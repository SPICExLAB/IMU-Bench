"""Reproduce Inertia-1 pretraining (AR-Transformer / PatchTST, "small" preset) on
the NHANES 20 Hz corpus built by build_nhanes_corpus.py -- no Lightning.

Recipe = the shipped Inertia-1 configs (inertia1/config/{default,methods/*,presets/small}.yaml
+ scripts/pretrain/*.sh) unless noted:
    window 30 s @ 20 Hz, triaxial acc in g, no normalisation, batch 1024,
    AdamW, linear warmup 500 steps then cosine to 1 % of peak, grad-clip 1.0,
    bf16 autocast, val every 5000 steps, keep best (val loss) + last.
      ar_transformer: lr 1e-3, wd 0.04, betas (0.9, 0.999)   [default.yaml + methods/ar_transformer.yaml]
      patchtst:       lr 1e-3, wd 1e-4, betas (0.9, 0.95), mask 0.4  [methods/patchtst.yaml]
    steps: paper App. A.4 says 170k optimisation steps (the .sh defaults to 100k);
           we use 170k by default (--max-steps).

Checkpoints are written in THIS repo's run-dir format so every downstream
config can point at them with ``pretrained_run: <run-name>``:
    runs/<run>/checkpoints/{best,latest}.pt = {"model": Inertia1Encoder.state_dict(),
                                              "cfg": <downstream-ready cfg>, "step": ..}
    runs/<run>/artifacts/encoder_meta.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from imu_lm.models.baselines.inertia1_encoder import Inertia1Encoder, G_MPS2  # noqa: E402

VAL_FRAC_PARTICIPANTS = 0.02


# --------------------------------------------------------------------------- data
def _is_val(seqn: str) -> bool:
    return int.from_bytes(hashlib.md5(f"val|{seqn}".encode()).digest()[:4], "little") / 2**32 < VAL_FRAC_PARTICIPANTS


def scan_corpus(root: str):
    pdir = os.path.join(root, "participants")
    train, val = [], []
    n_err = 0
    for fn in sorted(os.listdir(pdir)):
        if not fn.endswith(".json"):
            continue
        meta = json.load(open(os.path.join(pdir, fn)))
        if meta.get("error"):
            n_err += 1
        n = int(meta["n_windows"])
        if n <= 0:
            continue
        rec = (os.path.join(pdir, meta["seqn"] + ".npy"), n)
        (val if _is_val(meta["seqn"]) else train).append(rec)
    return train, val, n_err


class BlockShuffleDataset(IterableDataset):
    """Infinite stream of single 30 s windows: each worker keeps a shuffle buffer
    that it refills with random contiguous blocks (cheap sequential reads from a
    random participant memmap), and emits random buffer elements."""

    def __init__(self, files: List[tuple], block: int = 64, buffer_blocks: int = 96, seed: int = 0):
        self.files = files
        self.weights = np.array([n for _, n in files], dtype=np.float64)
        self.weights /= self.weights.sum()
        self.block, self.buffer_blocks, self.seed = block, buffer_blocks, seed

    def _read_block(self, rng):
        i = rng.choice(len(self.files), p=self.weights)
        path, n = self.files[i]
        arr = np.load(path, mmap_mode="r")
        b = min(self.block, n)
        s = rng.integers(0, n - b + 1)
        return np.ascontiguousarray(arr[s : s + b])  # [b, 3, 600] fp16

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        rng = np.random.default_rng(self.seed * 1000 + wid + int(time.time() * 1e3) % 100000)
        cap = self.block * self.buffer_blocks
        buf = np.empty((cap, 3, 600), dtype=np.float16)
        n = 0
        while True:
            while n < cap:
                blk = self._read_block(rng)
                m = min(len(blk), cap - n)
                buf[n : n + m] = blk[:m]
                n += m
            # emit a random half of the buffer, then refill
            idx = rng.permutation(n)[: max(1, n // 2)]
            for j in idx:
                yield torch.from_numpy(buf[j].astype(np.float32))
            keep = np.setdiff1d(np.arange(n), idx)
            buf[: len(keep)] = buf[keep]
            n = len(keep)


def load_val(files: List[tuple], per_participant: int = 200, max_total: int = 65536) -> torch.Tensor:
    chunks, tot = [], 0
    for path, n in files:
        arr = np.load(path, mmap_mode="r")
        take = min(per_participant, n)
        idx = np.linspace(0, n - 1, take).astype(int)
        chunks.append(np.asarray(arr[idx]).astype(np.float32))
        tot += take
        if tot >= max_total:
            break
    return torch.from_numpy(np.concatenate(chunks, 0))


# ------------------------------------------------------------------------ helpers
def downstream_cfg(variant: str, preset: str, run_name: str) -> Dict:
    """cfg saved in the checkpoint: what a downstream harness run needs to
    rebuild the adapter and preprocess inputs the Inertia-1 way (raw g)."""
    return {
        "inertia1": {"variant": variant, "preset": preset, "patch_len": 10, "stride": 10, "mask_ratio": 0.4,
                     "pretrain_window_sec": 30, "model_hz": 20,
                     "input_hz": 50, "acc_scale_to_g": 1.0 / G_MPS2, "acc_channels": [0, 1, 2],
                     "native_input": False, "pretrained_from": "NHANES (reproduction; no official weights)"},
        "vit1d": {"patch_size": 16, "max_patches_per_channel": 256, "pooling": "mean"},
        "data": {"sensor_columns": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]},
        "windowing": {"window_seconds": 10.24, "window_hop_ratio": 0.5},
        "preprocess": {"impute": {"enabled": True, "method": "linear", "max_missing_frac": 0.2},
                       "filter": {"enabled": False}, "normalize": {"enabled": False}},
        "objective": {"type": f"inertia1_pretrain_{variant}"},
        "run_name": run_name,
    }


def lr_lambda_for(variant: str, warmup: int, total: int, min_ratio: float = 0.01):
    def f(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        prog = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        if variant == "patchtst":
            return max(min_ratio, cos)                     # PatchTSTModule.configure_optimizers
        return min_ratio + (1.0 - min_ratio) * cos          # ARTransformerModule.configure_optimizers
    return f


def save_ckpt(path: str, model: Inertia1Encoder, cfg: Dict, step: int, extra: Dict, opt=None, sched=None):
    obj = {"model": model.state_dict(), "cfg": cfg, "step": step, **extra}
    if opt is not None:
        obj["optim"] = opt.state_dict()
        obj["sched"] = sched.state_dict()
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model: Inertia1Encoder, xval: torch.Tensor, device, bs: int = 2048) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(xval), bs):
        xb = xval[i : i + bs].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model.net.pretrain_loss(xb)
        tot += float(loss) * len(xb)
        n += len(xb)
    model.train()
    return tot / max(1, n)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["patchtst", "ar_transformer"], required=True)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--corpus", default="/projects/b53017/vbc1591/inertia1_nhanes20hz")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--runs-root", default=os.path.join(REPO, "runs"))
    ap.add_argument("--max-steps", type=int, default=170_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val-max", type=int, default=65536)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_dir = os.path.join(args.runs_root, args.run_name)
    for d in ("checkpoints", "logs", "artifacts"):
        os.makedirs(os.path.join(run_dir, d), exist_ok=True)
    metrics_path = os.path.join(run_dir, "logs", "metrics.txt")

    train_files, val_files, n_err = scan_corpus(args.corpus)
    n_train_win = sum(n for _, n in train_files)
    print(f"[data] train participants={len(train_files)} windows={n_train_win:,} ({n_train_win*30/3600:,.0f} h) | "
          f"val participants={len(val_files)} | unreadable={n_err}", flush=True)
    xval = load_val(val_files, max_total=args.val_max)
    print(f"[data] val windows={len(xval):,}", flush=True)

    # ---- model + optimiser (their hyper-parameters) -----------------------------
    cfg = downstream_cfg(args.variant, args.preset, args.run_name)
    model = Inertia1Encoder(cfg).to(device)
    model.native_input = True   # corpus is already 20 Hz g-unit windows
    n_params = sum(p.numel() for p in model.parameters())
    if args.variant == "patchtst":
        lr, wd, betas = 1e-3, 1e-4, (0.9, 0.95)
    else:
        lr, wd, betas = 1e-3, 0.04, (0.9, 0.999)
    lr = args.lr if args.lr is not None else lr
    wd = args.weight_decay if args.weight_decay is not None else wd
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_for(args.variant, args.warmup_steps, args.max_steps))
    print(f"[model] {args.variant}/{args.preset}: {n_params/1e6:.2f}M params | lr={lr} wd={wd} betas={betas} "
          f"| batch={args.batch_size} steps={args.max_steps}", flush=True)

    step, best_val = 0, float("inf")
    latest = os.path.join(run_dir, "checkpoints", "latest.pt")
    if args.resume and os.path.exists(latest):
        st = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optim"])
        sched.load_state_dict(st["sched"])
        step, best_val = int(st["step"]), float(st.get("best_val", float("inf")))
        print(f"[resume] from step {step} (best_val={best_val:.5f})", flush=True)

    with open(os.path.join(run_dir, "artifacts", "encoder_meta.json"), "w") as f:
        json.dump({"embedding_dim": model.embed_dim, "encoding": "raw_1d_imu", "backbone": "inertia1",
                   "objective": cfg["objective"]["type"], "variant": args.variant, "preset": args.preset,
                   "params": n_params, "input_spec": {"channels": 3, "hz": 20, "window_sec": 30, "units": "g",
                                                      "format": "[B, C, T]"}, "normalization": None},
                  f, indent=2)

    ds = BlockShuffleDataset(train_files, seed=args.seed + step)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)

    model.train()
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    for xb in dl:
        if step >= args.max_steps:
            break
        xb = xb.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model.net.pretrain_loss(xb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        run_loss += float(loss.detach()); run_n += 1

        if step % args.log_every == 0:
            el = time.time() - t0
            msg = f"step {step}/{args.max_steps} loss={run_loss/run_n:.5f} lr={sched.get_last_lr()[0]:.2e} " \
                  f"{step/el if step else 0:.1f} it/s (elapsed {el/60:.1f} min)"
            print(msg, flush=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": step, "train_loss": run_loss / run_n, "lr": sched.get_last_lr()[0]}) + "\n")
            run_loss, run_n = 0.0, 0

        if step % args.val_every == 0 or step == args.max_steps:
            vl = evaluate(model, xval, device)
            improved = vl < best_val
            best_val = min(best_val, vl)
            print(f"[val] step {step} val_loss={vl:.5f} best={best_val:.5f}{' *' if improved else ''}", flush=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": step, "val_loss": vl, "best_val": best_val}) + "\n")
            extra = {"val_loss": vl, "best_val": best_val, "variant": args.variant, "preset": args.preset}
            save_ckpt(latest, model, cfg, step, extra, opt, sched)
            if improved:
                save_ckpt(os.path.join(run_dir, "checkpoints", "best.pt"), model, cfg, step, extra)
    print(f"done: {step} steps, best_val={best_val:.5f}, {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
