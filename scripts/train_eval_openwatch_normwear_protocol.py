#!/usr/bin/env python3
"""NormWear-Base (Luo et al., official 136M checkpoint) under the §5f
fair-comparison protocol (scripts/openwatch_paper_protocol_common.py) —
identical split / windows / metrics as Mix-Token and our ViT1D MAE.

This is the OpenWatch paper's "NormWear-Base" row recipe (§3.3.2-3.3.3):
frozen backbone, embedding = mean over patches then concat over the 6 IMU
channels (NormWear's own downstream convention, 4608-d), classification head
= LayerNorm -> Linear(128) -> GELU -> Dropout -> Linear(K); class-weighted CE
(inverse frequency), AdamW, early stop on val macro-F1 patience 5, best-epoch
checkpoint, batch 16, <=100 epochs, seed 42. Input = raw device-axis window
at native 100Hz (NormWear resamples internally to 65Hz), NO z-score — this
matches EXPERIMENTS.md §5c/§5d, whose window-level reproduction landed
within 0.8pp of the paper's cited NormWear-Base* number.

Backbone embeddings are frozen -> computed once per (condition, split) and
cached; only the head trains.

Usage
-----
    python scripts/train_eval_openwatch_normwear_protocol.py --condition aug \
        --out-json results_openwatch_normwear_protocol_aug.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "baselines", "SSL"))

from NormWear.main_model import NormWearModel  # noqa: E402
from scripts.openwatch_paper_protocol_common import (  # noqa: E402
    NATIVE_HZ, NUM_CLASSES, SEED, build_protocol_split, build_window_arrays,
    inverse_frequency_weights, window_and_clip_metrics,
)

CKPT_PATH = os.path.join(REPO, "baselines/SSL/NormWear/checkpoints/normwear_pretrain_ckpt.pth")
EPOCHS = 100
BATCH = 16
PATIENCE = 5
LR = 1e-3        # UNSPECIFIED for the head; paper gives 1e-3 for Mix-Token
WD = 1e-3
DROPOUT = 0.2    # UNSPECIFIED ("dropout regularization on the classifier head")


class Head(nn.Module):
    """Paper §3.3.2: LayerNorm -> Linear(128) -> GELU -> Dropout -> Linear(K)."""

    def __init__(self, d_in, k, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 128), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(128, k))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def embed_windows(X: np.ndarray, model, device, cache_path: str | None) -> np.ndarray:
    """X [N, W, C] raw -> [N, 4608] NormWear embeddings."""
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)["E"]
    E = []
    for i, x in enumerate(X):
        x_t = torch.from_numpy(x.T[None]).float()  # [1, C, W] on CPU (get_embedding calls .numpy())
        out = model.get_embedding(x_t, sampling_rate=NATIVE_HZ, device=device)  # [1, nvar, P, E]
        E.append(out.mean(dim=2).flatten(start_dim=1)[0].cpu().numpy())
        if i % 500 == 0:
            print(f"  embedded {i}/{len(X)}")
    E = np.stack(E).astype(np.float32)
    if cache_path:
        np.savez(cache_path, E=E)
    return E


@torch.no_grad()
def predict(head, E, device, bs=1024):
    head.eval()
    out = []
    for i in range(0, len(E), bs):
        out.append(head(torch.from_numpy(E[i:i + bs]).to(device)).argmax(-1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["noaug", "aug"], required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--cache-dir", default="cache_openwatch_normwear_protocol")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.cache_dir, exist_ok=True)
    print("device:", device, "condition:", args.condition)

    train, val, test = build_protocol_split(args.condition)
    print(f"records train={len(train)} val={len(val)} test={len(test)}")

    model = None
    splits = {}
    for name, recs in [("train", train), ("val", val), ("test", test)]:
        X, y, cid = build_window_arrays(recs, flu=False)
        # val/test are identical across conditions -> share their cache
        tag = args.condition if name == "train" else "real"
        cache = os.path.join(args.cache_dir, f"{tag}_{name}.npz")
        if not os.path.exists(cache) and model is None:
            model = NormWearModel(weight_path=CKPT_PATH, optimized_cwt=True).to(device).eval()
            print(f"NormWear params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        t0 = time.time()
        E = embed_windows(X, model, device, cache)
        print(f"{name}: windows={len(y)} E={E.shape} ({time.time()-t0:.0f}s)")
        splits[name] = dict(E=E, y=y, cid=cid, clip_labels={i: r.label for i, r in enumerate(recs)})
    del model

    head = Head(splits["train"]["E"].shape[1], NUM_CLASSES).to(device)
    cw = torch.tensor(inverse_frequency_weights(splits["train"]["y"])).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

    tr = splits["train"]
    n = len(tr["y"])
    Etr = torch.from_numpy(tr["E"]).to(device)
    ytr = torch.from_numpy(tr["y"]).to(device)
    best_f1, best_state, best_epoch, bad = -1.0, None, -1, 0
    for epoch in range(EPOCHS):
        head.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(Etr[idx]), ytr[idx], weight=cw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)  # paper: global-norm clip for NormWear
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        va = splits["val"]
        vm = window_and_clip_metrics(va["y"], predict(head, va["E"], device), va["cid"], va["clip_labels"])
        f1 = vm["window"]["macro_f1"]
        print(f"epoch {epoch:3d} loss={tot/n:.4f} val_win_f1={f1:.4f} val_clip_acc={vm['clip']['acc']:.4f}")
        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[early stop] epoch={epoch} best_epoch={best_epoch} best_val_win_f1={best_f1:.4f}")
                break

    head.load_state_dict(best_state)
    result = {"model": "NormWear-Base (first-party, paper head)", "condition": args.condition,
              "params": 136_100_000, "best_epoch": best_epoch, "seed": args.seed}
    for split in ["val", "test"]:
        s = splits[split]
        result[split] = window_and_clip_metrics(s["y"], predict(head, s["E"], device), s["cid"], s["clip_labels"])
        print(f"[{split}] window acc={result[split]['window']['acc']:.4f} f1={result[split]['window']['macro_f1']:.4f} | "
              f"clip acc={result[split]['clip']['acc']:.4f} f1={result[split]['clip']['macro_f1']:.4f}")
    with open(args.out_json, "w") as fh:
        json.dump(result, fh, indent=2)
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
