#!/usr/bin/env python3
"""Joint multi-task fine-tuning of ONE shared encoder on several IMU-Bench tasks.

    python scripts/train_multitask_joint.py --name mt_body_odom \
        --tasks full_body_pose_nymeria odometry_nymeria_local --ratio 1 1 --steps 40000

The question (paper Section 5): adapting the MAE encoder to one task specialises it -- it helps a related task and
hurts a distant one. Is that a property of the tasks (they need incompatible features) or of single-task
adaptation? Here the encoder starts from the same MAE checkpoint as every single-task fine-tune and is updated by
ALL listed tasks: every optimiser step draws one batch from one task (tasks are visited in a fixed cycle given by
--ratio, e.g. 10 1 = ten pose steps per gesture step), computes THAT task's benchmark loss with that task's own
decoder, and updates the shared encoder plus that decoder. Nothing task-specific is re-implemented: loaders,
decoders, losses and preprocessing are the benchmark's own (bench/generated/configs/bench_<task>__mae_5src__
finetune__s42.yaml), built exactly as imu_lm/models/ViT1D/run.py builds them.

Optimisation: AdamW; the shared encoder gets --enc-lr, every decoder keeps the learning rate and weight decay of
its task's single-task fine-tune (they differ by 100x across tasks: pose 1e-3, odometry 1e-4, gesture 1e-5).
Linear warm-up, then cosine to 1 % of the peak. AMP as in the single-task runs.

Checkpoint selection is PER TASK: `best.pt` under runs/<name>__<task>/ is the encoder and decoder at that task's
own lowest validation loss, and `latest.pt` is the most recent validation step. This matters because the
single-task fine-tunes these runs are compared against all early-stop on their own validation split; taking the
final step for every task would compare a selected model against an unselected one. It also matters because the
tasks do not peak together -- in the first pair of runs full-body pose was best at 18k steps while microgesture,
whose training set is 13 batches, was already past its best at the 2k checkpoint.

Outputs, per task, a checkpoint in the single-task format so that the benchmark's scorers and frozen probes read it
unchanged:   runs/<name>__<task>/checkpoints/{best,latest}.pt   with keys model / <decoder key> / cfg / step.
Validation loss of every task is logged every --val-every steps (a prefix of each val loader, for speed).
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import logging
import math
import os
import sys
import time

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from imu_lm.data.windowing import compute_T_and_hop                                 # noqa: E402
from imu_lm.models.ViT1D.run import _load_encoder_state_dict                        # noqa: E402
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml                    # noqa: E402

log = logging.getLogger("multitask")
MAE_RUN = "vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic"


def task_cfg(task):
    return deep_update(load_yaml(os.path.join(REPO, "configs/base.yaml")),
                       load_yaml(os.path.join(REPO, f"bench/generated/configs/bench_{task}__mae_5src__finetune__s42.yaml")))


def _expand(root, files_cfg):
    if files_cfg is None or files_cfg == "all":
        return sorted(glob.glob(os.path.join(root, "*.pkl")))
    if isinstance(files_cfg, list):
        return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
    return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]


class Task:
    """loaders + decoder + loss of one benchmark task, around a SHARED encoder."""

    def __init__(self, name, encoder, device):
        self.name, self.cfg = name, task_cfg(name)
        cfg, enc = self.cfg, encoder
        self.obj_type = cfg_get(cfg, ["objective", "type"])
        ob = cfg_get(cfg, ["objective", self.obj_type], {}) or {}
        T, _ = compute_T_and_hop(cfg)
        n_patches = T // int(cfg_get(cfg, ["vit1d", "patch_size"], 16))
        self.lr = float(cfg_get(cfg, ["trainer", "optim", "lr"]))
        self.wd = float(cfg_get(cfg, ["trainer", "optim", "weight_decay"], 0.0))
        self.amp = bool(cfg_get(cfg, ["trainer", "amp"], False)) and device.type == "cuda"

        if self.obj_type == "pose":                                   # full-body pose (run.py, branch "pose")
            from imu_lm.data.nymeria_real_wrist_full_body_loader_FLU import make_nymeria_real_wrist_full_body_loaders_flu
            from imu_lm.models.ViT1D.model import ViT1DPoseDecoder
            from imu_lm.objectives import pose_1d as pose_obj
            assert str(ob.get("loader", "")).lower() == "nymeria_real_wrist_full_body_flu", ob.get("loader")
            J = int(ob.get("num_joints", 5))
            K = int((cfg_get(cfg, ["vit1d", "pose_decoder"], {}) or {}).get("sub_patches", 1))
            self.key = "pose_decoder"
            self.decoder = ViT1DPoseDecoder(embed_dim=enc.embed_dim, num_patches_per_channel=n_patches,
                                            in_channels=enc.in_channels, num_joints=J, cfg=cfg, sub_patches=K).to(device)
            body_model = None
            try:
                import core.articulate as art
                from core.paths import Paths
                body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=device)
            except Exception as e:                                      # same silent fallback as run.py
                log.warning("SMPL body model unavailable (%s): joint-position term off", e)
            jerk_w = float(ob.get("jerk_weight", ob.get("jitter_weight", 0.0)))
            jp_w = float(ob.get("joint_pos_weight", 0.0))
            dec = self.decoder

            def loss_fn(batch):
                x, y_full = batch[0], batch[1]
                if y_full.shape[-1] > dec.out_dim:
                    B, Pt, _ = y_full.shape
                    JL = y_full.shape[-1] // (K * 6)
                    y_pose = y_full.reshape(B, Pt, K, JL, 6)[:, :, :, JL - J:, :].reshape(B, Pt, dec.out_dim)
                else:
                    y_pose = y_full
                gt_pos = batch[2] if len(batch) >= 3 else None
                return pose_obj.forward_loss((x, y_pose), enc, dec, cfg, jerk_weight=jerk_w, joint_pos_weight=jp_w,
                                             body_model=body_model, y_full_rot=y_full.float(), gt_pos=gt_pos)
            self.loss_fn = loss_fn
            root_tr = ob.get("train_pkl_root") or ob.get("pkl_root"); root_va = ob.get("val_pkl_root") or root_tr
            self.loaders = make_nymeria_real_wrist_full_body_loaders_flu(
                cfg=cfg, pkl_root="/", train_files=_expand(root_tr, ob.get("train_files")),
                val_files=_expand(root_va, ob.get("val_files")), add_gravity=bool(ob.get("add_gravity", True)))

        elif self.obj_type == "odom_finetune_nymeria_real_wrist_flu":     # wrist-frame odometry
            from imu_lm.data.nymeria_real_wrist_odom_loader_FLU import make_nymeria_real_wrist_odom_loaders_flu
            from imu_lm.models.ViT1D.model import ViT1DOdomDecoder
            from imu_lm.objectives import odom_1d as odom_obj
            assert not bool(ob.get("freeze_encoder", False))
            self.key = "odom_decoder"
            self.decoder = ViT1DOdomDecoder(embed_dim=enc.embed_dim, num_patches_per_channel=n_patches,
                                            in_channels=enc.in_channels, cfg=cfg, out_dim=12).to(device)
            dec = self.decoder

            def loss_fn(batch):
                x, y = batch[0], batch[1]
                return odom_obj.forward_loss((x, y.reshape(y.shape[0], y.shape[1], -1)), enc, dec, cfg)
            self.loss_fn = loss_fn
            root_tr = ob.get("train_pkl_root"); root_va = ob.get("val_pkl_root") or root_tr
            self.loaders = make_nymeria_real_wrist_odom_loaders_flu(
                cfg=cfg, pkl_root="/", train_files=_expand(root_tr, ob.get("train_files")),
                val_files=_expand(root_va, ob.get("val_files")), add_gravity=bool(ob.get("add_gravity", True)))

        elif self.obj_type == "openwatch_gesture_finetune":               # microgesture
            from imu_lm.data.openwatch_loader import make_openwatch_loaders_from_cfg
            from imu_lm.models.ViT1D.run import LinearHead, _openwatch_ce_weight
            C = int(ob.get("num_classes", 59))
            self.key = "head"
            self.loaders = make_openwatch_loaders_from_cfg(cfg, objective_key="openwatch_gesture_finetune")
            self.decoder = LinearHead(enc.embed_dim, C).to(device)
            ce_w = _openwatch_ce_weight(self.loaders, ob, C)
            dec = self.decoder

            def loss_fn(batch):
                x, y = batch[0], batch[1]
                logits = dec(enc(x.float()))
                w = ce_w.to(logits.device) if ce_w is not None else None
                loss = torch.nn.functional.cross_entropy(logits, y.long(), weight=w)
                return loss, {"loss": float(loss.detach()), "acc": float((logits.argmax(-1) == y).float().mean())}
            self.loss_fn = loss_fn
        else:
            raise ValueError(f"{name}: objective {self.obj_type!r} has no multi-task adapter yet")

        self.train_loader, self.val_loader = self.loaders["train_loader"], self.loaders["val_loader"]
        self._it = None
        log.info("[%s] %s | train batches/epoch %d | val batches %d | decoder lr %g wd %g", name, self.obj_type,
                 len(self.train_loader), len(self.val_loader), self.lr, self.wd)

    def next_batch(self):
        while True:
            if self._it is None:
                self._it = iter(self.train_loader)
            try:
                b = next(self._it)
            except StopIteration:
                self._it = None
                continue
            if b is not None:
                return b


def to_dev(batch, device):
    return tuple(t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--ratio", nargs="+", type=int, default=None, help="steps per cycle for each task (default 1 each)")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--enc-lr", type=float, default=3e-4)
    ap.add_argument("--enc-wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratio = a.ratio or [1] * len(a.tasks)
    assert len(ratio) == len(a.tasks)

    # one shared encoder, built from the first task's config (the encoder block is identical across the benchmark)
    from imu_lm.models.baselines.inertia1_encoder import build_encoder
    cfg0 = task_cfg(a.tasks[0])
    encoder = build_encoder(cfg0).to(device)
    ck = os.path.join(REPO, "runs", MAE_RUN, "checkpoints", "best.pt")
    _load_encoder_state_dict(encoder, torch.load(ck, map_location="cpu", weights_only=False))
    log.info("shared encoder warm-started from %s", ck)
    tasks = [Task(t, encoder, device) for t in a.tasks]
    for t in tasks:                      # every task must describe the same encoder
        assert cfg_get(t.cfg, ["vit1d", "encoder"]) == cfg_get(cfg0, ["vit1d", "encoder"]), t.name

    groups = [{"params": list(encoder.parameters()), "lr": a.enc_lr, "weight_decay": a.enc_wd, "peak": a.enc_lr}]
    groups += [{"params": list(t.decoder.parameters()), "lr": t.lr, "weight_decay": t.wd, "peak": t.lr} for t in tasks]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=any(t.amp for t in tasks))
    cycle = list(itertools.chain.from_iterable([i] * r for i, r in enumerate(ratio)))
    log.info("tasks %s | cycle %s | %d steps | encoder lr %g", a.tasks, cycle, a.steps, a.enc_lr)

    def save_task(t, step, tag):
        d = os.path.join(REPO, "runs", f"{a.name}__{t.name}", "checkpoints"); os.makedirs(d, exist_ok=True)
        torch.save({"model": encoder.state_dict(), t.key: t.decoder.state_dict(), "cfg": t.cfg, "step": step,
                    "multitask": {"name": a.name, "tasks": a.tasks, "ratio": ratio, "enc_lr": a.enc_lr, "steps": a.steps}},
                   os.path.join(d, f"{tag}.pt"))

    def save(step, tag):
        for t in tasks:
            save_task(t, step, tag)

    @torch.no_grad()
    def validate(step):
        encoder.eval()
        out = {}
        for t in tasks:
            t.decoder.eval(); tot, n = 0.0, 0
            for i, b in enumerate(t.val_loader):
                if b is None:
                    continue
                if i >= a.val_batches:
                    break
                with torch.autocast("cuda", enabled=t.amp):
                    loss, _ = t.loss_fn(to_dev(b, device))
                tot += float(loss); n += 1
            out[t.name] = tot / max(n, 1); t.decoder.train()
        encoder.train()
        log.info("step=%d split=val %s", step, " ".join(f"{k}={v:.5f}" for k, v in out.items()))
        return out

    hist, best_val, best_step, t0 = [], {}, {}, time.time()
    encoder.train()
    for step in range(1, a.steps + 1):
        f = step / a.warmup if step <= a.warmup else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(1, a.steps - a.warmup)))
        for g in opt.param_groups:
            g["lr"] = g["peak"] * f
        t = tasks[cycle[(step - 1) % len(cycle)]]
        batch = to_dev(t.next_batch(), device)
        with torch.autocast("cuda", enabled=t.amp):
            loss, logs = t.loss_fn(batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(itertools.chain(encoder.parameters(), t.decoder.parameters()), 1.0)
        scaler.step(opt); scaler.update()
        if step % 200 == 0:
            log.info("step=%d split=train task=%s loss=%.5f lr_enc=%.2e (%.1f steps/s)", step, t.name, float(loss),
                     opt.param_groups[0]["lr"], step / (time.time() - t0))
        if step % a.val_every == 0 or step == a.steps:
            out = validate(step)
            hist.append({"step": step, **out})
            save(step, "latest")
            # Per-task checkpoint selection. The single-task fine-tunes this is compared against
            # all use early stopping on their own validation split, so taking the final step here
            # would compare a selected model against an unselected one. Tasks reach their optimum
            # at very different steps -- on the first run full-body pose was best at 18k while
            # microgesture, whose training set is 13 batches, was already past its best at 2k.
            for t in tasks:
                if out[t.name] < best_val.get(t.name, float("inf")):
                    best_val[t.name], best_step[t.name] = out[t.name], step
                    save_task(t, step, "best")
    log.info("best per task: %s", " ".join(f"{k}={best_val[k]:.5f}@{best_step[k]}" for k in best_val))
    for t in tasks:
        json.dump({"name": a.name, "tasks": a.tasks, "ratio": ratio, "steps": a.steps, "enc_lr": a.enc_lr,
                   "best_val": best_val, "best_step": best_step, "val_history": hist},
                  open(os.path.join(REPO, "runs", f"{a.name}__{t.name}", "multitask_summary.json"), "w"), indent=1)
    log.info("finished %s in %.0f min", a.name, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
