"""Full-body pose scorer.

The primary metric is the root-relative MPJPE (mm) over SMPL joints 1..23 that
the trainer computes on the full validation set at every eval
(imu_lm/objectives/pose_1d.py::_compute_val_mpjpe). We read it from
logs/metrics.txt at the step of the selected checkpoint (best.pt under the
bench recipe = min val mpjpe; legacy runs: best_mpjpe.pt), so the number is
exactly the trainer's, not a re-implementation.

Per-window sidecar (bootstrap CIs): `write_units_after_save` -- called by
bench.score.score_cell only AFTER the result JSON is written, and by
bench/rescore_units.py -- re-evaluates the selected checkpoint on the same val
loader the trainer used (same files, batch size, order, AMP setting) and stores
every window's MPJPE (mean over its 32 patches x 4 sub-patches x 23 joints) plus
its session and batch index. The trainer logged the UNWEIGHTED mean over val
batches of per-batch means, so the sidecar check recomputes that aggregation
(`reported_aggregation: batch_mean`) and compares it with the logged value; the
bootstrap statistic itself is the window mean (the two differ only through the
smaller last batch).
"""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, Tuple

from bench.score.common import acc_only_slice, load_ckpt_meta, parse_metrics_txt, save_units, val_row_at_step


def score(run_dir: str, ckpt_path: str, task: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    meta = load_ckpt_meta(ckpt_path)
    step = meta["step"]
    rows = parse_metrics_txt(run_dir, "val")
    if not rows:
        raise RuntimeError(f"no val rows in {run_dir}/logs/metrics.txt")
    row = val_row_at_step(rows, step)
    if row is None or "mpjpe_mm" not in row:
        raise RuntimeError(f"no mpjpe_mm val row at step {step} in {run_dir}")
    metrics = {"mpjpe_mm": float(row["mpjpe_mm"])}
    if "pose_mse" in row:
        metrics["val_pose_mse"] = float(row["pose_mse"])
    # sanity: the min over the whole curve (what best_mpjpe.pt tracks)
    min_row = min((r for r in rows if "mpjpe_mm" in r), key=lambda r: r["mpjpe_mm"])
    info = {"checkpoint_step": step, "min_val_mpjpe_mm": float(min_row["mpjpe_mm"]),
            "min_val_mpjpe_step": int(min_row["step"]), "n_val_evals": len(rows)}
    return metrics, info


def _per_window_mpjpe(pred_rot, y_full_rot, gt_pos, K: int, J: int, body_model):
    """imu_lm/objectives/pose_1d.py::_compute_val_mpjpe without the final reduction over windows:
    [B] root-relative MPJPE (mm) over joints 1..J_loader-1, every patch and sub-patch."""
    from imu_lm.objectives.pose_1d import _assemble_full_R
    B, Pt, _ = pred_rot.shape
    J_loader = y_full_rot.shape[-1] // (K * 6)
    full_R = _assemble_full_R(pred_rot, y_full_rot, B, Pt, K, J, J_loader)
    bm_device = body_model._J_regressor.device
    full_R_flat = full_R.reshape(B * Pt * K, -1).to(bm_device)
    _, pos_pred = body_model.forward_kinematics(full_R_flat)
    pos_pred = pos_pred.to(pred_rot.device)
    pos_pred_rel = (pos_pred - pos_pred[:, 0:1]).reshape(B, Pt, K, J_loader, 3)
    gt_pos_4d = gt_pos.reshape(B, Pt, K, J_loader, 3)
    diff = pos_pred_rel[:, :, :, 1:J_loader] - gt_pos_4d[:, :, :, 1:J_loader]
    return diff.norm(dim=-1).float().mean(dim=(1, 2, 3)) * 1000.0


def reeval_val_mpjpe(run_dir: str, ckpt_path: str, max_batches: int = 0, val_files=None) -> Dict[str, Any]:
    """Re-run the trainer's val MPJPE pass for `ckpt_path`; returns per-window arrays.
    max_batches / val_files: smoke tests only (a prefix of the val batches / a subset of sessions)."""
    import numpy as np
    import torch
    from torch.amp import autocast
    import core.articulate as art
    from core.paths import Paths
    from imu_lm.data.nymeria_real_wrist_full_body_loader_FLU import make_nymeria_real_wrist_full_body_loaders_flu
    from imu_lm.data.windowing import compute_T_and_hop
    from imu_lm.models.ViT1D.model import ViT1DPoseDecoder
    from imu_lm.models.baselines.inertia1_encoder import build_encoder
    from imu_lm.utils.helpers import cfg_get
    from bench.score.arm_pose import indexed_loader, window_sessions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_ckpt_meta(ckpt_path)
    ckpt, cfg = meta["_obj"], meta["cfg"]
    obj_type = cfg_get(cfg, ["objective", "type"], "pose_probe")
    ob = cfg_get(cfg, ["objective", obj_type], {}) or {}
    loader_type = str(ob.get("loader", "")).lower()
    if loader_type != "nymeria_real_wrist_full_body_flu":
        raise RuntimeError(f"full-body re-eval supports the nymeria_real_wrist_full_body_flu loader, got {loader_type!r}")
    num_joints = int(ob.get("num_joints", 5))
    K = int((cfg_get(cfg, ["vit1d", "pose_decoder"], {}) or {}).get("sub_patches", 1))

    encoder = build_encoder(cfg).to(device)
    encoder.load_state_dict(ckpt["model"], strict=True)
    encoder.eval()
    T, _ = compute_T_and_hop(cfg)
    decoder = ViT1DPoseDecoder(
        embed_dim=encoder.embed_dim, num_patches_per_channel=T // int(cfg_get(cfg, ["vit1d", "patch_size"], 16)),
        in_channels=encoder.in_channels, num_joints=num_joints, cfg=cfg, sub_patches=K,
    ).to(device)
    decoder.load_state_dict(ckpt["pose_decoder"], strict=True)
    decoder.eval()

    # val files exactly as imu_lm/models/ViT1D/run.py expands them for pose / pose_probe
    root = ob.get("val_pkl_root") or ob.get("train_pkl_root") or ob.get("pkl_root")
    files_cfg = ob.get("val_files")
    if val_files is not None:
        val_files = list(val_files)
    elif files_cfg is None or files_cfg == "all":
        val_files = sorted(glob.glob(os.path.join(root, "*.pkl")))
    elif isinstance(files_cfg, list):
        val_files = [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
    else:
        val_files = [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]
    loaders = make_nymeria_real_wrist_full_body_loaders_flu(
        cfg=cfg, pkl_root="/", train_files=[], val_files=val_files,
        add_gravity=bool(ob.get("add_gravity", True)))
    val_loader = loaders["val_loader"]
    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=device)   # trainer: cuda if available
    amp = bool(cfg_get(cfg, ["trainer", "amp"], False)) and device.type == "cuda"

    vals, batch_of, idxs = [], [], []
    nb = 0
    with torch.no_grad():
        for batch in indexed_loader(val_loader):
            if batch is None:
                continue
            x, y_full, gt_pos, bidx = batch
            x = acc_only_slice(x.to(device).float(), cfg)
            y_full, gt_pos = y_full.to(device).float(), gt_pos.to(device)
            with autocast("cuda", enabled=amp):
                tokens, _ = encoder.forward_tokens(x)
                pred = decoder(tokens)
                m = _per_window_mpjpe(pred, y_full, gt_pos, K, num_joints, body_model)
            vals.append(m.double().cpu().numpy())
            batch_of.append(np.full(len(bidx), nb, dtype=np.int64))
            idxs.append(bidx.numpy())
            nb += 1
            if max_batches and nb >= max_batches:
                break
    vals, batch_of, idxs = map(np.concatenate, (vals, batch_of, idxs))
    files, within = window_sessions(val_loader.dataset, idxs)
    return {"value": vals, "batch": batch_of, "files": files, "within": within, "step": int(meta["step"]),
            "best_val_mpjpe_in_ckpt": meta.get("best_val_mpjpe"), "amp": amp, "device": device.type,
            "n_batches": nb, "n_val_windows_total": len(val_loader.dataset), "max_batches": max_batches}


def write_units_after_save(run_dir: str, ckpt_path: str, task: Dict[str, Any], metrics: Dict[str, Any],
                           max_batches: int = 0, val_files=None):
    """Per-window sidecar via a val re-evaluation (see module docstring)."""
    import numpy as np
    from bench.score.common import nymeria_participant as participant_of
    r = reeval_val_mpjpe(run_dir, ckpt_path, max_batches=max_batches, val_files=val_files)
    v, b = r["value"], r["batch"]
    batch_mean = float(np.mean([v[b == k].mean() for k in np.unique(b)]))
    files = r["files"]
    return save_units(
        run_dir, ckpt_path, task, kind="mean",
        arrays={"unit_key": [f"{f}#{w}" for f, w in zip(files, r["within"])], "value": v, "batch": b},
        groups={"participant": [participant_of(f) for f in files], "session": files},
        metric_reported=metrics.get(task["metrics"]["primary"]), checkpoint_step=r["step"],
        notes="unit = 10.24 s val window; value = its MPJPE (mm, joints 1-23, all patches/sub-patches) from a "
              "re-evaluation of the checkpoint; JSON value = trainer log (batch-averaged); "
              "session = val pkl file; participant parsed from the file name"
              + (f"; SMOKE TEST: first {max_batches} val batches only" if max_batches else "")
              + (f"; SMOKE TEST: {len(val_files)} val sessions only" if val_files is not None else ""),
        extra_meta={"reported_aggregation": "batch_mean",
                    "check_metrics": {"window_mean_mm": float(v.mean()), "batch_mean_mm": batch_mean,
                                      "best_val_mpjpe_in_ckpt": r["best_val_mpjpe_in_ckpt"]},
                    "reeval": {k: r[k] for k in ("amp", "device", "n_batches", "n_val_windows_total", "max_batches")}})
