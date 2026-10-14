"""Arm-pose scorer: positional (cm) + angular (deg) error at the 5 right-arm
joints with GT ancestors, computed from a run's selected checkpoint.

Generalisation of scripts/eval_arm_pose_position_error.py: everything
(alt-attn, acc-only, window length, num_joints) is read from the checkpoint's
own resolved config instead of a hard-coded run list.
"""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, Tuple

from bench.score.common import acc_only_slice, load_ckpt_meta, parse_metrics_txt, save_units, val_row_at_step

RIGHT_ARM_CHAIN = [14, 17, 19, 21, 23]


class _WithIndex:
    """Map-style dataset wrapper that appends the dataset index to every (non-None) item."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        return None if item is None else (*item, i)


def _collate_with_index(batch):
    import torch
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    cols = list(zip(*batch))
    return (*[torch.stack(c, 0) for c in cols[:-1]], torch.as_tensor(cols[-1], dtype=torch.long))


def indexed_loader(loader):
    """Same batches (size / order / None-skipping) as `loader`, each with the dataset index of
    its windows appended as the last element."""
    from torch.utils.data import DataLoader
    return DataLoader(_WithIndex(loader.dataset), batch_size=loader.batch_size, shuffle=False,
                      num_workers=loader.num_workers, pin_memory=loader.pin_memory, drop_last=False,
                      collate_fn=_collate_with_index)


def window_sessions(ds, idx):
    """(session file basename, window number inside the session) for dataset indices `idx` of a
    NymeriaRealWristFullBodyDatasetFLU."""
    import numpy as np
    idx = np.asarray(idx, dtype=np.int64)
    si = np.asarray(ds._sess_idx)[idx]
    files = np.asarray([os.path.basename(f) for f in ds._session_files])[si]
    first = np.searchsorted(np.asarray(ds._sess_idx), si)
    return files, idx - first


def score(run_dir: str, ckpt_path: str, task: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    import torch
    from tqdm import tqdm
    import core.articulate as art
    from core.paths import Paths
    from imu_lm.data.nymeria_real_wrist_full_body_loader_FLU import make_nymeria_real_wrist_full_body_loaders_flu
    from imu_lm.data.windowing import compute_T_and_hop
    from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DEncoderAltAttn, ViT1DPoseDecoder
    from imu_lm.utils.helpers import cfg_get

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_ckpt_meta(ckpt_path)
    ckpt, cfg = meta["_obj"], meta["cfg"]

    obj_type = cfg_get(cfg, ["objective", "type"], "pose_probe")
    ob = cfg_get(cfg, ["objective", obj_type], {}) or {}
    num_joints = int(ob.get("num_joints", 5))
    assert num_joints == 5, f"arm_pose scorer expects 5 joints, got {num_joints}"
    val_dir = task["dataset"]["val"]
    val_files = sorted(glob.glob(os.path.join(val_dir, "*.pkl")))

    from imu_lm.models.baselines.inertia1_encoder import build_encoder as _build_encoder
    encoder = _build_encoder(cfg).to(device)   # ViT1D / alt-attn / Inertia-1 adapter per the ckpt cfg
    encoder.load_state_dict(ckpt["model"], strict=True)
    encoder.eval()

    T, _ = compute_T_and_hop(cfg)
    num_patches = T // int(cfg_get(cfg, ["vit1d", "patch_size"], 16))
    pose_decoder = ViT1DPoseDecoder(
        embed_dim=encoder.embed_dim, num_patches_per_channel=num_patches,
        in_channels=encoder.in_channels, num_joints=num_joints, cfg=cfg,
        sub_patches=int(ob.get("sub_patches", 1)),
    ).to(device)
    pose_decoder.load_state_dict(ckpt["pose_decoder"], strict=True)
    pose_decoder.eval()

    # Full-body loader on the same val sessions gives GT rotations for all 24 joints.
    loaders = make_nymeria_real_wrist_full_body_loaders_flu(
        cfg=cfg, pkl_root="/", train_files=[], val_files=val_files,
        target_hz=50, add_gravity=bool(ob.get("add_gravity", True)),
    )
    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))

    pos_errs, ang_errs, win_idx = [], [], []
    n_windows = 0
    with torch.no_grad():
        for x, y_full, _pos, bidx in tqdm(indexed_loader(loaders["val_loader"]), desc="arm_pose", leave=False):
            win_idx.append(bidx)
            x = acc_only_slice(x.to(device).float(), cfg)
            tokens, _ = encoder.forward_tokens(x)
            pred = pose_decoder(tokens)                       # [B, Pt, 5*6]
            B, Pt, _ = pred.shape
            n_windows += B
            pred_R = art.math.r6d_to_rotation_matrix(pred.reshape(-1, 6)).view(B * Pt, 5, 3, 3)
            gt_R_full = art.math.r6d_to_rotation_matrix(y_full.to(device).float().view(-1, 6)).view(B * Pt, 24, 3, 3)
            ang = art.math.angle_between(pred_R.reshape(-1, 3, 3), gt_R_full[:, RIGHT_ARM_CHAIN].reshape(-1, 3, 3))
            ang_errs.append(art.math.radian_to_degree(ang).reshape(-1).cpu())
            test_R = gt_R_full.clone()
            test_R[:, RIGHT_ARM_CHAIN] = pred_R
            _, gt_pos = body_model.forward_kinematics(gt_R_full.cpu())
            _, test_pos = body_model.forward_kinematics(test_R.cpu())
            d = (test_pos[:, RIGHT_ARM_CHAIN] - gt_pos[:, RIGHT_ARM_CHAIN]).norm(dim=-1)
            pos_errs.append(d.reshape(-1))

    metrics = {
        "pos_err_cm": float(torch.cat(pos_errs).mean().item() * 100.0),
        "ang_err_deg": float(torch.cat(ang_errs).mean().item()),
    }
    rows = parse_metrics_txt(run_dir, "val")
    row = val_row_at_step(rows, meta["step"])
    if row and "pose_mse" in row:
        metrics["val_pose_mse"] = float(row["pose_mse"])
    info = {"checkpoint_step": meta["step"], "n_val_windows": n_windows, "n_val_sessions": len(val_files)}
    try:
        _save_arm_units(run_dir, ckpt_path, task, loaders["val_loader"].dataset, torch.cat(win_idx).numpy(),
                        torch.cat(pos_errs), torch.cat(ang_errs), metrics, meta["step"])
    except Exception as e:     # never fail scoring over the sidecar
        print(f"[units] WARNING arm_pose sidecar: {e!r}")
    return metrics, info


def _save_arm_units(run_dir, ckpt_path, task, ds, idx, pos_errs, ang_errs, metrics, step):
    """Per-window sidecar: mean FK position error (cm) and angle error (deg) over the window's
    32 patches x 5 arm joints; every window has the same number of terms, so the mean over
    windows equals the reported mean over all terms (up to float32 summation order)."""
    from bench.score.common import nymeria_participant as participant_of
    n = len(idx)
    pos_w = pos_errs.double().reshape(n, -1).mean(dim=1).numpy() * 100.0
    ang_w = ang_errs.double().reshape(n, -1).mean(dim=1).numpy()
    files, within = window_sessions(ds, idx)
    save_units(run_dir, ckpt_path, task, kind="mean",
               arrays={"unit_key": [f"{f}#{w}" for f, w in zip(files, within)], "value": pos_w, "ang_err_deg": ang_w},
               groups={"participant": [participant_of(f) for f in files], "session": files},
               metric_reported=metrics.get(task["metrics"]["primary"]), checkpoint_step=step,
               notes="unit = 10.24 s val window; value = mean FK position error (cm) over its patches x 5 arm "
                     "joints; session = val pkl file; participant parsed from the file name")
