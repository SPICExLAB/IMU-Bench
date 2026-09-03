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

from bench.score.common import acc_only_slice, load_ckpt_meta, parse_metrics_txt, val_row_at_step

RIGHT_ARM_CHAIN = [14, 17, 19, 21, 23]


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

    pos_errs, ang_errs = [], []
    n_windows = 0
    with torch.no_grad():
        for x, y_full, _pos in tqdm(loaders["val_loader"], desc="arm_pose", leave=False):
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
    return metrics, info
