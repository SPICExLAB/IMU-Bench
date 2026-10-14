#!/usr/bin/env python3
"""Compute positional (cm) and angular (deg) error for the arm-pose task's
ViT1D columns, directly comparable to MobilePoser's arm-pose numbers.

The arm-pose loader (nymeria_real_wrist_loader_FLU.py) only ever labels the
5-joint right-arm chain (SMPL indices [14,17,19,21,23] -- collar, shoulder,
elbow, wrist, hand) with LOCAL rotations, and provides no root/ancestor
context or position ground truth, so the val_loss metric reported elsewhere
in EXPERIMENTS.md (pose-MSE on 6D rotations) has no physical units.

To get a genuine cm/deg number comparable to MobilePoser's own
FullMotionEvaluator-based arm-pose metric (same 5 joint indices), this script:
  1. Loads GT rotations for ALL 24 SMPL joints via the FULL-BODY loader
     (nymeria_real_wrist_full_body_loader_FLU.py) on the same val sessions.
  2. Runs each arm-pose checkpoint's encoder+ViT1DPoseDecoder on the same
     windows to get predicted rotations for just the 5 arm-chain joints.
  3. Composes a "test skeleton": GT rotations everywhere except the 5
     predicted joints, which come from the model. Runs forward_kinematics
     (core.articulate, same body model used throughout this repo) on both
     the composed skeleton and the all-GT skeleton, and measures Euclidean
     distance at the 5 target joints -> positional error (cm).
  4. Angular error (deg) is the direct local-rotation geodesic distance
     between predicted and GT at the 5 target joints (no FK needed; since
     both skeletons share identical GT ancestors up to those joints, this
     is mathematically identical to a "global" angle error here).

Caveat (documented in EXPERIMENTS.md): unlike MobilePoser, which predicts the
torso/ancestor chain too (compounding its own errors into the arm position),
this metric holds ancestors at ground truth -- it isolates "how well does the
model predict the arm chain's own shape" rather than a fully end-to-end
number. Expect this to look somewhat better than a true end-to-end metric
would.

Usage
-----
    python scripts/eval_arm_pose_position_error.py \
        --val-dir /projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist \
        --out-json results_arm_pose_position_error.json
"""

from __future__ import annotations

from imu_lm.utils.data_root import data_path as _data_path
import os
import sys
import glob
import json
import argparse

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DEncoderAltAttn, ViT1DPoseDecoder
from imu_lm.data.nymeria_real_wrist_full_body_loader_FLU import make_nymeria_real_wrist_full_body_loaders_flu
from imu_lm.data.windowing import compute_T_and_hop
import core.articulate as art
from core.paths import Paths

RIGHT_ARM_CHAIN = [14, 17, 19, 21, 23]

# (display name, run dir under runs/, alt_attention)
RUNS = [
    ("Ours (MAE)",              "vit1d_arm_pose_probe_nymeria_mae_multidata",             True),
    ("Ours (MAE) finetune",     "vit1d_arm_pose_finetune_nymeria_mae_multidata",          True),
    ("Ours (MAE, no alt-attn)", "vit1d_arm_pose_probe_nymeria_mae_noaltattn_multidata",   False),
    ("FOCAL",                   "vit1d_arm_pose_probe_nymeria_focal_multidata",           False),
    ("SimMTM",                  "vit1d_arm_pose_probe_nymeria_simmtm_multidata",          False),
    ("Scratch",                 "vit1d_arm_pose_supervised_nymeria_altattn_scratch",      True),
]


def build_cfg(alt_attention: bool) -> dict:
    return {
        "vit1d": {
            "patch_size": 16,
            "shared_kernel": "acc_gyro",
            "encoder": {
                "hidden_size": 384,
                "num_hidden_layers": 12,
                "num_attention_heads": 6,
                "intermediate_size": 1536,
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
                "layer_norm_eps": 1.0e-6,
                "alt_attention": alt_attention,
            },
            "pose_decoder": {
                "hidden_size": 192,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "mlp_ratio": 4.0,
                "dropout": 0.0,
            },
            "max_patches_per_channel": 256,
            "pooling": "mean",
        },
        "windowing": {"window_seconds": 10.24, "window_hop_ratio": 0.5},
        "data": {
            "sensor_columns": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
            "batch_size": 32, "eval_batch_size": 32, "num_workers": 4, "pin_memory": True,
        },
        "preprocess": {
            "impute": {"enabled": True, "method": "linear", "max_missing_frac": 0.2},
            "filter": {"enabled": True, "type": "lowpass", "low_hz": 0.25, "high_hz": 24, "order": 4},
            "normalize": {"enabled": True, "method": "zscore", "eps": 1.0e-6},
        },
    }


@torch.no_grad()
def eval_run(run_name: str, alt_attention: bool, val_files, body_model, device):
    ckpt_path = os.path.join("runs", run_name, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("cfg", {}).get("inertia1"):
        # External Inertia-1 baseline: its own preprocessing (raw g-scale, no
        # z-score) and encoder block live in the checkpoint cfg; the loader
        # still yields 6-ch [acc, gyro] and the adapter slices the acc channels.
        import copy as _copy
        cfg = _copy.deepcopy(ckpt["cfg"])
        cfg.setdefault("data", {}).update({"batch_size": 32, "eval_batch_size": 32, "num_workers": 4, "pin_memory": True})
        cfg["data"]["sensor_columns"] = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        cfg["data"].pop("acc_only", None)
    else:
        cfg = build_cfg(alt_attention)

    loaders = make_nymeria_real_wrist_full_body_loaders_flu(
        cfg=cfg, pkl_root="/", train_files=[], val_files=val_files,
        target_hz=50, add_gravity=True,
    )
    val_loader = loaders["val_loader"]

    from imu_lm.models.baselines.inertia1_encoder import build_encoder as _build_encoder
    encoder = _build_encoder(cfg).to(device)
    encoder.load_state_dict(ckpt["model"], strict=True)
    encoder.eval()

    T, _ = compute_T_and_hop(cfg)
    num_patches = T // int(cfg["vit1d"]["patch_size"])
    pose_decoder = ViT1DPoseDecoder(
        embed_dim=encoder.embed_dim,
        num_patches_per_channel=num_patches,
        in_channels=encoder.in_channels,
        num_joints=5,
        cfg=cfg,
        sub_patches=1,
    ).to(device)
    pose_decoder.load_state_dict(ckpt["pose_decoder"], strict=True)
    pose_decoder.eval()

    pos_errs, ang_errs = [], []
    for x, y_full, _pos in tqdm(val_loader, desc=run_name, leave=False):
        x = x.to(device).float()
        tokens, _ = encoder.forward_tokens(x)
        pred = pose_decoder(tokens)  # [B, Pt, 30]
        B, Pt, _ = pred.shape

        pred_R = art.math.r6d_to_rotation_matrix(pred.reshape(-1, 6)).view(B * Pt, 5, 3, 3)

        gt_r6d = y_full.to(device).float().view(-1, 6)
        gt_R_full = art.math.r6d_to_rotation_matrix(gt_r6d).view(B * Pt, 24, 3, 3)

        # angular error: direct local-rotation geodesic distance (deg)
        ang = art.math.angle_between(pred_R.reshape(-1, 3, 3), gt_R_full[:, RIGHT_ARM_CHAIN].reshape(-1, 3, 3))
        ang_errs.append(art.math.radian_to_degree(ang).reshape(-1).cpu())

        # positional error: compose GT-ancestor + predicted-arm skeleton, run FK, compare (cm)
        test_R = gt_R_full.clone()
        test_R[:, RIGHT_ARM_CHAIN] = pred_R
        _, gt_pos = body_model.forward_kinematics(gt_R_full.cpu())
        _, test_pos = body_model.forward_kinematics(test_R.cpu())
        d = (test_pos[:, RIGHT_ARM_CHAIN] - gt_pos[:, RIGHT_ARM_CHAIN]).norm(dim=-1)  # [B*Pt, 5]
        pos_errs.append(d.reshape(-1))

    pos_err_cm = float(torch.cat(pos_errs).mean().item() * 100)
    ang_err_deg = float(torch.cat(ang_errs).mean().item())
    return pos_err_cm, ang_err_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", default=_data_path("/projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist"))
    ap.add_argument("--max-val-files", type=int, default=101)
    ap.add_argument("--out-json", default="results_arm_pose_position_error.json")
    ap.add_argument("--runs", nargs="+", default=None, metavar="NAME=RUN_DIR[:alt]",
                    help="override the built-in RUNS list, e.g. 'Inertia-1 (PatchTST)=vit1d_arm_pose_probe_nymeria_inertia1_patchtst' "
                         "(append ':alt' for alt-attention ViT1D checkpoints)")
    args = ap.parse_args()
    runs = RUNS
    if args.runs:
        runs = []
        for spec in args.runs:
            name, rest = spec.split("=", 1)
            run_dir, _, flag = rest.partition(":")
            runs.append((name, run_dir, flag == "alt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))
    val_files = sorted(glob.glob(os.path.join(args.val_dir, "*.pkl")))[: args.max_val_files]
    print(f"val sessions: {len(val_files)}")

    results = {}
    for name, run_name, alt_attention in runs:
        pos_cm, ang_deg = eval_run(run_name, alt_attention, val_files, body_model, device)
        results[name] = {"pos_err_cm": pos_cm, "ang_err_deg": ang_deg, "run": run_name}
        print(f"{name}: pos_err={pos_cm:.2f}cm ang_err={ang_deg:.2f}deg")

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
