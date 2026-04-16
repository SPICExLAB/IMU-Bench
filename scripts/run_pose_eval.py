"""Pose regression evaluation for the ViT1D wrist model.

Loads a single Nymeria pkl sequence, runs the trained encoder + pose_decoder
over the full sequence, and:
    1. Reports per-joint geodesic (angular) error and MSE.
    2. Saves 2-D comparison plots for each right-arm joint.
    3. Visualizes the predicted arm pose vs ground truth in the body viewer
       (requires core.articulate; skipped gracefully if unavailable).

Usage
-----
    python scripts/run_pose_eval.py \\
        --config     configs/base.yaml \\
        --model-config configs/vit1d_pose.yaml \\
        --run        vit1d_nymeria_pose \\
        --pkl        /mnt/nas2/processed_data_test_body_Ty/some_seq.pkl \\
        [--ckpt      best]          # "best" | "latest" | absolute path
        [--max-frames 5000]         # clip sequence length (0 = full)
        [--fps       50]
        [--out-dir   data_vis/pose_eval]
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle

import numpy as np
from seaborn.objects import Paths
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as ScipyRot

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.nymeria_wrist_loader import (
    RIGHT_WRIST_CHAIN_JOINTS,
    _XSENS_RIGHT_WRIST_JOINT,
    _xsens_to_smpl_rotmat,
)
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DPoseDecoder
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Right-arm chain (5-joint mode)
JOINT_NAMES = {14: "R-Collar", 17: "R-Shoulder", 19: "R-Elbow", 21: "R-Wrist", 23: "R-Hand"}

# All 24 SMPL joint names (full-body mode)
SMPL_JOINT_NAMES = [
    "Pelvis",     "L-Hip",       "R-Hip",       "Spine1",
    "L-Knee",     "R-Knee",      "Spine2",      "L-Ankle",
    "R-Ankle",    "Spine3",      "L-Foot",      "R-Foot",
    "Neck",       "L-Collar",    "R-Collar",    "Head",
    "L-Shoulder", "R-Shoulder",  "L-Elbow",     "R-Elbow",
    "L-Wrist",    "R-Wrist",     "L-Hand",      "R-Hand",
]

logger = logging.getLogger(__name__)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def geodesic_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Angular distance in degrees between two sets of rotation matrices.

    Args:
        R_pred, R_gt: (..., 3, 3)
    Returns:
        angles: (...)  in degrees
    """
    R_diff = torch.matmul(R_pred, R_gt.transpose(-1, -2))
    trace  = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos    = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


# ---------------------------------------------------------------------------
# Data loading (no pypose / core.articulate)
# ---------------------------------------------------------------------------

def load_pkl_raw(pkl_path: str, fps: int = 50, max_frames: int = 0, num_joints: int = 5):
    """Load a Nymeria pkl and return IMU + GT rotations as numpy arrays.

    Parameters
    ----------
    num_joints : 5 or 24
        5  → return GT for right-arm chain (RIGHT_WRIST_CHAIN_JOINTS)  [N, 5,  6]
        24 → return GT for all SMPL body joints                        [N, 24, 6]

    Returns
    -------
    X       : [N, 6]             float32  — synthetic right-wrist acc + gyro
    gt_6d   : [N, num_joints, 6] float32  — GT 6D rotations
    smpl_R  : [N, 24, 3, 3]     float32  — full-body local rotation matrices (for vis)
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    xsens_pose = raw["gt_data"]["xsens_pose"]
    
    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float32)   # [N, 23, 4, 4]
    N = xsens_pose.shape[0]
    actual_fps = float(
        raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or fps
    )

    if max_frames > 0:
        N = min(N, max_frames)
        xsens_pose = xsens_pose[:N]

    # ---- IMU ----------------------------------------------------------------
    j   = _XSENS_RIGHT_WRIST_JOINT
    pos = xsens_pose[:, j, :3, 3]   # [N, 3]
    rot = xsens_pose[:, j, :3, :3]  # [N, 3, 3]

    acc                 = np.zeros((N, 3), dtype=np.float32)
    acc[1:-1]           = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) * (actual_fps ** 2)
    acc[0], acc[-1]     = acc[1], acc[-2]

    dR       = np.einsum("nij,njk->nik", rot[:-1].transpose(0, 2, 1), rot[1:])
    gyro_raw = ScipyRot.from_matrix(dR.reshape(-1, 3, 3)).as_rotvec() * actual_fps
    gyro     = np.zeros((N, 3), dtype=np.float32)
    gyro[:-1] = gyro_raw.astype(np.float32)
    gyro[-1]  = gyro[-2]

    X = np.concatenate([acc, gyro], axis=1)   # [N, 6]

    import core.articulate as art
    from core.paths import Paths

    # ---- Labels & full-body rotmats ----------------------------------------
    smpl_R_global = torch.from_numpy(_xsens_to_smpl_rotmat(xsens_pose[:, :, :3, :3]))   # [N, 24, 3, 3]
    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))
    smpl_R = (
        body_model
        .inverse_kinematics_R(smpl_R_global.reshape(smpl_R_global.shape[0], -1))
        .view(smpl_R_global.shape[0], 24, 3, 3)
        .contiguous().clone().cpu().numpy()
    )  # [N, 24, 3, 3]  local rotations

    if num_joints == 24:
        sel_R = smpl_R                              # [N, 24, 3, 3]
    else:
        sel_R = smpl_R[:, RIGHT_WRIST_CHAIN_JOINTS] # [N, 5,  3, 3]

    # 6D = first two columns of each rotation matrix
    gt_6d = np.concatenate([sel_R[..., 0], sel_R[..., 1]], axis=-1)  # [N, J, 6]

    return X, gt_6d, smpl_R


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    X: np.ndarray,
    encoder: ViT1DEncoder,
    pose_decoder: ViT1DPoseDecoder,
    cfg,
    device: torch.device,
) -> np.ndarray:
    """Slide a non-overlapping window over X and return per-frame 6D predictions.

    Returns
    -------
    pred_6d : [N, num_joints, 6]  float32
    """

    T, _hop    = compute_T_and_hop(cfg)
    vit_cfg    = cfg_get(cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))
    num_joints = int(cfg_get(cfg, ["objective", "pose", "num_joints"], 5))
    stats      = PreprocessStats()
    N          = len(X)

    # Pad so we always cover the full sequence
    pad   = (T - N % T) % T
    X_pad = np.concatenate([X, np.zeros((pad, 6), dtype=np.float32)], axis=0) if pad else X

    encoder.eval()
    pose_decoder.eval()

    pred_6d_full = np.zeros((len(X_pad), num_joints, 6), dtype=np.float32)

    n_windows = len(X_pad) // T
    for w in range(n_windows):
        start = w * T
        Xw    = X_pad[start:start + T]  # [T, 6]

        Xproc = preprocess_window(Xw, cfg, stats)
        if Xproc is None:
            continue

        x = torch.from_numpy(Xproc).float().unsqueeze(0).to(device)   # [1, 6, T]

        tokens, _npc = encoder.forward_tokens(x)     # [1, C*P_t, D]
        pred          = pose_decoder(tokens)           # [1, P_t, J*6]
        pred_np       = pred.squeeze(0).cpu().numpy() # [P_t, J*6]

        # Expand each patch prediction to all frames in that patch
        for p in range(pred_np.shape[0]):
            f0 = start + p * patch_size
            f1 = f0 + patch_size
            pred_6d_full[f0:f1] = pred_np[p].reshape(num_joints, 6)

    return pred_6d_full[:N]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _joint_names_for(num_joints: int):
    """Return ordered list of joint name strings for the given mode."""
    if num_joints == 24:
        return SMPL_JOINT_NAMES
    return [JOINT_NAMES[j] for j in RIGHT_WRIST_CHAIN_JOINTS]


def plot_joint_error(geodesic: np.ndarray, mse: np.ndarray, fps: int, out_dir: str,
                    num_joints: int = 5):
    """Bar chart of mean geodesic error and MSE per joint."""
    names = _joint_names_for(num_joints)
    x = np.arange(len(names))
    fig_w = max(12, len(names) * 0.55)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, 5))
    ax1.bar(x, geodesic, color="steelblue")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_ylabel("Mean Geodesic Error (°)"); ax1.set_title("Per-Joint Geodesic Error")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, mse, color="tomato")
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("MSE (6D)"); ax2.set_title("Per-Joint 6D MSE")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "joint_error_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved %s", path)


def plot_joint_timeseries(
    pred_6d: np.ndarray,       # [N, J, 6]
    gt_6d: np.ndarray,         # [N, J, 6]
    geodesic_seq: np.ndarray,  # [N, J]
    fps: int,
    out_dir: str,
    max_plot_frames: int = 1500,
    num_joints: int = 5,
):
    """One figure per joint: 6D components + geodesic error over time."""
    n = min(len(pred_6d), max_plot_frames)
    time = np.arange(n) / fps

    joint_names = _joint_names_for(num_joints)

    for ji, jname in enumerate(joint_names):
        fig, axes = plt.subplots(7, 1, figsize=(14, 18))

        for d in range(6):
            ax = axes[d]
            ax.plot(time, gt_6d[:n, ji, d], "b-", label="GT", linewidth=1.2, alpha=0.8)
            ax.plot(time, pred_6d[:n, ji, d], "r--", label="Pred", linewidth=1.0, alpha=0.8)
            ax.set_ylabel(f"6D[{d}]"); ax.grid(True, alpha=0.3)
            if d == 0:
                ax.set_title(f"{jname}  —  6D components & geodesic error")
                ax.legend(loc="upper right")

        ax = axes[6]
        ax.plot(time, geodesic_seq[:n, ji], "g-", linewidth=1.0)
        ax.axhline(geodesic_seq[:n, ji].mean(), color="r", linestyle="--",
                   label=f"mean={geodesic_seq[:n, ji].mean():.1f}°")
        ax.set_ylabel("Geodesic (°)"); ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"joint_{ji:02d}_{jname.replace('-','').replace(' ','_')}.png"
        path  = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=130)
        plt.close()
        logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    ap = argparse.ArgumentParser(description="Evaluate ViT1D pose decoder on a Nymeria sequence")
    ap.add_argument("--config",       required=True, help="Base config YAML")
    ap.add_argument("--model-config", required=True, help="Model config YAML (vit1d_pose.yaml)")
    ap.add_argument("--run",          required=True, help="Run name under runs/<run>")
    ap.add_argument("--pkl",          required=True, help="Path to a single Nymeria pkl file")
    ap.add_argument("--ckpt",         default="best", help="Checkpoint name: best | latest | abs path")
    ap.add_argument("--max-frames",   type=int, default=0, help="Max frames to evaluate (0 = full)")
    ap.add_argument("--fps",          type=int, default=50)
    ap.add_argument("--out-dir",      default="data_vis/pose_eval")
    args = ap.parse_args()

    base_cfg  = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    cfg = deep_update(base_cfg, model_cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ load
    runs_root = cfg.get("paths", {}).get("runs_root", "runs")
    run_dir   = os.path.join(runs_root, args.run)
    if args.ckpt in ("best", "latest"):
        ckpt_path = os.path.join(run_dir, "checkpoints", f"{args.ckpt}.pt")
    else:
        ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = state["cfg"]

    # Encoder
    encoder = ViT1DEncoder(ckpt_cfg).to(device)
    encoder.load_state_dict(state["model"])
    encoder.eval()

    # Pose decoder
    vit_cfg    = cfg_get(ckpt_cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))
    T_win, _   = compute_T_and_hop(ckpt_cfg)
    pose_decoder = ViT1DPoseDecoder(
        embed_dim=encoder.embed_dim,
        num_patches_per_channel=T_win // patch_size,
        in_channels=encoder.in_channels,
        num_joints=cfg_get(ckpt_cfg, ["objective", "pose", "num_joints"], 5),
        cfg=ckpt_cfg,
    ).to(device)
    pose_decoder.load_state_dict(state["pose_decoder"])
    pose_decoder.eval()
    logger.info("Encoder   : embed_dim=%d  in_channels=%d", encoder.embed_dim, encoder.in_channels)
    logger.info("PoseDecoder: num_joints=%d  patches/ch=%d",
                pose_decoder.num_joints if hasattr(pose_decoder, "num_joints") else 5,
                T_win // patch_size)

    num_joints = int(cfg_get(cfg, ["objective", "pose", "num_joints"], 5))

    # ------------------------------------------------------------------ data
    logger.info("Loading sequence: %s", os.path.basename(args.pkl))
    X, gt_6d, smpl_R_gt = load_pkl_raw(
        args.pkl, fps=args.fps, max_frames=args.max_frames, num_joints=num_joints
    )
    N = len(X)
    logger.info("Sequence length: %d frames (%.1f s)", N, N / args.fps)
    logger.info("num_joints: %d  →  gt_6d shape: %s", num_joints, gt_6d.shape)

    # ------------------------------------------------------------------ infer
    logger.info("Running inference ...")
    pred_6d = run_inference(X, encoder, pose_decoder, cfg, device)  # [N, J, 6]

    # ------------------------------------------------------------------ metrics
    pred_t = torch.from_numpy(pred_6d)   # [N, J, 6]
    gt_t   = torch.from_numpy(gt_6d)     # [N, J, 6]

    pred_R = rotation_6d_to_matrix(pred_t)   # [N, J, 3, 3]
    gt_R   = rotation_6d_to_matrix(gt_t)     # [N, J, 3, 3]

    # Geodesic error per frame per joint: [N, J]
    geo_seq = geodesic_error_deg(
        pred_R.reshape(-1, 3, 3), gt_R.reshape(-1, 3, 3)
    ).reshape(N, num_joints).numpy()

    # MSE on 6D representation per joint: [J]
    mse_per_joint = ((pred_6d - gt_6d) ** 2).mean(axis=(0, 2))   # [J]

    names = _joint_names_for(num_joints)
    print("\n=== Per-Joint Evaluation ===")
    print(f"{'Joint':<16} {'Geodesic (°)':>14} {'Geodesic std':>13} {'6D-MSE':>10}")
    print("-" * 57)
    for ji, jname in enumerate(names):
        print(
            f"{jname:<16} "
            f"{geo_seq[:, ji].mean():>14.2f} "
            f"{geo_seq[:, ji].std():>13.2f} "
            f"{mse_per_joint[ji]:>10.5f}"
        )
    print("-" * 57)
    print(
        f"{'Mean':<16} "
        f"{geo_seq.mean():>14.2f} "
        f"{geo_seq.std():>13.2f} "
        f"{mse_per_joint.mean():>10.5f}"
    )

    # ------------------------------------------------------------------ plots
    plot_joint_error(geo_seq.mean(0), mse_per_joint, args.fps, args.out_dir,
                     num_joints=num_joints)
    plot_joint_timeseries(pred_6d, gt_6d, geo_seq, args.fps, args.out_dir,
                          num_joints=num_joints)

    # ------------------------------------------------------------------ 3-D visualisation
    try:
        import core.articulate as art
        from core.paths import Paths

        logger.info("core.articulate found — launching body viewer ...")

        smpl_R_gt_t   = torch.from_numpy(smpl_R_gt).float().to(device)   # [N, 24, 3, 3]
        smpl_R_pred_t = smpl_R_gt_t.clone()
        pred_R_dev    = pred_R.to(device)  # [N, J, 3, 3]

        if num_joints == 24:
            # Full body: predicted rotations are already all 24 joints
            smpl_R_pred_t_masked = pred_R_dev
            smpl_R_gt_t_masked   = smpl_R_gt_t
        else:
            # 5-joint arm chain: start from identity for all joints,
            # then fill in the 5 predicted joints.
            identity = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
            smpl_R_pred_t_masked = identity.expand(N, 24, 3, 3).clone()
            smpl_R_gt_t_masked = identity.expand(N, 24, 3, 3).clone()
            for ci, jidx in enumerate(RIGHT_WRIST_CHAIN_JOINTS):
                smpl_R_pred_t_masked[:, jidx] = pred_R_dev[:, ci]
                smpl_R_gt_t_masked[:, jidx]   = smpl_R_gt_t[:, jidx]

        body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=device)

        print("\nLaunching body viewer (close browser tab to exit) ...")
        body_model.view_motion_web_with_gt(
            [smpl_R_pred_t_masked], None, None,
            [smpl_R_gt_t_masked],   None, None,
            fps=args.fps,
        )

    except (ImportError, Exception) as exc:
        logger.warning("Skipping 3-D visualization: %s", exc)
        logger.info("Plots saved to %s", args.out_dir)


if __name__ == "__main__":
    main()
