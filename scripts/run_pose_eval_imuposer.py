"""Single-sequence pose evaluation/visualization for IMUPoser FLU data.

This script is similar to scripts/run_pose_eval.py but loads IMUPoser sequence
PKLs via imu_lm.data.imuposer_wrist_loader_FLU.load_session_flu, so it does
not expect gt_data['xsens_pose'].

Outputs:
  - Per-joint geodesic + 6D MSE summary in terminal
  - Per-joint plots in --out-dir
  - Optional browser 3D visualization (pred vs GT) if core.articulate exists

Usage:
  python scripts/run_pose_eval_imuposer.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_pose_finetune_imuposer_realdata_nymeria_real_wrist_flu.yaml \
    --run vit1d_pose_finetune_imuposer_realdata_nymeria_real_wrist_flu \
    --pkl /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/imuposer_seq_00026.pkl \
    --ckpt best \
    --max-frames 5000 \
    --fps 50 \
    --out-dir data_vis/pose_eval_imuposer
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.imuposer_wrist_loader_FLU import _RIGHT_ARM_CHAIN, load_session_flu
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DPoseDecoder
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

logger = logging.getLogger(__name__)

JOINT_NAMES = {
    14: "R-Collar",
    17: "R-Shoulder",
    19: "R-Elbow",
    21: "R-Wrist",
    23: "R-Hand",
}


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 matrices."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def geodesic_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Angular distance (degrees) between two rotation-matrix tensors."""
    R_diff = torch.matmul(R_pred, R_gt.transpose(-1, -2))
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


@torch.no_grad()
def run_inference(
    X: np.ndarray,
    encoder: ViT1DEncoder,
    pose_decoder: ViT1DPoseDecoder,
    cfg,
    device: torch.device,
) -> np.ndarray:
    """Slide non-overlapping windows and return per-frame 6D predictions [N, J, 6]."""
    T, _ = compute_T_and_hop(cfg)
    vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))
    num_joints = int(cfg_get(cfg, ["objective", "pose", "num_joints"], 5))
    stats = PreprocessStats()
    N = len(X)

    pad = (T - N % T) % T
    X_pad = np.concatenate([X, np.zeros((pad, 6), dtype=np.float32)], axis=0) if pad else X

    encoder.eval()
    pose_decoder.eval()

    pred_6d_full = np.zeros((len(X_pad), num_joints, 6), dtype=np.float32)

    n_windows = len(X_pad) // T
    for w in range(n_windows):
        start = w * T
        Xw = X_pad[start:start + T]  # [T, 6]

        Xproc = preprocess_window(Xw, cfg, stats)
        if Xproc is None:
            continue

        x = torch.from_numpy(Xproc).float().unsqueeze(0).to(device)  # [1, 6, T]

        tokens, _ = encoder.forward_tokens(x)   # [1, C*P_t, D]
        pred = pose_decoder(tokens)             # [1, P_t, J*6]
        pred_np = pred.squeeze(0).cpu().numpy() # [P_t, J*6]

        for p in range(pred_np.shape[0]):
            f0 = start + p * patch_size
            f1 = f0 + patch_size
            pred_6d_full[f0:f1] = pred_np[p].reshape(num_joints, 6)

    return pred_6d_full[:N]


def _joint_names(num_joints: int) -> list[str]:
    if num_joints == 5:
        return [JOINT_NAMES[j] for j in _RIGHT_ARM_CHAIN]
    return [f"J{i}" for i in range(num_joints)]


def plot_joint_error(geodesic: np.ndarray, mse: np.ndarray, out_dir: str, num_joints: int):
    names = _joint_names(num_joints)
    x = np.arange(len(names))

    fig_w = max(12, len(names) * 0.55)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, 5))

    ax1.bar(x, geodesic, color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_ylabel("Mean Geodesic Error (deg)")
    ax1.set_title("Per-Joint Geodesic Error")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, mse, color="tomato")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("MSE (6D)")
    ax2.set_title("Per-Joint 6D MSE")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "joint_error_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved %s", path)


def plot_joint_timeseries(
    pred_6d: np.ndarray,
    gt_6d: np.ndarray,
    geodesic_seq: np.ndarray,
    fps: int,
    out_dir: str,
    num_joints: int,
    max_plot_frames: int = 1500,
):
    n = min(len(pred_6d), max_plot_frames)
    time = np.arange(n) / fps
    names = _joint_names(num_joints)

    for ji, jname in enumerate(names):
        fig, axes = plt.subplots(7, 1, figsize=(14, 18))

        for d in range(6):
            ax = axes[d]
            ax.plot(time, gt_6d[:n, ji, d], "b-", label="GT", linewidth=1.2, alpha=0.8)
            ax.plot(time, pred_6d[:n, ji, d], "r--", label="Pred", linewidth=1.0, alpha=0.8)
            ax.set_ylabel(f"6D[{d}]")
            ax.grid(True, alpha=0.3)
            if d == 0:
                ax.set_title(f"{jname} - 6D components & geodesic error")
                ax.legend(loc="upper right")

        ax = axes[6]
        ax.plot(time, geodesic_seq[:n, ji], "g-", linewidth=1.0)
        ax.axhline(
            geodesic_seq[:n, ji].mean(),
            color="r",
            linestyle="--",
            label=f"mean={geodesic_seq[:n, ji].mean():.1f} deg",
        )
        ax.set_ylabel("Geodesic (deg)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"joint_{ji:02d}_{jname.replace('-', '').replace(' ', '_')}.png"
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=130)
        plt.close()
        logger.info("Saved %s", path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    ap = argparse.ArgumentParser(description="Evaluate ViT1D pose decoder on one IMUPoser FLU sequence")
    ap.add_argument("--config", required=True, help="Base config YAML")
    ap.add_argument("--model-config", required=True, help="Model config YAML")
    ap.add_argument("--run", required=True, help="Run name under runs/<run>")
    ap.add_argument("--pkl", required=True, help="Path to one IMUPoser .pkl sequence")
    ap.add_argument("--ckpt", default="best", help="best | latest | abs path")
    ap.add_argument("--max-frames", type=int, default=0, help="Max frames to evaluate (0 = full)")
    ap.add_argument("--fps", type=int, default=50, help="Target sampling rate for loader/inference")
    ap.add_argument("--out-dir", default="data_vis/pose_eval_imuposer")
    ap.add_argument("--add-gravity", action="store_true", help="Add gravity back to accelerometer")
    args = ap.parse_args()

    base_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    cfg = deep_update(base_cfg, model_cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs_root = cfg.get("paths", {}).get("runs_root", "runs")
    run_dir = os.path.join(runs_root, args.run)
    ckpt_path = os.path.join(run_dir, "checkpoints", f"{args.ckpt}.pt") if args.ckpt in ("best", "latest") else args.ckpt
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = state["cfg"]

    encoder = ViT1DEncoder(ckpt_cfg).to(device)
    encoder.load_state_dict(state["model"])
    encoder.eval()

    vit_cfg = cfg_get(ckpt_cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))
    T_win, _ = compute_T_and_hop(ckpt_cfg)
    num_joints = int(cfg_get(ckpt_cfg, ["objective", "pose", "num_joints"], 5))

    pose_decoder = ViT1DPoseDecoder(
        embed_dim=encoder.embed_dim,
        num_patches_per_channel=T_win // patch_size,
        in_channels=encoder.in_channels,
        num_joints=num_joints,
        cfg=ckpt_cfg,
    ).to(device)
    pose_decoder.load_state_dict(state["pose_decoder"])
    pose_decoder.eval()

    logger.info("Encoder   : embed_dim=%d  in_channels=%d", encoder.embed_dim, encoder.in_channels)
    logger.info("PoseDecoder: num_joints=%d  patches/ch=%d", num_joints, T_win // patch_size)

    logger.info("Loading IMUPoser sequence: %s", os.path.basename(args.pkl))
    loaded = load_session_flu(args.pkl, target_hz=args.fps, add_gravity=args.add_gravity)
    if loaded is None:
        raise SystemExit(f"Failed to load IMUPoser sequence: {args.pkl}")

    X, gt_6d = loaded  # X [N,6], gt_6d [N,5,6] for IMUPoser arm chain
    if args.max_frames > 0:
        n = min(args.max_frames, len(X))
        X = X[:n]
        gt_6d = gt_6d[:n]

    N = len(X)
    logger.info("Sequence length: %d frames (%.1f s)", N, N / float(args.fps))

    logger.info("Running inference ...")
    pred_6d = run_inference(X, encoder, pose_decoder, cfg, device)

    pred_t = torch.from_numpy(pred_6d)
    gt_t = torch.from_numpy(gt_6d)

    pred_R = rotation_6d_to_matrix(pred_t)  # [N, J, 3, 3]
    gt_R = rotation_6d_to_matrix(gt_t)      # [N, J, 3, 3]

    geo_seq = geodesic_error_deg(
        pred_R.reshape(-1, 3, 3), gt_R.reshape(-1, 3, 3)
    ).reshape(N, num_joints).numpy()

    mse_per_joint = ((pred_6d - gt_6d) ** 2).mean(axis=(0, 2))

    names = _joint_names(num_joints)
    print("\n=== Per-Joint Evaluation (IMUPoser) ===")
    print(f"{'Joint':<16} {'Geodesic (deg)':>15} {'Geodesic std':>13} {'6D-MSE':>10}")
    print("-" * 60)
    for ji, name in enumerate(names):
        print(
            f"{name:<16} {geo_seq[:, ji].mean():>15.2f} "
            f"{geo_seq[:, ji].std():>13.2f} {mse_per_joint[ji]:>10.5f}"
        )
    print("-" * 60)
    print(
        f"{'Mean':<16} {geo_seq.mean():>15.2f} "
        f"{geo_seq.std():>13.2f} {mse_per_joint.mean():>10.5f}"
    )

    plot_joint_error(geo_seq.mean(0), mse_per_joint, args.out_dir, num_joints=num_joints)
    plot_joint_timeseries(
        pred_6d,
        gt_6d,
        geo_seq,
        fps=args.fps,
        out_dir=args.out_dir,
        num_joints=num_joints,
    )

    try:
        import core.articulate as art  # noqa: F401
        from core.paths import Paths  # noqa: F401

        logger.info("core.articulate found - launching body viewer ...")
        pred_R_dev = pred_R.to(device)
        gt_R_dev = gt_R.to(device)

        identity = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)  # [1,1,3,3]
        pred_full = identity.expand(N, 24, 3, 3).clone()
        gt_full = identity.expand(N, 24, 3, 3).clone()
        for ci, jidx in enumerate(_RIGHT_ARM_CHAIN):
            pred_full[:, jidx] = pred_R_dev[:, ci]
            gt_full[:, jidx] = gt_R_dev[:, ci]

        body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=device)

        print("\nLaunching body viewer (close browser tab to exit) ...")
        body_model.view_motion_web_with_gt(
            [pred_full], None, None,
            [gt_full], None, None,
            fps=args.fps,
        )
    except Exception as exc:
        logger.warning("Skipping 3D visualization: %s", exc)
        logger.info("Plots saved to %s", args.out_dir)


if __name__ == "__main__":
    main()
