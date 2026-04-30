"""vis_imuposer_odom.py

MeshCat visualization of GT vs predicted root odometry on an IMUPoser
sequence, driven by a ViT1D odometry model (e.g. vit1d_odom_imuposer_flu).

The model outputs per-patch root displacements in the Z-up world frame.
These are cumsum'd to obtain a trajectory that is plotted as a coloured
line in the ground plane.

    GREEN  — Ground-truth trajectory
    RED    — Predicted trajectory

A small sphere marker traces the current position along each trajectory.

Usage
-----
    python scripts/vis_imuposer_odom.py \\
        --pkl          /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/imuposer_seq_00000.pkl \\
        --config       configs/base.yaml \\
        --model-config configs/vit1d_odom_imuposer_flu.yaml \\
        --run          vit1d_odom_imuposer_flu \\
        [--ckpt        best]     \\
        [--fps         50]       \\
        [--max-frames  0]        \\
        [--device      cuda]
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
from time import sleep

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.imuposerdataset import (
    _resample_linear,
    _resample_nearest,
    _resample_rotmat_nlerp,
    _resample_rotmat_nearest,
)
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DOdomDecoder
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (match imuposer_wrist_odom_loader_FLU.py)
# ---------------------------------------------------------------------------
_SMPL_R_WRIST     = 21
_SMPL_PELVIS      = 0
_IMUPOSER_R_WRIST = 1

R_WRIST_FROM_FLU = np.array(
    [[-1.0,  0.0,  0.0],
     [ 0.0,  0.0,  1.0],
     [ 0.0,  1.0,  0.0]],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_imuposer_odom_vis(
    pkl_path: str,
    target_hz: float = 50.0,
    add_gravity: bool = False,
    max_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one IMUPoser pkl → (X_flu [N,6], pos_root [N,3]).

    pos_root is pelvis world position in the Z-up frame (metres).
    """
    from scipy.spatial.transform import Rotation as ScipyRot
    import core.articulate as art
    from core.paths import Paths

    with open(pkl_path, "rb") as fh:
        seq = pickle.load(fh)

    gt   = seq.get("gt_data", {})
    ri   = seq.get("real_imu_data", {})

    pose_local  = np.asarray(gt["pose"],     dtype=np.float32)
    joint_pos_w = np.asarray(gt["joint_pos"],dtype=np.float32)
    real_acc    = np.asarray(ri["real_acc"], dtype=np.float32)
    real_ori    = np.asarray(ri["real_ori"], dtype=np.float32)

    fps_raw = float(seq.get("sampling_rate_hz", target_hz))
    N = pose_local.shape[0]
    if max_frames > 0:
        N = min(N, max_frames)
    pose_local  = pose_local[:N]
    joint_pos_w = joint_pos_w[:N]
    real_acc    = real_acc[:N]
    real_ori    = real_ori[:N]

    if abs(fps_raw - target_hz) > 1e-3:
        new_len = int(round((N - 1) * target_hz / fps_raw)) + 1
        up = target_hz > fps_raw
        pose_local  = (_resample_rotmat_nearest if up else _resample_rotmat_nlerp)(
            torch.from_numpy(pose_local), new_len).numpy()
        joint_pos_w = (_resample_nearest if up else _resample_linear)(
            torch.from_numpy(joint_pos_w), new_len).numpy()
        real_acc    = (_resample_nearest if up else _resample_linear)(
            torch.from_numpy(real_acc), new_len).numpy()
        real_ori    = (_resample_rotmat_nearest if up else _resample_rotmat_nlerp)(
            torch.from_numpy(real_ori), new_len).numpy()
        N = new_len

    fps = float(target_hz)

    # SMPL FK → world-frame wrist rotation
    body_model = art.model.ParametricModel(str(Paths.SMPL_FILE), device=torch.device("cpu"))
    root_trans = torch.from_numpy(joint_pos_w[:, 0])
    pose_local_t = torch.from_numpy(pose_local.astype(np.float32))
    pose_global_t, _ = body_model.forward_kinematics_masked(
        pose_local_t, tran=root_trans, calc_mesh=False
    )
    R_world_from_wrist = pose_global_t[:, _SMPL_R_WRIST].numpy().astype(np.float64)

    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_wrist, R_WRIST_FROM_FLU)
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)

    acc_world = real_acc[:, _IMUPOSER_R_WRIST].astype(np.float64)
    if add_gravity:
        g_world = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        acc_world = acc_world - g_world
    acc_flu = np.einsum("nij,nj->ni", R_X_from_world, acc_world)

    R_world_from_sensor = real_ori[:, _IMUPOSER_R_WRIST].astype(np.float64)
    dR = np.einsum("nij,njk->nik",
                   R_world_from_sensor[:-1].transpose(0, 2, 1),
                   R_world_from_sensor[1:])
    omega_sensor = (
        ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
        .as_rotvec().astype(np.float64) * fps
    )
    omega_full = np.zeros((N, 3), dtype=np.float64)
    omega_full[:-1] = omega_sensor
    omega_full[-1]  = omega_sensor[-1]
    omega_world = np.einsum("nij,nj->ni", R_world_from_sensor, omega_full)
    gyro_flu    = np.einsum("nij,nj->ni", R_X_from_world, omega_world)

    X_flu    = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)
    pos_root = joint_pos_w[:, _SMPL_PELVIS].astype(np.float32)
    logger.info("Loaded %d frames (%.1f s @ %.0f Hz)", N, N / target_hz, target_hz)
    return X_flu, pos_root


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    X: np.ndarray,
    encoder: ViT1DEncoder,
    odom_decoder: ViT1DOdomDecoder,
    cfg: dict,
    device: torch.device,
) -> np.ndarray:
    """Return per-frame predicted displacement [N, 3] (one value per patch, repeated)."""
    T, _       = compute_T_and_hop(cfg)
    patch_size = int((cfg_get(cfg, ["vit1d"], {}) or {}).get("patch_size", 16))
    stats      = PreprocessStats()
    N          = len(X)

    pad   = (T - N % T) % T
    X_pad = np.concatenate([X, np.zeros((pad, 6), dtype=np.float32)]) if pad else X

    # pred_disp: per-patch displacement, broadcast to per-frame
    pred_disp = np.zeros((len(X_pad), 3), dtype=np.float32)
    for w in range(len(X_pad) // T):
        start     = w * T
        Xw        = X_pad[start : start + T]
        Xproc     = preprocess_window(Xw, cfg, stats)
        if Xproc is None:
            continue
        x         = torch.from_numpy(Xproc).float().unsqueeze(0).to(device)
        tokens, _ = encoder.forward_tokens(x)
        pred_np   = odom_decoder(tokens).squeeze(0).cpu().numpy()  # [P, 3]
        for p in range(pred_np.shape[0]):
            f0 = start + p * patch_size
            f1 = f0 + patch_size
            # distribute the patch displacement evenly across its frames
            pred_disp[f0:f1] = pred_np[p] / patch_size

    return pred_disp[:N]


# ---------------------------------------------------------------------------
# 2-D top-view image
# ---------------------------------------------------------------------------

def save_topview(
    pos_gt: np.ndarray,    # [N, 3] Z-up world positions
    pos_pred: np.ndarray,  # [N, 3] Z-up world positions
    out_path: str,
    seq_name: str = "",
) -> None:
    """Save a top-down (XY-plane) view of GT and predicted trajectories.

    In the Z-up world frame, the ground plane is XY:
      X → East-ish  (right in image)
      Y → North-ish (up in image)
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # --- trajectories ---
    ax.plot(pos_gt[:, 0],   pos_gt[:, 1],   color="#00AA44", linewidth=1.5,
            label="Ground truth", zorder=2)
    ax.plot(pos_pred[:, 0], pos_pred[:, 1], color="#DD2222", linewidth=1.5,
            label="Predicted",    zorder=2)

    # --- start / end markers ---
    for xy, color in [(pos_gt[0],   "#00AA44"),
                      (pos_pred[0], "#DD2222")]:
        ax.plot(xy[0], xy[1], "o", color=color, markersize=7, zorder=3)
    for xy, color in [(pos_gt[-1],   "#00AA44"),
                      (pos_pred[-1], "#DD2222")]:
        ax.plot(xy[0], xy[1], "s", color=color, markersize=7, zorder=3)

    # legend entry for start/end marker shapes
    start_patch = mpatches.Patch(color="grey", label="○ = start")
    end_patch   = mpatches.Patch(color="grey", label="□ = end")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [start_patch, end_patch],
              loc="best", fontsize=10)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    title = "Top-view trajectory"
    if seq_name:
        title += f"\n{seq_name}"
    ax.set_title(title, fontsize=12)
    ax.grid(True, linewidth=0.4, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Top-view image saved → %s", out_path)


# ---------------------------------------------------------------------------
# MeshCat visualization
# ---------------------------------------------------------------------------

def visualize(
    pos_root_gt: np.ndarray,    # [N, 3] pelvis world position (Z-up)
    pred_disp:   np.ndarray,    # [N, 3] per-frame predicted displacement (Z-up)
    fps: float,
):
    """Animate GT (green) and predicted (red) trajectories as line + marker."""
    import meshcat
    import meshcat.geometry as mcg
    import meshcat.transformations as mct

    N   = pos_root_gt.shape[0]
    vis = meshcat.Visualizer()
    print("\nMeshCat at:", vis.url())
    print("Open the URL above in a browser, then wait for animation.\n")
    print("  GREEN — Ground-truth trajectory")
    print("  RED   — Predicted trajectory\n")

    vis["/Background"].set_property("top_color",    [1, 1, 1])
    vis["/Background"].set_property("bottom_color", [1, 1, 1])

    ROOT = "scene"
    try:
        vis[ROOT].delete()
    except Exception:
        pass

    # IMUPoser world frame is Z-up → MeshCat is also Z-up; no rotation needed.

    # Build predicted trajectory via cumsum of per-frame displacements
    pos_pred = np.cumsum(pred_disp, axis=0)
    # align start to GT start
    pos_pred = pos_pred - pos_pred[0] + pos_root_gt[0]

    # ---- Draw full trajectories as polylines ----
    gt_line   = mcg.Line(
        mcg.PointsGeometry(pos_root_gt.T.astype(np.float32)),
        mcg.LineBasicMaterial(color=0x00AA44, linewidth=2),
    )
    pred_line = mcg.Line(
        mcg.PointsGeometry(pos_pred.T.astype(np.float32)),
        mcg.LineBasicMaterial(color=0xDD2222, linewidth=2),
    )
    vis[f"{ROOT}/gt_traj"].set_object(gt_line)
    vis[f"{ROOT}/pred_traj"].set_object(pred_line)

    # ---- Animated markers ----
    sphere    = mcg.Sphere(radius=0.05)
    gt_mat    = mcg.MeshLambertMaterial(color=0x00AA44)
    pred_mat  = mcg.MeshLambertMaterial(color=0xDD2222)
    vis[f"{ROOT}/gt_marker"].set_object(sphere,   gt_mat)
    vis[f"{ROOT}/pred_marker"].set_object(sphere, pred_mat)

    dt = 1.0 / fps
    logger.info("Animating %d frames at %.0f fps …", N, fps)

    for f in range(N):
        p_gt   = pos_root_gt[f]
        p_pred = pos_pred[f]

        vis[f"{ROOT}/gt_marker"].set_transform(
            mct.translation_matrix([p_gt[0],   p_gt[1],   p_gt[2]])
        )
        vis[f"{ROOT}/pred_marker"].set_transform(
            mct.translation_matrix([p_pred[0], p_pred[1], p_pred[2]])
        )
        sleep(dt)

    logger.info("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    ap = argparse.ArgumentParser(
        description="Visualize GT vs predicted odometry on IMUPoser in MeshCat"
    )
    ap.add_argument("--pkl",          required=True,
                    help="IMUPoser pkl (e.g. .../test/imuposer_seq_00000.pkl)")
    ap.add_argument("--config",       default="configs/base.yaml")
    ap.add_argument("--model-config", required=True,
                    help="Model config yaml (e.g. configs/vit1d_odom_imuposer_flu.yaml)")
    ap.add_argument("--run",          required=True,
                    help="Run name under runs/<run>/checkpoints/")
    ap.add_argument("--ckpt",         default="best",
                    help='"best" | "latest" | absolute path to .pt file')
    ap.add_argument("--fps",          type=float, default=50.0)
    ap.add_argument("--max-frames",   type=int,   default=0)
    ap.add_argument("--save-fig",     default="",
                    help="Path for the 2-D top-view PNG. "
                         "Defaults to <pkl_basename>_topview.png in the cwd.")
    ap.add_argument("--no-meshcat",   action="store_true",
                    help="Skip the interactive MeshCat animation (image only).")
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info("Device: %s", device)

    base_cfg  = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    cfg       = deep_update(base_cfg, model_cfg)

    runs_root = cfg.get("paths", {}).get("runs_root", "runs")
    if args.ckpt in ("best", "latest"):
        ckpt_path = os.path.join(runs_root, args.run, "checkpoints", f"{args.ckpt}.pt")
    else:
        ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    state    = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = state["cfg"]

    encoder = ViT1DEncoder(ckpt_cfg).to(device)
    encoder.load_state_dict(state["model"])
    encoder.eval()

    T_win, _   = compute_T_and_hop(ckpt_cfg)
    patch_size = int((cfg_get(ckpt_cfg, ["vit1d"], {}) or {}).get("patch_size", 16))

    odom_decoder = ViT1DOdomDecoder(
        embed_dim=encoder.embed_dim,
        num_patches_per_channel=T_win // patch_size,
        in_channels=encoder.in_channels,
        cfg=ckpt_cfg,
    ).to(device)
    odom_decoder.load_state_dict(state["odom_decoder"])
    odom_decoder.eval()

    odom_cfg    = cfg_get(ckpt_cfg, ["objective", "odom_imuposer_flu"], {}) or {}
    target_hz   = float(odom_cfg.get("target_hz", args.fps))
    add_gravity = bool(odom_cfg.get("add_gravity", False))

    logger.info("Loading: %s  (target_hz=%.0f  add_gravity=%s)",
                os.path.basename(args.pkl), target_hz, add_gravity)
    X_flu, pos_root_gt = load_imuposer_odom_vis(
        pkl_path    = args.pkl,
        target_hz   = target_hz,
        add_gravity = add_gravity,
        max_frames  = args.max_frames,
    )

    logger.info("Running inference …")
    pred_disp = run_inference(X_flu, encoder, odom_decoder, ckpt_cfg, device)

    # Build predicted trajectory (aligned to GT start)
    pos_pred = np.cumsum(pred_disp, axis=0)
    pos_pred = pos_pred - pos_pred[0] + pos_root_gt[0]

    # ---- 2-D top-view image ----
    fig_path = args.save_fig or (
        os.path.splitext(os.path.basename(args.pkl))[0] + "_topview.png"
    )
    save_topview(pos_root_gt, pos_pred, fig_path,
                 seq_name=os.path.basename(args.pkl))

    if not args.no_meshcat:
        logger.info("Launching MeshCat …")
        visualize(pos_root_gt, pred_disp, target_hz)


if __name__ == "__main__":
    main()
