"""vis_nymeria_odom.py

Top-view 2-D image + MeshCat animation of GT vs predicted root odometry on a
Nymeria real-wrist sequence, driven by a ViT1D odometry model
(e.g. vit1d_odom_nymeria).

The model outputs per-patch root displacements in the Y-up world frame.
These are cumsum'd to obtain a trajectory plotted from a top-down view.

    GREEN  — Ground-truth trajectory
    RED    — Predicted trajectory

Usage
-----
    python scripts/vis_nymeria_odom.py \\
        --pkl          /mnt/nas2/processed_data_test_body_Ty_rwrist/<session>.pkl \\
        --config       configs/base.yaml \\
        --model-config configs/vit1d_odom_nymeria.yaml \\
        --run          vit1d_odom_nymeria \\
        [--ckpt        best]          \\
        [--fps         50]            \\
        [--max-frames  0]             \\
        [--save-fig    topview.png]   \\
        [--no-meshcat]               \\
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
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DOdomDecoder
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_XSENS_ROOT_JOINT = 0  # pelvis

R_WRIST_FROM_FLU = np.array(
    [[-1.0, 0.0, 0.0],
     [ 0.0, 0.0, 1.0],
     [ 0.0, 1.0, 0.0]],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Data loading  (mirrors nymeria_real_wrist_odom_loader_FLU._load_…)
# ---------------------------------------------------------------------------

def _quat_xyzw_to_rotmat(quat: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as ScipyRot
    return ScipyRot.from_quat(quat).as_matrix().astype(np.float64)


def load_nymeria_odom_vis(
    pkl_path: str,
    target_hz: float = 50.0,
    add_gravity: bool = True,
    max_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one Nymeria pkl → (X_flu [N,6], pos_root [N,3], R_world_from_X [N,3,3]).

    pos_root is pelvis world position in the Y-up frame (metres).
    R_world_from_X is the wrist FLU orientation in world frame.
    """
    with open(pkl_path, "rb") as fh:
        raw = pickle.load(fh)

    if not raw.get("has_wrist_imu", False):
        raise RuntimeError(f"{pkl_path}: has_wrist_imu=False")

    wrist_imu  = raw["wrist_imu_data"]
    wrist_gt   = raw["wrist_gt_data"]
    xsens_pose = raw["gt_data"]["xsens_pose"]  # [N, 23, 4, 4]

    body_fps_raw  = float(raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or target_hz)
    wrist_fps_raw = float(raw.get("wrist_imu_sampling_rate_hz", body_fps_raw))

    body_ds  = max(1, round(body_fps_raw  / target_hz))
    wrist_ds = max(1, round(wrist_fps_raw / target_hz))

    def _to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float64)[::body_ds]  # [N_b, 23, 4, 4]

    acc_local  = _to_np(wrist_imu["accel"]).astype(np.float64)[::wrist_ds]
    gyro_local = _to_np(wrist_imu["gyro"]).astype(np.float64)[::wrist_ds]

    orient_raw = wrist_gt["orientation"]
    try:
        import pypose as pp
        if isinstance(orient_raw, pp.LieTensor):
            R_world_from_wrist = orient_raw.matrix().detach().cpu().numpy().astype(np.float64)[::wrist_ds]
        else:
            raise TypeError
    except (ImportError, TypeError):
        orient_np = _to_np(orient_raw).astype(np.float64)[::wrist_ds]
        R_world_from_wrist = _quat_xyzw_to_rotmat(orient_np)

    N = min(xsens_pose.shape[0], acc_local.shape[0])
    if max_frames > 0:
        N = min(N, max_frames)
    xsens_pose         = xsens_pose[:N]
    acc_local          = acc_local[:N]
    gyro_local         = gyro_local[:N]
    R_world_from_wrist = R_world_from_wrist[:N]

    # CPF → world
    acc_world  = np.einsum("nij,nj->ni", R_world_from_wrist, acc_local)
    gyro_world = np.einsum("nij,nj->ni", R_world_from_wrist, gyro_local)

    # add_gravity=True means acc already includes gravity; subtract if False
    if not add_gravity:
        g_world = np.array([0.0, -9.81, 0.0], dtype=np.float64)  # Y-up
        acc_world = acc_world + g_world

    # World → FLU
    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_wrist, R_WRIST_FROM_FLU)
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)
    acc_flu  = np.einsum("nij,nj->ni", R_X_from_world, acc_world)
    gyro_flu = np.einsum("nij,nj->ni", R_X_from_world, gyro_world)

    X_flu = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)

    # Root (pelvis) world position from Xsens 4×4 SE(3) matrices
    pos_root = xsens_pose[:, _XSENS_ROOT_JOINT, :3, 3].astype(np.float32)

    logger.info("Loaded %d frames (%.1f s @ %.0f Hz)", N, N / target_hz, target_hz)
    return X_flu, pos_root, R_world_from_X.astype(np.float32)


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
    """Return per-frame predicted displacement [N, 3] (global world frame).
    Only used for out_dim=3 (global displacement) models.
    """
    T, _       = compute_T_and_hop(cfg)
    patch_size = int((cfg_get(cfg, ["vit1d"], {}) or {}).get("patch_size", 16))
    stats      = PreprocessStats()
    N          = len(X)

    pad   = (T - N % T) % T
    X_pad = np.concatenate([X, np.zeros((pad, 6), dtype=np.float32)]) if pad else X

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
            pred_disp[f0:f1] = pred_np[p] / patch_size

    return pred_disp[:N]


@torch.no_grad()
def run_inference_local_vel(
    X_flu: np.ndarray,
    R_world_from_X_gt: np.ndarray,
    pos_root_gt: np.ndarray,
    encoder: ViT1DEncoder,
    odom_decoder: ViT1DOdomDecoder,
    cfg: dict,
    device: torch.device,
    fps: float,
) -> np.ndarray:
    """Return predicted trajectory [N, 3] for local-velocity model (out_dim=12).

    Integrates FLU-frame gyro from the GT initial orientation, then rotates
    each predicted sub-patch velocity to world frame and sums to a trajectory.
    """
    from scipy.spatial.transform import Rotation as ScipyRot
    T, _       = compute_T_and_hop(cfg)
    patch_size = int((cfg_get(cfg, ["vit1d"], {}) or {}).get("patch_size", 16))
    Q          = patch_size // 4
    dt_Q       = Q / fps
    dt         = 1.0 / fps
    stats      = PreprocessStats()
    N          = len(X_flu)
    P_t        = T // patch_size

    # Gyro integration (body-frame Euler)
    gyro_flu = X_flu[:, 3:6].astype(np.float64)
    R_int    = np.empty((N, 3, 3), dtype=np.float64)
    R_int[0] = R_world_from_X_gt[0].astype(np.float64)
    for t in range(1, N):
        dR       = ScipyRot.from_rotvec(gyro_flu[t - 1] * dt).as_matrix()
        R_int[t] = R_int[t - 1] @ dR

    pad   = (T - N % T) % T
    X_pad = np.concatenate([X_flu, np.zeros((pad, 6), dtype=np.float32)]) if pad else X_flu

    pos_pred    = np.empty((N, 3), dtype=np.float32)
    pos_pred[0] = pos_root_gt[0]
    cursor      = 0   # frame index into pos_pred

    for w in range(len(X_pad) // T):
        start  = w * T
        Xw     = X_pad[start : start + T]
        Xproc  = preprocess_window(Xw, cfg, stats)

        if Xproc is None:
            vel_pred = np.zeros((P_t, 4, 3), dtype=np.float32)
        else:
            x         = torch.from_numpy(Xproc).float().unsqueeze(0).to(device)
            tokens, _ = encoder.forward_tokens(x)
            vel_pred  = odom_decoder(tokens).squeeze(0).cpu().numpy().reshape(P_t, 4, 3)

        for p in range(P_t):
            for k in range(4):
                t_s   = start + p * patch_size + k * Q
                t_mid = t_s + Q // 2
                R_mid = R_int[min(t_mid, N - 1)]
                v_flu = vel_pred[p, k].astype(np.float64)
                delta = (R_mid @ v_flu * dt_Q).astype(np.float32)

                # Fill Q sub-patch frames with constant velocity
                for f in range(Q):
                    frame = t_s + f
                    if frame >= N:
                        break
                    if frame > 0:
                        pos_pred[frame] = pos_pred[frame - 1] + delta / Q

    return pos_pred


# ---------------------------------------------------------------------------
# 2-D top-view image  (Y-up world → ground plane is XZ)
# ---------------------------------------------------------------------------

def save_topview(
    pos_gt: np.ndarray,    # [N, 3] Y-up world positions
    pos_pred: np.ndarray,  # [N, 3] Y-up world positions
    out_path: str,
    seq_name: str = "",
) -> None:
    """Save a top-down view of GT and predicted trajectories.

    In the Y-up world frame the ground plane is XZ:
      X → right in the image
      Z → into the page → negated so it points upward in the image
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot( pos_gt[:, 0],  -pos_gt[:, 2],   color="#00AA44", linewidth=1.5,
            label="Ground truth", zorder=2)
    ax.plot(pos_pred[:, 0], -pos_pred[:, 2], color="#DD2222", linewidth=1.5,
            label="Predicted",    zorder=2)

    for xy, color in [(pos_gt[0],   "#00AA44"), (pos_pred[0], "#DD2222")]:
        ax.plot(xy[0], -xy[2], "o", color=color, markersize=7, zorder=3)
    for xy, color in [(pos_gt[-1],  "#00AA44"), (pos_pred[-1], "#DD2222")]:
        ax.plot(xy[0], -xy[2], "s", color=color, markersize=7, zorder=3)

    start_patch = mpatches.Patch(color="grey", label="○ = start")
    end_patch   = mpatches.Patch(color="grey", label="□ = end")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [start_patch, end_patch], loc="best", fontsize=10)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("−Z (m)", fontsize=11)
    title = "Top-view trajectory (ground plane)"
    if seq_name:
        title += f"\n{seq_name}"
    ax.set_title(title, fontsize=12)
    ax.grid(True, linewidth=0.4, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Top-view image saved → %s", out_path)


# ---------------------------------------------------------------------------
# MeshCat animation
# ---------------------------------------------------------------------------

def visualize(
    pos_gt: np.ndarray,    # [N, 3] Y-up world positions
    pos_pred: np.ndarray,  # [N, 3] Y-up world positions (pre-built trajectory)
    fps: float,
):
    import meshcat
    import meshcat.geometry as mcg
    import meshcat.transformations as mct

    N   = pos_gt.shape[0]
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

    # Nymeria is Y-up; rotate scene so Y-up → Z-up for MeshCat
    vis[ROOT].set_transform(mct.rotation_matrix(np.pi / 2.0, [1, 0, 0]))

    gt_line = mcg.Line(
        mcg.PointsGeometry(pos_gt.T.astype(np.float32)),
        mcg.LineBasicMaterial(color=0x00AA44, linewidth=2),
    )
    pred_line = mcg.Line(
        mcg.PointsGeometry(pos_pred.T.astype(np.float32)),
        mcg.LineBasicMaterial(color=0xDD2222, linewidth=2),
    )
    vis[f"{ROOT}/gt_traj"].set_object(gt_line)
    vis[f"{ROOT}/pred_traj"].set_object(pred_line)

    sphere   = mcg.Sphere(radius=0.05)
    gt_mat   = mcg.MeshLambertMaterial(color=0x00AA44)
    pred_mat = mcg.MeshLambertMaterial(color=0xDD2222)
    vis[f"{ROOT}/gt_marker"].set_object(sphere,   gt_mat)
    vis[f"{ROOT}/pred_marker"].set_object(sphere, pred_mat)

    dt = 1.0 / fps
    logger.info("Animating %d frames at %.0f fps …", N, fps)

    for f in range(N):
        vis[f"{ROOT}/gt_marker"].set_transform(
            mct.translation_matrix(pos_gt[f].tolist())
        )
        vis[f"{ROOT}/pred_marker"].set_transform(
            mct.translation_matrix(pos_pred[f].tolist())
        )
        sleep(dt)

    logger.info("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    ap = argparse.ArgumentParser(
        description="Visualize GT vs predicted odometry on Nymeria in MeshCat + PNG"
    )
    ap.add_argument("--pkl",          required=True,
                    help="Nymeria pkl with has_wrist_imu=True")
    ap.add_argument("--config",       default="configs/base.yaml")
    ap.add_argument("--model-config", required=True,
                    help="Model config yaml (e.g. configs/vit1d_odom_nymeria.yaml)")
    ap.add_argument("--run",          required=True,
                    help="Run name under runs/<run>/checkpoints/")
    ap.add_argument("--ckpt",         default="best",
                    help='"best" | "latest" | absolute path to .pt file')
    ap.add_argument("--fps",          type=float, default=50.0)
    ap.add_argument("--max-frames",   type=int,   default=0)
    ap.add_argument("--save-fig",     default="",
                    help="Path for the top-view PNG. "
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

    out_dim = int(state["odom_decoder"]["head.weight"].shape[0])
    odom_decoder = ViT1DOdomDecoder(
        embed_dim=encoder.embed_dim,
        num_patches_per_channel=T_win // patch_size,
        in_channels=encoder.in_channels,
        cfg=ckpt_cfg,
        out_dim=out_dim,
    ).to(device)
    odom_decoder.load_state_dict(state["odom_decoder"])
    odom_decoder.eval()
    use_local_vel = (out_dim == 12)

    # Config key varies by objective type
    _obj_type   = cfg_get(ckpt_cfg, ["objective", "type"], "odom_nymeria")
    odom_cfg    = (cfg_get(ckpt_cfg, ["objective", _obj_type], {}) or {})
    target_hz   = float(odom_cfg.get("target_hz", args.fps))
    add_gravity = bool(odom_cfg.get("add_gravity", True))

    logger.info("Loading: %s  (target_hz=%.0f  add_gravity=%s)",
                os.path.basename(args.pkl), target_hz, add_gravity)
    X_flu, pos_root_gt, R_world_from_X = load_nymeria_odom_vis(
        pkl_path    = args.pkl,
        target_hz   = target_hz,
        add_gravity = add_gravity,
        max_frames  = args.max_frames,
    )

    logger.info("Running inference … (out_dim=%d)", out_dim)
    if use_local_vel:
        logger.info("Local-velocity model: using gyro integration.")
        pos_pred = run_inference_local_vel(
            X_flu, R_world_from_X, pos_root_gt,
            encoder, odom_decoder, ckpt_cfg, device, target_hz,
        )
    else:
        pred_disp = run_inference(X_flu, encoder, odom_decoder, ckpt_cfg, device)
        pos_pred  = np.cumsum(pred_disp, axis=0)
        pos_pred  = pos_pred - pos_pred[0] + pos_root_gt[0]

    # ---- 2-D top-view image ----
    fig_path = args.save_fig or (
        os.path.splitext(os.path.basename(args.pkl))[0] + "_topview.png"
    )
    save_topview(pos_root_gt, pos_pred, fig_path, seq_name=os.path.basename(args.pkl))

    if not args.no_meshcat:
        logger.info("Launching MeshCat …")
        visualize(pos_root_gt, pos_pred, target_hz)


if __name__ == "__main__":
    main()
