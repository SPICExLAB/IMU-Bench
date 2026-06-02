"""vis_nymeria_full_body_pose_24j_tpose.py

Side-by-side MeshCat animation for the 24-joint sub-patch pose model
(vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_24j_tpose).

    LEFT  (blue)   — Ground-truth full-body SMPL pose
    RIGHT (orange) — Predicted full-body SMPL pose

Root orientation alignment
--------------------------
The model was trained with a T-pose reset: at the start of every 2.56 s window
the GT root (pelvis) orientation is normalised to identity before computing the
target 6D labels.  The model therefore predicts root rotation *relative to the
window's initial orientation* (R_pred_rel = R0⁻¹ @ R_gt).

To visualise with correct global orientation we invert this per-window:
    R_pred_abs[f] = R0 @ R_pred_rel[f]
where R0 = GT root rotation at the first sub-patch centre of that window.
This reconstructs the absolute pelvis orientation so the predicted body tracks
the ground-truth motion trajectory.

Usage
-----
    python scripts/vis_nymeria_full_body_pose_24j_tpose.py \\
        --pkl          /mnt/nas2/processed_data_test_body_Ty_rwrist/<session>.pkl \\
        --config       configs/base.yaml \\
        --model-config configs/vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_24j_tpose.yaml \\
        --run          vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_24j_tpose \\
        [--ckpt        best]
        [--fps         50]
        [--max-frames  500]
        [--gap         1.5]
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
import torch.nn.functional as F

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.nymeria_real_wrist_loader_FLU import (
    R_WRIST_FROM_FLU,
    _quat_xyzw_to_rotmat,
    xsens_to_SMPL,
)
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DPoseDecoder
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """[..., 6] → [..., 3, 3]."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sequence(pkl_path: str, target_hz: float, add_gravity: bool,
                  max_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Load Nymeria pkl → (X_flu [N,6], R_local [N,24,3,3])."""
    import core.articulate as art
    from core.paths import Paths

    with open(pkl_path, "rb") as fh:
        raw = pickle.load(fh)

    if not raw.get("has_wrist_imu", False):
        raise RuntimeError(f"{pkl_path}: has_wrist_imu=False")

    wrist_imu  = raw["wrist_imu_data"]
    wrist_gt   = raw["wrist_gt_data"]
    xsens_pose = raw["gt_data"]["xsens_pose"]

    body_fps  = float(raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or target_hz)
    wrist_fps = float(raw.get("wrist_imu_sampling_rate_hz", body_fps))
    body_ds   = max(1, round(body_fps  / target_hz))
    wrist_ds  = max(1, round(wrist_fps / target_hz))

    def _np(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float64)[::body_ds]

    acc_local  = _np(wrist_imu["accel"]).astype(np.float64)[::wrist_ds]
    gyro_local = _np(wrist_imu["gyro"]).astype(np.float64)[::wrist_ds]

    orient_raw = wrist_gt["orientation"]
    try:
        import pypose as pp
        if isinstance(orient_raw, pp.LieTensor):
            R_w2w = orient_raw.matrix().detach().cpu().numpy().astype(np.float64)[::wrist_ds]
        else:
            raise TypeError
    except (ImportError, TypeError):
        R_w2w = _quat_xyzw_to_rotmat(_np(orient_raw).astype(np.float64)[::wrist_ds])

    N = min(xsens_pose.shape[0], acc_local.shape[0])
    if max_frames > 0:
        N = min(N, max_frames)
    xsens_pose = xsens_pose[:N]
    acc_local  = acc_local[:N]
    gyro_local = gyro_local[:N]
    R_w2w      = R_w2w[:N]

    acc_world  = np.einsum("nij,nj->ni", R_w2w, acc_local)
    gyro_world = np.einsum("nij,nj->ni", R_w2w, gyro_local)
    if not add_gravity:
        acc_world = acc_world + np.array([0., -9.81, 0.])

    R_world_from_X = np.einsum("nij,jk->nik", R_w2w, R_WRIST_FROM_FLU)
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)
    acc_flu  = np.einsum("nij,nj->ni", R_X_from_world, acc_world)
    gyro_flu = np.einsum("nij,nj->ni", R_X_from_world, gyro_world)
    X_flu = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)

    body_model = art.model.ParametricModel(str(Paths.SMPL_FILE), device=torch.device("cpu"))
    xsens_R       = xsens_pose[:, :, :3, :3].astype(np.float32)
    smpl_global_R = xsens_to_SMPL(xsens_R)                          # [N, 24, 3, 3]
    smpl_local_t  = body_model.inverse_kinematics_R(
        torch.from_numpy(smpl_global_R).float()
    )
    R_local = smpl_local_t.cpu().numpy().astype(np.float32)          # [N, 24, 3, 3]

    logger.info("Loaded %d frames (%.1f s @ %.0f Hz)", N, N / target_hz, target_hz)
    return X_flu, R_local


# ---------------------------------------------------------------------------
# Sub-patch inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    X: np.ndarray,
    encoder: ViT1DEncoder,
    pose_decoder: ViT1DPoseDecoder,
    cfg: dict,
    device: torch.device,
) -> np.ndarray:
    """Return per-frame 24-joint 6D predictions [N, 24, 6] (T-pose-relative root)."""
    vit_cfg     = cfg_get(cfg, ["vit1d"], {}) or {}
    dec_cfg     = (vit_cfg.get("pose_decoder") or {})
    T, _        = compute_T_and_hop(cfg)
    patch_size  = int(vit_cfg.get("patch_size", 16))
    sub_patches = int(dec_cfg.get("sub_patches", 1))
    Q           = patch_size // sub_patches
    J           = pose_decoder.num_joints
    stats       = PreprocessStats()
    N           = len(X)
    Pt          = T // patch_size

    pad   = (T - N % T) % T
    X_pad = np.concatenate([X, np.zeros((pad, 6), dtype=np.float32)]) if pad else X

    pred_full = np.zeros((len(X_pad), J, 6), dtype=np.float32)
    for w in range(len(X_pad) // T):
        start = w * T
        Xproc = preprocess_window(X_pad[start : start + T], cfg, stats)
        if Xproc is None:
            continue
        x = torch.from_numpy(Xproc).float().unsqueeze(0).to(device)
        tokens, _ = encoder.forward_tokens(x)
        out = pose_decoder(tokens).squeeze(0).cpu().numpy()    # [Pt, K*J*6]
        out = out.reshape(Pt, sub_patches, J, 6)               # [Pt, K, J, 6]
        for p in range(Pt):
            for k in range(sub_patches):
                f0 = start + p * patch_size + k * Q
                f1 = f0 + Q
                pred_full[f0:f1] = out[p, k]

    return pred_full[:N]   # [N, J, 6]  — root is T-pose-relative


# ---------------------------------------------------------------------------
# Root orientation recovery (undo T-pose reset per window)
# ---------------------------------------------------------------------------

def recover_absolute_root(
    pred_6d: np.ndarray,       # [N, 24, 6]  T-pose-relative predictions
    R_local_gt: np.ndarray,    # [N, 24, 3, 3]  GT local rotations
    T_win: int,
    Q: int,                    # sub-patch size; R0 taken at frame Q//2
) -> np.ndarray:
    """Return pred rotation matrices [N, 24, 3, 3] with absolute root orientation.

    For each window the model predicted root relative to R0 (the GT root at the
    first sub-patch centre).  We invert: R_abs = R0 @ R_pred_rel.
    Joints 1-23 are parent-relative and need no correction.
    """
    pred_R = rot6d_to_matrix(torch.from_numpy(pred_6d)).numpy()   # [N, 24, 3, 3]
    N = len(pred_R)
    for w in range(0, N, T_win):
        end = min(w + T_win, N)
        # R0 = GT root at first sub-patch centre (frame Q//2 within window)
        r0_frame = min(w + Q // 2, N - 1)
        R0 = R_local_gt[r0_frame, 0].astype(np.float64)           # [3, 3]
        # pred_root_rel[f] → R0 @ pred_root_rel[f]
        pred_R[w:end, 0] = np.einsum(
            "ij,njk->nik", R0, pred_R[w:end, 0].astype(np.float64)
        ).astype(np.float32)
    return pred_R   # [N, 24, 3, 3]  — joint 0 now has absolute orientation


# ---------------------------------------------------------------------------
# SMPL FK
# ---------------------------------------------------------------------------

def smpl_fk(R_local: np.ndarray, device: torch.device,
            chunk: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[N, 24, 3, 3] → (verts [N,V,3], joints [N,24,3], faces [F,3])."""
    import core.articulate as art
    from core.paths import Paths

    body  = art.model.ParametricModel(str(Paths.SMPL_FILE), device=torch.device("cpu"))
    N     = R_local.shape[0]
    faces = body.face.astype(np.uint32)

    all_verts, all_joints = [], []
    for s in range(0, N, chunk):
        R_c = torch.from_numpy(R_local[s : s + chunk].astype(np.float32))
        with torch.no_grad():
            _, j, v = body.forward_kinematics_masked(R_c, tran=None, calc_mesh=True)
        all_verts.append(v.numpy().astype(np.float32))
        all_joints.append(j.numpy().astype(np.float64))

    return (
        np.concatenate(all_verts,  axis=0),
        np.concatenate(all_joints, axis=0),
        faces,
    )


# ---------------------------------------------------------------------------
# Per-joint comparison plots
# ---------------------------------------------------------------------------

SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1",
    "L_Knee", "R_Knee", "Spine2", "L_Ankle",
    "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]


def plot_all_joints(
    R_gt: np.ndarray,    # [N, 24, 3, 3]  GT local rotations
    R_pred: np.ndarray,  # [N, 24, 3, 3]  pred rotations (joint 0 absolute, rest parent-relative)
    fps: float,
    out_dir: str,
) -> None:
    """Save one PNG per SMPL joint comparing GT vs predicted Euler angles (ZYX, degrees).

    Joint 0 (Pelvis) uses absolute orientation (T-pose reset undone).
    Joints 1-23 are parent-relative, matching the training target.
    Files are named joint_00_pelvis.png … joint_23_r_hand.png.
    """
    from scipy.spatial.transform import Rotation as ScipyRot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    N           = R_gt.shape[0]
    num_joints  = R_gt.shape[1]
    t           = np.arange(N) / fps
    angle_labels = ["Yaw (Z, °)", "Pitch (Y, °)", "Roll (X, °)"]

    for j in range(num_joints):
        jname = SMPL_JOINT_NAMES[j] if j < len(SMPL_JOINT_NAMES) else f"Joint{j:02d}"
        euler_gt   = ScipyRot.from_matrix(R_gt[:, j].astype(np.float64)).as_euler("ZYX", degrees=True)
        euler_pred = ScipyRot.from_matrix(R_pred[:, j].astype(np.float64)).as_euler("ZYX", degrees=True)

        fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
        for i, (ax, lbl) in enumerate(zip(axes, angle_labels)):
            ax.plot(t, euler_gt[:, i],   color="#6699FF", linewidth=0.8, label="GT")
            ax.plot(t, euler_pred[:, i], color="#FF8C00", linewidth=0.8, label="Pred", alpha=0.85)
            ax.set_ylabel(lbl)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, linewidth=0.4, alpha=0.5)
        axes[-1].set_xlabel("Time (s)")
        suffix = "absolute" if j == 0 else "parent-relative"
        fig.suptitle(f"Joint {j:02d} — {jname} ({suffix})  GT vs Predicted", fontsize=12)
        fig.tight_layout()

        fname = os.path.join(out_dir, f"joint_{j:02d}_{jname.lower().replace(' ', '_')}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("All-joint plots saved → %s/  (%d files)", out_dir, num_joints)


# ---------------------------------------------------------------------------
# MeshCat animation
# ---------------------------------------------------------------------------

def animate(
    verts_gt: np.ndarray,
    verts_pred: np.ndarray,
    joints_gt: np.ndarray,
    faces: np.ndarray,
    fps: float,
    gap: float,
    T_win: int,
):
    """Side-by-side animation: GT (blue) left, predicted (orange) right.

    A thin grey vertical bar is drawn at the start of each window boundary
    to show where the T-pose reset happens.
    """
    import meshcat
    import meshcat.geometry as mcg
    import meshcat.transformations as mct

    N   = verts_gt.shape[0]
    vis = meshcat.Visualizer()
    print("\nMeshCat at:", vis.url())
    print("Open the URL above in a browser, then wait for animation.\n")
    print("  BLUE   (left)  = Ground-truth")
    print("  ORANGE (right) = Predicted (root aligned per-window)\n")

    vis["/Background"].set_property("top_color",    [1, 1, 1])
    vis["/Background"].set_property("bottom_color", [1, 1, 1])

    ROOT = "scene"
    try:
        vis[ROOT].delete()
    except Exception:
        pass

    # SMPL is Y-up; MeshCat viewer is Z-up
    vis[ROOT].set_transform(mct.rotation_matrix(np.pi / 2.0, [1, 0, 0]))

    gt_geom = mcg.TriangularMeshGeometry(verts_gt[0],   faces)
    gt_mat  = mcg.MeshLambertMaterial(color=0x6699FF, opacity=0.88)
    vis[f"{ROOT}/gt_body"].set_object(gt_geom, gt_mat)

    pred_geom = mcg.TriangularMeshGeometry(verts_pred[0], faces)
    pred_mat  = mcg.MeshLambertMaterial(color=0xFF8C00, opacity=0.88)
    vis[f"{ROOT}/pred_body"].set_object(pred_geom, pred_mat)

    root_offset  = joints_gt[0, 0].copy()
    pred_x_shift = np.array([gap, 0.0, 0.0], dtype=np.float64)
    dt = 1.0 / fps

    logger.info("Animating %d frames at %.0f fps (window=%d frames) …", N, fps, T_win)
    for f in range(N):
        shift = joints_gt[f, 0] - root_offset

        gt_geom.vertices = verts_gt[f] + shift
        vis[f"{ROOT}/gt_body"].set_object(gt_geom, gt_mat)

        pred_geom.vertices = verts_pred[f] + shift + pred_x_shift
        vis[f"{ROOT}/pred_body"].set_object(pred_geom, pred_mat)

        sleep(dt)

    logger.info("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    ap = argparse.ArgumentParser(
        description="Visualise 24-joint T-pose-model predictions vs GT in MeshCat"
    )
    ap.add_argument("--pkl",          required=True)
    ap.add_argument("--config",       default="configs/base.yaml")
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--run",          required=True)
    ap.add_argument("--ckpt",         default="best")
    ap.add_argument("--fps",          type=float, default=50.0)
    ap.add_argument("--max-frames",   type=int,   default=0)
    ap.add_argument("--gap",          type=float, default=1.5,
                    help="X offset (m) between GT and predicted bodies")
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir",      default="data_vis",
                    help="Directory to save root orientation PNG")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info("Device: %s", device)

    base_cfg  = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    cfg       = deep_update(base_cfg, model_cfg)

    runs_root = cfg.get("paths", {}).get("runs_root", "runs")
    ckpt_path = (
        os.path.join(runs_root, args.run, "checkpoints", f"{args.ckpt}.pt")
        if args.ckpt in ("best", "latest") else args.ckpt
    )
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    state    = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = state["cfg"]

    vit_cfg     = cfg_get(ckpt_cfg, ["vit1d"], {}) or {}
    dec_cfg     = (vit_cfg.get("pose_decoder") or {})
    pose_cfg    = cfg_get(ckpt_cfg, ["objective", "pose"], {}) or {}

    patch_size  = int(vit_cfg.get("patch_size", 16))
    sub_patches = int(dec_cfg.get("sub_patches", 1))
    Q           = patch_size // sub_patches
    num_joints  = int(pose_cfg.get("num_joints", 24))
    add_gravity = bool(pose_cfg.get("add_gravity", True))

    logger.info("patch_size=%d  sub_patches=%d  Q=%d  num_joints=%d",
                patch_size, sub_patches, Q, num_joints)

    encoder = ViT1DEncoder(ckpt_cfg).to(device)
    encoder.load_state_dict(state["model"])
    encoder.eval()

    T_win, _ = compute_T_and_hop(ckpt_cfg)

    pose_decoder = ViT1DPoseDecoder(
        embed_dim               = encoder.embed_dim,
        num_patches_per_channel = T_win // patch_size,
        in_channels             = encoder.in_channels,
        num_joints              = num_joints,
        cfg                     = ckpt_cfg,
        sub_patches             = sub_patches,
    ).to(device)
    pose_decoder.load_state_dict(state["pose_decoder"])
    pose_decoder.eval()

    # ---- Load sequence ---- #
    logger.info("Loading: %s", os.path.basename(args.pkl))
    X_flu, R_local_gt = load_sequence(
        args.pkl, args.fps, add_gravity, args.max_frames
    )

    # ---- Inference (root in T-pose-relative space) ---- #
    logger.info("Running inference …")
    pred_6d = run_inference(X_flu, encoder, pose_decoder, ckpt_cfg, device)
    # pred_6d: [N, 24, 6]  — joint 0 is T-pose-relative (R0⁻¹ @ R_gt_root)

    # ---- Recover absolute root orientation ---- #
    logger.info("Recovering absolute root orientation per window …")
    pred_R_abs = recover_absolute_root(pred_6d, R_local_gt, T_win, Q)
    # pred_R_abs: [N, 24, 3, 3]  — joint 0 now in world frame

    # ---- Per-joint comparison plots ---- #
    seq_name  = os.path.basename(args.pkl).replace(".pkl", "")
    plots_dir = os.path.join(args.out_dir, f"{args.run}_{seq_name}")
    plot_all_joints(R_local_gt, pred_R_abs, args.fps, plots_dir)

    # ---- FK ---- #
    logger.info("SMPL FK — GT …")
    verts_gt, joints_gt, faces = smpl_fk(R_local_gt, device)

    logger.info("SMPL FK — prediction …")
    verts_pred, _, _ = smpl_fk(pred_R_abs, device)

    # ---- Animate ---- #
    logger.info("Launching MeshCat …")
    animate(verts_gt, verts_pred, joints_gt, faces, args.fps, args.gap, T_win)


if __name__ == "__main__":
    main()
