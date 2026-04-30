"""vis_imuposer_arm_pose.py

Side-by-side MeshCat animation of GT vs predicted right-arm pose on an
IMUPoser sequence, driven by a ViT1D arm-pose model trained with the
FLU wrist frame (e.g. vit1d_pose_supervised_imuposer_flu).

    LEFT  (blue)   — Ground-truth:  T-pose body + GT right-arm rotations
    RIGHT (orange) — Predicted:     T-pose body + model right-arm rotations

Right-arm chain (SMPL joint indices)
-------------------------------------
  14 — R-Collar
  17 — R-Shoulder
  19 — R-Elbow
  21 — R-Wrist
  23 — R-Hand

Usage
-----
    python scripts/vis_imuposer_arm_pose.py \\
        --pkl          /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/s01_01.pkl \\
        --config       configs/base.yaml \\
        --model-config configs/vit1d_pose_supervised_imuposer_flu.yaml \\
        --run          vit1d_pose_supervised_imuposer_flu \\
        [--ckpt        best]     \\
        [--fps         50]       \\
        [--max-frames  0]        \\
        [--gap         1.5]      \\
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
from imu_lm.data.imuposerdataset import (
    _resample_linear,
    _resample_nearest,
    _resample_rotmat_nlerp,
    _resample_rotmat_nearest,
)
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DPoseDecoder
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RIGHT_WRIST_CHAIN_JOINTS = [14, 17, 19, 21, 23]  # R-Collar→R-Hand
ARM_JOINT_NAMES = ["R-Collar", "R-Shoulder", "R-Elbow", "R-Wrist", "R-Hand"]

# FLU body frame: F=-x_smpl, L=+z_smpl, U=+y_smpl  (right-wrist T-pose)
R_WRIST_FROM_FLU = np.array(
    [[-1.0,  0.0,  0.0],
     [ 0.0,  0.0,  1.0],
     [ 0.0,  1.0,  0.0]],
    dtype=np.float64,
)

_SMPL_R_WRIST = 21
_R_WRIST_REAL_IDX = 1  # index into real_imu_data arrays for right wrist


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_imuposer_flu_vis(
    pkl_path: str,
    target_hz: float = 50.0,
    add_gravity: bool = False,
    max_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one IMUPoser pkl and return **(X_flu [N,6], pose_local [N,24,3,3])**.

    X_flu       — right-wrist accelerometer + gyroscope in the FLU body frame.
    pose_local  — SMPL local rotation matrices [N, 24, 3, 3] resampled to
                  ``target_hz``, suitable for SMPL FK.

    Parameters
    ----------
    pkl_path    : absolute path to an IMUPoser ``.pkl`` file.
    target_hz   : target sampling rate (default 50 Hz, matching training).
    add_gravity : if True, add back gravity (Z-up: +9.81 on Z) to the
                  gravity-subtracted real_acc before projecting into FLU.
    max_frames  : if > 0, truncate to this many frames before resampling.
    """
    from scipy.spatial.transform import Rotation as ScipyRot
    import core.articulate as art
    from core.paths import Paths

    with open(pkl_path, "rb") as fh:
        seq = pickle.load(fh)

    gt   = seq.get("gt_data", {})
    ri   = seq.get("real_imu_data", {})

    pose_local  = np.asarray(gt["pose"],     dtype=np.float32)   # [N, 24, 3, 3]
    joint_pos_w = np.asarray(gt["joint_pos"],dtype=np.float32)   # [N, 24, 3]
    real_acc    = np.asarray(ri["real_acc"], dtype=np.float32)   # [N, 5, 3]
    real_ori    = np.asarray(ri["real_ori"], dtype=np.float32)   # [N, 5, 3, 3]

    fps_raw = float(seq.get("sampling_rate_hz", target_hz))
    N = pose_local.shape[0]
    if max_frames > 0:
        N = min(N, max_frames)
    pose_local  = pose_local[:N]
    joint_pos_w = joint_pos_w[:N]
    real_acc    = real_acc[:N]
    real_ori    = real_ori[:N]

    # ---- Resample to target_hz ----
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
        logger.info("Resampled %d → %d frames (%.0f→%.0f Hz)", pose_local.shape[0], N, fps_raw, target_hz)

    fps = float(target_hz)

    # ---- SMPL FK → world-frame wrist rotation R_world_from_wrist(t) ----
    body_model = art.model.ParametricModel(
        str(Paths.SMPL_FILE), device=torch.device("cpu")
    )
    root_trans = torch.from_numpy(joint_pos_w[:, 0])  # [N, 3]
    pose_local_t = torch.from_numpy(pose_local.astype(np.float32))
    pose_global_t, _ = body_model.forward_kinematics_masked(
        pose_local_t, tran=root_trans, calc_mesh=False
    )  # [N, 24, 3, 3]
    R_world_from_wrist = pose_global_t[:, _SMPL_R_WRIST].numpy().astype(np.float64)

    # ---- FLU frame ----
    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_wrist, R_WRIST_FROM_FLU)
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)

    # ---- Real accelerometer (Z-up world frame) → FLU ----
    acc_world = real_acc[:, _R_WRIST_REAL_IDX].astype(np.float64)
    if add_gravity:
        # real_acc is gravity-subtracted; add gravity back (Z-up: g=[0,0,-9.81])
        g_world = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        acc_world = acc_world - g_world  # equivalent to +9.81 on Z
    acc_flu = np.einsum("nij,nj->ni", R_X_from_world, acc_world)

    # ---- Real gyro: sensor-local omega → world → FLU ----
    R_world_from_sensor = real_ori[:, _R_WRIST_REAL_IDX].astype(np.float64)
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

    X_flu = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)
    return X_flu, pose_local  # [N,6], [N,24,3,3]


# ---------------------------------------------------------------------------
# SMPL FK
# ---------------------------------------------------------------------------

def smpl_fk(R_local_np: np.ndarray, device: torch.device, chunk: int = 512):
    """[N, 24, 3, 3] → (verts [N,V,3], joints [N,24,3], faces [F,3]).

    Processes in chunks to avoid GPU OOM on long sequences.
    Always runs on CPU (FK is memory-bound, not compute-bound).
    """
    import core.articulate as art
    from core.paths import Paths

    fk_device = torch.device("cpu")
    body  = art.model.ParametricModel(str(Paths.SMPL_FILE), device=fk_device)
    N     = R_local_np.shape[0]
    faces = body.face.astype(np.uint32)

    all_verts  = []
    all_joints = []
    for start in range(0, N, chunk):
        R_chunk = torch.from_numpy(R_local_np[start : start + chunk].astype(np.float32))
        with torch.no_grad():
            _, j, v = body.forward_kinematics_masked(R_chunk, tran=None, calc_mesh=True)
        all_verts.append(v.numpy().astype(np.float32))
        all_joints.append(j.numpy().astype(np.float64))

    return (
        np.concatenate(all_verts,  axis=0),
        np.concatenate(all_joints, axis=0),
        faces,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    X: np.ndarray,
    encoder: ViT1DEncoder,
    pose_decoder: ViT1DPoseDecoder,
    cfg: dict,
    device: torch.device,
    num_joints: int,
) -> np.ndarray:
    """Return per-frame 6D predictions [N, num_joints, 6]."""
    T, _       = compute_T_and_hop(cfg)
    patch_size = int((cfg_get(cfg, ["vit1d"], {}) or {}).get("patch_size", 16))
    stats      = PreprocessStats()
    N          = len(X)

    pad   = (T - N % T) % T
    X_pad = np.concatenate([X, np.zeros((pad, 6), dtype=np.float32)]) if pad else X

    pred_full = np.zeros((len(X_pad), num_joints, 6), dtype=np.float32)
    for w in range(len(X_pad) // T):
        start  = w * T
        Xw     = X_pad[start : start + T]
        Xproc  = preprocess_window(Xw, cfg, stats)
        if Xproc is None:
            continue
        x      = torch.from_numpy(Xproc).float().unsqueeze(0).to(device)
        tokens, _ = encoder.forward_tokens(x)
        pred_np = pose_decoder(tokens).squeeze(0).cpu().numpy()
        for p in range(pred_np.shape[0]):
            f0 = start + p * patch_size
            f1 = f0 + patch_size
            pred_full[f0:f1] = pred_np[p].reshape(num_joints, 6)

    return pred_full[:N]


# ---------------------------------------------------------------------------
# MeshCat animation
# ---------------------------------------------------------------------------

def animate(
    vis,                         # meshcat.Visualizer — already open
    verts_gt: np.ndarray,    # [N, V, 3]
    verts_pred: np.ndarray,  # [N, V, 3]
    joints_gt: np.ndarray,   # [N, 24, 3]  — root anchoring only
    faces: np.ndarray,
    fps: float,
    gap: float,
):
    """Side-by-side animation: GT (blue) left, predicted (orange) right.

    Both bodies are in T-pose for all non-arm joints; only the 5 right-arm
    chain joints animate.
    """
    import meshcat.geometry as mcg
    import meshcat.transformations as mct

    N   = verts_gt.shape[0]

    vis["/Background"].set_property("top_color",    [1, 1, 1])
    vis["/Background"].set_property("bottom_color", [1, 1, 1])
    vis["/Grid"].set_transform(mct.translation_matrix([0, 0, -1]))

    ROOT = "scene"
    try:
        vis[ROOT].delete()
    except Exception:
        pass

    # SMPL is Y-up; MeshCat viewer is Z-up → rotate 90° around X
    vis[ROOT].set_transform(mct.rotation_matrix(np.pi / 2.0, [1, 0, 0]))

    # ---- GT body (blue) ----
    gt_geom = mcg.TriangularMeshGeometry(verts_gt[0], faces)
    gt_mat  = mcg.MeshLambertMaterial(color=0x6699FF, opacity=0.88)
    vis[f"{ROOT}/gt_body"].set_object(gt_geom, gt_mat)

    # ---- Predicted body (orange) ----
    pred_geom = mcg.TriangularMeshGeometry(verts_pred[0], faces)
    pred_mat  = mcg.MeshLambertMaterial(color=0xFF8C00, opacity=0.88)
    vis[f"{ROOT}/pred_body"].set_object(pred_geom, pred_mat)

    root_offset  = joints_gt[0, 0].copy()
    pred_x_shift = np.array([gap, 0.0, 0.0], dtype=np.float64)

    dt = 1.0 / fps
    logger.info("Animating %d frames at %.0f fps …", N, fps)

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
        description="Visualize GT vs predicted right-arm pose on IMUPoser in MeshCat"
    )
    ap.add_argument("--pkl",          required=True,
                    help="IMUPoser pkl (e.g. /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/s01_01.pkl)")
    ap.add_argument("--config",       default="configs/base.yaml")
    ap.add_argument("--model-config", required=True,
                    help="Model config yaml (e.g. configs/vit1d_pose_supervised_imuposer_flu.yaml)")
    ap.add_argument("--run",          required=True,
                    help="Run name under runs/<run>/checkpoints/")
    ap.add_argument("--ckpt",         default="best",
                    help='"best" | "latest" | absolute path to .pt file')
    ap.add_argument("--fps",          type=float, default=None,
                    help="Target data sampling rate Hz (default: read from checkpoint config)")
    ap.add_argument("--play-fps",     type=float, default=None,
                    help="Playback animation speed in fps (default: same as data rate)")
    ap.add_argument("--max-frames",   type=int,   default=0,
                    help="Cap frames (0 = full sequence)")
    ap.add_argument("--gap",          type=float, default=1.5,
                    help="X offset (m) between GT and pred bodies")
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------ cfg
    base_cfg  = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    cfg       = deep_update(base_cfg, model_cfg)

    # ------------------------------------------------------------------ checkpoint
    runs_root = cfg.get("paths", {}).get("runs_root", "runs")
    run_dir   = os.path.join(runs_root, args.run)
    if args.ckpt in ("best", "latest"):
        ckpt_path = os.path.join(run_dir, "checkpoints", f"{args.ckpt}.pt")
    else:
        ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    state    = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = state["cfg"]

    def _resolve_num_joints(c):
        for path in (
            ["objective", "pose",          "num_joints"],
            ["objective", "pose_probe",    "num_joints"],
            ["objective", "pose_finetune", "num_joints"],
        ):
            v = cfg_get(c, path, None)
            if v is not None:
                return int(v)
        return None

    num_joints = _resolve_num_joints(ckpt_cfg) or _resolve_num_joints(cfg) or 5
    logger.info("num_joints: %d", num_joints)

    encoder = ViT1DEncoder(ckpt_cfg).to(device)
    encoder.load_state_dict(state["model"])
    encoder.eval()

    T_win, _   = compute_T_and_hop(ckpt_cfg)
    patch_size = int((cfg_get(ckpt_cfg, ["vit1d"], {}) or {}).get("patch_size", 16))

    pose_decoder = ViT1DPoseDecoder(
        embed_dim=encoder.embed_dim,
        num_patches_per_channel=T_win // patch_size,
        in_channels=encoder.in_channels,
        num_joints=num_joints,
        cfg=ckpt_cfg,
    ).to(device)
    pose_decoder.load_state_dict(state["pose_decoder"])
    pose_decoder.eval()

    ckpt_cfg.setdefault("objective", {}).setdefault("pose", {})["num_joints"] = num_joints

    # ------------------------------------------------------------------ MeshCat (start early so URL is ready while computing)
    import meshcat
    vis = meshcat.Visualizer()
    print("\nMeshCat at:", vis.url())
    print("Open the URL above in a browser. Meshes will appear after computation completes.\n")
    print("  BLUE   (left)  = Ground-truth right-arm")
    print("  ORANGE (right) = Predicted right-arm\n")

    # ------------------------------------------------------------------ data
    def _resolve_cfg_val(c, key, default):
        for section in ("pose_probe", "pose_finetune", "pose"):
            v = cfg_get(c, ["objective", section, key], None)
            if v is not None:
                return v
        return default

    target_hz   = float(args.fps if args.fps is not None else _resolve_cfg_val(ckpt_cfg, "target_hz", 30.0))
    add_gravity = bool(_resolve_cfg_val(ckpt_cfg, "add_gravity", False))
    play_fps    = float(args.play_fps) if args.play_fps is not None else target_hz

    logger.info("Loading: %s  (target_hz=%.0f  add_gravity=%s)",
                os.path.basename(args.pkl), target_hz, add_gravity)
    X_flu, pose_local = load_imuposer_flu_vis(
        pkl_path    = args.pkl,
        target_hz   = target_hz,
        add_gravity = add_gravity,
        max_frames  = args.max_frames,
    )
    N = len(X_flu)
    logger.info("Loaded %d frames (%.1f s @ %.0f Hz)", N, N / target_hz, target_hz)

    # ------------------------------------------------------------------ inference
    logger.info("Running inference …")
    pred_6d = run_inference(X_flu, encoder, pose_decoder, ckpt_cfg, device, num_joints)
    # pred_6d: [N, 5, 6] — 5 right-arm joints in 6D rotation representation

    # ------------------------------------------------------------------ SMPL FK — GT
    # GT body: T-pose for all joints except the 5 right-arm joints (GT rotations).
    logger.info("SMPL FK — GT …")
    identity      = np.eye(3, dtype=np.float32)
    R_gt_arm_only = np.tile(identity, (N, 24, 1, 1))
    for ji in RIGHT_WRIST_CHAIN_JOINTS:
        R_gt_arm_only[:, ji] = pose_local[:, ji]
    verts_gt, joints_gt, faces = smpl_fk(R_gt_arm_only, device)

    # ------------------------------------------------------------------ SMPL FK — pred
    # Predicted body: T-pose for all joints except the 5 right-arm joints (model output).
    logger.info("SMPL FK — prediction …")
    pred_R_arm = rotation_6d_to_matrix(
        torch.from_numpy(pred_6d)
    ).numpy().astype(np.float32)   # [N, 5, 3, 3]

    R_pred_local = np.tile(identity, (N, 24, 1, 1))
    for ci, ji in enumerate(RIGHT_WRIST_CHAIN_JOINTS):
        R_pred_local[:, ji] = pred_R_arm[:, ci]
    verts_pred, _, _ = smpl_fk(R_pred_local, device)

    # ------------------------------------------------------------------ animate
    logger.info("Launching MeshCat …")
    animate(vis, verts_gt, verts_pred, joints_gt, faces, play_fps, args.gap)


if __name__ == "__main__":
    main()
