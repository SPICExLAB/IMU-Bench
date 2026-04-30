"""vis_totalcapture_full_body_pose.py

Side-by-side MeshCat animation of GT vs predicted full-body SMPL pose on a
TotalCapture sequence, driven by a ViT1D full-body pose model trained with the
FLU wrist frame (e.g. vit1d_full_body_pose_supervised_totalcapture_flu).

    LEFT  (blue)   — Ground-truth full-body SMPL pose
    RIGHT (orange) — Predicted full-body SMPL pose

Usage
-----
    python scripts/vis_totalcapture_full_body_pose.py \\
        --pkl          /mnt/nas2/DIP_recalculate/s1_acting1.pkl \\
        --config       configs/base.yaml \\
        --model-config configs/vit1d_full_body_pose_supervised_totalcapture_flu.yaml \\
        --run          vit1d_full_body_pose_supervised_totalcapture_flu \\
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
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DPoseDecoder
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TotalCapture / SMPL constants
# ---------------------------------------------------------------------------
_SENSOR_REORDER  = [2, 3, 0, 1, 4, 5]
_RIGHT_WRIST_IDX = 0
_G_WORLD         = np.array([0.0, -9.81, 0.0], dtype=np.float64)

R_WRIST_FROM_FLU = np.array(
    [[-1.0, 0.0, 0.0],
     [ 0.0, 0.0, 1.0],
     [ 0.0, 1.0, 0.0]],
    dtype=np.float64,
)


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

def _resample_linear(x: np.ndarray, new_len: int) -> np.ndarray:
    old_len = x.shape[0]
    if old_len == new_len:
        return x
    t_old = np.linspace(0.0, 1.0, old_len)
    t_new = np.linspace(0.0, 1.0, new_len)
    flat  = x.reshape(old_len, -1)
    out   = np.stack(
        [np.interp(t_new, t_old, flat[:, c]) for c in range(flat.shape[1])],
        axis=1,
    )
    return out.reshape(new_len, *x.shape[1:])


def _resample_rotmat_nn(x: np.ndarray, new_len: int) -> np.ndarray:
    idx = np.round(np.linspace(0, x.shape[0] - 1, new_len)).astype(np.int64)
    return x[idx]


def load_totalcapture_flu(
    pkl_path: str,
    target_hz: float = 50.0,
    add_gravity: bool = True,
    max_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pkl → (X_flu [N,6], gt_aa [N,24,3])."""
    from scipy.spatial.transform import Rotation as ScipyRot

    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh, encoding="latin1")

    acc_raw = np.asarray(data["acc"], dtype=np.float64)
    ori_raw = np.asarray(data["ori"], dtype=np.float64)
    gt_raw  = np.asarray(data["gt"],  dtype=np.float64)

    N = min(acc_raw.shape[0], ori_raw.shape[0], gt_raw.shape[0])
    if max_frames > 0:
        N = min(N, max_frames)
    acc_raw = acc_raw[:N];  ori_raw = ori_raw[:N];  gt_raw = gt_raw[:N]

    fps_raw = 60.0
    if abs(fps_raw - target_hz) > 1e-3:
        new_len = int(round((N - 1) * target_hz / fps_raw)) + 1
        acc_raw = _resample_linear(acc_raw, new_len)
        ori_raw = _resample_rotmat_nn(ori_raw, new_len)
        gt_raw  = _resample_linear(gt_raw, new_len)
        N = new_len

    acc_all = acc_raw[:, _SENSOR_REORDER]
    ori_all = ori_raw[:, _SENSOR_REORDER]

    acc_world      = acc_all[:, _RIGHT_WRIST_IDX].copy()
    R_world_from_s = ori_all[:, _RIGHT_WRIST_IDX]

    if add_gravity:
        acc_world = acc_world - _G_WORLD

    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_s, R_WRIST_FROM_FLU)
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)

    acc_flu = np.einsum("nij,nj->ni", R_X_from_world, acc_world)

    dR = np.einsum("nij,njk->nik",
                   R_world_from_s[:-1].transpose(0, 2, 1),
                   R_world_from_s[1:])
    omega_local = (ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
                   .as_rotvec().astype(np.float64) * target_hz)
    omega_local_full      = np.zeros((N, 3), dtype=np.float64)
    omega_local_full[:-1] = omega_local
    omega_local_full[-1]  = omega_local[-1]
    omega_world = np.einsum("nij,nj->ni", R_world_from_s, omega_local_full)
    gyro_flu    = np.einsum("nij,nj->ni", R_X_from_world, omega_world)

    X_flu = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)
    gt_aa = gt_raw.reshape(N, 24, 3)
    logger.info("Loaded %d frames (%.1f s @ %.0f Hz)", N, N / target_hz, target_hz)
    return X_flu, gt_aa


# ---------------------------------------------------------------------------
# SMPL FK (chunked, CPU)
# ---------------------------------------------------------------------------

def smpl_fk(R_local_np: np.ndarray, device: torch.device, chunk: int = 512):
    """[N, 24, 3, 3] → (verts [N,V,3], joints [N,24,3], faces [F,3])."""
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
    verts_gt: np.ndarray,    # [N, V, 3]
    verts_pred: np.ndarray,  # [N, V, 3]
    joints_gt: np.ndarray,   # [N, 24, 3]
    faces: np.ndarray,
    fps: float,
    gap: float,
):
    """Side-by-side animation: GT (blue) left, predicted (orange) right."""
    import meshcat
    import meshcat.geometry as mcg
    import meshcat.transformations as mct

    N   = verts_gt.shape[0]
    vis = meshcat.Visualizer()
    print("\nMeshCat at:", vis.url())
    print("Open the URL above in a browser, then wait for animation.\n")
    print("  BLUE   (left)  = Ground-truth full-body pose")
    print("  ORANGE (right) = Predicted full-body pose\n")

    vis["/Background"].set_property("top_color",    [1, 1, 1])
    vis["/Background"].set_property("bottom_color", [1, 1, 1])
    vis["/Grid"].set_transform(mct.translation_matrix([0, 0, -1]))

    ROOT = "scene"
    try:
        vis[ROOT].delete()
    except Exception:
        pass

    # TotalCapture / SMPL is Y-up; rotate scene for MeshCat (Z-up viewer)
    vis[ROOT].set_transform(mct.rotation_matrix(np.pi / 2.0, [1, 0, 0]))

    gt_geom = mcg.TriangularMeshGeometry(verts_gt[0], faces)
    gt_mat  = mcg.MeshLambertMaterial(color=0x6699FF, opacity=0.88)
    vis[f"{ROOT}/gt_body"].set_object(gt_geom, gt_mat)

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
        description="Visualize GT vs predicted full-body pose on TotalCapture in MeshCat"
    )
    ap.add_argument("--pkl",          required=True,
                    help="DIP-format TotalCapture pkl (e.g. /mnt/nas2/DIP_recalculate/s1_acting1.pkl)")
    ap.add_argument("--config",       default="configs/base.yaml")
    ap.add_argument("--model-config", required=True,
                    help="Model config yaml (e.g. configs/vit1d_full_body_pose_supervised_totalcapture_flu.yaml)")
    ap.add_argument("--run",          required=True,
                    help="Run name under runs/<run>/checkpoints/")
    ap.add_argument("--ckpt",         default="best",
                    help='"best" | "latest" | absolute path to .pt file')
    ap.add_argument("--fps",          type=float, default=50.0)
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

    num_joints = _resolve_num_joints(ckpt_cfg) or _resolve_num_joints(cfg) or 24
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

    # ------------------------------------------------------------------ data
    pose_cfg    = cfg_get(ckpt_cfg, ["objective", "pose"], {}) or {}
    target_hz   = float(pose_cfg.get("target_hz",  args.fps))
    add_gravity = bool(pose_cfg.get("add_gravity", True))

    logger.info("Loading: %s  (target_hz=%.0f  add_gravity=%s)",
                os.path.basename(args.pkl), target_hz, add_gravity)
    X_flu, gt_aa = load_totalcapture_flu(
        pkl_path    = args.pkl,
        target_hz   = target_hz,
        add_gravity = add_gravity,
        max_frames  = args.max_frames,
    )
    N = len(X_flu)

    # ------------------------------------------------------------------ inference
    logger.info("Running inference …")
    pred_6d = run_inference(X_flu, encoder, pose_decoder, ckpt_cfg, device, num_joints)
    # pred_6d: [N, 24, 6]

    # ------------------------------------------------------------------ SMPL FK — GT
    from scipy.spatial.transform import Rotation as ScipyRot

    logger.info("SMPL FK — GT …")
    R_gt_local = (
        ScipyRot.from_rotvec(gt_aa.reshape(-1, 3))
        .as_matrix()
        .reshape(N, 24, 3, 3)
        .astype(np.float32)
    )
    verts_gt, joints_gt, faces = smpl_fk(R_gt_local, device)

    # ------------------------------------------------------------------ SMPL FK — pred
    logger.info("SMPL FK — prediction …")
    pred_R = rotation_6d_to_matrix(
        torch.from_numpy(pred_6d)
    ).numpy().astype(np.float32)   # [N, 24, 3, 3]
    verts_pred, _, _ = smpl_fk(pred_R, device)

    # ------------------------------------------------------------------ animate
    logger.info("Launching MeshCat …")
    animate(verts_gt, verts_pred, joints_gt, faces, target_hz, args.gap)


if __name__ == "__main__":
    main()
