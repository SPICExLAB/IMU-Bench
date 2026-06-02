"""Evaluation for the 24-joint sub-patch full-body pose model.

Metric groups reported
----------------------
  24-joint MPJPE / MPVPE  — T-pose-relative space: at the start of every
      2.56 s window the GT root (pelvis) orientation is reset to identity
      (R0^{-1} @ R_gt), matching the training normalisation.  Predictions are
      already in this space; comparing them against the normalised GT gives a
      fair, rotation-invariant measure of pose-change quality.

  23-joint MPJPE / MPVPE  — joints 1-23 only (parent-relative local rotations,
      unaffected by root normalisation).  No T-pose reset applied.

Usage
-----
    python scripts/run_pose_eval_24j_tpose.py \\
        --config       configs/base.yaml \\
        --model-config configs/vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_24j_tpose.yaml \\
        --run          vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_24j_tpose \\
        --test-dir     /mnt/nas2/processed_data_test_body_Ty_rwrist \\
        [--ckpt        best]
        [--out-dir     data_vis/pose_eval_24j_tpose]
        [--fps         50]
        [--max-frames  0]
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as ScipyRot
from tqdm import tqdm

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.nymeria_wrist_loader import _xsens_to_smpl_rotmat
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DPoseDecoder
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

logger = logging.getLogger(__name__)

SMPL_JOINT_NAMES = [
    "Pelvis", "L-Hip", "R-Hip", "Spine1", "L-Knee", "R-Knee",
    "Spine2", "L-Ankle", "R-Ankle", "Spine3", "L-Foot", "R-Foot",
    "Neck", "L-Collar", "R-Collar", "Head", "L-Shoulder", "R-Shoulder",
    "L-Elbow", "R-Elbow", "L-Wrist", "R-Wrist", "L-Hand", "R-Hand",
]

R_WRIST_FROM_FLU = np.array(
    [[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]], dtype=np.float64
)


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """[..., 6] → [..., 3, 3]  (6D rotation → rotation matrix, row-convention)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)  # rows of R


def mat_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] → [..., 6]  (first two rows)."""
    return R[..., :2, :].flatten(start_dim=-2)


def geodesic_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    R_diff = R_pred @ R_gt.transpose(-1, -2)
    trace  = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    return torch.rad2deg(torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)))


# ---------------------------------------------------------------------------
# Data loading (Nymeria real-wrist FLU format)
# ---------------------------------------------------------------------------

def load_sequence(pkl_path: str, fps: int, max_frames: int, add_gravity: bool):
    """Load a Nymeria pkl and return FLU IMU + full SMPL local rotations.

    Returns
    -------
    X            : [N, 6]       FLU-frame wrist IMU (acc + gyro)
    smpl_R_local : [N, 24, 3, 3]  SMPL local rotation matrices
    """
    import core.articulate as art
    from core.paths import Paths

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    if not raw.get("has_wrist_imu", False):
        raise ValueError("has_wrist_imu=False")

    gt_data   = raw.get("gt_data", {})
    wrist_imu = raw.get("wrist_imu_data", {})
    wrist_gt  = raw.get("wrist_gt_data",  {})

    xsens_pose = gt_data.get("xsens_pose")
    if xsens_pose is None:
        raise ValueError("xsens_pose missing")
    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float32)

    body_fps_raw  = float(raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or fps)
    wrist_fps_raw = float(raw.get("wrist_imu_sampling_rate_hz", body_fps_raw))
    body_ds  = max(1, round(body_fps_raw  / fps))
    wrist_ds = max(1, round(wrist_fps_raw / fps))

    xsens_pose = xsens_pose[::body_ds]

    def _np64(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy().astype(np.float64)
        return np.asarray(x, dtype=np.float64)

    acc_local  = _np64(wrist_imu["accel"])[::wrist_ds]
    gyro_local = _np64(wrist_imu["gyro"])[::wrist_ds]

    orient_raw = wrist_gt["orientation"]
    try:
        import pypose as pp
        _is_lt = isinstance(orient_raw, pp.LieTensor)
    except ImportError:
        _is_lt = False
    if _is_lt:
        R_w2wrist = orient_raw.matrix().detach().cpu().numpy().astype(np.float64)[::wrist_ds]
    else:
        R_w2wrist = ScipyRot.from_quat(_np64(orient_raw)[::wrist_ds]).as_matrix().astype(np.float64)

    N = min(xsens_pose.shape[0], acc_local.shape[0])
    if max_frames > 0:
        N = min(N, max_frames)
    xsens_pose = xsens_pose[:N]
    acc_local  = acc_local[:N]
    gyro_local = gyro_local[:N]
    R_w2wrist  = R_w2wrist[:N]

    acc_world  = np.einsum("nij,nj->ni", R_w2wrist, acc_local)
    gyro_world = np.einsum("nij,nj->ni", R_w2wrist, gyro_local)
    if not add_gravity:
        acc_world = acc_world + np.array([0., -9.81, 0.])

    R_world_from_X = np.einsum("nij,jk->nik", R_w2wrist, R_WRIST_FROM_FLU)
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)
    acc_flu  = np.einsum("nij,nj->ni", R_X_from_world, acc_world)
    gyro_flu = np.einsum("nij,nj->ni", R_X_from_world, gyro_world)
    X = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)

    # SMPL local rotations via IK
    smpl_R_global = torch.from_numpy(
        _xsens_to_smpl_rotmat(xsens_pose[:, :, :3, :3])
    )   # [N, 24, 3, 3]
    body_model_ik = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))
    smpl_R_local = (
        body_model_ik.inverse_kinematics_R(smpl_R_global.reshape(N, -1))
        .view(N, 24, 3, 3).contiguous().clone().cpu().numpy()
    )   # [N, 24, 3, 3]

    return X, smpl_R_local


# ---------------------------------------------------------------------------
# Inference — sub-patch aware
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    X: np.ndarray,
    encoder: ViT1DEncoder,
    pose_decoder: ViT1DPoseDecoder,
    cfg,
    device: torch.device,
) -> np.ndarray:
    """Return per-frame 24-joint 6D predictions [N, 24, 6].

    Sub-patch assignment: each sub-patch prediction is replicated across the
    Q=patch_size//sub_patches frames it covers.
    """
    vit_cfg    = cfg_get(cfg, ["vit1d"], {}) or {}
    dec_cfg    = (vit_cfg.get("pose_decoder") or {})
    T          = int(cfg_get(cfg, ["windowing", "window_seconds"], 2.56) *
                     cfg_get(cfg, ["windowing", "sample_rate_hz"], 50))
    patch_size = int(vit_cfg.get("patch_size", 16))
    sub_patches = int(dec_cfg.get("sub_patches", 1))
    Q          = patch_size // sub_patches          # sub-patch size in frames
    J          = pose_decoder.num_joints            # 24
    stats      = PreprocessStats()
    N          = len(X)

    pad   = (T - N % T) % T
    X_pad = np.concatenate([X, np.zeros((pad, 6), dtype=np.float32)]) if pad else X

    pred_full = np.zeros((len(X_pad), J, 6), dtype=np.float32)
    Pt = T // patch_size

    for w in range(len(X_pad) // T):
        start = w * T
        Xw    = X_pad[start : start + T]
        Xproc = preprocess_window(Xw, cfg, stats)
        if Xproc is None:
            continue
        x      = torch.from_numpy(Xproc).float().unsqueeze(0).to(device)
        tokens, _ = encoder.forward_tokens(x)
        out    = pose_decoder(tokens).squeeze(0).cpu().numpy()   # [Pt, K*J*6]
        out    = out.reshape(Pt, sub_patches, J, 6)              # [Pt, K, J, 6]

        for p in range(Pt):
            for k in range(sub_patches):
                f0 = start + p * patch_size + k * Q
                f1 = f0 + Q
                pred_full[f0:f1] = out[p, k]   # [J, 6] broadcast over Q frames

    return pred_full[:N]   # [N, J, 6]


# ---------------------------------------------------------------------------
# T-pose reset helper
# ---------------------------------------------------------------------------

def apply_tpose_reset_window(
    R_local: np.ndarray,   # [T_win, 24, 3, 3]  GT local rotations for one window
    Q: int,                # sub-patch size (frames); R0 taken at frame Q//2
) -> np.ndarray:
    """Normalise root (joint 0) so frame Q//2 has identity root → T-pose start.

    Returns [T_win, 24, 3, 3] with only joint 0 modified.
    """
    R0    = R_local[Q // 2, 0]       # [3, 3]  root rotation at first sub-patch centre
    R0_inv = R0.T                     # orthogonal inverse
    out   = R_local.copy()
    out[:, 0] = np.einsum("ij,njk->nik", R0_inv, R_local[:, 0])
    return out


# ---------------------------------------------------------------------------
# FK + MPJPE / MPVPE
# ---------------------------------------------------------------------------

def _fk_positions(R_local: np.ndarray, body_model, device: torch.device) -> np.ndarray:
    """Run SMPL FK → joint positions [N, 24, 3] in metres."""
    N  = R_local.shape[0]
    Rt = torch.from_numpy(R_local).float().to(device)
    result = body_model.forward_kinematics(Rt.reshape(N, -1))
    pos = result[1] if isinstance(result, (tuple, list)) else result
    if hasattr(pos, "detach"):
        pos = pos.detach().cpu().numpy()
    return pos   # [N, 24, 3]


def compute_mpjpe(pos_pred, pos_gt, joint_indices) -> np.ndarray:
    """Per-frame MPJPE in mm over selected joints. [N]."""
    return np.linalg.norm(
        pos_pred[:, joint_indices] - pos_gt[:, joint_indices], axis=-1
    ).mean(axis=-1) * 1000.0


def _fk_verts(R_local: np.ndarray, body_model, device: torch.device,
              batch_size: int = 64):
    """Run batched SMPL FK with mesh → (joint positions, vertices)."""
    N       = R_local.shape[0]
    all_jpos, all_verts = [], []
    for s in range(0, N, batch_size):
        e  = min(s + batch_size, N)
        Nb = e - s
        Rt = torch.from_numpy(R_local[s:e]).float().to(device)
        try:
            _, jpos, verts = body_model.forward_kinematics(
                Rt.reshape(Nb, -1), calc_mesh=True)
        except Exception:
            return None, None
        all_jpos.append(jpos.detach().cpu().numpy())
        all_verts.append(verts.detach().cpu().numpy())
    return (np.concatenate(all_jpos), np.concatenate(all_verts))


def compute_mpvpe(
    pred_R_full: np.ndarray,   # [N, 24, 3, 3]
    gt_R_full: np.ndarray,     # [N, 24, 3, 3]
    body_model,
    device: torch.device,
) -> np.ndarray:
    """Mean per-vertex position error in cm (pelvis-aligned). [N]"""
    jpos_gt, verts_gt = _fk_verts(gt_R_full, body_model, device)
    if verts_gt is None:
        return None
    jpos_pred, verts_pred = _fk_verts(pred_R_full, body_model, device)
    if verts_pred is None:
        return None

    verts_gt_a   = verts_gt   - jpos_gt[:, :1]
    verts_pred_a = verts_pred - jpos_pred[:, :1]
    return (np.linalg.norm(verts_pred_a - verts_gt_a, axis=-1).mean(axis=-1) * 100.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    ap = argparse.ArgumentParser(description="Eval 24-joint sub-patch pose model")
    ap.add_argument("--config",       required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--run",          required=True)
    ap.add_argument("--test-dir",     required=True)
    ap.add_argument("--ckpt",         default="best")
    ap.add_argument("--fps",          type=int, default=50)
    ap.add_argument("--max-frames",   type=int, default=0)
    ap.add_argument("--out-dir",      default="data_vis/pose_eval_24j_tpose")
    args = ap.parse_args()

    base_cfg  = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    cfg = deep_update(base_cfg, model_cfg)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Load checkpoint ---- #
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

    vit_cfg     = cfg_get(ckpt_cfg, ["vit1d"], {}) or {}
    dec_cfg     = (vit_cfg.get("pose_decoder") or {})
    pose_cfg    = cfg_get(ckpt_cfg, ["objective", "pose"], {}) or {}
    T_win, _    = compute_T_and_hop(ckpt_cfg)
    patch_size  = int(vit_cfg.get("patch_size", 16))
    sub_patches = int(dec_cfg.get("sub_patches", 1))
    Q           = patch_size // sub_patches
    num_joints  = int(pose_cfg.get("num_joints", 24))
    add_gravity = bool(pose_cfg.get("add_gravity", True))

    logger.info("patch_size=%d  sub_patches=%d  Q=%d (%.3fs)  num_joints=%d",
                patch_size, sub_patches, Q, Q / args.fps, num_joints)

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

    # ---- FK model (for MPJPE / MPVPE) ---- #
    try:
        import core.articulate as art
        from core.paths import Paths
        body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=device)
        has_fk = True
        logger.info("SMPL body model loaded for FK")
    except Exception as exc:
        body_model = None
        logger.warning("FK unavailable (%s) — skipping MPJPE / MPVPE", exc)
        has_fk = False

    # ---- Test files ---- #
    pkl_files = sorted(glob.glob(os.path.join(args.test_dir, "*.pkl")))
    if not pkl_files:
        raise SystemExit(f"No pkl files in {args.test_dir}")
    logger.info("Found %d test sequences", len(pkl_files))

    # Per-frame accumulators (over all sequences)
    mpjpe_24j_all = []   # 24-joint, T-pose-reset space
    mpvpe_24j_all = []
    mpjpe_23j_all = []   # 23-joint, no reset
    mpvpe_23j_all = []
    geo_24j_all   = []   # geodesic (24j, tpose)
    geo_23j_all   = []   # geodesic (23j)
    seq_names     = []
    seq_failed    = []

    for pkl_path in tqdm(pkl_files, desc="Evaluating", unit="seq"):
        seq_name = os.path.basename(pkl_path).replace(".pkl", "")
        try:
            X, smpl_R_local = load_sequence(
                pkl_path, args.fps, args.max_frames, add_gravity)
        except Exception as exc:
            logger.warning("Skipping %s: %s", seq_name, exc)
            seq_failed.append(seq_name)
            continue

        N    = len(X)
        pred = run_inference(X, encoder, pose_decoder, ckpt_cfg, device)
        # pred: [N, 24, 6]  — T-pose-normalised root for joint 0

        # Convert pred 6D → rotation matrices
        pred_R = rot6d_to_matrix(torch.from_numpy(pred)).numpy()   # [N, 24, 3, 3]

        # ------------------------------------------------------------------ #
        # 24-joint geodesic + MPJPE / MPVPE  (T-pose reset per window)
        # ------------------------------------------------------------------ #
        gt_R_24j_tpose   = smpl_R_local.copy()   # [N, 24, 3, 3] — will be filled window by window
        pred_R_24j_tpose = pred_R.copy()

        for w in range(0, N, T_win):
            end   = min(w + T_win, N)
            T_seg = end - w
            if T_seg < Q:
                continue
            gt_seg  = smpl_R_local[w:end]           # [T_seg, 24, 3, 3]
            gt_reset = apply_tpose_reset_window(gt_seg, Q)
            gt_R_24j_tpose[w:end] = gt_reset

        # Geodesic (24j)
        pred_t = torch.from_numpy(pred_R_24j_tpose).float()  # [N, 24, 3, 3]
        gt_t   = torch.from_numpy(gt_R_24j_tpose).float()
        geo_24 = geodesic_deg(pred_t.reshape(-1, 3, 3), gt_t.reshape(-1, 3, 3)
                              ).reshape(N, 24).mean(dim=-1).numpy()   # [N]
        geo_24j_all.extend(geo_24.tolist())

        if has_fk:
            pos_gt_24   = _fk_positions(gt_R_24j_tpose,   body_model, device)
            pos_pred_24 = _fk_positions(pred_R_24j_tpose, body_model, device)
            mpjpe_24 = compute_mpjpe(pos_pred_24, pos_gt_24, list(range(24)))
            mpjpe_24j_all.extend(mpjpe_24.tolist())

            mpvpe_24 = compute_mpvpe(pred_R_24j_tpose, gt_R_24j_tpose, body_model, device)
            if mpvpe_24 is not None:
                mpvpe_24j_all.extend(mpvpe_24.tolist())

        # ------------------------------------------------------------------ #
        # 23-joint geodesic + MPJPE / MPVPE  (no T-pose reset, joints 1-23)
        # ------------------------------------------------------------------ #
        pred_R_23j = pred_R[:, 1:]             # [N, 23, 3, 3]
        gt_R_23j   = smpl_R_local[:, 1:]      # [N, 23, 3, 3]

        pred_t23 = torch.from_numpy(pred_R_23j).float()
        gt_t23   = torch.from_numpy(gt_R_23j).float()
        geo_23 = geodesic_deg(pred_t23.reshape(-1, 3, 3), gt_t23.reshape(-1, 3, 3)
                              ).reshape(N, 23).mean(dim=-1).numpy()
        geo_23j_all.extend(geo_23.tolist())

        if has_fk:
            # Build full rotation matrices with pred joints 1-23, GT joint 0
            pred_R_23j_full = smpl_R_local.copy()
            pred_R_23j_full[:, 1:] = pred_R_23j

            pos_gt_23   = _fk_positions(smpl_R_local,   body_model, device)
            pos_pred_23 = _fk_positions(pred_R_23j_full, body_model, device)
            mpjpe_23 = compute_mpjpe(pos_pred_23, pos_gt_23, list(range(1, 24)))
            mpjpe_23j_all.extend(mpjpe_23.tolist())

            mpvpe_23 = compute_mpvpe(pred_R_23j_full, smpl_R_local, body_model, device)
            if mpvpe_23 is not None:
                mpvpe_23j_all.extend(mpvpe_23.tolist())

        seq_names.append(seq_name)

    if not seq_names:
        raise SystemExit("No sequences evaluated.")

    # ---- Print results ---- #
    total = sum(len(s) for s in [geo_24j_all])
    print(f"\n{'='*64}")
    print(f"  24-joint + 23-joint Pose Evaluation")
    print(f"  Run       : {args.run}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Sequences : {len(seq_names)}   Frames: {total}")
    print(f"{'='*64}")

    geo24 = np.array(geo_24j_all)
    geo23 = np.array(geo_23j_all)
    print(f"\n  24-joint (T-pose reset each window):")
    print(f"    Geodesic error  : {geo24.mean():.2f} ± {geo24.std():.2f} °")
    if mpjpe_24j_all:
        m = np.array(mpjpe_24j_all)
        print(f"    MPJPE           : {m.mean():.2f} ± {m.std():.2f} mm")
    if mpvpe_24j_all:
        v = np.array(mpvpe_24j_all)
        print(f"    MPVPE (pelvis-aligned): {v.mean():.2f} ± {v.std():.2f} cm")

    print(f"\n  23-joint (no T-pose reset, joints 1-23, GT root):")
    print(f"    Geodesic error  : {geo23.mean():.2f} ± {geo23.std():.2f} °")
    if mpjpe_23j_all:
        m = np.array(mpjpe_23j_all)
        print(f"    MPJPE           : {m.mean():.2f} ± {m.std():.2f} mm")
    if mpvpe_23j_all:
        v = np.array(mpvpe_23j_all)
        print(f"    MPVPE (pelvis-aligned): {v.mean():.2f} ± {v.std():.2f} cm")

    if seq_failed:
        print(f"\n  Skipped {len(seq_failed)} sequences: {seq_failed[:5]}"
              f"{'...' if len(seq_failed) > 5 else ''}")
    print()

    # ---- Plots ---- #
    def _save(fig, name):
        p = os.path.join(args.out_dir, name)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", p)

    # Geodesic histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, label, color in [
        (axes[0], geo24, "24-joint (T-pose reset)", "steelblue"),
        (axes[1], geo23, "23-joint (GT root)",      "tomato"),
    ]:
        ax.hist(data, bins=60, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(data.mean(), color="k", lw=1.4, ls="--",
                   label=f"Mean: {data.mean():.2f}°")
        ax.set_xlabel("Geodesic error (°)")
        ax.set_ylabel("Frame count")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Geodesic Error — {len(seq_names)} test sequences")
    fig.tight_layout()
    _save(fig, "geodesic_hist.png")

    if mpjpe_24j_all or mpjpe_23j_all:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, data, label, color in [
            (axes[0], np.array(mpjpe_24j_all) if mpjpe_24j_all else np.array([]),
             "24-joint MPJPE (T-pose reset)", "steelblue"),
            (axes[1], np.array(mpjpe_23j_all) if mpjpe_23j_all else np.array([]),
             "23-joint MPJPE (GT root)", "tomato"),
        ]:
            if len(data) == 0:
                ax.set_visible(False)
                continue
            ax.hist(data, bins=60, color=color, alpha=0.75, edgecolor="none")
            ax.axvline(data.mean(), color="k", lw=1.4, ls="--",
                       label=f"Mean: {data.mean():.2f} mm")
            ax.set_xlabel("MPJPE (mm)")
            ax.set_ylabel("Frame count")
            ax.set_title(label)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"MPJPE — {len(seq_names)} test sequences")
        fig.tight_layout()
        _save(fig, "mpjpe_hist.png")

    if mpvpe_24j_all or mpvpe_23j_all:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, data, label, color in [
            (axes[0], np.array(mpvpe_24j_all) if mpvpe_24j_all else np.array([]),
             "24-joint MPVPE (T-pose reset)", "mediumpurple"),
            (axes[1], np.array(mpvpe_23j_all) if mpvpe_23j_all else np.array([]),
             "23-joint MPVPE (GT root)", "teal"),
        ]:
            if len(data) == 0:
                ax.set_visible(False)
                continue
            ax.hist(data, bins=60, color=color, alpha=0.75, edgecolor="none")
            ax.axvline(data.mean(), color="k", lw=1.4, ls="--",
                       label=f"Mean: {data.mean():.2f} cm")
            ax.set_xlabel("MPVPE (cm, pelvis-aligned)")
            ax.set_ylabel("Frame count")
            ax.set_title(label)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"MPVPE — {len(seq_names)} test sequences")
        fig.tight_layout()
        _save(fig, "mpvpe_hist.png")

    logger.info("Done. Results in %s", args.out_dir)


if __name__ == "__main__":
    main()
