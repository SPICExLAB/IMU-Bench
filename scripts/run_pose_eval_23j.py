"""Evaluation for the 23-joint full-body pose model (no global orientation).

Metric groups reported
----------------------
  23-joint Geodesic / MPJPE / MPVPE  — joints 1-23 only (parent-relative local
      rotations).  The root/pelvis (joint 0) is not predicted by the model; GT
      root is spliced in for FK so that position errors are meaningful.

Usage
-----
    python scripts/run_pose_eval_23j.py \\
        --config       configs/base.yaml \\
        --model-config configs/vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_23j.yaml \\
        --run          vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_23j \\
        --test-dir     /mnt/nas2/processed_data_test_body_Ty_rwrist \\
        [--ckpt        best]
        [--out-dir     data_vis/pose_eval_23j]
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
    return torch.stack((b1, b2, b3), dim=-2)


def geodesic_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    R_diff = R_pred @ R_gt.transpose(-1, -2)
    trace  = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    return torch.rad2deg(torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)))


# ---------------------------------------------------------------------------
# Data loading (Nymeria real-wrist FLU format)
# ---------------------------------------------------------------------------

def load_sequence(pkl_path: str, fps: int, max_frames: int, add_gravity: bool,
                  body_model_ik=None):
    """Load a Nymeria pkl and return FLU IMU + full SMPL local rotations.

    Returns
    -------
    X            : [N, 6]        FLU-frame wrist IMU (acc + gyro)
    smpl_R_local : [N, 24, 3, 3] SMPL local rotation matrices (all 24 joints)
    """
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

    smpl_R_global = torch.from_numpy(
        _xsens_to_smpl_rotmat(xsens_pose[:, :, :3, :3])
    )   # [N, 24, 3, 3]

    if body_model_ik is None:
        import core.articulate as art
        from core.paths import Paths
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
    infer_batch_size: int = 64,
) -> np.ndarray:
    """Return per-frame 23-joint 6D predictions [N, 23, 6]."""
    vit_cfg     = cfg_get(cfg, ["vit1d"], {}) or {}
    dec_cfg     = (vit_cfg.get("pose_decoder") or {})
    T           = int(cfg_get(cfg, ["windowing", "window_seconds"], 2.56) *
                      cfg_get(cfg, ["windowing", "sample_rate_hz"], 50))
    patch_size  = int(vit_cfg.get("patch_size", 16))
    sub_patches = int(dec_cfg.get("sub_patches", 1))
    Q           = patch_size // sub_patches
    J           = pose_decoder.num_joints   # 23
    stats       = PreprocessStats()
    N           = len(X)

    pad   = (T - N % T) % T
    X_pad = np.concatenate([X, np.zeros((pad, 6), dtype=np.float32)]) if pad else X

    pred_full  = np.zeros((len(X_pad), J, 6), dtype=np.float32)
    Pt         = T // patch_size
    num_wins   = len(X_pad) // T

    # Preprocess all windows, track valid indices
    windows  = []
    win_idxs = []
    for w in range(num_wins):
        Xproc = preprocess_window(X_pad[w * T : (w + 1) * T], cfg, stats)
        if Xproc is not None:
            windows.append(Xproc)
            win_idxs.append(w)

    # Run encoder + decoder in batches
    for b_start in range(0, len(windows), infer_batch_size):
        batch_np  = windows[b_start : b_start + infer_batch_size]
        batch_idx = win_idxs[b_start : b_start + infer_batch_size]
        x = torch.from_numpy(np.stack(batch_np)).float().to(device)  # [B, C, T]
        tokens, _ = encoder.forward_tokens(x)
        outs = pose_decoder(tokens).cpu().numpy()  # [B, Pt, K*J*6]
        outs = outs.reshape(len(batch_np), Pt, sub_patches, J, 6)

        for i, w in enumerate(batch_idx):
            start = w * T
            for p in range(Pt):
                for k in range(sub_patches):
                    f0 = start + p * patch_size + k * Q
                    pred_full[f0 : f0 + Q] = outs[i, p, k]

    return pred_full[:N]   # [N, 23, 6]


# ---------------------------------------------------------------------------
# FK + MPJPE / MPVPE
# ---------------------------------------------------------------------------

def _fk_positions(R_local: np.ndarray, body_model, device: torch.device,
                  batch_size: int = 512) -> np.ndarray:
    """Run SMPL FK → joint positions [N, 24, 3] in metres, batched."""
    N    = R_local.shape[0]
    out  = []
    for s in range(0, N, batch_size):
        Rt = torch.from_numpy(R_local[s : s + batch_size]).float().to(device)
        result = body_model.forward_kinematics(Rt.reshape(Rt.shape[0], -1))
        pos = result[1] if isinstance(result, (tuple, list)) else result
        out.append(pos.detach().cpu().numpy())
    return np.concatenate(out, axis=0)   # [N, 24, 3]


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
    ap = argparse.ArgumentParser(description="Eval 23-joint pose model (no global orientation)")
    ap.add_argument("--config",       required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--run",          required=True)
    ap.add_argument("--test-dir",     required=True)
    ap.add_argument("--ckpt",         default="best")
    ap.add_argument("--fps",          type=int, default=50)
    ap.add_argument("--max-frames",   type=int, default=0)
    ap.add_argument("--out-dir",      default="data_vis/pose_eval_23j")
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

    from imu_lm.models.baselines.inertia1_encoder import build_encoder as _build_encoder
    encoder = _build_encoder(ckpt_cfg).to(device)   # ViT1D / alt-attn / Inertia-1 adapter per ckpt cfg
    encoder.load_state_dict(state["model"])
    encoder.eval()

    vit_cfg     = cfg_get(ckpt_cfg, ["vit1d"], {}) or {}
    dec_cfg     = (vit_cfg.get("pose_decoder") or {})
    pose_cfg    = cfg_get(ckpt_cfg, ["objective", "pose"], {}) or {}
    T_win, _    = compute_T_and_hop(ckpt_cfg)
    patch_size  = int(vit_cfg.get("patch_size", 16))
    sub_patches = int(dec_cfg.get("sub_patches", 1))
    Q           = patch_size // sub_patches
    num_joints  = int(pose_cfg.get("num_joints", 23))
    add_gravity = bool(pose_cfg.get("add_gravity", True))

    if num_joints != 23:
        raise SystemExit(
            f"This script expects a 23-joint model, but config has num_joints={num_joints}. "
            "Use run_pose_eval_24j_tpose.py for 24-joint models."
        )

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

    # ---- FK / IK models (loaded once, shared across all sequences) ---- #
    try:
        import core.articulate as art
        from core.paths import Paths
        body_model_ik = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))
        body_model    = art.model.ParametricModel(Paths.SMPL_FILE, device=device)
        has_fk = True
        logger.info("SMPL body model loaded for IK + FK")
    except Exception as exc:
        body_model_ik = None
        body_model    = None
        logger.warning("FK unavailable (%s) — skipping MPJPE / MPVPE", exc)
        has_fk = False

    # ---- Test files ---- #
    pkl_files = sorted(glob.glob(os.path.join(args.test_dir, "*.pkl")))
    if not pkl_files:
        raise SystemExit(f"No pkl files in {args.test_dir}")
    logger.info("Found %d test sequences", len(pkl_files))

    geo_23j_all   = []
    mpjpe_23j_all = []
    mpvpe_23j_all = []
    seq_names     = []
    seq_failed    = []

    for pkl_path in tqdm(pkl_files, desc="Evaluating", unit="seq"):
        seq_name = os.path.basename(pkl_path).replace(".pkl", "")
        try:
            X, smpl_R_local = load_sequence(
                pkl_path, args.fps, args.max_frames, add_gravity,
                body_model_ik=body_model_ik)
        except Exception as exc:
            logger.warning("Skipping %s: %s", seq_name, exc)
            seq_failed.append(seq_name)
            continue

        N    = len(X)
        pred = run_inference(X, encoder, pose_decoder, ckpt_cfg, device)
        # pred: [N, 23, 6] — joints 1-23 (no root)

        pred_R = rot6d_to_matrix(torch.from_numpy(pred)).numpy()   # [N, 23, 3, 3]

        # ------------------------------------------------------------------ #
        # 23-joint geodesic (joints 1-23, no T-pose reset)
        # ------------------------------------------------------------------ #
        gt_R_23j = smpl_R_local[:, 1:]   # [N, 23, 3, 3]

        pred_t = torch.from_numpy(pred_R).float()
        gt_t   = torch.from_numpy(gt_R_23j).float()
        geo_23 = geodesic_deg(pred_t.reshape(-1, 3, 3), gt_t.reshape(-1, 3, 3)
                              ).reshape(N, 23).mean(dim=-1).numpy()
        geo_23j_all.extend(geo_23.tolist())

        if has_fk:
            # Splice GT root back in so FK produces meaningful positions
            pred_R_full = smpl_R_local.copy()
            pred_R_full[:, 1:] = pred_R   # [N, 24, 3, 3]

            pos_gt   = _fk_positions(smpl_R_local, body_model, device)
            pos_pred = _fk_positions(pred_R_full,  body_model, device)
            mpjpe_23 = compute_mpjpe(pos_pred, pos_gt, list(range(1, 24)))
            mpjpe_23j_all.extend(mpjpe_23.tolist())

            mpvpe_23 = compute_mpvpe(pred_R_full, smpl_R_local, body_model, device)
            if mpvpe_23 is not None:
                mpvpe_23j_all.extend(mpvpe_23.tolist())

        seq_names.append(seq_name)

    if not seq_names:
        raise SystemExit("No sequences evaluated.")

    # ---- Print results ---- #
    total = len(geo_23j_all)
    print(f"\n{'='*64}")
    print(f"  23-joint Pose Evaluation  (joints 1-23, GT root)")
    print(f"  Run       : {args.run}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Sequences : {len(seq_names)}   Frames: {total}")
    print(f"{'='*64}")

    geo23 = np.array(geo_23j_all)
    print(f"\n  23-joint (no T-pose reset, GT root for FK):")
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

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(geo23, bins=60, color="tomato", alpha=0.75, edgecolor="none")
    ax.axvline(geo23.mean(), color="k", lw=1.4, ls="--",
               label=f"Mean: {geo23.mean():.2f}°")
    ax.set_xlabel("Geodesic error (°)")
    ax.set_ylabel("Frame count")
    ax.set_title(f"23-joint Geodesic Error — {len(seq_names)} test sequences")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "geodesic_hist.png")

    if mpjpe_23j_all:
        fig, ax = plt.subplots(figsize=(7, 4))
        m = np.array(mpjpe_23j_all)
        ax.hist(m, bins=60, color="steelblue", alpha=0.75, edgecolor="none")
        ax.axvline(m.mean(), color="k", lw=1.4, ls="--",
                   label=f"Mean: {m.mean():.2f} mm")
        ax.set_xlabel("MPJPE (mm)")
        ax.set_ylabel("Frame count")
        ax.set_title(f"23-joint MPJPE — {len(seq_names)} test sequences")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, "mpjpe_hist.png")

    if mpvpe_23j_all:
        fig, ax = plt.subplots(figsize=(7, 4))
        v = np.array(mpvpe_23j_all)
        ax.hist(v, bins=60, color="mediumpurple", alpha=0.75, edgecolor="none")
        ax.axvline(v.mean(), color="k", lw=1.4, ls="--",
                   label=f"Mean: {v.mean():.2f} cm")
        ax.set_xlabel("MPVPE (cm, pelvis-aligned)")
        ax.set_ylabel("Frame count")
        ax.set_title(f"23-joint MPVPE — {len(seq_names)} test sequences")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, "mpvpe_hist.png")

    logger.info("Done. Results in %s", args.out_dir)


if __name__ == "__main__":
    main()
