"""Dataset-level pose evaluation for the ViT1D right-arm model.

Runs inference over every pkl in a test directory, aggregates per-joint
rotation error and MPJPE (Mean Per-Joint Position Error) across the whole
test set, and saves summary PNGs.

Metrics
-------
  Geodesic error (°) — angular distance between predicted and GT rotation matrices
  MPJPE (mm)         — L2 distance between predicted and GT 3-D joint positions
                       derived via SMPL forward kinematics (requires core.articulate)

Output PNGs (--out-dir)
-----------------------
  per_joint_geodesic.png   — mean ± std geodesic error per right-arm joint
  per_joint_mpjpe.png      — mean ± std MPJPE  per right-arm joint
  geodesic_cdf.png         — CDF of per-frame geodesic error for each joint
  error_vs_sequence.png    — mean geodesic error per test sequence (sorted)

Usage
-----
    python scripts/run_pose_eval_dataset.py \\
        --config       configs/base.yaml \\
        --model-config configs/vit1d_pose.yaml \\
        --run          vit1d_nymeria_pose \\
        --test-dir     /mnt/nas2/processed_data_test_body_Ty \\
        [--ckpt        best]          # "best" | "latest" | abs path
        [--max-frames  0]             # per-sequence cap (0 = full)
        [--fps         50]
        [--out-dir     data_vis/pose_eval_dataset]
        [--num-joints  5]             # 5 (right-arm chain) or 24 (full body)
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
from imu_lm.data.nymeria_wrist_loader import (
    RIGHT_WRIST_CHAIN_JOINTS,
    _XSENS_RIGHT_WRIST_JOINT,
    _xsens_to_smpl_rotmat,
)
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DPoseDecoder
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

logger = logging.getLogger(__name__)

# Right-arm chain joint names (5-joint mode)
JOINT_NAMES_5  = ["R-Collar", "R-Shoulder", "R-Elbow", "R-Wrist", "R-Hand"]
SMPL_JOINT_NAMES = [
    "Pelvis", "L-Hip", "R-Hip", "Spine1", "L-Knee", "R-Knee",
    "Spine2", "L-Ankle", "R-Ankle", "Spine3", "L-Foot", "R-Foot",
    "Neck", "L-Collar", "R-Collar", "Head", "L-Shoulder", "R-Shoulder",
    "L-Elbow", "R-Elbow", "L-Wrist", "R-Wrist", "L-Hand", "R-Hand",
]


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


def geodesic_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    R_diff = torch.matmul(R_pred, R_gt.transpose(-1, -2))
    trace  = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos    = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


# ---------------------------------------------------------------------------
# Per-sequence data loading
# ---------------------------------------------------------------------------

def load_sequence(pkl_path: str, fps: int, max_frames: int, num_joints: int):
    """Load IMU, GT 6D rotations, and full SMPL local rotation matrices.

    Returns (X, gt_6d, smpl_R_local, actual_fps) or raises on failure.
      X             : [N, 6]             — right-wrist body-frame IMU
      gt_6d         : [N, num_joints, 6] — GT 6D rotations for queried joints
      smpl_R_local  : [N, 24, 3, 3]     — full SMPL local rotmats (for FK)
      actual_fps    : float
    """
    import core.articulate as art
    from core.paths import Paths

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    xsens_pose = raw["gt_data"]["xsens_pose"]
    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float32)   # [N, 23, 4, 4]

    actual_fps = float(
        raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or fps
    )
    N = xsens_pose.shape[0]
    if max_frames > 0:
        N = min(N, max_frames)
        xsens_pose = xsens_pose[:N]

    # ---- Body-frame wrist IMU (same as training) ---- #
    j   = _XSENS_RIGHT_WRIST_JOINT
    pos = xsens_pose[:, j, :3, 3]
    rot = xsens_pose[:, j, :3, :3]

    acc       = np.zeros((N, 3), dtype=np.float32)
    acc[1:-1] = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) * (actual_fps ** 2)
    acc[0]    = acc[1];  acc[-1] = acc[-2]

    dR       = np.einsum("nij,njk->nik", rot[:-1].transpose(0, 2, 1), rot[1:])
    gyro_raw = (ScipyRot.from_matrix(dR.reshape(-1, 3, 3)).as_rotvec() * actual_fps)
    gyro     = np.zeros((N, 3), dtype=np.float32)
    gyro[:-1] = gyro_raw.astype(np.float32);  gyro[-1] = gyro[-2]
    X = np.concatenate([acc, gyro], axis=1)   # [N, 6]

    # ---- GT SMPL local rotations via IK ---- #
    smpl_R_global = torch.from_numpy(
        _xsens_to_smpl_rotmat(xsens_pose[:, :, :3, :3])
    )  # [N, 24, 3, 3]
    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))
    smpl_R_local = (
        body_model
        .inverse_kinematics_R(smpl_R_global.reshape(N, -1))
        .view(N, 24, 3, 3)
        .contiguous().clone().cpu().numpy()
    )  # [N, 24, 3, 3]  local rotations

    if num_joints == 24:
        sel_R = smpl_R_local
    else:
        sel_R = smpl_R_local[:, RIGHT_WRIST_CHAIN_JOINTS]   # [N, 5, 3, 3]

    gt_6d = np.concatenate([sel_R[..., 0], sel_R[..., 1]], axis=-1)  # [N, J, 6]
    return X, gt_6d, smpl_R_local, actual_fps


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(X: np.ndarray, encoder, pose_decoder, cfg, device, fps: int) -> np.ndarray:
    """Return per-frame 6D predictions [N, num_joints, 6]."""
    T, _       = compute_T_and_hop(cfg)
    vit_cfg    = cfg_get(cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))
    num_joints = int(cfg_get(cfg, ["objective", "pose", "num_joints"], 5))
    stats      = PreprocessStats()
    N          = len(X)

    pad   = (T - N % T) % T
    X_pad = np.concatenate([X, np.zeros((pad, 6), dtype=np.float32)]) if pad else X

    pred_full = np.zeros((len(X_pad), num_joints, 6), dtype=np.float32)
    for w in range(len(X_pad) // T):
        start = w * T
        Xw    = X_pad[start:start + T]
        Xproc = preprocess_window(Xw, cfg, stats)
        if Xproc is None:
            continue
        x = torch.from_numpy(Xproc).float().unsqueeze(0).to(device)
        tokens, _ = encoder.forward_tokens(x)
        pred_np   = pose_decoder(tokens).squeeze(0).cpu().numpy()   # [Pt, J*6]
        for p in range(pred_np.shape[0]):
            f0 = start + p * patch_size
            f1 = f0 + patch_size
            pred_full[f0:f1] = pred_np[p].reshape(num_joints, 6)

    return pred_full[:N]


# ---------------------------------------------------------------------------
# MPJPE via forward kinematics
# ---------------------------------------------------------------------------

def compute_mpjpe_mm(
    pred_6d: np.ndarray,          # [N, num_joints, 6]
    smpl_R_gt_local: np.ndarray,  # [N, 24, 3, 3]
    num_joints: int,
    device: torch.device,
) -> np.ndarray:
    """Return MPJPE in millimetres per frame per joint  [N, num_joints].

    Uses SMPL forward kinematics (core.articulate) to convert local rotations
    → 3-D joint world positions, then computes per-joint L2 distance.

    Requires core.articulate. Returns None if unavailable.
    """
    try:
        import core.articulate as art
        from core.paths import Paths
    except ImportError:
        return None

    N = pred_6d.shape[0]

    # ---- Convert predicted 6D → local rotation matrices ---- #
    pred_R_sel = rotation_6d_to_matrix(
        torch.from_numpy(pred_6d)
    ).numpy()   # [N, num_joints, 3, 3]

    # Build full predicted local rotation matrices:
    # replace queried joints in GT, keep everything else from GT
    pred_R_full = smpl_R_gt_local.copy()   # [N, 24, 3, 3]
    if num_joints == 24:
        pred_R_full = pred_R_sel
    else:
        for ci, ji in enumerate(RIGHT_WRIST_CHAIN_JOINTS):
            pred_R_full[:, ji] = pred_R_sel[:, ci]

    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=device)

    def _fk_positions(R_local_np):
        # forward_kinematics returns (pose_global [N,24,3,3], joint_positions [N,24,3])
        # We need the joint POSITIONS (index 1), not the rotation matrices (index 0).
        R_t    = torch.from_numpy(R_local_np).float().to(device)
        R_flat = R_t.reshape(N, -1)
        try:
            result = body_model.forward_kinematics(R_flat)
            if isinstance(result, (tuple, list)):
                pos = result[1]   # index 1 = joint positions [N, 24, 3] in metres
            else:
                pos = result
        except AttributeError:
            return None
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().numpy()
        return pos  # [N, 24, 3]  in metres

    pos_gt   = _fk_positions(smpl_R_gt_local)
    pos_pred = _fk_positions(pred_R_full)
    if pos_gt is None or pos_pred is None:
        logger.warning("FK unavailable — skipping MPJPE")
        return None

    if num_joints == 24:
        sel_idx = list(range(24))
    else:
        sel_idx = RIGHT_WRIST_CHAIN_JOINTS

    # L2 distance in mm
    mpjpe = (
        np.linalg.norm(pos_pred[:, sel_idx] - pos_gt[:, sel_idx], axis=-1) * 1000.0
    )  # [N, num_joints]
    return mpjpe


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved  %s", path)


def plot_per_joint_bar(means, stds, ylabel, title, out_path, joint_names, color):
    fig, ax = plt.subplots(figsize=(max(8, len(joint_names) * 0.7 + 2), 5))
    x = np.arange(len(joint_names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=color, width=0.55, alpha=0.85,
                  error_kw=dict(ecolor="dimgray", lw=1.2))
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(means) * 0.015,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(joint_names, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    # Overall mean annotation
    overall = float(np.mean(means))
    ax.axhline(overall, color="r", lw=1.2, ls="--",
               label=f"Overall mean: {overall:.1f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_path)


def plot_cdf(all_errors_per_joint, joint_names, out_path, xlabel):
    """CDF of per-frame errors for each joint on one axes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.tab10
    for ji, (jname, errs) in enumerate(zip(joint_names, all_errors_per_joint)):
        errs_sorted = np.sort(errs)
        cdf = np.arange(1, len(errs_sorted) + 1) / len(errs_sorted)
        ax.plot(errs_sorted, cdf, label=jname, color=cmap(ji % 10), linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"CDF — {xlabel}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_error_per_sequence(seq_names, seq_geo_means, out_path):
    """Bar chart: mean geodesic error for each test sequence (sorted)."""
    order = np.argsort(seq_geo_means)
    names_sorted = [seq_names[i] for i in order]
    vals_sorted  = [seq_geo_means[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(10, len(seq_names) * 0.35 + 2), 5))
    x = np.arange(len(names_sorted))
    ax.bar(x, vals_sorted, color="steelblue", width=0.8, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names_sorted, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Mean geodesic error (°)")
    ax.set_title("Per-sequence mean geodesic error (all right-arm joints)")
    ax.axhline(np.mean(vals_sorted), color="r", lw=1.2, ls="--",
               label=f"Dataset mean: {np.mean(vals_sorted):.1f}°")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    ap = argparse.ArgumentParser(
        description="Dataset-level pose eval — geodesic + MPJPE across all test pkls"
    )
    ap.add_argument("--config",       required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--run",          required=True, help="Run name under runs/<run>")
    ap.add_argument("--test-dir",     required=True, help="Directory of Nymeria test pkl files")
    ap.add_argument("--ckpt",         default="best")
    ap.add_argument("--max-frames",   type=int, default=0, help="Per-sequence frame cap (0=all)")
    ap.add_argument("--fps",          type=int, default=50)
    ap.add_argument("--out-dir",      default="data_vis/pose_eval_dataset")
    ap.add_argument("--num-joints",   type=int, default=None,
                    help="Override num_joints (default: read from model config)")
    args = ap.parse_args()

    base_cfg  = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    cfg = deep_update(base_cfg, model_cfg)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Load checkpoint -------------------------------------------------- #
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

    encoder = ViT1DEncoder(ckpt_cfg).to(device)
    encoder.load_state_dict(state["model"])
    encoder.eval()

    T_win, _   = compute_T_and_hop(ckpt_cfg)
    vit_cfg    = cfg_get(ckpt_cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))

    # Resolve num_joints from the saved checkpoint config, checking both
    # the end-to-end *pose* key and the probe/finetune variants.
    def _resolve_num_joints(c):
        for path in (
            ["objective", "pose",            "num_joints"],
            ["objective", "pose_probe",       "num_joints"],
            ["objective", "pose_finetune",    "num_joints"],
        ):
            v = cfg_get(c, path, None)
            if v is not None:
                return int(v)
        return None

    num_joints = args.num_joints or _resolve_num_joints(ckpt_cfg) or _resolve_num_joints(cfg) or 5
    joint_names = SMPL_JOINT_NAMES if num_joints == 24 else JOINT_NAMES_5
    logger.info("num_joints=%d  joint_names=%s", num_joints, joint_names)

    pose_decoder = ViT1DPoseDecoder(
        embed_dim=encoder.embed_dim,
        num_patches_per_channel=T_win // patch_size,
        in_channels=encoder.in_channels,
        num_joints=num_joints,
        cfg=ckpt_cfg,
    ).to(device)
    pose_decoder.load_state_dict(state["pose_decoder"])
    pose_decoder.eval()

    # Override cfg so run_inference reads the right num_joints
    cfg.setdefault("objective", {}).setdefault("pose", {})["num_joints"] = num_joints

    # ---- Gather pkl files ------------------------------------------------- #
    pkl_files = sorted(glob.glob(os.path.join(args.test_dir, "*.pkl")))
    if not pkl_files:
        raise SystemExit(f"No pkl files found in {args.test_dir}")
    logger.info("Found %d pkl files in %s", len(pkl_files), args.test_dir)

    # ---- Per-sequence accumulators ---------------------------------------- #
    # geo_all[j] = flat list of per-frame geodesic errors for joint j
    geo_all     = [[] for _ in range(num_joints)]
    mpjpe_all   = [[] for _ in range(num_joints)]
    seq_names   = []
    seq_geo_means = []
    seq_failed  = []
    compute_mpjpe = True   # will be set False if FK is unavailable

    for pkl_path in tqdm(pkl_files, desc="Evaluating sequences", unit="seq"):
        seq_name = os.path.basename(pkl_path).replace(".pkl", "")
        try:
            X, gt_6d, smpl_R_local, actual_fps = load_sequence(
                pkl_path, args.fps, args.max_frames, num_joints
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", seq_name, exc)
            seq_failed.append(seq_name)
            continue

        pred_6d = run_inference(X, encoder, pose_decoder, cfg, device, int(actual_fps))
        N = len(pred_6d)

        # ---- Geodesic ---- #
        pred_t = torch.from_numpy(pred_6d)
        gt_t   = torch.from_numpy(gt_6d)
        pred_R = rotation_6d_to_matrix(pred_t)
        gt_R   = rotation_6d_to_matrix(gt_t)
        geo_seq = geodesic_error_deg(
            pred_R.reshape(-1, 3, 3), gt_R.reshape(-1, 3, 3)
        ).reshape(N, num_joints).numpy()   # [N, J]

        for j in range(num_joints):
            geo_all[j].extend(geo_seq[:, j].tolist())

        seq_geo_means.append(float(geo_seq.mean()))
        seq_names.append(seq_name)

        # ---- MPJPE ---- #
        if compute_mpjpe:
            mpjpe_seq = compute_mpjpe_mm(pred_6d, smpl_R_local, num_joints, device)
            if mpjpe_seq is None:
                compute_mpjpe = False
                logger.warning("FK unavailable — MPJPE will not be computed.")
            else:
                for j in range(num_joints):
                    mpjpe_all[j].extend(mpjpe_seq[:, j].tolist())

    if not seq_names:
        raise SystemExit("No sequences evaluated successfully.")

    # ---- Aggregate statistics --------------------------------------------- #
    geo_all_arr   = [np.array(g) for g in geo_all]   # per-joint flat arrays
    geo_means     = np.array([g.mean() for g in geo_all_arr])
    geo_stds      = np.array([g.std()  for g in geo_all_arr])

    # Print table
    total_frames = sum(len(g) for g in geo_all_arr)
    print(f"\n{'='*72}")
    print(f"  Dataset-level evaluation — {len(seq_names)} sequences  {total_frames} frames")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"{'='*72}")
    print(f"\n{'Joint':<16} {'Geodesic mean':>15} {'Geodesic std':>13}", end="")
    if compute_mpjpe and any(len(m) > 0 for m in mpjpe_all):
        print(f" {'MPJPE mean (mm)':>16} {'MPJPE std':>11}", end="")
    print()
    print("-" * (72 if not compute_mpjpe else 88))

    for ji, jname in enumerate(joint_names):
        row = f"{jname:<16} {geo_means[ji]:>14.2f}°  {geo_stds[ji]:>12.2f}°"
        if compute_mpjpe and len(mpjpe_all[ji]) > 0:
            m = np.array(mpjpe_all[ji])
            row += f"  {m.mean():>15.2f}   {m.std():>10.2f}"
        print(row)

    print("-" * (72 if not compute_mpjpe else 88))
    print(f"{'Mean':<16} {geo_means.mean():>14.2f}°  {geo_stds.mean():>12.2f}°", end="")
    if compute_mpjpe and any(len(m) > 0 for m in mpjpe_all):
        all_mpjpe_flat = np.concatenate([np.array(m) for m in mpjpe_all if len(m) > 0])
        print(f"  {all_mpjpe_flat.mean():>15.2f}   {all_mpjpe_flat.std():>10.2f}", end="")
    print()
    if seq_failed:
        print(f"\n  ⚠ Skipped {len(seq_failed)} sequences: {seq_failed[:5]}{'...' if len(seq_failed) > 5 else ''}")
    print()

    # ---- Save plots ------------------------------------------------------- #
    plot_per_joint_bar(
        geo_means, geo_stds,
        ylabel="Mean geodesic error (°)",
        title=f"Per-Joint Geodesic Error  ({len(seq_names)} test sequences)",
        out_path=os.path.join(args.out_dir, "per_joint_geodesic.png"),
        joint_names=joint_names,
        color="steelblue",
    )
    plot_cdf(
        geo_all_arr, joint_names,
        out_path=os.path.join(args.out_dir, "geodesic_cdf.png"),
        xlabel="Geodesic error (°)",
    )
    plot_error_per_sequence(
        seq_names, seq_geo_means,
        out_path=os.path.join(args.out_dir, "error_vs_sequence.png"),
    )

    if compute_mpjpe and any(len(m) > 0 for m in mpjpe_all):
        mpjpe_arr   = [np.array(m) for m in mpjpe_all]
        mpjpe_means = np.array([m.mean() for m in mpjpe_arr])
        mpjpe_stds  = np.array([m.std()  for m in mpjpe_arr])
        plot_per_joint_bar(
            mpjpe_means, mpjpe_stds,
            ylabel="MPJPE (mm)",
            title=f"Per-Joint MPJPE  ({len(seq_names)} test sequences)",
            out_path=os.path.join(args.out_dir, "per_joint_mpjpe.png"),
            joint_names=joint_names,
            color="tomato",
        )
        plot_cdf(
            mpjpe_arr, joint_names,
            out_path=os.path.join(args.out_dir, "mpjpe_cdf.png"),
            xlabel="MPJPE (mm)",
        )

    logger.info("All plots written to %s", args.out_dir)


if __name__ == "__main__":
    main()
