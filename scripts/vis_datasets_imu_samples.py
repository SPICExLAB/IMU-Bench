#!/usr/bin/env python3
"""vis_datasets_imu_samples.py

Plot ~10 raw IMU sample segments (~10 s each) for every dataset:
  • IMULM parquet: pamap2, recofit, samosa, shoaib, ut_watch, utd_mhad, wisdm, prism
  • OCDetect CSV  (FLU frame)
  • Nymeria real wrist pkl (FLU frame)

Output layout:
  data_vis/datasets_vis/<DatasetName>/sample_00.png
                                      sample_01.png
                                      ...

Usage
-----
    python scripts/vis_datasets_imu_samples.py
    python scripts/vis_datasets_imu_samples.py --n-samples 10 --seg-sec 10
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── constants ─────────────────────────────────────────────────────────────────
PARQUET_PATH  = "/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet"
OCDETECT_DIR  = "/mnt/nas2/yiquan_data/preprocessed"
NYMERIA_DIR   = "/mnt/nas2/processed_data_body_Ty_rwrist"
OUT_ROOT      = "data_vis/datasets_vis"

PARQUET_DATASETS = ["pamap2", "recofit", "samosa", "shoaib",
                    "ut_watch", "utd_mhad", "wisdm", "prism"]
PARQUET_IMU_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
OCDETECT_IMU_COLS = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]
HUMANML3D_DIR  = "/home/yiquan/HumanML3D/joints"
NYMERIA_SYN_DIR = "/mnt/nas2/processed_data_body_Ty_rwrist"

SAMPLE_HZ = 50   # all datasets resampled / stored at 50 Hz


# ── plotting helper ───────────────────────────────────────────────────────────

def _plot_sample(imu: np.ndarray, ch_names: list[str], title: str,
                 out_path: str, hz: float = SAMPLE_HZ) -> None:
    """imu: [T, 6]  float32."""
    T  = imu.shape[0]
    t  = np.arange(T) / hz

    acc_color  = "steelblue"
    gyro_color = "tomato"

    fig, axes = plt.subplots(6, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=11, fontweight="bold")

    for c, ax in enumerate(axes):
        color = acc_color if c < 3 else gyro_color
        ax.plot(t, imu[:, c], color=color, lw=0.9)
        ax.set_ylabel(ch_names[c], fontsize=7, rotation=0,
                      labelpad=60, va="center")
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", lw=0.3, alpha=0.5)

    axes[-1].set_xlabel("Time (s)", fontsize=9)

    # channel stats in title per axis
    for c, ax in enumerate(axes):
        mu  = imu[:, c].mean()
        std = imu[:, c].std()
        ax.set_title(f"μ={mu:.3f}  σ={std:.3f}", fontsize=6,
                     loc="right", pad=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── channel statistics ────────────────────────────────────────────────────────

def _plot_channel_stats(segments: list, ch_names: list[str],
                        title: str, out_path: str) -> None:
    """Bar chart of per-channel mean ± std computed across all segments.

    segments: list of [T, 6] arrays.
    Two panels: accelerometer (ch 0-2) and gyroscope (ch 3-5).
    """
    if not segments:
        return

    all_data = np.concatenate(segments, axis=0)   # [Total_T, 6]
    means = all_data.mean(axis=0)                  # [6]
    stds  = all_data.std(axis=0)                   # [6]

    fig, (ax_acc, ax_gyro) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Per-channel mean ± std  |  {title}  |  "
                 f"{len(segments)} segments  ({len(all_data)} samples total)",
                 fontsize=11, fontweight="bold")

    for ax, ch_range, colors, group_label, unit in [
        (ax_acc,  range(0, 3), ACC_COLORS,  "Accelerometer", "m/s²"),
        (ax_gyro, range(3, 6), GYRO_COLORS, "Gyroscope",     "rad/s"),
    ]:
        idxs   = list(ch_range)
        labels = [ch_names[c] for c in idxs]
        mu     = means[idxs]
        sigma  = stds[idxs]

        x = np.arange(len(idxs))
        bars = ax.bar(x, mu, yerr=sigma, color=colors, alpha=0.75,
                      capsize=6, error_kw={"elinewidth": 1.5, "ecolor": "black"})
        ax.axhline(0, color="grey", lw=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(f"Mean ({unit})", fontsize=9)
        ax.set_title(group_label, fontsize=10, loc="left")
        ax.grid(axis="y", lw=0.3, alpha=0.5)

        # annotate bars with mean ± std values
        for bar, m, s in zip(bars, mu, sigma):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.02 * (abs(mu).max() + abs(sigma).max() + 1e-6),
                    f"{m:.3f}\n±{s:.3f}",
                    ha="center", va="bottom", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  channel_stats.png saved")


# ── frequency distribution ───────────────────────────────────────────────────

ACC_COLORS  = ["#1f77b4", "#2ca02c", "#d62728"]   # F/L/U or x/y/z
GYRO_COLORS = ["#ff7f0e", "#9467bd", "#8c564b"]


def _plot_freq_distribution(segments: list, ch_names: list[str],
                             title: str, out_path: str, hz: float) -> None:
    """Welch PSD averaged over all segments. segments: list of [T, 6] arrays."""
    if not segments:
        return

    nperseg = min(256, min(s.shape[0] for s in segments))
    all_psds = [[] for _ in range(6)]
    freqs = None

    for seg in segments:
        for c in range(6):
            f, psd = welch(seg[:, c].astype(np.float64), fs=hz,
                           nperseg=nperseg, noverlap=nperseg // 2)
            if freqs is None:
                freqs = f
            all_psds[c].append(psd)

    if freqs is None:
        return

    fig, (ax_acc, ax_gyro) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"Welch PSD  |  {title}  |  {len(segments)} segments  @ {hz:.0f} Hz",
                 fontsize=11, fontweight="bold")

    for ax, ch_range, colors, group_label in [
        (ax_acc,  range(0, 3), ACC_COLORS,  "Accelerometer"),
        (ax_gyro, range(3, 6), GYRO_COLORS, "Gyroscope"),
    ]:
        for c, color in zip(ch_range, colors):
            arr     = np.array(all_psds[c])          # [N_segs, F]
            mean_db = 10 * np.log10(arr.mean(axis=0) + 1e-12)
            std_db  = 10 * np.log10(arr.std(axis=0)  + 1e-12)
            ax.plot(freqs, mean_db, color=color, lw=1.5, label=ch_names[c])
            ax.fill_between(freqs, mean_db - std_db, mean_db + std_db,
                            color=color, alpha=0.15)
        ax.set_ylabel("PSD (dB re 1 unit²/Hz)", fontsize=9)
        ax.set_title(group_label, fontsize=9, loc="left")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(lw=0.3, alpha=0.5)
        ax.axvline(hz / 2, color="grey", lw=0.8, linestyle="--",
                   label=f"Nyquist ({hz/2:.0f} Hz)")

    ax_gyro.set_xlabel("Frequency (Hz)", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  freq_distribution.png saved")


# ── OCDetect ──────────────────────────────────────────────────────────────────

def vis_ocdetect(n_samples: int, seg_sec: float) -> None:
    seg_len = int(seg_sec * SAMPLE_HZ)
    files   = sorted(glob.glob(os.path.join(OCDETECT_DIR, "OCDetect_*.csv")))
    files   = files[:n_samples]   # first n files, one sample each
    out_dir = os.path.join(OUT_ROOT, "OCDetect")
    ch_names = ["acc_F (m/s²)", "acc_L (m/s²)", "acc_U (m/s²)",
                "gyro_F (rad/s)", "gyro_L (rad/s)", "gyro_U (rad/s)"]

    print(f"[OCDetect] {len(files)} files → {out_dir}")
    segments = []
    for i, path in enumerate(files):
        df  = pd.read_csv(path, usecols=["ignore"] + OCDETECT_IMU_COLS,
                          low_memory=False)
        imu = df[OCDETECT_IMU_COLS].values.astype(np.float32)
        ign = df["ignore"].values.astype(np.int32)

        # Android → FLU: F=+Y, L=-X, U=+Z
        acc_x, acc_y   = imu[:, 0].copy(), imu[:, 1].copy()
        gyro_x, gyro_y = imu[:, 3].copy(), imu[:, 4].copy()
        imu[:, 0] =  acc_y;  imu[:, 1] = -acc_x    # acc_F=+Y, acc_L=-X
        imu[:, 3] =  gyro_y; imu[:, 4] = -gyro_x   # gyro_F=+Y, gyro_L=-X

        # pick first contiguous usable segment ≥ seg_len
        usable  = (ign == 0)
        changes = np.diff(usable.astype(np.int8), prepend=0, append=0)
        starts  = np.where(changes ==  1)[0]
        ends    = np.where(changes == -1)[0]

        seg = None
        for s, e in zip(starts, ends):
            if (e - s) >= seg_len:
                seg = imu[s : s + seg_len]
                break
        if seg is None:
            print(f"  [skip] {os.path.basename(path)} — no long-enough segment")
            continue

        fname = os.path.splitext(os.path.basename(path))[0]
        title = f"OCDetect | {fname} | {seg_sec:.0f}s"
        _plot_sample(seg, ch_names, title,
                     os.path.join(out_dir, f"sample_{i:02d}.png"))
        segments.append(seg)
        print(f"  sample_{i:02d}  {os.path.basename(path)}")

    _plot_freq_distribution(segments, ch_names, "OCDetect (FLU)",
                            os.path.join(out_dir, "freq_distribution.png"),
                            hz=float(SAMPLE_HZ))
    _plot_channel_stats(segments, ch_names, "OCDetect (FLU)",
                        os.path.join(out_dir, "channel_stats.png"))


# ── Nymeria real wrist ────────────────────────────────────────────────────────

_R_WRIST_FROM_FLU = np.array(
    [[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]], dtype=np.float64)


def _load_nymeria_imu_flu(pkl_path: str) -> np.ndarray | None:
    """Return [N, 6] float32 in FLU frame, or None on failure."""
    try:
        with open(pkl_path, "rb") as fh:
            raw = pickle.load(fh)
    except Exception as e:
        print(f"  [skip] {os.path.basename(pkl_path)}: {e}")
        return None

    if not raw.get("has_wrist_imu", True):
        return None

    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    wrist_imu = raw.get("wrist_imu_data", {})
    wrist_gt  = raw.get("wrist_gt_data",  {})
    if "accel" not in wrist_imu or "gyro" not in wrist_imu:
        return None

    fps_body  = float(raw.get("imu_sampling_rate_hz") or
                      raw.get("pose_sampling_rate_hz") or 50)
    fps_wrist = float(raw.get("wrist_imu_sampling_rate_hz", fps_body))
    ds = max(1, round(fps_wrist / SAMPLE_HZ))

    acc_l  = _np(wrist_imu["accel"]).astype(np.float64)[::ds]
    gyro_l = _np(wrist_imu["gyro"]).astype(np.float64)[::ds]
    orient = wrist_gt.get("orientation")
    if orient is None:
        return None

    try:
        import pypose as pp
        if isinstance(orient, pp.LieTensor):
            R_world = orient.matrix().detach().cpu().numpy().astype(np.float64)[::ds]
        else:
            raise TypeError
    except (ImportError, TypeError):
        from scipy.spatial.transform import Rotation as ScipyRot
        R_world = ScipyRot.from_quat(_np(orient).astype(np.float64)[::ds]).as_matrix()

    N = min(acc_l.shape[0], gyro_l.shape[0], R_world.shape[0])
    acc_l, gyro_l, R_world = acc_l[:N], gyro_l[:N], R_world[:N]

    acc_w  = np.einsum("nij,nj->ni", R_world, acc_l)
    gyro_w = np.einsum("nij,nj->ni", R_world, gyro_l)

    R_wfx = np.einsum("nij,jk->nik", R_world, _R_WRIST_FROM_FLU)
    R_xfw = R_wfx.transpose(0, 2, 1)
    imu_flu = np.concatenate([
        np.einsum("nij,nj->ni", R_xfw, acc_w),
        np.einsum("nij,nj->ni", R_xfw, gyro_w),
    ], axis=1).astype(np.float32)
    return imu_flu


def vis_nymeria(n_samples: int, seg_sec: float) -> None:
    seg_len  = int(seg_sec * SAMPLE_HZ)
    pkl_files = sorted(glob.glob(os.path.join(NYMERIA_DIR, "*.pkl")))[:n_samples * 2]
    out_dir   = os.path.join(OUT_ROOT, "Nymeria_real")
    ch_names  = ["acc_F (m/s²)", "acc_L (m/s²)", "acc_U (m/s²)",
                 "gyro_F (rad/s)", "gyro_L (rad/s)", "gyro_U (rad/s)"]

    print(f"[Nymeria] scanning {len(pkl_files)} pkl files → {out_dir}")
    saved = 0
    segments = []
    for path in pkl_files:
        if saved >= n_samples:
            break
        imu = _load_nymeria_imu_flu(path)
        if imu is None or len(imu) < seg_len:
            continue

        # pick middle segment to avoid edge artefacts
        mid   = len(imu) // 2
        start = max(0, mid - seg_len // 2)
        seg   = imu[start : start + seg_len]

        fname = os.path.splitext(os.path.basename(path))[0]
        title = f"Nymeria real wrist (FLU) | {fname[:40]} | {seg_sec:.0f}s"
        _plot_sample(seg, ch_names, title,
                     os.path.join(out_dir, f"sample_{saved:02d}.png"))
        segments.append(seg)
        print(f"  sample_{saved:02d}  {os.path.basename(path)}")
        saved += 1

    _plot_freq_distribution(segments, ch_names, "Nymeria real wrist (FLU)",
                            os.path.join(out_dir, "freq_distribution.png"),
                            hz=float(SAMPLE_HZ))
    _plot_channel_stats(segments, ch_names, "Nymeria real wrist (FLU)",
                        os.path.join(out_dir, "channel_stats.png"))


# ── Nymeria synthesized wrist IMU ────────────────────────────────────────────

def vis_nymeria_syn(n_samples: int, seg_sec: float) -> None:
    """Synthesize right-wrist FLU IMU from Nymeria xsens_pose body motion."""
    from imu_lm.data.nymeria_syn_wrist_loader_FLU import (
        _load_nymeria_syn_wrist_session_flu,
    )

    seg_len  = int(seg_sec * SAMPLE_HZ)
    out_dir  = os.path.join(OUT_ROOT, "Nymeria_syn")
    ch_names = ["acc_F (m/s²)", "acc_L (m/s²)", "acc_U (m/s²)",
                "gyro_F (rad/s)", "gyro_L (rad/s)", "gyro_U (rad/s)"]

    pkl_files = sorted(glob.glob(os.path.join(NYMERIA_SYN_DIR, "*.pkl")))
    if not pkl_files:
        print(f"[Nymeria_syn] no pkl files found under {NYMERIA_SYN_DIR}")
        return

    rng = np.random.default_rng(42)
    rng.shuffle(pkl_files)

    print(f"[Nymeria_syn] {len(pkl_files)} pkl files found → {out_dir}")
    saved    = 0
    segments = []

    for path in pkl_files:
        if saved >= n_samples:
            break

        imu = _load_nymeria_syn_wrist_session_flu(path, target_hz=SAMPLE_HZ,
                                                  add_gravity=True)
        if imu is None or len(imu) < seg_len:
            continue

        # pick middle segment
        mid   = len(imu) // 2
        start = max(0, mid - seg_len // 2)
        seg   = imu[start : start + seg_len]   # [seg_len, 6]

        fname = os.path.splitext(os.path.basename(path))[0]
        title = f"Nymeria syn (FLU) | {fname[:50]} | {seg_sec:.0f}s"
        _plot_sample(seg, ch_names, title,
                     os.path.join(out_dir, f"sample_{saved:02d}.png"),
                     hz=float(SAMPLE_HZ))
        segments.append(seg)
        print(f"  sample_{saved:02d}  {os.path.basename(path)}")
        saved += 1

    if saved == 0:
        print(f"  [warn] no valid clips for Nymeria_syn")
        return

    _plot_freq_distribution(segments, ch_names, "Nymeria syn wrist (FLU)",
                            os.path.join(out_dir, "freq_distribution.png"),
                            hz=float(SAMPLE_HZ))
    _plot_channel_stats(segments, ch_names, "Nymeria syn wrist (FLU)",
                        os.path.join(out_dir, "channel_stats.png"))


# ── HumanML3D synthetic wrist IMU ────────────────────────────────────────────

def _joints_to_imu_flu_geom(joints: np.ndarray, fps: float) -> np.ndarray:
    """Derive right-wrist FLU IMU from joint positions using geometric arm-chain frame.

    Mirrors nymeria_syn_wrist_loader_FLU.py but uses geometric R_world_from_wrist
    (from shoulder/elbow/wrist positions) instead of SMPL FK, so no AMASS needed.

    World frame: Y-up (HumanML3D canonical).  Gravity bias: +9.81 on Y.

    Returns [N, 6] float32: (acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U).
    """
    from scipy.spatial.transform import Rotation as ScipyRot
    from imu_lm.data.humanml3d_dataset_FLU import (
        _build_wrist_rotation, R_WRIST_FROM_FLU,
        _R_SHOULDER, _R_ELBOW, _R_WRIST,
    )

    N = joints.shape[0]
    joints_d = joints.astype(np.float64)

    # Geometric wrist rotation from arm-chain: R_world_from_wrist [N, 3, 3]
    R_wrist_world = _build_wrist_rotation(
        joints_d[:, _R_SHOULDER],
        joints_d[:, _R_ELBOW],
        joints_d[:, _R_WRIST],
    )

    # FLU frame expressed in world: R_world_from_FLU = R_wrist_world @ R_WRIST_FROM_FLU
    R_world_from_FLU = np.einsum("nij,jk->nik", R_wrist_world, R_WRIST_FROM_FLU)
    R_FLU_from_world = R_world_from_FLU.transpose(0, 2, 1)

    # World-frame linear acceleration (central finite diff) + gravity bias
    wrist = joints_d[:, _R_WRIST]
    a_world = np.zeros((N, 3), dtype=np.float64)
    a_world[1:-1] = (wrist[2:] - 2.0 * wrist[1:-1] + wrist[:-2]) * fps ** 2
    a_world[0] = a_world[1];  a_world[-1] = a_world[-2]
    a_world[:, 1] += 9.81   # Y-up world: static IMU reads +g on Y

    # Angular velocity in wrist local frame → world frame
    dR = np.einsum("nij,njk->nik",
                   R_wrist_world[:-1].transpose(0, 2, 1), R_wrist_world[1:])
    omega_local = np.zeros((N, 3), dtype=np.float64)
    omega_local[:-1] = (ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
                        .as_rotvec().astype(np.float64) * fps)
    omega_local[-1] = omega_local[-2]
    omega_world = np.einsum("nij,nj->ni", R_wrist_world, omega_local)

    # Project both into FLU
    acc_flu  = np.einsum("nij,nj->ni", R_FLU_from_world, a_world).astype(np.float32)
    gyro_flu = np.einsum("nij,nj->ni", R_FLU_from_world, omega_world).astype(np.float32)

    return np.concatenate([acc_flu, gyro_flu], axis=1)   # [N, 6]


def vis_humanml3d(n_samples: int, seg_sec: float) -> None:
    """Synthesize right-wrist FLU IMU from HumanML3D joint positions.

    Uses geometric arm-chain wrist frame (no AMASS required) + gravity bias,
    matching the convention of Nymeria syn/real wrist loaders.
    """
    from imu_lm.data.humanml3d_dataset import _resample_joints

    seg_len  = int(seg_sec * SAMPLE_HZ)
    npy_files = sorted(glob.glob(os.path.join(HUMANML3D_DIR, "*.npy")))
    if not npy_files:
        print(f"[HumanML3D] no .npy files found in {HUMANML3D_DIR}")
        return

    rng = np.random.default_rng(42)
    rng.shuffle(npy_files)

    out_dir  = os.path.join(OUT_ROOT, "HumanML3D")
    ch_names = ["acc_F (m/s²)", "acc_L (m/s²)", "acc_U (m/s²)",
                "gyro_F (rad/s)", "gyro_L (rad/s)", "gyro_U (rad/s)"]

    print(f"[HumanML3D] {len(npy_files)} clips found → {out_dir}")
    saved    = 0
    segments = []

    for path in npy_files:
        if saved >= n_samples:
            break

        try:
            raw = np.load(path, allow_pickle=True).astype(np.float32)  # [T_20hz, 52, 3]
        except Exception as e:
            print(f"  [skip] {os.path.basename(path)}: {e}")
            continue

        joints_20hz = raw[:, :22, :]               # [T_20hz, 22, 3]
        if joints_20hz.shape[0] < 10:
            continue

        joints_50hz = _resample_joints(joints_20hz) # [N, 22, 3]
        if joints_50hz.shape[0] < seg_len:
            continue

        imu = _joints_to_imu_flu_geom(joints_50hz, fps=float(SAMPLE_HZ))  # [N, 6]

        # pick middle segment
        mid   = len(imu) // 2
        start = max(0, mid - seg_len // 2)
        seg   = imu[start : start + seg_len]        # [seg_len, 6]

        clip_id = os.path.splitext(os.path.basename(path))[0]
        title   = f"HumanML3D | {clip_id} | {seg_sec:.0f}s @ {SAMPLE_HZ:.0f}Hz"
        _plot_sample(seg, ch_names, title,
                     os.path.join(out_dir, f"sample_{saved:02d}.png"),
                     hz=float(SAMPLE_HZ))
        segments.append(seg)
        print(f"  sample_{saved:02d}  {clip_id}  shape={joints_50hz.shape}")
        saved += 1

    if saved == 0:
        print(f"  [warn] no valid clips found in {HUMANML3D_DIR}")
        return

    _plot_freq_distribution(segments, ch_names, f"HumanML3D synthetic wrist @ {SAMPLE_HZ:.0f}Hz",
                            os.path.join(out_dir, "freq_distribution.png"),
                            hz=float(SAMPLE_HZ))
    _plot_channel_stats(segments, ch_names, "HumanML3D synthetic wrist",
                        os.path.join(out_dir, "channel_stats.png"))


# ── IMULM parquet ─────────────────────────────────────────────────────────────

def _infer_hz(sess_df: pd.DataFrame) -> float:
    """Infer sampling rate from timestamp_ns; fall back to SAMPLE_HZ if unclear."""
    ts = sess_df["timestamp_ns"].values
    diffs = np.diff(ts.astype(np.float64))
    nonzero = diffs[diffs > 0]
    if len(nonzero) < 2:
        return float(SAMPLE_HZ)
    median_ns = float(np.median(nonzero))
    return round(1e9 / median_ns)


def vis_parquet_dataset(dataset_name: str, n_samples: int, seg_sec: float) -> None:
    out_dir  = os.path.join(OUT_ROOT, dataset_name)
    ch_names = [c.replace("_", " ") for c in PARQUET_IMU_COLS]

    print(f"[{dataset_name}] loading from parquet …")
    try:
        df = pd.read_parquet(
            PARQUET_PATH,
            columns=["session_id", "timestamp_ns"] + PARQUET_IMU_COLS,
            filters=[("dataset", "==", dataset_name)],
        )
    except Exception as e:
        print(f"  [error] {e}")
        return

    if df.empty:
        print(f"  [skip] dataset '{dataset_name}' not found in parquet")
        return

    sessions = df["session_id"].unique()
    np.random.seed(42)
    np.random.shuffle(sessions)
    sessions = sessions[:n_samples * 2]   # over-sample in case some are short

    print(f"  {len(sessions)} sessions found → {out_dir}")
    saved    = 0
    segments = []
    ds_hz    = float(SAMPLE_HZ)   # updated on first successful session
    for sess in sessions:
        if saved >= n_samples:
            break
        sess_df = (df[df["session_id"] == sess]
                   .sort_values("timestamp_ns")
                   .reset_index(drop=True))

        hz      = _infer_hz(sess_df)
        seg_len = int(seg_sec * hz)
        imu     = sess_df[PARQUET_IMU_COLS].values.astype(np.float32)
        if len(imu) < seg_len:
            continue

        # pick middle segment
        mid   = len(imu) // 2
        start = max(0, mid - seg_len // 2)
        seg   = imu[start : start + seg_len]

        title = f"{dataset_name} | {str(sess)[:40]} | {seg_sec:.0f}s @ {hz:.0f}Hz"
        _plot_sample(seg, ch_names, title,
                     os.path.join(out_dir, f"sample_{saved:02d}.png"),
                     hz=hz)
        # resample to common hz for freq plot if needed
        segments.append(seg)
        ds_hz = hz
        print(f"  sample_{saved:02d}  sess={str(sess)[:30]}  hz={hz:.0f}")
        saved += 1

    if saved == 0:
        print(f"  [warn] no samples saved for {dataset_name}")
        return

    _plot_freq_distribution(segments, ch_names, f"{dataset_name} @ {ds_hz:.0f}Hz",
                            os.path.join(out_dir, "freq_distribution.png"),
                            hz=ds_hz)
    _plot_channel_stats(segments, ch_names, f"{dataset_name} @ {ds_hz:.0f}Hz",
                        os.path.join(out_dir, "channel_stats.png"))


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=10,
                   help="Number of sample plots per dataset (default 10)")
    p.add_argument("--seg-sec",   type=float, default=10.0,
                   help="Segment length in seconds (default 10)")
    p.add_argument("--datasets",  nargs="+", default=None,
                   help="Subset of datasets to run (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    n    = args.n_samples
    sec  = args.seg_sec

    all_datasets = PARQUET_DATASETS  # parquet ones
    if args.datasets:
        all_datasets = [d for d in all_datasets if d in args.datasets]
        do_ocdetect    = "OCDetect"    in args.datasets or "ocdetect"    in args.datasets
        do_nymeria     = "Nymeria"     in args.datasets or "nymeria"     in args.datasets
        do_nymeria_syn = "Nymeria_syn" in args.datasets or "nymeria_syn" in args.datasets
        do_humanml3d   = "HumanML3D"  in args.datasets or "humanml3d"   in args.datasets
    else:
        do_ocdetect    = True
        do_nymeria     = True
        do_nymeria_syn = True
        do_humanml3d   = True

    # ── OCDetect ──
    if do_ocdetect:
        vis_ocdetect(n, sec)

    # ── Nymeria real ──
    if do_nymeria:
        vis_nymeria(n, sec)

    # ── Nymeria synthesized ──
    if do_nymeria_syn:
        vis_nymeria_syn(n, sec)

    # ── HumanML3D synthetic wrist ──
    if do_humanml3d:
        vis_humanml3d(n, sec)

    # ── IMULM parquet datasets ──
    for ds_name in all_datasets:
        vis_parquet_dataset(ds_name, n, sec)

    print(f"\nDone. Plots saved under {OUT_ROOT}/")


if __name__ == "__main__":
    main()
