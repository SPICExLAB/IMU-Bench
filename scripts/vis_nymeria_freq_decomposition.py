"""Visualize frequency decomposition of one Nymeria real wrist IMU sequence.

Loads the first available pkl from processed_data_body_Ty_rwrist, transforms
the wrist IMU to the FLU body frame, then decomposes the signal into four
frequency bands using Butterworth low-pass filters:
  Band 1: 0–2 Hz
  Band 2: 2–5 Hz  (lp5 − lp2)
  Band 3: 5–12 Hz (lp12 − lp5)
  Band 4: 12–25 Hz residual (original − lp12)

Saves to /home/yiquan/TransfHAR/data_vis/nymeria_freq_decomposition.png
"""

from __future__ import annotations

import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PKL_ROOT   = "/mnt/nas2/processed_data_body_Ty_rwrist"
OUT_DIR    = "/home/yiquan/TransfHAR/data_vis"
OUT_FILE   = os.path.join(OUT_DIR, "nymeria_freq_decomposition.png")
FS         = 50.0          # Hz
PLOT_SEC   = 10.0          # seconds to show
CUTOFFS_HZ = [2.0, 5.0, 12.0]
CH_NAMES   = ["acc_F", "acc_L", "acc_U", "gyro_F", "gyro_L", "gyro_U"]
ADD_GRAVITY = True

R_WRIST_FROM_FLU = np.array(
    [[-1.0, 0.0, 0.0],
     [ 0.0, 0.0, 1.0],
     [ 0.0, 1.0, 0.0]],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def butter_lowpass(signal: np.ndarray, cutoff_hz: float, fs: float) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter along axis 0. signal: [T, C]."""
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        return signal.copy()
    b, a = butter(N=4, Wn=cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, signal, axis=0)


def _quat_xyzw_to_rotmat(quat: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(quat).as_matrix().astype(np.float64)


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_wrist_flu(pkl_path: str, add_gravity: bool = True) -> np.ndarray:
    """Load wrist IMU from pkl and transform to FLU body frame.

    Returns
    -------
    X : [T, 6]  float32  [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
    """
    with open(pkl_path, "rb") as fh:
        raw = pickle.load(fh)

    assert raw.get("has_wrist_imu", False), "no wrist IMU in this file"

    wrist_imu = raw["wrist_imu_data"]
    wrist_gt  = raw["wrist_gt_data"]

    wrist_hz = float(raw.get("wrist_imu_sampling_rate_hz", FS))
    ds = max(1, round(wrist_hz / FS))

    acc_local  = _to_numpy(wrist_imu["accel"]).astype(np.float64)[::ds]
    gyro_local = _to_numpy(wrist_imu["gyro"]).astype(np.float64)[::ds]

    orient_raw = wrist_gt["orientation"]
    try:
        import pypose as pp
        if isinstance(orient_raw, pp.LieTensor):
            R_wfcpf = orient_raw.matrix().detach().cpu().numpy().astype(np.float64)[::ds]
        else:
            R_wfcpf = _quat_xyzw_to_rotmat(_to_numpy(orient_raw).astype(np.float64)[::ds])
    except ImportError:
        R_wfcpf = _quat_xyzw_to_rotmat(_to_numpy(orient_raw).astype(np.float64)[::ds])

    N = min(acc_local.shape[0], R_wfcpf.shape[0])
    acc_local  = acc_local[:N]
    gyro_local = gyro_local[:N]
    R_wfcpf    = R_wfcpf[:N]

    # CPF → world
    acc_world  = np.einsum("nij,nj->ni", R_wfcpf, acc_local)
    gyro_world = np.einsum("nij,nj->ni", R_wfcpf, gyro_local)

    if not add_gravity:
        acc_world += np.array([0.0, -9.81, 0.0])

    # world → FLU
    R_wfX  = np.einsum("nij,jk->nik", R_wfcpf, R_WRIST_FROM_FLU)
    R_Xfw  = R_wfX.transpose(0, 2, 1)
    acc_flu  = np.einsum("nij,nj->ni", R_Xfw, acc_world)
    gyro_flu = np.einsum("nij,nj->ni", R_Xfw, gyro_world)

    return np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Pick first valid pkl
    pkl_files = sorted(f for f in os.listdir(PKL_ROOT) if f.endswith(".pkl"))
    assert pkl_files, f"No pkl files in {PKL_ROOT}"
    pkl_path = os.path.join(PKL_ROOT, pkl_files[0])
    print(f"Loading: {pkl_files[0]}")

    X = load_wrist_flu(pkl_path, add_gravity=ADD_GRAVITY)  # [T, 6]
    T_plot = int(PLOT_SEC * FS)
    X = X[:T_plot]
    t = np.arange(len(X)) / FS

    print(f"Signal shape after crop: {X.shape}  ({len(X)/FS:.1f}s)")

    # Frequency decomposition
    lp = [butter_lowpass(X, c, FS) for c in CUTOFFS_HZ]
    # Band-isolated components
    bands = [
        lp[0],                # 0–2 Hz
        lp[1] - lp[0],        # 2–5 Hz
        lp[2] - lp[1],        # 5–12 Hz
        X     - lp[2],        # 12–25 Hz residual
    ]
    band_labels = ["0–2 Hz", "2–5 Hz", "5–12 Hz", "12–25 Hz"]
    band_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    n_rows = 5  # original + 4 bands
    n_cols = len(CH_NAMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.2 * n_rows),
                             sharex=True, constrained_layout=True)

    row_labels = ["Original"] + band_labels
    row_colors = ["black"]   + band_colors

    for row in range(n_rows):
        sig = X if row == 0 else bands[row - 1]
        for col, ch in enumerate(CH_NAMES):
            ax = axes[row, col]
            ax.plot(t, sig[:, col], color=row_colors[row], linewidth=0.8)
            ax.set_xlim(0, PLOT_SEC)
            if row == 0:
                ax.set_title(ch, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, linewidth=0.3, alpha=0.5)

    fig.suptitle(
        f"Nymeria wrist IMU – frequency decomposition (FLU frame)\n{pkl_files[0]}",
        fontsize=11
    )
    fig.savefig(OUT_FILE, dpi=150)
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    main()
