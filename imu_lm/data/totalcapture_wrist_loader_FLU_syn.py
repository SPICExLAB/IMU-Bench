"""totalcapture_wrist_loader_FLU_syn.py
Synthetic right-wrist IMU for TotalCapture derived purely from GT SMPL poses.

Unlike ``totalcapture_wrist_loader_FLU.py``, which uses the DIP-precomputed ``acc``
and ``ori`` fields, this loader synthesises acc and gyro via SMPL FK:

  acc  : 2nd central finite difference of the wrist joint world position
  gyro : angular velocity derived from consecutive FK wrist rotations

Both are then projected into the FLU body frame, matching the convention of all
other ``*_FLU`` loaders.

Pipeline
--------
1.  Load ``gt`` [N, 72] SMPL axis-angle from a DIP-format TotalCapture pkl.
2.  Convert to local rotation matrices [N, 24, 3, 3] and run SMPL FK to obtain:
      R_world_from_wrist(t)  [N, 3, 3] – wrist world-frame rotation
      p_world_wrist(t)       [N, 3]    – wrist world position
    If ``official_dir`` is given, the Spine (pelvis) translation from the official
    ``gt_skel_gbl_pos.txt`` is used as root translation for FK; otherwise the pelvis
    is held at the origin (captures only arm-relative acceleration).
3.  Synthesise world-frame signals:
      a_world = d²p_world_wrist / dt²              (2nd central finite difference)
      ω_local = rotvec(R_world(t)^T @ R_world(t+1)) * fps   (wrist-local frame)
      ω_world = R_world_from_wrist @ ω_local
4.  Optional: add MEMS-style two-component noise to the gyro signal.
5.  Build the FLU frame:
      R_world_from_X(t) = R_world_from_wrist(t) @ R_WRIST_FROM_FLU
      R_X_from_world(t) = R_world_from_X(t)^T
6.  Express in FLU:
      acc_flu  = R_X_from_world @ a_world
      gyro_flu = R_X_from_world @ ω_world

FLU body frame (right wrist, T-pose)
--------------------------------------
  Forward (F, x) = −x_smpl   (finger direction)
  Left    (L, y) = +z_smpl   (thumb side)
  Up      (U, z) = +y_smpl   (dorsal / watch-face)

Outputs per window
------------------
  x : [6, T]           float32 – (acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U)
  y : [N_patches, J*6] float32 – 6D local rotations for right-arm chain
                                   J=5  → SMPL joints [14, 17, 19, 21, 23]
                                   J=24 → all SMPL joints
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRot
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.augmentations.transform import apply_augment
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SMPL right-arm chain
RIGHT_WRIST_CHAIN_JOINTS = [14, 17, 19, 21, 23]
NUM_CHAIN_JOINTS         = len(RIGHT_WRIST_CHAIN_JOINTS)
ROT6D_DIM                = 6

# SMPL joint indices
_SMPL_RIGHT_WRIST = 21

# World-frame gravity (Y-up, SMPL / TotalCapture convention)
_G_WORLD = np.array([0.0, -9.81, 0.0], dtype=np.float64)

# Official TotalCapture GT files
_INCHES_TO_METERS     = 0.0254
_OFFICIAL_GT_FILENAME = "gt_skel_gbl_pos.txt"

# ---------------------------------------------------------------------------
# FLU frame definition  (identical to all other *_FLU loaders)
# ---------------------------------------------------------------------------
R_WRIST_FROM_FLU = np.array(
    [
        [-1.0, 0.0, 0.0],   # F = −x_smpl
        [ 0.0, 0.0, 1.0],   # L = +z_smpl
        [ 0.0, 1.0, 0.0],   # U = +y_smpl
    ],
    dtype=np.float64,
)
assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] → [..., 6]  (first two columns, row-major)."""
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def _resample_linear(x: np.ndarray, new_len: int) -> np.ndarray:
    """Linear interpolation of [N, ...] → [new_len, ...]."""
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


def _resample_rotmat_nlerp(x: np.ndarray, new_len: int) -> np.ndarray:
    """Nearest-neighbour resample for rotation matrices [N, ..., 3, 3]."""
    old_len = x.shape[0]
    if old_len == new_len:
        return x
    idx = np.round(np.linspace(0, old_len - 1, new_len)).astype(np.int64)
    return x[idx]


def _compute_acc_world(pos_world: np.ndarray, fps: float) -> np.ndarray:
    """2nd central finite difference of wrist position → world-frame acc [N, 3]."""
    N   = pos_world.shape[0]
    acc = np.zeros((N, 3), dtype=np.float64)
    acc[1:-1] = (pos_world[2:] - 2.0 * pos_world[1:-1] + pos_world[:-2]) * (fps ** 2)
    acc[0]    = acc[1]
    acc[-1]   = acc[-2]
    return acc


def _compute_omega_local(R_world_from_joint: np.ndarray, fps: float) -> np.ndarray:
    """Angular velocity in the joint LOCAL frame from consecutive world rotations.

    dR_t = R_t^T @ R_{t+1},   omega_local(t) = rotvec(dR_t) * fps
    """
    dR = np.einsum(
        "nij,njk->nik",
        R_world_from_joint[:-1].transpose(0, 2, 1),
        R_world_from_joint[1:],
    )                          # [N-1, 3, 3]
    omega_raw = (
        ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
        .as_rotvec()
        .astype(np.float64) * fps
    )                          # [N-1, 3]
    N     = R_world_from_joint.shape[0]
    omega = np.zeros((N, 3), dtype=np.float64)
    omega[:-1] = omega_raw
    omega[-1]  = omega_raw[-1]
    return omega


def _add_mems_gyro_noise(
    gyro: np.ndarray,
    fps: float,
    white_noise_std: float = 0.02,
    bias_drift_std:  float = 0.002,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Two-component MEMS gyroscope noise model (IEEE Std 952).

    omega_measured(t) = omega_true(t)
                      + N(0, white_noise_std²)          [Angle Random Walk]
                      + integrated_random_walk(t)        [Bias Instability]

    Args:
        gyro:            [N, 3] clean angular velocity (rad/s)
        fps:             sampling rate (Hz)
        white_noise_std: white noise std (rad/s).  Default 0.02.
        bias_drift_std:  bias RW std per √second (rad/s/√s).  Default 0.002.
        rng:             optional RandomState for reproducibility.

    Returns:
        [N, 3] noisy gyro signal, same dtype as input.
    """
    if rng is None:
        rng = np.random.RandomState()
    N  = gyro.shape[0]
    dt = 1.0 / fps
    out = gyro.copy()
    out = out + rng.normal(0.0, white_noise_std, size=(N, 3))
    bias_steps = rng.normal(0.0, bias_drift_std * np.sqrt(dt), size=(N, 3))
    out = out + np.cumsum(bias_steps, axis=0)
    return out.astype(gyro.dtype)


def _load_gt_skel_spine(
    pkl_path: str,
    official_dir: str,
    target_hz: int = 50,
    native_hz: float = 60.0,
) -> Optional[np.ndarray]:
    """Load the Spine (root) world position from ``gt_skel_gbl_pos.txt``.

    Parses the pkl filename (e.g. ``s5_freestyle1.pkl``) to locate the
    corresponding official ground-truth file under *official_dir*
    (e.g. ``<official_dir>/S5/freestyle1/gt_skel_gbl_pos.txt``).

    Returns
    -------
    trans : [N, 3] float64 – absolute world position (Y-up, metres), resampled
            to *target_hz*.  ``None`` on any failure.
    """
    basename = os.path.splitext(os.path.basename(pkl_path))[0]
    parts    = basename.split("_", 1)
    if len(parts) != 2:
        logger.warning("[tc_syn_flu] cannot parse pkl filename: %s", basename)
        return None
    subject = parts[0].upper()   # s5 → S5
    motion  = parts[1]           # freestyle1

    gt_file = os.path.join(official_dir, subject, motion, _OFFICIAL_GT_FILENAME)
    if not os.path.isfile(gt_file):
        logger.warning("[tc_syn_flu] gt file not found: %s", gt_file)
        return None

    try:
        with open(gt_file, "r") as fh:
            header_cols = fh.readline().rstrip("\n\r").split("\t")
            try:
                spine_col = header_cols.index("Spine")
            except ValueError:
                logger.warning("[tc_syn_flu] 'Spine' column missing in %s", gt_file)
                return None
            rows: list = []
            for line in fh:
                line = line.rstrip("\n\r")
                if not line:
                    continue
                fields = line.split("\t")
                if spine_col >= len(fields):
                    continue
                xyz = [float(v) for v in fields[spine_col].split()]
                rows.append(xyz)
    except Exception as exc:
        logger.warning("[tc_syn_flu] error reading %s: %s", gt_file, exc)
        return None

    if not rows:
        logger.warning("[tc_syn_flu] empty GT file: %s", gt_file)
        return None

    trans = np.array(rows, dtype=np.float64) * _INCHES_TO_METERS
    trans[:, 0] *= -1   # negate X  (matches DIP preprocessing convention)
    trans[:, 2] *= -1   # negate Z

    N = trans.shape[0]
    if abs(native_hz - target_hz) > 1e-3:
        new_len = int(round((N - 1) * target_hz / native_hz)) + 1
        trans   = _resample_linear(trans, new_len)

    logger.debug("[tc_syn_flu] loaded GT trans %s: %d frames", basename, trans.shape[0])
    return trans


# ---------------------------------------------------------------------------
# Core session loader
# ---------------------------------------------------------------------------

def load_session_syn_flu(
    pkl_path: str,
    target_hz: int = 60,
    add_gravity: bool = True,
    official_dir: Optional[str] = None,
    gyro_white_noise_std: float = 0.0,
    gyro_bias_drift_std:  float = 0.0,
    rng: Optional[np.random.RandomState] = None,
    num_joints: int = 5,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load one TotalCapture DIP pkl and synthesise **(X, y)** in the FLU frame.

    Acceleration and gyro are derived entirely from GT SMPL FK, not from the
    DIP-precomputed ``acc`` / ``ori`` fields.

    Parameters
    ----------
    pkl_path    : path to a DIP-format TotalCapture ``.pkl`` file.
    target_hz   : target sampling rate; data is resampled from 60 Hz if needed.
    add_gravity : True  → subtract _G_WORLD from a_world so the signal matches
                          a real MEMS accelerometer reading (specific force).
                  False → keep the pure kinematic acceleration (gravity-free).
    official_dir : if given, root (Spine) translation is loaded from the official
                   ``gt_skel_gbl_pos.txt`` file and passed to SMPL FK.  This makes
                   the synthesised acc include body translational motion, matching
                   what a real accelerometer would read during locomotion.
                   If None, the pelvis is held at the origin (only arm-relative acc).
    gyro_white_noise_std : white noise std added to gyro (rad/s).  0 = disabled.
    gyro_bias_drift_std  : bias drift std (rad/s/√s).  0 = disabled.
    rng          : RandomState for reproducible noise; created internally if None.
    num_joints   : 5  → right-arm chain [14,17,19,21,23]  (30 dims per patch)
                   24 → all SMPL joints                    (144 dims per patch)

    Returns
    -------
    X : [N, 6]    float32 – ``[acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]``
    y : [N, J, 6] float32 – 6D local rotations; J = num_joints
    ``None`` on any failure.
    """
    import core.articulate as art
    from core.paths import Paths

    logger.info("[tc_syn_flu] loading: %s", os.path.basename(pkl_path))

    # ------------------------------------------------------------------
    # Load pickle
    # ------------------------------------------------------------------
    try:
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh, encoding="latin1")
    except Exception as exc:
        logger.warning("[tc_syn_flu] cannot open %s: %s", pkl_path, exc)
        return None

    gt_raw = data.get("gt")   # [N, 72] SMPL axis-angle

    if gt_raw is None:
        logger.warning("[tc_syn_flu] %s: missing 'gt' key.", pkl_path)
        return None

    gt_raw = np.asarray(gt_raw, dtype=np.float64)   # [N, 72]
    N      = gt_raw.shape[0]

    if N < 3:
        logger.warning("[tc_syn_flu] %s: too short (%d frames).", pkl_path, N)
        return None

    # ------------------------------------------------------------------
    # Resample to target_hz if needed  (raw = 60 Hz)
    # ------------------------------------------------------------------
    fps_raw = 60.0
    if abs(fps_raw - target_hz) > 1e-3:
        new_len = int(round((N - 1) * target_hz / fps_raw)) + 1
        gt_raw  = _resample_linear(gt_raw, new_len)
        N       = new_len
        logger.info(
            "[tc_syn_flu] resampled → %d frames (%.0f→%.0f Hz)", N, fps_raw, target_hz
        )

    fps = float(target_hz)

    # ------------------------------------------------------------------
    # Optional: load official root translation
    # ------------------------------------------------------------------
    root_trans_np: Optional[np.ndarray] = None
    if official_dir is not None:
        root_trans_np = _load_gt_skel_spine(
            pkl_path, official_dir, target_hz=target_hz, native_hz=fps_raw
        )
        if root_trans_np is not None:
            # Align lengths
            N = min(N, root_trans_np.shape[0])
            gt_raw        = gt_raw[:N]
            root_trans_np = root_trans_np[:N]

    # ------------------------------------------------------------------
    # Step 1 — GT pose → local rotation matrices [N, 24, 3, 3]
    # ------------------------------------------------------------------
    gt_aa    = gt_raw.reshape(N, 24, 3)
    R_local  = (
        ScipyRot.from_rotvec(gt_aa.reshape(-1, 3))
        .as_matrix()
        .reshape(N, 24, 3, 3)
        .astype(np.float32)
    )
    R_local_t = torch.from_numpy(R_local)   # [N, 24, 3, 3]

    # ------------------------------------------------------------------
    # Step 2 — SMPL FK → world-frame wrist rotation + position
    # ------------------------------------------------------------------
    body_model = art.model.ParametricModel(
        Paths.SMPL_FILE, device=torch.device("cpu")
    )

    root_trans_t: Optional[torch.Tensor] = None
    if root_trans_np is not None:
        root_trans_t = torch.from_numpy(root_trans_np.astype(np.float32))   # [N, 3]

    pose_global_t, joint_world_t = body_model.forward_kinematics_masked(
        R_local_t,
        tran=root_trans_t,
        calc_mesh=False,
    )   # [N, 24, 3, 3], [N, 24, 3]

    R_world_from_wrist = (
        pose_global_t[:, _SMPL_RIGHT_WRIST].cpu().numpy().astype(np.float64)
    )   # [N, 3, 3]
    p_world_wrist = (
        joint_world_t[:, _SMPL_RIGHT_WRIST].cpu().numpy().astype(np.float64)
    )   # [N, 3]

    # ------------------------------------------------------------------
    # Step 3 — synthesise world-frame acc and gyro
    # ------------------------------------------------------------------
    a_world = _compute_acc_world(p_world_wrist, fps)   # [N, 3]

    if add_gravity:
        # Subtract gravity to simulate real MEMS specific-force reading.
        # Y-up world: g_world = [0, -9.81, 0]
        # f_mems = a_kinematic - g_world  ≡  a_kinematic + [0, 9.81, 0]
        a_world = a_world - _G_WORLD

    omega_local = _compute_omega_local(R_world_from_wrist, fps)   # [N, 3]
    omega_world = np.einsum("nij,nj->ni", R_world_from_wrist, omega_local)   # [N, 3]

    # ------------------------------------------------------------------
    # Step 5 — build FLU frame per timestep
    #   R_world_from_X(t) = R_world_from_wrist(t) @ R_WRIST_FROM_FLU
    # ------------------------------------------------------------------
    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_wrist, R_WRIST_FROM_FLU)   # [N, 3, 3]
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)                                 # [N, 3, 3]

    # ------------------------------------------------------------------
    # Step 6 — project into FLU
    # ------------------------------------------------------------------
    acc_flu  = np.einsum("nij,nj->ni", R_X_from_world, a_world)      # [N, 3]
    gyro_flu = np.einsum("nij,nj->ni", R_X_from_world, omega_world)  # [N, 3]

    X = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)   # [N, 6]

    # ------------------------------------------------------------------
    # Labels — local 6D rotations for selected joints
    # ------------------------------------------------------------------
    label_R = R_local if num_joints == 24 else R_local[:, RIGHT_WRIST_CHAIN_JOINTS]
    y = matrix_to_rotation_6d(
        torch.from_numpy(label_R)
    ).numpy().astype(np.float32)   # [N, J, 6]

    logger.info(
        "[tc_syn_flu] %s: N=%d fps=%.0f syn-wrist-flu (root_trans=%s)",
        os.path.basename(pkl_path), N, fps,
        "official" if root_trans_np is not None else "none",
    )
    return X, y


# ---------------------------------------------------------------------------
# Window index builder
# ---------------------------------------------------------------------------

def _build_window_specs(
    session_lengths: List[int],
    T: int,
    hop: int,
) -> Tuple[np.ndarray, np.ndarray]:
    total    = sum(max(0, (n - T) // hop + 1) for n in session_lengths)
    sess_idx = np.empty(total, dtype=np.int32)
    starts   = np.empty(total, dtype=np.int32)
    offset   = 0
    for ki, n in enumerate(session_lengths):
        if n < T:
            continue
        n_win = (n - T) // hop + 1
        sess_idx[offset : offset + n_win] = ki
        starts[offset : offset + n_win]   = np.arange(0, n - T + 1, hop, dtype=np.int32)
        offset += n_win
    return sess_idx, starts


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TotalCaptureSynWristDatasetFLU(Dataset):
    """Sliding-window dataset backed by **synthetic** right-wrist IMU from GT SMPL FK.

    Input  X : [6, T]   float32 – [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
    Label  y : [P, J*6] float32 – per-patch 6D rotations; P = T // patch_size
    """

    def __init__(
        self,
        pkl_paths: List[str],
        cfg: Any,
        split_name: str = "train",
        target_hz: int = 60,
        add_gravity: bool = True,
        official_dir: Optional[str] = None,
        gyro_white_noise_std: float = 0.0,
        gyro_bias_drift_std:  float = 0.0,
        noise_seed: int = 0,
        num_joints: int = 5,
    ):
        super().__init__()
        self.cfg         = cfg
        self.split_name  = split_name
        self.target_hz   = target_hz
        self.add_gravity = add_gravity
        self.num_joints  = num_joints
        self.stats       = PreprocessStats()

        rng: Optional[np.random.RandomState] = (
            np.random.RandomState(noise_seed)
            if (gyro_white_noise_std > 0.0 or gyro_bias_drift_std > 0.0)
            else None
        )

        self._T, self._hop = compute_T_and_hop(cfg)
        vit_cfg            = cfg_get(cfg, ["vit1d"], {}) or {}
        self._patch_size   = int(vit_cfg.get("patch_size", 16))
        self._num_patches  = self._T // self._patch_size

        self._sessions: List[Tuple[np.ndarray, np.ndarray]] = []
        session_lengths: List[int] = []

        for path in tqdm(
            pkl_paths,
            desc=f"Loading {split_name} sessions (TotalCapture syn wrist FLU)",
            unit="session",
            dynamic_ncols=True,
        ):
            result = load_session_syn_flu(
                path,
                target_hz=target_hz,
                add_gravity=add_gravity,
                official_dir=official_dir,
                gyro_white_noise_std=gyro_white_noise_std,
                gyro_bias_drift_std=gyro_bias_drift_std,
                rng=rng,
                num_joints=num_joints,
            )
            if result is None:
                logger.warning("[tc_syn_flu] skipping: %s", path)
                continue
            X, y = result
            if X.shape[0] < self._T:
                logger.warning(
                    "[tc_syn_flu] %s too short (%d < %d), skipping",
                    os.path.basename(path), X.shape[0], self._T,
                )
                continue
            self._sessions.append((X, y))
            session_lengths.append(len(X))

        if not self._sessions:
            raise RuntimeError(
                "TotalCaptureSynWristDatasetFLU: no valid sessions loaded from "
                f"{len(pkl_paths)} pkl paths."
            )

        self._sess_idx, self._starts = _build_window_specs(
            session_lengths, self._T, self._hop
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        si    = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        T     = self._T

        X_raw, y_arr = self._sessions[si]
        Xw = X_raw[start : start + T]   # [T, 6]

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        Xproc    = apply_augment(Xproc.T, self.cfg).T   # [C, T] → augment → [C, T]
        x_tensor = torch.from_numpy(Xproc).float()      # [6, T]

        P        = self._patch_size
        J        = self.num_joints
        patch_labels = np.empty((self._num_patches, J * ROT6D_DIM), dtype=np.float32)
        for p in range(self._num_patches):
            frame = start + p * P + P // 2
            patch_labels[p] = y_arr[frame].reshape(-1)   # [J, 6] → [J*6]

        y_tensor = torch.from_numpy(patch_labels).float()   # [P, J*6]
        return x_tensor, y_tensor


# ---------------------------------------------------------------------------
# Collate helper
# ---------------------------------------------------------------------------

def collate_skip_none(
    batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_totalcapture_syn_wrist_loaders_flu(
    cfg: Any,
    pkl_root: str,
    train_files: Optional[List[str]] = None,
    val_files: Optional[List[str]] = None,
    test_files: Optional[List[str]] = None,
    target_hz: int = 60,
    add_gravity: bool = True,
    official_dir: Optional[str] = None,
    gyro_white_noise_std: float = 0.0,
    gyro_bias_drift_std:  float = 0.0,
    noise_seed: int = 0,
    num_joints: int = 5,
) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders for TotalCapture synthetic-wrist-FLU.

    Parameters
    ----------
    cfg          : experiment config (OmegaConf or dict)
    pkl_root     : directory containing the DIP-format ``.pkl`` session files
    train_files  : filenames relative to pkl_root (or absolute) for training
    val_files    : filenames for validation
    test_files   : filenames for testing
    target_hz    : target sampling rate (raw 60 Hz)
    add_gravity  : see ``load_session_syn_flu``
    official_dir : if given, root translation is loaded from official GT files
    gyro_white_noise_std / gyro_bias_drift_std : MEMS noise parameters
    noise_seed   : random seed for gyro noise
    num_joints   : 5 (right-arm chain) or 24 (all SMPL joints)

    Returns
    -------
    dict with keys ``'train_loader'``, ``'val_loader'``, ``'test_loader'``
    (only keys with non-empty file lists are included).
    """
    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],      256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],       0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],    False))

    def _abs(files: Optional[List[str]]) -> List[str]:
        return [
            f if os.path.isabs(f) else os.path.join(pkl_root, f)
            for f in (files or [])
        ]

    def _make(name: str, files: List[str], bs: int, shuffle: bool):
        if not files:
            return None
        ds = TotalCaptureSynWristDatasetFLU(
            pkl_paths=files,
            cfg=cfg,
            split_name=name,
            target_hz=target_hz,
            add_gravity=add_gravity,
            official_dir=official_dir,
            gyro_white_noise_std=gyro_white_noise_std,
            gyro_bias_drift_std=gyro_bias_drift_std,
            noise_seed=noise_seed,
            num_joints=num_joints,
        )
        persistent = num_workers > 0
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_skip_none,
            persistent_workers=persistent,
        )

    loaders: Dict[str, DataLoader] = {}
    for name, files, bs, shuf in [
        ("train", _abs(train_files), batch_size,      True),
        ("val",   _abs(val_files),   eval_batch_size, False),
        ("test",  _abs(test_files),  eval_batch_size, False),
    ]:
        loader = _make(name, files, bs, shuf)
        if loader is not None:
            loaders[f"{name}_loader"] = loader

    return loaders


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob
    import yaml

    logging.basicConfig(level=logging.INFO)

    cfg = yaml.safe_load(open("configs/base.yaml"))
    cfg.setdefault("vit1d", {})
    cfg["vit1d"].setdefault("patch_size", 16)

    pkl_files = sorted(glob.glob("/mnt/nas2/DIP_recalculate/s1_*.pkl"))
    if not pkl_files:
        raise SystemExit("No pkl files found — adjust the path above.")

    result = load_session_syn_flu(
        pkl_files[0],
        target_hz=60,
        add_gravity=True,
        official_dir=None,
    )
    if result is None:
        raise SystemExit("load_session_syn_flu returned None")
    X, y = result
    print(f"X shape: {X.shape}   y shape: {y.shape}")
    print(f"acc mean/std:  {X[:, :3].mean():.4f} / {X[:, :3].std():.4f}")
    print(f"gyro mean/std: {X[:, 3:].mean():.4f} / {X[:, 3:].std():.4f}")
