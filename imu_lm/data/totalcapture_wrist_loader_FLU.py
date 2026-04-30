"""totalcapture_wrist_loader_FLU.py
Sliding-window DataLoader for the TotalCapture dataset using the REAL right-wrist IMU
in the FLU body frame.

Data source
-----------
Raw DIP-format TotalCapture pkl files (e.g. ``/mnt/nas2/DIP_recalculate/s1_acting1.pkl``).
Each file contains:
  ``acc``  [N, 6, 3]      – kinematic acceleration in the WORLD frame (gravity-subtracted),
                            Y-up convention.
  ``ori``  [N, 6, 3, 3]  – R_world_from_sensor for each of the 6 IMU positions.
  ``gt``   [N, 72]        – SMPL local joint rotations in axis-angle (24 joints × 3).

Sensor ordering (after reindex [2, 3, 0, 1, 4, 5])
---------------------------------------------------
  0 – right wrist   ← used here
  1 – left wrist
  2 – head
  3 – sternum
  4 – right lower leg
  5 – left lower leg

FLU body frame  (same definition as all other ``*_FLU`` loaders)
-----------------------------------------------------------------
Defined at the right-wrist sensor in T-pose:
  Forward (F, x) = −x_smpl   (finger direction)
  Left    (L, y) = +z_smpl   (thumb side)
  Up      (U, z) = +y_smpl   (dorsal / watch-face)

Pipeline
--------
1.  Load acc (world, gravity-subtracted) and ori (R_world_from_sensor).
2.  Select right-wrist sensor (index 0 after reordering).
3.  Gravity handling:
      add_gravity=True  → add [0, 9.81, 0] to acc_world to simulate specific-force
                          reading of a real MEMS accelerometer (Y-up world).
      add_gravity=False → keep gravity-subtracted acc_world as-is.
4.  Build FLU frame:
      R_world_from_X(t) = ori(t) @ R_WRIST_FROM_FLU
      R_X_from_world(t) = R_world_from_X(t)^T
5.  Express acc in FLU:
      acc_flu = R_X_from_world @ acc_world
6.  Derive gyro from consecutive orientations, then express in FLU:
      dR          = ori(t-1)^T @ ori(t)          (sensor-local delta)
      omega_local = rotvec(dR) * fps
      omega_world = ori @ omega_local
      gyro_flu    = R_X_from_world @ omega_world
7.  Labels: axis-angle ``gt`` → rotation matrices → 6D for right-arm chain
    [14 (R_Collar), 17 (R_Shoulder), 19 (R_Elbow), 21 (R_Wrist), 23 (R_Hand)].

Outputs per window
------------------
  x : [6, T]           float32 – (acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U)
  y : [N_patches, 30]  float32 – 6D rotations for right-arm chain (5 joints × 6)
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
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

# Sensor reorder: DIP raw → standard order.
# After this indexing: left wrist, right wrist, left thigh, right thigh, head, pelvis
_SENSOR_REORDER    = [2, 3, 0, 1, 4, 5]
_RIGHT_WRIST_IDX   = 1   # index after reordering

# SMPL right-arm chain joint indices
RIGHT_WRIST_CHAIN_JOINTS = [14, 17, 19, 21, 23]
NUM_CHAIN_JOINTS         = len(RIGHT_WRIST_CHAIN_JOINTS)
ROT6D_DIM                = 6

# World-frame gravity vector (Y-up convention used by SMPL / TotalCapture DIP)
_G_WORLD = np.array([0.0, -9.81, 0.0], dtype=np.float64)

# SMPL kinematic tree: index = joint, value = parent joint (-1 = root pelvis)
_SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
]
_SMPL_RIGHT_WRIST = 21   # joint index for right wrist in SMPL

# ---------------------------------------------------------------------------
# FLU frame definition  (identical to all other *_FLU loaders)
# ---------------------------------------------------------------------------
# Columns are FLU axes expressed in the wrist sensor/SMPL local frame:
#   Forward (F) = −x_smpl
#   Left    (L) = +z_smpl
#   Up      (U) = +y_smpl
R_WRIST_FROM_FLU = np.array(
    [
        [-1.0, 0.0, 0.0],   # F
        [ 0.0, 0.0, 1.0],   # L
        [ 0.0, 1.0, 0.0],   # U
    ],
    dtype=np.float64,
)
assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smpl_global_rotations(R_local: np.ndarray) -> np.ndarray:
    """Compute world-frame rotation matrices for all SMPL joints via FK chain.

    Parameters
    ----------
    R_local : [N, 24, 3, 3]  local rotation matrices (SMPL convention)

    Returns
    -------
    R_global : [N, 24, 3, 3]  world-frame rotation matrices
    """
    R_global = np.empty_like(R_local)
    R_global[:, 0] = R_local[:, 0]   # pelvis / root
    for j in range(1, 24):
        p = _SMPL_PARENTS[j]
        R_global[:, j] = np.einsum("nij,njk->nik", R_global[:, p], R_local[:, j])
    return R_global


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] → [..., 6]  (first two columns, row-major)."""
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def _resample_linear(x: np.ndarray, new_len: int) -> np.ndarray:
    """Linearly interpolate array of shape [N, ...] to length new_len."""
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


# ---------------------------------------------------------------------------
# Core session loader
# ---------------------------------------------------------------------------

def load_session_flu(
    pkl_path: str,
    target_hz: int = 60,
    add_gravity: bool = True,
    use_gt_ori: bool = False,
    num_joints: int = 5,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load one TotalCapture DIP pkl and return **(X, y)** in the FLU frame.

    Parameters
    ----------
    num_joints : 5  → right-arm chain [14,17,19,21,23] (30 dims per patch)
                24 → all SMPL joints (144 dims per patch)

    Parameters
    ----------
    pkl_path    : path to a DIP-format TotalCapture ``.pkl`` file.
    target_hz   : target sampling rate; data is downsampled if needed (raw = 60 Hz).
    add_gravity : True  → simulate real MEMS accelerometer (add gravity to acc_world).
                  False → keep gravity-subtracted kinematic acceleration.
    use_gt_ori  : True  → derive wrist world orientation from GT SMPL FK (joint 21)
                          rather than from the ``ori`` IMU-fusion field.
                  False → use ``ori`` as-is (default; matches real deployment).

    Returns
    -------
    X : [N, 6]    float32 – ``[acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]``
    y : [N, 5, 6] float32 – 6D local rotations for SMPL right-arm chain [14,17,19,21,23]
    ``None`` on any failure.
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    logger.info("[totalcapture_flu] loading: %s", os.path.basename(pkl_path))

    # ------------------------------------------------------------------
    # Load pickle
    # ------------------------------------------------------------------
    try:
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh, encoding="latin1")
    except Exception as exc:
        logger.warning("[totalcapture_flu] cannot open %s: %s", pkl_path, exc)
        return None

    acc_raw = data.get("acc")   # [N, 6, 3]  world-frame kinematic acc (gravity-sub)
    ori_raw = data.get("ori")   # [N, 6, 3, 3]  R_world_from_sensor
    gt_raw  = data.get("gt")    # [N, 72]  axis-angle SMPL local rotations

    if acc_raw is None or ori_raw is None or gt_raw is None:
        logger.warning("[totalcapture_flu] %s: missing keys.", pkl_path)
        return None

    acc_raw = np.asarray(acc_raw, dtype=np.float64)   # [N_a, 6, 3]
    ori_raw = np.asarray(ori_raw, dtype=np.float64)   # [N_a, 6, 3, 3]
    gt_raw  = np.asarray(gt_raw,  dtype=np.float64)   # [N_g, 72]

    # ------------------------------------------------------------------
    # Align lengths (acc/ori may have one extra frame vs gt)
    # ------------------------------------------------------------------
    N = min(acc_raw.shape[0], ori_raw.shape[0], gt_raw.shape[0])
    if N < 3:
        logger.warning("[totalcapture_flu] %s: too short (%d frames).", pkl_path, N)
        return None

    acc_raw = acc_raw[:N]
    ori_raw = ori_raw[:N]
    gt_raw  = gt_raw[:N]

    # ------------------------------------------------------------------
    # Resample to target_hz if needed  (raw = 60 Hz)
    # ------------------------------------------------------------------
    fps_raw = 60.0
    if abs(fps_raw - target_hz) > 1e-3:
        new_len = int(round((N - 1) * target_hz / fps_raw)) + 1
        acc_raw = _resample_linear(acc_raw,       new_len)
        ori_raw = _resample_rotmat_nlerp(ori_raw, new_len)
        gt_raw  = _resample_linear(gt_raw,        new_len)
        N       = new_len
        logger.info(
            "[totalcapture_flu] resampled → %d frames (%.0f→%.0f Hz)", N, fps_raw, target_hz
        )

    fps = float(target_hz)

    # ------------------------------------------------------------------
    # Select right-wrist sensor
    # ------------------------------------------------------------------
    # Reorder: DIP raw [0..5]
    acc_all = acc_raw[:, _SENSOR_REORDER]   # [N, 6, 3]
    ori_all = ori_raw[:, _SENSOR_REORDER]   # [N, 6, 3, 3]

    acc_world_raw  = acc_all[:, _RIGHT_WRIST_IDX]   # [N, 3]  kinematic, world frame, Y-up
    
    use_gt_ori = True
    if use_gt_ori:
        # Use GT SMPL FK (joint 21) as the wrist world orientation.
        # This is cleaner than IMU fusion and avoids sensor-drift artefacts.
        gt_aa_early = gt_raw.reshape(N, 24, 3)
        R_local_gt  = (
            ScipyRot.from_rotvec(gt_aa_early.reshape(-1, 3))
            .as_matrix()
            .reshape(N, 24, 3, 3)
            .astype(np.float64)
        )
        R_world_from_s = _smpl_global_rotations(R_local_gt)[:, _SMPL_RIGHT_WRIST]   # [N,3,3]
        logger.info("[totalcapture_flu] using GT SMPL FK orientation for FLU frame")
    else:
        R_world_from_s = ori_all[:, _RIGHT_WRIST_IDX]   # [N, 3, 3]  R_world_from_sensor

    # ------------------------------------------------------------------
    # Gravity handling
    #   DIP acc_world = a_kinematic_world (gravity already subtracted).
    #   Real IMU specific force: f = a_kinematic - g_world
    #                              = a_kinematic - [0, -9.81, 0]
    #                              = a_kinematic + [0, 9.81, 0]
    # ------------------------------------------------------------------
    acc_world = acc_world_raw.copy()
    if add_gravity:
        acc_world = acc_world - _G_WORLD   # subtract [0, -9.81, 0]  ≡  add [0, 9.81, 0]

    # ------------------------------------------------------------------
    # Build FLU frame per timestep
    #   R_world_from_X(t) = R_world_from_sensor(t) @ R_WRIST_FROM_FLU
    #   R_X_from_world(t) = R_world_from_X(t)^T
    # ------------------------------------------------------------------
    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_s, R_WRIST_FROM_FLU)   # [N, 3, 3]
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)                             # [N, 3, 3]

    # ------------------------------------------------------------------
    # acc in FLU
    # ------------------------------------------------------------------
    acc_flu = np.einsum("nij,nj->ni", R_X_from_world, acc_world)   # [N, 3]

    # ------------------------------------------------------------------
    # Gyro: sensor-local delta-R → world → FLU
    #   dR          = R_world_from_s(t-1)^T @ R_world_from_s(t)
    #   omega_local = rotvec(dR) * fps
    #   omega_world = R_world_from_s(t) @ omega_local
    #   gyro_flu    = R_X_from_world(t) @ omega_world
    # ------------------------------------------------------------------
    dR = np.einsum(
        "nij,njk->nik",
        R_world_from_s[:-1].transpose(0, 2, 1),
        R_world_from_s[1:],
    )  # [N-1, 3, 3]

    omega_local = (
        ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
        .as_rotvec()
        .astype(np.float64)
        * fps
    )  # [N-1, 3]  angular velocity in sensor-local frame

    omega_local_full = np.zeros((N, 3), dtype=np.float64)
    omega_local_full[:-1] = omega_local
    omega_local_full[-1]  = omega_local[-1]   # repeat last frame

    omega_world = np.einsum("nij,nj->ni", R_world_from_s, omega_local_full)   # [N, 3]
    gyro_flu    = np.einsum("nij,nj->ni", R_X_from_world, omega_world)         # [N, 3]

    X = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)         # [N, 6]

    # ------------------------------------------------------------------
    # Labels: gt axis-angle → rotation matrices → right-arm chain local 6D
    # ------------------------------------------------------------------
    gt_aa  = gt_raw.reshape(N, 24, 3)                                           # [N, 24, 3]
    pose_R = (
        ScipyRot.from_rotvec(gt_aa.reshape(-1, 3))
        .as_matrix()
        .reshape(N, 24, 3, 3)
        .astype(np.float32)
    )                                                                            # [N, 24, 3, 3]
    if num_joints == 24:
        label_R = pose_R                                                         # [N, 24, 3, 3]
    else:
        label_R = pose_R[:, RIGHT_WRIST_CHAIN_JOINTS]                           # [N, 5, 3, 3]
    y = matrix_to_rotation_6d(
        torch.from_numpy(label_R)
    ).numpy()                                                                    # [N, J, 6]

    logger.info(
        "[totalcapture_flu] loaded %s: N=%d fps=%.0f real-wrist-flu",
        os.path.basename(pkl_path), N, fps,
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

class TotalCaptureRealWristDatasetFLU(Dataset):
    """Sliding-window dataset backed by real right-wrist IMU from TotalCapture DIP files.

    Input  X : [T, 6]   float32 – [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
    Label  y : [P, 30]  float32 – per-patch 6D rotations for 5-joint right-arm chain
               where P = T // patch_size
    """

    def __init__(
        self,
        pkl_paths: List[str],
        cfg: Any,
        split_name: str = "test",
        target_hz: int = 60,
        add_gravity: bool = True,
        use_gt_ori: bool = False,
        num_joints: int = 5,
    ):
        super().__init__()
        self.cfg         = cfg
        self.split_name  = split_name
        self.target_hz   = target_hz
        self.add_gravity = add_gravity
        self.use_gt_ori  = use_gt_ori
        self.num_joints  = num_joints
        self.stats       = PreprocessStats()

        self._T, self._hop = compute_T_and_hop(cfg)

        vit_cfg           = cfg_get(cfg, ["vit1d"], {}) or {}
        self._patch_size  = int(vit_cfg.get("patch_size", 16))
        self._num_patches = self._T // self._patch_size

        self._sessions: List[Tuple[np.ndarray, np.ndarray]] = []
        session_lengths: List[int] = []

        for path in tqdm(
            pkl_paths,
            desc=f"Loading {split_name} sessions (TotalCapture real wrist FLU)",
            unit="session",
            dynamic_ncols=True,
        ):
            result = load_session_flu(
                path,
                target_hz=target_hz,
                add_gravity=add_gravity,
                use_gt_ori=use_gt_ori,
                num_joints=num_joints,
            )
            if result is None:
                logger.warning("[totalcapture_flu] skipping: %s", path)
                continue
            X, y = result
            if X.shape[0] < self._T:
                logger.warning(
                    "[totalcapture_flu] %s too short (%d < %d), skipping",
                    os.path.basename(path), X.shape[0], self._T,
                )
                continue
            self._sessions.append((X, y))
            session_lengths.append(len(X))

        if not self._sessions:
            raise RuntimeError(
                "TotalCaptureRealWristDatasetFLU: no valid sessions loaded from "
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

        P = self._patch_size
        n_joints = self.num_joints
        patch_labels = np.empty(
            (self._num_patches, n_joints * ROT6D_DIM), dtype=np.float32
        )
        for p in range(self._num_patches):
            frame = start + p * P + P // 2
            patch_labels[p] = y_arr[frame].reshape(-1)   # [J, 6] → [J*6]

        y_tensor = torch.from_numpy(patch_labels).float()   # [P, 30]
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

def make_totalcapture_real_wrist_loaders_flu(
    cfg: Any,
    pkl_root: str,
    train_files: Optional[List[str]] = None,
    val_files: Optional[List[str]] = None,
    test_files: Optional[List[str]] = None,
    target_hz: int = 60,
    add_gravity: bool = True,
    use_gt_ori: bool = False,
    num_joints: int = 5,
) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders for TotalCapture real-wrist-FLU.

    TotalCapture is typically used as a **test-only** dataset; pass files only to
    ``test_files`` in the standard evaluation scenario.

    Parameters
    ----------
    cfg         : experiment config (OmegaConf or dict)
    pkl_root    : directory containing the DIP-format ``.pkl`` session files
    train_files : list of filenames (relative to pkl_root or absolute) for training
    val_files   : list of filenames for validation (optional)
    test_files  : list of filenames for testing (optional)
    target_hz   : target sampling rate (raw is 60 Hz)
    add_gravity : passed to the session loader (see ``load_session_flu``)
    use_gt_ori  : passed to the session loader (see ``load_session_flu``)

    Returns
    -------
    dict with keys ``'train_loader'``, ``'val_loader'``, ``'test_loader'``
    (only keys with non-empty file lists are present).
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
        ds = TotalCaptureRealWristDatasetFLU(
            pkl_paths=files,
            cfg=cfg,
            split_name=name,
            target_hz=target_hz,
            add_gravity=add_gravity,
            use_gt_ori=use_gt_ori,
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
    cfg["vit1d"]["patch_size"] = 16
    cfg.setdefault("data", {})
    cfg["data"]["batch_size"] = 8

    DIP_DIR = "/mnt/nas2/DIP_recalculate"
    pkls = sorted(glob.glob(os.path.join(DIP_DIR, "s1_*.pkl")))[:3]
    assert pkls, f"No pkl files found in {DIP_DIR}"

    print(f"Testing with {len(pkls)} pkl(s): {[os.path.basename(p) for p in pkls]}")

    result = load_session_flu(pkls[0], target_hz=60, add_gravity=True)
    assert result is not None
    X, y = result
    print(f"  X: {X.shape}  y: {y.shape}")
    acc_mag = np.linalg.norm(X[:, :3], axis=1)
    print(f"  acc_mag  mean={acc_mag.mean():.3f}  std={acc_mag.std():.3f}")
    gyro_mag = np.linalg.norm(X[:, 3:], axis=1)
    print(f"  gyro_mag mean={gyro_mag.mean():.3f}  std={gyro_mag.std():.3f}")

    ds = TotalCaptureRealWristDatasetFLU(pkls, cfg, split_name="test")
    print(f"  windows: {len(ds)}")
    item = next(x for x in (ds[i] for i in range(min(len(ds), 200))) if x is not None)
    print(f"  x: {item[0].shape}  y: {item[1].shape}  dtype: {item[0].dtype}")

    loaders = make_totalcapture_real_wrist_loaders_flu(
        cfg,
        pkl_root=DIP_DIR,
        test_files=[os.path.basename(p) for p in pkls],
    )
    for split, dl in loaders.items():
        if dl is not None:
            batch = next(iter(dl))
            if batch is not None:
                print(f"  {split}: x={batch[0].shape}  y={batch[1].shape}")
    print("Smoke test PASSED.")
