"""totalcapture_wrist_odom_loader_FLU.py
TotalCapture DIP pkl → right-wrist FLU IMU + per-patch root displacement labels.

Root displacement labels
------------------------
TotalCapture DIP pkls contain only SMPL pose parameters (no root translation).
Root displacement labels are therefore sourced from the **official TotalCapture
ground-truth skeleton files** (``gt_skel_gbl_pos.txt``) located in the raw release
directory (e.g. ``/mnt/nas2/TotalCapture/TotalCapture/raw/S1/acting1/``).

The Spine joint column is used as the root position proxy, following the DIP
preprocessing convention.  Coordinates are converted from inches to metres and
the X and Z axes are negated to match the SMPL/DIP Y-up convention.

  y[p] = trans[s + (p+1)*P − 1] − trans[s + p*P]
           ↑ last frame of patch p       ↑ first frame of patch p

FLU frame definition (identical to totalcapture_wrist_loader_FLU.py)
----------------------------------------------------------------------
  Forward (F, x) = −x_smpl   (finger direction)
  Left    (L, y) = +z_smpl   (thumb side)
  Up      (U, z) = +y_smpl   (watch face / dorsal)

IMU inputs per window
---------------------
  x : [6, T]           float32  – [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
  y : [N_patches, 3]   float32  – per-patch root displacement (Y-up world, metres)
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
# Constants  (match totalcapture_wrist_loader_FLU.py)
# ---------------------------------------------------------------------------
_SENSOR_REORDER  = [2, 3, 0, 1, 4, 5]
_RIGHT_WRIST_IDX = 0

_INCHES_TO_METERS    = 0.0254
_OFFICIAL_GT_FILENAME = 'gt_skel_gbl_pos.txt'

# World-frame gravity vector (Y-up, as used by SMPL / TotalCapture DIP)
_G_WORLD = np.array([0.0, -9.81, 0.0], dtype=np.float64)

R_WRIST_FROM_FLU = np.array(
    [[-1.0,  0.0,  0.0],
     [ 0.0,  0.0,  1.0],
     [ 0.0,  1.0,  0.0]],
    dtype=np.float64,
)

_SMPL_PARENTS    = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
_SMPL_RIGHT_WRIST = 21


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resample_linear(arr: np.ndarray, new_len: int) -> np.ndarray:
    old_len = arr.shape[0]
    if old_len == new_len:
        return arr
    old_idx = np.linspace(0, old_len - 1, new_len)
    lo      = np.floor(old_idx).astype(int)
    hi      = np.minimum(lo + 1, old_len - 1)
    t       = (old_idx - lo)[..., *([np.newaxis] * (arr.ndim - 1))]
    return arr[lo] * (1 - t) + arr[hi] * t


def _resample_rotmat_nlerp(mats: np.ndarray, new_len: int) -> np.ndarray:
    """NLERP for [N, ..., 3, 3] rotation matrices."""
    from scipy.spatial.transform import Rotation as R, Slerp
    old_len = mats.shape[0]
    if old_len == new_len:
        return mats
    shape    = mats.shape[1:]
    flat     = mats.reshape(old_len, -1, 3, 3)
    n_joints = flat.shape[1]
    out      = np.empty((new_len, n_joints, 3, 3), dtype=np.float64)
    t_old    = np.linspace(0, 1, old_len)
    t_new    = np.linspace(0, 1, new_len)
    for j in range(n_joints):
        slerp    = Slerp(t_old, R.from_matrix(flat[:, j]))
        out[:, j] = slerp(t_new).as_matrix()
    return out.reshape((new_len,) + shape)


def _smpl_global_rotations(R_local: np.ndarray) -> np.ndarray:
    """Compose local rotations up the SMPL kinematic chain. [N,24,3,3] → [N,24,3,3]."""
    N = R_local.shape[0]
    R_global = np.empty_like(R_local)
    R_global[:, 0] = R_local[:, 0]
    for j, p in enumerate(_SMPL_PARENTS):
        if p < 0:
            continue
        R_global[:, j] = R_global[:, p] @ R_local[:, j]
    return R_global


def _build_window_specs(session_lengths, T, hop):
    total    = sum(max(0, (n - T) // hop + 1) for n in session_lengths)
    sess_idx = np.empty(total, dtype=np.int32)
    starts   = np.empty(total, dtype=np.int32)
    offset   = 0
    for ki, n in enumerate(session_lengths):
        if n < T:
            continue
        n_win = (n - T) // hop + 1
        sess_idx[offset:offset + n_win] = ki
        starts[offset:offset + n_win]   = np.arange(0, n - T + 1, hop, dtype=np.int32)
        offset += n_win
    return sess_idx, starts


# ---------------------------------------------------------------------------
# Official GT translation loader
# ---------------------------------------------------------------------------

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
    basename = os.path.splitext(os.path.basename(pkl_path))[0]  # e.g. s5_freestyle1
    parts = basename.split('_', 1)
    if len(parts) != 2:
        logger.warning("[tc_odom_flu] cannot parse pkl filename: %s", basename)
        return None
    subject = parts[0].upper()   # s5 → S5
    motion  = parts[1]           # freestyle1

    gt_file = os.path.join(official_dir, subject, motion, _OFFICIAL_GT_FILENAME)
    if not os.path.isfile(gt_file):
        logger.warning("[tc_odom_flu] gt file not found: %s", gt_file)
        return None

    try:
        with open(gt_file, 'r') as fh:
            header_cols = fh.readline().rstrip('\n\r').split('\t')
            try:
                spine_col = header_cols.index('Spine')
            except ValueError:
                logger.warning("[tc_odom_flu] 'Spine' column missing in %s", gt_file)
                return None
            rows: list = []
            for line in fh:
                line = line.rstrip('\n\r')
                if not line:
                    continue
                fields = line.split('\t')
                if spine_col >= len(fields):
                    continue
                xyz = [float(v) for v in fields[spine_col].split()]
                rows.append(xyz)
    except Exception as exc:
        logger.warning("[tc_odom_flu] error reading %s: %s", gt_file, exc)
        return None

    if not rows:
        logger.warning("[tc_odom_flu] empty GT file: %s", gt_file)
        return None

    trans = np.array(rows, dtype=np.float64) * _INCHES_TO_METERS  # [N, 3]
    trans[:, 0] *= -1   # negate X  (matches DIP preprocessing convention)
    trans[:, 2] *= -1   # negate Z

    # Resample to target_hz if native rate differs
    N = trans.shape[0]
    if abs(native_hz - target_hz) > 1e-3:
        new_len = int(round((N - 1) * target_hz / native_hz)) + 1
        trans = _resample_linear(trans, new_len)

    logger.debug("[tc_odom_flu] loaded GT trans %s: %d frames", basename, trans.shape[0])
    return trans


# ---------------------------------------------------------------------------
# Session loader
# ---------------------------------------------------------------------------

def load_session_odom_flu(
    pkl_path: str,
    target_hz: int = 50,
    add_gravity: bool = True,
    use_gt_ori: bool = False,
    official_dir: str = '/mnt/nas2/TotalCapture/TotalCapture/raw',
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load one TotalCapture DIP pkl → (X_flu, trans_world).

    Returns
    -------
    X_flu       : [N, 6]  float32 – FLU-frame IMU  [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
    trans_world : [N, 3]  float32 – GT root (Spine) absolute world position (Y-up, metres)
                                    loaded from the official ``gt_skel_gbl_pos.txt``.
    ``None`` on failure.
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    logger.info("[totalcapture_odom_flu] loading: %s", os.path.basename(pkl_path))

    try:
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh, encoding="latin1")
    except Exception as exc:
        logger.warning("[totalcapture_odom_flu] cannot open %s: %s", pkl_path, exc)
        return None

    acc_raw = data.get("acc")    # [N, 6, 3]  kinematic world acc (gravity-sub, Y-up)
    ori_raw = data.get("ori")    # [N, 6, 3, 3]
    gt_raw  = data.get("gt")     # [N, 72]  SMPL axis-angle

    if acc_raw is None or ori_raw is None or gt_raw is None:
        logger.warning("[totalcapture_odom_flu] %s: missing keys.", pkl_path)
        return None

    acc_raw = np.asarray(acc_raw, dtype=np.float64)
    ori_raw = np.asarray(ori_raw, dtype=np.float64)
    gt_raw  = np.asarray(gt_raw,  dtype=np.float64)

    N = min(acc_raw.shape[0], ori_raw.shape[0], gt_raw.shape[0])
    if N < 3:
        logger.warning("[totalcapture_odom_flu] %s: too short (%d).", pkl_path, N)
        return None

    acc_raw = acc_raw[:N]
    ori_raw = ori_raw[:N]
    gt_raw  = gt_raw[:N]

    fps_raw = 60.0
    if abs(fps_raw - target_hz) > 1e-3:
        new_len = int(round((N - 1) * target_hz / fps_raw)) + 1
        acc_raw = _resample_linear(acc_raw,        new_len)
        ori_raw = _resample_rotmat_nlerp(ori_raw,  new_len)
        gt_raw  = _resample_linear(gt_raw,         new_len)
        N       = new_len
        logger.info("[totalcapture_odom_flu] resampled → %d frames (60→%d Hz)", N, target_hz)

    fps = float(target_hz)
    dt  = 1.0 / fps

    # --- GT root translation (Spine from official gt_skel_gbl_pos.txt) ------
    trans_world = _load_gt_skel_spine(pkl_path, official_dir,
                                      target_hz=target_hz, native_hz=fps_raw)
    if trans_world is None:
        logger.warning("[tc_odom_flu] skipping %s: GT trans unavailable.", pkl_path)
        return None

    # Align lengths (GT file and DIP pkl may differ by a few frames)
    N = min(N, trans_world.shape[0])
    trans_world = trans_world[:N]

    # --- Sensor selection --------------------------------------------------
    acc_all = acc_raw[:N, _SENSOR_REORDER]   # [N, 6, 3]
    ori_all = ori_raw[:N, _SENSOR_REORDER]   # [N, 6, 3, 3]
    acc_wrist_kin = acc_all[:, _RIGHT_WRIST_IDX]  # [N, 3] wrist kinematic acc for FLU input

    # --- Wrist orientation for FLU frame -----------------------------------
    if use_gt_ori:
        gt_aa   = gt_raw[:N].reshape(N, 24, 3)
        R_local = (ScipyRot.from_rotvec(gt_aa.reshape(-1, 3))
                   .as_matrix().reshape(N, 24, 3, 3).astype(np.float64))
        R_world_from_s = _smpl_global_rotations(R_local)[:, _SMPL_RIGHT_WRIST]
    else:
        R_world_from_s = ori_all[:, _RIGHT_WRIST_IDX]   # [N, 3, 3]

    # --- FLU frame ---------------------------------------------------------
    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_s, R_WRIST_FROM_FLU)
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)

    acc_wrist_mems = acc_wrist_kin.copy()
    if add_gravity:
        # subtract _G_WORLD = [0, -9.81, 0]  →  effectively adds +9.81 in Y
        acc_wrist_mems = acc_wrist_mems - _G_WORLD

    acc_flu  = np.einsum("nij,nj->ni", R_X_from_world, acc_wrist_mems)

    # --- Gyro --------------------------------------------------------------
    dR = np.einsum(
        "nij,njk->nik",
        R_world_from_s[:-1].transpose(0, 2, 1),
        R_world_from_s[1:],
    )
    omega_local = (ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
                   .as_rotvec().astype(np.float64) * fps)
    omega_local_full = np.zeros((N, 3), dtype=np.float64)
    omega_local_full[:-1] = omega_local
    omega_local_full[-1]  = omega_local[-1]
    omega_world = np.einsum("nij,nj->ni", R_world_from_s, omega_local_full)
    gyro_flu    = np.einsum("nij,nj->ni", R_X_from_world, omega_world)

    X = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)   # [N, 6]

    logger.info("[tc_odom_flu] loaded %s: N=%d fps=%d",
                os.path.basename(pkl_path), N, int(fps))
    return X, trans_world.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TotalCaptureOdomFLUDataset(Dataset):
    """Sliding-window dataset: right-wrist FLU IMU → per-patch root displacement labels.

    Displacement labels are read from the official TotalCapture GT skeleton files
    (``gt_skel_gbl_pos.txt``, Spine joint) and require *official_dir* pointing
    to the raw TotalCapture release directory.
    """

    def __init__(
        self,
        pkl_paths: List[str],
        cfg: Any,
        split_name: str = "train",
        target_hz: int = 50,
        add_gravity: bool = True,
        use_gt_ori: bool = False,
        official_dir: str = '/mnt/nas2/TotalCapture/TotalCapture/raw',
    ):
        super().__init__()
        self.cfg      = cfg
        self.stats    = PreprocessStats()
        self._T, self._hop = compute_T_and_hop(cfg)

        vit_cfg          = cfg_get(cfg, ["vit1d"], {}) or {}
        self._patch_size = int(vit_cfg.get("patch_size", 16))
        self._num_patches = self._T // self._patch_size

        self._sessions: List[Tuple[np.ndarray, np.ndarray]] = []
        session_lengths: List[int] = []

        for path in tqdm(pkl_paths,
                         desc=f"Loading {split_name} (TotalCapture odom FLU)",
                         unit="session", dynamic_ncols=True):
            result = load_session_odom_flu(
                path, target_hz=target_hz,
                add_gravity=add_gravity, use_gt_ori=use_gt_ori,
                official_dir=official_dir,
            )
            if result is None:
                continue
            X, trans_world = result
            if X.shape[0] < self._T:
                continue
            self._sessions.append((X, trans_world))
            session_lengths.append(X.shape[0])

        if not self._sessions:
            raise RuntimeError(
                f"TotalCaptureOdomFLUDataset: no valid sessions from {len(pkl_paths)} pkls."
            )

        self._sess_idx, self._starts = _build_window_specs(
            session_lengths, self._T, self._hop
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        si    = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        T, P  = self._T, self._patch_size

        X_raw, trans_world = self._sessions[si]
        Xw      = X_raw[start:start + T]          # [T, 6]
        trans_w = trans_world[start:start + T]    # [T, 3]  GT root position (Y-up, metres)

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None
        Xproc    = apply_augment(Xproc.T, self.cfg).T   # [6, T]
        x_tensor = torch.from_numpy(Xproc).float()

        # --- Per-patch displacement from GT root translation ---------------
        # y[p] = trans at last frame of patch p  −  trans at first frame of patch p
        patch_disp = np.empty((self._num_patches, 3), dtype=np.float32)
        for p in range(self._num_patches):
            patch_disp[p] = trans_w[(p + 1) * P - 1] - trans_w[p * P]

        y_tensor = torch.from_numpy(patch_disp).float()   # [Np, 3]
        return x_tensor, y_tensor


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_totalcapture_odom_flu_loaders(
    cfg: Any,
    pkl_root: str,
    train_files: Optional[List[str]] = None,
    val_files:   Optional[List[str]] = None,
    test_files:  Optional[List[str]] = None,
    target_hz:   int  = 50,
    add_gravity: bool = True,
    use_gt_ori:  bool = False,
    official_dir: str = '/mnt/nas2/TotalCapture/TotalCapture/raw',
) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders for TotalCapture odom in FLU frame."""
    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],      256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],       0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],    False))

    def _abs(files):
        return [f if os.path.isabs(f) else os.path.join(pkl_root, f)
                for f in (files or [])]

    def _make(name, files, bs, shuffle):
        if not files:
            return None
        ds = TotalCaptureOdomFLUDataset(
            pkl_paths=files, cfg=cfg, split_name=name,
            target_hz=target_hz, add_gravity=add_gravity, use_gt_ori=use_gt_ori,
            official_dir=official_dir,
        )
        persistent = num_workers > 0
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory,
                          drop_last=False, collate_fn=_collate,
                          persistent_workers=persistent)

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
