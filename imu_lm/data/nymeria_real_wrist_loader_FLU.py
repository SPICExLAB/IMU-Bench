from __future__ import annotations

import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# Constants (mirrored from nymeria_wrist_loader_FLU.py)
# ---------------------------------------------------------------------------

WRIST_SENSOR_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

RIGHT_WRIST_CHAIN_JOINTS = [14, 17, 19, 21, 23]
NUM_CHAIN_JOINTS = len(RIGHT_WRIST_CHAIN_JOINTS)
ROT6D_DIM = 6

# SMPL right wrist joint index
_SMPL_RIGHT_WRIST = 21

# ---------------------------------------------------------------------------
# FLU frame definition — identical to nymeria_wrist_loader_FLU.py
# ---------------------------------------------------------------------------
# Columns are FLU axes expressed in the wrist CPF (local) frame:
#   Forward (FLU x) = -x_wrist_cpf
#   Left    (FLU y) = +z_wrist_cpf
#   Up      (FLU z) = +y_wrist_cpf
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

_XSENS_TO_SMPL = [
    0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18,
    5, 11,  7, 6, 12,  8, 13,  9, 13,  9, 13,  9,
]


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def xsens_to_SMPL(xsens_rotmat: np.ndarray) -> np.ndarray:
    """Convert Xsens global rotation matrices [N,23,3,3] → SMPL global [N,24,3,3]."""
    perm = [1, 2, 0]
    xsens_rotmat = xsens_rotmat[:, :, perm, :]
    xsens_rotmat = xsens_rotmat[:, :, :, perm]

    xsens_t = torch.from_numpy(xsens_rotmat).float()
    Rx90 = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    Rz90 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    xsens_t = Rz90 @ (Rx90 @ xsens_t)

    smpl_rotmat = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(
        xsens_t.shape[0], 24, -1, -1
    ).clone()
    for smpl_idx, xsens_idx in enumerate(_XSENS_TO_SMPL):
        smpl_rotmat[:, smpl_idx] = xsens_t[:, xsens_idx]
    return smpl_rotmat.numpy()  # [N, 24, 3, 3]


def _quat_xyzw_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert (x, y, z, w) quaternions to rotation matrices. [N, 4] → [N, 3, 3]."""
    from scipy.spatial.transform import Rotation as ScipyRot
    return ScipyRot.from_quat(quat).as_matrix().astype(np.float64)


# ---------------------------------------------------------------------------
# Core session loader
# ---------------------------------------------------------------------------

def _load_nymeria_real_wrist_session_flu(
    pkl_path: str,
    target_hz: int,
    add_gravity: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load a Nymeria session using the REAL wrist IMU data.

    Transformation pipeline:
        wrist IMU local CPF  →  global CPF (world)  →  local FLU

    Steps
    -----
    1.  Read ``wrist_imu_data['accel']`` and ``wrist_imu_data['gyro']``
        (measured in the wrist sensor's local CPF frame).
    2.  Read ``wrist_gt_data['orientation']`` — CPF→world rotation stored as
        [N, 4] (x, y, z, w) quaternion or PyPose SO3.  Convert to rotation
        matrices R_world_from_wrist_cpf  [N, 3, 3].
    3.  Rotate acc and gyro to the world frame:
            acc_world  = R_world_from_wrist_cpf @ acc_local
            gyro_world = R_world_from_wrist_cpf @ gyro_local
    4.  Gravity handling:
            add_gravity=True  → keep signal as-is (real IMU includes gravity).
            add_gravity=False → subtract the static gravity from world-frame acc
                                (adds g_world = [0, −9.81, 0] to acc_world to
                                 recover pure linear acceleration; Y-up convention).
    5.  Build FLU frame per timestep:
            R_world_from_X(t) = R_world_from_wrist_cpf(t) @ R_WRIST_FROM_FLU
            R_X_from_world(t) = R_world_from_X(t)^T
    6.  Express acc and gyro in the FLU frame:
            acc_flu  = R_X_from_world @ acc_world
            gyro_flu = R_X_from_world @ gyro_world
    7.  Load xsens_pose → SMPL IK+FK → right-arm chain local rotations (labels).

    Parameters
    ----------
    pkl_path   : path to Nymeria .pkl session file
    target_hz  : target sampling rate; raw data is downsampled if needed
    add_gravity: True  → gravity included in output acc (real IMU behavior).
                 False → gravity removed (pure linear acceleration).

    Returns
    -------
    X_imu : [N, 6] float32   — [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
    y     : [N, 5, 6] float32 — 6D local rotations for right-arm chain joints
    """
    import core.articulate as art
    from core.paths import Paths

    logger.info("[nymeria_real_wrist_flu] loading: %s", os.path.basename(pkl_path))
    try:
        with open(pkl_path, "rb") as fh:
            raw = pickle.load(fh)
    except Exception as exc:
        logger.warning("[nymeria_real_wrist_flu] cannot open %s: %s", pkl_path, exc)
        return None

    # ------------------------------------------------------------------
    # Check required data
    # ------------------------------------------------------------------
    if not raw.get("has_wrist_imu", False):
        logger.warning(
            "[nymeria_real_wrist_flu] %s: has_wrist_imu=False, skipping", pkl_path
        )
        return None

    wrist_imu = raw.get("wrist_imu_data", {})
    wrist_gt  = raw.get("wrist_gt_data",  {})

    for key in ("accel", "gyro"):
        if key not in wrist_imu:
            logger.warning(
                "[nymeria_real_wrist_flu] %s: '%s' missing in wrist_imu_data", pkl_path, key
            )
            return None

    if "orientation" not in wrist_gt:
        logger.warning(
            "[nymeria_real_wrist_flu] %s: 'orientation' missing in wrist_gt_data", pkl_path
        )
        return None

    xsens_pose = raw.get("gt_data", {}).get("xsens_pose")
    if xsens_pose is None:
        logger.warning(
            "[nymeria_real_wrist_flu] %s: 'xsens_pose' missing (needed for labels)", pkl_path
        )
        return None

    # ------------------------------------------------------------------
    # Determine sampling rates and downsample to target_hz
    # ------------------------------------------------------------------
    body_fps_raw  = float(
        raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or target_hz
    )
    wrist_fps_raw = float(raw.get("wrist_imu_sampling_rate_hz", body_fps_raw))

    body_ds  = max(1, round(body_fps_raw  / target_hz))
    wrist_ds = max(1, round(wrist_fps_raw / target_hz))

    # Body pose (for labels)
    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float64)[::body_ds]   # [N_b, 23, 4, 4]

    # Wrist IMU signal (local wrist CPF frame)
    def _to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    acc_local  = _to_numpy(wrist_imu["accel"]).astype(np.float64)[::wrist_ds]   # [N_w, 3]
    gyro_local = _to_numpy(wrist_imu["gyro"]).astype(np.float64)[::wrist_ds]    # [N_w, 3]

    # Wrist GT orientation → R_world_from_wrist_cpf  [N_w, 3, 3]
    orient_raw = wrist_gt["orientation"]

    try:
        import pypose as pp
        _is_lietensor = isinstance(orient_raw, pp.LieTensor)
    except ImportError:
        _is_lietensor = False

    if _is_lietensor:
        R_world_from_wrist_cpf = (
            orient_raw.matrix().detach().cpu().numpy().astype(np.float64)[::wrist_ds]
        )  # [N_w, 3, 3]
    else:
        orient_np = _to_numpy(orient_raw).astype(np.float64)[::wrist_ds]  # [N_w, 4]
        R_world_from_wrist_cpf = _quat_xyzw_to_rotmat(orient_np)          # [N_w, 3, 3]

    # ------------------------------------------------------------------
    # Align lengths: body pose and wrist data may differ by a few frames
    # ------------------------------------------------------------------
    N = min(xsens_pose.shape[0], acc_local.shape[0])

    if N < 3:
        logger.warning(
            "[nymeria_real_wrist_flu] %s: too short after alignment (%d frames)", pkl_path, N
        )
        return None

    xsens_pose             = xsens_pose[:N]
    acc_local              = acc_local[:N]
    gyro_local             = gyro_local[:N]
    R_world_from_wrist_cpf = R_world_from_wrist_cpf[:N]

    # ------------------------------------------------------------------
    # Step 3: local CPF → world
    #   acc_world  = R_world_from_wrist_cpf @ acc_local
    #   gyro_world = R_world_from_wrist_cpf @ gyro_local
    # ------------------------------------------------------------------
    acc_world  = np.einsum("nij,nj->ni", R_world_from_wrist_cpf, acc_local)   # [N, 3]
    gyro_world = np.einsum("nij,nj->ni", R_world_from_wrist_cpf, gyro_local)  # [N, 3]

    # ------------------------------------------------------------------
    # Step 4: gravity handling
    #   Real IMU acc_world = a_linear_world - g_world
    #   where g_world = [0, -9.81, 0]  (Y-up, gravity points in -Y)
    #   add_gravity=True  → keep as-is   (gravity term present)
    #   add_gravity=False → a_linear_world = acc_world + g_world
    # ------------------------------------------------------------------
    if not add_gravity:
        g_world = np.array([0.0, -9.81, 0.0], dtype=np.float64)
        acc_world = acc_world + g_world

    # ------------------------------------------------------------------
    # Steps 5–6: world → local FLU
    #   R_world_from_X(t) = R_world_from_wrist_cpf(t) @ R_WRIST_FROM_FLU
    #   R_X_from_world(t) = R_world_from_X(t)^T
    #   acc_flu  = R_X_from_world @ acc_world
    #   gyro_flu = R_X_from_world @ gyro_world
    # ------------------------------------------------------------------
    R_world_from_X = np.einsum(
        "nij,jk->nik", R_world_from_wrist_cpf, R_WRIST_FROM_FLU
    )  # [N, 3, 3]
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)  # [N, 3, 3]

    acc_flu  = np.einsum("nij,nj->ni", R_X_from_world, acc_world)   # [N, 3]
    gyro_flu = np.einsum("nij,nj->ni", R_X_from_world, gyro_world)  # [N, 3]

    X_imu = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)  # [N, 6]

    # ------------------------------------------------------------------
    # Step 7: labels — SMPL IK+FK → right-arm chain local rotations
    # ------------------------------------------------------------------
    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))

    xsens_R       = xsens_pose[:, :, :3, :3].astype(np.float32)   # [N, 23, 3, 3]
    smpl_global_R = xsens_to_SMPL(xsens_R)                         # [N, 24, 3, 3]

    smpl_global_t = torch.from_numpy(smpl_global_R).float()
    smpl_local_t  = body_model.inverse_kinematics_R(smpl_global_t)  # [N, 24, 3, 3]
    smpl_local_R  = smpl_local_t.cpu().numpy().astype(np.float64)

    chain_R = smpl_local_R[:, RIGHT_WRIST_CHAIN_JOINTS]   # [N, 5, 3, 3]
    y = matrix_to_rotation_6d(
        torch.from_numpy(chain_R.astype(np.float32))
    ).numpy()   # [N, 5, 6]

    logger.info(
        "[nymeria_real_wrist_flu] loaded %s: N=%d fps=%.0f real-wrist-flu",
        os.path.basename(pkl_path), N, target_hz,
    )
    return X_imu, y


# ---------------------------------------------------------------------------
# Window index builder
# ---------------------------------------------------------------------------

def _build_window_specs(
    session_lengths: List[int],
    T: int,
    hop: int,
) -> Tuple[np.ndarray, np.ndarray]:
    total = sum(max(0, (n - T) // hop + 1) for n in session_lengths)
    sess_idx = np.empty(total, dtype=np.int32)
    starts   = np.empty(total, dtype=np.int32)
    offset = 0
    for ki, n in enumerate(session_lengths):
        if n < T:
            continue
        n_win = (n - T) // hop + 1
        sess_idx[offset:offset + n_win] = ki
        starts[offset:offset + n_win]   = np.arange(0, n - T + 1, hop, dtype=np.int32)
        offset += n_win
    return sess_idx, starts


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NymeriaRealWristDatasetFLU(Dataset):
    """
    Sliding-window dataset backed by REAL wrist IMU data from Nymeria pickle files.

    Input  X: [T, 6]   float32  — [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
    Label  y: [P, 30]  float32  — per-patch 6D rotations for the 5-joint right-arm chain
               where P = T // patch_size
    """

    def __init__(
        self,
        pkl_paths: List[str],
        cfg: Any,
        split_name: str = "train",
        target_hz: int = 50,
        add_gravity: bool = True,
    ):
        super().__init__()
        self.cfg        = cfg
        self.split_name = split_name
        self.target_hz  = target_hz
        self.add_gravity = add_gravity
        self.stats = PreprocessStats()

        self._T, self._hop = compute_T_and_hop(cfg)

        vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
        self._patch_size  = int(vit_cfg.get("patch_size", 16))
        self._num_patches = self._T // self._patch_size

        self._sessions: List[Tuple[np.ndarray, np.ndarray]] = []
        session_lengths: List[int] = []

        _n_workers = min(32, (os.cpu_count() or 4) * 2)
        _load = lambda p: _load_nymeria_real_wrist_session_flu(
            p, target_hz=target_hz, add_gravity=add_gravity
        )
        _ordered: List = [None] * len(pkl_paths)
        with ThreadPoolExecutor(max_workers=_n_workers) as _pool:
            _fut_to_idx = {_pool.submit(_load, p): i for i, p in enumerate(pkl_paths)}
            with tqdm(total=len(pkl_paths), desc=f"Loading {split_name} sessions (real wrist FLU)", unit="session", dynamic_ncols=True) as _pbar:
                for _fut in as_completed(_fut_to_idx):
                    _ordered[_fut_to_idx[_fut]] = _fut.result()
                    _pbar.update(1)
        for path, result in zip(pkl_paths, _ordered):
            if result is None:
                logger.warning("[nymeria_real_wrist_flu] skipping: %s", path)
                continue
            X, y = result
            self._sessions.append((X, y))
            session_lengths.append(len(X))

        if not self._sessions:
            raise RuntimeError(
                "NymeriaRealWristDatasetFLU: no valid sessions loaded from "
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
        Xw = X_raw[start : start + T]

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        if self.split_name == "train":
            Xproc = apply_augment(Xproc.T, self.cfg).T
        x_tensor = torch.from_numpy(Xproc).float()

        P = self._patch_size
        patch_labels = np.empty(
            (self._num_patches, NUM_CHAIN_JOINTS * ROT6D_DIM), dtype=np.float32
        )
        for p in range(self._num_patches):
            frame = start + p * P + P // 2
            patch_labels[p] = y_arr[frame].reshape(-1)

        y_tensor = torch.from_numpy(patch_labels).float()
        return x_tensor, y_tensor


def collate_skip_none(
    batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


def make_nymeria_real_wrist_loaders_flu(
    cfg: Any,
    pkl_root: str,
    train_files: List[str],
    val_files: Optional[List[str]] = None,
    test_files: Optional[List[str]] = None,
    target_hz: int = 50,
    add_gravity: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders for the real-wrist-FLU Nymeria dataset.

    Parameters
    ----------
    cfg         : experiment config (OmegaConf or dict)
    pkl_root    : directory that contains the .pkl session files
    train_files : list of filenames (relative to pkl_root or absolute)
    val_files   : list of filenames for validation (optional)
    test_files  : list of filenames for testing (optional)
    target_hz   : target sampling rate
    add_gravity : passed to the session loader (see _load_nymeria_real_wrist_session_flu)

    Returns
    -------
    dict with keys 'train_loader', 'val_loader', 'test_loader' (present only if files given)
    """
    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],      256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],       0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],    False))

    def _abs(files: List[str]) -> List[str]:
        return [
            f if os.path.isabs(f) else os.path.join(pkl_root, f)
            for f in files
        ]

    def _make(name: str, files: List[str], bs: int, shuffle: bool):
        if not files:
            return None
        ds = NymeriaRealWristDatasetFLU(
            pkl_paths=_abs(files),
            cfg=cfg,
            split_name=name,
            target_hz=target_hz,
            add_gravity=add_gravity,
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
        ("train", train_files,         batch_size,      True),
        ("val",   val_files  or [],    eval_batch_size, False),
        ("test",  test_files or [],    eval_batch_size, False),
    ]:
        loader = _make(name, files, bs, shuf)
        if loader is not None:
            loaders[f"{name}_loader"] = loader

    return loaders
