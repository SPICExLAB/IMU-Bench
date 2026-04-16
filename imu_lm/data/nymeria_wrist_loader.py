"""NymeriaWristDataset: PyTorch Dataset wrapping Nymeria pkl sessions.

Produces sliding windows of right wrist-region IMU data paired with SMPL
joint rotation labels for the right arm chain:

    x : [6, T]          float32  —  (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
    y : [N_patches, 30]  float32  —  per-patch 6D rotation for joints [14,17,19,21,23]
                                     flattened to 5×6=30 dims, one label per patch.
                                     N_patches = T // patch_size  (default 8 = 128//16)
                                     Each patch label is sampled at the patch center frame.

Right wrist IMU source
----------------------
pkl files contain gt_data['xsens_pose'] — [N, 23, 4, 4] transformation matrices
per Xsens body joint. We derive a synthetic right-wrist IMU from Xsens joint 9
(right wrist) WITHOUT importing pypose or core.articulate:

    acc  — second derivative of joint position  (xsens_pose[:, 9, :3, 3])  × fps²
    gyro — angular velocity from SO3 log map of consecutive rotation matrices

Label source
------------
``gt_data['xsens_pose'][:, :, :3, :3]`` — rotation matrices are converted to
SMPL order using the inlined xsens_to_SMPL mapping, then the first two columns
of each rotation matrix are taken as the 6D representation (no pypose needed).
We extract SMPL joints [14, 17, 19, 21, 23] (right arm chain) at the center
frame of each patch → y shape [N_patches, 30] per sample.

Preprocessing pipeline (identical to loaders.py)
-------------------------------------------------
impute → lowpass filter → z-score normalise → apply_augment

Usage
-----
    from imu_lm.data.nymeria_wrist_loader import (
        NymeriaWristDataset, make_nymeria_loaders
    )

    loaders = make_nymeria_loaders(
        cfg=cfg,
        pkl_root="/data/nymeria_processed",
        train_files=["seq_001.pkl", "seq_002.pkl"],
        val_files=["seq_003.pkl"],
    )
    train_loader = loaders["train_loader"]
    for x, y in train_loader:
        # x: [B, 6, 128]   y: [B, 8, 30]  (8 patches × 30-d rotation)
        ...
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
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

# Channel names emitted by this loader (matches base.yaml sensor_columns order)
WRIST_SENSOR_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# SMPL joint indices for the right arm kinematic chain
# 14: right collar  17: right shoulder  19: right elbow
# 21: right wrist   23: right hand
RIGHT_WRIST_CHAIN_JOINTS = [14, 17, 19, 21, 23]
NUM_CHAIN_JOINTS = len(RIGHT_WRIST_CHAIN_JOINTS)   # 5
ROT6D_DIM = 6
# Flat index slices into joint_rot [N, 24*6]
_CHAIN_INDICES = np.array(
    [j * ROT6D_DIM + d for j in RIGHT_WRIST_CHAIN_JOINTS for d in range(ROT6D_DIM)],
    dtype=np.int64,
)  # shape [30]

# ---------------------------------------------------------------------------
# Xsens → SMPL joint mapping (inlined from nymeriadataset.py so we don't
# need to import pypose / core.articulate)
# ---------------------------------------------------------------------------
# XSENS_TO_SMPL[smpl_idx] = xsens_idx
_XSENS_TO_SMPL = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18,
                   5, 11, 7,  6, 12,  8, 13, 9, 13,  9, 13,  9]

# Right-arm chain Xsens joint index used for synthetic IMU generation.
# SMPL joint 19 (right elbow) → Xsens joint 9 (closest proxy).
_XSENS_RIGHT_WRIST_JOINT = 9


# ---------------------------------------------------------------------------
# Session cache (per-worker, mirrors _SessionCache in loaders.py)
# ---------------------------------------------------------------------------

@dataclass
class _NymeriaSessionCache:
    path: Optional[str] = None
    X: Optional[np.ndarray] = None   # [N, 6]      float32  (acc xyz + gyro xyz)
    y: Optional[np.ndarray] = None   # [N, 5, 6]   float32  (6D rot, right arm chain)


# ---------------------------------------------------------------------------
# Window index helpers (mirrors _build_window_specs)
# ---------------------------------------------------------------------------

def _build_nymeria_window_specs(
    session_lengths: List[int],
    T: int,
    hop: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sess_idx, starts) arrays over all windows across sessions."""
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
# Session loader: returns 6-channel (acc+gyro) array and label array
# ---------------------------------------------------------------------------


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))



def xsens_to_SMPL(xsens_rotmat):
    """
    Convert Xsens rotation matrices to SMPL rotation matrices.
    
    Args:
        xsens_rotmat: Tensor of shape (N, 23, 3, 3) representing Xsens rotation matrices.
        
    Returns:
        Tensor of shape (N, 23, 3, 3) representing SMPL rotation matrices.
    """

    perm = [1, 2, 0] 
    xsens_rotmat = xsens_rotmat[:, :, perm, :]  
    xsens_rotmat = xsens_rotmat[:, :, :, perm]  

    if isinstance(xsens_rotmat, torch.Tensor):
        xsens_rotmat = xsens_rotmat.to(dtype=torch.float32)
    else:
        xsens_rotmat = torch.from_numpy(xsens_rotmat).float()
    
    Rx90 = torch.tensor([
        [1., 0., 0.],
        [0., 0., -1.],
        [0., 1., 0.]
    ], device=xsens_rotmat.device, dtype=xsens_rotmat.dtype)
    
    Rz90 = torch.tensor([
        [0., -1., 0.],
        [1.,  0., 0.],
        [0.,  0., 1.]
    ], device=xsens_rotmat.device, dtype=xsens_rotmat.dtype)
        
    xsens_rotmat[:, :, ...] = Rz90 @ (Rx90 @ xsens_rotmat[:,:, ...])
    
    smpl_rotmat = torch.eye(3).repeat(xsens_rotmat.shape[0], 24, 1, 1)
    for smpl_idx, xsens_idx in enumerate(_XSENS_TO_SMPL):
        smpl_rotmat[:, smpl_idx, :] = xsens_rotmat[:, xsens_idx, :]
            
    return smpl_rotmat.numpy()


# Alias used internally and by run_pose_eval.py
_xsens_to_smpl_rotmat = xsens_to_SMPL


def _load_nymeria_session(
    pkl_path: str,
    target_hz: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load one Nymeria pkl and return (X, y) without pypose/core.articulate.

        X : [N, 6]     float32  — synthetic right-wrist acc (3) + gyro (3)
                                   derived from gt_data['xsens_pose'] joint 9
        y : [N, 5, 6]  float32  — 6D rotation for RIGHT_WRIST_CHAIN_JOINTS
                                   (SMPL joints [14,17,19,21,23])

    IMU derivation
    --------------
    From gt_data['xsens_pose'] [N, 23, 4, 4]:
        positions:          pose[:, joint, :3, 3]
        rotation matrices:  pose[:, joint, :3, :3]

    acc  = second-order finite difference of positions × fps²
    gyro = angular velocity from SO3 log of consecutive rotation matrices × fps
           (uses scipy.spatial.transform.Rotation for numerical stability)

    Returns None if required data is unavailable.
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    logger.info("[nymeria_wrist] loading session: %s", os.path.basename(pkl_path))
    try:
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
    except Exception as exc:
        logger.warning("[nymeria_wrist] failed to open %s: %s", pkl_path, exc)
        return None

    gt_data = raw.get("gt_data", {})
    xsens_pose = gt_data.get("xsens_pose")
    if xsens_pose is None:
        logger.warning("[nymeria_wrist] %s: no 'xsens_pose' in gt_data; skipping.", pkl_path)
        return None

    # Convert to numpy float32 [N, 23, 4, 4]
    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float32)

    N = xsens_pose.shape[0]
    if N < 3:
        logger.warning("[nymeria_wrist] %s: too short (%d frames); skipping.", pkl_path, N)
        return None

    fps = float(raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or target_hz)

    # ------------------------------------------------------------------
    # X: synthetic right-wrist IMU from Xsens joint 9 (right wrist)
    # ------------------------------------------------------------------
    j = _XSENS_RIGHT_WRIST_JOINT
    pos = xsens_pose[:, j, :3, 3]   # [N, 3]  positions in world frame
    rot = xsens_pose[:, j, :3, :3]  # [N, 3, 3]  rotation matrices

    # Acceleration: second-order central difference of position × fps²
    acc = np.zeros((N, 3), dtype=np.float32)
    acc[1:-1] = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) * (fps ** 2)
    acc[0]    = acc[1]
    acc[-1]   = acc[-2]

    # Gyroscope: angular velocity from consecutive rotation matrices
    # dR[i] = rot[i].T @ rot[i+1]  →  rotvec * fps  =  angular velocity (rad/s)
    dR   = np.einsum("nij,njk->nik", rot[:-1].transpose(0, 2, 1), rot[1:])  # [N-1, 3, 3]
    gyro_raw = ScipyRot.from_matrix(dR.reshape(-1, 3, 3)).as_rotvec()  # [N-1, 3]
    gyro_raw = gyro_raw.reshape(N - 1, 3).astype(np.float32) * fps
    gyro = np.zeros((N, 3), dtype=np.float32)
    gyro[:-1] = gyro_raw
    gyro[-1]  = gyro_raw[-1]

    X = np.concatenate([acc, gyro], axis=1)  # [N, 6]

    # ------------------------------------------------------------------
    # y: 6D rotation labels for right arm chain joints (SMPL order)
    # ------------------------------------------------------------------
    xsens_R = xsens_pose[:, :, :3, :3]                  # [N, 23, 3, 3]
    smpl_R  = torch.from_numpy(_xsens_to_smpl_rotmat(xsens_R))            # [N, 24, 3, 3]
    import core.articulate as art
    from core.paths import Paths
    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))
    smpl_R = body_model.inverse_kinematics_R(smpl_R).view(smpl_R.shape[0], 24, 3, 3).contiguous().clone().cpu().numpy()  # [N, 24, 3, 3]
    # Extract right arm chain joints and convert to 6D  [N, 5, 3, 3] → [N, 5, 6]
    chain_R  = smpl_R[:, RIGHT_WRIST_CHAIN_JOINTS]       # [N, 5, 3, 3]
    y = matrix_to_rotation_6d(torch.from_numpy(chain_R))                           # [N, 5, 6]

    logger.info(
        "[nymeria_wrist] loaded %s: N=%d  fps=%.0f",
        os.path.basename(pkl_path), N, fps,
    )
    return X, y


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NymeriaWristDataset(Dataset):
    """Sliding-window dataset over Nymeria right-wrist-region IMU.

    Returns ``(x, y)`` pairs (or ``None`` for skipped windows) where::

        x : torch.Tensor  [6, T]          float32  — acc + gyro at right elbow
        y : torch.Tensor  [N_patches, 30] float32  — per-patch 6D rotation
                                                      for joints [14,17,19,21,23]
                                                      (collar→shoulder→elbow→wrist→hand)
                                                      flattened to 5×6=30 dims.
                                                      N_patches = T // patch_size

    ``patch_size`` is read from ``cfg.vit1d.patch_size`` (default 16), giving
    N_patches = 128 // 16 = 8 for the default window.

    The preprocessing and windowing pipelines are identical to
    ``imu_lm.data.loaders.WindowDataset``.

    Args:
        pkl_paths:  List of absolute paths to Nymeria ``.pkl`` session files.
        cfg:        Config dict (same structure as base.yaml).
        split_name: For logging only.
        target_hz:  Sampling rate used when loading each session (default 50).
    """

    def __init__(
        self,
        pkl_paths: List[str],
        cfg: Any,
        split_name: str = "train",
        target_hz: int = 50,
    ):
        super().__init__()
        self.cfg        = cfg
        self.split_name = split_name
        self.target_hz  = target_hz
        self.stats      = PreprocessStats()

        self._T, self._hop = compute_T_and_hop(cfg)

        # Patch size: align with ViT1D so each patch → one label
        vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
        self._patch_size  = int(vit_cfg.get("patch_size", 16))
        self._num_patches = self._T // self._patch_size

        # Cache config values used on every __getitem__
        self._handle_gaps = bool(cfg_get(cfg, ["windowing", "handle_gaps"], False))

        # Load all sessions eagerly at construction time.
        # Sessions are small enough to fit in RAM; this avoids per-worker
        # serialisation of long torch tensors (unlike the parquet path).
        self._sessions: List[Tuple[np.ndarray, np.ndarray]] = []  # [(X [N,6], y [N])]
        session_lengths: List[int] = []

        for path in tqdm(pkl_paths, desc=f"Loading {split_name} sessions", unit="session", dynamic_ncols=True):
            result = _load_nymeria_session(path, target_hz)
            if result is None:
                logger.warning("[nymeria_wrist] skipping session: %s", path)
                continue
            X, y = result
            self._sessions.append((X, y))
            session_lengths.append(len(X))

        if not self._sessions:
            raise RuntimeError(
                "NymeriaWristDataset: no valid sessions loaded from "
                f"{len(pkl_paths)} pkl paths."
            )

        self._sess_idx, self._starts = _build_nymeria_window_specs(
            session_lengths, self._T, self._hop
        )

        logger.info(
            "[nymeria_wrist] split=%s sessions=%d windows=%d T=%d hop=%d "
            "patch_size=%d num_patches=%d joints=%s",
            split_name,
            len(self._sessions),
            len(self._starts),
            self._T,
            self._hop,
            self._patch_size,
            self._num_patches,
            RIGHT_WRIST_CHAIN_JOINTS,
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._starts)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        si    = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        T     = self._T

        X, y_arr = self._sessions[si]  # X: [N,6], y_arr: [N,5,6]

        # Slice raw window: [T, 6]
        Xw = X[start:start + T]   # [T, 6]

        # Preprocessing (impute → filter → normalise) expects [T, C]
        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        # Augmentation (expects [T, C], returns [T, C])
        Xproc = apply_augment(Xproc.T, self.cfg).T

        # Final IMU tensor: [6, T] (channels first)
        x_tensor = torch.from_numpy(Xproc).float()   # [6, T]

        # Label: one [5*6=30] rotation vector per patch, sampled at each patch's
        # center frame.  Patch p spans [start + p*P, start + p*P + P) where P
        # is patch_size; center = start + p*P + P//2.
        P = self._patch_size
        patch_labels = np.empty((self._num_patches, NUM_CHAIN_JOINTS * ROT6D_DIM), dtype=np.float32)
        for p in range(self._num_patches):
            frame = start + p * P + P // 2
            patch_labels[p] = y_arr[frame].reshape(-1)   # [5, 6] → [30]

        y_tensor = torch.from_numpy(patch_labels).float()  # [N_patches, 30]

        return x_tensor, y_tensor


# ---------------------------------------------------------------------------
# Collate helper (mirrors loaders.py)
# ---------------------------------------------------------------------------

def collate_skip_none(
    batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Drop None entries and stack the rest."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

def make_nymeria_loaders(
    cfg: Any,
    pkl_root: str,
    train_files: List[str],
    val_files: Optional[List[str]] = None,
    test_files: Optional[List[str]] = None,
    target_hz: int = 50,
) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders for Nymeria right-wrist IMU.

    Each batch yields::

        x : [B, 6, T]            float32  — right elbow acc + gyro
        y : [B, N_patches, 30]   float32  — per-patch 6D rotation for
                                            joints [14,17,19,21,23] (5×6=30 dims)
                                            N_patches = T // patch_size (default 8)

    Args:
        cfg:         Config dict (same structure as base.yaml).
        pkl_root:    Directory that contains the ``.pkl`` session files.
        train_files: File names (relative to pkl_root) for training.
        val_files:   File names for validation (optional).
        test_files:  File names for test (optional).
        target_hz:   Sampling rate when loading sessions (must match cfg).

    Returns:
        Dict with one or more of ``"train_loader"``, ``"val_loader"``,
        ``"test_loader"`` keys, each a ``torch.utils.data.DataLoader``.
    """
    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],       256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"],  batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],        0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],     False))

    def _abs(files: List[str]) -> List[str]:
        # If a path is already absolute (e.g. expanded by run.py), keep it.
        return [f if os.path.isabs(f) else os.path.join(pkl_root, f)
                for f in files]

    def _make(name, files, bs, shuffle):
        if not files:
            return None
        ds = NymeriaWristDataset(
            pkl_paths=_abs(files),
            cfg=cfg,
            split_name=name,
            target_hz=target_hz,
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
        ("train", train_files,               batch_size,      True),
        ("val",   val_files   or [],         eval_batch_size, False),
        ("test",  test_files  or [],         eval_batch_size, False),
    ]:
        loader = _make(name, files, bs, shuf)
        if loader is not None:
            loaders[f"{name}_loader"] = loader
            logger.info(
                "[nymeria_wrist] %s_loader: sessions=%d windows=%d batch_size=%d",
                name, len(loader.dataset._sessions), len(loader.dataset), bs,
            )

    return loaders
