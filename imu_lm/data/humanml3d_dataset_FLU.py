"""
HumanML3D dataloaders — FLU wrist-IMU variant.

Identical to humanml3d_dataset.py **except** the synthetic right-wrist IMU is
expressed in the FLU (Forward-Left-Up) body frame, matching the convention of
``imu_lm/data/nymeria_wrist_loader_FLU.py``.

FLU frame definition (T-pose, right wrist, geometric arm-chain frame)
----------------------------------------------------------------------
_build_wrist_rotation builds a frame from shoulder→elbow→wrist positions:

    col 0 (x_geom) = normalize(z × upper_arm)  — lateral (≈ thumb side)
    col 1 (y_geom) = normalize(z × x)           — dorsal  (out of watch face)
    col 2 (z_geom) = normalize(wrist − elbow)   — toward fingertips

The FLU mapping from this geometric frame:

    Forward (FLU x)  =  +z_geom  (toward fingertips, away from arm)
    Left    (FLU y)  =  +x_geom  (toward thumb side of right wrist)
    Up      (FLU z)  =  +y_geom  (out of watch face, dorsal, away from skin)

Note: this is a DIFFERENT R_WRIST_FROM_FLU than nymeria_wrist_loader_FLU.py,
which uses the SMPL FK joint frame (x=proximal, y=dorsal, z=thumb).

IMU derivation pipeline (per frame, world frame → FLU)
-------------------------------------------------------
1.  Resample HumanML3D joint positions from 20 Hz → 50 Hz.
2.  Build wrist rotation matrix R_world_from_wrist(t) from arm-chain geometry
    at every frame (shoulder → elbow → wrist joint positions).
3.  Propagate FLU frame to world:
        R_world_from_X(t) = R_world_from_wrist(t) @ R_WRIST_FROM_FLU
4.  Compute world-frame linear acceleration of the wrist joint:
        a_world(t) = d²p_wrist / dt² × fps²
5.  Compute angular velocity in the wrist local frame via SO3 log-map, then
    rotate to world:
        ω_world(t) = R_world_from_wrist(t) @ ω_wrist_local(t)
6.  Project both into FLU:
        acc_flu(t)  = R_X_from_world(t) @ a_world(t)
        gyro_flu(t) = R_X_from_world(t) @ ω_world(t)

This is the **exact same pipeline** as ``nymeria_wrist_loader_FLU.py``; the
only difference is that R_world_from_wrist is estimated from joint positions
instead of SMPL FK rotation matrices.

Exported symbols
----------------
- ``_joints_to_imu_flu``            core IMU derivation (FLU frame)
- ``HumanML3DIMUDatasetFLU``        map-style Dataset
- ``HumanML3DIMUVarLenDatasetFLU``  iterable (variable T) Dataset
- ``make_humanml3d_imu_loaders_flu`` DataLoader factory
- ``imu_text_collate_fn``           (re-exported from base module)
"""

from __future__ import annotations

import logging
import os
import random as pyrandom
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from os.path import join as pjoin

# Re-export helpers that are shared with the base module
from imu_lm.data.humanml3d_dataset import (
    _resample_joints,
    _snap_T,
    imu_text_collate_fn,
    _HML3D_FPS,
    _TARGET_FPS,
    _R_SHOULDER,
    _R_ELBOW,
    _R_WRIST,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FLU frame definition — SMPL FK joint frame (same as nymeria_wrist_loader_FLU.py)
# ---------------------------------------------------------------------------
# R_world_from_wrist(t) is taken from SMPL FK pose_global[:, 21].
# In the SMPL T-pose the right-wrist local axes are:
#   x_smpl  = proximal direction (toward elbow)
#   y_smpl  = dorsal  (out of watch face)
#   z_smpl  = thumb side (lateral)
#
# FLU mapping (identical constants to nymeria_wrist_loader_FLU.py):
#   F (forward / fingertips) = −x_smpl
#   L (left    / thumb side) = +z_smpl
#   U (up      / watch face) = +y_smpl
#
# Columns of R_WRIST_FROM_FLU are the F, L, U axes in SMPL wrist-local coords:
R_WRIST_FROM_FLU: np.ndarray = np.array(
    [[-1.0,  0.0,  0.0],   # F
     [ 0.0,  0.0,  1.0],   # L
     [ 0.0,  1.0,  0.0]],  # U
    dtype=np.float64,
)

assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)

# SMPL right wrist joint index (same as nymeria_wrist_loader_FLU.py)
_SMPL_R_WRIST = 21


# ---------------------------------------------------------------------------
# Geometric arm-chain helper (kept for external / visualisation use)
# ---------------------------------------------------------------------------

def _build_wrist_rotation(
    shoulder: np.ndarray,   # [N, 3]
    elbow:    np.ndarray,   # [N, 3]
    wrist:    np.ndarray,   # [N, 3]
) -> np.ndarray:
    """Geometric wrist frame from arm-chain joint positions [N, 3, 3].

    Not used in the FLU IMU derivation pipeline (which uses SMPL FK rotation
    matrices from _load_wrist_rotmat_20hz). Kept for external / visualisation use.
    """
    eps = 1e-8
    z = wrist - elbow
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + eps)
    upper = elbow - shoulder
    upper = upper / (np.linalg.norm(upper, axis=-1, keepdims=True) + eps)
    x = np.cross(z, upper)
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y, axis=-1, keepdims=True) + eps)
    return np.stack([x, y, z], axis=-1).astype(np.float64)


# ---------------------------------------------------------------------------
# SMPL FK helpers — load wrist rotation matrices from AMASS data
# ---------------------------------------------------------------------------

def _rot_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping unit vector a → b (Rodrigues formula)."""
    a = a / np.linalg.norm(a);  b = b / np.linalg.norm(b)
    v = np.cross(a, b);  s = np.linalg.norm(v);  c = float(a @ b)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 - c) / (s * s)


# Module-level AMASS clip index cache  {root: {"clip.npy": (npz_path, s, e)}}
_amass_index_cache: Dict[str, Dict[str, Tuple[str, int, int]]] = {}


def _get_amass_index(root: str) -> Dict[str, Tuple[str, int, int]]:
    """Build (once per process) and return the AMASS clip index for this root."""
    if root in _amass_index_cache:
        return _amass_index_cache[root]
    index: Dict[str, Tuple[str, int, int]] = {}
    index_csv = pjoin(root, "index.csv")
    if os.path.exists(index_csv):
        with open(index_csv) as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) != 4:
                    continue
                src, s, e, name = parts
                try:
                    s, e = int(s), int(e)
                except ValueError:
                    continue
                pose_npy  = pjoin(root, src.lstrip("./ "))
                amass_npz = (pose_npy
                             .replace("/pose_data/", "/amass_data/")
                             .replace(".npy", ".npz"))
                index[name] = (amass_npz, s, e)
    _amass_index_cache[root] = index
    return index


# Lazy SMPL body model singleton (loaded once per process / DataLoader worker)
_smpl_body_model = None


def _get_smpl_body_model():
    global _smpl_body_model
    if _smpl_body_model is None:
        import torch
        import core.articulate as art
        from core.paths import Paths
        _smpl_body_model = art.model.ParametricModel(
            str(Paths.SMPL_FILE), device=torch.device("cpu")
        )
    return _smpl_body_model


def _resample_rotmat(R: np.ndarray, N_out: int) -> np.ndarray:
    """Resample [N_in, 3, 3] rotation matrices to [N_out, 3, 3] via SLERP."""
    from scipy.spatial.transform import Rotation as ScipyRot, Slerp
    N_in = R.shape[0]
    if N_in == N_out:
        return R
    t_in  = np.linspace(0.0, 1.0, N_in)
    t_out = np.linspace(0.0, 1.0, N_out)
    return Slerp(t_in, ScipyRot.from_matrix(R))(t_out).as_matrix().astype(np.float64)


def _load_wrist_rotmat_20hz(
    amass_path: str,
    start_f:    int,
    end_f:      int,
    target_N:   int,
) -> Optional[np.ndarray]:
    """Run SMPL FK on an AMASS clip; return canonical wrist rotation [N, 3, 3] at 20 Hz.

    Pipeline (mirrors nymeria_wrist_loader_FLU.py steps 2-4):
      1. Load AMASS poses, resample to 20 Hz, slice to [start_f : end_f].
      2. SMPL FK (no root translation) → pose_global[:, 21].
      3. Apply canonical Y-rotation (face +Z) matching HumanML3D.
      4. Return [min(N_amass, target_N), 3, 3].
    """
    if not os.path.exists(amass_path):
        return None
    try:
        d = np.load(amass_path)
    except Exception:
        return None

    import torch
    from core.articulate.math import axis_angle_to_rotation_matrix

    a_fps   = float(d["mocap_framerate"])
    stride  = max(1, int(round(a_fps / _HML3D_FPS)))
    poses20 = d["poses"][::stride]
    end     = end_f if end_f >= 0 else len(poses20)
    poses_c = poses20[start_f:end].astype(np.float32)
    N       = min(poses_c.shape[0], target_N)
    if N < 3:
        return None
    poses_c = poses_c[:N]

    body    = _get_smpl_body_model()
    pose_aa = torch.from_numpy(poses_c[:, :72].reshape(N, 24, 3))
    R_local = axis_angle_to_rotation_matrix(pose_aa.reshape(-1, 3)).reshape(N, 24, 3, 3)

    # Global rotations from FK
    R_global = body.forward_kinematics_R(R_local).detach().cpu().numpy()   # [N, 24, 3, 3]

    # Canonical Y-rotation: make person face +Z at frame 0 (same as HumanML3D)
    _, jnt_t = body.forward_kinematics_masked(R_local, tran=None, calc_mesh=False)
    jnt = jnt_t.detach().cpu().numpy()   # [N, 24, 3]
    r_hip = jnt[0, 2];  l_hip = jnt[0, 1]
    r_sdr = jnt[0, 17]; l_sdr = jnt[0, 16]
    across  = (r_hip - l_hip) + (r_sdr - l_sdr);  across[1] = 0.
    forward = np.cross(np.array([0., 1., 0.]), across)
    R_y     = _rot_a_to_b(forward, np.array([0., 0., 1.]))

    # R_canon = R_y @ R_world_from_wrist
    R_wrist = R_global[:, _SMPL_R_WRIST]                   # [N, 3, 3]
    return np.einsum("ij,njk->nik", R_y, R_wrist)          # [N, 3, 3]


# ---------------------------------------------------------------------------
# Core IMU derivation — FLU frame via SMPL FK (mirrors nymeria_wrist_loader_FLU.py)
# ---------------------------------------------------------------------------

def _joints_to_imu_flu(
    joints:        np.ndarray,   # [N, 22, 3]  canonical joint positions at target fps
    R_wrist_world: np.ndarray,   # [N, 3, 3]   SMPL FK wrist rotation at target fps
    fps:           float = _TARGET_FPS,
) -> np.ndarray:
    """Derive right-wrist IMU in the FLU body frame.

    Follows nymeria_wrist_loader_FLU.py exactly:

    1.  R_world_from_X(t) = R_wrist_world(t) @ R_WRIST_FROM_FLU
    2.  Linear acc from second derivative of canonical wrist position (from joints.npy).
    3.  Angular velocity: SO3 log of consecutive R_wrist_world, rotated to world.
    4.  Project both into FLU via R_X_from_world(t).

    Args:
        joints:        [N, 22, 3]  canonical joint positions (target fps)
        R_wrist_world: [N, 3, 3]   SMPL FK global wrist rotation (canonical, target fps)
        fps:           frame rate (default _TARGET_FPS = 50 Hz)

    Returns:
        [N, 6]  float32 — (acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U)
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    N     = joints.shape[0]
    wrist = joints[:, _R_WRIST].astype(np.float64)   # [N, 3]

    # ── Step 1: FLU frame in world ──────────────────────────────────────────
    R_world_from_X = np.einsum("nij,jk->nik", R_wrist_world, R_WRIST_FROM_FLU)  # [N, 3, 3]
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)                            # [N, 3, 3]

    # ── Step 2: world-frame linear acceleration of wrist joint ──────────────
    a_world = np.zeros((N, 3), dtype=np.float64)
    a_world[1:-1] = (wrist[2:] - 2.0 * wrist[1:-1] + wrist[:-2]) * (fps ** 2)
    a_world[0]    = a_world[1]
    a_world[-1]   = a_world[-2]

    # ── Step 3: angular velocity in wrist local frame → world ───────────────
    dR = np.einsum(
        "nij,njk->nik",
        R_wrist_world[:-1].transpose(0, 2, 1),
        R_wrist_world[1:],
    )  # [N-1, 3, 3]
    omega_wl       = np.zeros((N, 3), dtype=np.float64)
    omega_wl[:-1]  = (ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
                      .as_rotvec().astype(np.float64) * fps)
    omega_wl[-1]   = omega_wl[-2]
    omega_world    = np.einsum("nij,nj->ni", R_wrist_world, omega_wl)  # [N, 3]

    # ── Step 4: project into FLU ────────────────────────────────────────────
    acc_flu  = np.einsum("nij,nj->ni", R_X_from_world, a_world).astype(np.float32)
    gyro_flu = np.einsum("nij,nj->ni", R_X_from_world, omega_world).astype(np.float32)

    return np.concatenate([acc_flu, gyro_flu], axis=1)   # [N, 6]


# ---------------------------------------------------------------------------
# Clip loader (FLU version)
# ---------------------------------------------------------------------------

def _load_clip_imu_flu(
    root:       str,
    clip_id:    str,
    joints_num: int,
    patch_size: int,
    T_min:      int,
    T_max:      int,
) -> Optional[Dict]:
    """Load one HumanML3D clip and derive 50 Hz FLU IMU.

    Returns a dict::

        {'imu':  np.ndarray [N, 6]  float32   — FLU frame
         'T':    int                            actual length (multiple of patch_size)
         'caps': list[str]}

    or None if the clip is invalid / too short.
    """
    joints_path = pjoin(root, 'joints', f'{clip_id}.npy')
    text_path   = pjoin(root, 'texts',  f'{clip_id}.txt')

    if not os.path.exists(joints_path):
        return None

    joints_raw  = np.load(joints_path, allow_pickle=True).astype(np.float32)  # [T_20hz, 52, 3]
    joints_raw  = joints_raw[:, :joints_num, :]          # [T_20hz, 22, 3]

    joints_50hz = _resample_joints(joints_raw)           # [N, 22, 3]
    N_raw       = joints_50hz.shape[0]

    T = _snap_T(N_raw, patch_size, T_min, T_max)
    if T is None:
        return None

    joints_50hz = joints_50hz[:T]                        # [T, 22, 3]

    # ── SMPL FK wrist rotation at 20 Hz via AMASS (same approach as nymeria) ─
    amass_index = _get_amass_index(root)
    amass_entry = amass_index.get(f"{clip_id}.npy")
    if amass_entry is None:
        return None
    amass_path_clip, start_f, end_f = amass_entry
    R_wrist_20hz = _load_wrist_rotmat_20hz(
        amass_path_clip, start_f, end_f, target_N=joints_raw.shape[0]
    )
    if R_wrist_20hz is None:
        return None

    # Resample wrist rotation matrices 20 Hz → 50 Hz (SLERP), truncate to T
    R_wrist_50hz = _resample_rotmat(R_wrist_20hz, N_raw)[:T]   # [T, 3, 3]

    # ── FLU IMU derivation via SMPL FK (matches nymeria_wrist_loader_FLU.py) ─
    imu = _joints_to_imu_flu(joints_50hz, R_wrist_50hz, fps=_TARGET_FPS)  # [T, 6]

    # ── captions ─────────────────────────────────────────────────────────
    captions: List[str] = []
    if os.path.exists(text_path):
        with open(text_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('#')
                if parts and parts[0]:
                    captions.append(parts[0])
    if not captions:
        captions = ['']

    return {'imu': imu, 'T': T, 'caps': captions}


# ---------------------------------------------------------------------------
# Map-style Dataset
# ---------------------------------------------------------------------------

class HumanML3DIMUDatasetFLU(Dataset):
    """Map-style Dataset: one item per HumanML3D clip, FLU wrist IMU.

    Drop-in replacement for ``HumanML3DIMUDataset`` with FLU-frame IMU.

    Returns ``(x, caption)`` where::

        x:       torch.Tensor [6, T]   float32  — FLU acc (3) + FLU gyro (3), 50 Hz
        caption: str                             — one randomly selected annotation

    Args:
        root:        HumanML3D workspace root.
        split:       ``'train'``, ``'val'``, ``'test'``, etc.
        patch_size:  ViT1D patch size (T must be a multiple; default 16).
        T_min:       Minimum T at 50 Hz (default 64 = 1.28 s).
        T_max:       Maximum T at 50 Hz (default 512 = 10.24 s).
        joints_num:  Leading SMPL joints to keep (default 22).
        seed:        RNG seed for caption sampling (default 42).
    """

    def __init__(
        self,
        root:       str,
        split:      str = 'train',
        patch_size: int = 16,
        T_min:      int = 64,
        T_max:      int = 512,
        joints_num: int = 22,
        seed:       int = 42,
    ):
        self.patch_size = patch_size
        self.T_min      = T_min
        self.T_max      = T_max
        self.rng        = pyrandom.Random(seed)

        split_file = pjoin(root, 'HumanML3D', f'{split}.txt')
        with open(split_file) as f:
            all_ids = [l.strip() for l in f if l.strip()]

        self._clips: List[Dict] = []
        skipped = 0
        for clip_id in all_ids:
            clip = _load_clip_imu_flu(root, clip_id, joints_num, patch_size, T_min, T_max)
            if clip is None:
                skipped += 1
                continue
            clip['id'] = clip_id
            self._clips.append(clip)

        logger.info(
            '[HumanML3DIMUDatasetFLU] split=%s  loaded=%d  skipped=%d',
            split, len(self._clips), skipped,
        )

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        clip    = self._clips[idx]
        imu     = clip['imu']   # [T, 6]  FLU frame
        caption = self.rng.choice(clip['caps'])

        # Per-window z-score (identical to nymeria_wrist_loader_FLU.py)
        mean = imu.mean(axis=0, keepdims=True)
        std  = imu.std(axis=0,  keepdims=True) + 1e-8
        imu  = (imu - mean) / std

        x = torch.from_numpy(imu.T).float()   # [6, T]
        return x, caption


# ---------------------------------------------------------------------------
# Iterable variable-length Dataset
# ---------------------------------------------------------------------------

class HumanML3DIMUVarLenDatasetFLU(IterableDataset):
    """Infinite IterableDataset with variable T per batch — FLU wrist IMU.

    Drop-in replacement for ``HumanML3DIMUVarLenDataset`` with FLU-frame IMU.
    API is identical to ``NymeriaVarLenMAEDataset`` and
    ``HumanML3DIMUVarLenDataset``.

    Each batch yields::

        (x [B, 6, T],  texts List[str])           # without text_emb_cache
        (x [B, 6, T],  embs  FloatTensor [B, D])  # with    text_emb_cache

    T is sampled uniformly from multiples of ``patch_size`` in [T_min, T_max]
    at each batch boundary.

    Args:
        root:           HumanML3D workspace root.
        split:          ``'train'``, ``'val'``, etc.
        batch_size:     Samples per batch.
        patch_size:     ViT1D patch size; T is always a multiple (default 16).
        T_min:          Minimum T at 50 Hz (default 64 = 1.28 s).
        T_max:          Maximum T at 50 Hz (default 512 = 10.24 s).
        joints_num:     Leading SMPL joints (default 22).
        seed:           Base RNG seed.
        text_emb_cache: Optional pre-computed caption → embedding dict.
    """

    def __init__(
        self,
        root:           str,
        split:          str  = 'train',
        batch_size:     int  = 32,
        patch_size:     int  = 16,
        T_min:          int  = 64,
        T_max:          int  = 512,
        joints_num:     int  = 22,
        seed:           int  = 42,
        text_emb_cache: Optional[Dict[str, np.ndarray]] = None,
    ):
        super().__init__()
        self.root           = root
        self.batch_size     = batch_size
        self.patch_size     = patch_size
        self.joints_num     = joints_num
        self.T_min          = T_min
        self.T_max          = T_max
        self.seed           = seed
        self._text_emb_cache = text_emb_cache

        if text_emb_cache is not None:
            _sample_emb    = next(iter(text_emb_cache.values()))
            self._emb_dim  = int(_sample_emb.shape[0])
            self._zero_emb = np.zeros(self._emb_dim, dtype=np.float32)
            logger.info(
                '[HumanML3DIMUVarLenDatasetFLU] text_emb_cache: %d captions, dim=%d',
                len(text_emb_cache), self._emb_dim,
            )
        else:
            self._emb_dim  = None
            self._zero_emb = None

        self.allowed_Ts: List[int] = list(range(T_min, T_max + 1, patch_size))
        if not self.allowed_Ts:
            raise ValueError(f'No valid T in [{T_min}, {T_max}] with patch_size={patch_size}')

        split_file = pjoin(root, 'HumanML3D', f'{split}.txt')
        with open(split_file) as f:
            all_ids = [l.strip() for l in f if l.strip()]

        self._clip_ids: List[str] = [
            cid for cid in all_ids
            if os.path.exists(pjoin(root, 'joints', f'{cid}.npy'))
        ]

        if not self._clip_ids:
            raise RuntimeError(
                f'HumanML3DIMUVarLenDatasetFLU: no joints/*.npy found under {root}'
            )

        logger.info(
            '[HumanML3DIMUVarLenDatasetFLU] split=%s  clips=%d  '
            'T=[%d,%d]  bs=%d  (lazy, FLU frame)',
            split, len(self._clip_ids), T_min, T_max, batch_size,
        )

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed + (worker_info.id if worker_info is not None else 0)
        rng         = np.random.default_rng(worker_seed)
        py_rng      = pyrandom.Random(int(worker_seed))

        # Worker-local clip cache (loaded once, reused for life of worker)
        cache: Dict[str, Optional[Dict]] = {}

        def get_clip(cid: str) -> Optional[Dict]:
            if cid not in cache:
                cache[cid] = _load_clip_imu_flu(
                    self.root, cid, self.joints_num,
                    self.patch_size, self.T_min, self.T_max,
                )
            return cache[cid]

        clip_ids = self._clip_ids

        while True:
            T = int(rng.choice(self.allowed_Ts))

            x_list:    List[np.ndarray] = []
            text_list: List[str]        = []
            attempts = 0

            while len(x_list) < self.batch_size:
                attempts += 1
                if attempts > self.batch_size * 50:
                    break

                cid  = clip_ids[int(rng.integers(len(clip_ids)))]
                clip = get_clip(cid)
                if clip is None:
                    continue

                imu    = clip['imu']       # [clip_T, 6]  FLU
                clip_T = imu.shape[0]
                cap    = py_rng.choice(clip['caps'])

                if clip_T >= T:
                    max_start = clip_T - T
                    start     = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                    window    = imu[start : start + T]    # [T, 6]

                    # Per-window z-score (same as nymeria_wrist_loader_FLU.py)
                    mean   = window.mean(axis=0, keepdims=True)
                    std    = window.std(axis=0,  keepdims=True) + 1e-8
                    window = (window - mean) / std
                else:
                    # Shorter than T: normalise then zero-pad tail
                    mean   = imu.mean(axis=0, keepdims=True)
                    std    = imu.std(axis=0,  keepdims=True) + 1e-8
                    normed = (imu - mean) / std
                    window = np.zeros((T, 6), dtype=np.float32)
                    window[:clip_T] = normed

                x_list.append(window.T)   # [6, T]
                text_list.append(cap)

            if len(x_list) < self.batch_size:
                continue   # retry

            x_tensor = torch.from_numpy(np.stack(x_list, axis=0)).float()  # [B, 6, T]

            if self._text_emb_cache is not None:
                emb_list = [
                    self._text_emb_cache.get(cap, self._zero_emb)
                    for cap in text_list
                ]
                yield x_tensor, torch.from_numpy(np.stack(emb_list, axis=0)).float()
            else:
                yield x_tensor, text_list


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_humanml3d_imu_loaders_flu(
    root:           str,
    cfg:            Any                              = None,
    patch_size:     int                              = 16,
    T_min:          int                              = 64,
    T_max:          int                              = 512,
    batch_size:     int                              = 32,
    num_workers:    int                              = 4,
    seed:           int                              = 42,
    text_emb_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Optional[DataLoader]]:
    """Build train and val DataLoaders for HumanML3D with FLU wrist IMU.

    Drop-in replacement for ``make_humanml3d_imu_loaders`` — API is identical.
    ``cfg`` key overrides are the same::

        vit1d.patch_size              (int)
        windowing.min_window_seconds  (float)
        windowing.max_window_seconds  (float)
        data.batch_size               (int)
        data.num_workers              (int)
        trainer.seed                  (int)

    Returns a dict with keys ``'train_loader'`` and ``'val_loader'``.
    DataLoaders use ``batch_size=None`` (the dataset pre-batches).
    """
    if cfg is not None:
        from imu_lm.utils.helpers import cfg_get
        vit_cfg = cfg_get(cfg, ['vit1d'],    {}) or {}
        win_cfg = cfg_get(cfg, ['windowing'], {}) or {}
        dat_cfg = cfg_get(cfg, ['data'],     {}) or {}
        patch_size  = int(vit_cfg.get('patch_size', patch_size))
        T_min       = int(round(float(win_cfg.get('min_window_seconds', T_min / _TARGET_FPS)) * _TARGET_FPS / patch_size)) * patch_size
        T_max       = int(round(float(win_cfg.get('max_window_seconds', T_max / _TARGET_FPS)) * _TARGET_FPS / patch_size)) * patch_size
        batch_size  = int(dat_cfg.get('batch_size', batch_size))
        num_workers = int(dat_cfg.get('num_workers', num_workers))
        seed        = int(cfg_get(cfg, ['trainer', 'seed'], seed))

    loaders: Dict[str, Optional[DataLoader]] = {}
    for split, split_seed in [('train', seed), ('val', seed + 1000)]:
        split_file = pjoin(root, 'HumanML3D', f'{split}.txt')
        if not os.path.exists(split_file):
            logger.warning(
                '[make_humanml3d_imu_loaders_flu] split file not found: %s', split_file
            )
            loaders[f'{split}_loader'] = None
            continue

        dataset = HumanML3DIMUVarLenDatasetFLU(
            root           = root,
            split          = split,
            batch_size     = batch_size,
            patch_size     = patch_size,
            T_min          = T_min,
            T_max          = T_max,
            seed           = split_seed,
            text_emb_cache = text_emb_cache,
        )
        loaders[f'{split}_loader'] = DataLoader(
            dataset,
            batch_size         = None,
            num_workers        = num_workers if split == 'train' else min(num_workers, 2),
            pin_memory         = True,
            persistent_workers = num_workers > 0,
        )

    return loaders


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    root = sys.argv[1] if len(sys.argv) > 1 else '.'

    ds = HumanML3DIMUDatasetFLU(root=root, split='test', patch_size=16,
                                 T_min=64, T_max=512)
    print(f'Clips loaded : {len(ds)}')
    x, cap = ds[0]
    print(f'x shape      : {x.shape}   (expect [6, T])')
    print(f'caption      : {cap!r}')
    print(f'acc_F  mean/std : {x[0].mean():.4f} / {x[0].std():.4f}')
    print(f'gyro_F mean/std : {x[3].mean():.4f} / {x[3].std():.4f}')
