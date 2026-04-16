"""
HumanML3D dataloaders — FLU wrist-IMU variant.

Identical to humanml3d_dataset.py **except** the synthetic right-wrist IMU is
expressed in the FLU (Forward-Left-Up) body frame, matching the convention of
``imu_lm/data/nymeria_wrist_loader_FLU.py``.

FLU frame definition (T-pose, right wrist, SMPL convention)
------------------------------------------------------------
The FLU frame X is attached to the right wrist and defined by:

    Forward (FLU x)  =  −x_smpl_wrist   (pointing toward thumb)
    Left    (FLU y)  =  +z_smpl_wrist   (pointing out of back of hand)
    Up      (FLU z)  =  +y_smpl_wrist   (pointing toward elbow)

which gives the constant T-pose rotation matrix::

    R_wrist_from_flu = [[-1, 0, 0],
                        [ 0, 0, 1],
                        [ 0, 1, 0]]

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
# FLU frame definition  (identical to nymeria_wrist_loader_FLU.py)
# ---------------------------------------------------------------------------
# The columns of R_WRIST_FROM_FLU are the FLU axes expressed in wrist-local coords:
#   col 0 = F axis = -x_smpl
#   col 1 = L axis = +z_smpl
#   col 2 = U axis = +y_smpl
R_WRIST_FROM_FLU: np.ndarray = np.array(
    [[-1.0, 0.0, 0.0],
     [ 0.0, 0.0, 1.0],
     [ 0.0, 1.0, 0.0]],
    dtype=np.float64,
)

# Sanity checks (same as nymeria_wrist_loader_FLU.py)
assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Core helper: build R_world_from_wrist from joint positions
# ---------------------------------------------------------------------------

def _build_wrist_rotation(
    shoulder: np.ndarray,   # [N, 3]
    elbow:    np.ndarray,   # [N, 3]
    wrist:    np.ndarray,   # [N, 3]
) -> np.ndarray:
    """Construct world-frame wrist rotation matrix from arm-chain geometry.

    Matches the local arm-frame used in humanml3d_dataset.py but returns the
    full [N, 3, 3] rotation matrix instead of just the axes:

        z_axis = normalize(wrist  - elbow)      # forearm direction (up in SMPL)
        x_axis = normalize(z × upper_arm)       # lateral
        y_axis = z × x                          # completes RH frame

    Returns
    -------
    R : np.ndarray [N, 3, 3]  — R_world_from_wrist per frame
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

    # columns: x, y, z  →  shape [N, 3, 3]
    R = np.stack([x, y, z], axis=-1).astype(np.float64)   # [N, 3, 3]
    return R


# ---------------------------------------------------------------------------
# Core IMU derivation — FLU frame
# ---------------------------------------------------------------------------

def _joints_to_imu_flu(
    joints: np.ndarray,
    fps: float = _TARGET_FPS,
) -> np.ndarray:
    """Derive right-wrist synthetic IMU in the FLU body frame.

    Replicates the nymeria_wrist_loader_FLU.py pipeline using joint positions
    instead of SMPL FK rotation matrices:

    1.  Build R_world_from_wrist(t) from arm-chain geometry.
    2.  Propagate FLU frame: R_world_from_X(t) = R_world_from_wrist(t) @ R_WRIST_FROM_FLU
    3.  World-frame linear acc of wrist origin.
    4.  Angular velocity: ω_wrist_local → ω_world via R_world_from_wrist.
    5.  Project both into FLU via R_X_from_world(t).

    Args:
        joints: [N, 22, 3]  joint positions (world frame, already at target fps)
        fps:    frame rate (should equal _TARGET_FPS = 50 Hz)

    Returns:
        X: [N, 6]  float32  — (acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U)
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    N = joints.shape[0]

    shoulder = joints[:, _R_SHOULDER].astype(np.float64)  # [N, 3]
    elbow    = joints[:, _R_ELBOW   ].astype(np.float64)  # [N, 3]
    wrist    = joints[:, _R_WRIST   ].astype(np.float64)  # [N, 3]

    # ── Step 1: R_world_from_wrist  [N, 3, 3] ──────────────────────────────
    R_world_from_wrist = _build_wrist_rotation(shoulder, elbow, wrist)  # [N, 3, 3]

    # ── Step 2: propagate FLU frame to world ────────────────────────────────
    # R_world_from_X(t) = R_world_from_wrist(t) @ R_WRIST_FROM_FLU
    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_wrist, R_WRIST_FROM_FLU)  # [N, 3, 3]
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)                                # [N, 3, 3]

    # ── Step 3: world-frame linear acceleration of wrist ────────────────────
    a_world = np.zeros((N, 3), dtype=np.float64)
    a_world[1:-1] = (wrist[2:] - 2.0 * wrist[1:-1] + wrist[:-2]) * (fps ** 2)
    a_world[0]    = a_world[1]
    a_world[-1]   = a_world[-2]

    # ── Step 4: angular velocity in wrist local frame → world ───────────────
    # dR[t] = R_world_from_wrist[t]^T @ R_world_from_wrist[t+1]
    dR = np.einsum(
        "nij,njk->nik",
        R_world_from_wrist[:-1].transpose(0, 2, 1),
        R_world_from_wrist[1:],
    )  # [N-1, 3, 3]

    omega_wrist_local = (
        ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
        .as_rotvec()
        .astype(np.float64) * fps
    ).reshape(N - 1, 3)  # [N-1, 3]

    # Pad to length N (copy last value)
    omega_wl = np.zeros((N, 3), dtype=np.float64)
    omega_wl[:-1] = omega_wrist_local
    omega_wl[-1]  = omega_wrist_local[-1]

    # Rotate to world frame
    omega_world = np.einsum("nij,nj->ni", R_world_from_wrist, omega_wl)  # [N, 3]

    # ── Step 5: project into FLU ────────────────────────────────────────────
    acc_flu  = np.einsum("nij,nj->ni", R_X_from_world, a_world).astype(np.float32)   # [N, 3]
    gyro_flu = np.einsum("nij,nj->ni", R_X_from_world, omega_world).astype(np.float32)  # [N, 3]

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

    # ── FLU IMU derivation ────────────────────────────────────────────────
    imu = _joints_to_imu_flu(joints_50hz, fps=_TARGET_FPS)  # [T, 6]

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
