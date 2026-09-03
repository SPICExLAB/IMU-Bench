"""
HumanML3D dataloaders.

Two loaders are provided:

1. HumanML3DTestDataset / get_dataloader  (original)
   Joint-position level; returns raw motion tensors + text.

2. HumanML3DIMUDataset / HumanML3DIMUVarLenDataset / make_humanml3d_imu_loaders  (new)
   Derives a synthetic right-wrist IMU from SMPL joint positions (identical
   derivation to imu_lm/data/nymeria_wrist_loader.py) and pairs it with the
   clip's text caption.  Window lengths are always multiples of the ViT1D
   patch_size and lie in [T_min, T_max] — exactly the constraint of the
   vit1d_mae_nymeria_var_length pipeline.

   IMU channels  →  [6, T]:
     acc  (3)  — second-order finite difference of right-wrist world position × fps²
     gyro (3)  — angular velocity from consecutive local arm-frame rotation matrices
                 (same SO3 log-map formula as nymeria_wrist_loader.py)

   Source joints (SMPL indexing, first 22):
     r_shoulder = 17,  r_elbow = 19,  r_wrist = 21

   HumanML3D is stored at 20 fps; the IMU is resampled to 50 Hz before
   windowing (matching the Nymeria IMU sampling rate).

   Text label  →  one caption string randomly drawn from the clip annotations.

Each sample contains:
  - motion:   np.ndarray [T, 22, 3]  SMPL body joints (pelvis → wrists),
                                      coord-flipped, Z-up, 20 fps
  - captions: list[str]               all text descriptions for this clip
  - tokens:   list[str]               POS-tagged token strings for each caption
  - times:    list[tuple[float,float]]  (start, end) time offsets per caption
  - name:     str                     clip id, e.g. "004822"
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

logger = logging.getLogger(__name__)


# ── joint layout ──────────────────────────────────────────────────────────────
SMPL_JOINT_NAMES = [
    'pelvis',    'l_hip',    'r_hip',    'spine1',
    'l_knee',    'r_knee',   'spine2',   'l_ankle',
    'r_ankle',   'spine3',   'l_foot',   'r_foot',
    'neck',      'l_collar', 'r_collar', 'head',
    'l_shoulder','r_shoulder','l_elbow', 'r_elbow',
    'l_wrist',   'r_wrist',
]  # indices 0-21 (first 22 of SMPLH)


# ── dataset ───────────────────────────────────────────────────────────────────
class HumanML3DTestDataset(Dataset):
    """
    Parameters
    ----------
    root : str
        Workspace root, i.e. the folder that contains ``joints/``,
        ``texts/``, and ``HumanML3D/``.
    split : str
        One of ``'test'``, ``'train'``, ``'val'``, ``'train_val'``, ``'all'``.
    joints_num : int
        How many leading SMPL joints to keep (default 22, body only).
    min_frames : int | None
        Drop clips shorter than this many frames (None = keep all).
    max_frames : int | None
        Truncate clips longer than this many frames (None = keep all).
    """

    def __init__(
        self,
        root: str = '.',
        split: str = 'test',
        joints_num: int = 22,
        min_frames: int | None = 40,
        max_frames: int | None = 196,
    ):
        self.root = root
        self.joints_num = joints_num
        self.min_frames = min_frames
        self.max_frames = max_frames

        split_file = pjoin(root, 'HumanML3D', f'{split}.txt')
        with open(split_file) as f:
            all_ids = [line.strip() for line in f if line.strip()]

        # keep only clips whose joints file actually exists
        self.ids = [
            id_ for id_ in all_ids
            if os.path.exists(pjoin(root, 'joints', f'{id_}.npy'))
        ]

        missing = len(all_ids) - len(self.ids)
        if missing:
            print(f'[HumanML3DTestDataset] {missing}/{len(all_ids)} clips '
                  f'skipped (joints file not found). '
                  f'Run Stage 2 of raw_pose_processing.ipynb first.')

    # ── helpers ---------------------------------------------------------------
    def _load_motion(self, id_: str) -> np.ndarray:
        """Load [T, joints_num, 3] joint positions."""
        path = pjoin(self.root, 'joints', f'{id_}.npy')
        data = np.load(path).astype(np.float32)          # [T, 52, 3]
        data = data[:, :self.joints_num, :]               # [T, 22, 3]

        # optional length clipping
        if self.min_frames is not None and len(data) < self.min_frames:
            return None
        if self.max_frames is not None:
            data = data[:self.max_frames]
        return data

    @staticmethod
    def _load_text(path: str):
        """
        Parse a text annotation file.

        Returns
        -------
        captions : list[str]
        tokens   : list[str]   (raw 'word/POS word/POS …' strings)
        times    : list[tuple[float, float]]
        """
        captions, tokens, times = [], [], []
        if not os.path.exists(path):
            return captions, tokens, times
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('#')
                if len(parts) < 4:
                    continue
                caption, tok, start, end = parts[0], parts[1], parts[2], parts[3]
                captions.append(caption)
                tokens.append(tok)
                times.append((float(start), float(end)))
        return captions, tokens, times

    # ── dataset API ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        id_ = self.ids[idx]

        motion = self._load_motion(id_)
        # skip too-short clips at __getitem__ time (rare edge case)
        if motion is None:
            motion = np.zeros((self.min_frames, self.joints_num, 3),
                              dtype=np.float32)

        text_path = pjoin(self.root, 'texts', f'{id_}.txt')
        captions, token_strs, times = self._load_text(text_path)

        return {
            'name':     id_,
            'motion':   torch.from_numpy(motion),   # [T, 22, 3]
            'n_frames': motion.shape[0],
            'captions': captions,                   # list[str]
            'tokens':   token_strs,                 # list[str]
            'times':    times,                      # list[(float, float)]
        }


# ── collate ───────────────────────────────────────────────────────────────────
def collate_fn(batch: list[dict]) -> dict:
    """
    Pad variable-length motions to the longest sequence in the batch.

    Returns
    -------
    names    : list[str]                        length B
    motions  : FloatTensor [B, T_max, 22, 3]    zero-padded
    lengths  : LongTensor  [B]                  actual frame counts
    captions : list[list[str]]                  length B
    tokens   : list[list[str]]                  length B
    times    : list[list[tuple]]                length B
    """
    lengths = torch.tensor([s['n_frames'] for s in batch], dtype=torch.long)
    t_max = int(lengths.max().item())
    J = batch[0]['motion'].shape[1]

    motions = torch.zeros(len(batch), t_max, J, 3, dtype=torch.float32)
    for i, sample in enumerate(batch):
        t = sample['n_frames']
        motions[i, :t] = sample['motion']

    return {
        'names':    [s['name']     for s in batch],
        'motions':  motions,
        'lengths':  lengths,
        'captions': [s['captions'] for s in batch],
        'tokens':   [s['tokens']   for s in batch],
        'times':    [s['times']    for s in batch],
    }


# ── factory ───────────────────────────────────────────────────────────────────
def get_dataloader(
    root: str = '.',
    split: str = 'test',
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    joints_num: int = 22,
    min_frames: int | None = 40,
    max_frames: int | None = 196,
) -> DataLoader:
    dataset = HumanML3DTestDataset(
        root=root,
        split=split,
        joints_num=joints_num,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


# ── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    loader = get_dataloader(root=root, split='test', batch_size=4,
                            num_workers=0)

    print(f'Dataset size : {len(loader.dataset)}')
    batch = next(iter(loader))

    print(f'names        : {batch["names"]}')
    print(f'motions      : {batch["motions"].shape}')   # [4, T_max, 22, 3]
    print(f'lengths      : {batch["lengths"]}')
    print(f'captions[0]  : {batch["captions"][0]}')
    print(f'tokens[0][0] : {batch["tokens"][0][0] if batch["tokens"][0] else ""}')
    print(f'times[0]     : {batch["times"][0]}')


# =============================================================================
# IMU + Text dataloader
# =============================================================================

# Joint indices within the 22-joint SMPL layout used by HumanML3D
_R_SHOULDER = 17
_R_ELBOW    = 19
_R_WRIST    = 21

# HumanML3D native frame rate
_HML3D_FPS = 20.0

# Target IMU frame rate (matches Nymeria loader)
_TARGET_FPS = 50.0


# --------------------------------------------------------------------------- #
# Helper: derive synthetic IMU from [N, 22, 3] joint positions at 50 Hz       #
# --------------------------------------------------------------------------- #

def _joints_to_imu(joints: np.ndarray, fps: float = _TARGET_FPS) -> np.ndarray:
    """Derive right-wrist synthetic IMU from SMPL joint positions.

    Replicates the exact derivation in nymeria_wrist_loader.py:

    acc  = second-order finite difference of r_wrist position × fps²
    gyro = SO3 log of consecutive local arm-frame rotation matrices × fps
           (angular velocity in the sensor/body frame)

    Local arm frame built from r_shoulder (17), r_elbow (19), r_wrist (21):
        z_axis = normalize(r_wrist  - r_elbow)   # forearm direction
        x_axis = normalize(z_axis × (r_elbow - r_shoulder))  # lateral
        y_axis = z_axis × x_axis                 # completes RH frame

    Args:
        joints: [N, 22, 3]  joint positions (world frame, 50 Hz after upsampling)
        fps:    frame rate of ``joints`` (should be _TARGET_FPS after resample)

    Returns:
        X: [N, 6]  float32  — (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    N = joints.shape[0]

    # ── acceleration --------------------------------------------------------
    pos = joints[:, _R_WRIST, :]                  # [N, 3] world-frame position
    acc = np.zeros((N, 3), dtype=np.float32)
    acc[1:-1] = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) * (fps ** 2)
    acc[0]    = acc[1]
    acc[-1]   = acc[-2]

    # ── local arm frame → gyroscope -----------------------------------------
    shoulder = joints[:, _R_SHOULDER, :]          # [N, 3]
    elbow    = joints[:, _R_ELBOW,    :]          # [N, 3]
    wrist    = joints[:, _R_WRIST,    :]          # [N, 3]

    # forearm direction  (z axis)
    z = wrist - elbow
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    # lateral axis  (x = z × upper_arm)
    upper = elbow - shoulder
    upper = upper / (np.linalg.norm(upper, axis=-1, keepdims=True) + 1e-8)
    x = np.cross(z, upper)                        # [N, 3]
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    # y completes right-handed frame
    y = np.cross(z, x)                            # [N, 3]
    y = y / (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)

    # rotation matrices  [N, 3, 3]  (columns = x, y, z)
    R = np.stack([x, y, z], axis=-1).astype(np.float32)

    # angular velocity in body frame: dR[t] = R[t].T @ R[t+1]
    dR       = np.einsum("nij,njk->nik", R[:-1].transpose(0, 2, 1), R[1:])  # [N-1, 3, 3]
    gyro_raw = ScipyRot.from_matrix(dR.reshape(-1, 3, 3)).as_rotvec()        # [N-1, 3]
    gyro_raw = gyro_raw.reshape(N - 1, 3).astype(np.float32) * fps
    gyro     = np.zeros((N, 3), dtype=np.float32)
    gyro[:-1] = gyro_raw
    gyro[-1]  = gyro_raw[-1]

    return np.concatenate([acc, gyro], axis=1)    # [N, 6]


def _resample_joints(joints_20hz: np.ndarray, method: str = 'linear') -> np.ndarray:
    """Upsample [N, 22, 3] from 20 Hz to 50 Hz.

    method: 'linear' (default, original behavior) | 'cubic'.

    Bug fix (2026-07-25): linear interpolation has piecewise-constant
    velocity between the original 20Hz sample points -- its second
    derivative (acceleration, computed downstream via finite difference on
    the upsampled trajectory) is a series of artificial spikes exactly at
    each original 20Hz breakpoint, a pure numerical artifact of the
    interpolation method, not real motion. Confirmed empirically: switching
    to 'cubic' drops a 2nd-difference "roughness" proxy on real clips from
    ~14-22 down to ~0.2-0.7 (a 20-40x reduction) while leaving the overall
    acceleration MAGNITUDE essentially unchanged -- i.e. it removes a
    spurious artifact, not real signal. For comparison, real accelerometer
    data (WISDM/PAMAP2) shows this same roughness proxy at ~0.8-2.4 -- 'cubic'
    brings HumanML3D's synthetic acc in line with real sensor characteristics
    for the first time. Kept opt-in (default 'linear') so this doesn't
    silently change behavior for every other config already built on the
    original resampling.
    """
    from scipy.interpolate import interp1d

    N_orig  = joints_20hz.shape[0]
    dur     = (N_orig - 1) / _HML3D_FPS              # seconds
    t_orig  = np.linspace(0.0, dur, N_orig)
    t_new   = np.arange(0.0, dur + 1e-9, 1.0 / _TARGET_FPS)

    # clamp to avoid fp rounding beyond the original range
    t_new   = t_new[t_new <= dur]

    kind = 'cubic' if (method == 'cubic' and N_orig > 3) else 'linear'
    interp  = interp1d(t_orig, joints_20hz, axis=0, kind=kind, fill_value='extrapolate')
    return interp(t_new).astype(np.float32)           # [N_new, 22, 3]


def _snap_T(T_raw: int, patch_size: int, T_min: int, T_max: int) -> Optional[int]:
    """Snap T_raw to the nearest valid multiple of patch_size within [T_min, T_max].

    Returns None if T_raw < T_min (clip too short — should be skipped).
    """
    if T_raw < T_min:
        return None
    T_snapped = (min(T_raw, T_max) // patch_size) * patch_size
    return max(T_snapped, patch_size)


# --------------------------------------------------------------------------- #
# Session cache: [N, 6] IMU + [N, 22, 3] joints + captions list              #
# --------------------------------------------------------------------------- #

def _load_clip_imu(
    root: str,
    clip_id: str,
    joints_num: int,
    patch_size: int,
    T_min: int,
    T_max: int,
) -> Optional[Dict]:
    """Load one HumanML3D clip and derive 50 Hz IMU.

    Returns a dict::

        {'imu':  np.ndarray [N, 6]  float32
         'T':    int                 actual length N (multiple of patch_size)
         'caps': list[str]           non-empty caption strings}

    or None if the clip is invalid / too short.
    """
    joints_path = pjoin(root, 'joints', f'{clip_id}.npy')
    text_path   = pjoin(root, 'texts',  f'{clip_id}.txt')

    if not os.path.exists(joints_path):
        return None

    joints_raw = np.load(joints_path, allow_pickle=True).astype(np.float32)   # [T_20hz, 52, 3]
    joints_raw = joints_raw[:, :joints_num, :]             # [T_20hz, 22, 3]

    # Upsample to 50 Hz
    joints_50hz = _resample_joints(joints_raw)             # [N, 22, 3]
    N_raw       = joints_50hz.shape[0]

    T = _snap_T(N_raw, patch_size, T_min, T_max)
    if T is None:
        return None                                         # clip too short

    joints_50hz = joints_50hz[:T]                          # [T, 22, 3]

    # Derive IMU
    imu = _joints_to_imu(joints_50hz, fps=_TARGET_FPS)    # [T, 6]

    # Load captions
    captions: List[str] = []
    if os.path.exists(text_path):
        with open(text_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('#')
                if parts and parts[0]:
                    captions.append(parts[0])

    if not captions:
        captions = ['']   # fallback: empty caption

    return {'imu': imu, 'T': T, 'caps': captions}


# --------------------------------------------------------------------------- #
# Map-style Dataset (one sample = one clip window, T varies per clip)         #
# --------------------------------------------------------------------------- #

class HumanML3DIMUDataset(Dataset):
    """Map-style Dataset: one item per HumanML3D clip.

    Returns ``(x, caption)`` where::

        x:       torch.Tensor [6, T]   float32  — acc (3) + gyro (3) at 50 Hz
                                                   T is a multiple of ``patch_size``
                                                   in [T_min, T_max]
        caption: str                   — one randomly selected text annotation

    Window length T varies per clip (clips have different durations), so the
    default collate_fn pads x to the longest T in each batch.  Use
    ``imu_text_collate_fn`` below, or let DataLoader default-collate (which
    will raise an error on mismatched shapes) — pass ``collate_fn=imu_text_collate_fn``.

    Args:
        root:        HumanML3D workspace root (contains ``joints/`` and ``texts/``).
        split:       ``'train'``, ``'val'``, ``'test'``, ``'train_val'``, ``'all'``.
        patch_size:  ViT1D patch size (T must be a multiple; default 16).
        T_min:       Minimum T in frames at 50 Hz (default 64 = 1.28 s).
        T_max:       Maximum T in frames at 50 Hz (default 512 = 10.24 s).
        joints_num:  Leading SMPL joints to keep (default 22).
        seed:        RNG seed for caption sampling (default 42).
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        patch_size: int = 16,
        T_min: int = 64,
        T_max: int = 512,
        joints_num: int = 22,
        seed: int = 42,
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
            clip = _load_clip_imu(root, clip_id, joints_num, patch_size, T_min, T_max)
            if clip is None:
                skipped += 1
                continue
            clip['id'] = clip_id
            self._clips.append(clip)

        logger.info(
            '[HumanML3DIMUDataset] split=%s  loaded=%d  skipped=%d',
            split, len(self._clips), skipped,
        )

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        clip    = self._clips[idx]
        imu     = clip['imu']       # [T, 6]
        caption = self.rng.choice(clip['caps'])

        # z-score normalise per-window per-channel (no cfg dependency)
        mean = imu.mean(axis=0, keepdims=True)
        std  = imu.std(axis=0, keepdims=True) + 1e-8
        imu  = (imu - mean) / std

        x = torch.from_numpy(imu.T).float()   # [6, T]
        return x, caption


def imu_text_collate_fn(
    batch: List[Tuple[torch.Tensor, str]],
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Pad variable-length IMU windows to the longest T in the batch.

    Returns:
        x:       FloatTensor [B, 6, T_max]   zero-padded
        lengths: LongTensor  [B]             actual T per sample
        texts:   list[str]                   captions
    """
    xs, texts = zip(*batch)
    lengths   = torch.tensor([x.shape[-1] for x in xs], dtype=torch.long)
    T_max     = int(lengths.max().item())

    x_padded  = torch.zeros(len(xs), 6, T_max, dtype=torch.float32)
    for i, x in enumerate(xs):
        x_padded[i, :, :x.shape[-1]] = x

    return x_padded, lengths, list(texts)


# --------------------------------------------------------------------------- #
# IterableDataset (variable T per batch — matches NymeriaVarLenMAEDataset)    #
# --------------------------------------------------------------------------- #

class HumanML3DIMUVarLenDataset(IterableDataset):
    """Infinite IterableDataset yielding full IMU+text batches with variable T.

    API mirrors ``NymeriaVarLenMAEDataset`` so the two datasets are
    interchangeable in the training loop.

    When ``text_emb_cache`` is *not* supplied each batch is::

        (x [B, 6, T],  texts List[str])

    When ``text_emb_cache`` is supplied the pre-computed L2-normalised text
    embeddings are looked up at yield time and each batch becomes::

        (x [B, 6, T],  embs FloatTensor [B, D_clip])

    Using the cache removes the CLIP text encoder from the training-time
    critical path entirely — it is never loaded into GPU memory.

    T is sampled uniformly from multiples of ``patch_size`` in [T_min, T_max]
    at each batch boundary.  Clips shorter than T are skipped; clips longer are
    randomly cropped to T.

    Use with ``DataLoader(batch_size=None)`` so the pre-batched tensors pass
    through unchanged.

    Args:
        root:           HumanML3D workspace root.
        split:          ``'train'``, ``'val'``, etc.
        batch_size:     Number of (imu, text) pairs per yielded batch.
        patch_size:     ViT1D patch size; T is always a multiple (default 16).
        T_min:          Minimum T in frames at 50 Hz (default 64 = 1.28 s).
        T_max:          Maximum T in frames at 50 Hz (default 512 = 10.24 s).
        joints_num:     Leading SMPL joints (default 22).
        seed:           Base RNG seed.
        text_emb_cache: Optional dict mapping caption strings to pre-computed
                        L2-normalised CLIP embeddings (``np.ndarray [D_clip]``).
                        Produced by ``scripts/cache_clip_text_embeddings.py``.
                        When supplied the dataset yields embedding tensors
                        instead of strings and the CLIP text model is not needed
                        during training.
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        batch_size: int = 32,
        patch_size: int = 16,
        T_min: int = 64,
        T_max: int = 512,
        joints_num: int = 22,
        seed: int = 42,
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
        self._text_emb_cache: Optional[Dict[str, np.ndarray]] = text_emb_cache

        # Determine embedding dimension from the cache (needed for the zero fallback).
        if text_emb_cache is not None:
            # Peek at any entry to get D_clip
            _sample_emb = next(iter(text_emb_cache.values()))
            self._emb_dim = int(_sample_emb.shape[0])
            self._zero_emb = np.zeros(self._emb_dim, dtype=np.float32)
            logger.info(
                '[HumanML3DIMUVarLenDataset] text_emb_cache loaded: %d captions, dim=%d',
                len(text_emb_cache), self._emb_dim,
            )
        else:
            self._emb_dim  = None
            self._zero_emb = None

        self.allowed_Ts: List[int] = list(range(T_min, T_max + 1, patch_size))
        if not self.allowed_Ts:
            raise ValueError(f'No valid T in [{T_min}, {T_max}] with patch_size={patch_size}')

        # Lazy init: only check file existence — no heavy numpy/scipy work at startup.
        # Clips are loaded on-demand inside __iter__ and cached per worker.
        split_file = pjoin(root, 'HumanML3D', f'{split}.txt')
        with open(split_file) as f:
            all_ids = [l.strip() for l in f if l.strip()]

        self._clip_ids: List[str] = [
            cid for cid in all_ids
            if os.path.exists(pjoin(root, 'joints', f'{cid}.npy'))
        ]

        if not self._clip_ids:
            raise RuntimeError(f'HumanML3DIMUVarLenDataset: no joints/*.npy files found under {root}')

        logger.info(
            '[HumanML3DIMUVarLenDataset] split=%s  candidate_clips=%d  '
            'T_range=[%d, %d]  batch_size=%d  (lazy loading)',
            split, len(self._clip_ids), T_min, T_max, batch_size,
        )

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, List[str]]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed + (worker_info.id if worker_info is not None else 0)
        rng         = np.random.default_rng(worker_seed)
        py_rng      = pyrandom.Random(int(worker_seed))

        # Worker-local cache: clip_id -> loaded dict (or None if invalid).
        # Clips are loaded once on first access and reused for the life of the worker.
        cache: Dict[str, Optional[Dict]] = {}

        def get_clip(cid: str) -> Optional[Dict]:
            if cid not in cache:
                cache[cid] = _load_clip_imu(
                    self.root, cid, self.joints_num,
                    self.patch_size, self.T_min, self.T_max,
                )
            return cache[cid]

        clip_ids = self._clip_ids

        while True:
            # Sample a random window length for this batch
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

                imu    = clip['imu']      # [clip_T, 6]
                clip_T = imu.shape[0]
                cap    = py_rng.choice(clip['caps'])

                if clip_T >= T:
                    # Random crop of length T
                    max_start = clip_T - T
                    start     = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                    window    = imu[start : start + T]         # [T, 6]

                    # Per-window z-score normalisation
                    mean   = window.mean(axis=0, keepdims=True)
                    std    = window.std(axis=0,  keepdims=True) + 1e-8
                    window = (window - mean) / std
                else:
                    # Clip is shorter than T: normalise real data, then zero-pad
                    mean   = imu.mean(axis=0, keepdims=True)
                    std    = imu.std(axis=0,  keepdims=True) + 1e-8
                    normed = (imu - mean) / std                # [clip_T, 6]
                    window = np.zeros((T, 6), dtype=np.float32)
                    window[:clip_T] = normed                   # tail is zero-padded

                x_list.append(window.T)   # [6, T]
                text_list.append(cap)

            if len(x_list) < self.batch_size:
                continue   # could not fill a full batch; retry with new T

            x_tensor = torch.from_numpy(np.stack(x_list, axis=0)).float()  # [B, 6, T]

            if self._text_emb_cache is not None:
                # Look up pre-computed embeddings; fall back to zero vector for
                # any caption that is missing from the cache (should not happen
                # in normal use once the cache is built with all splits).
                emb_list = [
                    self._text_emb_cache.get(cap, self._zero_emb)
                    for cap in text_list
                ]
                emb_tensor = torch.from_numpy(
                    np.stack(emb_list, axis=0)
                ).float()                         # [B, D_clip]
                yield x_tensor, emb_tensor
            else:
                yield x_tensor, text_list


# --------------------------------------------------------------------------- #
# Loader factory                                                               #
# --------------------------------------------------------------------------- #

def make_humanml3d_imu_loaders(
    root: str,
    cfg: Any = None,
    patch_size: int = 16,
    T_min: int = 64,
    T_max: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    text_emb_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Optional[DataLoader]]:
    """Build train and val DataLoaders for the HumanML3D IMU+text dataset.

    If ``cfg`` is supplied, the following keys override the defaults::

        vit1d.patch_size            (int,   default 16)
        windowing.min_window_seconds (float, default 1.28  → T_min = 64 at 50 Hz)
        windowing.max_window_seconds (float, default 10.24 → T_max = 512 at 50 Hz)
        data.batch_size             (int,   default 32)
        data.num_workers            (int,   default 4)
        trainer.seed                (int,   default 42)

    Args:
        text_emb_cache: Optional pre-computed text embedding cache produced by
            ``scripts/cache_clip_text_embeddings.py``
            (``dict[caption_str, np.ndarray[D_clip]]``).  When supplied the
            dataset yields ``(x [B,6,T], embs [B,D_clip])`` tensors so the CLIP
            text encoder is never invoked at training time.

    Returns a dict with keys ``'train_loader'`` and ``'val_loader'``.
    DataLoaders use ``batch_size=None`` (the dataset pre-batches) so they are
    drop-in replacements for ``NymeriaVarLenMAEDataset`` loaders.
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
            logger.warning('[make_humanml3d_imu_loaders] split file not found: %s', split_file)
            loaders[f'{split}_loader'] = None
            continue

        dataset = HumanML3DIMUVarLenDataset(
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
            batch_size        = None,
            num_workers       = num_workers if split == 'train' else min(num_workers, 2),
            pin_memory        = True,
            persistent_workers= num_workers > 0,
        )

    return loaders


# ── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    loader = get_dataloader(root=root, split='test', batch_size=4,
                            num_workers=0)

    print(f'Dataset size : {len(loader.dataset)}')
    batch = next(iter(loader))

    print(f'names        : {batch["names"]}')
    print(f'motions      : {batch["motions"].shape}')   # [4, T_max, 22, 3]
    print(f'lengths      : {batch["lengths"]}')
    print(f'captions[0]  : {batch["captions"][0]}')
    print(f'tokens[0][0] : {batch["tokens"][0][0] if batch["tokens"][0] else ""}')
    print(f'times[0]     : {batch["times"][0]}')
