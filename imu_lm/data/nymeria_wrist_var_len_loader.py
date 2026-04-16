"""NymeriaVarLenMAEDataset: variable-length IterableDataset for MAE pretraining.

Each training batch randomly picks a window length T from the range
[min_window_seconds, max_window_seconds] (rounded to the nearest multiple of
patch_size), then draws `batch_size` windows of that length from the cached
sessions.  This trains the model to handle variable-length inputs.

Batch output::

    x : [batch_size, 6, T]   float32  — acc + gyro, T varies per batch
    y : [batch_size, 1]      float32  — dummy (MAE is self-supervised; labels ignored)

Usage with DataLoader (batch_size=None disables further batching)::

    loaders = make_nymeria_var_len_loaders(cfg, train_files, val_files)
    for x, _ in loaders["train_loader"]:
        # x: [B, 6, T]  where T is sampled fresh every iteration
        ...

Preprocessing pipeline
-----------------------
impute → lowpass filter → z-score normalise → apply_augment
(identical to NymeriaWristDataset; per-window statistics so T-agnostic)
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.augmentations.transform import apply_augment
from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)

# Xsens right-wrist joint index (same as nymeria_wrist_loader.py)
_XSENS_RIGHT_WRIST_JOINT = 9

CANONICAL_FPS = 50.0


# ---------------------------------------------------------------------------
# Session loader (IMU only — no pose labels needed for MAE)
# ---------------------------------------------------------------------------

def _load_imu_session(pkl_path: str) -> Optional[np.ndarray]:
    """Load one Nymeria pkl and return X [N, 6] (acc+gyro) or None on failure.

    Derived from xsens_pose joint 9 (right wrist):
      acc  — second-order finite difference of world-frame position × fps²
      gyro — SO3 log of consecutive rotation matrices × fps  (body frame)
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    try:
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
    except Exception as exc:
        logger.warning("[var_len] failed to open %s: %s", pkl_path, exc)
        return None

    gt_data    = raw.get("gt_data", {})
    xsens_pose = gt_data.get("xsens_pose")
    if xsens_pose is None:
        logger.warning("[var_len] %s: no 'xsens_pose'; skipping.", pkl_path)
        return None

    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float32)  # [N, 23, 4, 4]

    N = xsens_pose.shape[0]
    if N < 3:
        logger.warning("[var_len] %s: too short (%d frames); skipping.", pkl_path, N)
        return None

    fps = float(
        raw.get("imu_sampling_rate_hz")
        or raw.get("pose_sampling_rate_hz")
        or CANONICAL_FPS
    )

    j   = _XSENS_RIGHT_WRIST_JOINT
    pos = xsens_pose[:, j, :3, 3]      # [N, 3] world-frame position
    rot = xsens_pose[:, j, :3, :3]     # [N, 3, 3] rotation matrices

    # Acceleration: second-order central difference × fps²
    acc         = np.zeros((N, 3), dtype=np.float32)
    acc[1:-1]   = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) * (fps ** 2)
    acc[0]      = acc[1]
    acc[-1]     = acc[-2]

    # Gyroscope: angular velocity in body frame
    dR       = np.einsum("nij,njk->nik", rot[:-1].transpose(0, 2, 1), rot[1:])
    gyro_raw = ScipyRot.from_matrix(dR.reshape(-1, 3, 3)).as_rotvec()
    gyro_raw = gyro_raw.reshape(N - 1, 3).astype(np.float32) * fps
    gyro     = np.zeros((N, 3), dtype=np.float32)
    gyro[:-1] = gyro_raw
    gyro[-1]  = gyro_raw[-1]

    X = np.concatenate([acc, gyro], axis=1)  # [N, 6]
    logger.info(
        "[var_len] loaded %s: N=%d  fps=%.0f",
        os.path.basename(pkl_path), N, fps,
    )
    return X


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NymeriaVarLenMAEDataset(IterableDataset):
    """Infinite IterableDataset yielding full batches with variable window length T.

    Each call to ``__iter__`` produces an infinite stream of
    ``(x [batch_size, 6, T], dummy_y [batch_size, 1])`` tuples where T is
    sampled uniformly from ``allowed_Ts`` at each batch boundary.

    Use with ``DataLoader(batch_size=None)`` so the DataLoader does not try to
    re-batch the already-batched output.

    Args:
        pkl_paths:   Absolute paths to Nymeria ``.pkl`` session files.
        batch_size:  Number of windows per yielded batch.
        T_min:       Minimum window length in frames (must be multiple of patch_size).
        T_max:       Maximum window length in frames (must be multiple of patch_size).
        patch_size:  ViT patch size; T is always sampled as a multiple of this.
        cfg:         Config dict (for preprocessing parameters).
        split_name:  For logging only.
        target_hz:   Not currently used (FPS read from pkl metadata).
        seed:        Base RNG seed; per-worker seed = seed + worker_id.
    """

    def __init__(
        self,
        pkl_paths: List[str],
        batch_size: int,
        T_min: int,
        T_max: int,
        patch_size: int,
        cfg: Any,
        split_name: str = "train",
        target_hz: int = 50,
        seed: int = 42,
    ):
        super().__init__()
        self.pkl_paths  = pkl_paths
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.cfg        = cfg
        self.split_name = split_name
        self.target_hz  = target_hz
        self.seed       = seed

        # Build the list of allowed T values: multiples of patch_size in [T_min, T_max]
        self.allowed_Ts: List[int] = list(range(T_min, T_max + 1, patch_size))
        if not self.allowed_Ts:
            raise ValueError(
                f"No valid window lengths between T_min={T_min} and T_max={T_max} "
                f"with patch_size={patch_size}."
            )

        logger.info(
            "[var_len] split=%s  sessions=%d  T range=[%d, %d]  "
            "allowed_Ts=%d  batch_size=%d",
            split_name,
            len(pkl_paths),
            T_min,
            T_max,
            len(self.allowed_Ts),
            batch_size,
        )

        # Eagerly load all sessions so workers share memory (fork-safe on Linux)
        self._sessions: List[np.ndarray] = []  # each entry: [N, 6] float32
        for path in tqdm(
            pkl_paths,
            desc=f"Loading {split_name} sessions (var-len MAE)",
            unit="session",
            dynamic_ncols=True,
        ):
            X = _load_imu_session(path)
            if X is not None:
                self._sessions.append(X)

        if not self._sessions:
            raise RuntimeError(
                f"NymeriaVarLenMAEDataset: no valid sessions loaded from "
                f"{len(pkl_paths)} pkl paths."
            )

        logger.info(
            "[var_len] split=%s  loaded %d sessions", split_name, len(self._sessions)
        )

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Infinite batch generator with per-batch random window length."""
        worker_info = torch.utils.data.get_worker_info()
        worker_seed  = self.seed + (worker_info.id if worker_info is not None else 0)
        rng          = np.random.default_rng(worker_seed)
        stats        = PreprocessStats()

        while True:
            # Sample a random window length for this batch
            T = int(rng.choice(self.allowed_Ts))

            # Filter sessions that are long enough for this T
            valid_sessions = [s for s in self._sessions if len(s) >= T]
            if not valid_sessions:
                continue

            x_list: List[np.ndarray] = []
            attempts = 0
            while len(x_list) < self.batch_size:
                attempts += 1
                if attempts > self.batch_size * 20:
                    # Safety valve: avoid infinite loop if preprocessing keeps
                    # dropping windows (e.g. too many NaN frames)
                    break

                sess_idx = int(rng.integers(len(valid_sessions)))
                sess     = valid_sessions[sess_idx]
                start    = int(rng.integers(0, len(sess) - T + 1))
                Xw       = sess[start : start + T]  # [T, 6]

                # Preprocessing: impute → filter → z-score; returns [C, T] or None
                Xproc = preprocess_window(Xw, self.cfg, stats)
                if Xproc is None:
                    Xproc = Xw.T.astype(np.float32)  # fallback: skip preprocessing

                # Augmentation: apply_augment expects [T, C] → returns [T, C]
                Xproc = apply_augment(Xproc.T, self.cfg).T  # [6, T]

                x_list.append(Xproc)

            if len(x_list) < self.batch_size:
                continue  # could not fill a full batch; try again with a new T

            # [B, 6, T] float32
            x_tensor  = torch.from_numpy(np.stack(x_list, axis=0)).float()
            dummy_y   = torch.zeros(self.batch_size, 1, dtype=torch.float32)
            yield x_tensor, dummy_y


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

def make_nymeria_var_len_loaders(
    cfg: Any,
    train_files: List[str],
    val_files: List[str],
) -> Dict[str, Optional[DataLoader]]:
    """Build train and val DataLoaders for variable-length MAE pretraining.

    Returns DataLoaders with ``batch_size=None`` so the IterableDataset's
    pre-batched tensors pass through unchanged.

    Window length range is read from config::

        windowing:
          min_window_seconds: 1.28   # → T_min = 64 frames at 50 Hz
          max_window_seconds: 10.24  # → T_max = 512 frames at 50 Hz

    Batch size is read from ``data.batch_size`` (default 32).
    """
    vit_cfg   = cfg_get(cfg, ["vit1d"],    {}) or {}
    win_cfg   = cfg_get(cfg, ["windowing"], {}) or {}
    data_cfg  = cfg_get(cfg, ["data"],     {}) or {}
    trainer_cfg = cfg_get(cfg, ["trainer"], {}) or {}

    patch_size = int(vit_cfg.get("patch_size", 16))
    fps        = float(win_cfg.get("fps", CANONICAL_FPS))

    min_secs = float(win_cfg.get("min_window_seconds", 1.28))
    max_secs = float(win_cfg.get("max_window_seconds", 10.24))
    T_min    = int(round(min_secs * fps / patch_size)) * patch_size
    T_max    = int(round(max_secs * fps / patch_size)) * patch_size
    T_min    = max(T_min, patch_size)   # at least one patch

    batch_size  = int(data_cfg.get("batch_size", trainer_cfg.get("batch_size", 32)))
    num_workers = int(data_cfg.get("num_workers", 4))
    seed        = int(cfg_get(cfg, ["trainer", "seed"], 42))

    loaders: Dict[str, Optional[DataLoader]] = {}

    for split, files in [("train", train_files), ("val", val_files)]:
        if not files:
            loaders[f"{split}_loader"] = None
            continue
        dataset = NymeriaVarLenMAEDataset(
            pkl_paths  = files,
            batch_size = batch_size,
            T_min      = T_min,
            T_max      = T_max,
            patch_size = patch_size,
            cfg        = cfg,
            split_name = split,
            seed       = seed + (0 if split == "train" else 1000),
        )
        # batch_size=None: pass through the already-batched tensors unchanged
        loaders[f"{split}_loader"] = DataLoader(
            dataset,
            batch_size  = None,
            num_workers = num_workers if split == "train" else min(num_workers, 2),
            pin_memory  = True,
            persistent_workers = num_workers > 0,
        )

    return loaders
