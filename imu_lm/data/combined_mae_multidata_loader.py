"""combined_mae_multidata_loader.py
Combines every IMU source available under /projects/b53017/vbc1591 into one
variable-length MAE pretraining dataset for the ViT1D alt-attention encoder
(``vit1d_ssl_mae_varlen_altattn``).

Sources merged (config key -> on-disk location -> loader module reused):
  realdata     IMULM_master_dataset_v2.parquet (pamap2/recofit/samosa/shoaib/
               ut_watch/utd_mhad/wisdm)          -> imu_lm.data.loaders
  ocdetect     OCDetect_extracted/preprocessed   -> imu_lm.data.OCDetect_dataloader
  nymeria      processed_data_body_Ty_rwrist(+_test_) (real wrist)
                                                  -> nymeria_real_wrist_full_body_loader_FLU
  totalcapture DIP_recalculate (DIP-recalculated TotalCapture, s1-s5 —
               also referred to as "DIP-IMU"; same on-disk source, counted once)
                                                  -> totalcapture_wrist_loader_FLU
  imuposer     imuposer_dataset/P1..P10          -> imuposer_wrist_loader_FLU
  eatingspeed  EatingSpeed/{FD-I,FD-II,MO}       -> eatingspeed_wrist_loader
  motionx      Motion-Xplusplus (Synthesis)      -> motionx_wrist_loader_FLU
  humanml3d    HumanML3D / AMASS (Synthesis)     -> humanml3d_dataset_FLU
  sonic        bones-seed BVH mocap (Synthesis)  -> bones_seed_wrist_full_body_loader_FLU
  capture24    capture24/capture24 (acc-only)    -> capture24_loader
  openwatch    openwatch_raw/... (real wrist gesture, acc+gyro) -> openwatch_loader
  nhanes       nhanes_pax/{pax_g,pax_h} (NHANES 2011-2014 PAX80,
               acc-only, free-living)             -> nhanes_wrist_loader

Every source yields fixed-length ``[6, T_max]`` windows, ``T_max`` driven by
the shared ``windowing.window_seconds`` config value (all per-source loaders
read it via ``imu_lm.data.windowing.compute_T_and_hop`` except OCDetect, which
is given ``window_samples=T_max`` explicitly below). Labels are discarded
(this is unsupervised MAE pretraining) via ``XOnlyDataset``. A per-batch
random crop to ``L in [min_samples, max_samples]`` (a multiple of patch_size)
implements the "varlen" part of the objective — mirrors the crop-per-batch
scheme already used by ``mae_realdata_ocdetect_flu`` in
``imu_lm/models/ViT1D/run.py``.

HumanML3D/AMASS clips are individually shorter than T_max more often than
not (mocap clips, not long continuous recordings); to keep it stackable with
every sliding-window source, each clip is zero-padded on the right to
T_max samples. This means a random crop into a short clip's padded tail can
occasionally land on an all-zero segment — an accepted, minor tradeoff
matching the padding already used by ``HumanML3DIMUVarLenDatasetFLU``
elsewhere in this codebase.

Capture24 is accelerometer-only (3ch, no gyro): its windows are zero-padded
to 6ch and carry a per-sample ``channel_valid_mask`` marking the 3 fake gyro
channels invalid (see ``AccOnlyPadTo6ChDataset``). This mask rides alongside
every other source (all-True for the 6ch ones) all the way to the MAE
objective, which forces invalid channels to always be masked/hidden from the
encoder and always excludes them from the reconstruction loss — never fake
zeros pretending to be a target.

NHANES is also accelerometer-only (3ch) and uses the same
``AccOnlyPadTo6ChDataset`` wrap as Capture24. It is dramatically larger than
every other source combined (~2.77M raw person-hours across 14,693
participants), so it is always subsampled at the participant-file level
before any I/O (``sample_fraction``/``max_files``, see ``_build_nhanes``) --
without this cap, "natural" (proportional-to-size) sampling mode would let
NHANES dominate essentially every batch.
"""

from __future__ import annotations

import glob
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from imu_lm.data.augmentations.transform import apply_augment
from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)

DEFAULT_ROOTS = {
    "realdata_parquet":   "/projects/b53017/vbc1591/IMULM_master_dataset_v2.parquet",
    "ocdetect_dir":       "/projects/b53017/vbc1591/OCDetect_extracted/preprocessed",
    "nymeria_train_root": "/projects/b53017/vbc1591/processed_data_body_Ty_rwrist",
    "nymeria_val_root":   "/projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist",
    "totalcapture_root":  "/projects/b53017/vbc1591/DIP_recalculate",
    "imuposer_root":      "/projects/b53017/vbc1591/imuposer_dataset",
    "eatingspeed_root":   "/projects/b53017/vbc1591/EatingSpeed",
    "motionx_root":       "/projects/b53017/vbc1591/Motion-Xplusplus",
    "humanml3d_root":     "/projects/b53017/vbc1591/HumanML3D",
    "sonic_root":         "/projects/b53017/vbc1591/bones-seed",
    "capture24_root":     "/projects/b53017/vbc1591/capture24/capture24",
    "openwatch_root":     "/projects/b53017/vbc1591/openwatch_raw/open_watch_without_augmentations",
    "nhanes_root":        "/projects/b53017/vbc1591/nhanes_pax",
}

ALL_SOURCES = [
    "realdata", "ocdetect", "nymeria", "totalcapture", "imuposer",
    "eatingspeed", "motionx", "humanml3d", "sonic", "capture24", "openwatch",
    "nhanes",
]


# ---------------------------------------------------------------------------
# Generic label-stripping wrapper
# ---------------------------------------------------------------------------

class XOnlyDataset(Dataset):
    """Strips label(s)/caption from any ``(x, ...)`` dataset item →
    ``(x, channel_valid_mask)`` where ``channel_valid_mask`` is all-True
    (every channel is real data — the normal case for every source except
    capture24, see ``AccOnlyPadTo6ChDataset`` below).
    """

    def __init__(self, ds, name: str):
        self._ds = ds
        self.name = name

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        item = self._ds[idx]
        if item is None:
            return None
        x = item[0] if isinstance(item, (tuple, list)) else item
        ch = torch.ones(x.shape[0], dtype=torch.bool)
        wv = getattr(self._ds, "_window_valid", None)
        if wv is not None:
            v = int(wv[idx])
            if v < x.shape[-1]:      # short clip: temporal validity mask rides along
                return x, ch, (torch.arange(x.shape[-1]) < v)
        return x, ch


class PadToLenXOnlyDataset(Dataset):
    """Like ``XOnlyDataset`` but right-pads/truncates ``x`` to exactly ``T`` samples."""

    def __init__(self, ds, name: str, T: int):
        self._ds = ds
        self.name = name
        self.T = T

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        item = self._ds[idx]
        if item is None:
            return None
        x = item[0] if isinstance(item, (tuple, list)) else item
        C, cur_T = x.shape[0], x.shape[-1]
        channel_valid_mask = torch.ones(C, dtype=torch.bool)
        if cur_T == self.T:
            return x, channel_valid_mask
        if cur_T > self.T:
            return x[:, : self.T], channel_valid_mask
        padded = torch.zeros((C, self.T), dtype=x.dtype)
        padded[:, :cur_T] = x
        return padded, channel_valid_mask


class AccOnlyPadTo6ChDataset(Dataset):
    """Wraps an accelerometer-only ``(x [3, T], y)`` dataset (capture24) so it
    stacks with the 6-channel (acc+gyro) sources: zero-pads 3 fake "gyro"
    channels and returns a per-sample ``channel_valid_mask`` marking them
    invalid — [True, True, True, False, False, False]. The MAE objective
    (``imu_lm.objectives.masked_1d_freq_channel_mask``) uses this mask to
    force those channels to always be hidden from the encoder and to always
    exclude them from the reconstruction loss, regardless of the random SSL
    mask draw.
    """

    def __init__(self, ds, name: str):
        self._ds = ds
        self.name = name

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        item = self._ds[idx]
        if item is None:
            return None
        x = item[0] if isinstance(item, (tuple, list)) else item
        assert x.shape[0] == 3, f"AccOnlyPadTo6ChDataset expects 3-channel acc-only x, got {x.shape}"
        T = x.shape[-1]
        x6 = torch.zeros((6, T), dtype=x.dtype)
        x6[:3] = x
        channel_valid_mask = torch.tensor([True, True, True, False, False, False])
        return x6, channel_valid_mask


class CachedWindowDataset(Dataset):
    """Reads pre-windowed ``(x, channel_valid_mask)`` pairs straight from the
    ``.npy`` cache built by ``scripts/build_multidata_cache.py`` — skips all
    of the raw parsing / SMPL FK / windowing a source's normal builder does.

    ``x`` is memory-mapped (``mmap_mode='r'``) since some sources can be tens
    of GB at full scale; ``channel_valid_mask`` is tiny ([N, 6] bool) and
    loaded fully into RAM.

    The cache itself holds clean (deterministically preprocessed, NOT
    augmented) windows — see ``scripts/build_multidata_cache.py``'s
    ``_cfg_without_augmentation``. Augmentation (random_rotation,
    gaussian_noise, ...) is applied here, fresh, on every ``__getitem__``
    call, using ``cfg`` (the live training config) — exactly mirroring what
    every non-cached per-source loader does inline in its own
    ``__getitem__``. Without this, a cached run would see the exact same
    single random draw (e.g. one fixed octahedral rotation) for a window on
    every epoch instead of a fresh one.

    ``split_name`` gates that augmentation to the train split only. It
    defaults to ``"train"`` so a bare ``CachedWindowDataset(...)`` still
    augments, but ``_load_cached_source`` passes the real split name, so val
    windows come back clean.
    """

    def __init__(self, x_path: str, mask_path: str, name: str, cfg: Any,
                 expected_T: Optional[int] = None, split_name: str = "train"):
        from imu_lm.utils.helpers import cfg_get as _cg
        self._cache_is_normalized = bool(_cg(cfg, ["objective", "cache_is_normalized"], True))
        self._tmask = None                      # set by _load_cached_source when a *_tmask.npy exists
        self.name = name
        self.cfg = cfg
        self.split_name = split_name
        self._x = np.load(x_path, mmap_mode="r")            # [N, 6, T_max] float32
        self._mask = np.load(mask_path)                      # [N, 6] bool
        assert len(self._x) == len(self._mask), (
            f"{name}: x has {len(self._x)} rows but mask has {len(self._mask)}"
        )
        if expected_T is not None and self._x.shape[-1] != expected_T:
            raise ValueError(
                f"[combined_mae] cached '{name}' windows are T={self._x.shape[-1]} samples but the "
                f"current config's data.variable_length.max_samples={expected_T} — this cache was "
                f"built with a different windowing.window_seconds. Rebuild the cache (delete "
                f"{x_path}) or fix data.variable_length.max_samples to match."
            )

        # UNIT HARMONISATION (2026-08-20). The pool mixes accelerometer units: Capture24 and NHANES
        # store g (|mean acc| ~ 1.0), every other source stores m/s^2 (~9.8). Per-window z-scoring hid
        # this, but `normalize.method: physical` divides by a fixed constant, so a 9.81x inconsistency
        # would go straight into the model for 48 % of the pool. Detect the unit from the data (or set
        # sources.<name>.acc_units: g|m/s2) and convert to m/s^2 before normalising.
        self._acc_gain = 1.0
        if not self._cache_is_normalized:
            src = (_cg(cfg, ["objective", "mae_multidata_varlen_altattn_flu", "sources", name], {}) or {})
            u = str(src.get("acc_units", "auto")).lower()
            if u == "auto":
                import numpy as _np
                probe = _np.asarray(self._x[: min(200, len(self._x))][:, :3])
                mag = float(_np.linalg.norm(probe.mean(-1), axis=1).mean())   # |gravity| if present
                spread = float(probe.std())
                # g-scaled data has BOTH a ~1 gravity magnitude and a sub-g spread. Gravity-free
                # m/s^2 data (PADS userAcceleration) also has a small mean but a much larger spread.
                u = "g" if (mag < 3.0 and spread < 1.5) else "m/s2"
            self._acc_gain = 9.80665 if u == "g" else 1.0
            if self._acc_gain != 1.0:
                logger.info("[combined_mae] %s: acc stored in g -> scaling by 9.80665 to m/s^2", name)

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx: int):
        # .copy(): self._x is memory-mapped read-only; apply_augment (and
        # torch.from_numpy) require a writable array.
        x_np = np.asarray(self._x[idx]).copy()          # [C, T]
        # Train-only augmentation (bug fixed 2026-07-28): this used to call
        # apply_augment unconditionally, and _load_cached_source builds this
        # class for BOTH splits -- so with random_rotation enabled (as in
        # every vit1d_ssl_mae_varlen_altattn_multidata* config) the val set was
        # re-randomized on every pass, and that noisy val loss drove both
        # best.pt selection and early stopping. Matches the gating that
        # nymeria/openwatch/realdata already do (`if split_name == "train"`).
        # Caches built before 2026-08-19 have preprocess.normalize BAKED IN (per-window z-score),
        # so `preprocess.normalize.method` had no effect on a cached run. A cache built with
        # normalize disabled is flagged `cache_is_normalized: false`, and normalisation is then
        # applied here at load time -- which is what makes zscore vs physical a real training-time
        # choice rather than a property frozen into the .npy files.
        if not self._cache_is_normalized:
            from imu_lm.data.augmentations.preprocess import normalize_window
            if self._acc_gain != 1.0:
                x_np[:3] = x_np[:3] * self._acc_gain      # g -> m/s^2, so `physical` scaling is consistent
            x_np = normalize_window(x_np.T, self.cfg).T
        if self.split_name == "train":
            x_np = apply_augment(x_np.T, self.cfg).T      # apply_augment expects [T, C]
        x = torch.from_numpy(np.ascontiguousarray(x_np))
        mask = torch.from_numpy(self._mask[idx].copy())
        if self._tmask is not None:
            # short-clip support: the cache stores a per-window VALID LENGTH (int32 [N]); expand it
            # to a [T] bool mask — True where real signal, False where the window was zero-padded
            # because the source clip was shorter than one full window (mocap sources).
            v = int(self._tmask[idx])
            if v < x.shape[-1]:
                return x, mask, (torch.arange(x.shape[-1]) < v)
        return x, mask


def _load_cached_source(cache_dir: str, name: str, vl_max: int, cfg: Any) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    """Return (train_ds, val_ds) from `{cache_dir}/{name}_{split}_{x,mask}.npy`
    if both files exist for a split, else None for that split. Raises if a
    cached split's window length doesn't match the current config's
    data.variable_length.max_samples (stale cache from a different
    windowing.window_seconds) — better to fail loudly here than deep inside
    a collate_fn stack() error at the first mixed batch.

    ``cfg`` is the live training config — passed straight into
    CachedWindowDataset so augmentation reflects whatever this run actually
    asks for, not whatever was (deliberately not) baked into the cache."""

    # sources.<name>.max_windows / max_hours_cached: subsample a cached source at LOAD time
    # (evenly across the file, so all participants/sessions stay represented). Lets one full-size
    # cache serve small ablations without a rebuild -- and keeps the working set in page cache.
    def _one(split: str) -> Optional[Dataset]:
        x_path = os.path.join(cache_dir, f"{name}_{split}_x.npy")
        mask_path = os.path.join(cache_dir, f"{name}_{split}_mask.npy")
        if os.path.exists(x_path) and os.path.exists(mask_path):
            ds_ = CachedWindowDataset(x_path, mask_path, name, cfg,
                                      expected_T=vl_max, split_name=split)
            tpath = os.path.join(cache_dir, f"{name}_{split}_tmask.npy")
            if os.path.exists(tpath):
                ds_._tmask = np.load(tpath, mmap_mode="r")
                logger.info("[combined_mae] %s/%s: temporal mask found (short clips included)", name, split)
            from imu_lm.utils.helpers import cfg_get as _cg
            src = (_cg(cfg, ["objective", "mae_multidata_varlen_altattn_flu", "sources", name], {}) or {})
            # sources.<name>.subset_index_file: explicit window indices into THIS source's cached
            # train array (e.g. a diversity-selected NHANES subset from analysis/nhanes_diversity.py).
            # Takes precedence over max_windows/max_hours_cached for the train split; val keeps
            # the cap path (or subset_index_file_val if given).
            sub_path = src.get("subset_index_file" if split == "train" else "subset_index_file_val")
            if sub_path:
                import numpy as _np
                from torch.utils.data import Subset
                keep = _np.asarray(_np.load(sub_path)).astype(int)
                if keep.ndim != 1 or len(keep) == 0 or keep.min() < 0 or keep.max() >= len(ds_):
                    raise ValueError(f"[combined_mae] {name}/{split}: bad subset_index_file {sub_path} "
                                     f"(shape {keep.shape}, range [{keep.min() if len(keep) else None}, "
                                     f"{keep.max() if len(keep) else None}], cache has {len(ds_)} windows)")
                logger.info("[combined_mae] %s/%s: subset_index_file -> %d of %d windows (%.0f h)",
                            name, split, len(keep), len(ds_), len(keep) * 5.12 / 3600)
                return Subset(ds_, _np.sort(keep).tolist())
            cap = src.get("max_windows")
            if cap is None and src.get("max_hours_cached") is not None:
                cap = int(float(src["max_hours_cached"]) * 3600 / 5.12)
            if cap is not None:
                cap = int(cap) if split == "train" else max(1, int(cap) // 10)
            if split == "val" and src.get("max_hours_val") is not None:
                # explicit val cap (evenly strided) -- a 1,100 h multi-source val set is pure
                # overhead for early stopping; ~100 h total is plenty (2026-08-22)
                cap = int(float(src["max_hours_val"]) * 3600 / 5.12)
                if 0 < cap < len(ds_):
                    import numpy as _np
                    from torch.utils.data import Subset
                    keep = _np.linspace(0, len(ds_) - 1, cap).astype(int)   # evenly spread, not the first N
                    logger.info("[combined_mae] %s/%s: capped %d -> %d windows (%.0f h)",
                                name, split, len(ds_), cap, cap * 5.12 / 3600)
                    return Subset(ds_, keep.tolist())
            return ds_
        return None

    return _one("train"), _one("val")


def _fraction_subset(ds: Optional[Dataset], fraction: Optional[float], seed: int = 42) -> Optional[Dataset]:
    """Randomly keep a `fraction` of `ds` (deterministic seed). No-op if fraction is unset/>=1."""
    if ds is None or fraction is None:
        return ds
    fraction = float(fraction)
    n = len(ds)
    if fraction >= 1.0 or n == 0:
        return ds
    k = max(1, round(n * fraction))
    idx = np.random.default_rng(seed).choice(n, size=k, replace=False)
    return torch.utils.data.Subset(ds, idx.tolist())


def _unwrap_raw(ds: Dataset):
    """Peel back the label-stripping wrapper (XOnlyDataset / AccOnlyPadTo6ChDataset /
    PadToLenXOnlyDataset) to the underlying per-source dataset object, where
    the window-provenance attributes (`_keys`, `_session_files`, `_clips`,
    `.records`, or an opt-in `.provenance`) actually live."""
    return getattr(ds, "_ds", ds)


def _group_windows_by_session(name: str, train_ds: Dataset) -> Optional[Dict[str, List[int]]]:
    """Return {session_key: [window_idx into train_ds, ...]} for sources that
    expose per-window provenance, else None (capture24/eatingspeed today --
    see the scaling_mode docstring in make_combined_mae_multidata_loaders).
    Every session_key here is stable/deterministic across runs (same seed ->
    same grouping), which is what makes session_fraction/subject_fraction
    reproducible.

    Handles ``train_ds`` being a ``torch.utils.data.Subset`` (as every source
    becomes after any scaling_mode other than the un-subsampled default) by
    recursing into ``.dataset`` and remapping indices through ``.indices`` --
    without this, re-deriving groups from an already-scaled source (e.g. the
    composition-summary log below, or a second call on the same source) would
    silently return None for every scaling_mode that actually produces Subsets.
    """
    if isinstance(train_ds, torch.utils.data.Subset):
        base_groups = _group_windows_by_session(name, train_ds.dataset)
        if base_groups is None:
            return None
        pos_of = {int(underlying_idx): local_i for local_i, underlying_idx in enumerate(train_ds.indices)}
        out: Dict[str, List[int]] = {}
        for key, idxs in base_groups.items():
            local_idxs = [pos_of[i] for i in idxs if i in pos_of]
            if local_idxs:
                out[key] = local_idxs
        return out

    raw = _unwrap_raw(train_ds)

    if hasattr(raw, "_keys") and hasattr(raw, "_sess_idx"):
        # realdata: SessionKey(dataset, subject_id, session_id)
        groups: Dict[str, List[int]] = {}
        for i, ki in enumerate(raw._sess_idx):
            k = raw._keys[int(ki)]
            groups.setdefault(f"{k.dataset}/{k.subject_id}/{k.session_id}", []).append(i)
        return groups

    if hasattr(raw, "_session_files") and hasattr(raw, "_sess_idx"):
        # nymeria/totalcapture/motionx/sonic/imuposer: one file path per session
        groups = {}
        for i, si in enumerate(raw._sess_idx):
            f = raw._session_files[int(si)]
            groups.setdefault(f, []).append(i)
        return groups

    prov = getattr(raw, "provenance", None)
    if prov is not None and prov.get("kind") == "ocdetect_file":
        groups = {}
        for i, fid in enumerate(prov["file_ids"]):
            f = prov["files"][int(fid)]
            groups.setdefault(f, []).append(i)
        return groups

    if hasattr(raw, "_clips"):
        # humanml3d: one window per clip, no sliding-window session structure
        return {c["id"]: [i] for i, c in enumerate(raw._clips)}

    if hasattr(raw, "records"):
        # openwatch: one window per record (gesture CSV), path is the "session"
        groups = {}
        for i, rec in enumerate(raw.records):
            groups.setdefault(str(rec.path), []).append(i)
        return groups

    return None


def _subject_key_from_session_key(name: str, session_key: str) -> str:
    """Best-effort session -> subject mapping, source-specific. Falls back to
    treating the session itself as the subject (documented, conservative --
    used for sources with no independently-verifiable subject id in this
    codebase: nymeria [recording id, not subject-labeled on disk],
    motionx/sonic/humanml3d [synthetic clips, no subject concept])."""
    if name == "realdata":
        parts = session_key.split("/", 2)
        return parts[1] if len(parts) > 1 else session_key
    # Accept BOTH spellings (bug fixed 2026-07-28): callers thread the
    # ALL_SOURCES key "totalcapture", while "totalcapture_dip" is only the
    # XOnlyDataset display name set in _build_totalcapture. Matching solely on
    # "totalcapture_dip" meant this branch never ran, so subject_fraction
    # silently degraded to session_fraction for this source (keeping a
    # fraction of its ~40 sessions instead of its 4 train subjects) and the
    # composition summary reported subjects=<#sessions>.
    if name in ("totalcapture", "totalcapture_dip"):
        base = os.path.basename(session_key)
        return base.split("_", 1)[0] if "_" in base else base  # "s1_scenario1.pkl" -> "s1"
    if name == "imuposer":
        return os.path.basename(os.path.dirname(session_key))  # ".../P3/xxx.pkl" -> "P3"
    if name == "ocdetect":
        try:
            from imu_lm.data.OCDetect_dataloader import _subject_id_from_path
            return str(_subject_id_from_path(session_key))
        except Exception:
            return session_key
    if name == "openwatch":
        return session_key  # already 1 record = 1 session; caller uses rec.subject_id directly when available
    return session_key


def _group_fraction_subset(name: str, train_ds: Dataset, fraction: float, level: str,
                            seed: int = 42) -> Tuple[Optional[Dataset], Optional[str]]:
    """Deterministically keep a `fraction` of *groups* (sessions or subjects,
    per `level`) rather than a fraction of individual windows -- all windows
    belonging to a kept group are kept. Returns (subset_or_None, warning_or_None);
    warning is set (and the caller should fall back to plain window_fraction)
    when `name` has no group provenance available."""
    groups = _group_windows_by_session(name, train_ds)
    if groups is None:
        return None, f"{name}: no window-level provenance available, falling back to window_fraction"

    if level == "subject":
        by_group: Dict[str, List[int]] = {}
        for sess_key, idxs in groups.items():
            subj = _subject_key_from_session_key(name, sess_key)
            by_group.setdefault(subj, []).extend(idxs)
    else:
        by_group = groups

    keys = sorted(by_group.keys())  # sorted for determinism independent of dict insertion order
    n_keep = max(1, round(len(keys) * fraction))
    rng = np.random.default_rng(seed)
    kept_keys = set(rng.choice(keys, size=n_keep, replace=False).tolist())
    kept_idx = sorted(i for k in kept_keys for i in by_group[k])
    return torch.utils.data.Subset(train_ds, kept_idx), None


def _windows_per_session_subset(name: str, train_ds: Dataset, k: int,
                                 seed: int = 42) -> Tuple[Optional[Dataset], Optional[str]]:
    """Keep at most `k` windows per session (all sessions kept, deterministic
    per-session subsample) -- a diversity-preserving alternative to
    window_fraction that caps redundant same-session windows instead of
    dropping whole sessions."""
    groups = _group_windows_by_session(name, train_ds)
    if groups is None:
        return None, f"{name}: no window-level provenance available, falling back to window_fraction"

    rng = np.random.default_rng(seed)
    kept_idx: List[int] = []
    for sess_key in sorted(groups.keys()):
        idxs = groups[sess_key]
        if len(idxs) <= k:
            kept_idx.extend(idxs)
        else:
            kept_idx.extend(rng.choice(idxs, size=k, replace=False).tolist())
    kept_idx.sort()
    return torch.utils.data.Subset(train_ds, kept_idx), None


def _wrap(loader_key: str, loaders: Optional[Dict[str, DataLoader]], name: str):
    if not loaders:
        return None
    dl = loaders.get(loader_key)
    if dl is None:
        return None
    ds = dl.dataset
    if ds is None or len(ds) == 0:
        return None
    return XOnlyDataset(ds, name)


# ---------------------------------------------------------------------------
# Per-source builders. Each returns (train_ds_or_None, val_ds_or_None).
# ---------------------------------------------------------------------------

def _build_realdata(cfg, src_cfg: dict) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    from imu_lm.data.loaders import WindowDataset, build_session_index, make_splits

    parquet_path = src_cfg.get("dataset_path") or cfg_get(cfg, ["paths", "dataset_path"])
    train_on = src_cfg.get("train_on_datasets") or cfg_get(cfg, ["data", "train_on_datasets"], None)
    si = build_session_index(parquet_path, cfg, dataset_filter=list(train_on) if train_on else None)
    splits = make_splits(si, cfg)
    train_ds = WindowDataset(parquet_path, si, splits["train_keys"], cfg, split_name="train")
    val_ds = WindowDataset(parquet_path, si, splits["val_keys"], cfg, split_name="val")
    logger.info("[combined_mae] realdata: train=%d val=%d", len(train_ds), len(val_ds))
    return XOnlyDataset(train_ds, "realdata"), XOnlyDataset(val_ds, "realdata")


def _build_ocdetect(cfg, src_cfg: dict, vl_max: int,
                     collect_provenance: bool = False) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    from imu_lm.data.OCDetect_dataloader import (
        OCDetectDataset, _collect_windows, _list_files, select_ocd_files,
    )

    data_dir = src_cfg.get("data_dir", DEFAULT_ROOTS["ocdetect_dir"])
    max_files = src_cfg.get("max_files")
    sample_fraction = src_cfg.get("sample_fraction")
    max_hours = float(src_cfg.get("max_hours", 100000.0))
    splits = tuple(src_cfg.get("splits", [0.8, 0.1, 0.1]))
    win_sec = float(cfg_get(cfg, ["windowing", "window_seconds"], 10.24))

    # sample_fraction is applied at the file-selection stage (before any I/O),
    # since OCDetect is large enough (678 files, ~5,800h) that loading everything
    # just to discard most of it afterwards would be wasteful.
    if sample_fraction is not None:
        total_files = len(_list_files(data_dir))
        frac_files = max(1, round(total_files * float(sample_fraction)))
        max_files = min(int(max_files), frac_files) if max_files else frac_files
        logger.info("[combined_mae] ocdetect: sample_fraction=%.4g -> capping to %d/%d files",
                    float(sample_fraction), max_files, total_files)

    files = select_ocd_files(data_dir, max_files=int(max_files) if max_files else None)
    if not files:
        logger.warning("[combined_mae] ocdetect: no files found under %s, skipping", data_dir)
        return None, None

    # Split at the FILE level (shuffled, seeded), not the window level: windows
    # use a fixed 50% hop, so any two windows from the same file that end up on
    # opposite sides of a window-level cut are not independent (they overlap
    # temporally). The previous implementation sliced the file-concatenated,
    # unshuffled window array positionally -- whichever single file happened to
    # straddle that cut leaked ~50%-overlapping windows across train/val (see
    # analysis/mae_scaling_diagnostics/02_split_audit.py, and
    # analysis/mae_scaling_diagnostics/DIAGNOSIS.md §2). Splitting files first
    # guarantees no file's windows can ever appear on both sides, matching the
    # file/session-disjoint convention every other sliding-window source here
    # already uses (nymeria/totalcapture/imuposer/motionx/sonic).
    split_seed = int(src_cfg.get("split_seed", 42))
    n_files = len(files)
    n_train_files = max(1, int(round(splits[0] * n_files)))
    if n_files >= 2:
        n_train_files = min(n_train_files, n_files - 1)  # always leave >=1 file for val
    n_val_files = min(int(round(splits[1] * n_files)), n_files - n_train_files)
    if n_files >= 2 and n_val_files == 0:
        n_val_files = 1
    perm = np.random.default_rng(split_seed).permutation(n_files)
    train_files = sorted(files[i] for i in perm[:n_train_files])
    val_files = sorted(files[i] for i in perm[n_train_files:n_train_files + n_val_files])
    if not val_files:
        logger.warning("[combined_mae] ocdetect: only %d file(s) selected -- no file left for a "
                        "leak-free val split, val will be empty for this source", n_files)

    # "legacy" (default) keeps every pre-2026-08-20 cache/checkpoint reproducible;
    # "documented" applies the real Android->FLU rotation to acc AND gyro alike.
    # See OCDetect_dataloader._load_recording for what the two actually do.
    flu_convention = str(src_cfg.get("flu_convention", "legacy"))

    def _collect(file_list):
        if not file_list:
            return np.empty((0, 6, vl_max), dtype=np.float32), np.empty(0, dtype=np.int64), None
        if collect_provenance:
            X, y, file_ids = _collect_windows(file_list, apply_flu_transform=True,
                                               window_samples=vl_max, hop_samples=vl_max // 2,
                                               return_file_ids=True,
                                               flu_convention=flu_convention)
            return X, y, file_ids
        X, y = _collect_windows(file_list, apply_flu_transform=True,
                                 window_samples=vl_max, hop_samples=vl_max // 2,
                                 flu_convention=flu_convention)
        return X, y, None

    X_tr, y_tr, fid_tr = _collect(train_files)
    X_va, y_va, fid_va = _collect(val_files)

    # max_hours cap: applied to the train split only (this is what "keep
    # OCDetect from dominating the pretraining pool" was actually protecting
    # against; val sets are small and uncapped, matching every other source).
    max_wins = int(max_hours * 3600.0 / win_sec)
    if len(X_tr) > max_wins:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_tr), size=max_wins, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        if fid_tr is not None:
            fid_tr = fid_tr[idx]

    train_ds = OCDetectDataset(X_tr, y_tr, cfg=cfg)
    val_ds = OCDetectDataset(X_va, y_va, cfg=cfg)
    if collect_provenance:
        # `.provenance`: opt-in, diagnostic-only attribute -- not read by any
        # production training path. file_ids index into their OWN split's file
        # list (train_files/val_files respectively), which are by construction
        # disjoint.
        train_ds.provenance = {"kind": "ocdetect_file", "file_ids": fid_tr, "files": train_files}
        val_ds.provenance = {"kind": "ocdetect_file", "file_ids": fid_va, "files": val_files}
    logger.info("[combined_mae] ocdetect: train=%d windows/%d files, val=%d windows/%d files "
                "(file-disjoint split, seed=%d)",
                len(train_ds), len(train_files), len(val_ds), len(val_files), split_seed)
    train_out = XOnlyDataset(train_ds, "ocdetect")
    val_out = XOnlyDataset(val_ds, "ocdetect") if len(val_ds) > 0 else None
    return train_out, val_out


def _build_nymeria(cfg, src_cfg: dict) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    from imu_lm.data.nymeria_real_wrist_full_body_loader_FLU import (
        make_nymeria_real_wrist_full_body_loaders_flu,
    )

    train_root = src_cfg.get("train_root", DEFAULT_ROOTS["nymeria_train_root"])
    val_root = src_cfg.get("val_root", DEFAULT_ROOTS["nymeria_val_root"])
    train_files = sorted(glob.glob(os.path.join(train_root, "*.pkl")))
    val_files = sorted(glob.glob(os.path.join(val_root, "*.pkl")))
    max_train, max_val = src_cfg.get("max_train_files"), src_cfg.get("max_val_files")
    if max_train:
        train_files = train_files[: int(max_train)]
    if max_val:
        val_files = val_files[: int(max_val)]
    if not train_files:
        logger.warning("[combined_mae] nymeria: no train files found under %s, skipping", train_root)
        return None, None

    loaders = make_nymeria_real_wrist_full_body_loaders_flu(
        cfg=cfg, pkl_root="/", train_files=train_files, val_files=val_files,
        add_gravity=bool(src_cfg.get("add_gravity", True)),
    )
    return _wrap("train_loader", loaders, "nymeria"), _wrap("val_loader", loaders, "nymeria")


def _build_totalcapture(cfg, src_cfg: dict) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    from imu_lm.data.totalcapture_wrist_loader_FLU import make_totalcapture_real_wrist_loaders_flu

    root = src_cfg.get("root", DEFAULT_ROOTS["totalcapture_root"])
    train_subjects = src_cfg.get("train_subjects", ["s1", "s2", "s3", "s4"])
    val_subjects = src_cfg.get("val_subjects", ["s5"])

    def _expand(subjects):
        return [f for s in subjects for f in sorted(glob.glob(os.path.join(root, f"{s}_*.pkl")))]

    train_files, val_files = _expand(train_subjects), _expand(val_subjects)
    if not train_files:
        logger.warning("[combined_mae] totalcapture: no files found under %s, skipping", root)
        return None, None

    loaders = make_totalcapture_real_wrist_loaders_flu(
        cfg=cfg, pkl_root=root, train_files=train_files, val_files=val_files,
        target_hz=int(src_cfg.get("target_hz", 60)),
        add_gravity=bool(src_cfg.get("add_gravity", True)),
    )
    # Also referred to as "DIP-IMU" — same on-disk source (DIP_recalculate),
    # not double-counted as a separate dataset.
    return (_wrap("train_loader", loaders, "totalcapture_dip"),
            _wrap("val_loader", loaders, "totalcapture_dip"))


def _build_imuposer(cfg, src_cfg: dict) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    from imu_lm.data.imuposer_wrist_loader_FLU import make_imuposer_wrist_flu_loaders

    root = src_cfg.get("root", DEFAULT_ROOTS["imuposer_root"])
    val_participants = set(src_cfg.get("val_participants", ["P9", "P10"]))
    all_files = sorted(glob.glob(os.path.join(root, "P*", "*.pkl")))
    train_files = [f for f in all_files if os.path.basename(os.path.dirname(f)) not in val_participants]
    val_files = [f for f in all_files if os.path.basename(os.path.dirname(f)) in val_participants]
    max_train = src_cfg.get("max_train_files")
    if max_train:
        train_files = train_files[: int(max_train)]
    if not train_files:
        logger.warning("[combined_mae] imuposer: no files found under %s, skipping", root)
        return None, None

    loaders = make_imuposer_wrist_flu_loaders(
        cfg=cfg, train_files=train_files, val_files=val_files,
        target_hz=int(src_cfg.get("target_hz", 50)),
        add_gravity=bool(src_cfg.get("add_gravity", False)),
    )
    return _wrap("train_loader", loaders, "imuposer"), _wrap("val_loader", loaders, "imuposer")


def _build_eatingspeed(cfg, src_cfg: dict) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    from imu_lm.data.eatingspeed_wrist_loader import make_eatingspeed_loaders_random

    root = src_cfg.get("root", DEFAULT_ROOTS["eatingspeed_root"])
    subsets = src_cfg.get("subsets", ["FD-I", "FD-II", "MO"])
    max_participants = src_cfg.get("max_participants_per_subset")
    sample_fraction = src_cfg.get("sample_fraction")

    train_parts, val_parts = [], []
    for subset in subsets:
        try:
            loaders = make_eatingspeed_loaders_random(
                root=root, subset=subset, hand=str(src_cfg.get("hand", "both")),
                splits=tuple(src_cfg.get("splits", [0.8, 0.1, 0.1])),
                cfg=cfg,
                max_participants=int(max_participants) if max_participants else None,
            )
        except Exception:
            logger.exception("[combined_mae] eatingspeed: failed to load subset %s, skipping it", subset)
            continue
        # sample_fraction is applied after windowing (per-participant loading in this
        # loader is eager regardless of any cap, so there's no cheaper earlier point);
        # applied independently to train/val so the val split stays representative.
        tr = _fraction_subset(_wrap("train_loader", loaders, f"eatingspeed_{subset}"), sample_fraction)
        va = _fraction_subset(_wrap("val_loader", loaders, f"eatingspeed_{subset}"), sample_fraction)
        if tr is not None:
            train_parts.append(tr)
        if va is not None:
            val_parts.append(va)

    if not train_parts:
        logger.warning("[combined_mae] eatingspeed: no subsets loaded, skipping")
        return None, None
    train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_ds = (val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)) if val_parts else None
    return train_ds, val_ds


def _build_motionx(cfg, src_cfg: dict) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    from imu_lm.data.motionx_wrist_loader_FLU import make_motionx_wrist_loaders_flu

    root = src_cfg.get("root", DEFAULT_ROOTS["motionx_root"])
    try:
        loaders = make_motionx_wrist_loaders_flu(
            cfg=cfg, motionx_root=root,
            categories=src_cfg.get("categories"),
            val_frac=float(src_cfg.get("val_frac", 0.05)),
            test_frac=float(src_cfg.get("test_frac", 0.0)),
            max_clips_per_category=src_cfg.get("max_clips_per_category"),
            add_gravity=bool(src_cfg.get("add_gravity", True)),
            v1_zip=src_cfg.get("v1_zip"),
            v1_subsets=src_cfg.get("v1_subsets"),
        )
    except Exception:
        logger.exception("[combined_mae] motionx: failed to load, skipping")
        return None, None
    return _wrap("train_loader", loaders, "motionx"), _wrap("val_loader", loaders, "motionx")


def _build_sonic(cfg, src_cfg: dict) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    """"SONIC" (synthesis) == the bones-seed BVH mocap dataset on this deployment."""
    from imu_lm.data.bones_seed_wrist_full_body_loader_FLU import (
        make_bones_seed_wrist_full_body_loaders_flu,
    )

    root = src_cfg.get("root", DEFAULT_ROOTS["sonic_root"])
    try:
        loaders = make_bones_seed_wrist_full_body_loaders_flu(
            cfg=cfg,
            bones_seed_root=root,
            train_files=src_cfg.get("train_files", "all"),
            val_files=src_cfg.get("val_files"),
            add_gravity=bool(src_cfg.get("add_gravity", True)),
            max_train_sessions=src_cfg.get("max_train_sessions"),
            max_val_sessions=src_cfg.get("max_val_sessions"),
        )
    except Exception:
        logger.exception("[combined_mae] sonic (bones-seed): failed to load, skipping")
        return None, None
    return _wrap("train_loader", loaders, "sonic"), _wrap("val_loader", loaders, "sonic")


def _capped_split_name(root: str, split: str, cap: Optional[int]) -> str:
    """Return a split name whose backing .txt file has at most `cap` clip ids.

    ``HumanML3DIMUDatasetFLU`` eagerly runs SMPL FK for every id in the split
    file at construction time (~1s/clip; the full train split has ~23k ids,
    i.e. hours), so a post-hoc ``dataset._clips[:cap]`` slice does not save
    any time — the file must be capped *before* construction. When a cap is
    given and smaller than the real split, writes a small derived split file
    (``<split>_capped_<cap>.txt``) alongside the official ones and returns its
    name; otherwise returns ``split`` unchanged.

    Selection is a deterministic RANDOM sample, not ``ids[:cap]`` (changed
    2026-07-28). HumanML3D clip ids are assigned in AMASS-archive order, so
    taking the alphabetically-first ``cap`` ids kept a contiguous block of
    source sub-datasets rather than a representative sample of the split --
    at the default cap of 4000 of ~23k train clips, that silently selected a
    motion-content-correlated subset. The seed is fixed (42) so the choice is
    reproducible, and it is encoded in the derived file's name so the old
    ``<split>_capped_<cap>.txt`` files (head-selected) remain on disk,
    untouched and still loadable, for reproducing pre-fix runs.
    """
    if not cap:
        return split
    split_file = os.path.join(root, "HumanML3D", f"{split}.txt")
    with open(split_file) as f:
        ids = [l.strip() for l in f if l.strip()]
    cap = int(cap)
    if cap >= len(ids):
        return split
    seed = 42
    capped_name = f"{split}_capped_{cap}_seed{seed}"
    capped_file = os.path.join(root, "HumanML3D", f"{capped_name}.txt")
    if not os.path.exists(capped_file):
        idx = np.random.default_rng(seed).choice(len(ids), size=cap, replace=False)
        kept = [ids[i] for i in sorted(idx)]
        with open(capped_file, "w") as f:
            f.write("\n".join(kept) + "\n")
        logger.info("[combined_mae] humanml3d: wrote %s (%d/%d clips, random seed=%d)",
                    capped_file, cap, len(ids), seed)
    return capped_name


def _build_humanml3d(cfg, src_cfg: dict, vl_max: int) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    from imu_lm.data.humanml3d_dataset_FLU import HumanML3DIMUDatasetFLU

    root = src_cfg.get("root", DEFAULT_ROOTS["humanml3d_root"])
    patch_size = int(cfg_get(cfg, ["vit1d", "patch_size"], 16))
    t_min = int(src_cfg.get("t_min", 64))
    # Default caps: the full train split (~23k clips) takes hours to construct
    # (eager per-clip SMPL FK) — cap by default and let the config raise it
    # deliberately once the one-time build cost is acceptable.
    max_train = src_cfg.get("max_train_clips", 4000)
    max_val = src_cfg.get("max_val_clips", 500)

    try:
        train_split = _capped_split_name(root, "train", max_train)
        val_split = _capped_split_name(root, "val", max_val)
        train_raw = HumanML3DIMUDatasetFLU(root=root, split=train_split, patch_size=patch_size,
                                            T_min=t_min, T_max=vl_max, cfg=cfg,
                                            split_name="train")
        val_raw = HumanML3DIMUDatasetFLU(root=root, split=val_split, patch_size=patch_size,
                                          T_min=t_min, T_max=vl_max, cfg=cfg,
                                          split_name="val")
    except Exception:
        logger.exception("[combined_mae] humanml3d: failed to load under %s, skipping", root)
        return None, None

    if len(train_raw) == 0:
        return None, None

    train_ds = PadToLenXOnlyDataset(train_raw, "humanml3d", vl_max)
    val_ds = PadToLenXOnlyDataset(val_raw, "humanml3d", vl_max) if len(val_raw) else None
    return train_ds, val_ds


def _build_capture24(cfg, src_cfg: dict) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    """Capture24 is accelerometer-only (3ch) — padded to 6ch with the 3 fake
    "gyro" channels marked invalid via ``AccOnlyPadTo6ChDataset`` (see its
    docstring). ``sample_fraction`` is applied at the file-selection stage
    (before any I/O), matching the OCDetect pattern, since Capture-24 is also
    large enough (151 participants, ~4,000h) to make loading-then-discarding
    wasteful.
    """
    from imu_lm.data.capture24_loader import make_capture24_loaders_random, select_capture24_files

    root = src_cfg.get("root", DEFAULT_ROOTS["capture24_root"])
    max_files = src_cfg.get("max_files")
    sample_fraction = src_cfg.get("sample_fraction")

    if sample_fraction is not None:
        total_files = len(select_capture24_files(root, max_files=None))
        frac_files = max(1, round(total_files * float(sample_fraction)))
        max_files = min(int(max_files), frac_files) if max_files else frac_files
        logger.info("[combined_mae] capture24: sample_fraction=%.4g -> capping to %d/%d files",
                    float(sample_fraction), max_files, total_files)

    try:
        loaders = make_capture24_loaders_random(
            data_dir=root,
            splits=tuple(src_cfg.get("splits", [0.8, 0.1, 0.1])),
            target_hz=src_cfg.get("target_hz"),
            max_files=int(max_files) if max_files else None,
            cfg=cfg,
            all_windows=bool(src_cfg.get("all_windows", False)),
        )
    except Exception:
        logger.exception("[combined_mae] capture24: failed to load under %s, skipping", root)
        return None, None

    def _wrap_acc_only(loader_key):
        if not loaders:
            return None
        dl = loaders.get(loader_key)
        if dl is None or dl.dataset is None or len(dl.dataset) == 0:
            return None
        return AccOnlyPadTo6ChDataset(dl.dataset, "capture24")

    return _wrap_acc_only("train_loader"), _wrap_acc_only("val_loader")


def _build_openwatch(cfg, src_cfg: dict) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    """OpenWatch: real wrist-worn gesture recordings, 51 subjects, acc+gyro.
    Each gesture CSV is one fixed-length sample (resampled to the shared
    windowing.window_seconds length, same as every other source) — no sliding
    window needed. Subject-level train/val split (auto 80/10/10 by default,
    or override with explicit train_subjects/val_subjects)."""
    from imu_lm.data.openwatch_loader import make_openwatch_loaders

    root = src_cfg.get("root", DEFAULT_ROOTS["openwatch_root"])
    try:
        loaders = make_openwatch_loaders(
            cfg=cfg,
            root=root,
            train_subjects=src_cfg.get("train_subjects"),
            val_subjects=src_cfg.get("val_subjects"),
            test_subjects=src_cfg.get("test_subjects"),
            positions=src_cfg.get("positions"),
            gestures=src_cfg.get("gestures"),
            label_ids=src_cfg.get("label_ids"),
            include_ppg=bool(src_cfg.get("include_ppg", False)),
            coordinate_frame=str(src_cfg.get("coordinate_frame", "flu")),
            preload=bool(src_cfg.get("preload", True)),
        )
    except Exception:
        logger.exception("[combined_mae] openwatch: failed to load under %s, skipping", root)
        return None, None

    train_ds = _fraction_subset(_wrap("train_loader", loaders, "openwatch"), src_cfg.get("sample_fraction"))
    val_ds = _fraction_subset(_wrap("val_loader", loaders, "openwatch"), src_cfg.get("sample_fraction"))
    return train_ds, val_ds


def _build_nhanes(cfg, src_cfg: dict) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    """NHANES PAX80 is accelerometer-only (3ch) — padded to 6ch with the 3
    fake "gyro" channels marked invalid via ``AccOnlyPadTo6ChDataset``, same
    mechanism as ``capture24`` (see its docstring). ``sample_fraction`` is
    applied at the participant-archive-selection stage (before any I/O),
    same pattern as capture24/ocdetect — mandatory here since NHANES
    (~2.77M raw person-hours across 14,693 participants) otherwise dwarfs
    every other source combined.

    ``max_hours`` caps the **train** split's window budget DURING collection
    (passed through to ``make_nhanes_loaders_random`` as ``max_train_windows``),
    not after the fact — a first version of this builder collected every
    selected participant's full window set into memory before applying
    max_hours as a post-hoc Subset, which OOM-killed a real smoke test at
    101/132 participants (single-participant wear duration here can be
    100k+ windows, unlike every other source in this pool). Val is never
    capped, matching the convention every other capped source uses.
    """
    from imu_lm.data.nhanes_wrist_loader import make_nhanes_loaders_random, select_nhanes_files

    root = src_cfg.get("root", DEFAULT_ROOTS["nhanes_root"])
    max_files = src_cfg.get("max_files")
    sample_fraction = src_cfg.get("sample_fraction")
    max_hours = src_cfg.get("max_hours")
    win_sec = float(cfg_get(cfg, ["windowing", "window_seconds"], 10.24))
    hop_ratio = float(cfg_get(cfg, ["windowing", "window_hop_ratio"], 0.5))
    hop_sec = win_sec * hop_ratio

    if sample_fraction is not None:
        total_files = len(select_nhanes_files(root, max_files=None))
        frac_files = max(1, round(total_files * float(sample_fraction)))
        max_files = min(int(max_files), frac_files) if max_files else frac_files
        logger.info("[combined_mae] nhanes: sample_fraction=%.4g -> capping to %d/%d participants",
                    float(sample_fraction), max_files, total_files)

    max_hours_val = src_cfg.get("max_hours_val")

    max_train_windows = None
    if max_hours is not None:
        # windows are hop-spaced (not window_seconds-spaced), so this is the
        # count that actually corresponds to max_hours of DISTINCT wear time,
        # not an inflated ~2x-too-generous budget.
        max_train_windows = int(float(max_hours) * 3600.0 / hop_sec)
        logger.info("[combined_mae] nhanes: max_hours=%.1f -> max_train_windows=%d",
                    float(max_hours), max_train_windows)

    # max_hours_val: NHANES's per-participant window density is so much
    # higher than every other source's that leaving val uncapped (the
    # convention every other capped source here uses, since their overall
    # volume is small enough it doesn't matter) let NHANES alone make up
    # 92.8% of the entire 12-source val set in a real build -- see
    # nhanes_wrist_loader.make_nhanes_loaders_random's docstring/comment.
    max_val_windows = None
    if max_hours_val is not None:
        max_val_windows = int(float(max_hours_val) * 3600.0 / hop_sec)
        logger.info("[combined_mae] nhanes: max_hours_val=%.1f -> max_val_windows=%d",
                    float(max_hours_val), max_val_windows)

    try:
        loaders = make_nhanes_loaders_random(
            root=root,
            splits=tuple(src_cfg.get("splits", [0.9, 0.1])),
            target_hz=src_cfg.get("target_hz"),
            max_files=int(max_files) if max_files else None,
            max_train_windows=max_train_windows,
            max_val_windows=max_val_windows,
            cfg=cfg,
            spread_cap=bool(src_cfg.get("spread_across_participants", False)),
        )
    except Exception:
        logger.exception("[combined_mae] nhanes: failed to load under %s, skipping", root)
        return None, None

    def _wrap_acc_only(loader_key):
        if not loaders:
            return None
        dl = loaders.get(loader_key)
        if dl is None or dl.dataset is None or len(dl.dataset) == 0:
            return None
        return AccOnlyPadTo6ChDataset(dl.dataset, "nhanes")

    return _wrap_acc_only("train_loader"), _wrap_acc_only("val_loader")


_BUILDERS = {
    # `collect_provenance` is accepted (and ignored) by every builder except
    # ocdetect, which is the only source with no other way to recover
    # per-window file identity (see `_build_ocdetect`) -- everything else
    # already exposes it as plain attributes on the raw dataset object
    # (`_keys`/`_session_files`/`_clips`/`.records`, see
    # analysis/mae_scaling_diagnostics/lib/provenance.py) with zero extra cost.
    "realdata":     lambda cfg, sc, vl_max, **kw: _build_realdata(cfg, sc),
    "ocdetect":     lambda cfg, sc, vl_max, **kw: _build_ocdetect(
        cfg, sc, vl_max, collect_provenance=kw.get("collect_provenance", False)),
    "nymeria":      lambda cfg, sc, vl_max, **kw: _build_nymeria(cfg, sc),
    "totalcapture": lambda cfg, sc, vl_max, **kw: _build_totalcapture(cfg, sc),
    "imuposer":     lambda cfg, sc, vl_max, **kw: _build_imuposer(cfg, sc),
    "eatingspeed":  lambda cfg, sc, vl_max, **kw: _build_eatingspeed(cfg, sc),
    "motionx":      lambda cfg, sc, vl_max, **kw: _build_motionx(cfg, sc),
    "humanml3d":    lambda cfg, sc, vl_max, **kw: _build_humanml3d(cfg, sc, vl_max),
    "sonic":        lambda cfg, sc, vl_max, **kw: _build_sonic(cfg, sc),
    "capture24":    lambda cfg, sc, vl_max, **kw: _build_capture24(cfg, sc),
    "openwatch":    lambda cfg, sc, vl_max, **kw: _build_openwatch(cfg, sc),
    "nhanes":       lambda cfg, sc, vl_max, **kw: _build_nhanes(cfg, sc),
}


# ---------------------------------------------------------------------------
# Variable-length crop collate (mirrors mae_realdata_ocdetect_flu)
# ---------------------------------------------------------------------------

def _make_collate_fns(var_len: bool, vl_min: int, vl_max: int, patch_size: int):
    """Build train/val collate fns.

    Each dataset item is ``(x [C, T], channel_valid_mask [C] bool)`` — True
    where the channel is real data (every source except capture24, whose 3
    fake "gyro" channels are marked False; see ``AccOnlyPadTo6ChDataset``).
    Batches are returned as ``(x [B, C, T], dummy_y, channel_valid_mask [B, C])``
    — the 3-tuple consumed by
    ``imu_lm.objectives.masked_1d_freq_channel_mask.forward_loss``.
    channel_valid_mask has no time dimension, so it is stacked as-is
    (unaffected by the per-batch temporal crop below).
    """
    def _tstack(batch, T):
        """Per-sample temporal validity [B, T]: items carry one only if their source has short
        clips; everything else is all-True, so mixed batches stack cleanly."""
        if not any(len(b) > 2 for b in batch):
            return None
        out = []
        for b in batch:
            out.append(b[2] if len(b) > 2 else torch.ones(T, dtype=torch.bool))
        return torch.stack(out, 0)

    if not var_len:
        def _collate(batch):
            batch = [b for b in batch if b is not None]
            if not batch:
                return None
            xs = torch.stack([b[0] for b in batch], 0)
            channel_mask = torch.stack([b[1] for b in batch], 0)
            tm = _tstack(batch, xs.shape[-1])
            return (xs, torch.empty(0), channel_mask) if tm is None else (xs, torch.empty(0), channel_mask, tm)
        return _collate, _collate

    vl_lengths = list(range(vl_min, vl_max + 1, patch_size))

    def _collate_train(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        xs = torch.stack([b[0] for b in batch], 0)   # [B, C, Tmax]
        channel_mask = torch.stack([b[1] for b in batch], 0)   # [B, C]
        tm = _tstack(batch, xs.shape[-1])
        L = random.choice(vl_lengths)
        Tmax = xs.shape[-1]
        if L < Tmax:
            off = random.randint(0, Tmax - L)
            xs = xs[:, :, off:off + L]
            if tm is not None:
                tm = tm[:, off:off + L]
        if tm is None:
            return xs.contiguous(), torch.empty(0), channel_mask
        return xs.contiguous(), torch.empty(0), channel_mask, tm.contiguous()

    def _collate_val(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        xs = torch.stack([b[0] for b in batch], 0)[:, :, :vl_max]
        channel_mask = torch.stack([b[1] for b in batch], 0)
        tm = _tstack(batch, xs.shape[-1])
        if tm is None:
            return xs.contiguous(), torch.empty(0), channel_mask
        return xs.contiguous(), torch.empty(0), channel_mask, tm[:, :vl_max].contiguous()

    return _collate_train, _collate_val


# ---------------------------------------------------------------------------
# Top-level factory
# ---------------------------------------------------------------------------

def make_combined_mae_multidata_loaders(cfg: Any, collect_provenance: bool = False) -> Dict[str, DataLoader]:
    """Build train/val DataLoaders merging every enabled source.

    ``collect_provenance`` (default False, zero effect on normal training
    runs): when True, forces every source to build from its raw loader
    (bypassing ``cache_dir``, which stores plain tensors with no window
    identity) and, where the return dict also gains an
    ``"composition"`` entry: ``{name: {"train_n": int, "val_n": int}}``
    per-source window counts pre-``overall_fraction``, mirroring the doc-only
    logging below but returned programmatically for
    ``analysis/mae_scaling_diagnostics`` scripts. See
    ``analysis/mae_scaling_diagnostics/lib/provenance.py`` for how to recover
    per-window subject/session identity from the returned loaders' underlying
    dataset objects.

    Reads ``objective.mae_multidata_varlen_altattn_flu``::

        sources:
          realdata:     {enabled: true, train_on_datasets: [...]}
          ocdetect:     {enabled: true, data_dir: ..., max_hours: ..., max_files: ..., sample_fraction: ...}
          nymeria:      {enabled: true, train_root: ..., val_root: ...}
          totalcapture: {enabled: true, root: ..., train_subjects: [...], val_subjects: [...]}
          imuposer:     {enabled: true, root: ..., val_participants: [...]}
          eatingspeed:  {enabled: true, root: ..., subsets: [FD-I, FD-II, MO], sample_fraction: ...}
          motionx:      {enabled: true, root: ..., max_clips_per_category: ...}
          humanml3d:    {enabled: true, root: ..., max_train_clips: ...}
          sonic:        {enabled: true, root: ..., max_train_sessions: ..., max_val_sessions: ...}
          capture24:    {enabled: true, root: ..., sample_fraction: ...}   # acc-only, see AccOnlyPadTo6ChDataset
          openwatch:    {enabled: true, root: ..., train_subjects: [...], val_subjects: [...]}
        sampling_mode: balanced | natural       # default: balanced
        source_weights: {realdata: 1.0, ...}    # only used when sampling_mode == balanced
        cache_dir: /projects/b53017/vbc1591/mae_multidata_cache   # optional
        overall_fraction: 0.01                  # optional, for data-scaling-law runs
        stratify_overall_fraction: true         # optional, default false -- see below
        scaling_mode: window_fraction           # optional, default window_fraction -- see below
        windows_per_session: 20                 # only used when scaling_mode == windows_per_session

    ``cache_dir`` (optional): if set, each source first looks for
    ``{cache_dir}/{name}_{train,val}_{x,mask}.npy`` (built once via
    ``scripts/build_multidata_cache.py``) and loads directly from there —
    skipping all raw parsing / SMPL FK / windowing. Falls back to the normal
    (slow) builder for any source whose cache files aren't present, so a
    partially-built cache still works.

    Every source's ``sample_fraction`` (0-1) subsamples that source alone —
    applied before I/O where the underlying loader supports it (ocdetect,
    capture24), otherwise after windowing (eatingspeed).

    ``overall_fraction`` (0-1) subsamples the whole combined train set for
    data-scaling-law runs. Default mode ('pooled') draws one uniform sample
    over the already-concatenated pool of all sources -- simple, but at small
    fractions a source that's already a tiny share of the pool can be sampled
    down to zero windows purely by chance. Set
    ``stratify_overall_fraction: true`` to instead draw the fraction
    independently from each source's own windows, guaranteeing every enabled
    source keeps a nonzero, proportional presence at every fraction. Val set
    is always left full-size regardless of mode.

    ``scaling_mode`` (default ``window_fraction``) picks *what* gets
    subsampled by ``overall_fraction``, for testing whether random
    window-level scaling under/over-states real data diversity (see
    ``analysis/mae_scaling_diagnostics``):

      - ``window_fraction``: current behavior above (window_fraction is a synonym
        for the un-named default; `stratify_overall_fraction` still applies).
      - ``session_fraction``: keep `overall_fraction` of *sessions* per source
        (all windows from a kept session are kept) instead of a fraction of
        individual windows -- avoids splitting one session's windows across
        the kept/dropped boundary.
      - ``subject_fraction``: keep `overall_fraction` of *subjects* per source
        (all sessions/windows from a kept subject are kept). Falls back to
        session_fraction's session-key as the subject key for sources with no
        independently-verifiable subject id (nymeria, motionx, sonic,
        humanml3d -- see ``_subject_key_from_session_key``).
      - ``windows_per_session``: ignore `overall_fraction`; instead cap every
        session at `windows_per_session` windows (all sessions kept). A
        diversity-preserving alternative that trims redundant same-session
        windows rather than dropping whole sessions/subjects.
      - ``source_balanced_fraction``: like `window_fraction`, but the kept
        budget (`overall_fraction * total_train`) is split *equally* across
        enabled sources (capped by each source's availability) instead of
        proportionally to each source's natural size -- use to test whether
        large sources (ocdetect/capture24) dominating the pool is itself
        driving the saturation curve.

    Every mode except `window_fraction` requires per-window provenance,
    unavailable today for capture24/eatingspeed (see
    ``_group_windows_by_session``) -- those two sources silently fall back to
    plain `window_fraction` subsampling with a logged warning, every other
    source uses the requested mode.

    Before training, a composition summary (source/subject/session/window
    counts actually retained) is always logged at INFO regardless of mode --
    see the ``[combined_mae] scaling_mode=...`` log line below.

    Batches are ``(x [B, C, T], dummy_y, channel_valid_mask [B, C] bool)``.
    ``channel_valid_mask`` is all-True for every source except capture24
    (acc-only; its 3 zero-padded "gyro" channels are marked False). Consumed
    by ``imu_lm.objectives.masked_1d_freq_channel_mask.forward_loss``, which
    forces invalid channels to always be masked/hidden from the encoder and
    always excludes them from the reconstruction loss.
    """
    mm_cfg = cfg_get(cfg, ["objective", "mae_multidata_varlen_altattn_flu"], {}) or {}
    sources_cfg = mm_cfg.get("sources", {}) or {}

    vl_cfg = cfg_get(cfg, ["data", "variable_length"], {}) or {}
    var_len = bool(vl_cfg.get("enabled", False))
    vl_min = int(vl_cfg.get("min_samples", 64))
    vl_max = int(vl_cfg.get("max_samples", 512))
    patch_size = int(cfg_get(cfg, ["vit1d", "patch_size"], 16))

    cache_dir = mm_cfg.get("cache_dir")

    # Grouped scaling modes need per-window provenance (see
    # _group_windows_by_session), which cache_dir/.npy loading and ocdetect's
    # default builder path don't retain -- force it on regardless of the
    # caller-supplied collect_provenance when scaling_mode requires it.
    _needs_provenance_mode = str(mm_cfg.get("scaling_mode", "window_fraction")).lower() in (
        "session_fraction", "subject_fraction", "windows_per_session",
    )
    collect_provenance = collect_provenance or _needs_provenance_mode

    train_parts: List[Dataset] = []
    val_parts: List[Dataset] = []
    train_counts: Dict[str, int] = {}
    val_counts: Dict[str, int] = {}

    for name in ALL_SOURCES:
        src_cfg = sources_cfg.get(name, {}) or {}
        if not bool(src_cfg.get("enabled", True)):
            logger.info("[combined_mae] source '%s' disabled in config, skipping", name)
            continue

        train_ds = val_ds = None
        # `cache_name` lets one source read a differently-named cached slice, e.g.
        # realdata -> realdataclean_* (the leakage-free rebuild that drops the four
        # datasets which are also HAR evaluation sets). Falls back to the source name.
        cache_name = str(src_cfg.get("cache_name", name))
        if cache_dir and not collect_provenance:
            train_ds, val_ds = _load_cached_source(cache_dir, cache_name, vl_max, cfg)
            if train_ds is not None:
                logger.info("[combined_mae] %s: loaded from cache (%s as '%s') — train=%d val=%d",
                            name, cache_dir, cache_name, len(train_ds), len(val_ds) if val_ds else 0)

        if train_ds is None:
            train_ds, val_ds = _BUILDERS[name](cfg, src_cfg, vl_max, collect_provenance=collect_provenance)
        if train_ds is None:
            continue
        train_parts.append(train_ds)
        train_counts[name] = len(train_ds)
        if val_ds is not None:
            val_parts.append(val_ds)
            val_counts[name] = len(val_ds)

    if not train_parts:
        raise RuntimeError(
            "[combined_mae] no source produced any training data — check "
            "objective.mae_multidata_varlen_altattn_flu.sources in the config"
        )

    total_train = sum(train_counts.values())
    total_val = sum(len(v) for v in val_parts)
    logger.info(
        "[combined_mae] combined train=%d val=%d across %d sources: %s",
        total_train, total_val, len(train_counts),
        ", ".join(f"{k}={v} ({100.0*v/total_train:.1f}%)" for k, v in train_counts.items()),
    )

    # overall_fraction (0-1): for data-scaling-law runs -- subsamples the train set.
    # Val set is always left full-size either way, so every fraction is scored on the
    # same held-out data.
    #
    #   mode='pooled' (default, back-compat): one uniform draw over the
    #   *already-concatenated* pool of all sources, applied post-concat. In
    #   expectation this preserves each source's natural share of the mix, but
    #   only in expectation -- at small fractions, sources that are already a
    #   tiny share of the pool (e.g. totalcapture/imuposer, <0.1% each here) can
    #   be sampled down to *zero* windows purely by chance, so the low end of a
    #   pooled sweep can silently drop whole data domains rather than just
    #   seeing "less of the same mix".
    #
    #   mode='stratified' (mm_cfg.stratify_overall_fraction: true): draws
    #   `overall_fraction` independently from *each* source's own train windows
    #   before concatenation, so every enabled source keeps a nonzero,
    #   proportional presence at every fraction, including 0.1%.
    overall_fraction = mm_cfg.get("overall_fraction")
    stratify_fraction = bool(mm_cfg.get("stratify_overall_fraction", False))
    has_fraction = overall_fraction is not None and float(overall_fraction) < 1.0
    scaling_mode = str(mm_cfg.get("scaling_mode", "window_fraction")).lower()
    windows_per_session_k = mm_cfg.get("windows_per_session")
    seed = int(cfg_get(cfg, ["trainer", "seed"], 42))

    if scaling_mode == "windows_per_session":
        if windows_per_session_k is None:
            raise ValueError(
                "[combined_mae] scaling_mode='windows_per_session' requires "
                "mm_cfg.windows_per_session (int)"
            )
        names = list(train_counts.keys())
        new_parts = []
        for name, part in zip(names, train_parts):
            sub, warn = _windows_per_session_subset(name, part, int(windows_per_session_k), seed=seed)
            if sub is None:
                logger.warning("[combined_mae] %s", warn)
                sub = part
            new_parts.append(sub)
        train_parts = new_parts
        train_counts = {name: len(part) for name, part in zip(names, train_parts)}
        combined_train = ConcatDataset(train_parts)
        logger.info(
            "[combined_mae] scaling_mode=windows_per_session (k=%d) -> train=%d/%d windows "
            "across %d sources: %s (val unchanged, %d)",
            int(windows_per_session_k), sum(train_counts.values()), total_train, len(train_counts),
            ", ".join(f"{k}={v}" for k, v in train_counts.items()), total_val,
        )
    elif scaling_mode in ("session_fraction", "subject_fraction") and has_fraction:
        level = "session" if scaling_mode == "session_fraction" else "subject"
        names = list(train_counts.keys())
        new_parts = []
        for name, part in zip(names, train_parts):
            sub, warn = _group_fraction_subset(name, part, float(overall_fraction), level=level, seed=seed)
            if sub is None:
                logger.warning("[combined_mae] %s", warn)
                sub = _fraction_subset(part, float(overall_fraction), seed=seed)
            new_parts.append(sub)
        train_parts = new_parts
        train_counts = {name: len(part) for name, part in zip(names, train_parts)}
        combined_train = ConcatDataset(train_parts)
        logger.info(
            "[combined_mae] scaling_mode=%s overall_fraction=%.4g -> train=%d/%d windows "
            "across %d sources: %s (val unchanged, %d)",
            scaling_mode, float(overall_fraction), sum(train_counts.values()), total_train,
            len(train_counts), ", ".join(f"{k}={v}" for k, v in train_counts.items()), total_val,
        )
    elif scaling_mode == "source_balanced_fraction" and has_fraction:
        names = list(train_counts.keys())
        target_total = round(total_train * float(overall_fraction))
        per_source_budget = max(1, target_total // max(1, len(names)))
        new_parts = []
        for name, part in zip(names, train_parts):
            k = min(len(part), per_source_budget)
            fraction_eff = k / max(1, len(part))
            new_parts.append(_fraction_subset(part, fraction_eff, seed=seed))
        train_parts = new_parts
        train_counts = {name: len(part) for name, part in zip(names, train_parts)}
        combined_train = ConcatDataset(train_parts)
        logger.info(
            "[combined_mae] scaling_mode=source_balanced_fraction overall_fraction=%.4g "
            "(target %d windows/source) -> train=%d/%d windows across %d sources: %s (val unchanged, %d)",
            float(overall_fraction), per_source_budget, sum(train_counts.values()), total_train,
            len(train_counts), ", ".join(f"{k}={v}" for k, v in train_counts.items()), total_val,
        )
    elif scaling_mode == "index_file":
        # Diversity-controlled subsets (EXPERIMENTS.md diversity-saturation
        # experiment): train on an explicit, precomputed list of window indices
        # into the concatenated cache-order train pool. The index file is
        # produced by analysis/diversity_saturation/01_embed_select_analyze.py,
        # which iterates the SAME ConcatDataset built by this function from the
        # SAME config (cache path, no provenance), so indices correspond by
        # construction. Deliberately NOT in _needs_provenance_mode: this mode
        # must keep the cache path so the index space matches.
        subset_file = mm_cfg.get("subset_indices_file")
        if not subset_file:
            raise ValueError(
                "[combined_mae] scaling_mode='index_file' requires "
                "mm_cfg.subset_indices_file (path to .npy int index array)"
            )
        idx = np.load(subset_file)
        combined_train = ConcatDataset(train_parts)
        if idx.ndim != 1 or len(idx) == 0:
            raise ValueError(f"[combined_mae] bad index file {subset_file}: shape {idx.shape}")
        if int(idx.min()) < 0 or int(idx.max()) >= len(combined_train):
            raise ValueError(
                f"[combined_mae] index file {subset_file} range "
                f"[{int(idx.min())}, {int(idx.max())}] does not fit train pool "
                f"of {len(combined_train)} windows -- pool/config mismatch?"
            )
        combined_train = torch.utils.data.Subset(
            combined_train, sorted(int(i) for i in idx))
        logger.info(
            "[combined_mae] scaling_mode=index_file file=%s -> train=%d/%d windows "
            "(val unchanged, %d)",
            subset_file, len(combined_train), total_train, total_val,
        )
    elif has_fraction and stratify_fraction:
        names = list(train_counts.keys())
        train_parts = [_fraction_subset(part, float(overall_fraction), seed=seed)
                       for part in train_parts]
        train_counts = {name: len(part) for name, part in zip(names, train_parts)}
        combined_train = ConcatDataset(train_parts)
        logger.info(
            "[combined_mae] overall_fraction=%.4g (stratified per-source) -> train subsampled to "
            "%d/%d windows across %d sources: %s (val unchanged, %d)",
            float(overall_fraction), sum(train_counts.values()), total_train, len(train_counts),
            ", ".join(f"{k}={v}" for k, v in train_counts.items()), total_val,
        )
    else:
        combined_train = ConcatDataset(train_parts)
        if has_fraction:
            combined_train = _fraction_subset(combined_train, float(overall_fraction), seed=seed)
            logger.info(
                "[combined_mae] overall_fraction=%.4g (pooled) -> train subsampled to %d/%d windows "
                "(val unchanged, %d)",
                float(overall_fraction), len(combined_train), total_train, total_val,
            )

    combined_val = ConcatDataset(val_parts) if val_parts else combined_train

    if collect_provenance:
        # Cheap (just re-walks the already-in-memory provenance attributes) --
        # only computed when explicitly requested, so it costs nothing on
        # normal training runs.
        summary_lines = []
        for name, part in zip(train_counts.keys(), train_parts):
            groups = _group_windows_by_session(name, part)
            if groups is None:
                summary_lines.append(f"{name}: windows={len(part)} sessions=n/a subjects=n/a")
                continue
            n_sessions = len(groups)
            n_subjects = len({_subject_key_from_session_key(name, k) for k in groups})
            summary_lines.append(f"{name}: windows={len(part)} sessions={n_sessions} subjects={n_subjects}")
        logger.info(
            "[combined_mae] composition summary (scaling_mode=%s, overall_fraction=%s):\n  %s",
            scaling_mode, overall_fraction, "\n  ".join(summary_lines),
        )

    num_workers = int(cfg_get(cfg, ["data", "num_workers"], 4))
    pin_memory = bool(cfg_get(cfg, ["data", "pin_memory"], True))
    batch_size = int(cfg_get(cfg, ["data", "batch_size"], 256))
    eval_bs = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    persistent = num_workers > 0

    sampling_mode = str(mm_cfg.get("sampling_mode", "balanced")).lower()
    source_weights = mm_cfg.get("source_weights", {}) or {}
    pooled_window_fraction = (
        (has_fraction and not stratify_fraction and scaling_mode == "window_fraction")
        or scaling_mode == "index_file"
    )
    if pooled_window_fraction and sampling_mode == "balanced":
        raise ValueError(
            "[combined_mae] pooled overall_fraction (and index_file mode) is only supported with sampling_mode='natural' -- "
            "the balanced-mode WeightedRandomSampler is built from full per-source counts and "
            "would go out of range against the fraction-subsampled combined dataset. Every other "
            "scaling_mode (session_fraction/subject_fraction/windows_per_session/"
            "source_balanced_fraction) rebuilds train_parts per-source, so balanced sampling works "
            "fine with those."
        )
    if sampling_mode == "balanced":
        # Equal expected mass per enabled source per epoch (generalizes the
        # 50/50 WeightedRandomSampler used by mae_realdata_ocdetect_flu to N sources),
        # optionally reweighted via source_weights.
        weights: List[float] = []
        for name, part in zip(train_counts.keys(), train_parts):
            n = len(part)
            w = float(source_weights.get(name, 1.0)) / max(n, 1)
            weights.extend([w] * n)
        # Epoch length: by default n_sources x smallest source, which is tiny when one source
        # has a few hundred windows (imuposer: 137) -> thousands of micro-epochs. `epoch_windows`
        # overrides it (sampling with replacement, so the value is free to choose).
        num_samples = int(mm_cfg.get("epoch_windows", 0)) or len(train_counts) * min(train_counts.values())
        sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)
        train_sampler, train_shuffle = sampler, False
        logger.info("[combined_mae] sampler=balanced across %d sources (epoch=%d windows)",
                    len(train_counts), num_samples)
    else:
        train_sampler, train_shuffle = None, True
        logger.info("[combined_mae] sampler=natural ratio (epoch=%d windows)", len(combined_train))

    train_collate, val_collate = _make_collate_fns(var_len, vl_min, vl_max, patch_size)

    result = {
        "train_loader": DataLoader(
            combined_train, batch_size=batch_size,
            sampler=train_sampler, shuffle=train_shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=True, persistent_workers=persistent,
            collate_fn=train_collate,
        ),
        "val_loader": DataLoader(
            combined_val, batch_size=eval_bs, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=False, persistent_workers=persistent,
            collate_fn=val_collate,
        ),
    }
    if collect_provenance:
        # Programmatic counterpart to the composition summary log above, for
        # analysis/mae_scaling_diagnostics scripts -- {name: {"train_n", "val_n"}}
        # post-scaling (i.e. reflects whatever scaling_mode/overall_fraction did).
        result["composition"] = {
            name: {"train_n": len(part), "val_n": val_counts.get(name, 0)}
            for name, part in zip(train_counts.keys(), train_parts)
        }
    return result
