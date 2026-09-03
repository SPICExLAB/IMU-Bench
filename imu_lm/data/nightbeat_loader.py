"""NightbeatDB loader: overnight wrist accelerometer -> heart rate (bpm).

Nightbeat-DB (Moebus et al., IEEE BHI 2024, https://github.com/eth-siplab/Nightbeat)
is 40 participants x ~7 h of at-home overnight recordings from a wrist-worn
accelerometer, with a chest ECG reference. The task is *regression*: estimate
instantaneous heart rate from the ballistocardiogram (BCG) — the faint
body-recoil signal each heartbeat imparts to the wrist — in the accelerometer.

Run ``scripts/prepare_nightbeat_dataset.py`` first; it writes one
``{pid:02d}.npz`` per participant holding the 50 Hz acc trace, the 50 Hz ECG
validity flag, and beat times/rates. This module cuts windows from those
traces, so window length and hop are config-driven and need no re-prep.

Accelerometer-only source: the gyro mask-token convention
---------------------------------------------------------
NightbeatDB has no gyroscope, but the multidata MAE encoder was pretrained on
6 channels (acc + gyro). This loader follows the exact convention the
pretraining already established for its accelerometer-only sources (capture24,
NHANES — see ``imu_lm.data.combined_mae_multidata_loader.AccOnlyPadTo6ChDataset``
and ``imu_lm.objectives.masked_1d_freq_channel_mask``):

  * windows are zero-padded from 3 to 6 channels, and
  * each sample carries ``channel_valid_mask = [T,T,T,F,F,F]``.

The consumer turns ``~channel_valid_mask`` into a patch mask, so the encoder
replaces every gyro patch embedding with the *learned mask token* rather than
reading the zeros as if they were a real gyroscope at rest. That is precisely
the state the encoder saw for capture24/NHANES gyro throughout pretraining, so
it is in-distribution. ``scripts/eval_nightbeat_hr.py`` implements that step;
``gyro_mode="zero"`` there reproduces the naive alternative for comparison.

Preprocessing (impute/filter/normalize) runs on the 3 real channels *before*
padding — z-scoring an all-zero channel would divide by ``eps``.

Splits are grouped by participant. HR is strongly subject-specific (the paper's
per-participant-mean baseline already reaches 3.16 bpm MAE), so any split that
puts one participant's windows on both sides leaks the answer.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)

NIGHTBEAT_PREPARED_ROOT = "/projects/b53017/vbc1591/Nightbeat/prepared"
CANONICAL_HZ = 50.0

# Physiological guard on the reference HR. Beyond this the ECG beat detector is
# far more likely wrong than the sleeping participant.
HR_MIN_BPM = 30.0
HR_MAX_BPM = 200.0


@dataclass
class NightbeatWindowSpec:
    """One window: which participant, where it starts, and its HR target."""
    pid: int
    start: int          # index into the 50 Hz trace
    hr: float           # bpm, mean of per-beat HR inside the window
    n_beats: int
    hr_std: float       # within-window spread of per-beat HR (bpm)


class NightbeatHRDataset(Dataset):
    """Windows of wrist acc -> HR (bpm).

    Yields ``(x [6, T] float32, y float32, channel_valid_mask [6] bool)``.
    The 3 gyro channels are zeros and flagged invalid — see module docstring.
    """

    def __init__(
        self,
        specs: List[NightbeatWindowSpec],
        traces: Dict[int, np.ndarray],
        cfg: Any,
        T: int,
        pad_to_6ch: bool = True,
    ):
        self.specs = specs
        self.traces = traces
        self.cfg = cfg
        self.T = T
        self.pad_to_6ch = pad_to_6ch
        self.stats = PreprocessStats()

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int):
        spec = self.specs[idx]
        acc = self.traces[spec.pid][:, spec.start : spec.start + self.T]  # [3, T]

        # preprocess_window takes [T, C] and returns [C, T].
        xw = np.ascontiguousarray(acc.T)
        xproc = preprocess_window(xw, self.cfg, self.stats)
        if xproc is None:
            # impute dropped it; fall back to raw so __getitem__ stays total.
            xproc = xw.T.astype(np.float32)

        x3 = torch.from_numpy(np.ascontiguousarray(xproc)).float()  # [3, T]

        if not self.pad_to_6ch:
            return x3, torch.tensor(spec.hr, dtype=torch.float32), \
                torch.ones(3, dtype=torch.bool)

        x6 = torch.zeros((6, x3.shape[1]), dtype=x3.dtype)
        x6[:3] = x3
        channel_valid_mask = torch.tensor([True, True, True, False, False, False])
        return x6, torch.tensor(spec.hr, dtype=torch.float32), channel_valid_mask

    def pids(self) -> np.ndarray:
        return np.array([s.pid for s in self.specs], dtype=np.int64)

    def targets(self) -> np.ndarray:
        return np.array([s.hr for s in self.specs], dtype=np.float32)


def _window_targets(
    beat_t: np.ndarray,
    beat_hr: np.ndarray,
    t0: float,
    t1: float,
) -> Tuple[float, int, float]:
    """Mean per-beat HR over beats falling in [t0, t1). Returns (hr, n, std)."""
    lo = np.searchsorted(beat_t, t0, side="left")
    hi = np.searchsorted(beat_t, t1, side="left")
    if hi <= lo:
        return float("nan"), 0, float("nan")
    hrs = beat_hr[lo:hi]
    good = np.isfinite(hrs)
    if not good.any():
        return float("nan"), 0, float("nan")
    hrs = hrs[good]
    return float(hrs.mean()), int(hrs.size), float(hrs.std())


def load_participant(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as d:
        return {
            "acc": d["acc"],
            "isvalid": d["isvalid"],
            "beat_t": d["beat_t"],
            "beat_hr": d["beat_hr"],
            "pid": int(d["pid"]),
        }


def build_window_specs(
    data: Dict[str, np.ndarray],
    T: int,
    hop: int,
    min_valid_frac: float,
    min_beats: int,
    max_hr_std: float,
) -> List[NightbeatWindowSpec]:
    """Enumerate windows for one participant, keeping only usable ones.

    A window is kept when the ECG reference is valid across (nearly) all of it,
    it contains enough beats to define a rate, the resulting HR is
    physiological, and the HR is not wildly non-stationary inside the window
    (``max_hr_std``) — a window spanning an arousal has no single right answer.
    """
    acc = data["acc"]
    isvalid = data["isvalid"]
    beat_t = data["beat_t"]
    beat_hr = data["beat_hr"]
    pid = data["pid"]

    n = acc.shape[1]
    if n < T:
        return []

    # Cumulative validity gives each window's valid fraction in O(1).
    csum = np.concatenate([[0], np.cumsum(isvalid.astype(np.int64))])

    specs: List[NightbeatWindowSpec] = []
    for start in range(0, n - T + 1, hop):
        end = start + T
        valid_frac = (csum[end] - csum[start]) / float(T)
        if valid_frac < min_valid_frac:
            continue
        hr, n_beats, hr_std = _window_targets(
            beat_t, beat_hr, start / CANONICAL_HZ, end / CANONICAL_HZ
        )
        if n_beats < min_beats or not np.isfinite(hr):
            continue
        if hr < HR_MIN_BPM or hr > HR_MAX_BPM:
            continue
        if np.isfinite(max_hr_std) and hr_std > max_hr_std:
            continue
        specs.append(NightbeatWindowSpec(pid=pid, start=start, hr=hr,
                                         n_beats=n_beats, hr_std=hr_std))
    return specs


def split_participants(
    pids: Sequence[int],
    seed: int = 42,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> Tuple[List[int], List[int], List[int]]:
    """Deterministic participant-level split."""
    pids = sorted(pids)
    rng = np.random.RandomState(seed)
    order = np.array(pids)[rng.permutation(len(pids))]
    n = len(order)
    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    test = sorted(order[:n_test].tolist())
    val = sorted(order[n_test : n_test + n_val].tolist())
    train = sorted(order[n_test + n_val :].tolist())
    return train, val, test


def participant_folds(pids: Sequence[int], n_folds: int, seed: int = 42) -> List[List[int]]:
    """Split participants into ``n_folds`` disjoint test groups (grouped k-fold).

    Every participant lands in exactly one test fold, so pooling predictions
    across folds gives one held-out prediction per window in the dataset.
    """
    pids = sorted(pids)
    rng = np.random.RandomState(seed)
    order = np.array(pids)[rng.permutation(len(pids))]
    return [sorted(order[i::n_folds].tolist()) for i in range(n_folds)]


def make_nightbeat_datasets(
    cfg: Any,
    root: Optional[str] = None,
    pids: Optional[Sequence[int]] = None,
    pad_to_6ch: bool = True,
) -> Tuple[Dict[int, NightbeatHRDataset], Dict[int, np.ndarray], int]:
    """Build one dataset per participant (callers assemble splits from these).

    Returns ``(datasets_by_pid, traces_by_pid, T)``.
    """
    nb_cfg = cfg_get(cfg, ["data", "nightbeat"], {}) or {}
    root = Path(root or nb_cfg.get("root", NIGHTBEAT_PREPARED_ROOT))
    if not root.is_dir():
        raise FileNotFoundError(
            f"Nightbeat prepared root not found: {root}. "
            f"Run scripts/prepare_nightbeat_dataset.py first."
        )

    min_valid_frac = float(nb_cfg.get("min_valid_frac", 1.0))
    min_beats = int(nb_cfg.get("min_beats", 5))
    max_hr_std = float(nb_cfg.get("max_hr_std", 8.0))

    T, hop = compute_T_and_hop(cfg)

    files = sorted(root.glob("[0-9]*.npz"))
    if pids is not None:
        want = set(int(p) for p in pids)
        files = [f for f in files if int(f.stem) in want]
    if not files:
        raise FileNotFoundError(f"no participant npz files under {root}")

    datasets: Dict[int, NightbeatHRDataset] = {}
    traces: Dict[int, np.ndarray] = {}
    n_specs_total = 0
    for f in files:
        data = load_participant(f)
        pid = data["pid"]
        specs = build_window_specs(data, T, hop, min_valid_frac, min_beats, max_hr_std)
        if not specs:
            logger.warning("[nightbeat] pid %02d: no usable windows, skipping", pid)
            continue
        traces[pid] = data["acc"]
        datasets[pid] = NightbeatHRDataset(specs, traces, cfg, T, pad_to_6ch=pad_to_6ch)
        n_specs_total += len(specs)

    logger.info(
        "[nightbeat] %d participants, %d windows (T=%d, hop=%d, %.2fs @ %.0fHz)",
        len(datasets), n_specs_total, T, hop, T / CANONICAL_HZ, CANONICAL_HZ,
    )
    return datasets, traces, T


def concat_datasets(
    datasets: Dict[int, NightbeatHRDataset], pids: Sequence[int]
) -> Optional[NightbeatHRDataset]:
    """Merge per-participant datasets into one (they share the trace dict)."""
    pids = [p for p in pids if p in datasets]
    if not pids:
        return None
    specs: List[NightbeatWindowSpec] = []
    for p in pids:
        specs.extend(datasets[p].specs)
    ref = datasets[pids[0]]
    return NightbeatHRDataset(specs, ref.traces, ref.cfg, ref.T, pad_to_6ch=ref.pad_to_6ch)


def make_nightbeat_loaders(
    cfg: Any,
    root: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Participant-grouped train/val/test DataLoaders."""
    nb_cfg = cfg_get(cfg, ["data", "nightbeat"], {}) or {}
    datasets, _traces, T = make_nightbeat_datasets(cfg, root=root)

    train_p, val_p, test_p = split_participants(
        sorted(datasets.keys()),
        seed=seed,
        val_frac=float(nb_cfg.get("val_frac", 0.2)),
        test_frac=float(nb_cfg.get("test_frac", 0.2)),
    )

    data_cfg = cfg_get(cfg, ["data"], {}) or {}
    bs = int(data_cfg.get("batch_size", 256))
    eval_bs = int(data_cfg.get("eval_batch_size", bs))
    nw = int(data_cfg.get("num_workers", 4))
    pin = bool(data_cfg.get("pin_memory", True))

    out: Dict[str, Any] = {"T": T, "splits": {"train": train_p, "val": val_p, "test": test_p}}
    for name, plist, shuffle, b in (
        ("train_loader", train_p, True, bs),
        ("val_loader", val_p, False, eval_bs),
        ("test_loader", test_p, False, eval_bs),
    ):
        ds = concat_datasets(datasets, plist)
        out[name] = None if ds is None else DataLoader(
            ds, batch_size=b, shuffle=shuffle, num_workers=nw,
            pin_memory=pin, drop_last=False,
        )
    logger.info("[nightbeat] split by participant -> train=%s val=%s test=%s",
                train_p, val_p, test_p)
    return out
