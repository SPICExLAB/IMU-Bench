"""Nymeria real-wrist IMU <-> motion-narration caption dataloader — FLU frame.

Pairs the same real-wrist IMU signal used throughout this benchmark's Nymeria
tasks (`nymeria_real_wrist_loader_FLU.py`'s FLU-conversion pipeline,
duplicated here per this repo's established convention of copying the FLU
math per-loader rather than sharing it — see that file's own docstring for
why) with Nymeria's official motion-narration captions
(`narration/motion_narration.csv`, downloaded via
`scripts/download_nymeria_captions.py` from the manifest at
`nymeria_download_urls_caption.json`) for a CLIP-style IMU<->text alignment
task, mirroring `humanml3d_dataset_FLU.py`'s `HumanML3DIMUVarLenDatasetFLU`
API/shape exactly (same `__iter__` yield: `(x [B,6,T], text_list_or_embs)`).

Caption CSV format (one row per ~5s annotated segment, four independent
free-text description columns per row -- body posture, hands/arms motion,
legs/feet motion, focus/attention):

    request_id,gaia_id,start_time,end_time,annotator,creation_time,
    Describe my body posture,Describe my hands/arms motion,
    Describe my legs/feet motion,Describe my focus attention

``start_time``/``end_time`` are in the *raw recording's absolute clock*, not
the processed .pkl's own zero-based relative clock -- converting between the
two requires subtracting the pkl's own ``reference_time`` scalar (verified
empirically: for a co-located caption+pkl pair, ``start_time - reference_time``
lands inside ``[0, duration_seconds]``). Native wrist IMU rate in these pkls
is exactly 50 Hz (verified: ``wrist_imu_sampling_rate_hz == 50.0``, median
inter-sample dt == 0.02s), matching this benchmark's ``target_hz`` default, so
time -> frame-index conversion is a direct ``round(t * target_hz)`` with no
resampling/downsampling needed.

Unlike the arm-pose/full-body-pose Nymeria loaders, no SMPL IK/FK or pose
labels are needed here (language alignment has no pose target) -- so this
loader never touches ``xsens_pose``/``core.articulate``, only
``wrist_imu_data`` (accel/gyro) and ``wrist_gt_data`` (orientation, needed to
rotate the local CPF-frame IMU into FLU).

Train/val split reuses the exact same session-level boundary already
verified leakage-free against MAE pretraining elsewhere in this benchmark
(``processed_data_body_Ty_rwrist`` = 867 train sessions,
``processed_data_test_body_Ty_rwrist`` = 101 val sessions) -- a sequence is
usable here only if it has both a caption CSV *and* a pkl in the
corresponding root.
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import pickle
import random as pyrandom
from os.path import join as pjoin
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)

_TARGET_FPS = 50

# FLU frame definition -- identical to nymeria_real_wrist_loader_FLU.py
#   Forward (FLU x) = -x_wrist_cpf
#   Left    (FLU y) = +z_wrist_cpf
#   Up      (FLU z) = +y_wrist_cpf
R_WRIST_FROM_FLU = np.array(
    [
        [-1.0, 0.0, 0.0],
        [ 0.0, 0.0, 1.0],
        [ 0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

_DESCRIPTION_COLUMNS = [
    "Describe my body posture",
    "Describe my hands/arms motion",
    "Describe my legs/feet motion",
    "Describe my focus attention",
]

# Registry of Nymeria's official narration types (downloaded via
# scripts/download_nymeria_captions.py --file-key/--out-filename). All three
# share the same request_id/gaia_id/start_time/end_time/annotator/
# creation_time schema, differing only in filename and description column(s)
# -- verified directly against downloaded sample files, not assumed. None of
# the three is "motion-only": all reference rooms/objects/scene context just
# as much as motion_narration.csv does.
#   motion_narration: 236/1100 sequences, 4 aspect-columns, ~5s segments.
#   atomic_action: 845/1100 sequences (3.6x more coverage), 1 longer
#     free-text column, ~5s segments.
#   activity_summarization: 864/1100 sequences, 1 column, ~30s segments
#     (coarser activity-level summaries).
NYMERIA_CAPTION_SOURCES: Dict[str, Dict[str, Any]] = {
    "motion_narration": {
        "filename": "motion_narration.csv",
        "columns": _DESCRIPTION_COLUMNS,
    },
    "atomic_action": {
        "filename": "atomic_action.csv",
        "columns": ["Describe my atomic actions"],
    },
    "activity_summarization": {
        "filename": "activity_summarization.csv",
        "columns": ["Describe my activity"],
    },
    # LLM-augmented sibling of "atomic_action": adds a second, short,
    # HumanML3D-style motion-only caption per row alongside the original
    # detailed one (see scripts/generate_simplified_nymeria_captions.py +
    # scripts/apply_simplified_nymeria_captions.py). Both columns feed into
    # `caps`, so a sample randomly gets either style per draw -- same
    # multi-caption convention as motion_narration's 4 aspect-columns.
    "atomic_action_augmented": {
        "filename": "atomic_action_augmented.csv",
        "columns": ["Describe my atomic actions", "Describe my atomic actions (simplified)"],
    },
    # Narration-QUALITY experiment (2026-07-28): restrict motion_narration to
    # the single aspect column a WRIST-worn IMU can actually observe. The
    # full 4-column source randomly serves posture/legs/attention captions
    # ("C is standing", "C focuses on the laptop") that are invisible to a
    # wrist sensor -- for those draws the caption<->signal correspondence is
    # near-zero and the contrastive loss trains on noise. Motivated by the
    # diagnosis of UniMTS's flat ~2% val retrieval on
    # atomic_action_augmented (job 7679671) and its overfit-from-step-500
    # behavior on motion_narration (job 7581974).
    "motion_narration_arms": {
        "filename": "motion_narration.csv",
        "columns": ["Describe my hands/arms motion"],
    },
}


def _quat_xyzw_to_rotmat(quat: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as ScipyRot
    return ScipyRot.from_quat(quat).as_matrix().astype(np.float64)


# ---------------------------------------------------------------------------
# Caption CSV parsing
# ---------------------------------------------------------------------------

def _parse_caption_csv(
    csv_path: str,
    columns: List[str] = _DESCRIPTION_COLUMNS,
    merge_consecutive: bool = True,
    max_gap_seconds: float = 1.0,
) -> List[Tuple[float, float, List[str]]]:
    """Parse one Nymeria narration CSV into ``[(start_abs, end_abs, caps), ...]``.

    ``columns``: which description column(s) to read (default: the 4
    aspect-columns of ``motion_narration.csv``; pass
    ``NYMERIA_CAPTION_SOURCES["atomic_action"]["columns"]`` etc. for the
    other narration types, which have a single column instead). ``caps``
    collects every non-empty description across the given columns for that
    row (mirrors HumanML3D's multi-caption convention -- caller picks one at
    random per sample). Rows where every column is blank are skipped
    entirely (no fallback to ``['']`` here, unlike HumanML3D, since a
    segment with zero real captions carries no training signal and there
    are plenty of other segments).

    ``merge_consecutive`` (default ``True``): Nymeria's raw ~5s-per-row
    annotation granularity means a single sustained activity (e.g. "sitting
    playing an arcade game" for a minute) shows up as many consecutive rows
    with identical or near-identical captions -- confirmed empirically (only
    ~64% of val-split captions are unique; the single most common val
    caption repeats 51 times). This both floods the training/eval pool with
    duplicate captions (a real, measured contributor to this task's low
    retrieval accuracy -- 9-15% of any random batch has a near-duplicate
    caption elsewhere in the batch, cosine sim 0.90-0.95, that the
    multi-positive loss's tight 0.999 threshold treats as a false negative)
    and produces many short, low-context segments. Fixing it at the source
    (fewer, longer, more distinct segments) is more principled than papering
    over it downstream in the loss. Rows are sorted by ``start_time``, then
    consecutive rows are coalesced into one segment when their ``caps``
    lists are *exactly* identical (same annotation content, not just
    similar) and the gap between one row's end and the next's start is
    ``<= max_gap_seconds`` (guards against accidentally merging two
    unrelated occurrences of a generic caption far apart in time). Merging
    is conservative by design: it only ever reduces row count for truly
    contiguous identical annotations, never joins content-different or
    time-distant rows, and never fabricates or alters any caption text.
    """
    out: List[Tuple[float, float, List[str]]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start_abs = float(row["start_time"])
                end_abs = float(row["end_time"])
            except (KeyError, ValueError):
                continue
            if end_abs <= start_abs:
                continue
            caps = [
                row[col].strip()
                for col in columns
                if row.get(col) and row[col].strip()
            ]
            if not caps:
                continue
            out.append((start_abs, end_abs, caps))

    out.sort(key=lambda x: x[0])

    if not merge_consecutive or not out:
        return out

    merged: List[Tuple[float, float, List[str]]] = [out[0]]
    for start, end, caps in out[1:]:
        prev_start, prev_end, prev_caps = merged[-1]
        if caps == prev_caps and start - prev_end <= max_gap_seconds:
            merged[-1] = (prev_start, max(prev_end, end), prev_caps)
        else:
            merged.append((start, end, caps))
    return merged


# ---------------------------------------------------------------------------
# Session (.pkl) loader -- FLU wrist IMU only, no pose/labels
# ---------------------------------------------------------------------------

def _load_nymeria_session_imu_flu(
    pkl_path: str,
    target_hz: int = _TARGET_FPS,
    add_gravity: bool = True,
) -> Optional[Dict[str, Any]]:
    """Load one Nymeria .pkl's real wrist IMU, converted to the FLU frame.

    Same CPF-local -> world -> FLU pipeline as
    ``nymeria_real_wrist_loader_FLU.py::_load_nymeria_real_wrist_session_flu``,
    minus the xsens_pose/SMPL IK+FK label branch (not needed for language
    alignment) -- duplicated per this repo's established convention rather
    than imported, so this loader has no dependency on ``core.articulate``.

    Returns ``{'imu': [N,6] float32 FLU, 'reference_time': float,
    'duration_seconds': float, 'n_frames': int}`` or ``None`` if unusable.
    """
    try:
        with open(pkl_path, "rb") as fh:
            raw = pickle.load(fh)
    except Exception as exc:
        logger.warning("[nymeria_caption_flu] cannot open %s: %s", pkl_path, exc)
        return None

    if not raw.get("has_wrist_imu", False):
        return None

    wrist_imu = raw.get("wrist_imu_data", {})
    wrist_gt = raw.get("wrist_gt_data", {})
    if "accel" not in wrist_imu or "gyro" not in wrist_imu or "orientation" not in wrist_gt:
        return None

    def _to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    wrist_fps_raw = float(raw.get("wrist_imu_sampling_rate_hz", target_hz))
    wrist_ds = max(1, round(wrist_fps_raw / target_hz))

    acc_local = _to_numpy(wrist_imu["accel"]).astype(np.float64)[::wrist_ds]   # [N, 3]
    gyro_local = _to_numpy(wrist_imu["gyro"]).astype(np.float64)[::wrist_ds]   # [N, 3]

    orient_raw = wrist_gt["orientation"]
    try:
        import pypose as pp
        _is_lietensor = isinstance(orient_raw, pp.LieTensor)
    except ImportError:
        _is_lietensor = False
    if _is_lietensor:
        R_world_from_wrist_cpf = orient_raw.matrix().detach().cpu().numpy().astype(np.float64)[::wrist_ds]
    else:
        orient_np = _to_numpy(orient_raw).astype(np.float64)[::wrist_ds]
        R_world_from_wrist_cpf = _quat_xyzw_to_rotmat(orient_np)

    N = min(acc_local.shape[0], gyro_local.shape[0], R_world_from_wrist_cpf.shape[0])
    if N < 3:
        return None
    acc_local = acc_local[:N]
    gyro_local = gyro_local[:N]
    R_world_from_wrist_cpf = R_world_from_wrist_cpf[:N]

    acc_world = np.einsum("nij,nj->ni", R_world_from_wrist_cpf, acc_local)
    gyro_world = np.einsum("nij,nj->ni", R_world_from_wrist_cpf, gyro_local)

    if not add_gravity:
        acc_world = acc_world + np.array([0.0, -9.81, 0.0], dtype=np.float64)

    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_wrist_cpf, R_WRIST_FROM_FLU)
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)

    acc_flu = np.einsum("nij,nj->ni", R_X_from_world, acc_world)
    gyro_flu = np.einsum("nij,nj->ni", R_X_from_world, gyro_world)
    imu = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)  # [N, 6]

    return {
        "imu": imu,
        "reference_time": float(raw.get("reference_time", 0.0)),
        "duration_seconds": float(raw.get("duration_seconds", N / target_hz)),
        "n_frames": N,
    }


# XSens right-wrist joint index — same convention as nymeria_syn_wrist_loader_FLU.py
# (NOT the SMPL joint-21 index used elsewhere; this is XSens's own 23-joint skeleton).
_XSENS_RIGHT_WRIST_JOINT = 9

# Same FLU convention as nymeria_syn_wrist_loader_FLU.py (verified identical to the
# R_WRIST_FROM_FLU constant at the top of this file).
_SYN_R_WRIST_FROM_FLU = R_WRIST_FROM_FLU


def _load_nymeria_session_imu_flu_synthetic(
    pkl_path: str,
    target_hz: int = _TARGET_FPS,
    add_gravity: bool = True,
) -> Optional[Dict[str, Any]]:
    """Synthesize the right-wrist IMU from Nymeria's ground-truth XSens mocap
    (``gt_data['xsens_pose']``) instead of reading the real recorded hardware
    IMU (``wrist_imu_data``) -- the Nymeria analogue of how
    ``humanml3d_dataset_FLU.py`` derives HumanML3D's fully-synthetic IMU from
    clean SMPL mocap. Math identical to
    ``nymeria_syn_wrist_loader_FLU.py::_load_nymeria_syn_wrist_session_flu``
    (central-difference acceleration + gravity bias, SO3-log angular velocity,
    world->FLU rotation via joint 9's own orientation), but returns the same
    ``{'imu', 'reference_time', 'duration_seconds', 'n_frames'}`` dict shape as
    ``_load_nymeria_session_imu_flu`` above so it's a drop-in replacement for
    caption-alignment/windowing -- only the IMU *source* changes, nothing about
    how captions are matched to frames.

    Built to test whether Nymeria's language-alignment task is bottlenecked by
    real-sensor noise (this function removes it) vs. by the looser
    caption-to-segment temporal/semantic correspondence (this function does
    NOT fix that -- captions are still the same human annotations against the
    same segment timestamps).
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    try:
        with open(pkl_path, "rb") as fh:
            raw = pickle.load(fh)
    except Exception as exc:
        logger.warning("[nymeria_caption_flu] cannot open %s: %s", pkl_path, exc)
        return None

    xsens_pose = raw.get("gt_data", {}).get("xsens_pose")
    if xsens_pose is None:
        return None
    if hasattr(xsens_pose, "detach"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float64)  # [N, 23, 4, 4]
    if xsens_pose.shape[0] < 3:
        return None

    fps_raw = float(raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or target_hz)
    ds = max(1, round(fps_raw / target_hz))
    xsens_pose = xsens_pose[::ds]
    N = xsens_pose.shape[0]
    if N < 3:
        return None
    fps = float(target_hz)

    j = _XSENS_RIGHT_WRIST_JOINT
    pos = xsens_pose[:, j, :3, 3].copy()    # [N, 3] world-frame positions
    rot = xsens_pose[:, j, :3, :3].copy()   # [N, 3, 3] R_world_from_wrist

    acc_world = np.zeros((N, 3), dtype=np.float64)
    acc_world[1:-1] = (pos[2:] - 2.0 * pos[1:-1] + pos[:-2]) * (fps ** 2)
    acc_world[0] = acc_world[1]
    acc_world[-1] = acc_world[-2]
    if add_gravity:
        acc_world[:, 1] += 9.81  # Y-up world; real static IMU reads +g in the up axis

    dR_world = np.einsum("nij,nkj->nik", rot[1:], rot[:-1])
    gyro_raw = ScipyRot.from_matrix(dR_world).as_rotvec().astype(np.float64) * fps
    gyro_world = np.zeros((N, 3), dtype=np.float64)
    gyro_world[:-1] = gyro_raw
    gyro_world[-1] = gyro_raw[-1]

    R_world_from_FLU = np.einsum("nij,jk->nik", rot, _SYN_R_WRIST_FROM_FLU)
    R_FLU_from_world = R_world_from_FLU.transpose(0, 2, 1)
    acc_flu = np.einsum("nij,nj->ni", R_FLU_from_world, acc_world)
    gyro_flu = np.einsum("nij,nj->ni", R_FLU_from_world, gyro_world)
    imu = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)  # [N, 6]

    return {
        "imu": imu,
        "reference_time": float(raw.get("reference_time", 0.0)),
        "duration_seconds": float(raw.get("duration_seconds", N / target_hz)),
        "n_frames": N,
    }


# ---------------------------------------------------------------------------
# Sequence discovery (captions x pkl availability)
# ---------------------------------------------------------------------------

_PKL_SUFFIX = "_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl"


def _pkl_path_for_sequence(pkl_root: str, sid: str) -> Optional[str]:
    p = pjoin(pkl_root, f"{sid}{_PKL_SUFFIX}")
    return p if os.path.exists(p) else None


def _discover_sequences(
    captions_root: str, pkl_root: str, caption_filename: str = "motion_narration.csv",
) -> Dict[str, str]:
    """Return ``{sequence_id: pkl_path}`` for sequences with both a
    ``caption_filename`` under ``captions_root`` and a matching .pkl under
    ``pkl_root``."""
    out: Dict[str, str] = {}
    for csv_path in sorted(glob.glob(pjoin(captions_root, "*", caption_filename))):
        sid = os.path.basename(os.path.dirname(csv_path))
        pkl_path = _pkl_path_for_sequence(pkl_root, sid)
        if pkl_path is not None:
            out[sid] = pkl_path
    return out


# ---------------------------------------------------------------------------
# Iterable variable-length Dataset
# ---------------------------------------------------------------------------

class NymeriaCaptionVarLenDatasetFLU(IterableDataset):
    """Infinite IterableDataset with variable T per batch -- Nymeria FLU wrist IMU.

    API-compatible with ``HumanML3DIMUVarLenDatasetFLU``: each batch yields
    ``(x [B, 6, T], texts List[str])`` without a text cache, or
    ``(x [B, 6, T], embs FloatTensor [B, D])`` with one.

    Sampling: builds a flat pool of every (sequence, caption-row) pair at
    construction time (cheap -- just CSV parsing, no .pkl access), then
    samples uniformly from that pool per item -- so sessions with more
    annotated segments aren't systematically under- or over-weighted
    relative to their actual annotation density. Each drawn caption-row's
    time range is converted to a frame-index slice of that session's FLU IMU
    the first time the session is touched (lazy, per-worker cache, mirroring
    ``HumanML3DIMUVarLenDatasetFLU``'s own ``get_clip`` pattern) -- avoids
    eagerly unpickling every session (each pkl's full object graph includes
    heavy pose-label tensors this task never uses) at construction time.
    """

    def __init__(
        self,
        captions_root: str,
        pkl_root: str,
        split: str = "train",
        batch_size: int = 32,
        patch_size: int = 16,
        T_min: int = 64,
        T_max: int = 512,
        target_hz: int = _TARGET_FPS,
        add_gravity: bool = True,
        seed: int = 42,
        text_emb_cache: Optional[Dict[str, np.ndarray]] = None,
        pad_mode: str = "zero",
        cfg: Optional[Any] = None,
        caption_source: str = "motion_narration",
        imu_source: str = "real",
    ):
        super().__init__()
        self.captions_root = captions_root
        self.pkl_root = pkl_root
        self.split = split
        self.batch_size = batch_size
        self.pad_mode = pad_mode
        self.patch_size = patch_size
        self.target_hz = target_hz
        self.add_gravity = add_gravity
        self.T_min = T_min
        self.T_max = T_max
        self.seed = seed
        self.cfg = cfg
        self._text_emb_cache = text_emb_cache

        if imu_source not in ("real", "synthetic"):
            raise ValueError(f"Unknown imu_source={imu_source!r}, expected 'real' or 'synthetic'")
        self.imu_source = imu_source

        if caption_source not in NYMERIA_CAPTION_SOURCES:
            raise ValueError(
                f"Unknown caption_source={caption_source!r}, expected one of "
                f"{list(NYMERIA_CAPTION_SOURCES)}"
            )
        caption_filename = NYMERIA_CAPTION_SOURCES[caption_source]["filename"]
        caption_columns = NYMERIA_CAPTION_SOURCES[caption_source]["columns"]

        if text_emb_cache is not None:
            _sample_emb = next(iter(text_emb_cache.values()))
            self._emb_dim = int(_sample_emb.shape[0])
            self._zero_emb = np.zeros(self._emb_dim, dtype=np.float32)
        else:
            self._emb_dim = None
            self._zero_emb = None

        self.allowed_Ts: List[int] = list(range(T_min, T_max + 1, patch_size))
        if not self.allowed_Ts:
            raise ValueError(f"No valid T in [{T_min}, {T_max}] with patch_size={patch_size}")

        seq_to_pkl = _discover_sequences(captions_root, pkl_root, caption_filename)
        self._pkl_path_by_seq: Dict[str, str] = seq_to_pkl

        # Flat (sid, row_idx) pool -- cheap, CSV-only, no .pkl access yet.
        self._rows_by_seq: Dict[str, List[Tuple[float, float, List[str]]]] = {}
        self._flat_pool: List[Tuple[str, int]] = []
        for sid in sorted(seq_to_pkl.keys()):
            csv_path = pjoin(captions_root, sid, caption_filename)
            rows = _parse_caption_csv(csv_path, columns=caption_columns)
            if not rows:
                continue
            self._rows_by_seq[sid] = rows
            self._flat_pool.extend((sid, i) for i in range(len(rows)))

        if not self._flat_pool:
            raise RuntimeError(
                f"NymeriaCaptionVarLenDatasetFLU: no usable (sequence, caption) "
                f"pairs found under captions_root={captions_root} pkl_root={pkl_root} "
                f"caption_source={caption_source!r}"
            )

        logger.info(
            "[NymeriaCaptionVarLenDatasetFLU] split=%s  caption_source=%s  imu_source=%s  "
            "sequences=%d  caption_rows=%d  T=[%d,%d]  bs=%d  (lazy, FLU frame)",
            split, caption_source, imu_source, len(self._rows_by_seq), len(self._flat_pool),
            T_min, T_max, batch_size,
        )

    # ------------------------------------------------------------------
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed + (worker_info.id if worker_info is not None else 0)
        rng = np.random.default_rng(worker_seed)
        py_rng = pyrandom.Random(int(worker_seed))

        cache: Dict[str, Optional[Dict]] = {}

        _load_fn = (
            _load_nymeria_session_imu_flu_synthetic
            if self.imu_source == "synthetic"
            else _load_nymeria_session_imu_flu
        )

        def get_session(sid: str) -> Optional[Dict]:
            if sid not in cache:
                cache[sid] = _load_fn(
                    self._pkl_path_by_seq[sid], self.target_hz, self.add_gravity,
                )
            return cache[sid]

        apply_augment = None
        if self.cfg is not None:
            from imu_lm.data.augmentations.transform import apply_augment as _apply_augment
            apply_augment = _apply_augment

        pool = self._flat_pool

        while True:
            T = int(rng.choice(self.allowed_Ts))

            x_list: List[np.ndarray] = []
            text_list: List[str] = []
            attempts = 0

            while len(x_list) < self.batch_size:
                attempts += 1
                if attempts > self.batch_size * 50:
                    break

                sid, row_idx = pool[int(rng.integers(len(pool)))]
                session = get_session(sid)
                if session is None:
                    continue

                start_abs, end_abs, caps = self._rows_by_seq[sid][row_idx]
                start_rel = start_abs - session["reference_time"]
                end_rel = min(end_abs - session["reference_time"], session["duration_seconds"])
                if start_rel < 0 or end_rel <= start_rel:
                    continue

                start_idx = int(round(start_rel * self.target_hz))
                end_idx = min(int(round(end_rel * self.target_hz)), session["n_frames"])
                if end_idx - start_idx < self.patch_size:
                    continue

                seg = session["imu"][start_idx:end_idx]   # [seg_T, 6]
                seg_T = seg.shape[0]
                cap = py_rng.choice(caps)

                if seg_T >= T:
                    max_start = seg_T - T
                    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                    window = seg[start:start + T]
                    mean = window.mean(axis=0, keepdims=True)
                    std = window.std(axis=0, keepdims=True) + 1e-8
                    window = (window - mean) / std
                else:
                    mean = seg.mean(axis=0, keepdims=True)
                    std = seg.std(axis=0, keepdims=True) + 1e-8
                    normed = (seg - mean) / std
                    if self.pad_mode == "wrap":
                        window = np.pad(normed, ((0, T - seg_T), (0, 0)), mode="wrap").astype(np.float32)
                    else:
                        window = np.zeros((T, 6), dtype=np.float32)
                        window[:seg_T] = normed

                if apply_augment is not None:
                    window = apply_augment(window, self.cfg)

                x_list.append(window.T)   # [6, T]
                text_list.append(cap)

            if len(x_list) < self.batch_size:
                continue

            x_tensor = torch.from_numpy(np.stack(x_list, axis=0)).float()  # [B, 6, T]

            if self._text_emb_cache is not None:
                emb_list = [self._text_emb_cache.get(cap, self._zero_emb) for cap in text_list]
                yield x_tensor, torch.from_numpy(np.stack(emb_list, axis=0)).float()
            else:
                yield x_tensor, text_list


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_nymeria_caption_imu_loaders_flu(
    captions_root: str,
    train_pkl_root: str,
    val_pkl_root: str,
    cfg: Any = None,
    patch_size: int = 16,
    T_min: int = 64,
    T_max: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    text_emb_cache: Optional[Dict[str, np.ndarray]] = None,
    caption_source: str = "motion_narration",
    imu_source: str = "real",
) -> Dict[str, Optional[DataLoader]]:
    """Build train and val DataLoaders for Nymeria caption<->IMU alignment.

    ``caption_source``: which of ``NYMERIA_CAPTION_SOURCES`` to use
    (default ``'motion_narration'``; also ``'atomic_action'`` or
    ``'activity_summarization'``).

    ``imu_source``: ``'real'`` (default) reads the recorded hardware wrist
    IMU; ``'synthetic'`` instead derives a clean wrist IMU analytically from
    ground-truth XSens mocap (``gt_data['xsens_pose']``), the same way
    HumanML3D's own IMU is synthesized from SMPL mocap -- see
    ``_load_nymeria_session_imu_flu_synthetic``.

    ``cfg`` key overrides, same convention as ``make_humanml3d_imu_loaders_flu``::

        vit1d.patch_size              (int)
        windowing.min_window_seconds  (float)
        windowing.max_window_seconds  (float)
        windowing.pad_mode            (str, 'zero' or 'wrap'; default 'zero')
        data.batch_size               (int)
        data.num_workers              (int)
        trainer.seed                  (int)

    Train uses ``train_pkl_root`` (867-session split), val uses
    ``val_pkl_root`` (101-session split) -- the same session boundary
    already verified leakage-free against MAE pretraining elsewhere in this
    benchmark. Returns a dict with keys ``'train_loader'``/``'val_loader'``.
    """
    _target_fps = _TARGET_FPS
    pad_mode = "zero"
    if cfg is not None:
        from imu_lm.utils.helpers import cfg_get
        vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
        win_cfg = cfg_get(cfg, ["windowing"], {}) or {}
        dat_cfg = cfg_get(cfg, ["data"], {}) or {}
        patch_size = int(vit_cfg.get("patch_size", patch_size))
        T_min = int(round(float(win_cfg.get("min_window_seconds", T_min / _target_fps)) * _target_fps / patch_size)) * patch_size
        T_max = int(round(float(win_cfg.get("max_window_seconds", T_max / _target_fps)) * _target_fps / patch_size)) * patch_size
        batch_size = int(dat_cfg.get("batch_size", batch_size))
        num_workers = int(dat_cfg.get("num_workers", num_workers))
        seed = int(cfg_get(cfg, ["trainer", "seed"], seed))
        pad_mode = str(win_cfg.get("pad_mode", pad_mode))

    loaders: Dict[str, Optional[DataLoader]] = {}
    for split, split_seed, pkl_root in [
        ("train", seed, train_pkl_root),
        ("val", seed + 1000, val_pkl_root),
    ]:
        dataset = NymeriaCaptionVarLenDatasetFLU(
            captions_root=captions_root,
            pkl_root=pkl_root,
            split=split,
            batch_size=batch_size,
            patch_size=patch_size,
            T_min=T_min,
            T_max=T_max,
            seed=split_seed,
            text_emb_cache=text_emb_cache,
            pad_mode=pad_mode,
            cfg=cfg,
            caption_source=caption_source,
            imu_source=imu_source,
        )
        loaders[f"{split}_loader"] = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers if split == "train" else min(num_workers, 2),
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    return loaders
