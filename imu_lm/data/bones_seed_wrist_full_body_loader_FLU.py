"""Bones-Seed synthetic right-wrist IMU loader in FLU frame.

This loader reads Bones-Seed BVH body motion, runs simple forward kinematics,
uses the right hand/wrist trajectory to synthesize IMU signals, and returns
sliding windows compatible with the existing pose-regression loaders.

Input:
    x: [6, T] float32 = [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]

Label:
    y: [N_patches, J * 6] float32 = local BVH joint rotations in 6D.

The loader works best with an extracted BVH directory, e.g.
``/projects/b53017/vbc1591/bones-seed/soma_proportional/bvh``.  It can also
read ``tar.gz::member`` references, which is useful for quick checks but slow
for full training because gzip tar archives do not support efficient random
access.
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRot
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from imu_lm.data.synthetic_imu_sanitize import sanitize_synthetic_imu
from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.augmentations.transform import apply_augment
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)

ROT6D_DIM = 6

# A compact 24-joint body set, ordered from root/trunk to limbs.  These are BVH
# joint names, not SMPL joint indices.
DEFAULT_BODY_JOINTS_24 = [
    "Root",
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
]

RIGHT_WRIST_CANDIDATES = ("RightHand", "RightWrist", "RightForeArm")

# Same right-wrist FLU definition as the other *_FLU loaders.
R_WRIST_FROM_FLU = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass
class _BvhJoint:
    name: str
    parent: int
    offset: np.ndarray
    channels: List[str]
    channel_start: int


@dataclass
class _BvhMotion:
    joints: List[_BvhJoint]
    frames: np.ndarray
    fps: float


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def _rot6d_to_mat(r6d: np.ndarray) -> np.ndarray:
    a1 = r6d[:3].astype(np.float64)
    a2 = r6d[3:6].astype(np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0).astype(np.float32)


def _mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    return mat[:2, :].flatten().astype(np.float32)


def _axis_rotation(axis: str, degrees: float) -> np.ndarray:
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    if axis == "X":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == "Y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if axis == "Z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"unknown rotation axis: {axis}")


def _channels_to_rotation(values: np.ndarray, channels: Sequence[str]) -> np.ndarray:
    R = np.eye(3, dtype=np.float64)
    for value, channel in zip(values, channels):
        if channel.endswith("rotation"):
            R = R @ _axis_rotation(channel[0].upper(), float(value))
    return R


def _axis_rotation_batch(axis: str, degrees: np.ndarray) -> np.ndarray:
    """Batched version of ``_axis_rotation``: ``degrees`` is ``[N]`` -> returns ``[N, 3, 3]``."""
    a = np.deg2rad(degrees.astype(np.float64))
    c, s = np.cos(a), np.sin(a)
    n = a.shape[0]
    R = np.zeros((n, 3, 3), dtype=np.float64)
    if axis == "X":
        R[:, 0, 0] = 1.0
        R[:, 1, 1], R[:, 1, 2] = c, -s
        R[:, 2, 1], R[:, 2, 2] = s, c
    elif axis == "Y":
        R[:, 0, 0], R[:, 0, 2] = c, s
        R[:, 1, 1] = 1.0
        R[:, 2, 0], R[:, 2, 2] = -s, c
    elif axis == "Z":
        R[:, 0, 0], R[:, 0, 1] = c, -s
        R[:, 1, 0], R[:, 1, 1] = s, c
        R[:, 2, 2] = 1.0
    else:
        raise ValueError(f"unknown rotation axis: {axis}")
    return R


def _channels_to_rotation_batch(values: np.ndarray, channels: Sequence[str]) -> np.ndarray:
    """Batched version of ``_channels_to_rotation``.

    ``values`` is ``[n_frames, n_channels]`` (this joint's channel columns
    only), ``channels`` the matching channel-name list. Returns ``[n_frames, 3, 3]``
    — identical composition (``R = R @ R_axis`` per rotation channel, in
    channel order) to the scalar version, just batched over frames.
    """
    n_frames = values.shape[0]
    R = np.broadcast_to(np.eye(3, dtype=np.float64), (n_frames, 3, 3)).copy()
    for c_idx, channel in enumerate(channels):
        if channel.endswith("rotation"):
            R = np.matmul(R, _axis_rotation_batch(channel[0].upper(), values[:, c_idx]))
    return R


def _local_positions_batch(
    values: np.ndarray, channels: Sequence[str], offset: np.ndarray, scale: float,
) -> np.ndarray:
    """Batched version of the per-frame local-position accumulation in the
    original ``_forward_kinematics`` loop: ``offset`` (static) + any
    "position"-channel deltas for this frame, then scaled."""
    n_frames = values.shape[0]
    local_pos = np.broadcast_to(offset.astype(np.float64), (n_frames, 3)).copy()
    for c_idx, channel in enumerate(channels):
        if channel.endswith("position"):
            axis = "XYZ".index(channel[0].upper())
            local_pos[:, axis] += values[:, c_idx]
    local_pos *= scale
    return local_pos


def _open_bvh_text(path: str) -> str:
    if "::" not in path:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()

    tar_path, member = path.split("::", 1)
    with tarfile.open(tar_path, "r|gz") as tf:
        for info in tf:
            if info.name != member:
                continue
            fh = tf.extractfile(info)
            if fh is None:
                break
            return fh.read().decode("utf-8", errors="replace")
    raise FileNotFoundError(f"{member} not found in {tar_path}")


def _parse_bvh(path: str) -> _BvhMotion:
    lines = _open_bvh_text(path).splitlines()
    motion_idx = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")

    joints: List[_BvhJoint] = []
    stack: List[int] = []
    pending_joint: Optional[int] = None
    channel_cursor = 0
    ignore_depth = 0
    pending_end_site = False

    for raw in lines[:motion_idx]:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()

        if pending_end_site:
            if parts[0] == "{":
                ignore_depth = 1
                pending_end_site = False
            continue

        if ignore_depth:
            if "{" in parts:
                ignore_depth += 1
            if "}" in parts:
                ignore_depth -= 1
            continue

        if parts[0] == "End" and len(parts) > 1 and parts[1] == "Site":
            pending_end_site = True
            continue
        if parts[0] in {"ROOT", "JOINT"}:
            parent = stack[-1] if stack else -1
            pending_joint = len(joints)
            joints.append(
                _BvhJoint(
                    name=parts[1],
                    parent=parent,
                    offset=np.zeros(3, dtype=np.float64),
                    channels=[],
                    channel_start=channel_cursor,
                )
            )
        elif parts[0] == "{":
            if pending_joint is not None:
                stack.append(pending_joint)
                pending_joint = None
        elif parts[0] == "}":
            if stack:
                stack.pop()
        elif parts[0] == "OFFSET":
            joints[-1].offset = np.asarray([float(v) for v in parts[1:4]], dtype=np.float64)
        elif parts[0] == "CHANNELS":
            n = int(parts[1])
            joints[-1].channels = parts[2 : 2 + n]
            joints[-1].channel_start = channel_cursor
            channel_cursor += n

    frames_line = lines[motion_idx + 1].strip().split()
    frame_time_line = lines[motion_idx + 2].strip().split()
    n_frames = int(frames_line[-1])
    frame_time = float(frame_time_line[-1])
    data_lines = lines[motion_idx + 3 : motion_idx + 3 + n_frames]
    frames = np.asarray([[float(v) for v in line.split()] for line in data_lines], dtype=np.float64)
    if frames.shape[1] != channel_cursor:
        raise ValueError(
            f"{path}: expected {channel_cursor} BVH channels, got {frames.shape[1]}"
        )
    return _BvhMotion(joints=joints, frames=frames, fps=1.0 / frame_time)


def _resample_indices(n: int, src_fps: float, target_hz: int) -> np.ndarray:
    if target_hz <= 0 or abs(src_fps - float(target_hz)) < 1e-3:
        return np.arange(n, dtype=np.int64)
    duration = n / src_fps
    new_n = max(2, int(round(duration * float(target_hz))))
    idx = np.round(np.arange(new_n, dtype=np.float64) * src_fps / float(target_hz)).astype(np.int64)
    return np.clip(idx, 0, n - 1)


def _forward_kinematics(
    motion: _BvhMotion,
    target_hz: int,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], float]:
    """Vectorized BVH forward kinematics.

    Numerically identical to the original per-(frame, joint) Python double
    loop, just batched: the outer loop only runs over joints (~24), and every
    per-joint operation (local rotation/position, and the parent-composition
    step) is a single numpy call across all frames at once, instead of one
    Python-level call per frame. BVH parsing (``_parse_bvh``) always appends a
    joint's parent to ``motion.joints`` before any of its children (the BVH
    hierarchy syntax guarantees this), so a single forward pass over joints in
    list order is sufficient — no separate topological sort needed.
    """
    idx = _resample_indices(len(motion.frames), motion.fps, target_hz)
    frames = motion.frames[idx]     # [n_frames, n_total_channels]
    fps = float(target_hz) if target_hz > 0 else motion.fps

    n_frames = frames.shape[0]
    n_joints = len(motion.joints)
    world_pos = np.zeros((n_frames, n_joints, 3), dtype=np.float64)
    world_rot = np.zeros((n_frames, n_joints, 3, 3), dtype=np.float64)
    local_rot = np.zeros((n_frames, n_joints, 3, 3), dtype=np.float64)

    for j, joint in enumerate(motion.joints):
        values = frames[:, joint.channel_start : joint.channel_start + len(joint.channels)]  # [n_frames, n_ch]
        R_local = _channels_to_rotation_batch(values, joint.channels)          # [n_frames, 3, 3]
        local_rot[:, j] = R_local
        local_pos = _local_positions_batch(values, joint.channels, joint.offset, scale)  # [n_frames, 3]

        if joint.parent < 0:
            world_rot[:, j] = R_local
            world_pos[:, j] = local_pos
        else:
            parent = joint.parent
            world_rot[:, j] = np.matmul(world_rot[:, parent], R_local)
            world_pos[:, j] = world_pos[:, parent] + np.einsum(
                "nij,nj->ni", world_rot[:, parent], local_pos
            )

    return world_pos, world_rot, local_rot, [j.name for j in motion.joints], fps


def _choose_joint(names: Sequence[str], requested: Optional[str], candidates: Sequence[str]) -> int:
    if requested and requested in names:
        return names.index(requested)
    for name in candidates:
        if name in names:
            return names.index(name)
    raise KeyError(f"none of {list(candidates)} found in BVH joints")


def _compute_acc_world(pos_world: np.ndarray, fps: float, add_gravity: bool) -> np.ndarray:
    acc = np.zeros_like(pos_world, dtype=np.float64)
    acc[1:-1] = (pos_world[2:] - 2.0 * pos_world[1:-1] + pos_world[:-2]) * (fps ** 2)
    acc[0] = acc[1]
    acc[-1] = acc[-2]
    if add_gravity:
        acc[:, 1] += 9.81
    return acc


def _compute_gyro_world(R_world_from_wrist: np.ndarray, fps: float) -> np.ndarray:
    dR = np.einsum(
        "nij,njk->nik",
        R_world_from_wrist[:-1].transpose(0, 2, 1),
        R_world_from_wrist[1:],
    )
    omega_local = ScipyRot.from_matrix(dR).as_rotvec().astype(np.float64) * fps
    omega_world_raw = np.einsum("nij,nj->ni", R_world_from_wrist[:-1], omega_local)
    gyro = np.zeros((R_world_from_wrist.shape[0], 3), dtype=np.float64)
    gyro[:-1] = omega_world_raw
    gyro[-1] = omega_world_raw[-1]
    return gyro


def _load_bones_seed_session_flu(
    bvh_path: str,
    target_hz: int,
    add_gravity: bool = True,
    wrist_joint: Optional[str] = None,
    body_joints: Optional[Sequence[str]] = None,
    bvh_scale: float = 0.01,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return synthetic wrist IMU and selected local body rotations."""
    try:
        motion = _parse_bvh(bvh_path)
        world_pos, world_rot, local_rot, names, fps = _forward_kinematics(
            motion, target_hz=target_hz, scale=bvh_scale
        )
    except Exception as exc:
        logger.warning("[bones_seed_flu] failed to parse %s: %s", bvh_path, exc)
        return None

    if world_pos.shape[0] < 3:
        return None

    try:
        wrist_idx = _choose_joint(names, wrist_joint, RIGHT_WRIST_CANDIDATES)
    except KeyError as exc:
        logger.warning("[bones_seed_flu] %s: %s", bvh_path, exc)
        return None

    pos = world_pos[:, wrist_idx]
    rot = world_rot[:, wrist_idx]
    if not (np.isfinite(pos).all() and np.isfinite(rot).all()):
        # A malformed/corrupt BVH file (or a NaN/Inf sentinel value for an
        # untracked mocap frame) can propagate through FK into the wrist
        # joint's world rotation. Downstream, _compute_gyro_world feeds this
        # into a single batched ScipyRot.from_matrix() call over every frame
        # in the session — LAPACK's SVD-based rotation fitting can take a
        # pathologically long time (minutes+) on NaN input rather than
        # failing fast. Bail out here, before that call, instead.
        logger.warning(
            "[bones_seed_flu] %s: non-finite wrist pose (NaN/Inf) after FK — skipping", bvh_path
        )
        return None
    acc_world = _compute_acc_world(pos, fps=fps, add_gravity=add_gravity)
    gyro_world = _compute_gyro_world(rot, fps=fps)

    R_world_from_FLU = np.einsum("nij,jk->nik", rot, R_WRIST_FROM_FLU)
    R_FLU_from_world = R_world_from_FLU.transpose(0, 2, 1)
    acc_flu = np.einsum("nij,nj->ni", R_FLU_from_world, acc_world)
    gyro_flu = np.einsum("nij,nj->ni", R_FLU_from_world, gyro_world)
    X_imu = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)

    selected_names = list(body_joints or DEFAULT_BODY_JOINTS_24)
    missing = [name for name in selected_names if name not in names]
    if missing:
        logger.warning("[bones_seed_flu] %s missing body joints: %s", bvh_path, missing)
        return None
    joint_idx = [names.index(name) for name in selected_names]
    y_rot = local_rot[:, joint_idx].astype(np.float32)
    y = matrix_to_rotation_6d(torch.from_numpy(y_rot)).numpy().astype(np.float32)

    if not (np.isfinite(X_imu).all() and np.isfinite(y).all()):
        # Defense in depth: the wrist-pose check above catches the common
        # (and slow) case, but guard the final output too so a NaN/Inf can
        # never silently end up in a cached training window regardless of
        # which computation introduced it.
        logger.warning("[bones_seed_flu] %s: non-finite output (NaN/Inf) — skipping", bvh_path)
        return None

    # DEBUG, not INFO: this runs once per file inside a ThreadPoolExecutor
    # wrapped by a tqdm progress bar above — at INFO level, hundreds/thousands
    # of these lines interleave with (and visually bury) the bar's updates.
    logger.debug(
        "[bones_seed_flu] loaded %s: N=%d fps=%.2f joints=%d wrist=%s",
        os.path.basename(bvh_path), len(X_imu), fps, len(joint_idx), names[wrist_idx],
    )
    return X_imu, y


def _build_window_specs(session_lengths: List[int], T: int, hop: int,
                        min_keep: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding-window specs. When ``min_keep`` is set, sessions shorter than one full
    window (but >= min_keep samples) contribute ONE window starting at 0 — the window is
    zero-padded to T at fetch time and carries a temporal validity length (short-clip
    support, 2026-08-21; previously `if n < T: continue` silently discarded 70-95 % of
    the short-clip mocap sources)."""
    def n_win(n):
        if n >= T:
            return (n - T) // hop + 1
        return 1 if (min_keep is not None and n >= min_keep) else 0
    total = sum(n_win(n) for n in session_lengths)
    sess_idx = np.empty(total, dtype=np.int32)
    starts = np.empty(total, dtype=np.int32)
    offset = 0
    for ki, n in enumerate(session_lengths):
        k = n_win(n)
        if k == 0:
            continue
        if n >= T:
            sess_idx[offset:offset + k] = ki
            starts[offset:offset + k] = np.arange(0, n - T + 1, hop, dtype=np.int32)
        else:
            sess_idx[offset] = ki
            starts[offset] = 0
        offset += k
    return sess_idx, starts


class BonesSeedWristFullBodyDatasetFLU(Dataset):
    """Sliding-window dataset: Bones-Seed synthetic right-wrist IMU -> body pose."""

    def __init__(
        self,
        bvh_paths: List[str],
        cfg: Any,
        split_name: str = "train",
        target_hz: int = 50,
        add_gravity: bool = True,
        wrist_joint: Optional[str] = None,
        body_joints: Optional[Sequence[str]] = None,
        bvh_scale: float = 0.01,
    ):
        super().__init__()
        self.cfg = cfg
        self.split_name = split_name
        self.stats = PreprocessStats()
        self._T, self._hop = compute_T_and_hop(cfg)
        _vl0 = (cfg.get("data", {}) or {}).get("variable_length", {}) if isinstance(cfg, dict) else {}
        self._min_keep = int(_vl0.get("min_samples", 64)) if _vl0.get("include_short_clips", False) else None

        vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
        dec_cfg = vit_cfg.get("pose_decoder") or {}
        pose_cfg = cfg_get(cfg, ["objective", "pose"], {}) or cfg_get(
            cfg, ["objective", "pose_probe"], {}
        ) or {}

        self._patch_size = int(vit_cfg.get("patch_size", 16))
        self._num_patches = self._T // self._patch_size
        self._sub_patches = int(dec_cfg.get("sub_patches", 1))
        self._reset_root = bool(pose_cfg.get("reset_root_orientation", False))
        self._body_dim = len(body_joints or DEFAULT_BODY_JOINTS_24) * ROT6D_DIM

        self._sessions: List[Tuple[np.ndarray, np.ndarray]] = []
        self._session_files: List[str] = []  # bvh path per kept session, parallel to _sessions
        session_lengths: List[int] = []

        n_workers = min(16, max(1, os.cpu_count() or 1))
        ordered: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * len(bvh_paths)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            fut_to_idx = {
                pool.submit(
                    _load_bones_seed_session_flu,
                    p,
                    target_hz,
                    add_gravity,
                    wrist_joint,
                    body_joints,
                    bvh_scale,
                ): i
                for i, p in enumerate(bvh_paths)
            }
            with tqdm(
                total=len(bvh_paths),
                desc=f"Loading {split_name} sessions (Bones-Seed wrist FLU)",
                unit="session",
                dynamic_ncols=True,
            ) as pbar:
                for fut in as_completed(fut_to_idx):
                    ordered[fut_to_idx[fut]] = fut.result()
                    pbar.update(1)

        self._sanitize_stats = {"spike": 0, "cut": 0, "splits": 0}
        if not hasattr(self, "_min_keep"):      # defined later in this __init__ for the window specs; needed now
            _vl1 = (cfg.get("data", {}) or {}).get("variable_length", {}) if isinstance(cfg, dict) else {}
            self._min_keep = int(_vl1.get("min_samples", 64)) if _vl1.get("include_short_clips", False) else None
        for path, result in zip(bvh_paths, ordered):
            if result is None:
                logger.warning("[bones_seed_flu] skipping: %s", path)
                continue
            X, y = result
            # mocap glitches -> finite-difference spikes; interpolate 1-3 sample spikes, split at
            # tracking jumps, clamp to the +-16 g / +-2000 dps sensor range (synthetic_imu_sanitize)
            segs, st = sanitize_synthetic_imu(X, y, min_len=self._min_keep or 64)
            self._sanitize_stats["spike"] += st.get("spike_samples_interpolated", 0)
            self._sanitize_stats["cut"] += st.get("cut_samples", 0)
            self._sanitize_stats["splits"] += max(0, len(segs) - 1)
            for k, (Xs, ys) in enumerate(segs):
                self._sessions.append((Xs, ys))
                self._session_files.append(path if k == 0 else f"{path}#seg{k}")
                session_lengths.append(len(Xs))

        logger.info("[bones_seed_flu] sanitize: %d spike samples interpolated, %d samples cut, %d clip splits",
                    self._sanitize_stats["spike"], self._sanitize_stats["cut"], self._sanitize_stats["splits"])
        if not self._sessions:
            raise RuntimeError(
                "BonesSeedWristFullBodyDatasetFLU: no valid sessions loaded from "
                f"{len(bvh_paths)} BVH paths."
            )

        vl = (cfg.get("data", {}) or {}).get("variable_length", {}) if isinstance(cfg, dict) else {}
        self._min_keep = int(vl.get("min_samples", 64)) if vl.get("include_short_clips", False) else None
        self._sess_idx, self._starts = _build_window_specs(session_lengths, self._T, self._hop,
                                                           min_keep=self._min_keep)
        _lens = np.asarray(session_lengths, dtype=np.int64)
        self._window_valid = np.minimum(self._T, _lens[self._sess_idx] - self._starts).astype(np.int32)
        n_short = int((self._window_valid < self._T).sum())
        logger.info(
            "[bones_seed_flu] split=%s sessions=%d windows=%d (short=%d, min_keep=%s)",
            split_name, len(self._sessions), len(self._starts), n_short, self._min_keep,
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        si = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        X_raw, y_arr = self._sessions[si]
        Xw = X_raw[start : start + self._T]
        valid = int(self._window_valid[idx])

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        if self.split_name == "train":
            Xproc = apply_augment(Xproc.T, self.cfg).T
        if Xproc.shape[1] < self._T:      # short clip: zero-pad AFTER preprocess/augment
            pad = np.zeros((Xproc.shape[0], self._T - Xproc.shape[1]), dtype=Xproc.dtype)
            Xproc = np.concatenate([Xproc, pad], axis=1)
        x_tensor = torch.from_numpy(Xproc).float()

        P = self._patch_size
        K = self._sub_patches
        Q = P // K
        patch_labels = np.empty((self._num_patches, K * self._body_dim), dtype=np.float32)
        for p in range(self._num_patches):
            for k in range(K):
                frame = min(start + p * P + k * Q + Q // 2, len(y_arr) - 1)   # clamp: short clips
                patch_labels[p, k * self._body_dim : (k + 1) * self._body_dim] = (
                    y_arr[frame].reshape(-1)
                )

        if self._reset_root:
            R0 = _rot6d_to_mat(patch_labels[0, :6])
            R0_inv = R0.T
            for p in range(self._num_patches):
                for k in range(K):
                    base = k * self._body_dim
                    Ri = _rot6d_to_mat(patch_labels[p, base : base + 6])
                    patch_labels[p, base : base + 6] = _mat_to_rot6d(R0_inv @ Ri)

        return x_tensor, torch.from_numpy(patch_labels).float()


def collate_skip_none(
    batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


def _metadata_bvh_paths(
    bones_seed_root: str,
    metadata_csv: str,
    path_column: str,
    tar_fallback: bool = True,
) -> List[str]:
    out: List[str] = []
    tar_path = os.path.join(bones_seed_root, "soma_proportional.tar.gz")
    with open(metadata_csv, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rel = row.get(path_column)
            if not rel:
                continue
            abs_path = os.path.join(bones_seed_root, rel)
            if os.path.isfile(abs_path):
                out.append(abs_path)
            elif tar_fallback and os.path.isfile(tar_path):
                out.append(f"{tar_path}::{rel}")
    return out


def _expand_bvh_files(
    bvh_root: Optional[str],
    files: Optional[Any],
    bones_seed_root: Optional[str],
    metadata_csv: Optional[str],
    metadata_path_column: str,
    tar_fallback: bool,
) -> List[str]:
    if isinstance(files, list):
        root = bvh_root or bones_seed_root or ""
        return [f if os.path.isabs(f) or "::" in f else os.path.join(root, f) for f in files]
    if isinstance(files, str) and files not in {"", "all"}:
        root = bvh_root or bones_seed_root or ""
        return [files if os.path.isabs(files) or "::" in files else os.path.join(root, files)]

    if bvh_root:
        found = sorted(glob.glob(os.path.join(bvh_root, "**", "*.bvh"), recursive=True))
        if found:
            return found

    if metadata_csv and bones_seed_root:
        return _metadata_bvh_paths(
            bones_seed_root=bones_seed_root,
            metadata_csv=metadata_csv,
            path_column=metadata_path_column,
            tar_fallback=tar_fallback,
        )
    return []


def make_bones_seed_wrist_full_body_loaders_flu(
    cfg: Any,
    bvh_root: Optional[str] = None,
    train_files: Optional[Any] = "all",
    val_files: Optional[Any] = None,
    test_files: Optional[Any] = None,
    bones_seed_root: str = "/projects/b53017/vbc1591/bones-seed",
    metadata_csv: Optional[str] = None,
    metadata_path_column: str = "move_soma_proportional_path",
    target_hz: int = 50,
    add_gravity: bool = True,
    wrist_joint: Optional[str] = None,
    body_joints: Optional[Sequence[str]] = None,
    bvh_scale: float = 0.01,
    tar_fallback: bool = True,
    max_train_sessions: Optional[int] = None,
    max_val_sessions: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """Build train/val/test loaders for Bones-Seed BVH synthetic wrist IMU."""
    batch_size = int(cfg_get(cfg, ["data", "batch_size"], 256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers = int(cfg_get(cfg, ["data", "num_workers"], 0))
    pin_memory = bool(cfg_get(cfg, ["data", "pin_memory"], False))

    if metadata_csv is None and bones_seed_root:
        candidate = os.path.join(bones_seed_root, "metadata", "seed_metadata_v004.csv")
        metadata_csv = candidate if os.path.isfile(candidate) else None

    def _files(files_cfg: Any, limit: Optional[int]) -> List[str]:
        paths = _expand_bvh_files(
            bvh_root=bvh_root,
            files=files_cfg,
            bones_seed_root=bones_seed_root,
            metadata_csv=metadata_csv,
            metadata_path_column=metadata_path_column,
            tar_fallback=tar_fallback,
        )
        if limit is not None:
            paths = paths[: int(limit)]
        return paths

    train_paths = _files(train_files, max_train_sessions)
    val_paths = _files(val_files, max_val_sessions) if val_files else []
    test_paths = _files(test_files, None) if test_files else []

    # If no explicit validation set is provided, keep a small deterministic tail
    # split so the factory is immediately usable for quick experiments.
    if train_paths and not val_paths:
        n_val = max(1, int(round(0.05 * len(train_paths)))) if len(train_paths) > 1 else 0
        if n_val:
            val_paths = train_paths[-n_val:]
            train_paths = train_paths[:-n_val]

    def _make(name: str, paths: List[str], bs: int, shuffle: bool):
        if not paths:
            return None
        ds = BonesSeedWristFullBodyDatasetFLU(
            bvh_paths=paths,
            cfg=cfg,
            split_name=name,
            target_hz=target_hz,
            add_gravity=add_gravity,
            wrist_joint=wrist_joint,
            body_joints=body_joints,
            bvh_scale=bvh_scale,
        )
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_skip_none,
            persistent_workers=num_workers > 0,
        )

    loaders: Dict[str, DataLoader] = {}
    for name, paths, bs, shuffle in [
        ("train", train_paths, batch_size, True),
        ("val", val_paths, eval_batch_size, False),
        ("test", test_paths, eval_batch_size, False),
    ]:
        loader = _make(name, paths, bs, shuffle)
        if loader is not None:
            loaders[f"{name}_loader"] = loader
    return loaders
