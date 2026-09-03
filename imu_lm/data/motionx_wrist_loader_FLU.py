"""motionx_wrist_loader_FLU.py
Synthetic right-wrist IMU for Motion-X++ derived purely from GT SMPL-X body pose.

Motion-X++ (https://github.com/IDEA-Research/Motion-X) ships full-body pose as
per-frame SMPL-X parameter vectors packed into a flat 322-dim "smplx322" layout
(``motion/motion_generation/smplx322/<category>.zip``, one ``.npy`` per clip).
This layout has already been globally optimised (SLAM root trajectory + foot
contact refinement, see the dataset ``readme.md``), so ``trans`` is a physically
plausible world-frame root translation — exactly what's needed to synthesise a
realistic wrist accelerometer signal (it captures whole-body locomotion, not
just arm-relative motion).

This loader never extracts the archives to disk: it reads individual ``.npy``
members directly out of the zip via ``zipfile``, following the same synthesis
recipe as ``totalcapture_wrist_loader_FLU_syn.py``:

  acc  : 2nd central finite difference of the right-wrist joint world position
  gyro : angular velocity derived from consecutive FK wrist rotations

Pipeline
--------
1.  Read one ``[T, 322]`` clip array from ``motion/motion_generation/smplx322/<cat>.zip``
    and slice it into the documented SMPL-X fields:
        root_orient = motion[:,   0:  3]
        pose_body   = motion[:,   3: 66]   (21 joints, axis-angle)
        pose_hand   = motion[:,  66:156]   (unused — doesn't affect wrist FK)
        pose_jaw    = motion[:, 156:159]   (unused)
        face_expr   = motion[:, 159:209]   (unused)
        face_shape  = motion[:, 209:309]   (unused)
        trans       = motion[:, 309:312]   (global root translation, metres)
        betas       = motion[:, 312:322]   (unused — see note below)
2.  Build local rotation matrices ``R_local [T, 24, 3, 3]`` for the SMPL
    kinematic tree (joint 0 = root_orient, joints 1-21 = pose_body, joints
    22-23 = identity — Motion-X++ has no single "hand" joint, only fingers,
    and finger pose cannot move the wrist itself).
3.  Run SMPL forward kinematics (``core.articulate.model.ParametricModel``,
    the same male body model + FK code already used by the TotalCapture and
    Nymeria synthetic-wrist loaders in this codebase) to obtain:
      R_world_from_wrist(t)  [T, 3, 3] — wrist world-frame rotation
      p_world_wrist(t)       [T, 3]    — wrist world position
4.  Synthesise world-frame signals:
      a_world = d²p_world_wrist / dt²              (2nd central finite difference)
      ω_local = rotvec(R_world(t)^T @ R_world(t+1)) * fps
      ω_world = R_world_from_wrist @ ω_local
5.  Optional: add MEMS-style two-component noise to the gyro signal.
6.  Build the FLU frame and project:
      R_world_from_X(t) = R_world_from_wrist(t) @ R_WRIST_FROM_FLU
      acc_flu  = R_world_from_X(t)^T @ a_world
      gyro_flu = R_world_from_X(t)^T @ ω_world

Note on ``betas``: forward kinematics uses the model's mean (zero) shape, not
the per-clip Motion-X++ ``betas``, since those live in the SMPL-X shape space
and are not directly compatible with the SMPL model file used for FK here.
This matches the existing ``totalcapture_wrist_loader_FLU_syn.py`` convention,
which also ignores shape and computes FK on the mean body.

FLU body frame (right wrist, T-pose) — identical convention to every other
``*_FLU`` loader in this directory:
  Forward (F, x) = −x_smpl   (finger direction)
  Left    (L, y) = +z_smpl   (thumb side)
  Up      (U, z) = +y_smpl   (dorsal / watch-face)

Outputs per window
------------------
  x : [6, T]           float32 – (acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U)
  y : [N_patches, J*6] float32 – 6D local rotations for the right-arm chain
                                   J=4  → SMPL joints [14, 17, 19, 21]
                                          (right_collar, right_shoulder,
                                           right_elbow, right_wrist)
                                   J=24 → all SMPL joints (hands are identity)

Native frame rate: Motion-X / Motion-X++ motion is captured/annotated at
30 FPS; override ``native_hz`` if a given release differs.
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from typing import Any, Dict, List, Optional, Tuple

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# smplx322 field offsets (Motion-X / Motion-X++ convention)
_ROOT_ORIENT_SLICE = slice(0, 3)
_POSE_BODY_SLICE   = slice(3, 66)
_TRANS_SLICE       = slice(309, 312)
_SMPLX322_DIM      = 322

# SMPL right-arm chain (same joint indices as totalcapture_wrist_loader_FLU_syn.py)
RIGHT_WRIST_CHAIN_JOINTS = [14, 17, 19, 21]
NUM_CHAIN_JOINTS         = len(RIGHT_WRIST_CHAIN_JOINTS)
ROT6D_DIM                = 6

# SMPL joint indices
_SMPL_RIGHT_WRIST = 21
_SMPL_NUM_JOINTS  = 24

# Motion-X / Motion-X++ native capture rate (see module docstring)
_MOTIONX_NATIVE_HZ = 30.0

# World-frame gravity (Y-up, SMPL convention)
_G_WORLD = np.array([0.0, -9.81, 0.0], dtype=np.float64)

# ---------------------------------------------------------------------------
# FLU frame definition (identical to all other *_FLU loaders)
# ---------------------------------------------------------------------------
R_WRIST_FROM_FLU = np.array(
    [
        [-1.0, 0.0, 0.0],   # F = −x_smpl
        [ 0.0, 0.0, 1.0],   # L = +z_smpl
        [ 0.0, 1.0, 0.0],   # U = +y_smpl
    ],
    dtype=np.float64,
)
assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Small numeric helpers (self-contained, mirrors totalcapture_wrist_loader_FLU_syn.py)
# ---------------------------------------------------------------------------

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] → [..., 6]  (first two columns, row-major)."""
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def _resample_linear(x: np.ndarray, new_len: int) -> np.ndarray:
    """Linear interpolation of [N, ...] → [new_len, ...]."""
    old_len = x.shape[0]
    if old_len == new_len:
        return x
    t_old = np.linspace(0.0, 1.0, old_len)
    t_new = np.linspace(0.0, 1.0, new_len)
    flat  = x.reshape(old_len, -1)
    out   = np.stack(
        [np.interp(t_new, t_old, flat[:, c]) for c in range(flat.shape[1])],
        axis=1,
    )
    return out.reshape(new_len, *x.shape[1:])


def _compute_acc_world(pos_world: np.ndarray, fps: float) -> np.ndarray:
    """2nd central finite difference of wrist position → world-frame acc [N, 3]."""
    N   = pos_world.shape[0]
    acc = np.zeros((N, 3), dtype=np.float64)
    acc[1:-1] = (pos_world[2:] - 2.0 * pos_world[1:-1] + pos_world[:-2]) * (fps ** 2)
    acc[0]    = acc[1]
    acc[-1]   = acc[-2]
    return acc


def _compute_omega_local(R_world_from_joint: np.ndarray, fps: float) -> np.ndarray:
    """Angular velocity in the joint LOCAL frame from consecutive world rotations.

    dR_t = R_t^T @ R_{t+1},   omega_local(t) = rotvec(dR_t) * fps
    """
    dR = np.einsum(
        "nij,njk->nik",
        R_world_from_joint[:-1].transpose(0, 2, 1),
        R_world_from_joint[1:],
    )                          # [N-1, 3, 3]
    omega_raw = (
        ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
        .as_rotvec()
        .astype(np.float64) * fps
    )                          # [N-1, 3]
    N     = R_world_from_joint.shape[0]
    omega = np.zeros((N, 3), dtype=np.float64)
    omega[:-1] = omega_raw
    omega[-1]  = omega_raw[-1]
    return omega


def _add_mems_gyro_noise(
    gyro: np.ndarray,
    fps: float,
    white_noise_std: float = 0.02,
    bias_drift_std:  float = 0.002,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Two-component MEMS gyroscope noise model (IEEE Std 952).

    omega_measured(t) = omega_true(t)
                      + N(0, white_noise_std²)          [Angle Random Walk]
                      + integrated_random_walk(t)        [Bias Instability]
    """
    if rng is None:
        rng = np.random.RandomState()
    N  = gyro.shape[0]
    dt = 1.0 / fps
    out = gyro.copy()
    out = out + rng.normal(0.0, white_noise_std, size=(N, 3))
    bias_steps = rng.normal(0.0, bias_drift_std * np.sqrt(dt), size=(N, 3))
    out = out + np.cumsum(bias_steps, axis=0)
    return out.astype(gyro.dtype)


# ---------------------------------------------------------------------------
# Clip discovery — enumerate .npy members inside the smplx322 zips
# ---------------------------------------------------------------------------

_DEFAULT_CATEGORIES = (
    "animation", "haa500", "humman", "idea400", "kungfu", "music", "perform",
)

ClipSpec = Tuple[str, str, str]   # (category, zip_path, member_name)


def list_motionx_clips(
    motionx_root: str,
    categories: Optional[List[str]] = None,
    max_clips_per_category: Optional[int] = None,
) -> List[ClipSpec]:
    """Enumerate every smplx322 clip without extracting any archive.

    Args:
        motionx_root: path to the Motion-X++ release root (contains ``motion/``).
        categories:   subset of {animation, haa500, humman, idea400, kungfu,
                      music, perform}; ``None`` → all.
        max_clips_per_category: optional cap, useful for smoke tests.

    Returns:
        list of ``(category, zip_path, member_name)`` tuples.
    """
    categories = categories or list(_DEFAULT_CATEGORIES)
    smplx322_dir = os.path.join(motionx_root, "motion", "motion_generation", "smplx322")

    specs: List[ClipSpec] = []
    for cat in categories:
        zip_path = os.path.join(smplx322_dir, f"{cat}.zip")
        if not os.path.isfile(zip_path):
            logger.warning("[motionx] category zip not found: %s", zip_path)
            continue
        with zipfile.ZipFile(zip_path) as zf:
            members = [
                n for n in zf.namelist()
                if n.endswith(".npy") and not n.endswith("/")
            ]
        if max_clips_per_category is not None:
            members = members[:max_clips_per_category]
        specs.extend((cat, zip_path, m) for m in members)

    logger.info("[motionx] discovered %d clips across %d categories", len(specs), len(categories))
    return specs


# Motion-X v1 (`motionx_smplx.zip`, 2023-10): ONE archive laid out as
# motion_data/smplx_322/<subset>/subset_XXXX/<clip>.npy, same [T, 322] float32 smplx322
# rows at 30 FPS, 11 subsets / 73.3 h.  7 subsets (animation, haa500, humman, idea400,
# kungfu, music, perform) are re-cuts of the SAME footage that Motion-X++ ships (clip
# names differ, so they cannot be name-deduped) -> only enable the 4 subsets absent from
# Motion-X++: fitness 28.5 h, game_motion 10.2 h, aist 3.0 h, dance 0.3 h.
V1_UNIQUE_SUBSETS = ("fitness", "game_motion", "aist", "dance")


def list_motionx_v1_clips(
    zip_path: str,
    subsets: Optional[List[str]] = None,
    max_clips_per_subset: Optional[int] = None,
) -> List[ClipSpec]:
    """Enumerate Motion-X v1 clips inside the single ``motionx_smplx.zip``.

    Returns ``(f"v1_{subset}", zip_path, member)`` tuples — the same ClipSpec the
    Motion-X++ path uses, so the dataset class needs no changes.
    """
    subsets = [x.lower() for x in (subsets or V1_UNIQUE_SUBSETS)]
    if not os.path.isfile(zip_path):
        logger.warning("[motionx] v1 zip not found: %s", zip_path)
        return []
    specs: List[ClipSpec] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".npy")]
    per: Dict[str, List[str]] = {}
    for n in names:
        parts = n.split("/")
        if len(parts) >= 4 and parts[0] == "motion_data":
            per.setdefault(parts[2].lower(), []).append(n)
    for sub in subsets:
        members = sorted(per.get(sub, []))
        if not members:
            logger.warning("[motionx] v1 subset %s has no clips in %s", sub, zip_path)
        if max_clips_per_subset is not None:
            members = members[:max_clips_per_subset]
        specs.extend((f"v1_{sub}", zip_path, m) for m in members)
    logger.info("[motionx] v1: discovered %d clips across %d subsets", len(specs), len(subsets))
    return specs


def split_motionx_clips(
    clip_specs: List[ClipSpec],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> Tuple[List[ClipSpec], List[ClipSpec], List[ClipSpec]]:
    """Deterministic clip-level random split (Motion-X++ ships no official split)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(clip_specs))
    n_val  = int(round(len(clip_specs) * val_frac))
    n_test = int(round(len(clip_specs) * test_frac))
    val_idx   = idx[:n_val]
    test_idx  = idx[n_val : n_val + n_test]
    train_idx = idx[n_val + n_test :]
    to_list = lambda ii: [clip_specs[i] for i in ii]
    return to_list(train_idx), to_list(val_idx), to_list(test_idx)


# ---------------------------------------------------------------------------
# Core clip loader
# ---------------------------------------------------------------------------

def _read_smplx322(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    """Read one ``[T, 322]`` clip array straight out of an open zip archive."""
    data = zf.read(member)
    return np.load(io.BytesIO(data))


def synthesize_wrist_flu_from_smplx322(
    motion: np.ndarray,
    body_model: "Any",
    target_hz: int = 50,
    native_hz: float = _MOTIONX_NATIVE_HZ,
    add_gravity: bool = True,
    gyro_white_noise_std: float = 0.0,
    gyro_bias_drift_std:  float = 0.0,
    rng: Optional[np.random.RandomState] = None,
    num_joints: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesise **(X, y)** in the FLU frame from one smplx322 ``[T, 322]`` clip.

    Args:
        motion:     ``[T, 322]`` float array, Motion-X++ smplx322 layout.
        body_model: a ``core.articulate.model.ParametricModel`` (SMPL, 24 joints).
        target_hz:  resample target; Motion-X++ is natively ``native_hz`` FPS.
        add_gravity: subtract world gravity so ``acc`` reads like a real MEMS
                     accelerometer (specific force) instead of pure kinematic accel.
        num_joints: 4 → right-arm chain [14, 17, 19, 21]; 24 → all SMPL joints.

    Returns:
        X : [N, 6]    float32 – [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
        y : [N, J, 6] float32 – 6D local rotations; J = num_joints
    """
    motion = np.asarray(motion, dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != _SMPLX322_DIM:
        raise ValueError(f"expected [T, {_SMPLX322_DIM}] smplx322 array, got {motion.shape}")

    N = motion.shape[0]

    # ------------------------------------------------------------------
    # Resample to target_hz if needed (native = native_hz FPS)
    # ------------------------------------------------------------------
    if abs(native_hz - target_hz) > 1e-3:
        new_len = int(round((N - 1) * target_hz / native_hz)) + 1
        motion  = _resample_linear(motion, new_len)
        N       = new_len

    fps = float(target_hz)

    root_orient = motion[:, _ROOT_ORIENT_SLICE]                 # [N, 3]
    pose_body   = motion[:, _POSE_BODY_SLICE].reshape(N, 21, 3)  # [N, 21, 3]
    trans       = motion[:, _TRANS_SLICE]                        # [N, 3]

    # ------------------------------------------------------------------
    # Step 1 — pose → local rotation matrices [N, 24, 3, 3]
    # ------------------------------------------------------------------
    aa_body = np.concatenate([root_orient[:, None, :], pose_body], axis=1)   # [N, 22, 3]
    R_body  = (
        ScipyRot.from_rotvec(aa_body.reshape(-1, 3))
        .as_matrix()
        .reshape(N, 22, 3, 3)
    )
    R_local = np.tile(np.eye(3), (N, _SMPL_NUM_JOINTS, 1, 1))   # [N, 24, 3, 3]
    R_local[:, :22] = R_body
    R_local_t = torch.from_numpy(R_local.astype(np.float32))     # [N, 24, 3, 3]

    # ------------------------------------------------------------------
    # Step 2 — SMPL FK → world-frame wrist rotation + position
    # ------------------------------------------------------------------
    tran_t = torch.from_numpy(trans.astype(np.float32))          # [N, 3]
    pose_global_t, joint_world_t = body_model.forward_kinematics_masked(
        R_local_t,
        tran=tran_t,
        calc_mesh=False,
    )   # [N, 24, 3, 3], [N, 24, 3]

    R_world_from_wrist = (
        pose_global_t[:, _SMPL_RIGHT_WRIST].cpu().numpy().astype(np.float64)
    )   # [N, 3, 3]
    p_world_wrist = (
        joint_world_t[:, _SMPL_RIGHT_WRIST].cpu().numpy().astype(np.float64)
    )   # [N, 3]

    # ------------------------------------------------------------------
    # Step 3 — synthesise world-frame acc and gyro
    # ------------------------------------------------------------------
    a_world = _compute_acc_world(p_world_wrist, fps)   # [N, 3]

    if add_gravity:
        # f_mems = a_kinematic - g_world  ≡  a_kinematic + [0, 9.81, 0]  (Y-up)
        a_world = a_world - _G_WORLD

    omega_local = _compute_omega_local(R_world_from_wrist, fps)                # [N, 3]
    omega_world = np.einsum("nij,nj->ni", R_world_from_wrist, omega_local)     # [N, 3]

    if gyro_white_noise_std > 0.0 or gyro_bias_drift_std > 0.0:
        omega_world = _add_mems_gyro_noise(
            omega_world, fps,
            white_noise_std=gyro_white_noise_std,
            bias_drift_std=gyro_bias_drift_std,
            rng=rng,
        )

    # ------------------------------------------------------------------
    # Step 4 — build FLU frame per timestep
    # ------------------------------------------------------------------
    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_wrist, R_WRIST_FROM_FLU)   # [N, 3, 3]
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)                                 # [N, 3, 3]

    acc_flu  = np.einsum("nij,nj->ni", R_X_from_world, a_world)      # [N, 3]
    gyro_flu = np.einsum("nij,nj->ni", R_X_from_world, omega_world)  # [N, 3]

    X = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)   # [N, 6]

    # ------------------------------------------------------------------
    # Labels — local 6D rotations for selected joints
    # ------------------------------------------------------------------
    label_R = R_local if num_joints == _SMPL_NUM_JOINTS else R_local[:, RIGHT_WRIST_CHAIN_JOINTS]
    y = matrix_to_rotation_6d(
        torch.from_numpy(label_R.astype(np.float32))
    ).numpy().astype(np.float32)   # [N, J, 6]

    return X, y


def load_clip_syn_flu(
    zf: zipfile.ZipFile,
    member: str,
    body_model: "Any",
    target_hz: int = 50,
    native_hz: float = _MOTIONX_NATIVE_HZ,
    add_gravity: bool = True,
    gyro_white_noise_std: float = 0.0,
    gyro_bias_drift_std:  float = 0.0,
    rng: Optional[np.random.RandomState] = None,
    num_joints: int = 4,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read + synthesise one clip. Returns ``None`` on any failure (too short, bad file)."""
    try:
        motion = _read_smplx322(zf, member)
    except Exception as exc:
        logger.warning("[motionx_wrist_flu] cannot read %s: %s", member, exc)
        return None

    if motion.ndim != 2 or motion.shape[1] != _SMPLX322_DIM or motion.shape[0] < 3:
        logger.warning("[motionx_wrist_flu] skipping malformed/short clip: %s (%s)", member, motion.shape)
        return None

    return synthesize_wrist_flu_from_smplx322(
        motion, body_model,
        target_hz=target_hz, native_hz=native_hz, add_gravity=add_gravity,
        gyro_white_noise_std=gyro_white_noise_std, gyro_bias_drift_std=gyro_bias_drift_std,
        rng=rng, num_joints=num_joints,
    )


# ---------------------------------------------------------------------------
# Window index builder (identical scheme to totalcapture_wrist_loader_FLU_syn.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MotionXWristDatasetFLU(Dataset):
    """Sliding-window dataset backed by **synthetic** right-wrist IMU from
    Motion-X++ GT SMPL-X body pose (see module docstring for the full pipeline).

    Input  X : [6, T]   float32 – [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
    Label  y : [P, J*6] float32 – per-patch 6D rotations; P = T // patch_size
    """

    def __init__(
        self,
        clip_specs: List[ClipSpec],
        cfg: Any,
        split_name: str = "train",
        target_hz: int = 50,
        native_hz: float = _MOTIONX_NATIVE_HZ,
        add_gravity: bool = True,
        gyro_white_noise_std: float = 0.0,
        gyro_bias_drift_std:  float = 0.0,
        noise_seed: int = 0,
        num_joints: int = 4,
    ):
        super().__init__()
        self.cfg         = cfg
        self.split_name  = split_name
        self.target_hz   = target_hz
        self.add_gravity = add_gravity
        self.num_joints  = num_joints
        self.stats       = PreprocessStats()

        rng: Optional[np.random.RandomState] = (
            np.random.RandomState(noise_seed)
            if (gyro_white_noise_std > 0.0 or gyro_bias_drift_std > 0.0)
            else None
        )

        self._T, self._hop = compute_T_and_hop(cfg)
        _vl = (cfg.get("data", {}) or {}).get("variable_length", {}) if isinstance(cfg, dict) else {}
        self._min_keep = int(_vl.get("min_samples", 64)) if _vl.get("include_short_clips", False) else None
        vit_cfg            = cfg_get(cfg, ["vit1d"], {}) or {}
        self._patch_size   = int(vit_cfg.get("patch_size", 16))
        self._num_patches  = self._T // self._patch_size

        import core.articulate as art
        from core.paths import Paths
        body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))

        self._sessions: List[Tuple[np.ndarray, np.ndarray]] = []
        self._session_files: List[str] = []  # "{category}/{member}" per kept session, parallel to _sessions
        session_lengths: List[int] = []
        self._sanitize_stats = {"spike": 0, "cut": 0, "splits": 0}

        # Group by zip so each archive's central directory is only opened once.
        by_zip: Dict[str, List[Tuple[str, str]]] = {}
        for cat, zip_path, member in clip_specs:
            by_zip.setdefault(zip_path, []).append((cat, member))

        pbar = tqdm(
            total=len(clip_specs),
            desc=f"Loading {split_name} clips (Motion-X++ syn wrist FLU)",
            unit="clip",
            dynamic_ncols=True,
        )
        for zip_path, cat_members in by_zip.items():
            with zipfile.ZipFile(zip_path) as zf:
                for cat, member in cat_members:
                    result = load_clip_syn_flu(
                        zf, member, body_model,
                        target_hz=target_hz, native_hz=native_hz, add_gravity=add_gravity,
                        gyro_white_noise_std=gyro_white_noise_std,
                        gyro_bias_drift_std=gyro_bias_drift_std,
                        rng=rng, num_joints=num_joints,
                    )
                    pbar.update(1)
                    if result is None:
                        continue
                    X, y = result
                    floor = self._min_keep if self._min_keep is not None else self._T
                    # mocap glitches -> finite-difference spikes (27 % of cached windows > 10 g before
                    # this): interpolate 1-3 sample spikes, split at tracking jumps, clamp to the
                    # +-16 g / +-2000 dps sensor range (synthetic_imu_sanitize)
                    segs, st = sanitize_synthetic_imu(X, y, min_len=floor)
                    self._sanitize_stats["spike"] += st.get("spike_samples_interpolated", 0)
                    self._sanitize_stats["cut"] += st.get("cut_samples", 0)
                    self._sanitize_stats["splits"] += max(0, len(segs) - 1)
                    for k, (Xs, ys) in enumerate(segs):
                        self._sessions.append((Xs, ys))
                        self._session_files.append(f"{cat}/{member}" + ("" if k == 0 else f"#seg{k}"))
                        session_lengths.append(len(Xs))
        pbar.close()
        logger.info("[motionx] sanitize: %d spike samples interpolated, %d samples cut, %d clip splits",
                    self._sanitize_stats["spike"], self._sanitize_stats["cut"], self._sanitize_stats["splits"])

        if not self._sessions:
            raise RuntimeError(
                "MotionXWristDatasetFLU: no valid clips loaded from "
                f"{len(clip_specs)} clip specs."
            )

        self._sess_idx, self._starts = _build_window_specs(
            session_lengths, self._T, self._hop, min_keep=self._min_keep
        )
        _lens = np.asarray(session_lengths, dtype=np.int64)
        self._window_valid = np.minimum(self._T, _lens[self._sess_idx] - self._starts).astype(np.int32)

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        si    = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        T     = self._T

        X_raw, y_arr = self._sessions[si]
        Xw = X_raw[start : start + T]   # [T, 6]

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        Xproc    = apply_augment(Xproc.T, self.cfg).T   # [C, T] → augment → [C, T]
        if Xproc.shape[1] < T:                          # short clip: zero-pad after preprocess/augment
            pad = np.zeros((Xproc.shape[0], T - Xproc.shape[1]), dtype=Xproc.dtype)
            Xproc = np.concatenate([Xproc, pad], axis=1)
        x_tensor = torch.from_numpy(Xproc).float()      # [6, T]

        P        = self._patch_size
        J        = self.num_joints
        patch_labels = np.empty((self._num_patches, J * ROT6D_DIM), dtype=np.float32)
        for p in range(self._num_patches):
            frame = min(start + p * P + P // 2, len(y_arr) - 1)   # clamp: short clips
            patch_labels[p] = y_arr[frame].reshape(-1)   # [J, 6] → [J*6]

        y_tensor = torch.from_numpy(patch_labels).float()   # [P, J*6]
        return x_tensor, y_tensor


# ---------------------------------------------------------------------------
# Collate helper
# ---------------------------------------------------------------------------

def collate_skip_none(
    batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_motionx_wrist_loaders_flu(
    cfg: Any,
    motionx_root: str,
    categories: Optional[List[str]] = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    split_seed: int = 0,
    max_clips_per_category: Optional[int] = None,
    target_hz: int = 50,
    native_hz: float = _MOTIONX_NATIVE_HZ,
    add_gravity: bool = True,
    gyro_white_noise_std: float = 0.0,
    gyro_bias_drift_std:  float = 0.0,
    noise_seed: int = 0,
    num_joints: int = 4,
    v1_zip: Optional[str] = None,
    v1_subsets: Optional[List[str]] = None,
) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders for Motion-X++ synthetic-wrist-FLU.

    ``v1_zip`` additionally pulls Motion-X v1 subsets (default: the 4 that Motion-X++
    does not ship — see ``V1_UNIQUE_SUBSETS``) into the same clip pool before splitting.

    Motion-X++ ships no official split, so clips are randomly partitioned
    (clip-level, not frame-level) using ``split_seed``.

    Parameters
    ----------
    cfg          : experiment config (OmegaConf or dict)
    motionx_root : Motion-X++ release root (contains ``motion/``), e.g.
                   ``/projects/b53017/vbc1591/Motion-Xplusplus``
    categories   : subset of {animation, haa500, humman, idea400, kungfu,
                   music, perform}; ``None`` → all
    val_frac / test_frac : clip-level split fractions
    max_clips_per_category : optional cap per category (useful for smoke tests)
    target_hz    : output sampling rate (native is ``native_hz``, default 30)
    add_gravity  : see ``synthesize_wrist_flu_from_smplx322``
    gyro_white_noise_std / gyro_bias_drift_std : MEMS noise parameters
    num_joints   : 4 (right-arm chain) or 24 (all SMPL joints)

    Returns
    -------
    dict with keys ``'train_loader'``, ``'val_loader'``, ``'test_loader'``
    """
    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],      256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],       0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],    False))

    all_specs = list_motionx_clips(
        motionx_root, categories=categories, max_clips_per_category=max_clips_per_category
    )
    if v1_zip:
        all_specs = all_specs + list_motionx_v1_clips(
            v1_zip, subsets=v1_subsets, max_clips_per_subset=max_clips_per_category
        )
    train_specs, val_specs, test_specs = split_motionx_clips(
        all_specs, val_frac=val_frac, test_frac=test_frac, seed=split_seed
    )

    def _make(name: str, specs: List[ClipSpec], bs: int, shuffle: bool):
        if not specs:
            return None
        ds = MotionXWristDatasetFLU(
            clip_specs=specs,
            cfg=cfg,
            split_name=name,
            target_hz=target_hz,
            native_hz=native_hz,
            add_gravity=add_gravity,
            gyro_white_noise_std=gyro_white_noise_std,
            gyro_bias_drift_std=gyro_bias_drift_std,
            noise_seed=noise_seed,
            num_joints=num_joints,
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
    for name, specs, bs, shuf in [
        ("train", train_specs, batch_size,      True),
        ("val",   val_specs,   eval_batch_size, False),
        ("test",  test_specs,  eval_batch_size, False),
    ]:
        loader = _make(name, specs, bs, shuf)
        if loader is not None:
            loaders[f"{name}_loader"] = loader

    return loaders


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO)

    cfg = yaml.safe_load(open("configs/base.yaml"))
    cfg.setdefault("vit1d", {})
    cfg["vit1d"].setdefault("patch_size", 16)

    root = "/projects/b53017/vbc1591/Motion-Xplusplus"
    specs = list_motionx_clips(root, categories=["humman"], max_clips_per_category=3)
    if not specs:
        raise SystemExit("No clips found — adjust `root` above.")

    import core.articulate as art
    from core.paths import Paths
    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))

    cat, zip_path, member = specs[0]
    with zipfile.ZipFile(zip_path) as zf:
        result = load_clip_syn_flu(zf, member, body_model, target_hz=50, add_gravity=True)
    if result is None:
        raise SystemExit("load_clip_syn_flu returned None")
    X, y = result
    print(f"clip: {cat}/{member}")
    print(f"X shape: {X.shape}   y shape: {y.shape}")
    print(f"acc mean/std:  {X[:, :3].mean():.4f} / {X[:, :3].std():.4f}")
    print(f"gyro mean/std: {X[:, 3:].mean():.4f} / {X[:, 3:].std():.4f}")
