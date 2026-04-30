"""imuposer_wrist_local_disp_loader_FLU.py
IMUPoser right-wrist IMU (FLU body frame) → per-patch pelvis-root velocity in
the **wrist FLU sensor frame**.

Motivation
----------
Expressing the pelvis velocity in the wrist FLU frame keeps inputs and labels
in a consistent local coordinate system.  The network receives acc/gyro already
in FLU, and the velocity label is also in FLU, so the body-frame direction of
motion is directly observable without a separate global-yaw estimate.

IMU derivation
--------------
  acc  : real_imu_data["real_acc"][:, 1]  –  Z-up world, gravity-subtracted real acc
  gyro : real_imu_data["real_ori"][:, 1]  –  Z-up world, R_world_from_sensor

Data flow
---------
1.  SMPL FK on gt_data["pose"] + joint_pos[:,0] as root_trans →
      R_world_from_wrist(t)   [N, 3, 3]  – wrist world rotation (for FLU frame)
      pos_root(t)             [N, 3]     – pelvis world position  (for velocity labels)
2.  Build FLU frame from wrist:
      R_world_from_X(t) = R_world_from_wrist(t) @ R_WRIST_FROM_FLU
      R_X_from_world(t) = R_world_from_X(t)^T
3.  Project real acc and gyro into FLU.
4.  Store pos_root and R_X_from_world for label computation.

Per-patch label
---------------
Each patch p (P frames) is divided into 4 sub-patches of Q = P // 4 frames.
For sub-patch k (k = 0 … 3):

    t_start = start + p*P + k*Q
    t_end   = t_start + Q
    t_mid   = t_start + Q // 2              (midpoint — used for FLU frame)

    v_world[k] = (pos_root[t_end] − pos_root[t_start]) / (Q / fps)
    v_flu[k]   = R_X_from_world[t_mid] @ v_world[k]   (pelvis vel in wrist FLU, m/s)

    shape  : [N_patches, 4, 3]  float32

FLU frame convention (right wrist, T-pose)
------------------------------------------
  F (x) = −x_smpl  (finger / forward direction)
  L (y) = +z_smpl  (thumb / left side)
  U (z) = +y_smpl  (dorsal / watch-face / up)

Inputs / outputs per window
---------------------------
  x : [6, T]              float32  – (acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U)
  y : [N_patches, 4, 3]   float32  – pelvis velocity in wrist FLU frame (m/s)
                                       y[p, k] = velocity at sub-patch k of patch p
"""

from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRot
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from imu_lm.data.imuposerdataset import (
    _resample_linear,
    _resample_nearest,
    _resample_rotmat_nlerp,
    _resample_rotmat_nearest,
)
from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.augmentations.transform import apply_augment
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.utils.helpers import cfg_get

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SMPL_R_WRIST     = 21   # right wrist joint index in SMPL
_SMPL_PELVIS      = 0    # pelvis (root) joint index in SMPL
_IMUPOSER_R_WRIST = 1    # sensor index in real_acc / real_ori

# T-pose FLU frame at the right wrist (same as all other *_FLU loaders)
R_WRIST_FROM_FLU = np.array(
    [[-1.0,  0.0,  0.0],   # F = −x_smpl
     [ 0.0,  0.0,  1.0],   # L = +z_smpl
     [ 0.0,  1.0,  0.0]],  # U = +y_smpl
    dtype=np.float64,
)
assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Window index builder
# ---------------------------------------------------------------------------

def _build_window_specs(session_lengths, T, hop):
    total    = sum(max(0, (n - T) // hop + 1) for n in session_lengths)
    sess_idx = np.empty(total, dtype=np.int32)
    starts   = np.empty(total, dtype=np.int32)
    offset   = 0
    for ki, n in enumerate(session_lengths):
        if n < T:
            continue
        n_win = (n - T) // hop + 1
        sess_idx[offset : offset + n_win] = ki
        starts[offset : offset + n_win]   = np.arange(0, n - T + 1, hop, dtype=np.int32)
        offset += n_win
    return sess_idx, starts


# ---------------------------------------------------------------------------
# Session loader
# ---------------------------------------------------------------------------

def load_session_local_disp_flu(
    pkl_path: str,
    target_hz: int = 50,
    add_gravity: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load one IMUPoser pkl → (X_flu, pos_root, R_world_from_pelvis).

    Parameters
    ----------
    pkl_path    : path to an IMUPoser ``.pkl`` file.
    target_hz   : target sampling rate (data resampled if raw ≠ target).
    add_gravity : if True, adds gravity back to real_acc so it matches a full
                  MEMS specific-force reading (Z-up world: adds +9.81 in Z).

    Returns
    -------
    X_flu              : [N, 6]    float32  – FLU acc + gyro
    pos_root           : [N, 3]    float32  – pelvis world position (Z-up, metres)
    R_world_from_pelvis: [N, 3, 3] float32  – pelvis world-frame rotation from FK
    ``None`` on any failure.
    """
    import core.articulate as art
    from core.paths import Paths

    _log.info("[imuposer_local_disp] loading: %s", os.path.basename(pkl_path))

    try:
        with open(pkl_path, "rb") as fh:
            seq_data = pickle.load(fh)
    except Exception as exc:
        _log.warning("[imuposer_local_disp] cannot open %s: %s", pkl_path, exc)
        return None

    gt          = seq_data.get("gt_data", {})
    pose_local  = gt.get("pose")        # [N, 24, 3, 3]
    joint_pos_w = gt.get("joint_pos")   # [N, 24, 3]

    ri       = seq_data.get("real_imu_data", {})
    real_acc = ri.get("real_acc")       # [N, 5, 3]
    real_ori = ri.get("real_ori")       # [N, 5, 3, 3]

    if pose_local is None or joint_pos_w is None:
        _log.warning("[imuposer_local_disp] %s: missing gt fields.", pkl_path)
        return None
    if real_acc is None or real_ori is None:
        _log.warning("[imuposer_local_disp] %s: missing real_imu_data fields.", pkl_path)
        return None

    def _to_tensor(v):
        if torch.is_tensor(v):
            return v.float()
        return torch.as_tensor(np.asarray(v, dtype=np.float32))

    pose_local  = _to_tensor(pose_local)   # [N, 24, 3, 3]
    joint_pos_w = _to_tensor(joint_pos_w)  # [N, 24, 3]
    real_acc    = _to_tensor(real_acc)     # [N, 5, 3]
    real_ori    = _to_tensor(real_ori)     # [N, 5, 3, 3]

    fps_raw = float(seq_data.get("sampling_rate_hz", target_hz))
    N       = pose_local.shape[0]
    if N < 3:
        _log.warning("[imuposer_local_disp] %s: too short (%d frames).", pkl_path, N)
        return None

    # ------------------------------------------------------------------
    # Resample to target_hz if needed
    # ------------------------------------------------------------------
    if abs(fps_raw - target_hz) > 1e-3:
        new_len = int(round((N - 1) * target_hz / fps_raw)) + 1
        up = target_hz > fps_raw
        pose_local  = (_resample_rotmat_nearest if up else _resample_rotmat_nlerp)(pose_local,  new_len)
        joint_pos_w = (_resample_nearest        if up else _resample_linear      )(joint_pos_w, new_len)
        real_acc    = (_resample_nearest        if up else _resample_linear      )(real_acc,    new_len)
        real_ori    = (_resample_rotmat_nearest if up else _resample_rotmat_nlerp)(real_ori,    new_len)
        N           = new_len
        _log.info("[imuposer_local_disp] resampled → %d frames (%.0f→%.0f Hz)", N, fps_raw, target_hz)

    fps = float(target_hz)

    # ------------------------------------------------------------------
    # SMPL FK → wrist world rotation (for FLU IMU) + pelvis world rotation (for labels)
    # ------------------------------------------------------------------
    body_model  = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))
    root_trans  = joint_pos_w[:, 0]   # [N, 3]

    pose_global_t, _ = body_model.forward_kinematics_masked(
        pose_local, tran=root_trans, calc_mesh=False,
    )   # [N, 24, 3, 3]

    R_world_from_wrist = pose_global_t[:, _SMPL_R_WRIST].cpu().numpy().astype(np.float64)  # [N,3,3]

    # ------------------------------------------------------------------
    # Build FLU frame from wrist
    # ------------------------------------------------------------------
    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_wrist, R_WRIST_FROM_FLU)  # [N,3,3]
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)                                # [N,3,3]

    # ------------------------------------------------------------------
    # Real acc → FLU
    # ------------------------------------------------------------------
    acc_world = real_acc[:, _IMUPOSER_R_WRIST].numpy().astype(np.float64)   # [N, 3]
    if add_gravity:
        # Z-up world: g = [0, 0, -9.81], a_meas = a_kin - g → add [0, 0, +9.81]
        acc_world = acc_world - np.array([0.0, 0.0, -9.81])

    acc_flu = np.einsum("nij,nj->ni", R_X_from_world, acc_world)            # [N, 3]

    # ------------------------------------------------------------------
    # Real gyro → FLU
    # ------------------------------------------------------------------
    R_ws = real_ori[:, _IMUPOSER_R_WRIST].numpy().astype(np.float64)        # [N, 3, 3]
    dR   = np.einsum("nij,njk->nik", R_ws[:-1].transpose(0, 2, 1), R_ws[1:])  # [N-1,3,3]
    omega_sensor = np.zeros((N, 3), dtype=np.float64)
    omega_sensor[:-1] = (
        ScipyRot.from_matrix(dR.reshape(-1, 3, 3)).as_rotvec().astype(np.float64) * fps
    )
    omega_sensor[-1]  = omega_sensor[-2]

    omega_world = np.einsum("nij,nj->ni", R_ws, omega_sensor)               # [N, 3]
    gyro_flu    = np.einsum("nij,nj->ni", R_X_from_world, omega_world)      # [N, 3]

    X_flu = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)  # [N, 6]

    # ------------------------------------------------------------------
    # Root position + FLU-from-world rotation for label computation
    # ------------------------------------------------------------------
    pos_root     = joint_pos_w[:, _SMPL_PELVIS].numpy().astype(np.float32)   # [N, 3]
    R_X_fw_f32  = R_X_from_world.astype(np.float32)                          # [N, 3, 3]

    _log.info(
        "[imuposer_local_disp] %s: N=%d fps=%.0f",
        os.path.basename(pkl_path), N, fps,
    )
    return X_flu, pos_root, R_X_fw_f32


# ---------------------------------------------------------------------------
# Sliding-window Dataset
# ---------------------------------------------------------------------------

class IMUPoserWristLocalDispFLUDataset(Dataset):
    """Sliding-window dataset: right-wrist FLU IMU → per-patch pelvis velocity in FLU frame.

    Each item::

        x : [6, T]              float32 – FLU-frame wrist acc + gyro (real sensor)
        y : [N_patches, 4, 3]   float32 – pelvis velocity (m/s) in wrist FLU frame

    Each patch p is split into 4 sub-patches of Q = patch_size // 4 frames::

        t_start  = start + p*P + k*Q
        t_end    = t_start + Q
        t_mid    = t_start + Q // 2
        v_world  = (pos_root[t_end] − pos_root[t_start]) / (Q / fps)
        y[p, k]  = R_X_from_world[t_mid] @ v_world   (3-D velocity in FLU, m/s)

    patch_size must be divisible by 4.
    """

    def __init__(
        self,
        pkl_paths:   list[str],
        cfg:         dict,
        split_name:  str  = "train",
        target_hz:   int  = 50,
        add_gravity: bool = False,
    ):
        super().__init__()
        self.cfg        = cfg
        self.split_name = split_name
        self.stats      = PreprocessStats()

        vit_cfg            = cfg_get(cfg, ["vit1d"], {}) or {}
        self._patch_size   = int(vit_cfg.get("patch_size", 16))
        self._T, self._hop = compute_T_and_hop(cfg)
        self._num_patches  = self._T // self._patch_size
        self._target_hz    = float(target_hz)
        assert self._T % self._patch_size == 0, (
            f"T={self._T} must be divisible by patch_size={self._patch_size}"
        )
        assert self._patch_size % 4 == 0, (
            f"patch_size={self._patch_size} must be divisible by 4 "
            "(4 velocity sub-patches per patch)"
        )

        # Each session stores (X_flu [N,6], pos_root [N,3], R_X_from_world [N,3,3])
        self._sessions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        session_lengths: list[int] = []

        for pkl in tqdm(pkl_paths, desc=f"[imuposer_local_disp] loading {split_name}"):
            result = load_session_local_disp_flu(
                pkl, target_hz=target_hz, add_gravity=add_gravity
            )
            if result is None:
                continue
            X_flu, pos_root, R_X_fw = result
            # Need at least T + P frames so the last patch has a valid end-frame
            if X_flu.shape[0] < self._T + self._patch_size:
                _log.warning(
                    "[imuposer_local_disp] %s too short (%d), skipping",
                    os.path.basename(pkl), X_flu.shape[0],
                )
                continue
            self._sessions.append((X_flu, pos_root, R_X_fw))
            session_lengths.append(X_flu.shape[0] - self._patch_size)

        if not self._sessions:
            raise RuntimeError(
                "IMUPoserWristLocalDispFLUDataset: no valid sessions loaded from "
                f"{len(pkl_paths)} pkl paths."
            )

        self._sess_idx, self._starts = _build_window_specs(
            session_lengths, self._T, self._hop
        )
        _log.info(
            "[imuposer_local_disp] split=%s  sessions=%d  windows=%d  T=%d  hop=%d",
            split_name, len(self._sessions), len(self._starts), self._T, self._hop,
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int):
        si    = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        T     = self._T
        P     = self._patch_size

        X_flu, pos_root, R_X_fw = self._sessions[si]   # [N,6], [N,3], [N,3,3]

        Xw    = X_flu[start : start + T]             # [T, 6]
        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        Xproc    = apply_augment(Xproc.T, self.cfg).T   # [C, T]
        x_tensor = torch.from_numpy(Xproc).float()       # [6, T]

        # Per-patch label: 4 × 3-D velocity in wrist FLU frame (m/s)
        # Each patch is divided into 4 sub-patches of Q frames.
        # For sub-patch k:  t_start = start + p*P + k*Q
        #                   t_end   = t_start + Q
        #                   t_mid   = t_start + Q//2  (FLU frame evaluated here)
        #   v_world = (pos_root[t_end] − pos_root[t_start]) / (Q / fps)
        #   v_flu   = R_X_from_world[t_mid] @ v_world
        Q     = P // 4                            # frames per sub-patch (e.g. 4 @ 50 Hz)
        dt_Q  = Q / self._target_hz              # duration of one sub-patch (seconds)
        vel   = np.empty((self._num_patches, 4, 3), dtype=np.float32)
        for p in range(self._num_patches):
            t_p = start + p * P
            for k in range(4):
                t_s   = t_p + k * Q
                t_e   = t_s + Q
                t_mid = t_s + Q // 2
                v_w   = (pos_root[t_e].astype(np.float64)
                         - pos_root[t_s].astype(np.float64)) / dt_Q   # [3] world
                vel[p, k] = (R_X_fw[t_mid].astype(np.float64) @ v_w).astype(np.float32)

        y_tensor = torch.from_numpy(vel).float()   # [N_patches, 4, 3]
        return x_tensor, y_tensor


# ---------------------------------------------------------------------------
# Collate helper
# ---------------------------------------------------------------------------

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_imuposer_local_disp_flu_loaders(
    cfg:         dict,
    train_files: list[str],
    val_files:   list[str] | None = None,
    test_files:  list[str] | None = None,
    target_hz:   int  = 50,
    add_gravity: bool = False,
) -> dict[str, DataLoader]:
    """Build train / val / test DataLoaders for IMUPoser FLU-frame velocity.

    Each batch::

        x : [B, 6, T]              float32 – right-wrist FLU IMU (real sensor)
        y : [B, N_patches, 4, 3]   float32 – pelvis velocity in wrist FLU frame (m/s)
                                              y[b, p, k] = velocity at sub-patch k of patch p

    Parameters
    ----------
    cfg          : experiment config (OmegaConf or dict).
    train_files  : list of absolute paths to training pkl files.
    val_files    : list of absolute paths to validation pkl files (or None).
    test_files   : list of absolute paths to test pkl files (or None).
    target_hz    : target sampling rate.
    add_gravity  : see ``load_session_local_disp_flu``.
    """
    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],      256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],       0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],    False))
    persistent      = num_workers > 0

    def _make(name: str, files: list[str], bs: int, shuffle: bool):
        if not files:
            return None
        ds = IMUPoserWristLocalDispFLUDataset(
            pkl_paths=files, cfg=cfg, split_name=name,
            target_hz=target_hz, add_gravity=add_gravity,
        )
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

    loaders = {}
    for name, files, bs, shuf in [
        ("train", train_files,       batch_size,      True),
        ("val",   val_files  or [],  eval_batch_size, False),
        ("test",  test_files or [],  eval_batch_size, False),
    ]:
        loader = _make(name, files, bs, shuf)
        if loader is not None:
            loaders[f"{name}_loader"] = loader
    return loaders


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob
    import yaml
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)

    cfg = yaml.safe_load(open("configs/base.yaml"))
    cfg.setdefault("vit1d", {})
    cfg["vit1d"]["patch_size"] = 16
    cfg.setdefault("data", {})
    cfg["data"]["batch_size"] = 4

    pkls = sorted(glob.glob(
        "/mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/*.pkl"
    ))[:3]
    assert pkls, "No pkl files found – update the glob path."

    print(f"Testing with {len(pkls)} pkl(s)")
    result = load_session_local_disp_flu(pkls[0])
    assert result is not None
    X, pos, R_X_fw = result
    print(f"  X: {X.shape}  pos_root: {pos.shape}  R_X_from_world: {R_X_fw.shape}")

    # Verify velocity at first sub-patch of first patch (Q=4, fps=50)
    fps = 50.0
    P, Q = 16, 4
    dt_Q    = Q / fps
    v_world = (pos[Q].astype(float) - pos[0].astype(float)) / dt_Q
    v_flu   = R_X_fw[Q // 2].astype(float) @ v_world
    print(f"  world velocity [0..Q]: {v_world}")
    print(f"  FLU   velocity [0..Q]: {v_flu}")
    print("  (F=finger-fwd, L=thumb-side, U=watch-face-up)")
