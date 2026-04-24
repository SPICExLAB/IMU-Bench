"""imuposer_wrist_odom_loader_FLU.py
IMUPoser right-wrist IMU (FLU body frame) → root displacement odometry labels.

IMU derivation (FLU frame — same pipeline as imuposer_wrist_loader_FLU.py)
---------------------------------------------------------------------------
Uses **real** right-wrist sensor readings from the IMUPoser pkl:
  real_imu_data["real_acc"][:, 1]  –  real accelerometer, Z-up world frame
                                        (gravity-subtracted; add_gravity=True to restore)
  real_imu_data["real_ori"][:, 1]  –  real R_world_from_sensor (Z-up world frame)

FLU body frame (T-pose, right wrist):
  Forward (FLU x) = -x_smpl
  Left    (FLU y) = +z_smpl
  Up      (FLU z) = +y_smpl

Pipeline:
  1. SMPL FK on gt_data["pose"] → R_world_from_wrist(t)
  2. R_world_from_X = R_world_from_wrist @ R_WRIST_FROM_FLU
     R_X_from_world = R_world_from_X^T
  3. acc_flu  = R_X_from_world @ acc_world           (real accelerometer → FLU)
  4. gyro_flu = R_X_from_world @ (R_world_from_sensor @ omega_sensor)
                                                     (real gyro → FLU)

Root displacement labels
------------------------
  pos_root[t] = gt_data["joint_pos"][t, 0]   (pelvis world position, Z-up)
  y[p]        = pos_root[start + (p+1)*P] − pos_root[start + p*P]
  shape       : [N_patches, 3]  (XYZ in Z-up world frame, metres)

Inputs per window
-----------------
  x : [6, T]            float32  – (acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U)
  y : [N_patches, 3]    float32  – per-patch root displacement (metres)
"""

from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from imu_lm.data.imuposerdataset import (
    matrix_to_rotation_6d,
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
# Constants — match imuposer_wrist_loader_FLU.py exactly
# ---------------------------------------------------------------------------
_SMPL_R_WRIST     = 21          # SMPL right wrist joint index
_SMPL_PELVIS      = 0           # SMPL pelvis (root) joint index for displacement labels
_IMUPOSER_R_WRIST = 1           # sensor index in real_acc / real_ori: 0=pelvis, 1=R-wrist

R_WRIST_FROM_FLU = np.array(
    [[-1.0,  0.0,  0.0],   # Forward
     [ 0.0,  0.0,  1.0],   # Left
     [ 0.0,  1.0,  0.0]],  # Up
    dtype=np.float64,
)
assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Window index builder
# ---------------------------------------------------------------------------

def _build_window_specs(session_lengths, T, hop):
    # sessions are windowed to length-P so last patch has a valid end-frame
    total = sum(max(0, (n - T) // hop + 1) for n in session_lengths)
    sess_idx = np.empty(total, dtype=np.int32)
    starts   = np.empty(total, dtype=np.int32)
    offset = 0
    for ki, n in enumerate(session_lengths):
        if n < T:
            continue
        n_win = (n - T) // hop + 1
        sess_idx[offset:offset + n_win] = ki
        starts[offset:offset + n_win]   = np.arange(0, n - T + 1, hop, dtype=np.int32)
        offset += n_win
    return sess_idx, starts


# ---------------------------------------------------------------------------
# Session loader
# ---------------------------------------------------------------------------

def load_session_odom_flu(
    pkl_path: str,
    target_hz: int = 50,
    add_gravity: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load one IMUPoser pkl → (X_flu, pos_root).

    Parameters
    ----------
    pkl_path    : str   Path to IMUPoser ``.pkl`` file.
    target_hz   : int   Target sampling rate.
    add_gravity : bool  If True, add gravity back to real_acc so it matches a
                        full MEMS specific-force reading (Z-up world: +9.81 in Z).

    Returns
    -------
    X_flu    : [N, 6]  float32 – ``[acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]``
    pos_root : [N, 3]  float32 – pelvis world position (Z-up, metres)  used for labels
    ``None`` on failure.
    """
    from scipy.spatial.transform import Rotation as ScipyRot
    import core.articulate as art
    from core.paths import Paths

    _log.info("[imuposer_odom_flu] loading: %s", os.path.basename(pkl_path))

    try:
        with open(pkl_path, "rb") as fh:
            seq_data = pickle.load(fh)
    except Exception as exc:
        _log.warning("[imuposer_odom_flu] cannot open %s: %s", pkl_path, exc)
        return None

    gt          = seq_data.get("gt_data", {})
    pose_local  = gt.get("pose")        # [N, 24, 3, 3]  SMPL local rotations
    joint_pos_w = gt.get("joint_pos")   # [N, 24, 3]     Z-up world positions

    ri       = seq_data.get("real_imu_data", {})
    real_acc = ri.get("real_acc")       # [N, 5, 3]   real acc, Z-up world
    real_ori = ri.get("real_ori")       # [N, 5, 3, 3] R_world_from_sensor, Z-up world

    if pose_local is None or joint_pos_w is None:
        _log.warning("[imuposer_odom_flu] %s: missing gt fields.", pkl_path)
        return None
    if real_acc is None or real_ori is None:
        _log.warning("[imuposer_odom_flu] %s: missing real_imu_data fields.", pkl_path)
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
        _log.warning("[imuposer_odom_flu] %s: too short (%d frames).", pkl_path, N)
        return None

    # --- Resample if needed -----------------------------------------------
    if abs(fps_raw - target_hz) > 1e-3:
        new_len = int(round((N - 1) * target_hz / fps_raw)) + 1
        up = target_hz > fps_raw
        pose_local  = (_resample_rotmat_nearest if up else _resample_rotmat_nlerp)(pose_local,  new_len)
        joint_pos_w = (_resample_nearest        if up else _resample_linear      )(joint_pos_w, new_len)
        real_acc    = (_resample_nearest        if up else _resample_linear      )(real_acc,    new_len)
        real_ori    = (_resample_rotmat_nearest if up else _resample_rotmat_nlerp)(real_ori,    new_len)
        N           = new_len
        _log.info("[imuposer_odom_flu] resampled → %d frames (%.0f→%.0f Hz)", N, fps_raw, target_hz)

    fps = float(target_hz)

    # --- Step 1: SMPL FK → R_world_from_wrist(t) -------------------------
    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))
    root_trans = joint_pos_w[:, 0]   # [N, 3]  pelvis world position

    pose_global_t, _ = body_model.forward_kinematics_masked(
        pose_local, tran=root_trans, calc_mesh=False,
    )  # [N, 24, 3, 3]

    R_world_from_wrist = pose_global_t[:, _SMPL_R_WRIST].cpu().numpy().astype(np.float64)  # [N,3,3]

    # --- Step 2: FLU frame to world frame ---------------------------------
    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_wrist, R_WRIST_FROM_FLU)  # [N,3,3]
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)                                # [N,3,3]

    # --- Step 3: acc world → FLU ------------------------------------------
    acc_world = real_acc[:, _IMUPOSER_R_WRIST].numpy().astype(np.float64)  # [N, 3]

    if add_gravity:
        # World frame is Z-up: g_world = [0, 0, -9.81]  →  a_meas = a_kin - g_world
        g_world = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        acc_world = acc_world - g_world   # adds +9.81 in Z

    acc_flu = np.einsum("nij,nj->ni", R_X_from_world, acc_world)  # [N, 3]

    # --- Step 4: gyro world → FLU -----------------------------------------
    R_world_from_sensor = real_ori[:, _IMUPOSER_R_WRIST].numpy().astype(np.float64)  # [N, 3, 3]

    dR = np.einsum("nij,njk->nik",
                   R_world_from_sensor[:-1].transpose(0, 2, 1),
                   R_world_from_sensor[1:])                           # [N-1, 3, 3]
    omega_sensor = np.zeros((N, 3), dtype=np.float64)
    omega_sensor[:-1] = (
        ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
        .as_rotvec()
        .astype(np.float64) * fps
    )
    omega_sensor[-1] = omega_sensor[-2]

    omega_world = np.einsum("nij,nj->ni", R_world_from_sensor, omega_sensor)  # [N, 3]
    gyro_flu    = np.einsum("nij,nj->ni", R_X_from_world, omega_world)         # [N, 3]

    X_flu = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)  # [N, 6]

    # --- Root (pelvis) world-frame position for displacement labels -------
    # joint_pos_w[:, 0] is the pelvis position in Z-up world frame (metres)
    pos_root = joint_pos_w[:, _SMPL_PELVIS].numpy().astype(np.float32)  # [N, 3]

    _log.info("[imuposer_odom_flu] %s: N=%d fps=%.0f", os.path.basename(pkl_path), N, fps)
    return X_flu, pos_root


# ---------------------------------------------------------------------------
# Sliding-window Dataset
# ---------------------------------------------------------------------------

class IMUPoserWristOdomFLUDataset(Dataset):
    """Sliding-window dataset: IMUPoser right-wrist FLU IMU → root displacement.

    Each item::

        x : [6, T]           float32  – FLU-frame wrist acc + gyro (real sensor)
        y : [N_patches, 3]   float32  – per-patch root (pelvis) XYZ displacement (metres)
                                         y[p] = pos_root[start+(p+1)*P] − pos_root[start+p*P]
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
        assert self._T % self._patch_size == 0, (
            f"T={self._T} must be divisible by patch_size={self._patch_size}"
        )

        self._sessions: list[tuple[np.ndarray, np.ndarray]] = []  # [(X_flu, pos_root)]
        session_lengths: list[int] = []

        for pkl in tqdm(pkl_paths, desc=f"[imuposer_odom_flu] loading {split_name}"):
            result = load_session_odom_flu(pkl, target_hz=target_hz, add_gravity=add_gravity)
            if result is None:
                continue
            X_flu, pos_root = result
            # Need at least T + patch_size frames so the last patch has an end-frame
            if X_flu.shape[0] < self._T + self._patch_size:
                _log.warning("[imuposer_odom_flu] %s too short (%d), skipping", pkl, X_flu.shape[0])
                continue
            self._sessions.append((X_flu, pos_root))
            # Usable window length: N − patch_size (last patch needs pos_root[start+T+P])
            session_lengths.append(X_flu.shape[0] - self._patch_size)

        if not self._sessions:
            raise RuntimeError(
                f"IMUPoserWristOdomFLUDataset: no valid sessions loaded from "
                f"{len(pkl_paths)} pkl paths."
            )

        self._sess_idx, self._starts = _build_window_specs(
            session_lengths, self._T, self._hop
        )

        _log.info(
            "[imuposer_odom_flu] split=%s  sessions=%d  windows=%d  T=%d  hop=%d",
            split_name, len(self._sessions), len(self._starts), self._T, self._hop,
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int):
        si    = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        T     = self._T
        P     = self._patch_size

        X_flu, pos_root = self._sessions[si]  # [N, 6], [N, 3]

        Xw    = X_flu[start : start + T]             # [T, 6]
        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        Xproc    = apply_augment(Xproc.T, self.cfg).T  # [C, T]
        x_tensor = torch.from_numpy(Xproc).float()     # [6, T]

        # Per-patch root displacement
        # y[p] = pos_root[start + (p+1)*P] − pos_root[start + p*P]
        disp = np.empty((self._num_patches, 3), dtype=np.float32)
        for p in range(self._num_patches):
            f0 = start + p * P
            f1 = f0 + P
            disp[p] = pos_root[f1] - pos_root[f0]

        y_tensor = torch.from_numpy(disp).float()  # [N_patches, 3]
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

def make_imuposer_odom_flu_loaders(
    cfg:         dict,
    train_files: list[str],
    val_files:   list[str] | None = None,
    test_files:  list[str] | None = None,
    target_hz:   int  = 50,
    add_gravity: bool = False,
) -> dict[str, DataLoader]:
    """Build train / val / test DataLoaders for IMUPoser odometry in FLU frame.

    Each batch::

        x : [B, 6, T]           float32  – right-wrist FLU IMU (real sensor)
        y : [B, N_patches, 3]   float32  – per-patch root displacement (metres)
    """
    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],      256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],       0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],    False))
    persistent      = num_workers > 0

    def _make(name: str, files: list[str], bs: int, shuffle: bool):
        if not files:
            return None
        ds = IMUPoserWristOdomFLUDataset(
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
    X, pos = load_session_odom_flu(pkls[0])
    print(f"  X: {X.shape}  pos_root: {pos.shape}")
    print(f"  acc_mag mean={np.linalg.norm(X[:, :3], axis=1).mean():.3f}")

    ds = IMUPoserWristOdomFLUDataset(pkls, cfg, split_name="test")
    print(f"  windows: {len(ds)}")
    item = next(b for b in (ds[i] for i in range(min(len(ds), 50))) if b is not None)
    print(f"  x: {item[0].shape}  y: {item[1].shape}")

    loaders = make_imuposer_odom_flu_loaders(cfg, train_files=pkls, val_files=pkls)
    for split, dl in loaders.items():
        if dl:
            batch = next(iter(dl))
            if batch:
                print(f"  {split}: x={batch[0].shape}  y={batch[1].shape}")
    print("Smoke test PASSED.")
