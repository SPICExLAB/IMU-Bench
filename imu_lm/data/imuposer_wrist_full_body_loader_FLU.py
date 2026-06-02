"""imuposer_wrist_full_body_loader_FLU.py
IMUPoser right-wrist loader in the FLU body frame — full SMPL body labels.

Identical IMU derivation pipeline to imuposer_wrist_loader_FLU.py.  The only
difference is that labels include **all 24 SMPL joints** (24 × 6D = 144 dims)
instead of only the right-arm chain (5 joints × 6D = 30 dims).

FLU body frame — defined at the right-wrist joint in T-pose:
   Forward (FLU x) = -x_smpl   (fingertip direction in SMPL wrist-local coords)
   Left    (FLU y) = +z_smpl   (thumb side)
   Up      (FLU z) = +y_smpl   (dorsal / watch-face)

Pipeline:
  1.  SMPL FK on local rotations → world-frame wrist rotation R_world_from_wrist(t)
                                    and wrist world position p_world_wrist(t).
  2.  Propagate the FLU frame (X) to world using the T-pose constant:
          R_world_from_X(t) = R_world_from_wrist(t) @ R_WRIST_FROM_FLU
      Inverse: R_X_from_world(t) = R_world_from_X(t)^T
  3.  Real right-wrist accelerometer  (real_imu_data["real_acc"][:, 1]) is already in
      world frame (Z-up, gravity-subtracted).  Express it in X:
          acc_flu = R_X_from_world(t) @ acc_world(t)
  4.  Real gyro: derive omega in world frame from consecutive real orientations,
      then express in X:
          omega_world = R_world_from_sensor(t) @ omega_sensor(t)
          gyro_flu    = R_X_from_world(t) @ omega_world(t)

Inputs
------
IMUPoser .pkl files containing:
  gt_data["pose"]                  – [N, 24, 3, 3] SMPL local rotations  (labels + FK)
  gt_data["joint_pos"]             – [N, 24, 3]   world-space joint positions Z-up
  real_imu_data["real_acc"][:, 1]  – [N, 3]  real R-wrist accelerometer, Z-up world frame
  real_imu_data["real_ori"][:, 1]  – [N, 3, 3]  real R-wrist R_world_from_sensor

Outputs per window
------------------
  x : [6, T]             float32  – (acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U)
  y : [N_patches, 144]   float32  – 6D rotations for all 24 SMPL joints
"""

from __future__ import annotations

import os
import pickle
import logging

import numpy as np
import torch
from tqdm import tqdm

from imu_lm.data.imuposerdataset import (
    matrix_to_rotation_6d,
    _resample_linear,
    _resample_nearest,
    _resample_rotmat_nlerp,
    _resample_rotmat_nearest,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SMPL joint indices — full body
# ---------------------------------------------------------------------------
_SMPL_R_WRIST  = 21
_ALL_JOINTS    = list(range(24))   # all 24 SMPL joints

NUM_BODY_JOINTS = 24
ROT6D_DIM       = 6

# ---------------------------------------------------------------------------
# T-pose FLU definition  (matches imuposer_wrist_loader_FLU.py exactly)
# ---------------------------------------------------------------------------
R_WRIST_FROM_FLU = np.array(
    [[-1.0,  0.0,  0.0],   # F
     [ 0.0,  0.0,  1.0],   # L
     [ 0.0,  1.0,  0.0]],  # U
    dtype=np.float64,
)
assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Session-level loader
# ---------------------------------------------------------------------------

def load_session_full_body_flu(
    pkl_path: str,
    target_hz: int = 50,
    add_gravity: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load one IMUPoser pkl and return **(X, y)** in the FLU frame.

    Parameters
    ----------
    pkl_path    : str   Path to an IMUPoser ``.pkl`` sequence file.
    target_hz   : int   Target sampling rate; data is resampled if needed.
    add_gravity : bool
        If True, add back the gravity vector to the (gravity-subtracted) real_acc.
        Default: False.

    Returns
    -------
    X : [N, 6]     float32 – ``[acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]``
    y : [N, 24, 6] float32 – 6D rotations for all 24 SMPL joints (local rotations)
    ``None`` on any failure.
    """
    from scipy.spatial.transform import Rotation as ScipyRot
    import core.articulate as art
    from core.paths import Paths

    _log.info("[imuposer_full_body_flu] loading: %s", os.path.basename(pkl_path))

    try:
        with open(pkl_path, "rb") as fh:
            seq_data = pickle.load(fh)
    except Exception as exc:
        _log.warning("[imuposer_full_body_flu] cannot open %s: %s", pkl_path, exc)
        return None

    gt          = seq_data.get("gt_data", {})
    pose_local  = gt.get("pose")        # [N, 24, 3, 3]  SMPL local rotations
    joint_pos_w = gt.get("joint_pos")   # [N, 24, 3]     Z-up world positions

    ri       = seq_data.get("real_imu_data", {})
    real_acc = ri.get("real_acc")       # [N, 5, 3]      real accelerometer, Z-up world
    real_ori = ri.get("real_ori")       # [N, 5, 3, 3]   R_world_from_sensor, Z-up world

    if pose_local is None or joint_pos_w is None:
        _log.warning("[imuposer_full_body_flu] %s: missing gt fields.", pkl_path)
        return None
    if real_acc is None or real_ori is None:
        _log.warning("[imuposer_full_body_flu] %s: missing real_imu_data fields.", pkl_path)
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
        _log.warning("[imuposer_full_body_flu] %s: too short (%d frames).", pkl_path, N)
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
        _log.info(
            "[imuposer_full_body_flu] resampled → %d frames (%.0f→%.0f Hz)",
            N, fps_raw, target_hz,
        )

    fps = float(target_hz)

    # ------------------------------------------------------------------
    # Step 1 — SMPL FK → world-frame wrist rotation R_world_from_wrist(t)
    # ------------------------------------------------------------------
    body_model = art.model.ParametricModel(
        Paths.SMPL_FILE, device=torch.device("cpu")
    )
    root_trans = joint_pos_w[:, 0]   # [N, 3]  pelvis world position

    pose_global_t, _ = body_model.forward_kinematics_masked(
        pose_local,
        tran=root_trans,
        calc_mesh=False,
    )  # [N, 24, 3, 3], [N, 24, 3]

    R_world_from_wrist = pose_global_t[:, _SMPL_R_WRIST].cpu().numpy().astype(np.float64)  # [N,3,3]

    # ------------------------------------------------------------------
    # Step 2 — propagate X (= FLU) to world frame
    # ------------------------------------------------------------------
    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_wrist, R_WRIST_FROM_FLU)  # [N,3,3]
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)                                # [N,3,3]

    # ------------------------------------------------------------------
    # Step 3 — real accelerometer: already in Z-up world frame → FLU
    # ------------------------------------------------------------------
    acc_world = real_acc[:, 1].numpy().astype(np.float64)            # [N, 3]

    if add_gravity:
        g_world   = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        acc_world = acc_world - g_world

    acc_flu = np.einsum("nij,nj->ni", R_X_from_world, acc_world)    # [N, 3]

    # ------------------------------------------------------------------
    # Step 4 — real gyro: sensor-local omega → world → FLU
    # ------------------------------------------------------------------
    R_world_from_sensor = real_ori[:, 1].numpy().astype(np.float64)  # [N, 3, 3]

    dR = np.einsum("nij,njk->nik",
                   R_world_from_sensor[:-1].transpose(0, 2, 1),
                   R_world_from_sensor[1:])                           # [N-1, 3, 3]
    omega_raw = (
        ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
        .as_rotvec()
        .astype(np.float64) * fps
    )  # [N-1, 3]  in sensor local frame
    omega_sensor = np.zeros((N, 3), dtype=np.float64)
    omega_sensor[:-1] = omega_raw
    omega_sensor[-1]  = omega_raw[-1]

    omega_world = np.einsum("nij,nj->ni", R_world_from_sensor, omega_sensor)  # [N, 3]
    gyro_flu    = np.einsum("nij,nj->ni", R_X_from_world, omega_world)        # [N, 3]

    X = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)         # [N, 6]

    # ------------------------------------------------------------------
    # Labels: 6D rotation for ALL 24 SMPL joints (local rotations)
    # ------------------------------------------------------------------
    all_R = pose_local[:, _ALL_JOINTS]                                # [N, 24, 3, 3]
    y     = matrix_to_rotation_6d(all_R).numpy().astype(np.float32)  # [N, 24, 6]

    _log.info(
        "[imuposer_full_body_flu] %s: N=%d  fps=%.0f",
        os.path.basename(pkl_path), N, fps,
    )
    return X, y


# ---------------------------------------------------------------------------
# Sliding-window Dataset
# ---------------------------------------------------------------------------

class IMUPoserWristFullBodyFLUDataset(torch.utils.data.Dataset):
    """Sliding-window dataset over IMUPoser sessions — full SMPL body labels.

    Each item:
        x : [6, T]            float32  – (acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U)
        y : [N_patches, 144]  float32  – all 24 joint 6D rotations per time patch
    """

    def __init__(
        self,
        pkl_paths:   list[str],
        cfg:         dict,
        split_name:  str  = "train",
        target_hz:   int  = 50,
        add_gravity: bool = False,
    ):
        from imu_lm.data.augmentations.preprocess import PreprocessStats
        from imu_lm.data.windowing import compute_T_and_hop
        from imu_lm.utils.helpers import cfg_get

        self.cfg         = cfg
        self.split_name  = split_name
        self.add_gravity = add_gravity
        self.stats       = PreprocessStats()

        vit_cfg              = cfg_get(cfg, ["vit1d"], {}) or {}
        self._patch_size     = int(vit_cfg.get("patch_size", 16))
        self._T, self._hop   = compute_T_and_hop(cfg)
        self._num_patches    = self._T // self._patch_size
        assert self._T % self._patch_size == 0, (
            f"T={self._T} must be divisible by patch_size={self._patch_size}"
        )

        self._sessions: list[tuple[np.ndarray, np.ndarray]] = []
        self._sess_idx: list[int] = []
        self._starts:   list[int] = []

        for pkl in tqdm(pkl_paths, desc=f"[imuposer_full_body_flu] loading {split_name}"):
            result = load_session_full_body_flu(pkl, target_hz=target_hz, add_gravity=add_gravity)
            if result is None:
                continue
            X, y = result          # X: [N, 6],  y: [N, 24, 6]
            if X.shape[0] < self._T:
                continue

            si = len(self._sessions)
            self._sessions.append((X, y))

            n_windows = (X.shape[0] - self._T) // self._hop + 1
            for s in range(n_windows):
                self._sess_idx.append(si)
                self._starts.append(s * self._hop)

        self._sess_idx = np.array(self._sess_idx, dtype=np.int32)
        self._starts   = np.array(self._starts,   dtype=np.int32)

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int):
        from imu_lm.data.augmentations.preprocess import preprocess_window
        from imu_lm.data.augmentations.transform  import apply_augment

        si    = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        T     = self._T
        X, y  = self._sessions[si]          # [N, 6], [N, 24, 6]

        Xw = X[start : start + T]           # [T, 6]

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        Xproc    = apply_augment(Xproc.T, self.cfg).T          # [C, T]
        x_tensor = torch.from_numpy(Xproc).float()             # [6, T]

        # Labels: one 144-dim vector per patch (sampled at patch centre)
        P = self._patch_size
        patch_labels = np.empty(
            (self._num_patches, NUM_BODY_JOINTS * ROT6D_DIM), dtype=np.float32
        )
        for p in range(self._num_patches):
            frame = start + p * P + P // 2
            patch_labels[p] = y[frame].reshape(-1)    # [24, 6] → [144]

        y_tensor = torch.from_numpy(patch_labels).float()     # [N_patches, 144]
        return x_tensor, y_tensor


# ---------------------------------------------------------------------------
# Collate helper
# ---------------------------------------------------------------------------

def collate_skip_none(batch):
    """Drop ``None`` entries (bad windows) and stack the rest."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

class IMUPoserWristFullBody23FLUDataset(IMUPoserWristFullBodyFLUDataset):
    """Like IMUPoserWristFullBodyFLUDataset but drops the root (pelvis) joint.

    Labels are 23 joints × 6D = 138 dims per patch (SMPL joints 1–23).
    """

    _NUM_JOINTS_23 = 23

    def __getitem__(self, idx: int):
        from imu_lm.data.augmentations.preprocess import preprocess_window
        from imu_lm.data.augmentations.transform  import apply_augment

        si    = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        T     = self._T
        X, y  = self._sessions[si]          # [N, 6], [N, 24, 6]

        Xw = X[start : start + T]

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        Xproc    = apply_augment(Xproc.T, self.cfg).T
        x_tensor = torch.from_numpy(Xproc).float()

        P = self._patch_size
        patch_labels = np.empty(
            (self._num_patches, self._NUM_JOINTS_23 * ROT6D_DIM), dtype=np.float32
        )
        for p in range(self._num_patches):
            frame = start + p * P + P // 2
            patch_labels[p] = y[frame, 1:].reshape(-1)   # drop joint 0 (pelvis root)

        y_tensor = torch.from_numpy(patch_labels).float()
        return x_tensor, y_tensor


def make_imuposer_wrist_full_body_23_flu_loaders(
    cfg:          dict,
    train_files:  list[str],
    val_files:    list[str] | None = None,
    test_files:   list[str] | None = None,
    target_hz:    int  = 50,
    add_gravity:  bool = False,
) -> dict[str, torch.utils.data.DataLoader]:
    """Build train / val / test DataLoaders — 23 SMPL joints (no pelvis) in FLU frame.

    Each batch yields::

        x : [B, 6, T]            float32  – right-wrist IMU in FLU body frame
        y : [B, N_patches, 138]  float32  – joints 1–23 6D rotations per patch
    """
    from imu_lm.utils.helpers import cfg_get

    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],       256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"],  batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],        0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],     False))
    persistent      = num_workers > 0

    def _make(name: str, files: list[str], bs: int, shuffle: bool):
        if not files:
            return None
        ds = IMUPoserWristFullBody23FLUDataset(
            pkl_paths=files, cfg=cfg, split_name=name, target_hz=target_hz,
            add_gravity=add_gravity,
        )
        return torch.utils.data.DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=shuffle,
            persistent_workers=persistent,
            collate_fn=collate_skip_none,
        )

    return {
        "train_loader": _make("train", train_files,       batch_size,      True),
        "val_loader":   _make("val",   val_files  or [],  eval_batch_size, False),
        "test_loader":  _make("test",  test_files or [],  eval_batch_size, False),
    }


def make_imuposer_wrist_full_body_flu_loaders(
    cfg:          dict,
    train_files:  list[str],
    val_files:    list[str] | None = None,
    test_files:   list[str] | None = None,
    target_hz:    int  = 50,
    add_gravity:  bool = False,
) -> dict[str, torch.utils.data.DataLoader]:
    """Build train / val / test DataLoaders — full SMPL body in FLU frame.

    Each batch yields::

        x : [B, 6, T]            float32  – right-wrist IMU in FLU body frame
        y : [B, N_patches, 144]  float32  – all 24 joint 6D rotations per patch
    """
    from imu_lm.utils.helpers import cfg_get

    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],       256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"],  batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],        0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],     False))
    persistent      = num_workers > 0

    def _make(name: str, files: list[str], bs: int, shuffle: bool):
        if not files:
            return None
        ds = IMUPoserWristFullBodyFLUDataset(
            pkl_paths=files, cfg=cfg, split_name=name, target_hz=target_hz,
            add_gravity=add_gravity,
        )
        return torch.utils.data.DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=shuffle,
            persistent_workers=persistent,
            collate_fn=collate_skip_none,
        )

    return {
        "train_loader": _make("train", train_files,       batch_size,      True),
        "val_loader":   _make("val",   val_files  or [],  eval_batch_size, False),
        "test_loader":  _make("test",  test_files or [],  eval_batch_size, False),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import glob
    import yaml

    logging.basicConfig(level=logging.INFO)

    cfg = yaml.safe_load(open("configs/base.yaml"))
    cfg.setdefault("vit1d", {}); cfg["vit1d"]["patch_size"] = 16
    cfg.setdefault("data", {}); cfg["data"]["batch_size"] = 8

    pkls = sorted(glob.glob(
        "/mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/*.pkl"
    ))[:3]
    assert pkls, "No pkl files found – update the glob path."

    print(f"Testing with {len(pkls)} pkl(s): {[os.path.basename(p) for p in pkls]}")

    X, y = load_session_full_body_flu(pkls[0])
    print(f"  X: {X.shape}  y: {y.shape}")   # expect [N, 6], [N, 24, 6]
    acc_mag = np.linalg.norm(X[:, :3], axis=1)
    print(f"  acc_mag  mean={acc_mag.mean():.3f}  std={acc_mag.std():.3f}")

    ds = IMUPoserWristFullBodyFLUDataset(pkls, cfg, split_name="test")
    print(f"  windows: {len(ds)}")
    item = next(x for x in (ds[i] for i in range(min(len(ds), 200))) if x is not None)
    print(f"  x: {item[0].shape}  y: {item[1].shape}")  # expect [6, T], [P, 144]

    loaders = make_imuposer_wrist_full_body_flu_loaders(cfg, train_files=pkls, val_files=pkls)
    for split, dl in loaders.items():
        if dl is not None:
            batch = next(iter(dl))
            if batch is not None:
                print(f"  {split}: x={batch[0].shape}  y={batch[1].shape}")
    print("Smoke test PASSED.")
