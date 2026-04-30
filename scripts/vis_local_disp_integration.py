from __future__ import annotations
import argparse, os, pickle, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("scripts/vis_local_disp_integration.py")), ".."))
sys.path.insert(0, ".")
from imu_lm.data.imuposer_wrist_local_odom_loader_FLU import (
    load_session_local_disp_flu,
    R_WRIST_FROM_FLU,
    _IMUPOSER_R_WRIST,
)
from imu_lm.data.imuposerdataset import _resample_rotmat_nlerp


def extract_velocity_labels(pos_root, R_X_from_world, patch_size, fps):
    N, P, Q, dt_Q = pos_root.shape[0], patch_size, patch_size // 4, (patch_size // 4) / fps
    M = (N - P) // P
    n_sub = M * 4
    v_flu, t_mids, t_starts = (np.zeros((n_sub, 3)), np.zeros(n_sub, np.int32), np.zeros(n_sub, np.int32))
    for p in range(M):
        for k in range(4):
            i = p * 4 + k
            t_s, t_e, t_mid = p * P + k * Q, p * P + k * Q + Q, p * P + k * Q + Q // 2
            t_starts[i], t_mids[i] = t_s, t_mid
            v_w = (pos_root[t_e].astype(np.float64) - pos_root[t_s].astype(np.float64)) / dt_Q
            v_flu[i] = R_X_from_world[t_mid].astype(np.float64) @ v_w
    return v_flu, t_mids, t_starts


def integrate_gyro(gyro_flu, R0_wfx, fps):
    N, dt = gyro_flu.shape[0], 1.0 / fps
    R = np.zeros((N, 3, 3))
    R[0] = R0_wfx
    for t in range(N - 1):
        R[t + 1] = R[t] @ ScipyRot.from_rotvec(gyro_flu[t] * dt).as_matrix()
    return R


def integrate_trajectory(v_flu, t_starts, t_mids, R_wfx, pos0, fps, Q):
    dt_Q = Q / fps
    pos = np.zeros((len(v_flu) + 1, 3))
    pos[0] = pos0
    for i, (tm, vf) in enumerate(zip(t_mids, v_flu)):
        pos[i + 1] = pos[i] + R_wfx[tm] @ vf * dt_Q
    return pos


def _load_real_ori_wrist(pkl_path: str, target_hz: int) -> np.ndarray:
    """Load and resample real_ori[:, wrist] from a pkl → R_world_from_FLU [N,3,3]."""
    with open(pkl_path, "rb") as fh:
        seq_data = pickle.load(fh, encoding="latin1" if sys.version_info[0] >= 3 else "bytes")
    ri = seq_data.get("real_imu_data", {})
    real_ori = ri.get("real_ori")                              # [N, 5, 3, 3]
    real_ori = np.asarray(real_ori, dtype=np.float64)
    N_raw    = real_ori.shape[0]
    fps_raw  = float(seq_data.get("sampling_rate_hz", target_hz))
    if abs(fps_raw - target_hz) > 1e-3:
        import torch
        new_len  = int(round((N_raw - 1) * target_hz / fps_raw)) + 1
        real_ori = _resample_rotmat_nlerp(
            torch.as_tensor(real_ori.astype(np.float32)), new_len
        ).numpy().astype(np.float64)
    R_ws  = real_ori[:, _IMUPOSER_R_WRIST]                    # [N, 3, 3]  R_world_from_sensor
    # Apply T-pose FLU offset:  R_world_from_FLU = R_world_from_sensor @ R_sensor_from_FLU
    # R_sensor_from_FLU == R_WRIST_FROM_FLU (same T-pose convention)
    R_wfx = np.einsum("nij,jk->nik", R_ws, R_WRIST_FROM_FLU) # [N, 3, 3]  R_world_from_FLU
    return R_wfx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="/mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/imuposer_seq_00033.pkl")
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--target-hz",  type=int, default=50)
    ap.add_argument("--out",        default=None)
    args = ap.parse_args()
    fps, P, Q = float(args.target_hz), args.patch_size, args.patch_size // 4

    print(f"Loading: {args.pkl}")
    result = load_session_local_disp_flu(args.pkl, target_hz=args.target_hz)
    assert result is not None
    X_flu, pos_root, R_X_fw = result
    N, M = pos_root.shape[0], (pos_root.shape[0] - P) // P
    print(f"  Frames={N}  patches={M}  sub-patches={M*4}  Q={Q} frames = {Q/fps:.2f}s")

    # Real IMU wrist orientation (R_world_from_FLU from raw sensor, not FK-derived)
    R_realori_wfx = _load_real_ori_wrist(args.pkl, args.target_hz)

    gyro_flu = X_flu[:, 3:6].astype(np.float64)
    v_flu_gt, t_mids, t_starts = extract_velocity_labels(pos_root, R_X_fw, P, fps)

    t_bounds = np.append(t_starts, t_starts[-1] + Q)
    pos_gt   = pos_root[t_bounds].astype(np.float64)
    t_sec    = t_bounds / fps

    R_gt_wfx   = R_X_fw.transpose(0, 2, 1).astype(np.float64)
    pos_gtR    = integrate_trajectory(v_flu_gt, t_starts, t_mids, R_gt_wfx,   pos_root[0], fps, Q)

    R_gyro_wfx = integrate_gyro(gyro_flu, R_X_fw[0].T, fps)
    pos_gyro   = integrate_trajectory(v_flu_gt, t_starts, t_mids, R_gyro_wfx, pos_root[0], fps, Q)

    # GT vel + real IMU wrist orientation
    pos_realori = integrate_trajectory(v_flu_gt, t_starts, t_mids, R_realori_wfx, pos_root[0], fps, Q)

    err_gtR     = np.linalg.norm(pos_gt - pos_gtR,     axis=1)
    err_gyro    = np.linalg.norm(pos_gt - pos_gyro,    axis=1)
    err_realori = np.linalg.norm(pos_gt - pos_realori, axis=1)
    err_gyro_h    = np.linalg.norm(pos_gt[:, :2] - pos_gyro[:, :2],    axis=1)
    err_realori_h = np.linalg.norm(pos_gt[:, :2] - pos_realori[:, :2], axis=1)

    print(f"GT-R      label check     : 3D MAE={err_gtR.mean():.7f} m  max={err_gtR.max():.7f} m")
    print(f"GT vel + real wrist ori   : 3D MAE={err_realori.mean():.4f} m  max={err_realori.max():.4f} m  Horiz MAE={err_realori_h.mean():.4f} m")
    print(f"Gyro-R integration        : 3D MAE={err_gyro.mean():.4f} m  max={err_gyro.max():.4f} m  Horiz MAE={err_gyro_h.mean():.4f} m")

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"FLU-velocity labels + gyro integration — {os.path.basename(args.pkl)}\n"
        f"patch={P} fr @ {fps:.0f} Hz={P/fps:.2f}s, 4 sub-patches x {Q/fps:.2f}s each",
        fontsize=11)
    kw_gt      = dict(color="green",  lw=1.8, label="GT")
    kw_gtR     = dict(color="blue",   lw=1.2, ls="--",  label=f"GT vel + FK ori    MAE={err_gtR.mean():.6f}m (label check)")
    kw_realori = dict(color="orange", lw=1.2, ls=":",   label=f"GT vel + real wrist ori  MAE={err_realori.mean():.4f}m")
    kw_gyro    = dict(color="red",    lw=1.2, ls="-.",  label=f"GT vel + gyro ori  MAE={err_gyro.mean():.4f}m (gyro drift)")

    ax1 = fig.add_subplot(2, 2, 1, aspect="equal")
    ax1.plot(pos_gt[:,0],      pos_gt[:,1],      **kw_gt)
    ax1.plot(pos_gtR[:,0],     pos_gtR[:,1],     **kw_gtR)
    ax1.plot(pos_realori[:,0], pos_realori[:,1], **kw_realori)
    ax1.plot(pos_gyro[:,0],    pos_gyro[:,1],    **kw_gyro)
    ax1.plot(pos_gt[0,0], pos_gt[0,1], "g^", ms=9, zorder=5, label="Start")
    ax1.set(xlabel="World X (m)", ylabel="World Y (m)", title="Top-down (Z-up world)")
    ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t_sec, pos_gt[:,2],      **kw_gt)
    ax2.plot(t_sec, pos_gtR[:,2],     **kw_gtR)
    ax2.plot(t_sec, pos_realori[:,2], **kw_realori)
    ax2.plot(t_sec, pos_gyro[:,2],    **kw_gyro)
    ax2.set(xlabel="Time (s)", ylabel="World Z (m)", title="Height over time")
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t_sec, err_gtR,     color="blue",   lw=1.2, label="GT vel + FK ori   (≈0, label check)")
    ax3.plot(t_sec, err_realori, color="orange", lw=1.2, label="GT vel + real wrist ori")
    ax3.plot(t_sec, err_gyro,    color="red",    lw=1.2, label="GT vel + gyro ori")
    ax3.set(xlabel="Time (s)", ylabel="Error (m)", title="Cumulative 3D position error")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    v_mag = np.linalg.norm(v_flu_gt, axis=1)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.bar(t_mids / fps, v_mag, width=Q/fps*0.8, color="purple", alpha=0.7)
    ax4.set(xlabel="Time (s)", ylabel="Speed (m/s)", title="GT pelvis speed in wrist FLU frame")
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
