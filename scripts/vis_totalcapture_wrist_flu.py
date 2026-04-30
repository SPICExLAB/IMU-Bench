"""Visualize a TotalCapture (DIP-format) session in MeshCat with the FLU wrist frame.

The body mesh is driven by SMPL FK from the ``gt`` axis-angle rotations in the pkl.
On top of the right-wrist IMU sensor origin, three colored arrows show the FLU frame
defined in ``totalcapture_wrist_loader_FLU.py``:

    F  (forward / finger direction)  →  RED   = −x_smpl
    L  (left    / thumb side)        →  GREEN = +z_smpl
    U  (up      / watch-face dorsal) →  BLUE  = +y_smpl

An optional matplotlib panel shows the derived acc_flu and gyro_flu signals
for the first ``--plot-seconds`` seconds of the session.

Usage
-----
    python scripts/vis_totalcapture_wrist_flu.py \\
        --pkl  /mnt/nas2/DIP_recalculate/s1_acting1.pkl \\
        [--fps 60]                  \\
        [--max-frames 0]            \\  # 0 = full sequence
        [--axis-len 0.12]           \\
        [--add-gravity]             \\  # flag: simulate real MEMS acc
        [--plot-seconds 10]         \\  # 0 = skip signal plot
        [--out-dir data_vis/totalcapture_flu]
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from time import sleep

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants  (identical to totalcapture_wrist_loader_FLU.py)
# ---------------------------------------------------------------------------
_SENSOR_REORDER  = [2, 3, 0, 1, 4, 5]
_RIGHT_WRIST_IDX = 0   # after reordering

_G_WORLD = np.array([0.0, -9.81, 0.0], dtype=np.float64)   # Y-up gravity

# FLU columns expressed in wrist SMPL-local frame
R_WRIST_FROM_FLU = np.array(
    [[-1.0, 0.0, 0.0],   # F
     [ 0.0, 0.0, 1.0],   # L
     [ 0.0, 1.0, 0.0]],  # U
    dtype=np.float64,
)
assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)

_SMPL_RIGHT_WRIST = 21   # SMPL joint index for right wrist

AXIS_COLORS = [0xFF2222, 0x22CC22, 0x2255FF]   # F=red, L=green, U=blue
AXIS_NAMES  = ["F", "L", "U"]

# SMPL kinematic tree (index = joint, value = parent, -1 = root)
_SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
]


def _smpl_global_rotations(R_local: np.ndarray) -> np.ndarray:
    """R_local [N, 24, 3, 3] (SMPL local) → R_global [N, 24, 3, 3] (world)."""
    R_global = np.empty_like(R_local)
    R_global[:, 0] = R_local[:, 0]
    for j in range(1, 24):
        p = _SMPL_PARENTS[j]
        R_global[:, j] = np.einsum("nij,njk->nik", R_global[:, p], R_local[:, j])
    return R_global


# ---------------------------------------------------------------------------
# Load & derive FLU IMU  (mirrors load_session_flu)
# ---------------------------------------------------------------------------

def _load_pkl(pkl_path: str, fps_raw: float = 60.0, target_hz: float = 60.0,
              add_gravity: bool = True, use_gt_ori: bool = False, max_frames: int = 0):
    """Load TotalCapture DIP pkl; return acc_flu [N,3], gyro_flu [N,3],
    R_world_from_flu [N,3,3], gt_aa [N,24,3] and effective fps.

    use_gt_ori=True  : derive wrist world orientation from GT SMPL FK (cleaner).
    use_gt_ori=False : use ori IMU-fusion field as-is (matches real deployment).
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh, encoding="latin1")

    acc_raw = np.asarray(data["acc"], dtype=np.float64)   # [N, 6, 3]
    ori_raw = np.asarray(data["ori"], dtype=np.float64)   # [N, 6, 3, 3]
    gt_raw  = np.asarray(data["gt"],  dtype=np.float64)   # [N, 72]

    N = min(acc_raw.shape[0], ori_raw.shape[0], gt_raw.shape[0])
    if max_frames > 0:
        N = min(N, max_frames)
    acc_raw = acc_raw[:N];  ori_raw = ori_raw[:N];  gt_raw = gt_raw[:N]

    # Resample if needed
    if abs(fps_raw - target_hz) > 1e-3:
        new_len = int(round((N - 1) * target_hz / fps_raw)) + 1

        def _lin(x):
            old = np.linspace(0, 1, x.shape[0])
            new = np.linspace(0, 1, new_len)
            flat = x.reshape(x.shape[0], -1)
            return np.stack([np.interp(new, old, flat[:, c]) for c in range(flat.shape[1])],
                            axis=1).reshape(new_len, *x.shape[1:])

        def _nn_rot(x):
            idx = np.round(np.linspace(0, x.shape[0] - 1, new_len)).astype(np.int64)
            return x[idx]

        acc_raw = _lin(acc_raw)
        ori_raw = _nn_rot(ori_raw)
        gt_raw  = _lin(gt_raw)
        N       = new_len
        logger.info("Resampled → %d frames (%.0f→%.0f Hz)", N, fps_raw, target_hz)

    fps = float(target_hz)

    # Select right-wrist sensor
    acc_all = acc_raw[:, _SENSOR_REORDER]   # [N, 6, 3]
    ori_all = ori_raw[:, _SENSOR_REORDER]   # [N, 6, 3, 3]
    acc_world_raw  = acc_all[:, _RIGHT_WRIST_IDX]   # [N, 3]  kinematic, Y-up world

    if use_gt_ori:
        # GT SMPL FK: axis-angle → local rotmats → global chain → joint 21
        gt_aa_early = gt_raw.reshape(N, 24, 3)
        from scipy.spatial.transform import Rotation as _ScipyRotLocal
        R_local_gt  = (_ScipyRotLocal.from_rotvec(gt_aa_early.reshape(-1, 3))
                       .as_matrix().reshape(N, 24, 3, 3).astype(np.float64))
        R_world_from_s = _smpl_global_rotations(R_local_gt)[:, _SMPL_RIGHT_WRIST]  # [N,3,3]
        logger.info("Using GT SMPL FK orientation for FLU frame")
    else:
        R_world_from_s = ori_all[:, _RIGHT_WRIST_IDX]   # [N, 3, 3]  actual sensor frame

    # Gravity
    acc_world = acc_world_raw.copy()
    if add_gravity:
        acc_world = acc_world - _G_WORLD

    # FLU frame
    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_s, R_WRIST_FROM_FLU)  # [N,3,3]
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)

    acc_flu = np.einsum("nij,nj->ni", R_X_from_world, acc_world)   # [N, 3]

    # Gyro
    dR = np.einsum("nij,njk->nik",
                   R_world_from_s[:-1].transpose(0, 2, 1),
                   R_world_from_s[1:])                             # [N-1, 3, 3]
    omega_local = (ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
                   .as_rotvec().astype(np.float64) * fps)           # [N-1, 3]
    omega_local_full = np.zeros((N, 3), dtype=np.float64)
    omega_local_full[:-1] = omega_local
    omega_local_full[-1]  = omega_local[-1]
    omega_world = np.einsum("nij,nj->ni", R_world_from_s, omega_local_full)
    gyro_flu    = np.einsum("nij,nj->ni", R_X_from_world, omega_world)   # [N, 3]

    gt_aa = gt_raw.reshape(N, 24, 3)   # [N, 24, 3] axis-angle

    return acc_flu, gyro_flu, R_world_from_X, gt_aa, fps


# ---------------------------------------------------------------------------
# SMPL FK
# ---------------------------------------------------------------------------

def _smpl_fk(gt_aa: np.ndarray, device="cpu"):
    """gt_aa [N,24,3] axis-angle → verts [N,V,3], joints [N,24,3], pose_global [N,24,3,3]."""
    from scipy.spatial.transform import Rotation as ScipyRot
    import core.articulate as art
    from core.paths import Paths

    N = gt_aa.shape[0]
    body = art.model.ParametricModel(str(Paths.SMPL_FILE), device=torch.device(device))

    # axis-angle → local rotation matrices
    R_local = (ScipyRot.from_rotvec(gt_aa.reshape(-1, 3))
               .as_matrix()
               .reshape(N, 24, 3, 3).astype(np.float32))
    R_local_t = torch.from_numpy(R_local).float()

    logger.info("Running SMPL FK ...")
    pose_global_t, joints_t, verts_t = body.forward_kinematics_masked(
        R_local_t, tran=None, calc_mesh=True,
    )

    verts  = verts_t.detach().cpu().numpy().astype(np.float32)          # [N, V, 3]
    joints = joints_t.detach().cpu().numpy().astype(np.float64)         # [N, 24, 3]
    R_wrist = pose_global_t[:, _SMPL_RIGHT_WRIST].cpu().numpy().astype(np.float64)  # [N,3,3]
    faces  = body.face.astype(np.uint32)
    return verts, joints, R_wrist, faces


# ---------------------------------------------------------------------------
# Signal plot
# ---------------------------------------------------------------------------

def _plot_signals(acc_flu: np.ndarray, gyro_flu: np.ndarray, fps: float,
                  plot_seconds: float, out_dir: str, pkl_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(acc_flu), int(plot_seconds * fps))
    t = np.arange(n) / fps

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ch_labels = ["F", "L", "U"]
    colors     = ["#d62728", "#2ca02c", "#1f77b4"]

    for ci in range(3):
        axes[0].plot(t, acc_flu[:n, ci],  color=colors[ci], label=ch_labels[ci], linewidth=0.8)
        axes[1].plot(t, gyro_flu[:n, ci], color=colors[ci], label=ch_labels[ci], linewidth=0.8)

    axes[0].set_ylabel("Acc (m/s²)")
    axes[0].set_title(f"{pkl_name} — FLU wrist IMU")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].axhline(0, color="grey", linewidth=0.4)

    axes[1].set_ylabel("Gyro (rad/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].axhline(0, color="grey", linewidth=0.4)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pkl_name}_flu_signals.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Signal plot saved: %s", out_path)


# ---------------------------------------------------------------------------
# T-pose visualization
# ---------------------------------------------------------------------------

def _show_tpose(axis_len: float):
    """Render a static SMPL T-pose with the FLU frame arrows at the right wrist.

    In T-pose all SMPL local rotations are identity, so:
        R_world_from_wrist = I  →  R_world_from_FLU = R_WRIST_FROM_FLU
    The three columns of R_WRIST_FROM_FLU give the F/L/U directions directly
    in SMPL world space (Y-up).
    """
    import meshcat
    import meshcat.geometry as mcg
    import meshcat.transformations as mct

    logger.info("Computing SMPL T-pose ...")
    # All local rotations = identity → T-pose
    gt_aa_tpose = np.zeros((1, 24, 3), dtype=np.float32)
    verts, joints, _, faces = _smpl_fk(gt_aa_tpose)

    p_wrist = joints[0, _SMPL_RIGHT_WRIST]   # [3] wrist origin in Y-up world

    vis = meshcat.Visualizer()
    print("\nMeshCat at:", vis.url())
    print("Open the URL above in a browser to see the T-pose.")
    print("  RED   = F (forward / finger direction = −x_smpl)")
    print("  GREEN = L (left / thumb side          = +z_smpl)")
    print("  BLUE  = U (up / watch-face dorsal     = +y_smpl)\n")

    vis["/Background"].set_property("top_color",    [1, 1, 1])
    vis["/Background"].set_property("bottom_color", [1, 1, 1])
    vis["/Grid"].set_transform(mct.translation_matrix([0, 0, -1]))

    ROOT = "scene"
    try:
        vis[ROOT].delete()
    except Exception:
        pass

    # Y-up → Z-up for MeshCat
    vis[ROOT].set_transform(mct.rotation_matrix(np.pi / 2.0, [1, 0, 0]))

    # Body mesh
    vis[f"{ROOT}/body"].set_object(
        mcg.TriangularMeshGeometry(verts[0], faces),
        mcg.MeshLambertMaterial(color=0x6699FF, opacity=0.90),
    )

    # Wrist origin sphere
    vis[f"{ROOT}/wrist_origin"].set_object(
        mcg.Sphere(0.012),
        mcg.MeshLambertMaterial(color=0xFFFF00),
    )
    vis[f"{ROOT}/wrist_origin"].set_transform(mct.translation_matrix(p_wrist.tolist()))

    # FLU axes — in T-pose R_world_from_wrist=I, so columns of R_WRIST_FROM_FLU
    # give axis directions directly in world space.
    for ci, (name, color) in enumerate(zip(AXIS_NAMES, AXIS_COLORS)):
        direction = R_WRIST_FROM_FLU[:, ci]          # column ci = FLU basis vector in wrist frame = world frame in T-pose
        tip = p_wrist + direction * axis_len
        pts = np.stack([p_wrist, tip], axis=1).astype(np.float32)   # (3, 2)
        vis[f"{ROOT}/flu/{name}"].set_object(
            mcg.Line(
                mcg.PointsGeometry(pts),
                mcg.LineBasicMaterial(color=color, linewidth=12),
            )
        )

    input("Press Enter to exit ...")


# ---------------------------------------------------------------------------
# MeshCat animation
# ---------------------------------------------------------------------------

def _animate(verts, joints, R_world_from_flu_sensor, faces, fps, axis_len):
    """Run MeshCat animation loop."""
    import meshcat
    import meshcat.geometry as mcg
    import meshcat.transformations as mct

    N = verts.shape[0]

    vis = meshcat.Visualizer()
    print("\nMeshCat at:", vis.url())
    print("Open the URL above in a browser, then wait for animation.\n")

    vis["/Background"].set_property("top_color",    [1, 1, 1])
    vis["/Background"].set_property("bottom_color", [1, 1, 1])
    vis["/Grid"].set_transform(mct.translation_matrix([0, 0, -1]))

    ROOT = "scene"
    try:
        vis[ROOT].delete()
    except Exception:
        pass

    # TotalCapture / SMPL is Y-up; rotate scene so it looks right in MeshCat (Z-up)
    vis[ROOT].set_transform(mct.rotation_matrix(np.pi / 2.0, [1, 0, 0]))

    # Body mesh
    mesh_geom = mcg.TriangularMeshGeometry(verts[0], faces)
    mesh_mat  = mcg.MeshLambertMaterial(color=0x6699FF, opacity=0.90)
    vis[f"{ROOT}/body"].set_object(mesh_geom, mesh_mat)

    # Pre-register FLU axis lines
    for name, color in zip(AXIS_NAMES, AXIS_COLORS):
        pts = np.zeros((3, 2), dtype=np.float32)
        vis[f"{ROOT}/flu/{name}"].set_object(
            mcg.Line(
                mcg.PointsGeometry(pts),
                mcg.LineBasicMaterial(color=color, linewidth=12),
            )
        )

    # Wrist origin sphere
    vis[f"{ROOT}/wrist_origin"].set_object(
        mcg.Sphere(0.012),
        mcg.MeshLambertMaterial(color=0xFFFF00),
    )

    dt = 1.0 / fps
    logger.info("Animating %d frames at %.0f fps ...", N, fps)

    for f in range(N):
        # Body mesh
        mesh_geom.vertices = verts[f]
        vis[f"{ROOT}/body"].set_object(mesh_geom, mesh_mat)

        # Wrist origin (from FK joint positions)
        p_wrist = joints[f, _SMPL_RIGHT_WRIST]
        vis[f"{ROOT}/wrist_origin"].set_transform(
            mct.translation_matrix(p_wrist)
        )

        # FLU axes: columns of R_world_from_flu (built from actual sensor ori)
        for ci, (name, color) in enumerate(zip(AXIS_NAMES, AXIS_COLORS)):
            direction = R_world_from_flu_sensor[f, :, ci]   # [3]
            tip = p_wrist + direction * axis_len
            pts = np.stack([p_wrist, tip], axis=1).astype(np.float32)   # (3,2)
            vis[f"{ROOT}/flu/{name}"].set_object(
                mcg.Line(
                    mcg.PointsGeometry(pts),
                    mcg.LineBasicMaterial(color=color, linewidth=12),
                )
            )

        sleep(dt)

    logger.info("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    ap = argparse.ArgumentParser(
        description="Visualize TotalCapture session with FLU wrist frame in MeshCat"
    )
    ap.add_argument("--pkl",          default="",
                    help="Path to DIP-format TotalCapture .pkl file (not needed with --tpose)")
    ap.add_argument("--fps",          type=float, default=60.0,
                    help="Target FPS (raw TotalCapture is 60 Hz)")
    ap.add_argument("--max-frames",   type=int,   default=0,
                    help="Max frames to animate (0 = full sequence)")
    ap.add_argument("--axis-len",     type=float, default=0.55,
                    help="Length of each FLU axis arrow in metres")
    ap.add_argument("--add-gravity",  action="store_true", default=True,
                    help="Simulate real MEMS accelerometer (add gravity to acc_world)")
    ap.add_argument("--no-gravity",   dest="add_gravity", action="store_false",
                    help="Keep gravity-subtracted kinematic acceleration")
    ap.add_argument("--use-gt-ori",   action="store_true", default=True,
                    help="Derive FLU frame from GT SMPL FK (joint 21) instead of ori field")
    ap.add_argument("--plot-seconds", type=float, default=10.0,
                    help="Seconds of signal to plot (0 = skip)")
    ap.add_argument("--out-dir",      default="data_vis/totalcapture_flu",
                    help="Directory for saving signal plots")
    ap.add_argument("--no-vis",       action="store_true",
                    help="Skip MeshCat 3-D animation (signal plot only)")
    ap.add_argument("--tpose",        action="store_true",
                    help="Show a static SMPL T-pose with FLU axes (no pkl required)")
    args = ap.parse_args()

    # T-pose mode — no pkl needed
    if args.tpose:
        _show_tpose(args.axis_len)
        return

    pkl_name = os.path.splitext(os.path.basename(args.pkl))[0]
    logger.info("Loading: %s  (add_gravity=%s)", pkl_name, args.add_gravity)

    acc_flu, gyro_flu, R_world_from_flu_sensor, gt_aa, fps = _load_pkl(
        pkl_path    = args.pkl,
        fps_raw     = 60.0,
        target_hz   = args.fps,
        add_gravity = args.add_gravity,
        use_gt_ori  = args.use_gt_ori,
        max_frames  = args.max_frames,
    )
    N = acc_flu.shape[0]
    logger.info("Loaded %d frames (%.1f s @ %.0f Hz)", N, N / fps, fps)
    logger.info("acc_flu  range: [%.2f, %.2f] m/s²",
                acc_flu.min(), acc_flu.max())
    logger.info("gyro_flu range: [%.2f, %.2f] rad/s",
                gyro_flu.min(), gyro_flu.max())

    # Signal plot
    if args.plot_seconds > 0:
        _plot_signals(acc_flu, gyro_flu, fps, args.plot_seconds, args.out_dir, pkl_name)

    # 3-D visualization
    if not args.no_vis:
        verts, joints, _, faces = _smpl_fk(gt_aa)
        _animate(verts, joints, R_world_from_flu_sensor, faces, fps, args.axis_len)


if __name__ == "__main__":
    main()
