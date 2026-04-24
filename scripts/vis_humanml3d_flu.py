"""Visualize a HumanML3D clip in MeshCat: SMPL body mesh + FLU wrist frame.

Coordinate systems
------------------
HumanML3D joints (joints/*.npy) are in a canonical world frame:
    Y-up, person standing on Y=0 floor, frame-0 pelvis at XZ origin, faces +Z.

Geometric arm-chain frame (same as humanml3d_dataset_FLU.py):
    col 0  x_geom = normalize(z_geom × upper_arm)  — lateral, thumb side
    col 1  y_geom = normalize(z_geom × x_geom)     — dorsal,  watch face
    col 2  z_geom = normalize(wrist − elbow)        — fingertips direction

FLU mapping (R_WRIST_FROM_FLU = same matrix as humanml3d_dataset_FLU.py):
    F  (forward / fingertips)  RED   = +z_geom
    L  (left    / thumb side)  GREEN = +x_geom
    U  (up      / watch face)  BLUE  = +y_geom

SMPL mesh pipeline:
    1. Load AMASS .npz found via HumanML3D/index.csv
    2. Resample 100 Hz → 20 Hz, slice to clip range
    3. SMPL FK with NO root translation (root stays at origin)
    4. Rotate mesh: same Y-rotation as HumanML3D canonicalization (face +Z)
    5. Translate each frame by pelvis from joints.npy → mesh follows trajectory

FLU orientation uses SMPL FK wrist rotation. The FLU origin is anchored to the
canonicalized FK right-wrist joint so the axes stay fixed on the rendered mesh.

Usage
-----
    python scripts/vis_humanml3d_flu.py \\
        --npy /path/to/HumanML3D/joints/000021.npy \\
        [--humanml3d-root /path/to/HumanML3D]  \\
        [--fps 20]  [--max-frames 0]  [--axis-len 0.25]  [--loop]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from time import sleep

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SMPL joint indices (first 22, same as humanml3d_dataset.py)
# ---------------------------------------------------------------------------
_R_SHOULDER = 17
_R_ELBOW    = 19
_R_WRIST    = 21

# ---------------------------------------------------------------------------
# FLU frame — SMPL FK joint frame (same as nymeria_wrist_loader_FLU.py
#              and the updated humanml3d_dataset_FLU.py)
# ---------------------------------------------------------------------------
# R_world_from_wrist comes from SMPL FK pose_global[:, 21].
# In SMPL T-pose:
#   x_smpl = proximal (toward elbow)
#   y_smpl = dorsal   (out of watch face)
#   z_smpl = thumb side (lateral)
#
# FLU mapping:
#   F (fingertips) = −x_smpl
#   L (thumb side) = +z_smpl
#   U (watch face) = +y_smpl
R_WRIST_FROM_FLU: np.ndarray = np.array(
    [[-1.0,  0.0,  0.0],   # F
     [ 0.0,  0.0,  1.0],   # L
     [ 0.0,  1.0,  0.0]],  # U
    dtype=np.float64,
)
assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)

AXIS_NAMES  = ["F", "L", "U"]
AXIS_COLORS = [0xFF2222, 0x22CC22, 0x2255FF]   # red, green, blue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_wrist_rotation(shoulder, elbow, wrist):
    """R_world_from_wrist [N,3,3] from arm-chain joint positions.
    Columns: x_geom (lateral), y_geom (dorsal), z_geom (fingertips).
    Identical to humanml3d_dataset_FLU.py._build_wrist_rotation.
    """
    eps = 1e-8
    z = wrist - elbow
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + eps)
    upper = elbow - shoulder
    upper = upper / (np.linalg.norm(upper, axis=-1, keepdims=True) + eps)
    x = np.cross(z, upper)
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y, axis=-1, keepdims=True) + eps)
    return np.stack([x, y, z], axis=-1).astype(np.float64)   # [N, 3, 3]


def _rot_a_to_b(a, b):
    """Rotation matrix mapping unit vector a → b (Rodrigues)."""
    a = a / np.linalg.norm(a);  b = b / np.linalg.norm(b)
    v = np.cross(a, b);  s = np.linalg.norm(v);  c = float(a @ b)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 - c) / (s * s)


def _find_clip(npy_path, root):
    """Return (amass_npz_path, start_frame, end_frame) for a joints/*.npy."""
    clip = os.path.basename(npy_path)
    with open(os.path.join(root, "index.csv")) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            src, s, e, name = parts
            try:
                s, e = int(s), int(e)
            except ValueError:
                continue
            if name == clip:
                pose_npy  = os.path.join(root, src.lstrip("./"))
                amass_npz = (pose_npy
                             .replace("/pose_data/", "/amass_data/")
                             .replace(".npy", ".npz"))
                return amass_npz, s, e
    raise FileNotFoundError(f"Clip {clip} not in {root}/index.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(
        description="HumanML3D: SMPL mesh + FLU wrist frame in MeshCat"
    )
    ap.add_argument("--npy",            required=True,
                    help="Path to HumanML3D joints/*.npy")
    ap.add_argument("--humanml3d-root", default="/home/yiquan/HumanML3D",
                    help="Root of HumanML3D (contains joints/, index.csv, …)")
    ap.add_argument("--fps",            type=int,   default=20)
    ap.add_argument("--max-frames",     type=int,   default=0, help="0 = full clip")
    ap.add_argument("--axis-len",       type=float, default=0.25,
                    help="FLU arrow length in metres")
    ap.add_argument("--loop",           action="store_true")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # 1. Load canonical joint positions from joints.npy
    #    Shape [N, 52, 3]: Y-up, floor Y=0, pelvis at XZ=0, faces +Z, 20 Hz
    # ------------------------------------------------------------------
    jnpy = np.load(args.npy).astype(np.float64)   # [N, 52, 3]
    N    = jnpy.shape[0]
    if args.max_frames > 0:
        N    = min(N, args.max_frames)
        jnpy = jnpy[:N]
    logger.info("joints.npy: %d frames  (%s)", N, os.path.basename(args.npy))

    # ------------------------------------------------------------------
    # 2. Locate AMASS .npz via index.csv
    # ------------------------------------------------------------------
    amass_path, start_f, end_f = _find_clip(args.npy, args.humanml3d_root)
    logger.info("AMASS: %s  [%d:%s]",
                os.path.relpath(amass_path, args.humanml3d_root),
                start_f, end_f if end_f >= 0 else "end")

    d     = np.load(amass_path)
    a_fps = float(d["mocap_framerate"])   # typically 100 Hz
    poses = d["poses"]                    # [T_amass, 156]  axis-angle per joint

    stride  = max(1, int(round(a_fps / 20)))
    poses20 = poses[::stride]
    end_f   = end_f if end_f >= 0 else len(poses20)
    poses_c = poses20[start_f:end_f].astype(np.float32)[:N]   # [N, 156]
    N       = poses_c.shape[0]
    jnpy    = jnpy[:N]
    logger.info("Clip after resample/slice: %d frames", N)

    # ------------------------------------------------------------------
    # 3. SMPL FK — no root translation; root stays at origin.
    #    We'll add the canonical pelvis from jnpy afterwards.
    # ------------------------------------------------------------------
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import core.articulate as art
    from core.paths import Paths
    from core.articulate.math import axis_angle_to_rotation_matrix

    body  = art.model.ParametricModel(str(Paths.SMPL_FILE), device=torch.device("cpu"))
    faces = body.face.astype(np.uint32)

    pose_aa = torch.from_numpy(poses_c[:, :72].reshape(N, 24, 3))
    R_local = axis_angle_to_rotation_matrix(pose_aa.reshape(-1, 3)).reshape(N, 24, 3, 3)

    logger.info("Running SMPL FK …")
    pose_global_t, jnt_t, vrt_t = body.forward_kinematics_masked(
        R_local, tran=None, calc_mesh=True
    )
    raw_j    = jnt_t.detach().cpu().numpy().astype(np.float64)        # [N, 24, 3]
    raw_v    = vrt_t.detach().cpu().numpy().astype(np.float64)        # [N, 6890, 3]
    R_global = pose_global_t.detach().cpu().numpy().astype(np.float64) # [N, 24, 3, 3]

    # ------------------------------------------------------------------
    # 4. Canonicalize:
    #    a) Y-rotation so person faces +Z (matching HumanML3D)
    #    b) Translate each frame so pelvis = jnpy[:, 0, :]
    # ------------------------------------------------------------------
    r_hip  = raw_j[0, 2];  l_hip  = raw_j[0, 1]
    r_sdr  = raw_j[0, 17]; l_sdr  = raw_j[0, 16]
    across  = (r_hip - l_hip) + (r_sdr - l_sdr);  across[1] = 0.
    forward = np.cross(np.array([0., 1., 0.]), across)
    R_y     = _rot_a_to_b(forward, np.array([0., 0., 1.]))

    raw_j = np.einsum("ij,nkj->nki", R_y, raw_j)   # rotate FK joints
    raw_j = raw_j + jnpy[:, 0:1, :]                  # add canonical pelvis

    raw_v = np.einsum("ij,nvj->nvi", R_y, raw_v)   # rotate mesh
    raw_v = raw_v + jnpy[:, 0:1, :]                  # add canonical pelvis

    verts_np = raw_v.astype(np.float32)              # [N, 6890, 3]

    # ------------------------------------------------------------------
    # 5. FLU frame — SMPL FK (same as nymeria_wrist_loader_FLU.py)
    #
    #    R_world_from_wrist(t) = R_y @ pose_global[:, 21, :, :]   (canonical)
    #    R_world_from_FLU(t)   = R_world_from_wrist(t) @ R_WRIST_FROM_FLU
    # ------------------------------------------------------------------
    R_wrist_world = np.einsum("ij,njk->nik", R_y, R_global[:, _R_WRIST])  # [N,3,3]
    R_world_flu   = np.einsum("nij,jk->nik", R_wrist_world, R_WRIST_FROM_FLU)  # [N,3,3]
    wrist_pos     = raw_j[:, _R_WRIST]   # [N, 3] canonical FK wrist position

    # ------------------------------------------------------------------
    # 6. MeshCat streaming (same pattern as vis_wrist_flu.py)
    # ------------------------------------------------------------------
    import meshcat
    import meshcat.geometry as mcg
    import meshcat.transformations as mct

    vis = meshcat.Visualizer()
    print(f"\nMeshCat at: {vis.url()}\n")

    vis["/Background"].set_property("top_color",    [1, 1, 1])
    vis["/Background"].set_property("bottom_color", [1, 1, 1])
    vis["/Grid"].set_transform(mct.translation_matrix([0, 0, -1]))

    ROOT = "scene"
    try:
        vis[ROOT].delete()
    except Exception:
        pass

    # HumanML3D is Y-up; rotate scene so it renders upright in MeshCat (Z-up viewer)
    vis[ROOT].set_transform(mct.rotation_matrix(np.pi / 2.0, [1, 0, 0]))

    # Body mesh
    mesh_geom = mcg.TriangularMeshGeometry(verts_np[0], faces)
    mesh_mat  = mcg.MeshLambertMaterial(color=0x6699FF, opacity=0.90)
    vis[f"{ROOT}/body"].set_object(mesh_geom, mesh_mat)

    # Pre-register FLU axis lines
    for name, color in zip(AXIS_NAMES, AXIS_COLORS):
        vis[f"{ROOT}/flu/{name}"].set_object(
            mcg.Line(
                mcg.PointsGeometry(np.zeros((3, 2), dtype=np.float32)),
                mcg.LineBasicMaterial(color=color, linewidth=12),
            )
        )

    # Wrist origin sphere (yellow)
    vis[f"{ROOT}/wrist_origin"].set_object(
        mcg.Sphere(0.012),
        mcg.MeshLambertMaterial(color=0xFFFF00),
    )

    dt       = 1.0 / args.fps
    axis_len = float(args.axis_len)
    logger.info("Animating %d frames at %d fps …  (Ctrl-C to stop)", N, args.fps)

    def _play_once():
        for f in range(N):
            # Body mesh
            mesh_geom.vertices = verts_np[f]
            vis[f"{ROOT}/body"].set_object(mesh_geom, mesh_mat)

            # Wrist sphere
            vis[f"{ROOT}/wrist_origin"].set_transform(
                mct.translation_matrix(wrist_pos[f])
            )

            # FLU arrows — columns of R_world_flu[f] are world-frame unit axes
            for ci, (name, color) in enumerate(zip(AXIS_NAMES, AXIS_COLORS)):
                tip = wrist_pos[f] + R_world_flu[f, :, ci] * axis_len
                pts = np.stack([wrist_pos[f], tip], axis=1).astype(np.float32)
                vis[f"{ROOT}/flu/{name}"].set_object(
                    mcg.Line(
                        mcg.PointsGeometry(pts),
                        mcg.LineBasicMaterial(color=color, linewidth=12),
                    )
                )

            sleep(dt)

    try:
        while True:
            _play_once()
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\nStopped.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
