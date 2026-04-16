"""Visualize a Nymeria sequence in MeshCat with the FLU wrist coordinate frame.

The body mesh is driven by SMPL FK from the Nymeria xsens_pose data.
On top of the right-wrist joint, three colored line segments are drawn per frame
to show the FLU (X) coordinate defined in nymeria_wrist_loader_FLU.py:

    F  (forward, FLU x)  →  RED   = -x_smpl in wrist-local T-pose
    L  (left,    FLU y)  →  GREEN = +z_smpl in wrist-local T-pose
    U  (up,      FLU z)  →  BLUE  = +y_smpl in wrist-local T-pose

Usage
-----
    python scripts/vis_wrist_flu.py \\
        --pkl  /mnt/nas2/processed_data_test_body_Ty/some_seq.pkl \\
        [--fps 50]            \\
        [--max-frames 2000]   \\
        [--axis-len 0.12]     # length of each FLU axis arrow in metres
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from time import sleep

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# T-pose FLU definition (matches nymeria_wrist_loader_FLU.py)
# ---------------------------------------------------------------------------
# Forward (FLU x) = -x_smpl in wrist-local frame
# Left    (FLU y) = +z_smpl
# Up      (FLU z) = +y_smpl
#
# Columns of R_WRIST_FROM_FLU are the FLU unit axes expressed in
# wrist-local SMPL coordinates.
R_WRIST_FROM_FLU = np.array(
    [[-1.0, 0.0, 0.0],   # F column
     [ 0.0, 0.0, 1.0],   # L column
     [ 0.0, 1.0, 0.0]],  # U column
    dtype=np.float64,
)

# Sanity-check
assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)

# SMPL right wrist joint index
_SMPL_RIGHT_WRIST = 21

# Xsens → SMPL joint mapping
_XSENS_TO_SMPL = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18,
                   5, 11,  7, 6, 12,  8, 13,  9, 13,  9, 13,  9]


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def xsens_to_SMPL_rotmat(xsens_rotmat: np.ndarray) -> np.ndarray:
    """Xsens global rotations [N,23,3,3] → SMPL global [N,24,3,3]."""
    perm = [1, 2, 0]
    xr = xsens_rotmat[:, :, perm, :][:, :, :, perm]
    xt = torch.from_numpy(xr).float()

    Rx90 = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    Rz90 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    xt = Rz90 @ (Rx90 @ xt)

    smpl = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(xt.shape[0], 24, -1, -1).clone()
    for smpl_idx, xsens_idx in enumerate(_XSENS_TO_SMPL):
        smpl[:, smpl_idx] = xt[:, xsens_idx]
    return smpl.numpy()


# ---------------------------------------------------------------------------
# MeshCat helper: draw a single line segment as a PointsGeometry Line
# ---------------------------------------------------------------------------

def _make_axis_line(vis, path: str, color_hex: int):
    """Pre-register a Line object under ``path``; update vertices to animate."""
    import meshcat.geometry as mcg
    pts = np.zeros((3, 2), dtype=np.float32)   # two endpoints
    vis[path].set_object(
        mcg.Line(
            mcg.PointsGeometry(pts),
            mcg.LineBasicMaterial(color=color_hex, linewidth=5),
        )
    )


def _update_axis(vis, path: str, origin: np.ndarray, direction: np.ndarray, length: float, color_hex: int):
    """Push updated endpoints for a single axis line."""
    import meshcat.geometry as mcg
    tip = origin + direction * length
    pts = np.stack([origin, tip], axis=1).astype(np.float32)  # (3,2)
    vis[path].set_object(
        mcg.Line(
            mcg.PointsGeometry(pts),
            mcg.LineBasicMaterial(color=color_hex, linewidth=5),
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    ap = argparse.ArgumentParser(
        description="Visualize a Nymeria sequence + wrist FLU frame in MeshCat"
    )
    ap.add_argument("--pkl",         required=True, help="Path to Nymeria .pkl file")
    ap.add_argument("--fps",         type=int,   default=50)
    ap.add_argument("--max-frames",  type=int,   default=0, help="0 = full sequence")
    ap.add_argument("--axis-len",    type=float, default=0.25,
                    help="Length of each FLU axis arrow in metres")
    args = ap.parse_args()

    # ------------------------------------------------------------------ data
    logger.info("Loading: %s", os.path.basename(args.pkl))
    with open(args.pkl, "rb") as fh:
        raw = pickle.load(fh)

    xsens_pose = raw["gt_data"]["xsens_pose"]
    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float32)   # [N, 23, 4, 4]

    N = xsens_pose.shape[0]
    if args.max_frames > 0:
        N = min(N, args.max_frames)
        xsens_pose = xsens_pose[:N]

    logger.info("Sequence length: %d frames  (%.1f s)", N, N / args.fps)

    # ------------------------------------------------------------------ SMPL
    import core.articulate as art
    from core.paths import Paths

    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))

    xsens_R      = xsens_pose[:, :, :3, :3]          # [N,23,3,3]
    smpl_global  = xsens_to_SMPL_rotmat(xsens_R)      # [N,24,3,3]
    smpl_global_t = torch.from_numpy(smpl_global).float()

    # IK: global → local rotations
    smpl_local_t = body_model.inverse_kinematics_R(smpl_global_t)  # [N,24,3,3]

    # Root translation from Xsens joint 0
    root_trans = torch.from_numpy(xsens_pose[:, 0, :3, 3]).float()  # [N,3]

    # FK: local → world joint rotations + joint positions + mesh vertices
    logger.info("Running SMPL FK (mesh) ...")
    pose_global_t, joint_world_t, verts_t = body_model.forward_kinematics_masked(
        smpl_local_t,
        tran=root_trans,
        calc_mesh=True,
    )  # [N,24,3,3], [N,24,3], [N,V,3]

    verts_np = verts_t.detach().cpu().numpy().astype(np.float32)          # [N,V,3]
    R_world_from_wrist = pose_global_t[:, _SMPL_RIGHT_WRIST].cpu().numpy().astype(np.float64)  # [N,3,3]
    p_world_wrist      = joint_world_t[:, _SMPL_RIGHT_WRIST].cpu().numpy().astype(np.float64)  # [N,3]

    # ------------------------------------------------------------------
    # FLU axes in world frame per timestep
    #
    #   R_world_from_FLU(t) = R_world_from_wrist(t) @ R_WRIST_FROM_FLU
    #
    # The three columns are the F, L, U unit vectors in world frame.
    # ------------------------------------------------------------------
    R_world_from_flu = np.einsum("nij,jk->nik",
                                  R_world_from_wrist,
                                  R_WRIST_FROM_FLU)   # [N,3,3]
    # columns: F=[:,0], L=[:,1], U=[:,2]

    faces = body_model.face.astype(np.uint32)

    # ------------------------------------------------------------------ MeshCat
    import meshcat
    import meshcat.geometry as mcg
    import meshcat.transformations as mct

    vis = meshcat.Visualizer()
    print("\nMeshCat at:", vis.url())
    print("Open the URL above in a browser, then wait for animation to start.\n")

    vis["/Background"].set_property("top_color",    [1, 1, 1])
    vis["/Background"].set_property("bottom_color", [1, 1, 1])
    vis["/Grid"].set_transform(mct.translation_matrix([0, 0, -1]))

    ROOT = "scene"
    try:
        vis[ROOT].delete()
    except Exception:
        pass

    # Nymeria data is Y-up; rotate the scene so it looks correct in MeshCat (Z-up)
    vis[ROOT].set_transform(mct.rotation_matrix(np.pi / 2.0, [1, 0, 0]))

    # ---- body mesh ----
    mesh_geom = mcg.TriangularMeshGeometry(
        verts_np[0],
        faces,
    )
    mesh_mat = mcg.MeshLambertMaterial(color=0x6699FF, opacity=0.90)
    vis[f"{ROOT}/body"].set_object(mesh_geom, mesh_mat)

    # ---- FLU axis lines (pre-register, updated each frame) ----
    #  F = red   L = green   U = blue
    AXIS_COLORS = [0xFF2222, 0x22CC22, 0x2255FF]
    AXIS_NAMES  = ["F", "L", "U"]
    for name, color in zip(AXIS_NAMES, AXIS_COLORS):
        pts = np.zeros((3, 2), dtype=np.float32)
        vis[f"{ROOT}/flu/{name}"].set_object(
            mcg.Line(
                mcg.PointsGeometry(pts),
                mcg.LineBasicMaterial(color=color, linewidth=12),
            )
        )

    # ---- wrist origin sphere ----
    vis[f"{ROOT}/wrist_origin"].set_object(
        mcg.Sphere(0.012),
        mcg.MeshLambertMaterial(color=0xFFFF00),
    )

    dt = 1.0 / args.fps
    axis_len = float(args.axis_len)

    logger.info("Animating %d frames at %.0f fps ...", N, args.fps)
    for f in range(N):
        # Body mesh
        mesh_geom.vertices = verts_np[f]
        vis[f"{ROOT}/body"].set_object(mesh_geom, mesh_mat)

        # Wrist origin
        vis[f"{ROOT}/wrist_origin"].set_transform(
            mct.translation_matrix(p_world_wrist[f])
        )

        # FLU axes: each column of R_world_from_flu[f] is a unit axis in world frame
        for ci, (name, color) in enumerate(zip(AXIS_NAMES, AXIS_COLORS)):
            origin    = p_world_wrist[f]           # [3]
            direction = R_world_from_flu[f, :, ci] # [3] — ci-th column
            tip = origin + direction * axis_len
            pts = np.stack([origin, tip], axis=1).astype(np.float32)  # (3,2)
            vis[f"{ROOT}/flu/{name}"].set_object(
                mcg.Line(
                    mcg.PointsGeometry(pts),
                    mcg.LineBasicMaterial(color=color, linewidth=12),
                )
            )

        sleep(dt)

    logger.info("Done.")


if __name__ == "__main__":
    main()
