r"""
    SMPL/MANO/SMPLH parametric model. Modified from https://github.com/CalciferZh/SMPL.
"""


__all__ = ['ParametricModel']


import os
import pickle
import torch
import tqdm
import numpy as np
from . import math as M


class ParametricModel:
    r"""
    SMPL/MANO/SMPLH parametric model.
    """
    def __init__(self, official_model_file: str, use_pose_blendshape=False, vert_mask=None, device=torch.device('cpu')):
        r"""
        Init an SMPL/MANO/SMPLH parametric model.

        :param official_model_file: Path to the official model to be loaded.
        :param use_pose_blendshape: Whether to use the pose blendshape.
        :param vert_mask: List of selected vertex index. None for all vertices.
        :param device: torch.device, cpu or cuda.
        """
        with open(official_model_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        self._J_regressor = torch.from_numpy(data['J_regressor'].toarray()).float().to(device)
        self._skinning_weights = torch.from_numpy(data['weights']).float().to(device)
        self._posedirs = torch.from_numpy(data['posedirs']).float().to(device)
        self._shapedirs = torch.from_numpy(np.array(data['shapedirs'])).float().to(device)
        self._v_template = torch.from_numpy(data['v_template']).float().to(device)
        self._J = torch.from_numpy(data['J']).float().to(device)
        self.face = data['f']
        self.parent = data['kintree_table'][0].tolist()
        self.parent[0] = None
        self.use_pose_blendshape = use_pose_blendshape
        self.vert_mask = vert_mask

    def save_obj_mesh(self, vertex_position, file_name='a.obj'):
        r"""
        Export an obj mesh using the input vertex position.

        :param vertex_position: Vertex position in shape [num_vertex, 3].
        :param file_name: Output obj file name.
        """
        with open(file_name, 'w') as fp:
            for v in vertex_position:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.face + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    @staticmethod
    def save_unity_motion(pose: torch.Tensor = None, tran: torch.Tensor = None, output_dir='saved_motions/'):
        r"""
        Write motions into text files used by Unity3D `LoadMotion()`.

        :param pose: Pose tensor in shape [num_frames, *] that can reshape to [num_frame, num_joint, 3, 3]
                    (rotation matrices).
        :param tran: Translation tensor in shape [num_frames, 3] for root positions.
        :param output_dir: Output directory path.
        """
        os.makedirs(output_dir, exist_ok=True)

        if pose is not None:
            f = open(os.path.join(output_dir, 'pose.txt'), 'w')
            pose = M.rotation_matrix_to_axis_angle(pose).view(pose.shape[0], -1)
            f.write('\n'.join([','.join(['%.4f' % _ for _ in p]) for p in pose]))
            f.close()

        if tran is not None:
            f = open(os.path.join(output_dir, 'tran.txt'), 'w')
            f.write('\n'.join([','.join(['%.5f' % _ for _ in t]) for t in tran.view(tran.shape[0], 3)]))
            f.close()

    def get_zero_pose_joint_and_vertex(self, shape: torch.Tensor = None):
        r"""
        Get the joint and vertex positions in zero pose. Root joint is aligned at zero.

        :param shape: Tensor for model shapes that can reshape to [batch_size, 10]. Use None for the mean(zero) shape.
        :return: Joint tensor in shape [batch_size, num_joint, 3] and vertex tensor in shape [batch_size, num_vertex, 3]
                 if shape is not None. Otherwise [num_joint, 3] and [num_vertex, 3] assuming the mean(zero) shape.
        """
        if shape is None:
            j, v = self._J - self._J[:1], (self._v_template if self.vert_mask is None else self._v_template[self.vert_mask, :]) - self._J[:1]
        else:
            shape = shape.view(-1, 10)
            v = torch.tensordot(shape, self._shapedirs, dims=([1], [2])) + self._v_template
            j = torch.matmul(self._J_regressor, v)
            j, v = j - j[:, :1], (v if self.vert_mask is None else v[:, self.vert_mask]) - j[:, :1]
        return j, v

    def bone_vector_to_joint_position(self, bone_vec: torch.Tensor):
        r"""
        Calculate joint positions in the base frame from bone vectors (position difference of child and parent joint)
        in the base frame. (torch, batch)

        Notes
        -----
        bone_vec[:, i] is the vector from parent[i] to i.

        Args
        -----
        :param bone_vec: Bone vector tensor in shape [batch_size, *] that can reshape to [batch_size, num_joint, 3].
        :return: Joint position, in shape [batch_size, num_joint, 3].
        """
        return M.bone_vector_to_joint_position(bone_vec, self.parent)

    def joint_position_to_bone_vector(self, joint_pos: torch.Tensor):
        r"""
        Calculate bone vectors (position difference of child and parent joint) in the base frame from joint positions
        in the base frame. (torch, batch)

        Notes
        -----
        bone_vec[:, i] is the vector from parent[i] to i.

        Args
        -----
        :param joint_pos: Joint position tensor in shape [batch_size, *] that can reshape to [batch_size, num_joint, 3].
        :return: Bone vector, in shape [batch_size, num_joint, 3].
        """
        return M.joint_position_to_bone_vector(joint_pos, self.parent)

    def forward_kinematics_R(self, R_local: torch.Tensor):
        r"""
        :math:`R_global = FK(R_local)`

        Forward kinematics that computes the global rotation of each joint from local rotations. (torch, batch)

        Notes
        -----
        A joint's *local* rotation is expressed in its parent's frame.

        A joint's *global* rotation is expressed in the base (root's parent) frame.

        Args
        -----
        :param R_local: Joint local rotation tensor in shape [batch_size, *] that can reshape to
                        [batch_size, num_joint, 3, 3] (rotation matrices).
        :return: Joint global rotation, in shape [batch_size, num_joint, 3, 3].
        """
        return M.forward_kinematics_R(R_local, self.parent)

    def forward_kinematics_masked(
        self,
        pose: torch.Tensor,
        shape: torch.Tensor = None,
        tran: torch.Tensor = None,
        calc_mesh: bool = False,
        vi_mask: torch.Tensor = None,   # e.g., torch.tensor([1961, 5424, 876, 4362, 411, 3021])
    ):
        """
        Same outputs as forward_kinematics, but if calc_mesh and vi_mask are provided,
        only computes vertex positions for the masked vertices to save memory.
        """
        def add_tran(x):
            return x if tran is None else x + tran.view(-1, 1, 3)

        device = pose.device
        pose = pose.view(pose.shape[0], -1, 3, 3)

        # zero-pose joints (Jx3) and vertices (Vx3); expand to batch
        j, v0 = [_.expand(pose.shape[0], -1, -1) for _ in self.get_zero_pose_joint_and_vertex(shape)]
        # local->global joint transforms
        T_local = M.transformation_matrix(pose, self.joint_position_to_bone_vector(j))
        T_global = self.forward_kinematics_T(T_local)
        pose_global, joint_global = M.decode_transformation_matrix(T_global)
        if not calc_mesh:
            return pose_global, add_tran(joint_global)

        # ---- Only operate on masked vertices from here on ----
        if vi_mask is None:
            # fall back to full computation if mask not given
            vi_mask = torch.arange(v0.shape[1], device=device)

        vi_mask = vi_mask.to(device=device, dtype=torch.long)

        # adjust transforms so they map from joint origin to world (as in your original code)
        T_global = T_global.clone()  # avoid in-place on shared tensors
        T_global[..., -1:] -= torch.matmul(T_global, M.append_zero(j, dim=-1).unsqueeze(-1))

        # Skinning weights: [V, J] (in your code dims=([1],[1]) + permute -> [B, V, 4,4])
        # Slice only rows we need: [V_sel, J]
        W_sel = self._skinning_weights.index_select(0, vi_mask)  # [V_sel, J]

        # Blend joint transforms only for selected vertices:
        # tensordot over joint axis -> [B, 4,4, V_sel] then permute -> [B, V_sel, 4,4]
        T_vertex_sel = torch.tensordot(T_global, W_sel, dims=([1], [1])).permute(0, 3, 1, 2)

        # Base vertex positions, masked
        v_sel = v0.index_select(1, vi_mask)  # [B, V_sel, 3]

        # Optional pose blendshapes — slice posedirs on vertex axis only
        if self.use_pose_blendshape:
            # r: [B, 9*(J-1)]
            r = (pose[:, 1:] - torch.eye(3, device=device)).flatten(1)
            # posedirs is typically [3, V, P] where P=9*(J-1); slice vertex axis
            posedirs_sel = self._posedirs.index_select(1, vi_mask)  # [3, V_sel, P]
            # tensordot over P -> [B, 3, V_sel] then permute to [B, V_sel, 3]
            dv = torch.tensordot(r, posedirs_sel, dims=([1], [2])).permute(0, 2, 1)
            v_sel = v_sel + dv

        # Transform only selected vertices
        vertex_global_sel = torch.matmul(T_vertex_sel, M.append_one(v_sel, dim=-1).unsqueeze(-1)).squeeze(-1)[..., :3]
        return pose_global, add_tran(joint_global), add_tran(vertex_global_sel)
    
    def inverse_kinematics_R(self, R_global: torch.Tensor):
        r"""
        :math:`R_local = IK(R_global)`

        Inverse kinematics that computes the local rotation of each joint from global rotations. (torch, batch)

        Notes
        -----
        A joint's *local* rotation is expressed in its parent's frame.

        A joint's *global* rotation is expressed in the base (root's parent) frame.

        Args
        -----
        :param R_global: Joint global rotation tensor in shape [batch_size, *] that can reshape to
                         [batch_size, num_joint, 3, 3] (rotation matrices).
        :return: Joint local rotation, in shape [batch_size, num_joint, 3, 3].
        """
        return M.inverse_kinematics_R(R_global, self.parent)

    def forward_kinematics_T(self, T_local: torch.Tensor):
        r"""
        :math:`T_global = FK(T_local)`

        Forward kinematics that computes the global homogeneous transformation of each joint from
        local homogeneous transformations. (torch, batch)

        Notes
        -----
        A joint's *local* transformation is expressed in its parent's frame.

        A joint's *global* transformation is expressed in the base (root's parent) frame.

        Args
        -----
        :param T_local: Joint local transformation tensor in shape [batch_size, *] that can reshape to
                        [batch_size, num_joint, 4, 4] (homogeneous transformation matrices).
        :return: Joint global transformation matrix, in shape [batch_size, num_joint, 4, 4].
        """
        return M.forward_kinematics_T(T_local, self.parent)

    def inverse_kinematics_T(self, T_global: torch.Tensor):
        r"""
        :math:`T_local = IK(T_global)`

        Inverse kinematics that computes the local homogeneous transformation of each joint from
        global homogeneous transformations. (torch, batch)

        Notes
        -----
        A joint's *local* transformation is expressed in its parent's frame.

        A joint's *global* transformation is expressed in the base (root's parent) frame.

        Args
        -----
        :param T_global: Joint global transformation tensor in shape [batch_size, *] that can reshape to
                        [batch_size, num_joint, 4, 4] (homogeneous transformation matrices).
        :return: Joint local transformation matrix, in shape [batch_size, num_joint, 4, 4].
        """
        return M.inverse_kinematics_T(T_global, self.parent)

    def forward_kinematics(self, pose: torch.Tensor, shape: torch.Tensor = None, tran: torch.Tensor = None,
                           calc_mesh=False):
        r"""
        Forward kinematics that computes the global joint rotation, joint position, and additionally
        mesh vertex position from poses, shapes, and translations. (torch, batch)

        :param pose: Joint local rotation tensor in shape [batch_size, *] that can reshape to
                     [batch_size, num_joint, 3, 3] (rotation matrices).
        :param shape: Tensor for model shapes that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param tran: Root position tensor in shape [batch_size, 3]. Use None for the zero positions.
        :param calc_mesh: Whether to calculate mesh vertex positions.
        :return: Joint global rotation in [batch_size, num_joint, 3, 3],
                 joint position in [batch_size, num_joint, 3],
                 and additionally mesh vertex position in [batch_size, num_vertex, 3] if calc_mesh is True.
        """
        def add_tran(x):
            return x if tran is None else x + tran.view(-1, 1, 3)

        pose = pose.view(pose.shape[0], -1, 3, 3)
        j, v = [_.expand(pose.shape[0], -1, -1) for _ in self.get_zero_pose_joint_and_vertex(shape)]
        T_local = M.transformation_matrix(pose, self.joint_position_to_bone_vector(j))
        T_global = self.forward_kinematics_T(T_local)
        pose_global, joint_global = M.decode_transformation_matrix(T_global)
        if calc_mesh is False:
            return pose_global, add_tran(joint_global)

        if self.use_pose_blendshape:
            r = (pose[:, 1:] - torch.eye(3, device=pose.device)).flatten(1)
            posedirs = self._posedirs if self.vert_mask is None else self._posedirs[self.vert_mask, :, :]
            v = v + torch.tensordot(r, posedirs, dims=([1], [2]))
        T_global[..., -1:] -= torch.matmul(T_global, M.append_zero(j, dim=-1).unsqueeze(-1))
        skinning_weights = self._skinning_weights if self.vert_mask is None else self._skinning_weights[self.vert_mask, :]
        T_vertex = torch.tensordot(T_global, skinning_weights, dims=([1], [1])).permute(0, 3, 1, 2)
        vertex_global = torch.matmul(T_vertex, M.append_one(v, dim=-1).unsqueeze(-1)).squeeze(-1)[..., :3]
        return pose_global, add_tran(joint_global), add_tran(vertex_global)

    def view_joint(self, joint_list: list, fps=60, distance_between_subjects=0.8):
        r"""
        View model joint (single frame or a sequence).

        Notes
        -----
        If num_frame == 1, only show one picture.

        Args
        -----
        :param joint_list: List in length [num_subject] of tensors that can all reshape to [num_frame, num_joint, 3].
        :param fps: Sequence FPS.
        :param distance_between_subjects: Distance in meters between subjects. 0.2 for hand and 0.8 for body is good.
        """
        import vctoolkit as vc
        import vctoolkit.viso3d as vo3d
        joint_list = [(j.view(-1, len(self.parent), 3) - j.view(-1, len(self.parent), 3)[:1, :1]).cpu().numpy()
                      for j in joint_list]

        v_list, f_list = [], []
        f = vc.joints_to_mesh(joint_list[0][0], self.parent)[1]
        for i in range(len(joint_list)):
            v = np.stack([vc.joints_to_mesh(frame, self.parent)[0] for frame in joint_list[i]])
            v[:, :, 0] += distance_between_subjects * i
            v_list.append(v)
            f_list.append(f.copy())
            f += v.shape[1]

        verts = np.concatenate(v_list, axis=1)
        faces = np.concatenate(f_list)
        if verts.shape[0] > 1:
            vo3d.render_sequence_3d(verts, faces, 720, 720, 'a.avi', fps, visible=True)
        else:
            vo3d.vis_mesh(verts[0], faces)

    def view_mesh(self, vertex_list: list, fps=60, distance_between_subjects=0.8):
        r"""
        View model mesh (single frame or a sequence).

        Notes
        -----
        If num_frame == 1, only show one picture.

        Args
        -----
        :param vertex_list: List in length [num_subject] of tensors that can all reshape to [num_frame, num_vertex, 3].
        :param fps: Sequence FPS.
        :param distance_between_subjects: Distance in meters between subjects. 0.2 for hand and 0.8 for body is good.
        """
        import vctoolkit.viso3d as vo3d
        v_list, f_list = [], []
        f = self.face.copy()
        for i in range(len(vertex_list)):
            v = vertex_list[i].clone().view(-1, self._v_template.shape[0], 3)
            v[:, :, 0] += distance_between_subjects * i
            v_list.append(v)
            f_list.append(f.copy())
            f += v.shape[1]

        verts = torch.cat(v_list, dim=1).cpu().numpy()
        faces = np.concatenate(f_list)
        if verts.shape[0] > 1:
            vo3d.render_sequence_3d(verts, faces, 720, 720, 'a.avi', fps, visible=True)
        else:
            vo3d.vis_mesh(verts[0], faces)

    def view_motion(self, pose_list: list, tran_list: list = None, fps=60, distance_between_subjects=0.8):
        r"""
        View model motion (poses and translations) (single frame or a sequence).

        Notes
        -----
        If num_frame == 1, only show one picture.

        Args
        -----
        :param pose_list: List in length [num_subject] of tensors that can all reshape to [num_frame, num_joint, 3, 3].
        :param tran_list: List in length [num_subject] of tensors that can all reshape to [num_frame, 3].
        :param fps: Sequence FPS.
        :param distance_between_subjects: Distance in meters between subjects. 0.2 for hand and 0.8 for body is good.
        """
        vert_mask = self.vert_mask
        self.vert_mask = None
        verts = []
        for i in range(len(pose_list)):
            pose = pose_list[i].view(-1, len(self.parent), 3, 3)
            tran = tran_list[i].view(-1, 3) - tran_list[i].view(-1, 3)[:1] if tran_list else None
            verts.append(self.forward_kinematics(pose, tran=tran, calc_mesh=True)[2])
        self.view_mesh(verts, fps, distance_between_subjects=distance_between_subjects)
        self.vert_mask = vert_mask

    def view_mesh_overlay(self, verts, images, K, Tcw=torch.eye(4), fps=60):
        r"""
        View model mesh overlay (single frame or a sequence).
        Warning: open3d has bugs. This function may not run correctly.

        Notes
        -----
        If num_frame == 1, only show one picture.

        Args
        -----
        :param verts: Tensor that can reshape to [num_frame, num_vertex, 3].
        :param images: Numpy uint8 array that can expand to [num_frame, height, width, 3].
        :param K: Camera intrinsic tensor in shape [3, 3].
        :param Tcw: Camera extrinsic tensor in shape [4, 4].
        :param fps: Sequence FPS.
        """
        import vctoolkit.viso3d as vo3d
        import cv2

        verts = verts.cpu().view(-1, self._v_template.shape[0], 3).numpy()
        images = np.broadcast_to(images, (verts.shape[0], images.shape[-3], images.shape[-2], 3))

        param = vo3d.o3d.camera.PinholeCameraParameters()
        param.intrinsic = vo3d.o3d.camera.PinholeCameraIntrinsic()
        param.intrinsic.intrinsic_matrix = K.numpy()
        param.extrinsic = Tcw.numpy()

        viewer = vo3d.o3d.visualization.Visualizer()
        viewer.create_window(width=images.shape[-2], height=images.shape[-3], visible=False)
        mesh = vo3d.create_o3d_mesh(verts[0], self.face)
        viewer.add_geometry(mesh)
        view_control = viewer.get_view_control()
        view_control.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        viewer.get_render_option().background_color = [0, 0, 0]
        padw = images.shape[-2] - view_control.convert_to_pinhole_camera_parameters().intrinsic.width
        padh = images.shape[-3] - view_control.convert_to_pinhole_camera_parameters().intrinsic.height
        assert padw >= 0 and padh >= 0
        wb, we = padw // 2, images.shape[-2] + padw // 2 - padw
        hb, he = padh // 2, images.shape[-3] + padh // 2 - padh

        if len(verts) == 1:
            viewer.poll_events()
            viewer.update_renderer()
            frame = (np.asarray(viewer.capture_screen_float_buffer()) * 255).astype(np.uint8)
            mask = np.tile(frame.astype(np.bool8).max(axis=2, keepdims=True), (1, 1, 3))
            im = images[0].copy()
            im[hb:he, wb:we][mask] = frame[mask]
            cv2.imshow('overlay', im)
            cv2.waitKey(0)
        else:
            writer = cv2.VideoWriter('a.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (images[0].shape[1], images[0].shape[0]))
            for i in tqdm.trange(len(verts)):
                mesh.vertices = vo3d.o3d.utility.Vector3dVector(verts[i])
                mesh.compute_vertex_normals()
                viewer.update_geometry(mesh)
                viewer.poll_events()
                viewer.update_renderer()
                frame = (np.asarray(viewer.capture_screen_float_buffer()) * 255).astype(np.uint8)
                mask = np.tile(frame.astype(np.bool8).max(axis=2, keepdims=True), (1, 1, 3))
                im = images[i].copy()
                im[hb:he, wb:we][mask] = frame[mask]
                cv2.imshow('overlay', im)
                cv2.waitKey(1)
                writer.write(im)
            writer.release()
        cv2.destroyWindow('overlay')

    def view_motion_overlay(self, pose, tran, images, K, Tcw=torch.eye(4), fps=60):
        r"""
        View model motion (poses and translations) overlay (single frame or a sequence).
        Warning: open3d has bugs. This function may not run correctly.

        Notes
        -----
        If num_frame == 1, only show one picture.

        Args
        -----
        :param pose: Tensor that can reshape to [num_frame, num_joint, 3, 3].
        :param tran: Tensor that can reshape to [num_frame, 3].
        :param images: Numpy uint8 array that can expand to [num_frame, height, width, 3].
        :param K: Camera intrinsic tensor in shape [3, 3].
        :param Tcw: Camera extrinsic tensor in shape [4, 4].
        :param fps: Sequence FPS.
        """
        vert_mask = self.vert_mask
        self.vert_mask = None
        verts = self.forward_kinematics(pose.view(-1, len(self._J), 3, 3), tran=tran.view(-1, 3), calc_mesh=True)[2]
        self.view_mesh_overlay(verts, images, K, Tcw, fps)
        self.vert_mask = vert_mask

    def is_inside(self, points: torch.Tensor, vertices: torch.Tensor):
        r"""
        Check if the points are inside the model.
        Modified from https://github.com/YuliangXiu/ICON/blob/master/lib/common/render_utils.py

        Args
        -----
        :param points: Tensor in shape [num_point, 3].
        :param vertices: Tensor in shape [num_vertex, 3].
        :return: Bool tensor in shape [num_points].
        """
        centered_tris = vertices[None, self.face.astype(int)] - points[:, None, None]
        norms = torch.norm(centered_tris, dim=-1)
        cross_prod = torch.cross(centered_tris[:, :, 1], centered_tris[:, :, 2], dim=-1)
        numerator = (centered_tris[:, :, 0] * cross_prod).sum(dim=-1)
        dot01 = (centered_tris[:, :, 0] * centered_tris[:, :, 1]).sum(dim=-1)
        dot12 = (centered_tris[:, :, 1] * centered_tris[:, :, 2]).sum(dim=-1)
        dot02 = (centered_tris[:, :, 0] * centered_tris[:, :, 2]).sum(dim=-1)
        denominator = norms.prod(dim=-1) + dot01 * norms[:, :, 2] + dot02 * norms[:, :, 1] + dot12 * norms[:, :, 0]
        solid_angle = torch.atan2(numerator, denominator)
        return (solid_angle.sum(dim=-1) / (2 * torch.pi)).round().bool()

    def physical_properties(self, shape=None, res=0.02):
        r"""
        Compute the physical properties (mass, com, inertia) of the model.

        Args
        -----
        :param shape: Tensor for model shape in [10]. Use None for the mean(zero) shape.
        :param res: Resolution for mesh voxelization.
        :return: Mass tensor in shape [num_joint],
                 center of mass tensor in shape [num_joint, 3] expressed in the joint frame, and
                 inertia tensor in shape [num_joint, 3, 3] expressed in the CoM frame.
        """
        from scipy.interpolate import LinearNDInterpolator
        vert_mask = self.vert_mask
        self.vert_mask = None
        joint, vert = self.get_zero_pose_joint_and_vertex(shape)
        joint = joint.squeeze(0)
        vert = vert.squeeze(0)
        self.vert_mask = vert_mask

        # slice the mesh along x-axis and find inner points
        point = []
        for x in torch.arange(vert[:, 0].min().item(), vert[:, 0].max().item(), res):
            v = vert[(vert[:, 0] - x).abs() < max(0.04, res)]
            g = [x, torch.arange(v[:, 1].min().item(), v[:, 1].max().item(), res), torch.arange(v[:, 2].min().item(), v[:, 2].max().item(), res)]
            g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).view(-1, 3).to(vert.device)
            point.append(g[self.is_inside(g, vert)])
        point = torch.cat(point)

        # compute points' weighted mass seen from each joint
        weight_fn = LinearNDInterpolator(vert.cpu(), self._skinning_weights.cpu())
        weight = torch.from_numpy(weight_fn(point.cpu()).T).float().to(vert.device)
        weighted_mass = weight * 1000 * res ** 3

        # compute physical properties
        mass = torch.einsum('ij->i', weighted_mass)
        com = torch.einsum('ij,ijk->ik', weighted_mass, point[None] - joint[:, None]) / mass.unsqueeze(-1)
        local_point_cross = M.hat(point[None] - com[:, None])
        inertia = torch.einsum('ij,ijkl->ikl', weighted_mass, -local_point_cross.matmul(local_point_cross))

        # # visualization
        # from articulate.utils.open3d import PointCloud3DViewer
        # from articulate.utils.color import ColorBar
        # colorbar = ColorBar()
        # colorbar.show()
        # with PointCloud3DViewer() as pcviewer:
        #     for i in range(len(self._J)):
        #         pcviewer.update(point.cpu(), colorbar(weight[i].cpu()), reset_view_point=True)
        #         pcviewer.pause()

        return mass, com, inertia
    
    def view_motion_web_with_gt(
        self,
        pose_list,
        tran_list=None,
        head_list=None,
        pose_list_gt=None,
        tran_list_gt=None,
        head_list_gt=None,
        fps=60,
        colors=None,
        colors_gt=None,
        distance_between_subjects=0.8,
        gt_group_offset=1.0,   # place GT meshes / lines along +Z by this amount (MeshCat coords)
        zmq_url=None,
        y_up=True,            # NEW: if True, apply a global Y-up -> MeshCat(Z-up) transform
    ):
        """
        Visualize predicted and (optionally) ground-truth sequences in MeshCat.

        Notes on axes
        -------------
        MeshCat uses Z-up by convention. If your data is Y-up and you want the viewer
        to *look* Y-up, set y_up=True. Internally MeshCat stays Z-up; we rotate the
        whole scene by -90deg around X.
        """
        import numpy as np
        import meshcat
        import meshcat.geometry as mcg
        import meshcat.transformations as mct
        from time import sleep

        # ---------- 1. sanity checks ----------
        n_subj = len(pose_list)
        assert tran_list is None or len(tran_list) == n_subj, "tran_list len does not match pose_list"
        assert head_list is None or len(head_list) == n_subj, "head_list len does not match pose_list"

        n_subj_gt = len(pose_list_gt) if pose_list_gt is not None else 0
        if pose_list_gt is not None:
            assert tran_list_gt is None or len(tran_list_gt) == n_subj_gt, "tran_list_gt len does not match pose_list_gt"
            assert head_list_gt is None or len(head_list_gt) == n_subj_gt, "head_list_gt len does not match pose_list_gt"

        faces_template = self.face  # (F_idx,3)

        # We'll mount everything under a single scene root so we can rotate it once.
        ROOT = "world"

        # ---------- helper to build geometry packs ----------
        def build_group(pose_list_, tran_list_, head_list_, x_offset_per_subj=0.0, z_offset_group=0.0):
            """Return dict with verts_cat, faces_cat, traj_all, head_all, F, V."""
            if pose_list_ is None:
                return None

            verts_all, faces_all = [], []
            traj_all, head_all = [], []

            for sid in range(len(pose_list_)):
                pose = pose_list_[sid].view(-1, len(self.parent), 3, 3)  # [F,J,3,3]

                # Normalize to the first translation if provided (matches user's behavior)
                if (tran_list_ is not None) and (tran_list_[sid] is not None):
                    tran = tran_list_[sid].view(-1, 3) - tran_list_[sid][:1]
                else:
                    tran = None

                if (head_list_ is not None) and (head_list_[sid] is not None):
                    if (tran_list_ is not None) and (tran_list_[sid] is not None):
                        head = head_list_[sid].view(-1, 3) - tran_list_[sid][:1].to(head_list_[sid].device)
                    else:
                        head = head_list_[sid].view(-1, 3)
                else:
                    head = None

                # Forward kinematics → verts sequence [F,V,3]
                verts_seq = (
                    self.forward_kinematics(pose, tran=tran, calc_mesh=True)[2]
                    .detach()
                    .cpu()
                    .numpy()
                )

                # Space out subjects along +X, and apply a group +Z offset (for GT)
                verts_seq[..., 0] += sid * distance_between_subjects + x_offset_per_subj
                verts_seq[..., 2] += z_offset_group

                # Keep the raw trajectories (offset to match mesh)
                if tran is not None:
                    t_np = tran.detach().cpu().numpy()
                    t_np[:, 0] += sid * distance_between_subjects + x_offset_per_subj
                    t_np[:, 2] += z_offset_group
                    traj_all.append(t_np)
                else:
                    traj_all.append(None)

                if head is not None:
                    h_np = head.detach().cpu().numpy()
                    h_np[:, 0] += sid * distance_between_subjects + x_offset_per_subj
                    h_np[:, 2] += z_offset_group
                    head_all.append(h_np)
                else:
                    head_all.append(None)

                verts_all.append(verts_seq)
                faces_all.append(faces_template + len(self._v_template) * sid)

            verts_cat = np.concatenate(verts_all, axis=1)  # [F, N*V, 3]
            faces_cat = np.concatenate(faces_all, axis=0)  # [N*T, 3]
            F = verts_cat.shape[0]
            V = verts_cat.shape[1]
            return dict(verts_cat=verts_cat, faces_cat=faces_cat, traj_all=traj_all, head_all=head_all, F=F, V=V)

        # Build predicted and GT packs
        pack_pred = build_group(pose_list, tran_list, head_list, x_offset_per_subj=0.0, z_offset_group=0.0)
        pack_gt   = build_group(pose_list_gt, tran_list_gt, head_list_gt, x_offset_per_subj=0.0, z_offset_group=gt_group_offset) if pose_list_gt is not None else None

        # ---------- 2. connect MeshCat ----------
        vis = meshcat.Visualizer(zmq_url=zmq_url)
        url = vis.url()
        print("MeshCat at:", url)

        # Set white background
        vis["/Background"].set_property("top_color", [1, 1, 1])
        vis["/Background"].set_property("bottom_color", [1, 1, 1])

        # Move the grid/floor down by 1m (in MeshCat Z-up coords, this is -1 in Z)
        vis["/Grid"].set_transform(mct.translation_matrix([0, 0, -1]))

        # Clear prior content under ROOT (optional but nice)
        try:
            vis[ROOT].delete()
        except Exception:
            pass

        # ---------- 2b. apply global Y-up transform ----------
        # MeshCat is Z-up internally. If your *data* is Y-up and you want the viewer
        # to behave like Y-up, rotate the entire scene by -90deg about X:
        #
        #   (x, y, z)_data  ->  (x, z, -y)_meshcat
        #
        if y_up:
            T_yup_to_zup = mct.rotation_matrix(np.pi / 2.0, [1, 0, 0])
            vis[ROOT].set_transform(T_yup_to_zup)
        else:
            vis[ROOT].set_transform(np.eye(4))

        # ---------- 3. create MeshCat meshes ----------
        default_color_pred = 0x6699FF
        default_color_gt   = 0xFFAA33

        # Predicted mesh
        mesh_pred = mcg.TriangularMeshGeometry(
            pack_pred["verts_cat"][0].astype("float32"),
            pack_pred["faces_cat"].astype("uint32"),
        )
        material_pred = mcg.MeshLambertMaterial(color=default_color_pred, vertexColors=False)

        if colors and len(colors) == n_subj:
            col_arr = np.zeros((pack_pred["V"], 3), dtype=np.uint8)
            v_offset = 0
            for sid, col in enumerate(colors):
                r = (col >> 16) & 0xFF
                g = (col >> 8) & 0xFF
                b = col & 0xFF
                col_arr[v_offset : v_offset + len(self._v_template)] = np.array([r, g, b], dtype=np.uint8)
                v_offset += len(self._v_template)
            mesh_pred = mcg.TriangularMeshGeometry(
                pack_pred["verts_cat"][0].astype("float32"),
                pack_pred["faces_cat"].astype("uint32"),
                mcg.ColorArray(col_arr),
            )
            material_pred = mcg.MeshLambertMaterial(vertexColors=True)

        vis[f"{ROOT}/body"].set_object(mesh_pred, material_pred)

        # GT mesh (optional)
        if pack_gt is not None:
            mesh_gt = mcg.TriangularMeshGeometry(
                pack_gt["verts_cat"][0].astype("float32"),
                pack_gt["faces_cat"].astype("uint32"),
            )
            material_gt = mcg.MeshLambertMaterial(color=default_color_gt, vertexColors=False)

            if colors_gt and len(colors_gt) == n_subj_gt:
                col_arr_gt = np.zeros((pack_gt["V"], 3), dtype=np.uint8)
                v_offset = 0
                for sid, col in enumerate(colors_gt):
                    r = (col >> 16) & 0xFF
                    g = (col >> 8) & 0xFF
                    b = col & 0xFF
                    col_arr_gt[v_offset : v_offset + len(self._v_template)] = np.array([r, g, b], dtype=np.uint8)
                    v_offset += len(self._v_template)
                mesh_gt = mcg.TriangularMeshGeometry(
                    pack_gt["verts_cat"][0].astype("float32"),
                    pack_gt["faces_cat"].astype("uint32"),
                    mcg.ColorArray(col_arr_gt),
                )
                material_gt = mcg.MeshLambertMaterial(vertexColors=True)

            vis[f"{ROOT}/body_gt"].set_object(mesh_gt, material_gt)

        # ---------- 3b. trajectories + moving markers ----------
        # Predicted trajectories / heads are drawn in the same coordinate system as the meshes.
        for sid in range(n_subj):
            traj = pack_pred["traj_all"][sid]
            if traj is not None:
                pts = traj.T.astype("float32")  # (3,F)
                line_col = 0xFF0000  # red
                vis[f"{ROOT}/traj/{sid}"].set_object(
                    mcg.Line(
                        mcg.PointsGeometry(pts),
                        mcg.LineBasicMaterial(color=line_col, linewidth=50),
                    )
                )
                vis[f"{ROOT}/marker/{sid}"].set_object(mcg.Sphere(0.02), mcg.MeshLambertMaterial(color=line_col))
                vis[f"{ROOT}/marker/{sid}"].set_transform(mct.translation_matrix(traj[0]))

        for sid in range(n_subj):
            head = pack_pred["head_all"][sid]
            if head is not None:
                pts = head.T.astype("float32")
                line_col = 0xFF0000  # red
                vis[f"{ROOT}/traj_head/{sid}"].set_object(
                    mcg.Line(
                        mcg.PointsGeometry(pts),
                        mcg.LineBasicMaterial(color=line_col, linewidth=50),
                    )
                )
                vis[f"{ROOT}/marker_head/{sid}"].set_object(mcg.Sphere(0.02), mcg.MeshLambertMaterial(color=line_col))
                vis[f"{ROOT}/marker_head/{sid}"].set_transform(mct.translation_matrix(head[0]))

        # GT (optional), use green
        if pack_gt is not None:
            for sid in range(n_subj_gt):
                traj = pack_gt["traj_all"][sid]
                if traj is not None:
                    pts = traj.T.astype("float32")
                    line_col = 0x00FF00  # green
                    vis[f"{ROOT}/traj_gt/{sid}"].set_object(
                        mcg.Line(
                            mcg.PointsGeometry(pts),
                            mcg.LineBasicMaterial(color=line_col, linewidth=50),
                        )
                    )
                    vis[f"{ROOT}/marker_gt/{sid}"].set_object(mcg.Sphere(0.02), mcg.MeshLambertMaterial(color=line_col))
                    vis[f"{ROOT}/marker_gt/{sid}"].set_transform(mct.translation_matrix(traj[0]))

            for sid in range(n_subj_gt):
                head = pack_gt["head_all"][sid]
                if head is not None:
                    pts = head.T.astype("float32")
                    line_col = 0x00FF00  # green
                    vis[f"{ROOT}/traj_head_gt/{sid}"].set_object(
                        mcg.Line(
                            mcg.PointsGeometry(pts),
                            mcg.LineBasicMaterial(color=line_col, linewidth=50),
                        )
                    )
                    vis[f"{ROOT}/marker_head_gt/{sid}"].set_object(mcg.Sphere(0.02), mcg.MeshLambertMaterial(color=line_col))
                    vis[f"{ROOT}/marker_head_gt/{sid}"].set_transform(mct.translation_matrix(head[0]))

        # ---------- 4. animation ----------
        dt = 1.0 / fps
        F_pred = pack_pred["F"]
        F_gt   = pack_gt["F"] if pack_gt is not None else 0
        F_max  = max(F_pred, F_gt)

        for f in range(F_max):
            # Update predicted mesh/markers
            if f < F_pred:
                mesh_pred.vertices = pack_pred["verts_cat"][f].astype("float32")
                vis[f"{ROOT}/body"].set_object(mesh_pred, material_pred)

                if tran_list is not None:
                    for sid in range(n_subj):
                        traj = pack_pred["traj_all"][sid]
                        if traj is not None:
                            vis[f"{ROOT}/marker/{sid}"].set_transform(mct.translation_matrix(traj[f]))

                if head_list is not None:
                    for sid in range(n_subj):
                        head = pack_pred["head_all"][sid]
                        if head is not None:
                            vis[f"{ROOT}/marker_head/{sid}"].set_transform(mct.translation_matrix(head[f]))

            # Update GT mesh/markers
            if pack_gt is not None and f < F_gt:
                mesh_gt.vertices = pack_gt["verts_cat"][f].astype("float32")
                vis[f"{ROOT}/body_gt"].set_object(mesh_gt, material_gt)

                if tran_list_gt is not None:
                    for sid in range(n_subj_gt):
                        traj = pack_gt["traj_all"][sid]
                        if traj is not None:
                            vis[f"{ROOT}/marker_gt/{sid}"].set_transform(mct.translation_matrix(traj[f]))

                if head_list_gt is not None:
                    for sid in range(n_subj_gt):
                        head = pack_gt["head_all"][sid]
                        if head is not None:
                            vis[f"{ROOT}/marker_head_gt/{sid}"].set_transform(mct.translation_matrix(head[f]))

            sleep(dt)