import os
import pickle
import torch
import pypose as pp
import torch.nn.functional as F
from tqdm import tqdm
from .dataset import Sequence
import core.articulate as art
from core.paths import Paths


VI_MASK_6 = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
JI_MASK_6 = torch.tensor([18, 19, 1, 2, 15, 0])

# IMUPoser raw to CPF world
ZUP_TO_CPF_WORLD = torch.tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0.]], dtype=torch.float32)


def mat2quat(R: torch.Tensor):
    """
    Convert rotation matrices to quaternions (x, y, z, w) - PyPose convention.

    Args:
        R: (..., 3, 3) rotation matrices

    Returns:
        quat: (..., 4) quaternions in (x, y, z, w)
    """
    assert R.shape[-2:] == (3, 3)

    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    trace = m00 + m11 + m22

    quat = torch.zeros(R.shape[0], 4, dtype=R.dtype, device=R.device)

    # Case 1: trace > 0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2  # s = 4 * qw
    mask = trace > 0
    quat[mask, 3] = 0.25 * s[mask]  # qw
    quat[mask, 0] = (m21[mask] - m12[mask]) / s[mask]  # qx
    quat[mask, 1] = (m02[mask] - m20[mask]) / s[mask]  # qy
    quat[mask, 2] = (m10[mask] - m01[mask]) / s[mask]  # qz

    # Case 2: m00 > m11 and m00 > m22
    mask = (~(trace > 0)) & (m00 > m11) & (m00 > m22)
    s = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-10)) * 2  # s = 4 * qx
    quat[mask, 3] = (m21[mask] - m12[mask]) / s[mask]  # qw
    quat[mask, 0] = 0.25 * s[mask]  # qx
    quat[mask, 1] = (m01[mask] + m10[mask]) / s[mask]  # qy
    quat[mask, 2] = (m02[mask] + m20[mask]) / s[mask]  # qz

    # Case 3: m11 > m22
    mask = (~(trace > 0)) & (~((m00 > m11) & (m00 > m22))) & (m11 > m22)
    s = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=1e-10)) * 2  # s = 4 * qy
    quat[mask, 3] = (m02[mask] - m20[mask]) / s[mask]  # qw
    quat[mask, 0] = (m01[mask] + m10[mask]) / s[mask]  # qx
    quat[mask, 1] = 0.25 * s[mask]  # qy
    quat[mask, 2] = (m12[mask] + m21[mask]) / s[mask]  # qz

    # Case 4: else
    mask = (~(trace > 0)) & (~((m00 > m11) & (m00 > m22))) & (~(m11 > m22))
    s = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=1e-10)) * 2  # s = 4 * qz
    quat[mask, 3] = (m10[mask] - m01[mask]) / s[mask]  # qw
    quat[mask, 0] = (m02[mask] + m20[mask]) / s[mask]  # qx
    quat[mask, 1] = (m12[mask] + m21[mask]) / s[mask]  # qy
    quat[mask, 2] = 0.25 * s[mask]  # qz

    # Normalize
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-10)

    return quat.reshape(batch_shape + (4,))


# def rot_to_6d(R: torch.Tensor):
#     """
#     Convert rotation matrix [..., 3, 3] to 6D representation [..., 6].
#     Takes first two columns of the rotation matrix.
#     """
#     return R[..., :, :2].transpose(-1, -2).reshape(*R.shape[:-2], 6)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def _resample_linear(x: torch.Tensor, new_len: int) -> torch.Tensor:
    """Linear resample a tensor with leading time dim [N, ...] to [new_len, ...]."""
    if x.shape[0] == new_len:
        return x
    if x.shape[0] < 2:
        return x.repeat(new_len, *([1] * (x.ndim - 1)))

    x_shape = x.shape
    x_flat = x.reshape(x_shape[0], -1).transpose(0, 1).unsqueeze(0)  # [1, C, N]
    y_flat = F.interpolate(x_flat, size=new_len, mode='linear', align_corners=True)
    y = y_flat.squeeze(0).transpose(0, 1).reshape(new_len, *x_shape[1:])
    return y


def _resample_nearest(x: torch.Tensor, new_len: int) -> torch.Tensor:
    """Nearest-neighbor resample a tensor with leading time dim [N, ...] to [new_len, ...]."""
    if x.shape[0] == new_len:
        return x
    if x.shape[0] < 2:
        return x.repeat(new_len, *([1] * (x.ndim - 1)))

    x_shape = x.shape
    x_flat = x.reshape(x_shape[0], -1).transpose(0, 1).unsqueeze(0)  # [1, C, N]
    y_flat = F.interpolate(x_flat, size=new_len, mode='nearest')
    y = y_flat.squeeze(0).transpose(0, 1).reshape(new_len, *x_shape[1:])
    return y


def _resample_rotmat_nlerp(rot: torch.Tensor, new_len: int) -> torch.Tensor:
    """Resample rotation matrices [N, ..., 3, 3] using quaternion nlerp."""
    if rot.shape[0] == new_len:
        return rot
    if rot.shape[0] < 2:
        return rot.repeat(new_len, *([1] * (rot.ndim - 1)))

    n_old = rot.shape[0]
    trailing_shape = rot.shape[1:-2]

    rot_flat = rot.reshape(n_old, -1, 3, 3)  # [N, K, 3, 3]
    K = rot_flat.shape[1]

    quat = mat2quat(rot_flat.reshape(-1, 3, 3)).reshape(n_old, K, 4)  # [N, K, 4]

    src_pos = torch.linspace(0.0, float(n_old - 1), steps=new_len, device=rot.device, dtype=rot.dtype)
    idx0 = torch.floor(src_pos).long()
    idx1 = torch.clamp(idx0 + 1, max=n_old - 1)
    alpha = (src_pos - idx0.to(src_pos.dtype)).view(new_len, 1, 1)  # [new_len, 1, 1]

    q0 = quat[idx0]  # [new_len, K, 4]
    q1 = quat[idx1]  # [new_len, K, 4]

    # Hemisphere correction for smooth interpolation
    same_hemi = (q0 * q1).sum(dim=-1, keepdim=True) >= 0
    q1 = torch.where(same_hemi, q1, -q1)

    q = (1.0 - alpha) * q0 + alpha * q1
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-10)

    rot_new = pp.SO3(q.reshape(-1, 4)).matrix().reshape(new_len, K, 3, 3)
    return rot_new.reshape(new_len, *trailing_shape, 3, 3)


def _resample_rotmat_nearest(rot: torch.Tensor, new_len: int) -> torch.Tensor:
    """Nearest-neighbor resample rotation matrices [N, ..., 3, 3]."""
    if rot.shape[0] == new_len:
        return rot
    if rot.shape[0] < 2:
        return rot.repeat(new_len, *([1] * (rot.ndim - 1)))

    n_old = rot.shape[0]
    src_pos = torch.linspace(0.0, float(n_old - 1), steps=new_len, device=rot.device, dtype=rot.dtype)
    idx = torch.round(src_pos).long()
    idx = torch.clamp(idx, 0, n_old - 1)
    return rot[idx]


def _rotmat_to_euler_zyx_deg(rot: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices [N, 3, 3] to Euler angles [yaw, pitch, roll] in degrees (ZYX order)."""
    if rot.ndim != 3 or rot.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [N, 3, 3], got {tuple(rot.shape)}")

    # ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    sy = torch.sqrt(rot[:, 0, 0] * rot[:, 0, 0] + rot[:, 1, 0] * rot[:, 1, 0])
    singular = sy < 1e-6

    yaw = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
    pitch = torch.atan2(-rot[:, 2, 0], sy)
    roll = torch.atan2(rot[:, 2, 1], rot[:, 2, 2])

    # Fallback for gimbal-lock region
    yaw_s = torch.atan2(-rot[:, 0, 1], rot[:, 1, 1])
    roll_s = torch.zeros_like(roll)
    yaw = torch.where(singular, yaw_s, yaw)
    roll = torch.where(singular, roll_s, roll)

    return torch.rad2deg(torch.stack([yaw, pitch, roll], dim=-1))


def _syn_acc(v: torch.Tensor, smooth_n: int = 5, TARGET_FPS: int = 50) -> torch.Tensor:
    """Synthesize acceleration from vertex positions [N, K, 3]."""
    mid = smooth_n // 2
    scale_factor = TARGET_FPS ** 2

    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * scale_factor for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))

    if mid != 0 and v.shape[0] > smooth_n * 2:
        acc[smooth_n:-smooth_n] = torch.stack(
            [
                (v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * scale_factor / smooth_n ** 2
                for i in range(0, v.shape[0] - smooth_n * 2)
            ]
        )
    return acc


def _compute_angular_velocity_from_rotation(rot_mats: torch.Tensor, fps: float) -> torch.Tensor:
    """Compute angular velocity [N, 3] from rotation matrices [N, 3, 3]."""
    if rot_mats.ndim != 3 or rot_mats.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [N, 3, 3], got {tuple(rot_mats.shape)}")

    n = rot_mats.shape[0]
    if n < 2:
        return torch.zeros((n, 3), dtype=rot_mats.dtype, device=rot_mats.device)

    dt = 1.0 / float(fps)
    R_t = rot_mats[:-1]
    R_t1 = rot_mats[1:]
    R_rel = torch.matmul(R_t.transpose(-1, -2), R_t1)

    R_rel_quat = mat2quat(R_rel)
    R_rel_so3 = pp.SO3(R_rel_quat)
    omega = R_rel_so3.Log() / dt

    out = torch.zeros((n, 3), dtype=rot_mats.dtype, device=rot_mats.device)
    out[:-1] = omega
    out[-1] = omega[-1]
    return out


# def _syn_vel(v: torch.Tensor, fps: float = 50.0) -> torch.Tensor:
#     """
#     v: [T, D]
#     return vel: [T, D]
#     """

#     T = v.shape[0]
#     vel = torch.zeros_like(v)
#     if T < 2:
#         return vel

#     extra_dims = [1] * (v.ndim - 1)
#     dt = torch.full((T - 1,), 1.0 / float(fps), dtype=v.dtype, device=v.device)
#     dt = dt.reshape(T - 1, *extra_dims)
#     dt = torch.clamp(dt, min=1e-6)

#     vel[1:] = (v[1:] - v[:-1]) / dt
#     vel[0] = vel[1]
#     return vel


def _syn_vel(v: torch.Tensor, fps: float = 50.0, smooth_n: int = 5) -> torch.Tensor:
    """
    Synthesize velocity from position by finite difference.

    Args:
        v: [T, D] or [T, J, 3] or generally [T, ...]
        fps: sampling rate
        smooth_n: temporal smoothing kernel size.
                  0 or 1 means no smoothing.

    Returns:
        vel: same shape as v
    """
    if not torch.is_tensor(v):
        raise TypeError(f"Expected torch.Tensor, got {type(v)}")
    if v.ndim < 2:
        raise ValueError(f"Expected v with shape [T, ...], got {v.shape}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    T = v.shape[0]
    vel = torch.zeros_like(v)
    if T < 2:
        return vel

    x = v

    # Optional temporal smoothing along time axis
    if smooth_n is not None and smooth_n > 1:
        if smooth_n % 2 == 0:
            raise ValueError(f"smooth_n should be odd, got {smooth_n}")

        orig_shape = x.shape
        # [T, ...] -> [N_feat, 1, T]
        x_reshape = x.reshape(T, -1).transpose(0, 1).unsqueeze(1)

        pad = smooth_n // 2
        kernel = torch.ones(1, 1, smooth_n, dtype=x.dtype, device=x.device) / smooth_n

        # replicate padding avoids edge shrinkage
        x_pad = torch.nn.functional.pad(x_reshape, (pad, pad), mode="replicate")
        x_smooth = torch.nn.functional.conv1d(x_pad, kernel)

        # back to original shape
        x = x_smooth.squeeze(1).transpose(0, 1).reshape(orig_shape)

    dt = 1.0 / float(fps)

    # central difference for interior, forward/backward at boundaries
    vel[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    vel[0] = (x[1] - x[0]) / dt
    vel[-1] = (x[-1] - x[-2]) / dt

    return vel

def _foot_ground_probs(joint: torch.Tensor) -> torch.Tensor:
    """
    Compute foot-ground contacts from world-frame joint trajectories.

    Args:
        joint: [N, 24, 3] world-frame joint positions.

    Returns:
        [N, 2] int tensor: [left_foot_contact, right_foot_contact].
    """
    if joint.ndim != 3 or joint.shape[1] < 12 or joint.shape[2] != 3:
        raise ValueError(f"Expected joint shape [N, 24, 3]-like, got {tuple(joint.shape)}")

    dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
    dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
    lfoot_contact = (dist_lfeet < 0.008).int()
    rfoot_contact = (dist_rfeet < 0.008).int()
    lfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int, device=joint.device), lfoot_contact))
    rfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int, device=joint.device), rfoot_contact))
    return torch.stack((lfoot_contact, rfoot_contact), dim=1)


def _vertical_motion_labels(
    joint: torch.Tensor,
    foot_vel_threshold: float = 0.008,
    window: int = 10,
    height_threshold: float = 0.15,
    smooth: bool = True,
    smooth_kernel: int = 10,
) -> torch.Tensor:
    """
    Compute binary vertical-motion labels from world-frame support-foot height change.
    """
    if joint.ndim != 3 or joint.shape[1] < 12 or joint.shape[2] != 3:
        raise ValueError(f"Expected joint shape [N, 24, 3]-like, got {tuple(joint.shape)}")

    N = joint.shape[0]
    device = joint.device

    lfy = joint[:, 10, 1]
    rfy = joint[:, 11, 1]

    dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
    dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)

    l_contact = torch.cat([
        torch.zeros(1, dtype=torch.int, device=device),
        (dist_lfeet < foot_vel_threshold).int()
    ])
    r_contact = torch.cat([
        torch.zeros(1, dtype=torch.int, device=device),
        (dist_rfeet < foot_vel_threshold).int()
    ])

    support_y = torch.zeros(N, dtype=joint.dtype, device=device)
    both_contact = (l_contact == 1) & (r_contact == 1)
    left_only = (l_contact == 1) & (r_contact == 0)
    right_only = (l_contact == 0) & (r_contact == 1)
    neither = (l_contact == 0) & (r_contact == 0)

    support_y[left_only] = lfy[left_only]
    support_y[right_only] = rfy[right_only]
    support_y[both_contact] = torch.minimum(lfy[both_contact], rfy[both_contact])

    for i in range(1, N):
        if neither[i]:
            support_y[i] = support_y[i - 1]

    delta_h = torch.zeros(N, dtype=joint.dtype, device=device)
    if N > window:
        delta_h[window:] = torch.abs(support_y[window:] - support_y[:-window])

    labels = (delta_h > height_threshold).int()

    if smooth and smooth_kernel > 1:
        pad = smooth_kernel // 2
        x = labels.float().unsqueeze(0).unsqueeze(0)
        x = F.pad(x, (pad, pad), mode='replicate')
        x = F.avg_pool1d(x, kernel_size=smooth_kernel, stride=1)
        labels = (x.squeeze() > 0.5).int()

    return labels


def _transform_world_vectors(v: torch.Tensor, world_rot: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors [..., 3] with world_rot [3, 3]."""
    return torch.einsum('ij,...j->...i', world_rot, v)


def _transform_world_rotations(rot: torch.Tensor, world_rot: torch.Tensor) -> torch.Tensor:
    """Change world frame for body→world rotations [..., 3, 3] via left multiplication."""
    return torch.einsum('ij,...jk->...ik', world_rot, rot)


class imuposer(Sequence):
    """
    IMUPoser dataset loader for IOPoser with head + hip IMU.

    Loads a single preprocessed IMUPoser .pkl file.
    
    Usage:
        data_root = '/mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz/test'
        data_name = 'imuposer_seq_00011.pkl'
        dataset = imuposer(data_root=data_root, data_name=data_name)
    """

    # Class-level cache to avoid reloading
    _cached_data = {}

    def __init__(
        self,
        data_root: str,  # Directory containing the .pkl file
        data_name: str,  # Name of the .pkl file (e.g., 'imuposer_seq_00011.pkl')
        coordinate: str = "glob_coord",
        gravity: float = 9.81,
        remove_g: bool = False,
        maximum_length: int = 500000000,
        pelvis_relative: bool = True,
        buffer_size: int = 300,  # frames to invalidate at sequence boundaries
        input_coordinate: str = 'glob_coord',  # 'body_coord' or 'glob_coord'
        target_hz: int = 50,
        **kwargs
    ):
        super().__init__()
        # Construct full path to the .pkl file
        filepath = os.path.join(data_root, data_name+".pkl")
        cache_key = filepath

        # Load from cache or disk
        if cache_key not in imuposer._cached_data:
            print(f"Loading IMUPoser data from {filepath}...")
            with open(filepath, 'rb') as f:
                seq_data = pickle.load(f)
            imuposer._cached_data[cache_key] = seq_data
            print(f"Loaded sequence: {seq_data.get('sequence_name', data_name)}")
        
        seq_data = imuposer._cached_data[cache_key]
        # Get sequence info
        num_frames = seq_data['num_samples']
        fps = seq_data['sampling_rate_hz']
        print(f"Total frames: {num_frames}, FPS: {fps}")

        # Extract data from single sequence
        all_data = self._extract_sequence_data(seq_data, pelvis_relative)

        # Resample sequence to target_hz if needed
        if target_hz is not None and abs(float(fps) - float(target_hz)) > 1e-6:
            old_len = num_frames
            new_len = int(round((old_len - 1) * float(target_hz) / float(fps))) + 1
            is_upsample = float(target_hz) > float(fps)

            vec_resample = _resample_nearest if is_upsample else _resample_linear
            rot_resample = _resample_rotmat_nearest if is_upsample else _resample_rotmat_nlerp
            all_data['head_acc'] = vec_resample(all_data['head_acc'], new_len)
            all_data['head_gyro'] = vec_resample(all_data['head_gyro'], new_len)
            all_data['head_vel'] = vec_resample(all_data['head_vel'], new_len)
            all_data['head_ori'] = rot_resample(all_data['head_ori'], new_len)

            all_data['hip_acc'] = vec_resample(all_data['hip_acc'], new_len)
            all_data['hip_gyro'] = vec_resample(all_data['hip_gyro'], new_len)
            all_data['hip_vel'] = vec_resample(all_data['hip_vel'], new_len)
            all_data['hip_ori'] = rot_resample(all_data['hip_ori'], new_len)

            all_data['velocity'] = vec_resample(all_data['velocity'], new_len)
            all_data['position'] = vec_resample(all_data['position'], new_len)
            all_data['joint_pos'] = vec_resample(all_data['joint_pos'], new_len)
            all_data['joint_pos_global'] = vec_resample(all_data['joint_pos_global'], new_len)
            all_data['joint_rot'] = vec_resample(all_data['joint_rot'], new_len)
            all_data['pose'] = rot_resample(all_data['pose'], new_len)
            all_data['contact'] = _resample_nearest(all_data['contact'].float(), new_len).to(all_data['contact'].dtype)

            all_data['real_imu_data']['real_acc'] = vec_resample(all_data['real_imu_data']['real_acc'], new_len)
            all_data['real_imu_data']['real_ori'] = rot_resample(all_data['real_imu_data']['real_ori'], new_len)
            # 0 - left wrist
            # 1 - right wrist
            # 2 - left front pocket
            # 3 - right front pocket
            # 4 - head
            
            # Keep GT orientation time-aligned with other signals
            gt_ori = all_data['gt_orientation']
            if isinstance(gt_ori, pp.LieTensor):
                gt_ori_mat = rot_resample(gt_ori.matrix(), new_len)
                all_data['gt_orientation'] = gt_ori_mat
            elif gt_ori.shape[-2:] == (3, 3):
                all_data['gt_orientation'] = rot_resample(gt_ori, new_len)
            elif gt_ori.shape[-1] == 4:
                q = vec_resample(gt_ori, new_len)
                q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-10)
                all_data['gt_orientation'] = q
            else:
                raise ValueError(f"Unsupported gt_orientation shape for resampling: {tuple(gt_ori.shape)}")

            num_frames = new_len
            fps = float(target_hz)
            print(f"Resampled sequence from {old_len}@{seq_data['sampling_rate_hz']}Hz to {num_frames}@{fps}Hz")

        # Convert head orientation to PyPose SO3 (via quaternion)
        head_ori_quat = mat2quat(all_data['head_ori'])  # [N, 4] in (x, y, z, w)
        head_orientation_so3 = pp.SO3(head_ori_quat)

        # Keep GT orientation from file for comparison plots/debugging
        if isinstance(all_data['gt_orientation'], pp.LieTensor):
            gt_orientation_mat = all_data['gt_orientation'].matrix()
        elif all_data['gt_orientation'].shape[-1] == 4:
            gt_orientation_mat = pp.SO3(all_data['gt_orientation']).matrix()
        elif all_data['gt_orientation'].shape[-2:] == (3, 3):
            gt_orientation_mat = all_data['gt_orientation']
        else:
            raise ValueError(f"Unsupported gt_orientation shape: {tuple(all_data['gt_orientation'].shape)}")

        # Generate time and dt
        time = torch.arange(num_frames, dtype=torch.float32) / fps
        dt = torch.full((num_frames,), 1.0 / fps, dtype=torch.float32)
        # Create mask - all valid for single sequence
        # Note: buffer_size was used for invalidating sequence boundaries when concatenating
        # multiple sequences, but for single-file loading, we don't need it since each
        # sequence is independent. Set all frames as valid.
        mask = torch.ones(num_frames, dtype=torch.bool)

        # Flatten joint data (local convention, aligned with datasets/nymeriadataset.py)
        joint_pos_local = all_data['joint_pos']
        joint_pos_flat_global = all_data['joint_pos_global']
        joint_pos_flat_local = joint_pos_local.reshape(num_frames, -1)  # [N, 72]
        joint_rot_local_6d = matrix_to_rotation_6d(all_data['pose'])
        joint_rot_flat = joint_rot_local_6d.reshape(num_frames, -1)  # [N, 144]

        # if input_coordinate == 'body_coord':
        #     head_acc = all_data['head_acc']
        #     hip_acc = all_data['hip_acc']
            
        #     head_vel = all_data['head_vel']
        #     hip_vel = all_data['hip_vel']
        # else:
        #     head_acc = torch.einsum('nij,nj->ni', all_data['head_ori'], all_data['head_acc'])
        #     hip_acc = torch.einsum('nij,nj->ni', all_data['hip_ori'], all_data['hip_acc'])
    
        #     head_vel = torch.einsum('nij,nj->ni', all_data['head_ori'], all_data['head_vel'])
        #     hip_vel = torch.einsum('nij,nj->ni', all_data['hip_ori'], all_data['hip_vel'])
        
        if input_coordinate == 'glob_coord':
            head_acc = torch.einsum('nij,nj->ni', all_data['head_ori'], all_data['head_acc'])
            hip_acc = torch.einsum('nij,nj->ni', all_data['hip_ori'], all_data['hip_acc'])
            
            head_vel = all_data['head_vel']
            hip_vel = all_data['hip_vel']
        else:
            raise NotImplementedError(f"Input coordinate {input_coordinate} not supported yet. Only 'glob_coord' is implemented.")

        zup_to_cpf_world = ZUP_TO_CPF_WORLD.to(
            device=all_data['real_imu_data']['real_acc'].device,
            dtype=all_data['real_imu_data']['real_acc'].dtype,
        )
        real_head_acc_cpf_world = _transform_world_vectors(all_data['real_imu_data']['real_acc'][:, 4, :], zup_to_cpf_world)
        real_head_ori_cpf_world = _transform_world_rotations(all_data['real_imu_data']['real_ori'][:, 4, :, :], zup_to_cpf_world)
        
        real_hip_acc_cpf_world = _transform_world_vectors(all_data['real_imu_data']['real_acc'][:, 3, :], zup_to_cpf_world)
        real_hip_ori_cpf_world = _transform_world_rotations(all_data['real_imu_data']['real_ori'][:, 3, :, :], zup_to_cpf_world)
        
        # Store data
        self.data = {
            # Head IMU
            'acc': head_acc,                                # [N, 3]
            'gyro': all_data['head_gyro'],                  # [N, 3]
            'head_ori': all_data['head_ori'],               # [N, 3, 3]
            'head_vel': head_vel,                           # [N, 3]
            
            # Hip IMU
            'hip_acc': hip_acc,                             # [N, 3]
            'hip_gyro': all_data['hip_gyro'],               # [N, 3]
            'hip_ori': all_data['hip_ori'],                 # [N, 3, 3]
            'hip_vel': hip_vel,                             # [N, 3]
            
            # Also store hip as right_hip for compatibility
            'right_hip_acc': hip_acc,                       # [N, 3]
            'right_hip_rot': matrix_to_rotation_6d(all_data['hip_ori']), # [N, 6]
            
            # Timing
            'time': time,
            'dt': dt,
            'mask': mask,
            
            # Ground truth
            'velocity': all_data['velocity'],               # [N, 3]
            'gt_translation': all_data['position'],         # [N, 3]
            'gt_orientation': head_orientation_so3,         # SO3 (head orientation)
            'gt_orientation_mat': gt_orientation_mat,       # [N, 3, 3] from gt['orientation']
            
            # Full body
            'joint_pos': joint_pos_flat_local,                    # [N, 72]
            'joint_rot': joint_rot_flat,                    # [N, 144]
            'pose': all_data['pose'],                       # [N, 24, 3, 3]
            'contact': all_data['contact'],                 # [N, 2]
            
            # # Real IMU data for comparison/debugging
            # 'real_head_acc': real_head_acc_cpf_world,  # [N, 3], z-up -> y-up CPF world
            # 'real_head_ori': real_head_ori_cpf_world,  # [N, 3, 3], z-up -> y-up CPF world
            # 'real_right_hip_acc': real_hip_acc_cpf_world,  # [N, 3], z-up -> y-up CPF world
            # 'real_right_hip_ori': real_hip_ori_cpf_world,  # [N, 3, 3], z-up -> y-up CPF world
            
            'real_head_acc': all_data['real_imu_data']['real_acc'][:, 4, :],  # [N, 3], original z-up head acc
            'real_head_ori': all_data['real_imu_data']['real_ori'][:, 4, :, :],  # [N, 3, 3], original z-up head ori
            'real_right_hip_acc': all_data['real_imu_data']['real_acc'][:, 3, :],  # [N, 3], original z-up hip acc
            'real_right_hip_ori': all_data['real_imu_data']['real_ori'][:, 3, :, :],  # [N, 3, 3], original z-up hip ori
            'real_left_hip_acc': all_data['real_imu_data']['real_acc'][:, 2, :],  # [N, 3], original z-up left hip acc
            'real_left_hip_ori': all_data['real_imu_data']['real_ori'][:, 2, :, :],  # [N, 3, 3], original z-up left hip ori
        }

        # Synthesize IMU from pose+translation (same approach as datasets/imu2scenedataset.py)
        try:
            body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))
            pose_local = all_data['pose'].to(dtype=torch.float32, device=torch.device("cpu"))
            body_tran = all_data['position'].to(dtype=torch.float32, device=torch.device("cpu"))

            pose_global, _, vertex_global_trans = body_model.forward_kinematics_masked(
                pose_local,
                tran=body_tran,
                calc_mesh=True,
                vi_mask=VI_MASK_6,
            )

            body_v_acc = _syn_acc(vertex_global_trans, smooth_n=5, TARGET_FPS=int(round(float(fps)))).float()
            body_v_rot = pose_global[:, JI_MASK_6].float()

            self.data['synth_head_acc'] = body_v_acc[:, 4, :].to(head_acc.device)
            self.data['synth_head_ori'] = body_v_rot[:, 4, :, :].to(head_acc.device)
            self.data['synth_head_gyro'] = _compute_angular_velocity_from_rotation(self.data['synth_head_ori'], fps)

            self.data['synth_hip_acc'] = body_v_acc[:, 3, :].to(head_acc.device)
            self.data['synth_hip_ori'] = body_v_rot[:, 3, :, :].to(head_acc.device)
            self.data['synth_hip_gyro'] = _compute_angular_velocity_from_rotation(self.data['synth_hip_ori'], fps)

            print(f"Synthesized head/hip IMU from pose for {num_frames} frames")
        except Exception as e:
            print(f"Warning: failed to synthesize head/hip IMU from pose: {e}")

        # MobilePoser input: [N, 60]
        # Sensor slot format: [acc(3), rotmat(9)] packed into 12 dims
        self.data['mobileposer_input'] = torch.zeros((num_frames, 60), dtype=head_acc.dtype, device=head_acc.device)

        # Head IMU -> slot index 4
        # self.data['mobileposer_input'][:, 12 * 4:12 * 4 + 3] = (self.data['synth_head_acc'] + torch.tensor([0.0, 9.8, 0.0], device=head_acc.device)) / 30.0
        # self.data['mobileposer_input'][:, 12 * 4 + 3:12 * 4 + 12] = self.data['synth_head_ori'].reshape(num_frames, -1)
        # self.data['mobileposer_input'][:, 12 * 4:12 * 4 + 3] = (self.data['real_head_acc'] + torch.tensor([0.0, 9.8, 0.0], device=head_acc.device)) / 30.0
        # self.data['mobileposer_input'][:, 12 * 4 + 3:12 * 4 + 12] = self.data['real_head_ori'].reshape(num_frames, -1)

        # self.data['mobileposer_input'][:, 12 * 4:12 * 4 + 3] = (self.data['real_left_hip_acc']  + torch.tensor([0.0, 9.8, 0.0], device=head_acc.device)) / 30.0
        # self.data['mobileposer_input'][:, 12 * 4 + 3:12 * 4 + 12] =  self.data['real_left_hip_ori'].reshape(num_frames, -1)
        
        # # Hip IMU -> slot index 3
        # self.data['mobileposer_input'][:, 12 * 3:12 * 3 + 3] = self.data['synth_hip_acc'] / 30.0
        # self.data['mobileposer_input'][:, 12 * 3 + 3:12 * 3 + 12] = self.data['synth_hip_ori'].reshape(num_frames, -1)
        self.data['mobileposer_input'][:, 12 * 3:12 * 3 + 3] = self.data['real_right_hip_acc'] / 30.0
        self.data['mobileposer_input'][:, 12 * 3 + 3:12 * 3 + 12] = self.data['real_right_hip_ori'].reshape(num_frames, -1)
        
        # # Apply coordinate transform if needed
        # if coordinate == "glob_coord":
        #     # Transform velocity to world frame
        #     self.data['velocity'] = head_orientation_so3 @ self.data['velocity']
        #     # Transform acceleration to world frame
        #     self.data['acc'] = head_orientation_so3 @ self.data['acc']

        # Compatibility keys for datasets/dataset_motion.py
        # Use meaningful aliases when available and safe zero defaults otherwise.
        joint_world = all_data['joint_pos_global']

        if 'joint_vel' not in self.data:
            joint_vel = _syn_vel(joint_pos_flat_global, fps=float(fps))
            self.data['joint_vel'] = joint_vel.flatten(start_dim=1)  # [N, 72]
        if 'foot_contact' not in self.data:
            self.data['foot_contact'] = _foot_ground_probs(joint_world).to(dtype=torch.int)
        if 'updown_label' not in self.data:
            self.data['updown_label'] = _vertical_motion_labels(joint_world).to(dtype=torch.int)

        if 'pelvis_vel' not in self.data:
            self.data['pelvis_vel'] = self.data['velocity']
        if 'pocket_vel' not in self.data:
            self.data['pocket_vel'] = hip_vel

        if 'right_elbow_acc' not in self.data:
            self.data['right_elbow_acc'] = torch.zeros((num_frames, 3), dtype=head_acc.dtype, device=head_acc.device)
        if 'right_elbow_rot' not in self.data:
            self.data['right_elbow_rot'] = torch.zeros((num_frames, 6), dtype=head_acc.dtype, device=head_acc.device)
        if 'right_elbow_gyro' not in self.data:
            self.data['right_elbow_gyro'] = torch.zeros((num_frames, 3), dtype=head_acc.dtype, device=head_acc.device)

        self.coordinate = coordinate

        # Truncate if needed
        if self.get_length() > maximum_length:
            for key in self.data:
                if isinstance(self.data[key], (torch.Tensor, pp.LieTensor)):
                    self.data[key] = self.data[key][:maximum_length]

        print(f"\n=== Loaded IMUPoser sequence: {data_name} ===")
        print(f"  Total frames: {self.get_length()}")
        print(f"  FPS: {fps}")
        print(f"  Coordinate: {coordinate}")
        print(f"  Pelvis-relative joint_pos: {pelvis_relative}")
        print(f"  Has hip IMU: True")

    def _extract_sequence_data(self, seq_data, pelvis_relative: bool):
        """Extract data from a single sequence dict."""
        imu = seq_data['imu_data']
        gt = seq_data['gt_data']
        
        # Get joint positions
        joint_pos = gt['joint_pos'].clone()  # [N, 24, 3]
        
        # Convert to pelvis-relative coordinates
        if pelvis_relative:
            pelvis_pos = joint_pos[:, 0:1, :]  # [N, 1, 3]
            joint_pos = joint_pos - pelvis_pos  # [N, 24, 3]

        return {
            # Head IMU
            'head_acc': imu['head_acc'],
            'head_gyro': imu['head_gyro'],
            'head_ori': imu['head_ori'],
            'head_vel': imu['head_vel'],
            
            # Hip IMU
            'hip_acc': imu['hip_acc'],
            'hip_gyro': imu['hip_gyro'],
            'hip_ori': imu['hip_ori'],
            'hip_vel': imu['hip_vel'],
            
            # Ground truth
            'velocity': gt['velocity'],
            'gt_orientation': gt['orientation'],
            'position': gt['position'],
            'joint_pos': joint_pos,
            'joint_pos_global': gt['joint_pos'],  # Keep global joint positions for compatibility
            'joint_rot': gt['joint_rot_6d'],
            'pose': gt['pose'],
            'contact': gt['contact'],
            
            'real_imu_data': {
                'real_acc': seq_data['real_imu_data']['real_acc'],
                'real_ori': seq_data['real_imu_data']['real_ori'],        
            }
        }

    def get_length(self):
        """Number of frames in the sequence"""
        return self.data['acc'].shape[0]


# ===========================================================================
# IMUPoser right-wrist FLU sliding-window dataset
# ===========================================================================
# Mirrors nymeria_wrist_loader_FLU.py but sourced from IMUPoser pkl files.
#
# IMUPoser world frame: Z-up  (confirmed by ZUP_TO_CPF_WORLD constant above)
#   → world_up = [0, 0, 1]
#
# SMPL right-arm joints used:
#   19 – R_Elbow   (elbow proxy, arm direction)
#   21 – R_Wrist   (IMU placement, position + global rotation)
#   23 – R_Hand    (finger-root proxy, forward direction)
#
# FLU body frame (right wrist watch):
#   F (+X) – toward fingertips  (wrist→hand)
#   L (+Y) – thumb side          (cross(U, F))
#   U (+Z) – dorsal / watch-face  (palm normal, toward sky when arm horizontal)
#
# Outputs per window:
#   x : [6, T]         – acc_flu + gyro_flu  (linear, no gravity)
#   y : [N_patches, 30] – 6D rotations for right arm chain [14,17,19,21,23]
# ===========================================================================

_SMPL_R_ELBOW     = 19
_SMPL_R_WRIST     = 21
_SMPL_R_HAND      = 23
_RIGHT_ARM_CHAIN  = [14, 17, 19, 21, 23]   # same as nymeria_wrist_loader_FLU

_IMUPOSER_WORLD_UP = None   # resolved lazily to np.array([0.,0.,1.])
_MIN_CROSS_MAG_IP  = None   # resolved lazily


def _imuposer_world_up() -> "np.ndarray":
    import numpy as _np
    return _np.array([0.0, 0.0, 1.0], dtype=_np.float64)   # Z-up


def _build_flu_axes_imuposer(
    f: "np.ndarray",
    elbow_to_wrist: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Orthonormal FLU axes for Z-up world frame (IMUPoser convention).

    Forward  = f (wrist → hand, already unit-length)
    Up       = palm normal toward sky (dorsal), u·[0,0,1] ≥ 0
    Left     = cross(U, F)
    """
    import numpy as _np
    world_up = _imuposer_world_up()
    min_cross = _np.sin(_np.radians(10.0))

    cross_fv = _np.cross(f, elbow_to_wrist)
    mag = _np.linalg.norm(cross_fv)
    if mag < min_cross:
        cross_fv = _np.cross(f, world_up)
        mag = _np.linalg.norm(cross_fv)
        if mag < 1e-6:
            cross_fv = _np.cross(f, _np.array([1.0, 0.0, 0.0]))
            mag = _np.linalg.norm(cross_fv)

    u = cross_fv / mag
    if _np.dot(u, world_up) < 0:
        u = -u

    l = _np.cross(u, f)
    l = l / (_np.linalg.norm(l) + 1e-8)
    f = _np.cross(l, u)
    f = f / (_np.linalg.norm(f) + 1e-8)

    return f, l, u


def _compute_flu_per_frame_imuposer(
    joint_pos_world: "np.ndarray",   # [N, 24, 3]  Z-up world
) -> "np.ndarray":                   # [N, 3, 3]   R_world_from_flu
    """Per-frame FLU rotation matrices from SMPL joint positions."""
    import numpy as _np

    N = joint_pos_world.shape[0]

    elbow = joint_pos_world[:, _SMPL_R_ELBOW].astype(_np.float64)  # [N, 3]
    wrist = joint_pos_world[:, _SMPL_R_WRIST].astype(_np.float64)  # [N, 3]
    hand  = joint_pos_world[:, _SMPL_R_HAND ].astype(_np.float64)  # [N, 3]

    f_raw = hand - wrist
    f_mag = _np.linalg.norm(f_raw, axis=1, keepdims=True)
    f_vec = f_raw / _np.maximum(f_mag, 1e-8)     # [N, 3]

    v_raw = wrist - elbow
    v_mag = _np.linalg.norm(v_raw, axis=1, keepdims=True)
    v_vec = v_raw / _np.maximum(v_mag, 1e-8)     # [N, 3]

    R_out = _np.empty((N, 3, 3), dtype=_np.float64)
    for t in range(N):
        f, l, u = _build_flu_axes_imuposer(f_vec[t], v_vec[t])
        R_out[t] = _np.stack([f, l, u], axis=1)  # columns = FLU axes

    return R_out.astype(_np.float32)              # [N, 3, 3]


def _load_imuposer_session_flu(
    pkl_path: str,
    target_hz: int = 50,
) -> "tuple[np.ndarray, np.ndarray] | None":
    """Load one IMUPoser pkl and return (X, y) in FLU frame.

    Parameters
    ----------
    pkl_path : str
        Path to an IMUPoser ``.pkl`` sequence file.
    target_hz : int
        Target sampling rate; if the file differs, data is resampled.

    Returns
    -------
    X : [N, 6]    float32  – acc_flu (linear) + gyro_flu
    y : [N, 5, 6] float32  – 6D rotations for SMPL joints [14,17,19,21,23]
    None on failure.
    """
    import numpy as _np
    import logging as _logging
    from scipy.spatial.transform import Rotation as ScipyRot
    import core.articulate as art
    from core.paths import Paths

    _log = _logging.getLogger(__name__)
    _log.info("[imuposer_wrist_flu] loading: %s", os.path.basename(pkl_path))

    try:
        with open(pkl_path, "rb") as fh:
            seq_data = pickle.load(fh)
    except Exception as exc:
        _log.warning("[imuposer_wrist_flu] cannot open %s: %s", pkl_path, exc)
        return None

    gt   = seq_data.get("gt_data", {})
    pose_local   = gt.get("pose")           # [N, 24, 3, 3]  local rotations
    joint_pos_w  = gt.get("joint_pos")      # [N, 24, 3]     Z-up world
    position     = gt.get("position")       # [N, 3]         root translation

    if pose_local is None or joint_pos_w is None or position is None:
        _log.warning("[imuposer_wrist_flu] %s: missing gt fields.", pkl_path)
        return None

    # Convert to tensors
    if not torch.is_tensor(pose_local):
        pose_local  = torch.as_tensor(pose_local,  dtype=torch.float32)
    if not torch.is_tensor(joint_pos_w):
        joint_pos_w = torch.as_tensor(joint_pos_w, dtype=torch.float32)
    if not torch.is_tensor(position):
        position = torch.as_tensor(position, dtype=torch.float32)

    fps_raw = float(seq_data.get("sampling_rate_hz", target_hz))
    N       = pose_local.shape[0]

    if N < 3:
        _log.warning("[imuposer_wrist_flu] %s: too short (%d frames).", pkl_path, N)
        return None

    # -----------------------------------------------------------------------
    # Resample to target_hz if needed
    # -----------------------------------------------------------------------
    if abs(fps_raw - target_hz) > 1e-3:
        new_len    = int(round((N - 1) * target_hz / fps_raw)) + 1
        up         = target_hz > fps_raw
        pose_local  = (_resample_rotmat_nearest if up else _resample_rotmat_nlerp)(pose_local, new_len)
        joint_pos_w = (_resample_nearest        if up else _resample_linear      )(joint_pos_w, new_len)
        position    = (_resample_nearest        if up else _resample_linear      )(position, new_len)
        N           = new_len
        _log.info("[imuposer_wrist_flu] resampled %d→%d frames (%.0f→%.0f Hz)",
                  pose_local.shape[0], N, fps_raw, target_hz)

    fps = float(target_hz)

    # -----------------------------------------------------------------------
    # FK: local → global rotation matrices  [N, 24, 3, 3]
    # -----------------------------------------------------------------------
    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))
    pose_global, _ = body_model.forward_kinematics_masked(
        pose_local,
        tran=position,
        calc_mesh=False,
        vi_mask=torch.tensor([], dtype=torch.long),
    )                                                   # [N, 24, 3, 3]

    jpos = joint_pos_w.numpy().astype(_np.float64)     # [N, 24, 3]
    R_global_wrist = pose_global[:, _SMPL_R_WRIST].numpy().astype(_np.float64)   # [N, 3, 3]

    # -----------------------------------------------------------------------
    # Linear acceleration of R_Wrist (second-order finite differences, no gravity)
    # -----------------------------------------------------------------------
    pos_wrist = jpos[:, _SMPL_R_WRIST]                 # [N, 3]
    a_world   = _np.zeros((N, 3), dtype=_np.float64)
    a_world[1:-1] = (pos_wrist[2:] - 2 * pos_wrist[1:-1] + pos_wrist[:-2]) * fps ** 2
    a_world[0]    = a_world[1]
    a_world[-1]   = a_world[-2]

    # -----------------------------------------------------------------------
    # Angular velocity of R_Wrist in its own local frame
    # -----------------------------------------------------------------------
    dR = _np.einsum("nij,njk->nik",
                    R_global_wrist[:-1].transpose(0, 2, 1),
                    R_global_wrist[1:])                 # [N-1, 3, 3]
    omega_raw = (
        ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
        .as_rotvec()
        .astype(_np.float64) * fps
    )
    omega_wrist = _np.zeros((N, 3), dtype=_np.float64)
    omega_wrist[:-1] = omega_raw
    omega_wrist[-1]  = omega_raw[-1]

    # -----------------------------------------------------------------------
    # FLU frame (per-frame, from SMPL joint positions)
    # -----------------------------------------------------------------------
    R_wf = _compute_flu_per_frame_imuposer(jpos).astype(_np.float64)   # [N, 3, 3] world_from_flu
    R_fw = R_wf.transpose(0, 2, 1)                                       # R_flu_from_world

    acc_flu  = _np.einsum("nij,nj->ni", R_fw, a_world)                             # [N, 3]
    R_flu_from_wrist = _np.einsum("nij,njk->nik", R_fw, R_global_wrist)            # [N, 3, 3]
    gyro_flu = _np.einsum("nij,nj->ni", R_flu_from_wrist, omega_wrist)             # [N, 3]

    X = _np.concatenate([acc_flu, gyro_flu], axis=1).astype(_np.float32)           # [N, 6]

    # -----------------------------------------------------------------------
    # Labels: 6D rotation for right arm chain [14, 17, 19, 21, 23] (local)
    # -----------------------------------------------------------------------
    chain_R = pose_local[:, _RIGHT_ARM_CHAIN]                    # [N, 5, 3, 3]
    y = matrix_to_rotation_6d(chain_R).numpy().astype(_np.float32)   # [N, 5, 6]

    _log.info(
        "[imuposer_wrist_flu] loaded %s: N=%d  fps=%.0f",
        os.path.basename(pkl_path), N, fps,
    )
    return X, y


# ---------------------------------------------------------------------------
# Sliding-window Dataset
# ---------------------------------------------------------------------------

class IMUPoserWristFLUDataset(torch.utils.data.Dataset):
    """Sliding-window dataset over IMUPoser sessions with right-wrist FLU IMU.

    Each item:
        x : [6, T]           float32  – (acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U)
        y : [N_patches, 30]  float32  – right-arm 6D rotations per time patch
    """

    def __init__(
        self,
        pkl_paths: "list[str]",
        cfg: dict,
        split_name: str = "train",
        target_hz: int = 50,
    ):
        from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
        from imu_lm.data.augmentations.transform import apply_augment
        from imu_lm.data.windowing import compute_T_and_hop
        from imu_lm.utils.helpers import cfg_get
        import numpy as _np

        self.cfg        = cfg
        self.split_name = split_name
        self.stats      = PreprocessStats()

        vit_cfg         = cfg_get(cfg, ["vit1d"], {}) or {}
        self._patch_size     = int(vit_cfg.get("patch_size", 16))
        self._T, self._hop   = compute_T_and_hop(cfg)
        self._num_patches    = self._T // self._patch_size
        assert self._T % self._patch_size == 0, \
            f"T={self._T} must be divisible by patch_size={self._patch_size}"

        # Load all sessions
        self._sessions: list[tuple["np.ndarray", "np.ndarray"]] = []
        self._sess_idx: list[int]  = []
        self._starts:   list[int]  = []

        for pkl in tqdm(pkl_paths, desc=f"[imuposer_wrist_flu] loading {split_name}"):
            result = _load_imuposer_session_flu(pkl, target_hz=target_hz)
            if result is None:
                continue
            X, y = result       # X: [N, 6], y: [N, 5, 6]
            if X.shape[0] < self._T:
                continue

            si = len(self._sessions)
            self._sessions.append((X, y))

            n_windows = (X.shape[0] - self._T) // self._hop + 1
            for s in range(n_windows):
                self._sess_idx.append(si)
                self._starts.append(s * self._hop)

        self._sess_idx = _np.array(self._sess_idx, dtype=_np.int32)
        self._starts   = _np.array(self._starts,   dtype=_np.int32)

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int):
        from imu_lm.data.augmentations.preprocess import preprocess_window
        from imu_lm.data.augmentations.transform import apply_augment
        import numpy as _np

        si    = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        T     = self._T
        X, y  = self._sessions[si]         # [N, 6], [N, 5, 6]

        Xw = X[start : start + T]          # [T, 6]

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        Xproc = apply_augment(Xproc.T, self.cfg).T    # [C, T] → augment(T,C) → [C, T]
        x_tensor = torch.from_numpy(Xproc).float()    # [6, T] channels-first

        P = self._patch_size
        patch_labels = _np.empty(
            (self._num_patches, len(_RIGHT_ARM_CHAIN) * 6), dtype=_np.float32
        )
        for p in range(self._num_patches):
            frame = start + p * P + P // 2
            patch_labels[p] = y[frame].reshape(-1)    # [5, 6] → [30]

        y_tensor = torch.from_numpy(patch_labels).float()   # [N_patches, 30]
        return x_tensor, y_tensor


def collate_skip_none_imuposer(batch):
    """Drop None entries (bad windows) and stack the rest."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_imuposer_wrist_flu_loaders(
    cfg: dict,
    train_files: "list[str]",
    val_files:   "list[str] | None" = None,
    test_files:  "list[str] | None" = None,
    target_hz:   int = 50,
) -> "dict[str, torch.utils.data.DataLoader]":
    """Build train/val/test DataLoaders with right-wrist FLU IMU from IMUPoser.

    Each batch yields::

        x : [B, 6, T]            float32  – right-wrist IMU in FLU body frame
        y : [B, N_patches, 30]   float32  – right-arm 6D rotations per patch

    Parameters
    ----------
    cfg : dict              Config dict (base.yaml structure).
    train_files : list[str] Absolute paths to training pkl files.
    val_files   : list[str] Absolute paths to validation pkl files (optional).
    test_files  : list[str] Absolute paths to test pkl files (optional).
    target_hz   : int       Resample all sequences to this rate (default 50).
    """
    from imu_lm.utils.helpers import cfg_get

    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],      256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],       0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],    False))
    persistent      = num_workers > 0

    def _make(name, files, bs, shuffle):
        if not files:
            return None
        ds = IMUPoserWristFLUDataset(
            pkl_paths=files, cfg=cfg, split_name=name, target_hz=target_hz
        )
        return torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=(shuffle), persistent_workers=persistent,
            collate_fn=collate_skip_none_imuposer,
        )

    loaders = {}
    train_loader = _make("train", train_files,        batch_size,      True)
    val_loader   = _make("val",   val_files or [],    eval_batch_size, False)
    test_loader  = _make("test",  test_files or [],   eval_batch_size, False)

    if train_loader: loaders["train_loader"] = train_loader
    if val_loader:   loaders["val_loader"]   = val_loader
    if test_loader:  loaders["test_loader"]  = test_loader
    return loaders


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Example usage
    data_root = '/mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test'
    data_name = "imuposer_seq_00013"
    
    dataset = imuposer(
        data_root=data_root,
        data_name=data_name, 
        coordinate='glob_coord',
        gravity=9.81,
        remove_g=False,
        maximum_length=500000000,
        pelvis_relative=True
    )

    print("\nDataset length (frames):", dataset.get_length())
    print("Sample keys:", dataset.data.keys())
    print("\n=== Data shapes ===")
    for key, val in dataset.data.items():
        if hasattr(val, 'shape'):
            print(f"  {key}: {val.shape}")
        elif hasattr(val, 'lshape'):
            print(f"  {key}: {val.lshape} (SO3)")

    # Print velocity preview
    velocity = dataset.data['velocity']
    print("\n=== Velocity Preview ===")
    print(f"velocity shape: {velocity.shape}")
    print("first 5 frames:")
    print(velocity[:5])
    print("last 5 frames:")
    print(velocity[-5:])

    # Plot velocity and save
    velocity_np = velocity.cpu().numpy()
    time_np = dataset.data['time'].cpu().numpy()

    fig_vel, ax_vel = plt.subplots(figsize=(12, 4))
    ax_vel.plot(time_np, velocity_np[:, 0], 'r-', label='vx', linewidth=0.8)
    ax_vel.plot(time_np, velocity_np[:, 1], 'g-', label='vy', linewidth=0.8)
    ax_vel.plot(time_np, velocity_np[:, 2], 'b-', label='vz', linewidth=0.8)
    ax_vel.set_title('Velocity vs Time')
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.legend()
    ax_vel.grid(True, alpha=0.3)
    fig_vel.tight_layout()

    vis_dir = "data_vis"
    os.makedirs(vis_dir, exist_ok=True)
    velocity_plot_path = os.path.join(vis_dir, f'{data_name}_velocity.png')
    fig_vel.savefig(velocity_plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved velocity plot to {velocity_plot_path}")
    plt.close(fig_vel)

    # Compare joint_vel[:, 3*15:3*15+3] with global velocity components
    if 'joint_vel' in dataset.data and 'velocity' in dataset.data:
        joint_vel_15 = dataset.data['joint_vel'][:, 3 * 0:3 * 0 + 3].cpu().numpy()  # [N, 3]
        gt_velocity = dataset.data['velocity'].cpu().numpy()                            # [N, 3]

        n_cmp = min(time_np.shape[0], joint_vel_15.shape[0], gt_velocity.shape[0])
        t_cmp = time_np[:n_cmp]
        fig_jv_cmp, axes_jv_cmp = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes_jv_cmp[0].plot(t_cmp, joint_vel_15[:n_cmp, 0], 'r-', linewidth=1.0, label='joint15 vel x')
        axes_jv_cmp[0].plot(t_cmp, joint_vel_15[:n_cmp, 1], 'g-', linewidth=1.0, label='joint15 vel y')
        axes_jv_cmp[0].plot(t_cmp, joint_vel_15[:n_cmp, 2], 'b-', linewidth=1.0, label='joint15 vel z')
        axes_jv_cmp[0].set_ylabel('joint15 velocity (m/s)')
        axes_jv_cmp[0].set_title('Joint 15 Velocity Components')
        axes_jv_cmp[0].legend()
        axes_jv_cmp[0].grid(True, alpha=0.3)

        axes_jv_cmp[1].plot(t_cmp, gt_velocity[:n_cmp, 0], 'r-', linewidth=1.0, label='velocity x')
        axes_jv_cmp[1].plot(t_cmp, gt_velocity[:n_cmp, 1], 'g-', linewidth=1.0, label='velocity y')
        axes_jv_cmp[1].plot(t_cmp, gt_velocity[:n_cmp, 2], 'b-', linewidth=1.0, label='velocity z')
        axes_jv_cmp[1].set_xlabel('Time (s)')
        axes_jv_cmp[1].set_ylabel('velocity (m/s)')
        axes_jv_cmp[1].set_title('GT Velocity Components')
        axes_jv_cmp[1].legend()
        axes_jv_cmp[1].grid(True, alpha=0.3)

        fig_jv_cmp.suptitle(f'joint_vel[:, 45:48] vs GT velocity - {data_name}', fontsize=13, fontweight='bold')
        fig_jv_cmp.tight_layout()

        jv_cmp_path = os.path.join(vis_dir, f'{data_name}_joint_vel15_vs_velocity.png')
        fig_jv_cmp.savefig(jv_cmp_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved joint_vel-vs-velocity comparison to {jv_cmp_path}")
        plt.close(fig_jv_cmp)
    else:
        print("Skipping joint_vel-vs-velocity plot: required keys are unavailable.")

    # Plot IMU signals: head/hip acceleration and orientation matrices
    head_acc = dataset.data['acc'].cpu().numpy()                # [N, 3]
    head_ori = dataset.data['head_ori'].cpu().numpy()           # [N, 3, 3]
    hip_acc = dataset.data['hip_acc'].cpu().numpy()             # [N, 3]
    hip_ori = dataset.data['hip_ori'].cpu().numpy()             # [N, 3, 3]

    vis_dir = "data_vis"
    os.makedirs(vis_dir, exist_ok=True)

    fig_imu, axes_imu = plt.subplots(2, 2, figsize=(14, 9))

    axes_imu[0, 0].plot(head_acc)
    axes_imu[0, 0].set_title('Head Acceleration')
    axes_imu[0, 0].set_xlabel('Frame')
    axes_imu[0, 0].set_ylabel('m/s²')
    axes_imu[0, 0].legend(['x', 'y', 'z'])
    axes_imu[0, 0].grid(True, alpha=0.3)

    axes_imu[0, 1].plot(head_ori.reshape(head_ori.shape[0], -1))
    axes_imu[0, 1].set_title('Head Orientation (3x3 flattened)')
    axes_imu[0, 1].set_xlabel('Frame')
    axes_imu[0, 1].set_ylabel('Rotation matrix value')
    axes_imu[0, 1].grid(True, alpha=0.3)

    axes_imu[1, 0].plot(hip_acc)
    axes_imu[1, 0].set_title('Hip Acceleration')
    axes_imu[1, 0].set_xlabel('Frame')
    axes_imu[1, 0].set_ylabel('m/s²')
    axes_imu[1, 0].legend(['x', 'y', 'z'])
    axes_imu[1, 0].grid(True, alpha=0.3)

    axes_imu[1, 1].plot(hip_ori.reshape(hip_ori.shape[0], -1))
    axes_imu[1, 1].set_title('Hip Orientation (3x3 flattened)')
    axes_imu[1, 1].set_xlabel('Frame')
    axes_imu[1, 1].set_ylabel('Rotation matrix value')
    axes_imu[1, 1].grid(True, alpha=0.3)

    fig_imu.suptitle(f'IMU Signals - {data_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    imu_plot_path = os.path.join(vis_dir, f'{data_name}_imu_signals.png')
    plt.savefig(imu_plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved IMU signal visualization to {imu_plot_path}")
    plt.close(fig_imu)

    # Compare real IMU vs processed IMU (acc + orientation YPR)
    if all(k in dataset.data for k in ['real_head_acc', 'real_head_ori', 'real_right_hip_acc', 'real_right_hip_ori']):
        real_head_acc = dataset.data['real_head_acc'].cpu().numpy()
        real_hip_acc = dataset.data['real_right_hip_acc'].cpu().numpy()

        proc_head_ori = dataset.data['head_ori'].cpu()
        proc_hip_ori = dataset.data['hip_ori'].cpu()
        real_head_ori = dataset.data['real_head_ori'].cpu()
        real_hip_ori = dataset.data['real_right_hip_ori'].cpu()

        proc_head_ypr = _rotmat_to_euler_zyx_deg(proc_head_ori).numpy()
        real_head_ypr = _rotmat_to_euler_zyx_deg(real_head_ori).numpy()
        proc_hip_ypr = _rotmat_to_euler_zyx_deg(proc_hip_ori).numpy()
        real_hip_ypr = _rotmat_to_euler_zyx_deg(real_hip_ori).numpy()

        n_real_cmp = min(
            len(time_np),
            head_acc.shape[0], hip_acc.shape[0],
            real_head_acc.shape[0], real_hip_acc.shape[0],
            proc_head_ypr.shape[0], real_head_ypr.shape[0],
            proc_hip_ypr.shape[0], real_hip_ypr.shape[0],
        )
        t_real_cmp = time_np[:n_real_cmp]

        fig_real_cmp, axes_real_cmp = plt.subplots(4, 3, figsize=(18, 14), sharex=True)

        axis_names = ['x', 'y', 'z']
        ypr_names = ['yaw', 'pitch', 'roll']

        # Row 0: Head acceleration x/y/z
        for j in range(3):
            ax = axes_real_cmp[0, j]
            ax.plot(t_real_cmp, head_acc[:n_real_cmp, j], 'b-', linewidth=1.0, label='processed')
            ax.plot(t_real_cmp, real_head_acc[:n_real_cmp, j], 'm--', linewidth=1.0, label='real')
            ax.set_title(f'Head Acc {axis_names[j]}')
            ax.set_ylabel('m/s²')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Row 1: Head orientation yaw/pitch/roll
        for j in range(3):
            ax = axes_real_cmp[1, j]
            ax.plot(t_real_cmp, proc_head_ypr[:n_real_cmp, j], 'b-', linewidth=1.0, label='processed')
            ax.plot(t_real_cmp, real_head_ypr[:n_real_cmp, j], 'm--', linewidth=1.0, label='real')
            ax.set_title(f'Head {ypr_names[j]}')
            ax.set_ylabel('deg')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Row 2: Hip acceleration x/y/z
        for j in range(3):
            ax = axes_real_cmp[2, j]
            ax.plot(t_real_cmp, hip_acc[:n_real_cmp, j], 'b-', linewidth=1.0, label='processed')
            ax.plot(t_real_cmp, real_hip_acc[:n_real_cmp, j], 'm--', linewidth=1.0, label='real')
            ax.set_title(f'Hip Acc {axis_names[j]}')
            ax.set_ylabel('m/s²')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Row 3: Hip orientation yaw/pitch/roll
        for j in range(3):
            ax = axes_real_cmp[3, j]
            ax.plot(t_real_cmp, proc_hip_ypr[:n_real_cmp, j], 'b-', linewidth=1.0, label='processed')
            ax.plot(t_real_cmp, real_hip_ypr[:n_real_cmp, j], 'm--', linewidth=1.0, label='real')
            ax.set_title(f'Hip {ypr_names[j]}')
            ax.set_ylabel('deg')
            ax.set_xlabel('Time (s)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        fig_real_cmp.suptitle(f'Real vs Processed IMU (Head/Hip) - {data_name}', fontsize=14, fontweight='bold')
        fig_real_cmp.tight_layout()

        real_cmp_path = os.path.join(vis_dir, f'{data_name}_real_vs_processed_imu.png')
        fig_real_cmp.savefig(real_cmp_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved real-vs-processed IMU comparison to {real_cmp_path}")
        plt.close(fig_real_cmp)
    else:
        print("Skipping real-vs-processed plot: real IMU keys are unavailable.")

    # Compare GT orientation vs hip orientation in yaw/pitch/roll
    gt_ori_mat = dataset.data['gt_orientation_mat'].cpu()
    hip_ori_mat = dataset.data['hip_ori'].cpu()

    gt_euler = _rotmat_to_euler_zyx_deg(gt_ori_mat).numpy()   # [N, 3]
    hip_euler = _rotmat_to_euler_zyx_deg(hip_ori_mat).numpy() # [N, 3]

    n_cmp = min(time_np.shape[0], gt_euler.shape[0], hip_euler.shape[0])
    t_cmp = time_np[:n_cmp]
    gt_euler = gt_euler[:n_cmp]
    hip_euler = hip_euler[:n_cmp]

    fig_cmp, axes_cmp = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    labels = ['Yaw (deg)', 'Pitch (deg)', 'Roll (deg)']
    for i in range(3):
        axes_cmp[i].plot(t_cmp, gt_euler[:, i], 'k-', linewidth=1.0, label='GT orientation')
        axes_cmp[i].plot(t_cmp, hip_euler[:, i], 'c--', linewidth=1.0, label='Hip IMU orientation')
        axes_cmp[i].set_ylabel(labels[i])
        axes_cmp[i].grid(True, alpha=0.3)
        axes_cmp[i].legend(loc='upper right')

    axes_cmp[-1].set_xlabel('Time (s)')
    fig_cmp.suptitle(f'GT vs Hip Orientation (Yaw/Pitch/Roll) - {data_name}', fontsize=14, fontweight='bold')
    fig_cmp.tight_layout()

    cmp_plot_path = os.path.join(vis_dir, f'{data_name}_gt_vs_hip_ypr.png')
    fig_cmp.savefig(cmp_plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved GT vs Hip Yaw/Pitch/Roll comparison to {cmp_plot_path}")
    plt.close(fig_cmp)

    # Compare synthesized vs measured head/hip IMU (acc + orientation YPR)
    if all(k in dataset.data for k in ['synth_head_acc', 'synth_head_ori', 'synth_hip_acc', 'synth_hip_ori']):
        synth_head_acc = dataset.data['synth_head_acc'].cpu().numpy()
        synth_hip_acc = dataset.data['synth_hip_acc'].cpu().numpy()
        synth_head_ori = dataset.data['synth_head_ori'].cpu()
        synth_hip_ori = dataset.data['synth_hip_ori'].cpu()

        meas_head_ori = dataset.data['head_ori'].cpu()
        meas_hip_ori = dataset.data['hip_ori'].cpu()

        meas_head_ypr = _rotmat_to_euler_zyx_deg(meas_head_ori).numpy()
        synth_head_ypr = _rotmat_to_euler_zyx_deg(synth_head_ori).numpy()
        meas_hip_ypr = _rotmat_to_euler_zyx_deg(meas_hip_ori).numpy()
        synth_hip_ypr = _rotmat_to_euler_zyx_deg(synth_hip_ori).numpy()

        n_syn = min(
            len(time_np),
            head_acc.shape[0], hip_acc.shape[0],
            synth_head_acc.shape[0], synth_hip_acc.shape[0],
            meas_head_ypr.shape[0], synth_head_ypr.shape[0],
            meas_hip_ypr.shape[0], synth_hip_ypr.shape[0],
        )
        t_syn = time_np[:n_syn]

        fig_syn, axes_syn = plt.subplots(4, 3, figsize=(18, 14), sharex=True)

        axis_names = ['x', 'y', 'z']
        ypr_names = ['yaw', 'pitch', 'roll']

        # Row 0: Head acceleration x/y/z (separate subplots)
        for j in range(3):
            ax = axes_syn[0, j]
            ax.plot(t_syn, head_acc[:n_syn, j], 'b-', linewidth=1.0, label='measured')
            ax.plot(t_syn, synth_head_acc[:n_syn, j], 'r--', linewidth=1.0, label='synthesized')
            ax.set_title(f'Head Acc {axis_names[j]}')
            ax.set_ylabel('m/s²')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Row 1: Head orientation yaw/pitch/roll (separate subplots)
        for j in range(3):
            ax = axes_syn[1, j]
            ax.plot(t_syn, meas_head_ypr[:n_syn, j], 'b-', linewidth=1.0, label='measured')
            ax.plot(t_syn, synth_head_ypr[:n_syn, j], 'r--', linewidth=1.0, label='synthesized')
            ax.set_title(f'Head {ypr_names[j]}')
            ax.set_ylabel('deg')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Row 2: Hip acceleration x/y/z (separate subplots)
        for j in range(3):
            ax = axes_syn[2, j]
            ax.plot(t_syn, hip_acc[:n_syn, j], 'b-', linewidth=1.0, label='measured')
            ax.plot(t_syn, synth_hip_acc[:n_syn, j], 'r--', linewidth=1.0, label='synthesized')
            ax.set_title(f'Hip Acc {axis_names[j]}')
            ax.set_ylabel('m/s²')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Row 3: Hip orientation yaw/pitch/roll (separate subplots)
        for j in range(3):
            ax = axes_syn[3, j]
            ax.plot(t_syn, meas_hip_ypr[:n_syn, j], 'b-', linewidth=1.0, label='measured')
            ax.plot(t_syn, synth_hip_ypr[:n_syn, j], 'r--', linewidth=1.0, label='synthesized')
            ax.set_title(f'Hip {ypr_names[j]}')
            ax.set_ylabel('deg')
            ax.set_xlabel('Time (s)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        fig_syn.suptitle(f'Synthesized vs Measured IMU (Per-Component Subplots) - {data_name}', fontsize=14, fontweight='bold')
        fig_syn.tight_layout()

        synth_cmp_path = os.path.join(vis_dir, f'{data_name}_synth_vs_measured_imu.png')
        fig_syn.savefig(synth_cmp_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved synthesized-vs-measured IMU comparison to {synth_cmp_path}")
        plt.close(fig_syn)
    else:
        print("Skipping synthesized-vs-measured plot: synthesized IMU keys are unavailable.")

    # Visualize gt_translation
    gt_translation = dataset.data['gt_translation'].numpy()  # [N, 3]
    
    # Create output directory
    vis_dir = "data_vis"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create figure with multiple views
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top-down view (X-Z plane)
    ax1 = axes[0, 0]
    ax1.plot(gt_translation[:, 0], gt_translation[:, 2], 'b-', linewidth=0.8)
    ax1.scatter(gt_translation[0, 0], gt_translation[0, 2], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(gt_translation[-1, 0], gt_translation[-1, 2], c='red', s=100, marker='x', label='End', zorder=5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title('Top-Down View (X-Z Plane)')
    ax1.axis('equal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Side view (X-Y plane)
    ax2 = axes[0, 1]
    ax2.plot(gt_translation[:, 0], gt_translation[:, 1], 'b-', linewidth=0.8)
    ax2.scatter(gt_translation[0, 0], gt_translation[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter(gt_translation[-1, 0], gt_translation[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m) - Height')
    ax2.set_title('Side View (X-Y Plane)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Front view (Z-Y plane)
    ax3 = axes[1, 0]
    ax3.plot(gt_translation[:, 2], gt_translation[:, 1], 'b-', linewidth=0.8)
    ax3.scatter(gt_translation[0, 2], gt_translation[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax3.scatter(gt_translation[-1, 2], gt_translation[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
    ax3.set_xlabel('Z (m)')
    ax3.set_ylabel('Y (m) - Height')
    ax3.set_title('Front View (Z-Y Plane)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Time series of each axis
    ax4 = axes[1, 1]
    time = dataset.data['time'].numpy()
    ax4.plot(time, gt_translation[:, 0], 'r-', label='X', linewidth=0.8)
    ax4.plot(time, gt_translation[:, 1], 'g-', label='Y (Height)', linewidth=0.8)
    ax4.plot(time, gt_translation[:, 2], 'b-', label='Z', linewidth=0.8)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position (m)')
    ax4.set_title('Position vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add main title
    seq_name = data_name.replace('.pkl', '')
    fig.suptitle(f'Ground Truth Translation - {seq_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(vis_dir, f'{seq_name}_gt_translation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to {output_path}")
    
    plt.close()

# python -m datasets.imuposerdataset
