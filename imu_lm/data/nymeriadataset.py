import os
import pickle
import numpy as np
import torch
import pypose as pp
from .dataset import Sequence
import matplotlib.pyplot as plt
import torch.nn.functional as F

import core.articulate as art
from core.paths import Paths

def mat2quat(R: torch.Tensor):
    """
    Convert rotation matrices to quaternions (x, y, z, w)

    Args:
        R: (..., 3, 3) rotation matrices

    Returns:
        quat: (..., 4) quaternions in (x, y, z, w)
    """
    assert R.shape[-2:] == (3, 3)

    m00 = R[..., 0, 0]
    m11 = R[..., 1, 1]
    m22 = R[..., 2, 2]

    trace = m00 + m11 + m22

    qw = torch.sqrt(torch.clamp(trace + 1.0, min=0.0)) / 2.0
    qx = (R[..., 2, 1] - R[..., 1, 2]) / (4.0 * qw)
    qy = (R[..., 0, 2] - R[..., 2, 0]) / (4.0 * qw)
    qz = (R[..., 1, 0] - R[..., 0, 1]) / (4.0 * qw)

    quat = torch.stack([qx, qy, qz, qw], dim=-1)
    return quat

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

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def add_hip_orientation_noise(mobileposer_input: torch.Tensor, noise_std_deg: float) -> torch.Tensor:

    squeeze_batch = False
    if mobileposer_input.dim() == 2:
        mobileposer_input = mobileposer_input.unsqueeze(0)
        squeeze_batch = True

    x = mobileposer_input.clone()
    B, T, C = x.shape

    noise_std_rad = noise_std_deg * torch.pi / 180.0
    noise_axis_angle = torch.randn(B, T, 3, device=x.device, dtype=x.dtype) * noise_std_rad
    noise_rot = pp.so3(noise_axis_angle.reshape(-1, 3)).Exp().matrix().reshape(B, T, 3, 3)

    hip_rot9 = x[..., 12 * 3 + 3:12 * 3 + 12]
    rot9_mat = hip_rot9.reshape(B, T, 3, 3)
    noisy_rot = torch.matmul(noise_rot, rot9_mat)
    x[..., 12 * 3 + 3:12 * 3 + 12] = noisy_rot.reshape(B, T, 9)

    if squeeze_batch:
        x = x.squeeze(0)
    return x


def apply_fixed_rotation_noise_to_acc_and_rotmat(
    hip_acc: torch.Tensor,
    hip_rot_mat: torch.Tensor,
    noise_std_deg: float,
    seed: int = 0,
):
    if hip_acc.ndim != 2 or hip_acc.shape[-1] != 3:
        raise ValueError(f"Expected hip_acc shape [N, 3], got {tuple(hip_acc.shape)}")
    if hip_rot_mat.ndim != 3 or hip_rot_mat.shape[-2:] != (3, 3):
        raise ValueError(f"Expected hip_rot_mat shape [N, 3, 3], got {tuple(hip_rot_mat.shape)}")
    if hip_acc.shape[0] != hip_rot_mat.shape[0]:
        raise ValueError(f"Length mismatch: hip_acc has {hip_acc.shape[0]} frames, hip_rot_mat has {hip_rot_mat.shape[0]}")

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    noise_std_rad = noise_std_deg * torch.pi / 180.0
    fixed_noise_axis_angle = torch.randn(
        3, device=hip_rot_mat.device, dtype=hip_rot_mat.dtype
    ) * noise_std_rad
    fixed_noise_rot = pp.so3(fixed_noise_axis_angle.unsqueeze(0)).Exp().matrix().squeeze(0)  # [3, 3]

    hip_acc_noisy = torch.matmul(fixed_noise_rot.unsqueeze(0), hip_acc.unsqueeze(-1)).squeeze(-1)
    hip_rot_mat_noisy = torch.matmul(fixed_noise_rot.unsqueeze(0), hip_rot_mat)

    return hip_acc_noisy, hip_rot_mat_noisy, fixed_noise_rot

def _rotmat_to_euler_zyx_deg(rot_mats: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to CPF-frame yaw/pitch/roll in degrees.

    CPF convention used in this project:
      - X: left
      - Y: up
      - Z: forward

    Angle definitions:
      - yaw   : rotation around Y axis
      - pitch : rotation around X axis
      - roll  : rotation around Z axis

    Args:
        rot_mats: [N, 3, 3]

    Returns:
        euler_deg: [N, 3] with columns [yaw, pitch, roll]
    """
    if isinstance(rot_mats, np.ndarray):
        rot_mats = torch.from_numpy(rot_mats)

    if not torch.is_tensor(rot_mats):
        raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(rot_mats)}")

    rot_mats = rot_mats.to(dtype=torch.float32)

    if rot_mats.ndim != 3 or rot_mats.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [N, 3, 3], got {rot_mats.shape}")

    yaw = torch.atan2(rot_mats[:, 0, 2], rot_mats[:, 2, 2])
    pitch = -torch.asin(torch.clamp(rot_mats[:, 1, 2], -1.0, 1.0))
    roll = torch.atan2(rot_mats[:, 1, 0], rot_mats[:, 1, 1])

    euler_rad = torch.stack([yaw, pitch, roll], dim=-1)
    return torch.rad2deg(euler_rad)


def _unwrap_euler_deg(euler_deg: np.ndarray) -> np.ndarray:
    """
    Unwrap Euler angle sequences (in degrees) along time axis to avoid +/-180 jumps.

    Args:
        euler_deg: [N, 3] array in degrees

    Returns:
        [N, 3] unwrapped angles in degrees
    """
    if euler_deg.ndim != 2 or euler_deg.shape[1] != 3:
        raise ValueError(f"Expected [N, 3], got {euler_deg.shape}")
    return np.rad2deg(np.unwrap(np.deg2rad(euler_deg), axis=0))


def _relative_rotation_error_deg(rot_a, rot_b) -> torch.Tensor:
    """
    Compute geodesic relative rotation error between two rotation sequences.

    Args:
        rot_a: [N, 3, 3], torch.Tensor or np.ndarray
        rot_b: [N, 3, 3], torch.Tensor or np.ndarray

    Returns:
        err_deg: [N] angular error in degrees
    """
    if isinstance(rot_a, np.ndarray):
        rot_a = torch.from_numpy(rot_a)
    if isinstance(rot_b, np.ndarray):
        rot_b = torch.from_numpy(rot_b)

    if not torch.is_tensor(rot_a) or not torch.is_tensor(rot_b):
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(rot_a)} and {type(rot_b)}")

    rot_a = rot_a.to(dtype=torch.float32)
    rot_b = rot_b.to(dtype=torch.float32)

    if rot_a.ndim != 3 or rot_b.ndim != 3 or rot_a.shape[-2:] != (3, 3) or rot_b.shape[-2:] != (3, 3):
        raise ValueError(f"Expected both inputs shape [N, 3, 3], got {rot_a.shape} and {rot_b.shape}")
    if rot_a.shape[0] != rot_b.shape[0]:
        raise ValueError(f"Sequence length mismatch: {rot_a.shape[0]} vs {rot_b.shape[0]}")

    rel = torch.matmul(rot_a.transpose(-1, -2), rot_b)  # [N, 3, 3]
    tr = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    err_rad = torch.acos(cos_theta)
    return torch.rad2deg(err_rad)

vi_mask = torch.tensor([
    1961, 5424, 876, 4362, 411, 3021,
    3050,   # chest
    6720,    # left foot
    5700 # left wrist
])

ji_mask = torch.tensor([
    18, 19, 1, 2, 15, 0,
    12,   # chest
    10,    # left foot
    20     # left wrist
])

def _syn_vel(v, smooth_n=4, fps=200):
    """
    v: [T, D]
    return vel: [T, D]
    """
    T = v.shape[0]
    vel = torch.zeros_like(v)

    if smooth_n <= 1:
        # forward difference
        vel[:-1] = (v[1:] - v[:-1]) * fps
        vel[-1] = vel[-2]
        return vel

    m = smooth_n // 2

    # centered difference
    vel[m:-m] = (v[m+1:-m+1] - v[m-1:-m-1]) * (fps / 2)

    # boundary handling
    vel[:m] = vel[m]
    vel[-m:] = vel[-m-1]

    return vel

def _syn_acc(v, smooth_n=4, TARGET_FPS=200):
    """Synthesize accelerations from vertex positions."""
    mid = smooth_n // 2
    scale_factor = TARGET_FPS ** 2 

    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * scale_factor for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))

    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * scale_factor / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc


def _vertical_motion_labels(
    joint,
    foot_vel_threshold=0.008,
    window=10,
    height_threshold=0.15,
    smooth=True,
    smooth_kernel=10,
):
    r"""
    Compute binary vertical-motion labels using long-window support-foot height change.

    Args:
        joint: Tensor of shape (N, 24, 3), world-frame joint positions
        foot_vel_threshold: threshold for foot contact detection
        window: number of frames for long-term height difference
        height_threshold: threshold for vertical change over the window (meters)
        smooth: whether to smooth the final labels
        smooth_kernel: kernel size for label smoothing

    Returns:
        labels: Tensor of shape (N,)
            1 -> significant vertical motion (e.g. up/down stairs)
            0 -> no significant vertical motion
    """

    N = joint.shape[0]
    device = joint.device

    # Foot y positions
    lfy = joint[:, 10, 1]
    rfy = joint[:, 11, 1]

    # Foot-contact detection from foot velocity
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

    # Estimate support foot height
    support_y = torch.zeros(N, dtype=joint.dtype, device=device)

    both_contact = (l_contact == 1) & (r_contact == 1)
    left_only = (l_contact == 1) & (r_contact == 0)
    right_only = (l_contact == 0) & (r_contact == 1)
    neither = (l_contact == 0) & (r_contact == 0)

    support_y[left_only] = lfy[left_only]
    support_y[right_only] = rfy[right_only]
    support_y[both_contact] = torch.minimum(lfy[both_contact], rfy[both_contact])

    # Carry forward when no foot is in contact
    for i in range(1, N):
        if neither[i]:
            support_y[i] = support_y[i - 1]

    # Long-window height change
    delta_h = torch.zeros(N, dtype=joint.dtype, device=device)
    if N > window:
        delta_h[window:] = torch.abs(support_y[window:] - support_y[:-window])

    # Binary label
    labels = (delta_h > height_threshold).int()

    # Optional smoothing
    if smooth and smooth_kernel > 1:
        pad = smooth_kernel // 2
        x = labels.float().unsqueeze(0).unsqueeze(0)
        x = F.pad(x, (pad, pad), mode='replicate')
        x = F.avg_pool1d(x, kernel_size=smooth_kernel, stride=1)
        labels = (x.squeeze() > 0.5).int()

    return labels

def _foot_ground_probs(joint):
    r"""
    Compute foot-ground contact probabilities.
    
    Args:
        joint: Tensor of shape (N, 24, 3) representing joint positions.
        
    Returns:
        Tensor of shape (N, 2) where each row is [left_foot_contact, right_foot_contact].
        Contact is 1 if foot velocity is below threshold, 0 otherwise.
    """
    # Joint indices 10 and 11 are left and right foot (toes) in SMPL
    dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
    dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
    lfoot_contact = (dist_lfeet < 0.008).int()
    rfoot_contact = (dist_rfeet < 0.008).int()
    lfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), lfoot_contact))
    rfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), rfoot_contact))
    return torch.stack((lfoot_contact, rfoot_contact), dim=1)


def _compute_fused_velocity(joint_local, foot_contact, pelvis_vel, 
                             gravity_velocity=0.018, prob_threshold=(0.5, 0.9)):
    """
    Compute fused velocity by blending root velocity with foot-contact-based velocity.
    Similar to MobilePoser's velocity fusion approach.
    
    IMPORTANT: Uses LOCAL joint positions (relative to pelvis, with zero translation)
    not global joint positions. When foot is grounded in world frame, it moves backward
    relative to pelvis in local frame, indicating pelvis moves forward.
    
    Args:
        joint_local: Tensor of shape (N, 24, 3) - joint positions in LOCAL frame (no translation)
        foot_contact: Tensor of shape (N, 2) - foot contact binary [left, right]
        pelvis_vel: Tensor of shape (N, 3) - root/pelvis velocity from differentiation
        gravity_velocity: float - gravity compensation velocity (default: 0.018 from MobilePoser)
        prob_threshold: tuple - (min, max) threshold for blending weight computation
        
    Returns:
        fused_vel: Tensor of shape (N, 3) - fused velocity
    """
    # Get foot positions in LOCAL frame (relative to pelvis)
    lfoot_local = joint_local[:, 10, :]  # [N, 3] - left foot (toes)
    rfoot_local = joint_local[:, 11, :]  # [N, 3] - right foot (toes)
    gravity_vel = torch.tensor([0, gravity_velocity, 0], dtype=lfoot_local.dtype, device=lfoot_local.device)
    
    # Foot displacement in local frame: when foot is grounded in world,
    # foot moves backward relative to pelvis → body moves forward
    # displacement = foot[t-1] - foot[t] → positive when foot moves backward → body moves forward
    lfoot_disp = torch.cat([torch.zeros(1, 3, device=lfoot_local.device), lfoot_local[:-1] - lfoot_local[1:]], dim=0)  # [N, 3]
    rfoot_disp = torch.cat([torch.zeros(1, 3, device=rfoot_local.device), rfoot_local[:-1] - rfoot_local[1:]], dim=0)  # [N, 3]
    
    # Select which foot to use based on contact (higher contact = more grounded)
    foot_contact_f = foot_contact.float()  # [N, 2]
    contact_idx = foot_contact_f.max(dim=1).indices.view(-1, 1)  # [N, 1]
    contact_vel = torch.where(
        contact_idx == 0,
        lfoot_disp,  # left foot grounded
        rfoot_disp   # right foot grounded
    )  # [N, 3]
    
    fps = 50  # Assuming 50 Hz sampling rate; adjust if different
    contact_vel = contact_vel * fps

    # Root velocity from differentiation
    pred_vel = pelvis_vel.float()  # [N, 3]
    
    # Compute blending weight from contact probability
    max_contact_prob = foot_contact_f.max(dim=1).values.sigmoid()  # [N]
    # Clamp and normalize to [0, 1]
    weight = (max_contact_prob.clamp(prob_threshold[0], prob_threshold[1]) - prob_threshold[0]) / \
             (prob_threshold[1] - prob_threshold[0])
    weight = weight.view(-1, 1)  # [N, 1]
    
    # Fused velocity: lerp(pred_vel, contact_vel, weight)
    # weight=0 -> pred_vel (no contact), weight=1 -> contact_vel (foot grounded)
    fused_vel = pred_vel * (1 - weight) + contact_vel * weight  # [N, 3]
    
    return fused_vel


def _compute_velocity_from_position(positions, time):
    """
    Compute velocity from positions using finite difference.
    
    Args:
        positions: np.ndarray or torch.Tensor of shape [N, ...] - positions over time
        time: np.ndarray or torch.Tensor of shape [N] - timestamps
        
    Returns:
        velocity: torch.Tensor of same shape as positions - velocity in global frame
    """
    # Convert to numpy for computation
    if isinstance(positions, torch.Tensor):
        x = positions.detach().cpu().numpy()
    else:
        x = positions
    
    if isinstance(time, torch.Tensor):
        t = time.detach().cpu().numpy()
    else:
        t = time
    
    # Compute dt with appropriate shape for broadcasting
    # Add extra dimensions to match positions rank
    extra_dims = len(x.shape) - 1
    dt_shape = [-1] + [1] * extra_dims
    dt = (t[1:] - t[:-1]).reshape(dt_shape)  # [N-1, 1, ...]
    dt = np.maximum(dt, 1e-6)  # Avoid division by zero
    
    # Compute velocity via finite difference
    v = np.zeros_like(x)
    v[1:] = (x[1:] - x[:-1]) / dt
    v[0] = v[1]  # Copy first valid velocity to index 0
    
    return torch.from_numpy(v)


def _compute_angular_velocity_from_rotation(rot_mats, fps):
    """
    Compute angular velocity (gyro) from rotation matrices using SO3 logarithm.

    Supports:
        rot_mats: [N, 3, 3]
        rot_mats: [N, K, 3, 3]

    Args:
        rot_mats: torch.Tensor of shape [N, 3, 3] or [N, K, 3, 3]
        fps: float - sampling rate in Hz

    Returns:
        angular_vel:
            [N, 3]    if input is [N, 3, 3]
            [N, K, 3] if input is [N, K, 3, 3]
    """
    if rot_mats.ndim not in (3, 4) or rot_mats.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected rot_mats to have shape [N, 3, 3] or [N, K, 3, 3], but got {rot_mats.shape}"
        )

    dt = 1.0 / fps

    # Case 1: single rotation track [N, 3, 3]
    if rot_mats.ndim == 3:
        N = rot_mats.shape[0]
        if N < 2:
            return torch.zeros(N, 3, dtype=rot_mats.dtype, device=rot_mats.device)

        R_t = rot_mats[:-1]   # [N-1, 3, 3]
        R_t1 = rot_mats[1:]   # [N-1, 3, 3]
        R_rel = torch.bmm(R_t.transpose(-1, -2), R_t1)  # [N-1, 3, 3]

        R_rel_quat = mat2quat(R_rel)  # [N-1, 4]
        R_rel_so3 = pp.SO3(R_rel_quat)
        omega = R_rel_so3.Log() / dt  # [N-1, 3]

        angular_vel = torch.zeros(N, 3, dtype=rot_mats.dtype, device=rot_mats.device)
        angular_vel[:-1] = omega
        angular_vel[-1] = omega[-1]
        return angular_vel

    # Case 2: batched rotation tracks [N, K, 3, 3]
    N, K = rot_mats.shape[:2]
    if N < 2:
        return torch.zeros(N, K, 3, dtype=rot_mats.dtype, device=rot_mats.device)

    R_t = rot_mats[:-1]    # [N-1, K, 3, 3]
    R_t1 = rot_mats[1:]    # [N-1, K, 3, 3]

    # Relative rotations along time for each track independently
    R_rel = torch.matmul(R_t.transpose(-1, -2), R_t1)   # [N-1, K, 3, 3]

    # Flatten batch dims for PyPose
    R_rel_flat = R_rel.reshape(-1, 3, 3)                # [(N-1)*K, 3, 3]
    R_rel_quat = mat2quat(R_rel_flat)                   # [(N-1)*K, 4]
    R_rel_so3 = pp.SO3(R_rel_quat)
    omega = (R_rel_so3.Log() / dt).reshape(N - 1, K, 3)  # [N-1, K, 3]

    angular_vel = torch.zeros(N, K, 3, dtype=rot_mats.dtype, device=rot_mats.device)
    angular_vel[:-1] = omega
    angular_vel[-1] = omega[-1]

    return angular_vel


def _integrate_velocity_to_trajectory(velocity, time, initial_position=None):
    """
    Integrate velocity over time to compute trajectory (position).
    
    Args:
        velocity: torch.Tensor or np.ndarray of shape [N, D] - velocity over time
        time: torch.Tensor or np.ndarray of shape [N] - timestamps
        initial_position: torch.Tensor or np.ndarray of shape [D] - initial position (default: zeros)
        
    Returns:
        trajectory: torch.Tensor of shape [N, D] - integrated trajectory
    """
    # Convert to numpy for computation
    if isinstance(velocity, torch.Tensor):
        v = velocity.detach().cpu().numpy()
    else:
        v = velocity
    
    if isinstance(time, torch.Tensor):
        t = time.detach().cpu().numpy()
    else:
        t = time
    
    # Compute dt
    dt = np.zeros(len(t))
    dt[1:] = t[1:] - t[:-1]
    dt[0] = dt[1]  # Use same dt for first frame
    
    # Initialize trajectory
    N, D = v.shape
    traj = np.zeros_like(v)
    
    # Set initial position
    if initial_position is not None:
        if isinstance(initial_position, torch.Tensor):
            traj[0] = initial_position.detach().cpu().numpy()
        else:
            traj[0] = initial_position
    
    # Integrate: x[t+1] = x[t] + v[t] * dt
    for i in range(1, N):
        traj[i] = traj[i-1] + v[i] * dt[i]
    
    return torch.from_numpy(traj).float()


XSENS_TO_SMPL = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18, 
                            5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]

def xsens_to_SMPL(xsens_rotmat):
    """
    Convert Xsens rotation matrices to SMPL rotation matrices.
    
    Args:
        xsens_rotmat: Tensor of shape (N, 23, 3, 3) representing Xsens rotation matrices.
        
    Returns:
        Tensor of shape (N, 23, 3, 3) representing SMPL rotation matrices.
    """

    perm = [1, 2, 0] 
    xsens_rotmat = xsens_rotmat[:, :, perm, :]  
    xsens_rotmat = xsens_rotmat[:, :, :, perm]  

    if isinstance(xsens_rotmat, torch.Tensor):
        xsens_rotmat = xsens_rotmat.to(dtype=torch.float32)
    else:
        xsens_rotmat = torch.from_numpy(xsens_rotmat).float()
    
    Rx90 = torch.tensor([
        [1., 0., 0.],
        [0., 0., -1.],
        [0., 1., 0.]
    ], device=xsens_rotmat.device, dtype=xsens_rotmat.dtype)
    
    Rz90 = torch.tensor([
        [0., -1., 0.],
        [1.,  0., 0.],
        [0.,  0., 1.]
    ], device=xsens_rotmat.device, dtype=xsens_rotmat.dtype)
        
    xsens_rotmat[:, :, ...] = Rz90 @ (Rx90 @ xsens_rotmat[:,:, ...])
    
    smpl_rotmat = torch.eye(3).repeat(xsens_rotmat.shape[0], 24, 1, 1)
    for smpl_idx, xsens_idx in enumerate(XSENS_TO_SMPL):
        smpl_rotmat[:, smpl_idx, :] = xsens_rotmat[:, xsens_idx, :]
            
    return smpl_rotmat

class nymeria(Sequence):
    def __init__(
        self,
        data_root: str,
        data_name: str,
        coordinate: str = None,
        gravity: float = 9.81,
        remove_g: bool = False,
        maximum_length: int = 10000,
        use_body: bool = True,
        target_hz: int = 50,
        use_root_vel: bool = False,
        rot_from_gyro: bool = False,
        mobileposer_head_only: bool = False,
        mobileposer_position: str = None ,
        mobileposer_use_3_input: bool = False,
        **kwargs
    ):
        super().__init__()
        # Load pickle file - try .pkl first, then .pickle
        pkl_file = os.path.join(data_root, data_name + '.pkl')
        if not os.path.exists(pkl_file):
            pkl_file = os.path.join(data_root, data_name + '.pickle')
        with open(pkl_file, 'rb') as f:
            raw = pickle.load(f)
        data_sampling_rate_hz = raw['imu_sampling_rate_hz']
        downsampling_rate = max(1, round( data_sampling_rate_hz / target_hz))
        for key in ["time", "timestamps_imu", "gt_orientation", "accel", "gyro", 'orientation', 'position', 'velocity', "xsens_pose"]:
            if key in raw['imu_data']:
                raw['imu_data'][key] = raw['imu_data'][key][::downsampling_rate]
            if key in raw['gt_data']:
                raw['gt_data'][key] = raw['gt_data'][key][::downsampling_rate]
                    
        # Initialize data dictionary
        self.data = {}

        # Process IMU data
        if 'imu_data' in raw:
            # Single-rate: use common timestamps
            if 'time' in raw['imu_data']:
                self.data['time'] = raw['imu_data']['time']
            # Handle accelerometer data - already in CPF frame from preprocessing
            if 'accel' in raw['imu_data']:
                self.data['acc'] = raw['imu_data']['accel']
            # Handle gyroscope data - already in CPF frame from preprocessing
            if 'gyro' in raw['imu_data']:
                self.data['gyro'] = raw['imu_data']['gyro'] # world frame gyro
                
        # imu_dat = rot_non_yaw @ imu_data
        
        # Process ground truth data
        if 'gt_data' in raw:
            # Single-rate: use common timestamps
            if 'time' not in self.data and 'time' in raw['gt_data']:
                self.data['time'] = raw['gt_data']['time']
                
            # Handle orientation data - now stored as CPF→World quaternions
            if 'orientation' in raw['gt_data']:
                orientation = raw['gt_data']['orientation']
                if isinstance(orientation, torch.Tensor) and not isinstance(orientation, pp.LieTensor):
                    # Check if it's a quaternion [qx, qy, qz, qw] (PyPose order)
                    if orientation.shape[-1] == 4:
                        # Convert to PyPose SO3 - represents CPF→World rotation
                        self.data['gt_orientation'] = pp.SO3(orientation)
                        print(f"   ✓ Loaded CPF→World orientations: {orientation.shape}")
                    else:
                        print(f"Warning: Unexpected orientation shape: {orientation.shape}")
                        self.data['gt_orientation'] = orientation
                else:
                    self.data['gt_orientation'] = orientation
                
                # self.data['gt_orientation_yaw_only'] = self._heading_only_from_so3(self.data['gt_orientation'])
            
            # Handle position data
            if 'position' in raw['gt_data']:
                self.data['gt_translation'] = raw['gt_data']['position']
                
            # Handle velocity data - already in CPF frame from preprocessing
            if 'velocity' in raw['gt_data']:
                self.data['velocity'] = raw['gt_data']['velocity']
            
            self.gravity = torch.tensor([0,-gravity,0],dtype=self.data['acc'].dtype)
            self.gravity = self.gravity.expand(self.data['acc'].shape[0], -1)

            if 'xsens_pose' in raw['gt_data'] and use_body:
                self.data['xsens_pose'] = raw['gt_data']['xsens_pose']
                self.data['body_translation'] = self.data['xsens_pose'][:,0, :3, 3].contiguous().clone() # tensor [N, 23, 3]
            
                self.data['root_translation'] = self.data['body_translation'] # alias
                
                # Compute root velocity from root translation
                t = self.data.get('time_pose', self.data['time'])  # [N]
                self.data['pelvis_vel'] = _compute_velocity_from_position(
                    self.data['root_translation'], t
                )  # [N, 3] in global frame

                self.body_smpl = xsens_to_SMPL(self.data['xsens_pose'][:,:, :3, :3].contiguous().clone()) # tensor [N, 23, 3, 3]
                
                # self.data["head_translation"] = self.data['xsens_pose'][:,15, :3, 3] # [N, 3]
                global_pose = self.body_smpl
                self.body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))
                local_poses = self.body_model.inverse_kinematics_R(global_pose).view(global_pose.shape[0], 24, 3, 3).contiguous().clone()   
                self.device = torch.device("cpu")
                self.pose_global, self.joint_global, self.vertex_global_trans = self.body_model.forward_kinematics_masked(local_poses.to(self.device), tran=self.data['body_translation'].to(self.device),calc_mesh=True, vi_mask=vi_mask)
                _, self.joint_pos_local = self.body_model.forward_kinematics_masked(local_poses.to(self.device), tran=torch.zeros_like(self.data['body_translation']).to(self.device), calc_mesh=False, vi_mask=vi_mask)
                self.data['joint_pos'] = self.joint_pos_local.flatten(start_dim=1).cpu() # (N, 24*3)
                self.data['pose_rotmat'] = local_poses # (N, 24, 3, 3)
                self.data['joint_rot'] = matrix_to_rotation_6d(self.data['pose_rotmat']).flatten(start_dim=1) # (N, 24*6)
                
                # Compute joint velocity from joint positions [N, 24, 3] -> [N, 24*3]
                t = self.data.get('time_pose', self.data['time'])
                joint_vel_3d = _compute_velocity_from_position(self.joint_global, t)  # [N, 24, 3]
                self.data['joint_vel'] = joint_vel_3d.flatten(start_dim=1).float()  # (N, 24*3)
          
                # Compute foot-ground contact probabilities
                self.data['foot_contact'] = _foot_ground_probs(self.joint_global).clone()  # (N, 2)
                self.data['left_foot_y'] = self.joint_global[:, 10, 1].detach().cpu().float()   # [N]
                self.data['right_foot_y'] = self.joint_global[:, 11, 1].detach().cpu().float()  # [N]
                
                self.data['updown_label'] = _vertical_motion_labels(self.joint_global) # [N]
                
                # Compute fused velocity (similar to MobilePoser)

                self.data['fused_vel'] = _compute_fused_velocity(
                    self.joint_pos_local, 
                    self.data['foot_contact'], 
                    self.data['pelvis_vel']
                )  # [N, 3]
                
                # Integrate fused velocity to get trajectory
                t = self.data.get('time_pose', self.data['time'])
                self.data['fused_vel_traj'] = _integrate_velocity_to_trajectory(
                    self.data['fused_vel'], t, 
                    initial_position=self.data['root_translation'][0]
                )  # [N, 3]
                
                self.data['body_v_acc'] = _syn_acc(self.vertex_global_trans, smooth_n=4, TARGET_FPS=target_hz) # [N, 6, 3]
                
                from .imu_synthesis import syn_imu_from_smpl
                local_poses = local_poses.to(self.device)
                root_tran = self.data['root_translation'].to(self.device)

                self.data['body_v_rot'] = self.pose_global[:, ji_mask] # [N, 6, 3, 3]
                
                self.data['body_v_acc_2'], self.data['body_v_gyro'], self.data['body_v_rot_2'] = syn_imu_from_smpl(local_poses, root_tran)

                # Compute vertex velocities from positions
                t = self.data.get('time_pose', self.data['time'])  # [N]
                self.data['body_v_vel'] = _compute_velocity_from_position(self.vertex_global_trans, t)  # [N, K, 3]

                self.data['right_hip_acc'] = self.data['body_v_acc'][:,3,:] # [N, 3]
                self.data['right_hip_vel'] = self.data['body_v_vel'][:,3,:] # [N, 3] - velocity in global frame
                self.data['right_hip_rot_mat'] = self.data['body_v_rot'][:,3,:,:] # [N, 3, 3]
                self.data['right_hip_rot'] = matrix_to_rotation_6d(self.data['body_v_rot'][:,3,:,:]).flatten(start_dim=1) # [N, 6]
                
                # Head acceleration from vertex positions (ji_mask[4] = 15 is head joint)
                self.data['head_acc_from_vertex'] = self.data['body_v_acc'][:,4,:] # [N, 3] - global frame
                # Head IMU acceleration transformed to global frame
                self.data['head_acc_imu_global'] = (self.data['gt_orientation'] @ self.data['acc']).clone() # [N, 3] - global frame
                # Head IMU gyro transformed to global frame
                self.data['head_gyro_imu_global'] = (self.data['gt_orientation'] @ self.data['gyro']).clone() # [N, 3] - global frame
                # # Head rotation from body model (for comparison with IMU orientation)
                self.data['head_rot_from_body'] = self.data['body_v_rot'][:,4,:,:].clone() # [N, 3, 3] - global frame
                # # Head gyro from body model (for comparison with IMU gyro)
                # self.data['head_gyro_from_body'] = self.data['body_v_gyro'][:,4,:].clone() # [N, 3] - global frame
                
                self.data['right_elbow_acc'] = self.data['body_v_acc'][:,1,:] # [N, 3]
                self.data['right_elbow_vel'] = self.data['body_v_vel'][:,1,:] # [N, 3] - velocity in global frame
                self.data['right_elbow_rot'] = matrix_to_rotation_6d(self.data['body_v_rot'][:,1,:,:]).flatten(start_dim=1) # [N, 6]
                
                # Compute right elbow gyro (angular velocity) from rotation matrices
                self.data['right_elbow_gyro'] = _compute_angular_velocity_from_rotation(
                    self.data['body_v_rot'][:, 1, :, :], target_hz
                )  # [N, 3]
                
                self.data['pocket_vel'] = self.data['right_hip_vel'] # alias for right hip velocity in global frame
                
                # if use_root_vel:
                #     self.data['gt_translation'] = self.data['root_translation']
                
                if not mobileposer_use_3_input:
                    self.data['mobileposer_input'] = torch.zeros([local_poses.shape[0],60])
                    
                    # self.data['mobileposer_input'][:, 12*4:12*4+3] = (self.data['gt_orientation'] @ self.data['acc'])/30
                    # self.data['mobileposer_input'][:, 12*4+3:12*4+12] = self.data['gt_orientation'].matrix().reshape(local_poses.shape[0], -1)
                    
                    # self.data['mobileposer_input'][:, 12*4:12*4+3] = (self.data['body_v_acc'][:,3,:] - self.gravity.to(self.data['body_v_acc'].device))/30       
                    # self.data['mobileposer_input'][:, 12*4+3:12*4+12] = self.data['body_v_rot'][:,3,:,:].reshape(local_poses.shape[0], -1)
                    
                    if not mobileposer_head_only:
                        if mobileposer_position == "foot":
                            self.data['mobileposer_input'][:, 12*3:12*3+3] = (self.data['body_v_acc'][:,7,:])
                            self.data['mobileposer_input'][:, 12*3+3:12*3+12] = self.data['body_v_rot'][:,7,:,:].reshape(local_poses.shape[0], -1)
                        elif mobileposer_position == "chest":
                            self.data['mobileposer_input'][:, 12*3:12*3+3] = (self.data['body_v_acc'][:,6,:])
                            self.data['mobileposer_input'][:, 12*3+3:12*3+12] = self.data['body_v_rot'][:,6,:,:].reshape(local_poses.shape[0], -1)
                        elif mobileposer_position == "wrist":
                            self.data['mobileposer_input'][:, 12*3:12*3+3] = (self.data['body_v_acc'][:,8,:])
                            self.data['mobileposer_input'][:, 12*3+3:12*3+12] = self.data['body_v_rot'][:,8,:,:].reshape(local_poses.shape[0], -1)
                        else:    
                            # noisy_hip_acc, noisy_hip_rot_mat, _ = apply_fixed_rotation_noise_to_acc_and_rotmat(
                            #     hip_acc=self.data['right_hip_acc'],
                            #     hip_rot_mat=self.data['right_hip_rot_mat'],
                            #     noise_std_deg=0,
                            #     seed=0,
                            # )
                            # self.data['mobileposer_input'][:, 12*3:12*3+3] = noisy_hip_acc / 30
                            # self.data['mobileposer_input'][:, 12*3+3:12*3+12] = noisy_hip_rot_mat.reshape(local_poses.shape[0], -1)
                            
                            self.data['mobileposer_input'][:, 12*3:12*3+3] = self.data['right_hip_acc'] / 30
                            self.data['mobileposer_input'][:, 12*3+3:12*3+12] = self.data['right_hip_rot_mat'].reshape(local_poses.shape[0], -1)
               
                            # self.data['mobileposer_input'][:, 12*3:12*3+3] = (self.data['body_v_acc_2'][:,3,:])/30
                            
                            # self.data['mobileposer_input'][:, 12*3+3:12*3+12] = self.data['body_v_rot_2'][:,3,:,:].reshape(local_poses.shape[0], -1)
                            
                            # noise_sigma = 0.1
                            # self.data['mobileposer_input'][:, 12*3:12*3+3] += noise_sigma * torch.randn_like(self.data['mobileposer_input'][:, 12*3:12*3+3])
                            # self.data['mobileposer_input'][:, 12*3+3:12*3+12] += noise_sigma * torch.randn_like(self.data['mobileposer_input'][:, 12*3+3:12*3+12])
                            
                            # self.data['body_v_acc'] = _syn_acc(self.vertex_global_trans, smooth_n=2, TARGET_FPS=target_hz)[:,:6,:] # [N, 6, 3]
                            # self.data['right_hip_acc'] = self.data['body_v_acc'][:,3,:]
                            # self.data['mobileposer_input'][:, 12*3:12*3+3] = (self.data['right_hip_acc']) #+ (torch.randn_like(self.data['right_hip_acc']) * 0.1)/30
                            # self.data['mobileposer_input'][:, 12*3+3:12*3+12] = self.data['right_hip_rot_mat'].reshape(local_poses.shape[0], -1)
                else:
                    self.data['mobileposer_input'] = torch.zeros([local_poses.shape[0],60])
                    # head
                    self.data['mobileposer_input'][:, 12*4:12*4+3] = (self.data['body_v_acc'][:,4,:])
                    self.data['mobileposer_input'][:, 12*4+3:12*4+6] = self.data['body_v_gyro'][:,4,:]
                    self.data['mobileposer_input'][:, 12*4+6:12*4+12] = matrix_to_rotation_6d(self.data['body_v_rot'][:,4,:,:])
                    
                    # right hip
                    self.data['mobileposer_input'][:, 12*3:12*3+3] = (self.data['body_v_acc'][:,3,:])
                    self.data['mobileposer_input'][:, 12*3+3:12*3+6] = (self.data['body_v_gyro'][:,3,:])
                    self.data['mobileposer_input'][:, 12*3+6:12*3+12] = matrix_to_rotation_6d(self.data['body_v_rot'][:,3,:,:])
                    
                self.data['xsens_pose'] = None  # Free up memory
                self.data['pose_rotmat'] = None  # Free up memory
                self.body_model = None
                
                del self.vertex_global_trans, self.pose_global, self.joint_global
                
            else:
                self.data['right_hip_acc'] = torch.zeros((self.get_length(), 3), dtype=torch.float32)
                self.data['right_hip_rot'] = torch.zeros((self.get_length(), 6), dtype=torch.float32)
                self.data['joint_pos'] = torch.zeros((self.get_length(), 24*3), dtype=torch.float32)
                self.data['joint_vel'] = torch.zeros((self.get_length(), 24*3), dtype=torch.float32)
                self.data['joint_rot'] = torch.zeros((self.get_length(), 24*6), dtype=torch.float32)
                self.data['pelvis_vel'] = torch.zeros((self.get_length(), 3), dtype=torch.float32)
                self.data['pocket_vel'] = torch.zeros((self.get_length(), 3), dtype=torch.float32)
                self.data['foot_contact'] = torch.zeros((self.get_length(), 2), dtype=torch.int)
                self.data['left_foot_y'] = torch.zeros((self.get_length(),), dtype=torch.float32)
                self.data['right_foot_y'] = torch.zeros((self.get_length(),), dtype=torch.float32)
                self.data['fused_vel'] = torch.zeros((self.get_length(), 3), dtype=torch.float32)
                self.data['fused_vel_traj'] = torch.zeros((self.get_length(), 3), dtype=torch.float32)
        
        # Single-rate dt calculation
        if 'dt' in raw.get('imu_data', {}):
            self.data['dt'] = raw['imu_data']['dt']
        else:
            time = self.data['time']
            dt = torch.zeros_like(time)
            if len(time) > 1:
                dt[1:] = time[1:] - time[:-1]
                dt[0] = dt[1]  # Set first dt to second dt
            self.data['dt'] = dt

        if rot_from_gyro:
            # Right IMU: integrate gyro to get orientation (CPF frame)
            R0   = self.data['gt_orientation'][0]                  # [] SO3 (LieTensor)
            gyro = self.data['gyro'][:-1]                          # [N-1, 3] - Right IMU
            dt   = self.data['dt'].reshape(-1) if self.data['dt'].ndim > 1 else self.data['dt']  # [N]
            dt   = dt[:-1]                                         # [N-1] to match gyro

            # keep everything on the same device/dtype
            gyro = gyro.to(R0.device)
            dt   = dt.to(R0.device)

            theta = gyro * dt.unsqueeze(-1)                        # [N-1, 3] radians
            delta = pp.so3(theta).Exp()                            # [N-1] SO3 increments
            Rrel  = pp.cumprod(delta, dim=0, left=False)           # [N-1] SO3 (prefix products)

            # include the initial pose as the first element; total length N
            Rseq = torch.cat([R0.unsqueeze(0), (R0 * Rrel)], dim=0)  # [N] SO3

            self.data['orientation_from_gyro'] = Rseq
            # self.data['gt_orientation'] = Rseq
            
            print(f"   ✓ Replaced GT orientations with GYRO-INTEGRATED rotations: {Rseq.shape}")
                
        # Create mask (all valid by default)
        mask_size = len(self.data.get('time_pose', self.data.get('time', [])))
        self.data['mask'] = torch.ones(mask_size, dtype=torch.bool)
                
        cpf_velocities = self.data["velocity"].clone()  # Clone to preserve original
        
        # Get CPF→World orientations and convert to rotation matrices
        orientations = self.data["gt_orientation"]
        R_cpf_to_world = orientations.matrix().float()  # Shape: [N, 3, 3]
        world_velocities = torch.bmm(R_cpf_to_world, cpf_velocities.unsqueeze(-1)).squeeze(-1)
        
        self.data['vertical_vel'] = world_velocities[:,1:2]

        # Replace velocity with world frame velocity
        if coordinate == "glob_coord":
            self.data["velocity"] = world_velocities
        
        assert coordinate in [None, "glob_coord", "body_coord"], "Coordinate must be None, 'glob_coord' or 'body_coord'"
        self.coordinate = coordinate
        
        if remove_g:
            self.data['acc'] = self.data['acc'] + self.data["gt_orientation"].Inv() @ self.gravity
            
        if self.coordinate == "glob_coord":
            self.data['acc'] = self.data["gt_orientation"] @ self.data['acc']
        
        # Compute head acceleration from gt_translation using simple double finite difference
        if 'gt_translation' in self.data:
            pos = self.data['gt_translation']  # [N, 3]
            dt = 1.0 / target_hz
            # First derivative: velocity
            vel = torch.zeros_like(pos)
            vel[1:] = (pos[1:] - pos[:-1]) / dt
            vel[0] = vel[1]
            # Second derivative: acceleration
            acc = torch.zeros_like(vel)
            acc[1:] = (vel[1:] - vel[:-1]) / dt
            acc[0] = acc[1]
            self.data['head_acc_from_gt_translation'] = acc  # [N, 3] - global frame
        
        # if use_root_vel:
        #     self.data['velocity'] =  self.data['root_vel']
        
        keep = {"acc","gyro","dt","time","mask", "velocity","joint_pos","joint_vel","joint_rot","right_hip_acc","right_hip_rot","right_hip_rot_mat","right_elbow_acc","right_elbow_rot","right_elbow_gyro","gt_translation","gt_orientation", "vertical_vel", "pelvis_vel", "pocket_vel", "foot_contact", "left_foot_y", "right_foot_y", "fused_vel", "fused_vel_traj", "updown_label", "root_translation", "mobileposer_input", "head_acc_from_vertex", "head_acc_imu_global", "head_gyro_imu_global", "head_gyro_from_body", "head_acc_from_gt_translation", "head_rot_from_body", "body_v_acc", "body_v_acc_2", "body_v_rot", "body_v_rot_2"}
        self.data = {k: v for k, v in self.data.items() if k in keep}
        if self.get_length() > maximum_length:
            for key in ["velocity", "gt_translation", "gt_orientation", "time", "mask", "dt", "acc", "gyro", "body_translation", "root_translation", "root_vel", "joint_pos", "joint_rot", "left_hip_acc", "left_hip_rot", "left_foot_y", "right_foot_y", "vertical_vel", "pelvis_vel", "pocket_vel"]:
                if key in self.data:
                    self.data[key] = self.data[key][:maximum_length]
        # Print summary
        print(f"\n=== Loaded Nymeria dataset: {data_name} ===")
        del raw
        import gc; gc.collect()
        
    def get_length(self):
        """Number of samples equals number of time stamps"""

        return self.data['time'].shape[0]

    def vis_gt_smpl_without_translation(
        self,
        start: int = 0,
        max_frames: int = 1000,
        fps: int = 50,
    ):
        """
        Visualize GT SMPL motion without root translation.

        This uses local GT joint rotations (`joint_rot`) and renders in-place
        (translation is not provided to the viewer).
        """
        if 'joint_rot' not in self.data or self.data['joint_rot'] is None:
            raise ValueError("'joint_rot' is not available in dataset")

        if max_frames <= 0:
            raise ValueError(f"max_frames must be > 0, got {max_frames}")

        total_len = self.get_length()
        if total_len < 1:
            raise ValueError("Empty dataset: no frames to visualize")

        start = max(0, int(start))
        end = min(start + int(max_frames), total_len)
        if end - start < 1:
            raise ValueError(f"No valid frames in range start={start}, max_frames={max_frames}")

        joint_rot_6d = self.data['joint_rot'][start:end].detach().cpu().float().view(-1, 24, 6)
        local_poses = rotation_6d_to_matrix(joint_rot_6d).view(-1, 24, 3, 3).contiguous()

        body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))
        print(f"Visualizing GT SMPL without translation: frames [{start}:{end}] ({end - start} frames)")
        body_model.view_motion_web_with_gt([local_poses], None, None, None, None, None, fps=fps)

    def vis_pose(self, max_frames = 1000):
        self.vis_gt_smpl_without_translation(start=0, max_frames=max_frames, fps=60)

    def plot_body_smpl_joint15_rotation(
        self,
        start=0,
        max_frames=1000,
        save_path="data_vis/nymeria_body_smpl_joint15_rotation.png",
        unwrap=True,
    ):
        """
        Plot yaw/pitch/roll of self.body_smpl[:, 15, :, :].
        """
        if not hasattr(self, 'body_smpl') or self.body_smpl is None:
            raise ValueError("'self.body_smpl' is not available in dataset")

        if self.body_smpl.ndim != 4 or self.body_smpl.shape[1] <= 15 or self.body_smpl.shape[-2:] != (3, 3):
            raise ValueError(f"Expected self.body_smpl shape [N, >=16, 3, 3], got {tuple(self.body_smpl.shape)}")

        total_len = self.body_smpl.shape[0]
        end = min(start + max_frames, total_len)
        if end - start < 2:
            raise ValueError("Need at least 2 frames to plot self.body_smpl[:,15]")

        t_src = self.data.get('time_pose', self.data.get('time', None))
        if t_src is None:
            time = np.arange(end - start, dtype=np.float32)
        else:
            time = t_src[start:end].detach().cpu().numpy()

        rot_joint15 = self.body_smpl[start:end, 15].detach().cpu()
        euler_ypr = _rotmat_to_euler_zyx_deg(rot_joint15)
        if hasattr(euler_ypr, 'detach'):
            euler_ypr = euler_ypr.detach().cpu().numpy()

        if unwrap:
            euler_ypr = _unwrap_euler_deg(euler_ypr)

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        labels = ['yaw', 'pitch', 'roll']

        for i, lbl in enumerate(labels):
            axes[i].plot(time, euler_ypr[:, i], linewidth=1.4, label=f'body_smpl[:,15] {lbl}')
            axes[i].set_ylabel(f'{lbl} (deg)')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='best')

        axes[-1].set_xlabel('time (s)')
        title_suffix = ' (Unwrapped)' if unwrap else ''
        fig.suptitle(f"self.body_smpl[:,15] Rotation (Yaw/Pitch/Roll){title_suffix}", fontsize=13)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"Saved self.body_smpl[:,15] rotation plot to: {save_path}")

    def plot_head_imu_comparison(self, start=0, max_frames=1000, save_path=None, rotation_save_path=None):
        end = min(start + max_frames, self.get_length())
        time = self.data['time'][start:end].detach().cpu().numpy()

        acc_imu = self.data['head_acc_imu_global'][start:end].detach().cpu().numpy()
        acc_syn = self.data['head_acc_from_vertex'][start:end].detach().cpu().numpy()

        gyro_imu = self.data['head_gyro_imu_global'][start:end].detach().cpu().numpy()
        gyro_syn = self.data['head_gyro_from_body'][start:end].detach().cpu().numpy()

        rot_imu = self.data['gt_orientation'][start:end].matrix().detach().cpu()
        rot_syn = self.data['head_rot_from_body'][start:end].detach().cpu()
        euler_imu = _rotmat_to_euler_zyx_deg(rot_imu).numpy()  # [N, 3] yaw,pitch,roll
        euler_syn = _rotmat_to_euler_zyx_deg(rot_syn).numpy()  # [N, 3] yaw,pitch,roll

        fig, axes = plt.subplots(2, 3, figsize=(18, 7), sharex='col')
        labels = ['x', 'y', 'z']

        for i in range(3):
            axes[0, i].plot(time, acc_imu[:, i], label='self.data head acc', linewidth=1.2)
            axes[0, i].plot(time, acc_syn[:, i], label='syn_imu_from_smpl head acc', linewidth=1.2)
            axes[0, i].set_ylabel(f'acc_{labels[i]}')
            axes[0, i].grid(True, alpha=0.3)

            axes[1, i].plot(time, gyro_imu[:, i], label='self.data head gyro', linewidth=1.2)
            axes[1, i].plot(time, gyro_syn[:, i], label='syn_imu_from_smpl head gyro', linewidth=1.2)
            axes[1, i].set_ylabel(f'gyro_{labels[i]}')
            axes[1, i].set_xlabel('time')
            axes[1, i].grid(True, alpha=0.3)

        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, legend_labels, loc='upper right')

        fig.suptitle('Head IMU comparison (acc + gyro): self.data vs syn_imu_from_smpl', fontsize=13)
        fig.tight_layout(rect=[0, 0, 0.98, 0.96])

        fig_rot, axes_rot = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        rot_labels = ['yaw', 'pitch', 'roll']
        for i in range(3):
            axes_rot[i].plot(time, euler_imu[:, i], label=f'self.data {rot_labels[i]}', linewidth=1.2)
            axes_rot[i].plot(time, euler_syn[:, i], label=f'syn_imu_from_smpl {rot_labels[i]}', linewidth=1.2)
            axes_rot[i].set_ylabel(f'{rot_labels[i]} (deg)')
            axes_rot[i].grid(True, alpha=0.3)
        axes_rot[-1].set_xlabel('time')
        handles_rot, labels_rot = axes_rot[0].get_legend_handles_labels()
        if handles_rot:
            fig_rot.legend(handles_rot, labels_rot, loc='upper right')
        fig_rot.suptitle('Head rotation Euler comparison (yaw, pitch, roll)', fontsize=13)
        fig_rot.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=200)
            if rotation_save_path is None:
                root, ext = os.path.splitext(save_path)
                rotation_save_path = f"{root}_rotation{ext}" if ext else f"{save_path}_rotation.png"

        if rotation_save_path is not None:
            fig_rot.savefig(rotation_save_path, dpi=200)

        plt.show()

    def print_and_plot_right_hip_acc(
        self,
        start=0,
        max_frames=1000,
        save_path="data_vis/nymeria_right_hip_acc.png",
    ):
        if 'right_hip_acc' not in self.data:
            raise ValueError("'right_hip_acc' is not available in dataset")
        if 'time' not in self.data:
            raise ValueError("'time' is not available in dataset")

        print("self.data['right_hip_acc']:")
        print(self.data['right_hip_acc'])

        end = min(start + max_frames, self.get_length())
        if end - start < 2:
            raise ValueError("Need at least 2 frames to plot right_hip_acc")

        time = self.data['time'][start:end].detach().cpu().numpy()
        hip_acc = self.data['right_hip_acc'][start:end].detach().cpu().numpy()

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        labels = ['x', 'y', 'z']

        for i in range(3):
            axes[i].plot(time, hip_acc[:, i], linewidth=1.4, label=f'right_hip_acc_{labels[i]}')
            axes[i].set_ylabel(f'acc_{labels[i]} (m/s²)')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='best')

        axes[3].plot(time, np.linalg.norm(hip_acc, axis=1), linewidth=1.4, label='|right_hip_acc|')
        axes[3].set_ylabel('|acc| (m/s²)')
        axes[3].set_xlabel('time (s)')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc='best')

        fig.suptitle('Right Hip Acceleration', fontsize=13)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"Saved right_hip_acc plot to: {save_path}")

    def plot_right_hip_acc_histogram(
        self,
        start=0,
        max_frames=1000,
        bins=80,
        save_path="data_vis/nymeria_right_hip_acc_hist.png",
    ):
        if 'right_hip_acc' not in self.data:
            raise ValueError("'right_hip_acc' is not available in dataset")

        end = min(start + max_frames, self.get_length())
        if end - start < 2:
            raise ValueError("Need at least 2 frames to plot right_hip_acc histogram")

        hip_acc = self.data['right_hip_acc'][start:end].detach().cpu().numpy()

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
        labels = ['x', 'y', 'z']

        for i, axis_name in enumerate(labels):
            axes[i].hist(hip_acc[:, i], bins=bins, alpha=0.8)
            axes[i].set_ylabel('count')
            axes[i].set_xlabel(f'acc_{axis_name} (m/s²)')
            axes[i].set_title(f'right_hip_acc {axis_name}-axis histogram')
            axes[i].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"Saved right_hip_acc histogram plot to: {save_path}")

    def plot_mobileposer_hip_noise_comparison(
        self,
        start=0,
        max_frames=1000,
        noise_std_deg=5.0,
        acc_noise_sigma=0.03,
        save_path="data_vis/nymeria_mobileposer_hip_noise_comparison.png",
        seed=0,
    ):
        if 'mobileposer_input' not in self.data:
            raise ValueError("'mobileposer_input' is not available in dataset")
        if 'time' not in self.data:
            raise ValueError("'time' is not available in dataset")

        end = min(start + max_frames, self.get_length())
        if end - start < 2:
            raise ValueError("Need at least 2 frames to plot mobileposer noise comparison")

        time = self.data['time'][start:end].detach().cpu().numpy()
        mobileposer_input = self.data['mobileposer_input'][start:end].detach().cpu()

        # Hip slot in mobileposer_input (index 3): [acc(3), rot(9)]
        hip_acc_before = mobileposer_input[:, 12 * 3:12 * 3 + 3].clone()
        hip_rot_before = mobileposer_input[:, 12 * 3 + 3:12 * 3 + 12].clone()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Orientation noise (fixed rotation perturbation, in degrees)
        noise_std_rad = noise_std_deg * torch.pi / 180.0
        fixed_noise_axis_angle = torch.randn(
            3, device=hip_rot_before.device, dtype=hip_rot_before.dtype
        ) * noise_std_rad
        fixed_noise_rot = pp.so3(fixed_noise_axis_angle.unsqueeze(0)).Exp().matrix().squeeze(0)  # [3, 3]

        rot_before_mat = hip_rot_before.reshape(-1, 3, 3)
        rot_after_mat = torch.matmul(fixed_noise_rot.unsqueeze(0), rot_before_mat)
        hip_rot_after = rot_after_mat.reshape(-1, 9)

        # Acceleration noise: apply the same fixed rotation perturbation
        hip_acc_after = torch.matmul(
            fixed_noise_rot.unsqueeze(0),
            hip_acc_before.unsqueeze(-1),
        ).squeeze(-1)

        hip_acc_before = hip_acc_before.numpy()
        hip_acc_after = hip_acc_after.numpy()

        rot_before_mat = hip_rot_before.reshape(-1, 3, 3)
        rot_after_mat = hip_rot_after.reshape(-1, 3, 3)
        euler_before = _rotmat_to_euler_zyx_deg(rot_before_mat).detach().cpu().numpy()
        euler_after = _rotmat_to_euler_zyx_deg(rot_after_mat).detach().cpu().numpy()
        # euler_before = _unwrap_euler_deg(euler_before)
        # euler_after = _unwrap_euler_deg(euler_after)

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        root, ext = os.path.splitext(save_path)
        if ext == "":
            ext = ".png"
        rot_save_path = f"{root}_rot{ext}"
        acc_save_path = f"{root}_acc{ext}"

        # -------- Rotation comparison (Yaw/Pitch/Roll) --------
        fig_rot, axes_rot = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        rot_labels = ['yaw', 'pitch', 'roll']
        for i in range(3):
            axes_rot[i].plot(time, euler_before[:, i], linewidth=1.2, label='before fixed rotation noise')
            axes_rot[i].plot(time, euler_after[:, i], linewidth=1.0, label='after fixed rotation noise', linestyle='--')
            axes_rot[i].set_ylabel(f'{rot_labels[i]} (deg)')
            axes_rot[i].grid(True, alpha=0.3)
            if i == 0:
                axes_rot[i].legend(loc='best')
        axes_rot[-1].set_xlabel('time (s)')
        fig_rot.suptitle(
            f"MobilePoser Hip Rotation Before/After Fixed Orientation Noise (std={noise_std_deg} deg)",
            fontsize=13,
        )
        fig_rot.tight_layout()
        fig_rot.savefig(rot_save_path, dpi=200)
        plt.close(fig_rot)

        # -------- Acceleration comparison --------
        fig_acc, axes_acc = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        acc_labels = ['acc_x', 'acc_y', 'acc_z']
        for i in range(3):
            axes_acc[i].plot(time, hip_acc_before[:, i], linewidth=1.2, label='hip_acc before fixed rotation noise')
            axes_acc[i].plot(time, hip_acc_after[:, i], linewidth=1.0, label='hip_acc after fixed rotation noise', linestyle='--')
            axes_acc[i].set_ylabel(acc_labels[i])
            axes_acc[i].grid(True, alpha=0.3)
            axes_acc[i].legend(loc='best')

        axes_acc[3].plot(time, np.linalg.norm(hip_acc_before, axis=1), linewidth=1.2, label='|hip_acc| before fixed rotation noise')
        axes_acc[3].plot(time, np.linalg.norm(hip_acc_after, axis=1), linewidth=1.0, label='|hip_acc| after fixed rotation noise', linestyle='--')
        axes_acc[3].set_ylabel('|acc|')
        axes_acc[3].set_xlabel('time (s)')
        axes_acc[3].grid(True, alpha=0.3)
        axes_acc[3].legend(loc='best')

        fig_acc.suptitle(
            f"MobilePoser Hip Acc Before/After Fixed Rotation Noise (std={noise_std_deg} deg)",
            fontsize=13,
        )
        fig_acc.tight_layout()
        fig_acc.savefig(acc_save_path, dpi=200)
        plt.close(fig_acc)

        print(f"Saved hip rotation noise comparison plot to: {rot_save_path}")
        print(f"Saved hip acceleration noise comparison plot to: {acc_save_path}")

    def plot_right_hip_fixed_rotation_noise_comparison(
        self,
        start=0,
        max_frames=1000,
        noise_std_deg=10.0,
        save_path="data_vis/nymeria_right_hip_fixed_rotation_noise_comparison.png",
        seed=0,
        apply_to_data=False,
    ):
        required_keys = ['time', 'right_hip_acc', 'right_hip_rot_mat']
        for k in required_keys:
            if k not in self.data:
                raise ValueError(f"'{k}' is not available in dataset")

        end = min(start + max_frames, self.get_length())
        if end - start < 2:
            raise ValueError("Need at least 2 frames to plot right hip fixed-noise comparison")

        time = self.data['time'][start:end].detach().cpu().numpy()
        hip_acc_before = self.data['right_hip_acc'][start:end].detach().cpu().clone()
        hip_rot_before = self.data['right_hip_rot_mat'][start:end].detach().cpu().clone()

        hip_acc_after, hip_rot_after, fixed_noise_rot = apply_fixed_rotation_noise_to_acc_and_rotmat(
            hip_acc=hip_acc_before,
            hip_rot_mat=hip_rot_before,
            noise_std_deg=noise_std_deg,
            seed=seed,
        )

        if apply_to_data:
            self.data['right_hip_acc'][start:end] = hip_acc_after.to(self.data['right_hip_acc'].device)
            self.data['right_hip_rot_mat'][start:end] = hip_rot_after.to(self.data['right_hip_rot_mat'].device)
            if 'right_hip_rot' in self.data and self.data['right_hip_rot'] is not None:
                self.data['right_hip_rot'][start:end] = matrix_to_rotation_6d(hip_rot_after).reshape(-1, 6).to(self.data['right_hip_rot'].device)

        hip_acc_before_np = hip_acc_before.numpy()
        hip_acc_after_np = hip_acc_after.numpy()
        euler_before = _rotmat_to_euler_zyx_deg(hip_rot_before).detach().cpu().numpy()
        euler_after = _rotmat_to_euler_zyx_deg(hip_rot_after).detach().cpu().numpy()

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        root, ext = os.path.splitext(save_path)
        if ext == "":
            ext = ".png"
        rot_save_path = f"{root}_rot{ext}"
        acc_save_path = f"{root}_acc{ext}"

        fig_rot, axes_rot = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        rot_labels = ['yaw', 'pitch', 'roll']
        for i in range(3):
            axes_rot[i].plot(time, euler_before[:, i], linewidth=1.2, label='right_hip_rot_mat before fixed noise')
            axes_rot[i].plot(time, euler_after[:, i], linewidth=1.0, label='right_hip_rot_mat after fixed noise', linestyle='--')
            axes_rot[i].set_ylabel(f'{rot_labels[i]} (deg)')
            axes_rot[i].grid(True, alpha=0.3)
            if i == 0:
                axes_rot[i].legend(loc='best')
        axes_rot[-1].set_xlabel('time (s)')
        fig_rot.suptitle(
            f"Right Hip Rotation Before/After Fixed Orientation Noise (std={noise_std_deg} deg)",
            fontsize=13,
        )
        fig_rot.tight_layout()
        fig_rot.savefig(rot_save_path, dpi=200)
        plt.close(fig_rot)

        fig_acc, axes_acc = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        acc_labels = ['acc_x', 'acc_y', 'acc_z']
        for i in range(3):
            axes_acc[i].plot(time, hip_acc_before_np[:, i], linewidth=1.2, label='right_hip_acc before fixed noise')
            axes_acc[i].plot(time, hip_acc_after_np[:, i], linewidth=1.0, label='right_hip_acc after fixed noise', linestyle='--')
            axes_acc[i].set_ylabel(acc_labels[i])
            axes_acc[i].grid(True, alpha=0.3)
            axes_acc[i].legend(loc='best')

        axes_acc[3].plot(time, np.linalg.norm(hip_acc_before_np, axis=1), linewidth=1.2, label='|right_hip_acc| before fixed noise')
        axes_acc[3].plot(time, np.linalg.norm(hip_acc_after_np, axis=1), linewidth=1.0, label='|right_hip_acc| after fixed noise', linestyle='--')
        axes_acc[3].set_ylabel('|acc|')
        axes_acc[3].set_xlabel('time (s)')
        axes_acc[3].grid(True, alpha=0.3)
        axes_acc[3].legend(loc='best')

        fig_acc.suptitle(
            f"Right Hip Acc Before/After Fixed Rotation Noise (std={noise_std_deg} deg)",
            fontsize=13,
        )
        fig_acc.tight_layout()
        fig_acc.savefig(acc_save_path, dpi=200)
        plt.close(fig_acc)

        print(f"Fixed noise rotation matrix used:\n{fixed_noise_rot.detach().cpu().numpy()}")
        print(f"Saved right hip rotation fixed-noise comparison plot to: {rot_save_path}")
        print(f"Saved right hip acceleration fixed-noise comparison plot to: {acc_save_path}")

    def plot_right_hip_body_v_comparison(
        self,
        start=0,
        max_frames=1000,
        save_path="data_vis/nymeria_right_hip_body_v_comparison.png",
    ):
        required_keys = ['time', 'body_v_acc', 'body_v_acc_2', 'body_v_rot', 'body_v_rot_2']
        for k in required_keys:
            if k not in self.data:
                raise ValueError(f"'{k}' is not available in dataset")

        end = min(start + max_frames, self.get_length())
        if end - start < 2:
            raise ValueError("Need at least 2 frames to plot right hip body_v comparison")

        time = self.data['time'][start:end].detach().cpu().numpy()

        # compare right hip only (index 3)
        acc_ref = self.data['body_v_acc'][start:end, 3, :].detach().cpu().numpy()
        acc_syn = self.data['body_v_acc_2'][start:end, 3, :].detach().cpu().numpy()

        rot_ref = self.data['body_v_rot'][start:end, 3, :, :].detach().cpu()
        rot_syn = self.data['body_v_rot_2'][start:end, 3, :, :].detach().cpu()
        euler_ref = _rotmat_to_euler_zyx_deg(rot_ref).numpy()  # [N, 3] yaw,pitch,roll
        euler_syn = _rotmat_to_euler_zyx_deg(rot_syn).numpy()  # [N, 3] yaw,pitch,roll

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)

        acc_labels = ['x', 'y', 'z']
        for i in range(3):
            axes[0, i].plot(time, acc_ref[:, i], linewidth=1.3, label='body_v_acc[:,3,:]')
            axes[0, i].plot(time, acc_syn[:, i], linewidth=0.8, label='body_v_acc_2[:,3,:]', linestyle='--')
            axes[0, i].set_ylabel(f'acc_{acc_labels[i]} (m/s²)')
            axes[0, i].grid(True, alpha=0.3)
            if i == 0:
                axes[0, i].legend(loc='best')

        rot_labels = ['yaw', 'pitch', 'roll']
        for i in range(3):
            axes[1, i].plot(time, euler_ref[:, i], linewidth=1.3, label='body_v_rot[:,3,:,:]')
            axes[1, i].plot(time, euler_syn[:, i], linewidth=0.8, label='body_v_rot_2[:,3,:,:]', linestyle='--')
            axes[1, i].set_ylabel(f'{rot_labels[i]} (deg)')
            axes[1, i].set_xlabel('time (s)')
            axes[1, i].grid(True, alpha=0.3)
            if i == 0:
                axes[1, i].legend(loc='best')

        fig.suptitle('Right Hip Comparison: body_v_acc/body_v_rot vs _2', fontsize=13)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"Saved right hip body_v comparison plot to: {save_path}")

    def _unwrap_euler_deg(euler_deg: np.ndarray) -> np.ndarray:
        """
        Unwrap Euler angles along time axis.

        Args:
            euler_deg: [N, 3] array in degrees, columns = yaw, pitch, roll

        Returns:
            [N, 3] array in degrees after temporal unwrapping
        """
        if euler_deg.ndim != 2 or euler_deg.shape[1] != 3:
            raise ValueError(f"Expected euler_deg shape [N, 3], got {euler_deg.shape}")

        euler_rad = np.deg2rad(euler_deg)
        euler_rad_unwrapped = np.unwrap(euler_rad, axis=0)
        return np.rad2deg(euler_rad_unwrapped)


    def plot_orientation_comparison_ypr(
        self,
        start=0,
        max_frames=1000,
        save_path="data_vis/nymeria_orientation_comparison_ypr.png",
        unwrap=True,
    ):
        end = min(start + max_frames, self.get_length())
        if end - start < 2:
            raise ValueError("Need at least 2 frames to plot orientation comparison")

        required_keys = ['time', 'gt_orientation', 'right_hip_rot_mat', 'head_rot_from_body']
        for k in required_keys:
            if k not in self.data:
                raise ValueError(f"'{k}' is not available in dataset")

        # time
        time = self.data['time'][start:end].detach().cpu().numpy()

        # rotations
        # gt_orientation is assumed to be a PyPose SO3-like object supporting .matrix()
        rot_gt = self.data['gt_orientation'][start:end].matrix().detach().cpu().numpy()   # [N, 3, 3]
        rot_hip = self.data['right_hip_rot_mat'][start:end].detach().cpu().numpy()        # [N, 3, 3]
        rot_head = self.data['head_rot_from_body'][start:end].detach().cpu().numpy()       # [N, 3, 3]

        # convert to Euler ZYX (yaw, pitch, roll) in degrees
        euler_gt = _rotmat_to_euler_zyx_deg(rot_gt)       # [N, 3]
        euler_hip = _rotmat_to_euler_zyx_deg(rot_hip)     # [N, 3]
        euler_head = _rotmat_to_euler_zyx_deg(rot_head)   # [N, 3]

        # in case _rotmat_to_euler_zyx_deg returns torch.Tensor
        if hasattr(euler_gt, "detach"):
            euler_gt = euler_gt.detach().cpu().numpy()
        if hasattr(euler_hip, "detach"):
            euler_hip = euler_hip.detach().cpu().numpy()
        if hasattr(euler_head, "detach"):
            euler_head = euler_head.detach().cpu().numpy()

        if unwrap:
            euler_gt = _unwrap_euler_deg(euler_gt)
            euler_hip = _unwrap_euler_deg(euler_hip)
            euler_head = _unwrap_euler_deg(euler_head)

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        labels = ['yaw', 'pitch', 'roll']

        for i in range(3):
            axes[i].plot(time, euler_gt[:, i], label='gt_orientation', linewidth=1.4)
            axes[i].plot(time, euler_hip[:, i], label='right_hip_rot_mat', linewidth=1.2)
            axes[i].plot(time, euler_head[:, i], label='head_rot_from_body', linewidth=1.2)
            axes[i].set_ylabel(f'{labels[i]} (deg)')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='best')

        axes[-1].set_xlabel('time (s)')
        title_suffix = ' (Unwrapped)' if unwrap else ''
        fig.suptitle(f'Orientation Comparison (Yaw / Pitch / Roll){title_suffix}', fontsize=13)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"Saved orientation comparison plot to: {save_path}")

    def plot_body_v_rot_index5_ypr(
        self,
        start=0,
        max_frames=1000,
        save_path="data_vis/nymeria_body_v_rot_idx5_ypr.png",
        unwrap=True,
    ):
        """
        Plot yaw/pitch/roll of self.data['body_v_rot'][:, 5, :, :].
        """
        if 'body_v_rot' not in self.data:
            raise ValueError("'body_v_rot' is not available in dataset")
        if 'time' not in self.data:
            raise ValueError("'time' is not available in dataset")

        end = min(start + max_frames, self.get_length())
        if end - start < 2:
            raise ValueError("Need at least 2 frames to plot body_v_rot[:,5,:,:]")

        time = self.data['time'][start:end].detach().cpu().numpy()
        rot_idx5 = self.data['body_v_rot'][start:end, 5, :, :].detach().cpu()  # [N, 3, 3]

        euler_ypr = _rotmat_to_euler_zyx_deg(rot_idx5)
        if hasattr(euler_ypr, "detach"):
            euler_ypr = euler_ypr.detach().cpu().numpy()

        if unwrap:
            euler_ypr = _unwrap_euler_deg(euler_ypr)

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        labels = ['yaw', 'pitch', 'roll']

        for i in range(3):
            axes[i].plot(time, euler_ypr[:, i], linewidth=1.4, label=f'body_v_rot[:,5] {labels[i]}')
            axes[i].set_ylabel(f'{labels[i]} (deg)')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='best')

        axes[-1].set_xlabel('time (s)')
        title_suffix = ' (Unwrapped)' if unwrap else ''
        fig.suptitle(f"body_v_rot[:,5,:,:] Yaw/Pitch/Roll{title_suffix}", fontsize=13)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"Saved body_v_rot[:,5,:,:] yaw/pitch/roll plot to: {save_path}")

    def plot_relative_rotation_error(
        self,
        start=0,
        max_frames=1000,
        save_path="data_vis/nymeria_relative_rotation_error.png",
    ):
        end = min(start + max_frames, self.get_length())
        if end - start < 2:
            raise ValueError("Need at least 2 frames to plot relative rotation error")

        required_keys = ['time', 'gt_orientation', 'head_rot_from_body', 'right_hip_rot_mat']
        for k in required_keys:
            if k not in self.data:
                raise ValueError(f"'{k}' is not available in dataset")

        time = self.data['time'][start:end].detach().cpu().numpy()

        rot_gt = self.data['gt_orientation'][start:end].matrix().detach().cpu()     # [N, 3, 3]
        rot_head = self.data['head_rot_from_body'][start:end].detach().cpu()         # [N, 3, 3]
        rot_hip = self.data['right_hip_rot_mat'][start:end].detach().cpu()           # [N, 3, 3]

        yaw_gt = _rotmat_to_euler_zyx_deg(rot_gt).detach().cpu().numpy()[:, 0]
        yaw_head = _rotmat_to_euler_zyx_deg(rot_head).detach().cpu().numpy()[:, 0]
        yaw_hip = _rotmat_to_euler_zyx_deg(rot_hip).detach().cpu().numpy()[:, 0]

        err_gt_head = (yaw_gt - yaw_head + 180.0) % 360.0 - 180.0
        err_hip_head = (yaw_hip - yaw_head + 180.0) % 360.0 - 180.0

        abs_err_gt_head = np.abs(err_gt_head)
        abs_err_hip_head = np.abs(err_hip_head)

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        axes[0].plot(time, err_gt_head, label='gt_orientation vs head_rot_from_body', linewidth=1.4)
        axes[0].set_ylabel('yaw diff (deg)')
        axes[0].set_title(
            f"GT vs Head (Yaw) | MAE={abs_err_gt_head.mean():.2f}°, median|.|={np.median(abs_err_gt_head):.2f}°, max|.|={abs_err_gt_head.max():.2f}°"
        )
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best')

        axes[1].plot(time, err_hip_head, label='right_hip_rot_mat vs head_rot_from_body', linewidth=1.4)
        axes[1].set_ylabel('yaw diff (deg)')
        axes[1].set_xlabel('time (s)')
        axes[1].set_title(
            f"Hip vs Head (Yaw) | MAE={abs_err_hip_head.mean():.2f}°, median|.|={np.median(abs_err_hip_head):.2f}°, max|.|={abs_err_hip_head.max():.2f}°"
        )
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best')

        fig.suptitle('Relative Rotation Difference (Yaw Only)', fontsize=13)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"Saved relative rotation error plot to: {save_path}")

    def plot_velocity_estimated_from_acc(self, start=0, max_frames=1000, save_path="data_vis/nymeria_velocity_from_acc.png"):
        end = min(start + max_frames, self.get_length())
        if end - start < 2:
            raise ValueError("Need at least 2 frames to integrate acceleration")

        acc = self.data['acc'][start:end].detach().cpu()
        dt = self.data['dt'][start:end].detach().cpu()
        if dt.ndim > 1:
            dt = dt.reshape(-1)

        if 'velocity' in self.data and self.data['velocity'].shape[0] > start:
            init_vel = self.data['velocity'][start].detach().cpu().to(acc.dtype)
        else:
            init_vel = torch.zeros(3, dtype=acc.dtype)

        vel_est = torch.zeros_like(acc)
        vel_est[0] = init_vel
        for i in range(1, vel_est.shape[0]):
            dti = dt[i].clamp(min=1e-6)
            vel_est[i] = vel_est[i - 1] + acc[i - 1] * dti

        t = torch.cumsum(dt, dim=0).numpy()
        vel_np = vel_est.numpy()

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        labels = ['x', 'y', 'z']
        for i in range(3):
            axes[i].plot(t, vel_np[:, i], linewidth=1.4, label=f'estimated vel_{labels[i]}')
            axes[i].set_ylabel(f'vel_{labels[i]} (m/s)')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='best')

        axes[3].plot(t, torch.norm(vel_est, dim=-1).numpy(), linewidth=1.4, label='estimated |vel|')
        axes[3].set_ylabel('|vel| (m/s)')
        axes[3].set_xlabel('time')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc='best')

        fig.suptitle('Velocity Estimated from Integrated Acceleration', fontsize=13)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"Saved velocity-estimated-from-acc plot to: {save_path}")

    def plot_velocity_estimated_vs_gt(self, start=0, max_frames=1000, save_path="data_vis/nymeria_velocity_from_acc_vs_gt.png"):
        end = min(start + max_frames, self.get_length())
        if end - start < 2:
            raise ValueError("Need at least 2 frames to integrate acceleration")

        if 'velocity' not in self.data:
            raise ValueError("GT velocity ('velocity') is not available in dataset")

        acc = self.data['acc'][start:end].detach().cpu()
        dt = self.data['dt'][start:end].detach().cpu()
        gt_vel = self.data['velocity'][start:end].detach().cpu().to(acc.dtype)

        if dt.ndim > 1:
            dt = dt.reshape(-1)

        init_vel = gt_vel[0]

        vel_est = torch.zeros_like(acc)
        vel_est[0] = init_vel
        for i in range(1, vel_est.shape[0]):
            dti = dt[i].clamp(min=1e-6)
            vel_est[i] = vel_est[i - 1] + acc[i - 1] * dti

        t = torch.cumsum(dt, dim=0).numpy()
        vel_est_np = vel_est.numpy()
        gt_vel_np = gt_vel.numpy()

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        labels = ['x', 'y', 'z']
        for i in range(3):
            axes[i].plot(t, vel_est_np[:, i], linewidth=1.4, label=f'est vel_{labels[i]}')
            axes[i].plot(t, gt_vel_np[:, i], linewidth=1.2, alpha=0.9, label=f'gt vel_{labels[i]}')
            axes[i].set_ylabel(f'vel_{labels[i]} (m/s)')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='best')

        axes[3].plot(t, torch.norm(vel_est, dim=-1).numpy(), linewidth=1.4, label='est |vel|')
        axes[3].plot(t, torch.norm(gt_vel, dim=-1).numpy(), linewidth=1.2, alpha=0.9, label='gt |vel|')
        axes[3].set_ylabel('|vel| (m/s)')
        axes[3].set_xlabel('time')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc='best')

        fig.suptitle('Velocity Comparison: Estimated from Acc vs GT Velocity', fontsize=13)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"Saved velocity comparison plot to: {save_path}")

    def plot_trajectory_from_acc_double_integration_vs_gt(
        self,
        start=0,
        max_frames=1000,
        save_path="data_vis/nymeria_trajectory_from_acc_vs_gt_xz.png",
    ):
        end = min(start + max_frames, self.get_length())
        if end - start < 2:
            raise ValueError("Need at least 2 frames to integrate acceleration into trajectory")

        if 'gt_translation' not in self.data:
            raise ValueError("GT trajectory ('gt_translation') is not available in dataset")

        acc = self.data['acc'][start:end].detach().cpu()
        dt = self.data['dt'][start:end].detach().cpu()
        gt_traj = self.data['gt_translation'][start:end].detach().cpu().to(acc.dtype)

        if dt.ndim > 1:
            dt = dt.reshape(-1)

        if 'velocity' in self.data and self.data['velocity'].shape[0] > start:
            init_vel = self.data['velocity'][start].detach().cpu().to(acc.dtype)
        else:
            init_vel = torch.zeros(3, dtype=acc.dtype)

        traj_est = torch.zeros_like(gt_traj)
        vel_est = torch.zeros_like(acc)
        traj_est[0] = gt_traj[0]
        vel_est[0] = init_vel

        for i in range(1, traj_est.shape[0]):
            dti = dt[i].clamp(min=1e-6)
            vel_est[i] = vel_est[i - 1] + acc[i - 1] * dti
            traj_est[i] = traj_est[i - 1] + vel_est[i - 1] * dti + 0.5 * acc[i - 1] * dti * dti

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        gt_np = gt_traj.numpy()
        est_np = traj_est.numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        planes = [
            (0, 2, 'X (m)', 'Z (m)', 'XZ'),
            (0, 1, 'X (m)', 'Y (m)', 'XY'),
            (1, 2, 'Y (m)', 'Z (m)', 'YZ'),
        ]

        for ax, (ix, iy, xlabel, ylabel, plane_name) in zip(axes, planes):
            ax.plot(gt_np[:, ix], gt_np[:, iy], linewidth=2.0, label='gt trajectory')
            ax.plot(est_np[:, ix], est_np[:, iy], linewidth=1.8, label='acc double integration trajectory')
            ax.scatter(gt_np[0, ix], gt_np[0, iy], c='green', s=50, label='start')
            ax.scatter(gt_np[-1, ix], gt_np[-1, iy], c='red', s=50, label='gt end')
            ax.scatter(est_np[-1, ix], est_np[-1, iy], c='orange', s=50, label='est end')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{plane_name} View')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right')
        fig.suptitle('Trajectory: Acc Double Integration vs GT (XZ / XY / YZ)', fontsize=13)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"Saved trajectory comparison plot to: {save_path}")

        return {'traj_est': traj_est, 'traj_gt': gt_traj, 'vel_est': vel_est}

    def plot_updown_label_and_gt_trajectory_xyz(
        self,
        start=0,
        max_frames=1000,
        save_path="data_vis/nymeria_updown_and_gt_xyz.png",
    ):
        end = min(start + max_frames, self.get_length())
        if end - start < 2:
            raise ValueError("Need at least 2 frames to plot")

        if 'gt_translation' not in self.data:
            raise ValueError("GT trajectory ('gt_translation') is not available in dataset")
        if 'updown_label' not in self.data:
            raise ValueError("'updown_label' is not available in dataset")

        time = self.data['time'][start:end].detach().cpu().numpy()
        gt_traj = self.data['gt_translation'][start:end].detach().cpu().numpy()  # [N, 3]
        updown = self.data['updown_label'][start:end].detach().cpu().numpy()     # [N]
        root_traj = None
        if 'root_translation' in self.data:
            root_traj = self.data['root_translation'][start:end].detach().cpu().numpy()  # [N, 3]
        left_foot_y = None
        right_foot_y = None
        if 'left_foot_y' in self.data and 'right_foot_y' in self.data:
            left_foot_y = self.data['left_foot_y'][start:end].detach().cpu().numpy()
            right_foot_y = self.data['right_foot_y'][start:end].detach().cpu().numpy()
        elif 'joint_pos' in self.data:
            # Fallback: use local joint positions if explicit foot-y tracks are unavailable.
            joint_pos = self.data['joint_pos'][start:end].detach().cpu().numpy()
            left_foot_y = joint_pos[:, 10 * 3 + 1]
            right_foot_y = joint_pos[:, 11 * 3 + 1]

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        axes[0].plot(time, gt_traj[:, 1], label='gt y', linewidth=1.3)
        if root_traj is not None:
            axes[0].plot(time, root_traj[:, 1], label='root y', linewidth=1.3)
        if left_foot_y is not None and right_foot_y is not None:
            axes[0].plot(time, left_foot_y, label='left foot y', linewidth=1.1, alpha=0.9)
            axes[0].plot(time, right_foot_y, label='right foot y', linewidth=1.1, alpha=0.9)
        axes[0].set_ylabel('position (m)')
        axes[0].set_title('GT/Root/Foot trajectory (y)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best')

        axes[1].step(time, updown, where='post', label='updown_label', linewidth=1.4)
        axes[1].set_ylabel('updown label')
        axes[1].set_xlabel('time')
        axes[1].set_yticks([0, 1])
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].set_title('Up/down motion label')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best')

        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"Saved updown + GT trajectory plot to: {save_path}")

    
if __name__ == "__main__":
    data_root="/mnt/nas2/processed_data_test_body_Ty"
    data_name="20231222_s0_denise_carter_act6_26wfkz_cpfbody_imu50hz_pose50hz"
    
    # 20231222_s0_denise_carter_act2_d8mdi5_cpfbody_imu50hz_pose50hz
    # 20231222_s0_denise_carter_act6_26wfkz_cpfbody_imu50hz_pose50hz
    
    # 20230911_s0_angela_gomez_act3_8wxz7b_cpfbody_imu50hz_pose50hz
    coordinate="glob_coord"

    # Load dataset
    dataset = nymeria(
        data_root=data_root,
        data_name=data_name,
        coordinate=coordinate,
        remove_g=True,
        maximum_length=50000000,
        target_hz=50,
    )
    # dataset.plot_head_imu_comparison(max_frames=300, save_path="data_vis/head_imu_comparison.png")
    # dataset.plot_velocity_estimated_from_acc(max_frames=300, save_path="data_vis/nymeria_velocity_from_acc.png")
    # dataset.plot_velocity_estimated_vs_gt(max_frames=300, save_path="data_vis/nymeria_velocity_from_acc_vs_gt.png")
    # dataset.plot_trajectory_from_acc_double_integration_vs_gt(max_frames=300, save_path="data_vis/nymeria_trajectory_from_acc_vs_gt_xz.png")
    # dataset.plot_updown_label_and_gt_trajectory_xyz(max_frames=300000, save_path="data_vis/nymeria_updown_and_gt_xyz.png")
    dataset.print_and_plot_right_hip_acc(max_frames=3000, save_path="data_vis/nymeria_right_hip_acc.png")
    dataset.plot_right_hip_acc_histogram(
        max_frames=3000,
        bins=80,
        save_path="data_vis/nymeria_right_hip_acc_hist.png",
    )
    dataset.plot_mobileposer_hip_noise_comparison(
        max_frames=2000,
        save_path="data_vis/nymeria_mobileposer_hip_noise_comparison.png",
        seed=0,
        noise_std_deg=15.0,
    )
    dataset.plot_right_hip_fixed_rotation_noise_comparison(
        max_frames=2000,
        noise_std_deg=10.0,
        save_path="data_vis/nymeria_right_hip_fixed_rotation_noise_comparison.png",
        seed=0,
        apply_to_data=False,
    )
    dataset.plot_right_hip_body_v_comparison(
        max_frames=2000,
        save_path="data_vis/nymeria_right_hip_body_v_comparison.png",
    )
    dataset.plot_orientation_comparison_ypr(max_frames=2000, save_path="data_vis/nymeria_orientation_comparison_ypr.png", unwrap=True)
    dataset.plot_relative_rotation_error(max_frames=2000, save_path="data_vis/nymeria_relative_rotation_error.png")
    dataset.plot_body_v_rot_index5_ypr(
        max_frames=2000,
        save_path="data_vis/nymeria_body_v_rot_idx5_ypr.png",
        unwrap=True,
    )
    # dataset.vis_gt_smpl_without_translation(start=0, max_frames=2000, fps=50)
    dataset.plot_body_smpl_joint15_rotation(start=0, max_frames=2000)


# python -m datasets.nymeriadataset
