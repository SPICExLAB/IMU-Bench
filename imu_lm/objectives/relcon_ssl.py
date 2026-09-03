"""RelCon SSL baseline (Xu et al., ICLR 2025, "RelCon: Relative Contrastive
Learning for a Motion Foundation Model for Wearable Data"), integrated with
the same "different pretext task, same backbone" comparison methodology as
FOCAL/SimMTM -- see ssl_vit_baselines.py's module docstring.

Two-stage design, faithfully porting the real algorithm from the vendored
RelCon repo (baselines/SSL/relcon):

  Stage 1 (auxiliary, frozen afterward) -- MotifDist: trains a small
    cross-attention distance network (their own ``CrossAttn.Net``, imported
    directly from the vendored repo) via query/augmented-key reconstruction,
    using their own published augmentation set (noise/scaling/rotation/
    negate/time_flip/channel_shuffle/time_segment_permutation/time_warp,
    2 of 8 chosen per sample, following https://arxiv.org/abs/2011.11542).
    Lower reconstruction error <=> more motif-similar pair. This is a
    SEPARATE checkpoint, not part of the downstream-facing encoder -- it
    only supplies a ranking signal to Stage 2. Trained standalone via
    scripts/train_relcon_motifdist.py (not through run.py -- it has no
    ViT1DEncoder in the loop at all).

  Stage 2 (the actual SSL baseline evaluated downstream) -- RelCon: trains
    the SHARED ViT1DEncoder (plain, non-alt-attn, matching FOCAL/SimMTM) via
    the paper's ``relative_contrastive_loss`` (ported verbatim from
    relcon/models/RelCon/RelCon_Model.py -- pure PyTorch, dataset-agnostic),
    using the FROZEN Stage-1 MotifDist network to rank each anchor's
    candidate set (every other window in the same training batch -- the
    "rotated batch" trick from their own run_one_epoch) by motif-distance,
    then training the encoder so cosine similarity between anchor/candidate
    embeddings respects that ranking. Wired into run.py's existing
    ``mae_multidata_varlen_altattn_flu`` objective's ssl_method switch
    (alongside "simmtm"/"focal") -- see that branch in run.py.

Documented simplification vs. the original paper: RelCon also uses
"within-user" candidates (other windows from the same participant/session).
This benchmark's generic multidata pretraining loader
(combined_mae_multidata_loader.py) draws windows across all 11 sources
without session-grouped batching, so within-user candidates are not
constructed here -- only the between-window ("rotated batch") candidates
from RelCon_Model.py's run_one_epoch are used (equivalent to their own
``withinuser_cands=0``). Also, like FOCAL/SimMTM in this same codebase
(see ssl_vit_baselines.py's focal_forward_loss/simmtm_forward_loss_varlen),
this does not special-case capture24's zero-padded gyro channels via
channel_valid_mask -- the same minor, accepted skew already documented
there.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.utils.helpers import cfg_get

_RELCON_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "baselines", "SSL", "relcon",
)
if _RELCON_REPO not in sys.path:
    sys.path.insert(0, _RELCON_REPO)

from relcon.nets.CrossAttn.CrossAttn_Net import Net as _CrossAttnNet  # noqa: E402
from relcon.models.MotifDist.utils.augmentations import (  # noqa: E402
    noise_transform, scaling_transform, rotation_transform, negate_transform,
    time_flip_transform, channel_shuffle_transform,
    time_segment_permutation_transform, time_warp_transform,
    axis_angle_to_rotation_matrix_3d,
)


# ══════════════════════════════════════════════════════════════════════════
# Stage 1 -- MotifDist distance network
# ══════════════════════════════════════════════════════════════════════════

_TRANSFORMS = [
    noise_transform, scaling_transform, rotation_transform, negate_transform,
    time_flip_transform, channel_shuffle_transform,
    time_segment_permutation_transform, time_warp_transform,
]


def _apply_transform(x_tc: np.ndarray, fn) -> np.ndarray:
    """x_tc: [T, C] (6 ch: acc_F,acc_L,acc_U,gyro_F,gyro_L,gyro_U).
    rotation_transform is a genuine 3D rotation (needs exactly 3 channels) --
    applied identically to the acc and gyro triplets (rotating the sensor
    rotates both readings the same way); every other transform is
    channel-count agnostic and applied as-is via the vendored functions."""
    x_b = x_tc[None, :, :]  # fake batch dim, matching the vendored code's own convention
    if fn is rotation_transform and x_b.shape[-1] == 6:
        axes = np.random.uniform(low=-1, high=1, size=(1, 3))
        angles = np.random.uniform(low=-np.pi, high=np.pi, size=(1,))
        R = axis_angle_to_rotation_matrix_3d(axes, angles)  # [1,3,3]
        out = x_b.copy()
        out[:, :, :3] = np.matmul(x_b[:, :, :3], R)
        out[:, :, 3:6] = np.matmul(x_b[:, :, 3:6], R)
        return out[0]
    return fn(x_b)[0]


def two_random_augments(x_tc: np.ndarray) -> np.ndarray:
    """Apply 2 of the 8 published augmentations (randomly chosen, matching
    the vendored crossattn_augdataset.__getitem__)."""
    idxs = np.random.choice(np.arange(len(_TRANSFORMS)), 2, replace=False)
    out = x_tc
    for i in idxs:
        out = _apply_transform(out, _TRANSFORMS[i])
    return out


class MotifDistNet(nn.Module):
    """Thin wrapper around the vendored CrossAttn.Net, matching this repo's
    module conventions (nn.Module subclass, built with plain kwargs)."""

    def __init__(self, in_channels: int = 6, embed_dim: int = 64,
                 kernel_size: int = 15, double_receptivefield: int = 2):
        super().__init__()
        self.net = _CrossAttnNet(
            query_dimsize=in_channels, key_dimsize=in_channels,
            kernel_size=kernel_size, embed_dim=embed_dim,
            double_receptivefield=double_receptivefield,
        )

    def forward(self, query_in, key_in):
        return self.net(query_in, key_in)

    def calc_distance(self, anchor: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """anchor/candidate: [B, T, C] -> [B] reconstruction-error distance.
        Always run without grad (frozen use, Stage 2) -- Stage 1 training
        calls .forward()/motifdist_forward_loss directly instead."""
        with torch.no_grad():
            recon, _ = self.net(anchor, candidate)
            dist = torch.sum((recon - anchor) ** 2, dim=(1, 2))
        return dist


def motifdist_forward_loss(batch, net: MotifDistNet, cfg) -> Tuple[torch.Tensor, Dict]:
    """x [B,C,T] -> two augmented views (query=original, key=augmented) ->
    cross-attention reconstruction loss (query reconstructed from key)."""
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    x = x.float()
    B, C, T = x.shape
    x_np = x.detach().cpu().numpy().transpose(0, 2, 1)  # [B,T,C]
    x_aug_np = np.stack([two_random_augments(x_np[i]) for i in range(B)], axis=0)

    query_tc = x.transpose(1, 2)  # [B,T,C]
    key_tc = torch.from_numpy(x_aug_np.astype(np.float32)).to(x.device)
    recon, _ = net(query_tc, key_tc)
    loss = torch.sum((recon - query_tc) ** 2) / B
    return loss, {"loss": float(loss.detach())}


# ══════════════════════════════════════════════════════════════════════════
# Stage 2 -- RelCon relative contrastive loss on the shared ViT1DEncoder
# ══════════════════════════════════════════════════════════════════════════

def _pooled_embed(encoder, x):
    """x [B,C,T] -> [B,D] single global embedding (mean over channels AND
    time-patches) -- matches the original RelCon paper's global-avg-pool
    ResNet1D design. Downstream tasks still use the repo's own
    features()-style per-patch pooling on the same trained encoder weights
    (architecture-only transfer -- unaffected by how Stage 2 pools for its
    own loss, exactly like FOCAL/SimMTM's checkpoints)."""
    tokens, Pt = encoder.forward_tokens(x)
    B, C, D = x.shape[0], encoder.in_channels, encoder.embed_dim
    return tokens.reshape(B, C, Pt, D).mean(dim=(1, 2))


def relative_contrastive_loss(emb_ancs, emb_cands, sortedinds, tau=0.1):
    """Ported verbatim (math unchanged) from
    baselines/SSL/relcon/relcon/models/RelCon/RelCon_Model.py::relative_contrastive_loss.
    emb_ancs: [BS, D]. emb_cands: [BS, BS-1, D] (every OTHER anchor in the
    batch, in original index order skipping self -- see relcon_forward_loss).
    sortedinds: [BS-1, BS] ascending motif-distance order (row 0 = most
    similar candidate for that batch column)."""
    bs, candset_size, channels = emb_cands.shape
    loss = torch.zeros(bs, device=emb_ancs.device)
    for batch_idx in range(bs):
        emb_anc_idx = emb_ancs[batch_idx, :].contiguous().view(1, -1)
        emb_cands_idx = emb_cands[batch_idx, :]  # [(BS-1), D]
        emb_cands_idx_revsorted = emb_cands_idx[torch.flip(sortedinds[:, batch_idx], dims=(0,))]
        sim_revsorted = F.cosine_similarity(emb_anc_idx, emb_cands_idx_revsorted, dim=1) / tau
        exp_sim_revsorted = torch.exp(sim_revsorted)
        cumsum_exp_sim_revsorted = torch.cumsum(exp_sim_revsorted, dim=0)
        logsoftmax = sim_revsorted[1:] - torch.log(cumsum_exp_sim_revsorted[1:])
        loss[batch_idx] = torch.sum(-logsoftmax)
    return torch.mean(loss)


def relcon_forward_loss(batch, encoder, motifdist_net: MotifDistNet, cfg) -> Tuple[torch.Tensor, Dict]:
    """x [B,C,T] -> between-window candidate ranking via frozen MotifDist,
    encoder embeddings via the shared ViT1DEncoder, relative contrastive loss."""
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    x = x.float()
    B = x.shape[0]
    obj = cfg_get(cfg, ["objective"], {}) or {}
    tau = float(obj.get("tau", 0.1))
    x_tc = x.transpose(1, 2)  # [B,T,C]

    distances = []
    for idx in range(1, B):
        rotated = torch.cat((x_tc[idx:], x_tc[:idx]), dim=0)
        distances.append(motifdist_net.calc_distance(anchor=x_tc, candidate=rotated))
    distances = torch.stack(distances)              # [(B-1), B]
    _, sortedinds = torch.sort(distances, dim=0)     # ascending: index 0 = most similar

    emb = _pooled_embed(encoder, x)                  # [B, D]
    emb_cands = torch.stack(
        [torch.cat((emb[b + 1:], emb[:b]), dim=0) for b in range(B)], dim=0
    )                                                 # [B, B-1, D]

    loss = relative_contrastive_loss(emb, emb_cands, sortedinds, tau=tau)
    return loss, {"loss": float(loss.detach())}
