"""SSL baseline objectives on the SHARED ViT1DEncoder.

For a controlled "which SSL objective is best?" comparison, all baselines run on
the identical ViT1DEncoder (same capacity, same data) — only the objective head
and loss differ. The authentic loss math is reused from the vendored repos.

    limu_bert : masked-patch reconstruction (temporal/span masking) + MSE.
                On a patch encoder this is masked reconstruction (≈ MAE) — the
                LIMU-BERT objective, minus its parameter-shared tiny backbone.
    simmtm    : geometric masking → series-wise contrastive aggregation +
                point reconstruction (ContrastiveWeight + AggregationRebuild +
                AutomaticWeightedLoss).
    focal     : two augmented views → factorised shared/private contrastive +
                orthogonality + temporal-locality ranking. ViT1D's (channel,
                time-patch) token grid maps directly to FOCAL's (modality,
                sub-segment) structure.

Each forward_loss matches the trainer contract: (batch, encoder, head, cfg).
``head`` is the objective-specific module (decoder / projas), built in run.py.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.utils.helpers import cfg_get
from imu_lm.objectives.masked_1d import generate_patch_mask, _patch_mask_to_signal_mask

# Authentic vendored loss pieces
from imu_lm.models.baselines.simmtm_ssl import (
    ContrastiveWeight, AggregationRebuild, AutomaticWeightedLoss,
    data_transform_masked4cl,
)
from imu_lm.models.baselines.focal_ssl import FOCALLoss


# ══════════════════════════════════════════════════════════════════════════════
# LIMU-BERT — masked-patch reconstruction on ViT1D
# ══════════════════════════════════════════════════════════════════════════════

def limubert_forward_loss(batch, encoder, decoder, cfg) -> Tuple[torch.Tensor, Dict]:
    """Span/temporal-masked reconstruction. ``decoder`` is a ViT1DDecoder."""
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    x = x.float()
    B, C, T = x.shape
    obj = cfg_get(cfg, ["objective"], {}) or {}
    mask_ratio = float(obj.get("mask_ratio", 0.15))
    strategy   = str(obj.get("mask_strategy", "temporal"))   # span ≈ temporal patches
    ps = encoder.patch_size
    Pt = T // ps

    patch_mask = generate_patch_mask(B, Pt, C, mask_ratio, strategy, x.device)
    vis, vis_pad = encoder(x, mask=patch_mask, return_keep_padding_mask=True)
    recon = decoder(vis, patch_mask, Pt, C, visible_padding_mask=vis_pad)

    sm = _patch_mask_to_signal_mask(patch_mask, Pt, C, ps)
    Tr = recon.shape[-1]
    if Tr < sm.shape[-1]:
        sm, x = sm[:, :, :Tr], x[:, :, :Tr]
    loss = F.mse_loss(recon[sm], x[sm])
    return loss, {"loss": float(loss.detach()),
                  "mask_ratio_act": float(patch_mask.float().mean())}


# ══════════════════════════════════════════════════════════════════════════════
# SimMTM — contrastive aggregation + reconstruction on ViT1D
# ══════════════════════════════════════════════════════════════════════════════

class SimMTMHead(nn.Module):
    """Projection (for contrastive) + reconstruction head over ViT1D tokens."""

    def __init__(self, embed_dim, num_tokens, recon_dim, proj_dim=128,
                 temperature=0.2, positive_nums=3):
        super().__init__()
        flat = num_tokens * embed_dim
        self.proj = nn.Sequential(
            nn.Linear(flat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, proj_dim))
        self.recon = nn.Linear(flat, recon_dim)
        self.contrastive = ContrastiveWeight(temperature, positive_nums)
        self.aggregation = AggregationRebuild(temperature)
        self.awl = AutomaticWeightedLoss(2)


def simmtm_forward_loss(batch, encoder, head: SimMTMHead, cfg) -> Tuple[torch.Tensor, Dict]:
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    x = x.float()
    obj = cfg_get(cfg, ["objective"], {}) or {}
    masking_ratio = float(obj.get("masking_ratio", 0.5))
    lm            = int(obj.get("lm", 3))
    positive_nums = int(obj.get("positive_nums", 3))

    masked_m, _ = data_transform_masked4cl(x, masking_ratio, lm, positive_nums)
    x_om = torch.cat([x, masked_m.to(x.device)], dim=0)        # [(1+P)*B, C, T]

    tokens, _ = encoder.forward_tokens(x_om)                   # [N', M, D]
    flat = tokens.reshape(tokens.size(0), -1)
    z = head.proj(flat)

    loss_cl, sim, _, _ = head.contrastive(z)
    _, agg = head.aggregation(sim, tokens)                     # [N', M, D]
    pred = head.recon(agg.reshape(agg.size(0), -1))            # [N', C*T]
    loss_rb = F.mse_loss(pred, x_om.reshape(x_om.size(0), -1).detach())
    loss = head.awl(loss_cl, loss_rb)
    return loss, {"loss": float(loss.detach()),
                  "loss_cl": float(loss_cl.detach()),
                  "loss_rb": float(loss_rb.detach())}


class SimMTMHeadVarLen(nn.Module):
    """Length-agnostic SimMTM head: pooled projection + per-patch reconstruction.

    Unlike SimMTMHead (fixed flat Linear over N*D), this works for any token
    count: the contrastive projection pools tokens (mean → D), and the
    reconstruction is per-token (D → patch_size), so it adapts to variable
    window lengths.
    """

    def __init__(self, embed_dim, patch_size, proj_dim=128,
                 temperature=0.2, positive_nums=3):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, proj_dim))
        self.recon = nn.Linear(embed_dim, patch_size)          # per-token → patch_size
        self.contrastive = ContrastiveWeight(temperature, positive_nums)
        self.aggregation = AggregationRebuild(temperature)
        self.awl = AutomaticWeightedLoss(2)


def simmtm_forward_loss_varlen(batch, encoder, head: SimMTMHeadVarLen, cfg) -> Tuple[torch.Tensor, Dict]:
    """SimMTM on the shared ViT1D backbone.

    Two objective flags restore what SimMTM's own ``Model.pretrain`` does and this
    port originally did not (both default to False, i.e. the behaviour every
    ``*_5src``/``*_4src`` SimMTM row in the leaderboard was trained with):

    ``visible_point_instance_norm``
        SimMTM normalises inside the model, not in the dataloader: per sample and
        per channel, mean/std over the VISIBLE points only, detached, and the
        reconstruction is de-normalised back to the input's own units before the
        MSE (SimMTM_Forecasting/models/SimMTM.py:172-178 and 205-207). With the
        benchmark's per-window z-score in the dataloader this is a near no-op --
        the stats are already ~(0, 1) and the recon loss is scale-free. Fed
        un-z-scored input instead, it is the real objective: window amplitude
        survives into the encoder input statistics and weights the recon loss.

    ``recon_on_original_only``
        ``pred_batch_x = dec_out[:batch_x.shape[0]]`` — only the un-masked copy is
        reconstructed, against the original series (SimMTM.py:208-211). This port
        instead took the MSE over all (1+positive_nums) copies against ``x_om``,
        whose masked copies are zero at the masked points. Harmless under z-score,
        but in physical units "predict 0 at the masked points" is an amplitude-
        weighted error term SimMTM never had, so the faithful arm turns it off.
    """
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    x = x.float()                                              # [B, C, L]
    obj = cfg_get(cfg, ["objective"], {}) or {}
    masking_ratio = float(obj.get("masking_ratio", 0.5))
    lm            = int(obj.get("lm", 3))
    positive_nums = int(obj.get("positive_nums", 3))
    inst_norm     = bool(obj.get("visible_point_instance_norm", False))
    recon_orig    = bool(obj.get("recon_on_original_only", False))

    B = x.shape[0]
    masked_m, keep_m = data_transform_masked4cl(x, masking_ratio, lm, positive_nums)
    x_om = torch.cat([x, masked_m.to(x.device)], dim=0)        # [(1+P)*B, C, L]

    x_in = x_om
    means = stdev = None
    if inst_norm:
        # keep_m is True where the point SURVIVED the geometric mask (SimMTM's mask == 1);
        # the first B rows are the un-masked copy, so their mask is all ones.
        keep = torch.cat([torch.ones_like(x),
                          keep_m.to(x.device).float()], dim=0)          # [N', C, L]
        denom = keep.sum(dim=-1, keepdim=True).clamp(min=1.0)
        means = ((x_om * keep).sum(dim=-1, keepdim=True) / denom).detach()
        xc = (x_om - means).masked_fill(keep == 0, 0.0)
        stdev = torch.sqrt((xc * xc).sum(dim=-1, keepdim=True) / denom + 1e-5).detach()
        x_in = xc / stdev

    tokens, Pt = encoder.forward_tokens(x_in)                  # [N', C*Pt, D], Pt=L//ps
    C, ps = encoder.in_channels, head.patch_size

    z = head.proj(tokens.mean(dim=1))                          # pool tokens → [N', D] → proj
    loss_cl, sim, _, _ = head.contrastive(z)
    _, agg = head.aggregation(sim, tokens)                     # [N', C*Pt, D]
    pred_patches = head.recon(agg)                             # [N', C*Pt, ps]
    pred = pred_patches.reshape(pred_patches.size(0), C, Pt * ps)   # [N', C, L]
    if inst_norm:
        pred = pred * stdev + means                            # de-Normalization
    if recon_orig:
        loss_rb = F.mse_loss(pred[:B], x_om[:B].detach())
    else:
        loss_rb = F.mse_loss(pred, x_om.detach())
    loss = head.awl(loss_cl, loss_rb)
    return loss, {"loss": float(loss.detach()),
                  "loss_cl": float(loss_cl.detach()),
                  "loss_rb": float(loss_rb.detach())}


# ══════════════════════════════════════════════════════════════════════════════
# FOCAL — factorised contrastive on ViT1D's (channel, time-patch) token grid
# ══════════════════════════════════════════════════════════════════════════════

_MOD_CH = {"acc": (0, 3), "gyro": (3, 6)}


class FOCALHead(nn.Module):
    """Per-modality projection heads over ViT1D tokens."""

    def __init__(self, embed_dim, modalities, emb_dim=128):
        super().__init__()
        assert emb_dim % 2 == 0, "emb_dim must be even (shared/private split)"
        self.modalities = list(modalities)
        self.projectors = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                             nn.Linear(embed_dim, emb_dim))
            for m in self.modalities
        })


def _augment_view(x):
    noise = torch.randn_like(x) * 0.05
    scale = 1.0 + torch.randn(x.shape[0], x.shape[1], 1, device=x.device) * 0.1
    out = x * scale + noise
    shift = int(torch.randint(-5, 6, (1,)).item())
    if shift != 0:
        out = torch.roll(out, shifts=shift, dims=-1)
    return out


def _focal_embed(x, encoder, head: FOCALHead):
    """x [B, C, T] → {modality: [B*Pt, emb_dim]} using ViT1D token grid."""
    B, C, T = x.shape
    Pt = T // encoder.patch_size
    tokens, _ = encoder.forward_tokens(x)                      # [B, C*Pt, D] channel-major
    grid = tokens.reshape(B, C, Pt, tokens.shape[-1])          # [B, C, Pt, D]
    out = {}
    for m in head.modalities:
        lo, hi = _MOD_CH[m]
        feat = grid[:, lo:hi].mean(dim=1)                      # pool channels → [B, Pt, D]
        proj = head.projectors[m](feat)                       # [B, Pt, emb_dim]
        out[m] = proj.reshape(B * Pt, -1)                     # [B*Pt, emb_dim]
    return out


_focal_loss_cache = {}


def focal_forward_loss(batch, encoder, head: FOCALHead, cfg) -> Tuple[torch.Tensor, Dict]:
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    x = x.float()
    device = x.device
    Pt = x.shape[-1] // encoder.patch_size
    obj = cfg_get(cfg, ["objective"], {}) or {}

    key = (str(device), Pt)
    loss_fn = _focal_loss_cache.get(key)
    if loss_fn is None:
        loss_fn = FOCALLoss(
            device=device, seq_len=Pt, modalities=head.modalities,
            temperature=float(obj.get("temperature", 0.07)),
            inter_rank_margin=float(obj.get("inter_rank_margin", 1.0)),
            shared_w=float(obj.get("shared_contrastive_loss_weight", 1.0)),
            private_w=float(obj.get("private_contrastive_loss_weight", 1.0)),
            ortho_w=float(obj.get("orthogonal_loss_weight", 3.0)),
            rank_w=float(obj.get("rank_loss_weight", 5.0)),
        )
        _focal_loss_cache[key] = loss_fn

    f1 = _focal_embed(_augment_view(x), encoder, head)
    f2 = _focal_embed(_augment_view(x), encoder, head)
    return loss_fn(f1, f2)
