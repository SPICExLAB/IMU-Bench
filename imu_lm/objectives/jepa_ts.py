"""I-JEPA (Assran et al. 2023) on the SHARED ViT1DEncoder and the multi-source cached corpus.

A context encoder (the trainable encoder) sees only the context patches of a window; an EMA target encoder sees the
whole window; a narrow transformer predictor, given the context tokens and a mask token with the positional embedding
of every target patch, predicts the target encoder's (layer-normalised) representation at the target patches. The
loss is the smooth-L1 distance between prediction and target, as in the reference implementation
(github.com/facebookresearch/ijepa, `F.smooth_l1_loss`); nothing is reconstructed in signal space. The EMA momentum
rises linearly from 0.996 to 1.0 over training.

Masking follows the paper's multi-block scheme on the time axis: `n_targets` target blocks, each a contiguous span of
15-20 % of the window, and one context block of 85-100 % of the window from which the target patches are removed.
The blocks cover all sensor channels (a target is a span of time, not a channel). One layout is drawn per batch and
circularly shifted by an independent offset per sample, so every sample in the batch removes the same NUMBER of
tokens, which the encoder's batched token dropping requires; target coverage is capped at `max_target_frac` of the
window so that at least one context patch always remains, also on the shortest (1.28 s) windows.

`JEPAState` (target encoder, predictor, step counter) is one module so the trainer's extra-module slot checkpoints it
and `--resume latest` continues the momentum schedule. The predictor is the repository's `JEPA1DPredictor` (width and
depth from `vit1d.jepa_predictor`, by default the same 192 x 4 as the MAE decoder). Validation masks come from a
generator seeded by the batch content.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.models.ViT1D.model import JEPA1DPredictor
from imu_lm.objectives.ssl_wearables_mtl import batch_seed
from imu_lm.utils.helpers import cfg_get


class JEPAState(nn.Module):
    def __init__(self, encoder: nn.Module, cfg: Any):
        super().__init__()
        top = cfg_get(cfg, ["objective"], {}) or {}
        obj = dict(top); obj.update(top.get("jepa", {}) or {})          # hyper-parameters live in objective.jepa.*
        self.target = copy.deepcopy(encoder)
        self.target.requires_grad_(False)
        self.predictor = JEPA1DPredictor(embed_dim=encoder.embed_dim, cfg=cfg)
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.ema_start = float(obj.get("ema_momentum", 0.996))
        self.ema_end = float(obj.get("ema_momentum_end", 1.0))
        self.max_steps = int(cfg_get(cfg, ["trainer", "max_steps"], 160000))
        self.n_targets = int(obj.get("n_targets", 4))
        self.target_scale = tuple(obj.get("target_scale", (0.15, 0.2)))
        self.context_scale = tuple(obj.get("context_scale", (0.85, 1.0)))
        self.max_target_frac = float(obj.get("max_target_frac", 0.75))

    def trainable_parameters(self):
        return list(self.predictor.parameters())

    def ema_momentum(self) -> float:
        t = min(int(self.step), self.max_steps) / max(1, self.max_steps)
        return self.ema_start + (self.ema_end - self.ema_start) * t

    @torch.no_grad()
    def ema_update(self, encoder: nn.Module) -> float:
        m = self.ema_momentum()
        for ps, pt in zip(encoder.parameters(), self.target.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        return m


def _span(Pt: int, scale, gen: torch.Generator) -> Tuple[int, int]:
    lo, hi = scale
    w = max(1, min(Pt, int(round((lo + (hi - lo) * torch.rand((), generator=gen).item()) * Pt))))
    s = int(torch.randint(0, Pt - w + 1, (1,), generator=gen).item())
    return s, w


def block_masks(B: int, Pt: int, C: int, st: JEPAState, gen: torch.Generator, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """(removed [B, C*Pt], target [B, C*Pt]); removed = everything the context encoder does not see (targets and the
    part outside the context block), target = the patches the loss is taken on. Channel-major token order."""
    tgt = torch.zeros(Pt, dtype=torch.bool)
    for _ in range(st.n_targets):
        s, w = _span(Pt, st.target_scale, gen)
        cand = tgt.clone(); cand[s:s + w] = True
        if int(cand.sum()) <= max(1, int(st.max_target_frac * Pt)):
            tgt = cand
    if not tgt.any():
        tgt[int(torch.randint(0, Pt, (1,), generator=gen).item())] = True
    s, w = _span(Pt, st.context_scale, gen)
    ctx = torch.zeros(Pt, dtype=torch.bool); ctx[s:s + w] = True
    ctx &= ~tgt
    if not ctx.any():                                     # the context block fell entirely on targets
        ctx = ~tgt
    removed_t, target_t = ~ctx, tgt
    shifts = torch.randint(0, Pt, (B,), generator=gen)
    ar = torch.arange(Pt)
    idx = (ar[None, :] - shifts[:, None]) % Pt           # per-sample circular shift of the same layout
    removed = removed_t[idx]; target = target_t[idx]     # [B, Pt]
    removed = removed[:, None, :].expand(B, C, Pt).reshape(B, C * Pt)
    target = target[:, None, :].expand(B, C, Pt).reshape(B, C * Pt)
    return removed.to(device), target.to(device)


def forward_loss(batch, encoder: nn.Module, st: JEPAState, cfg: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    x = x.float()
    B, C, L = x.shape
    Pt = L // int(encoder.patch_size)
    gen = torch.Generator()
    gen.manual_seed(batch_seed(x) if not encoder.training else int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
    removed, target = block_masks(B, Pt, C, st, gen, x.device)

    ctx_tokens, ctx_pad = encoder(x, mask=removed, return_keep_padding_mask=True)          # [B, N_ctx, D]
    st.target.eval()
    with torch.no_grad():
        full, _ = st.target.forward_tokens(x)                                                # [B, N, D]
        tgt = F.layer_norm(full.float(), [full.shape[-1]])
    pred = st.predictor(ctx_tokens, removed, Pt, C, ctx_padding_mask=ctx_pad).float()       # [B, N, D]
    loss = F.smooth_l1_loss(pred[target], tgt[target])

    logs = {"loss": float(loss.detach()), "target_frac": float(target.float().mean()),
            "context_frac": float((~removed).float().mean()),
            "cosine_sim": float(F.cosine_similarity(pred[target].detach(), tgt[target], dim=-1).mean())}
    if encoder.training:
        logs["ema_momentum"] = st.ema_update(encoder)
        st.step += 1
    return loss, logs
