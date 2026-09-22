"""DINO (Caron et al. 2021) on the SHARED ViT1DEncoder and the multi-source cached corpus.

Self-distillation with no labels: a student encoder and an EMA teacher encoder score the same window under different
views; the teacher's centred and sharpened prototype distribution on the two GLOBAL views is the target for the
student's distribution on every view (global and local), with the cross-entropy summed over all pairs of distinct
views ("multi-crop"). Collapse is prevented as in the paper: centring (EMA mean of the teacher output subtracted),
sharpening (teacher temperature below the student's), a weight-normalised prototype layer whose norm is fixed, the
prototypes frozen for the first steps, and a cosine EMA momentum 0.996 -> 1. No iBOT patch term and no KoLeo (those
belong to DINOv2), so what is measured is the DINO objective itself.

Time-series views. A "crop" is a contiguous span of the window: global crops cover 50-100 % of the window, local crops
15-40 % (2 patches at least); one span length per view per batch (so the batch stacks), an independent offset per
sample. Every view also receives its own random octahedral rotation of the two sensors, a per-channel gain
N(1, 0.1) and Gaussian noise (0.05 in z-score units), which are the augmentations the FOCAL baseline uses for its
two views. The representation entering the projection head is the mean over the encoder's tokens, the same pooled
vector every downstream classification head of the benchmark reads, so the encoder's state dict is identical in
layout to the MAE rows' (no CLS token is added).

Everything that is not the student encoder lives in `DINOState` (teacher encoder, both heads, centre, step counter),
ONE module, so that the trainer's single extra-module slot checkpoints and restores it and `--resume latest` continues
the EMA and temperature schedules. Validation views come from a generator seeded by the batch content, so the
validation loss that selects the checkpoint compares like with like across evaluations.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.models.ViT1D.model import ViT1DDINOHead
from imu_lm.objectives.ssl_wearables_mtl import batch_seed
from imu_lm.utils.helpers import cfg_get


class DINOState(nn.Module):
    def __init__(self, encoder: nn.Module, cfg: Any):
        super().__init__()
        top = cfg_get(cfg, ["objective"], {}) or {}
        obj = dict(top); obj.update(top.get("dino", {}) or {})          # hyper-parameters live in objective.dino.*
        head = obj.get("head", obj.get("dino_head", {})) or {}
        K = int(head.get("out_dim", 4096))
        kw = dict(in_dim=encoder.embed_dim, out_dim=K, hidden_dim=int(head.get("hidden_dim", 2048)),
                  bottleneck_dim=int(head.get("bottleneck_dim", 256)))
        self.teacher = copy.deepcopy(encoder)
        self.teacher.requires_grad_(False)
        self.student_head = ViT1DDINOHead(**kw)
        self.teacher_head = ViT1DDINOHead(**kw)                 # weight_norm modules do not deepcopy: build + copy
        with torch.no_grad():
            for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
                pt.copy_(ps)
        self.teacher_head.requires_grad_(False)
        self.register_buffer("center", torch.zeros(1, K))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        # schedule / loss hyper-parameters (DINO defaults for ViT-S unless overridden)
        self.student_temp = float(obj.get("student_temp", 0.1))
        self.teacher_temp_start = float(obj.get("warmup_teacher_temp", 0.04))
        self.teacher_temp_end = float(obj.get("teacher_temp", 0.07))
        self.teacher_temp_steps = int(obj.get("warmup_teacher_temp_steps", 16000))
        self.center_momentum = float(obj.get("center_momentum", 0.9))
        self.ema_start = float(obj.get("ema_momentum", 0.996))
        self.max_steps = int(cfg_get(cfg, ["trainer", "max_steps"], 160000))
        self.freeze_last_layer_steps = int(obj.get("freeze_last_layer_steps", 1500))
        self.n_local = int(obj.get("n_local_crops", obj.get("local_crops", 4)))
        self.global_scale = tuple(obj.get("global_crop_scale", obj.get("global_scale", (0.5, 1.0))))
        self.local_scale = tuple(obj.get("local_crop_scale", obj.get("local_scale", (0.15, 0.4))))
        self.gain_std = float(obj.get("gain_std", 0.1))
        self.noise_std = float(obj.get("noise_std", 0.05))
        self.rotate_views = bool(obj.get("octahedral_rotation", obj.get("rotate_views", True)))
        # DINO's freeze_last_layer: zero the prototype-direction gradient for the first steps
        self.student_head.last_layer.weight_v.register_hook(
            lambda g: torch.zeros_like(g) if int(self.step) < self.freeze_last_layer_steps else g)

    def trainable_parameters(self):
        return list(self.student_head.parameters())

    # ---- schedules -------------------------------------------------------------------------------------------------
    def teacher_temp(self) -> float:
        s = int(self.step)
        if self.teacher_temp_steps > 0 and s < self.teacher_temp_steps:
            a = s / self.teacher_temp_steps
            return self.teacher_temp_start + a * (self.teacher_temp_end - self.teacher_temp_start)
        return self.teacher_temp_end

    def ema_momentum(self) -> float:
        t = min(int(self.step), self.max_steps) / max(1, self.max_steps)
        return 1.0 - (1.0 - self.ema_start) * (1.0 + math.cos(math.pi * t)) / 2.0

    @torch.no_grad()
    def ema_update(self, encoder: nn.Module) -> float:
        m = self.ema_momentum()
        for ps, pt in zip(encoder.parameters(), self.teacher.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        return m


# --------------------------------------------------------------------------------------------------------------------
# views
# --------------------------------------------------------------------------------------------------------------------
def octahedral_rotations(n: int, gen: torch.Generator) -> torch.Tensor:
    """n random proper signed permutation matrices (the 24 rotations of the cube), [n, 3, 3] on the CPU."""
    perm = torch.argsort(torch.rand(n, 3, generator=gen), dim=1)
    signs = torch.randint(0, 2, (n, 3), generator=gen) * 2 - 1
    inv = (perm[:, 0] > perm[:, 1]).int() + (perm[:, 0] > perm[:, 2]).int() + (perm[:, 1] > perm[:, 2]).int()
    det = signs.prod(dim=1) * (1 - 2 * (inv % 2))
    signs[:, 0] *= torch.where(det < 0, -1, 1)
    R = torch.zeros(n, 3, 3)
    R[torch.arange(n)[:, None], torch.arange(3)[None, :], perm] = signs.float()
    return R


def _crop_len(L: int, scale, patch: int, gen: torch.Generator, min_patches: int = 2) -> int:
    lo, hi = scale
    frac = lo + (hi - lo) * torch.rand((), generator=gen).item()
    p = max(min_patches, min(L // patch, int(round(frac * L / patch))))
    return p * patch


def make_view(x: torch.Tensor, Lc: int, st: DINOState, gen: torch.Generator) -> torch.Tensor:
    """Random span of length Lc per sample + rotation + gain + noise. Random numbers come from `gen` (CPU)."""
    B, C, L = x.shape
    off = torch.randint(0, L - Lc + 1, (B,), generator=gen)
    idx = (off[:, None] + torch.arange(Lc)[None, :]).to(x.device)
    v = torch.gather(x, -1, idx[:, None, :].expand(B, C, Lc))
    if st.rotate_views and C >= 3:
        R = octahedral_rotations(B, gen).to(x.device, x.dtype)
        parts = [torch.bmm(R, v[:, s:s + 3]) for s in range(0, (C // 3) * 3, 3)]
        if C % 3:
            parts.append(v[:, (C // 3) * 3:])
        v = torch.cat(parts, dim=1)
    if st.gain_std > 0:
        v = v * (1.0 + st.gain_std * torch.randn(B, C, 1, generator=gen)).to(x.device, x.dtype)
    if st.noise_std > 0:
        v = v + st.noise_std * torch.randn(v.shape, generator=gen).to(x.device, x.dtype)
    return v.contiguous()


def _pooled_logits(encoder: nn.Module, head: nn.Module, v: torch.Tensor) -> torch.Tensor:
    tokens, _ = encoder.forward_tokens(v)
    return head(tokens.mean(dim=1)).float()


# --------------------------------------------------------------------------------------------------------------------
def forward_loss(batch, encoder: nn.Module, st: DINOState, cfg: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    x = x.float()
    B, C, L = x.shape
    patch = int(encoder.patch_size)
    gen = torch.Generator()
    gen.manual_seed(batch_seed(x) if not encoder.training else int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))

    lens: List[int] = [_crop_len(L, st.global_scale, patch, gen), _crop_len(L, st.global_scale, patch, gen)]
    if st.n_local > 0:
        ll = _crop_len(L, st.local_scale, patch, gen)
        lens += [min(ll, min(lens[:2]))] * st.n_local
    views = [make_view(x, Lc, st, gen) for Lc in lens]

    st.teacher.eval(); st.teacher_head.eval()
    with torch.no_grad():
        t_out = torch.stack([_pooled_logits(st.teacher, st.teacher_head, v) for v in views[:2]])      # [2, B, K]
        # validation is scored at the FINAL teacher temperature: the warm-up (0.04 -> 0.07 over 16k steps) would
        # otherwise raise the cross-entropy from one evaluation to the next and trip early stopping / best.pt
        tau_t = st.teacher_temp() if encoder.training else st.teacher_temp_end
        t_prob = F.softmax((t_out - st.center) / tau_t, dim=-1)
    s_logp = [F.log_softmax(_pooled_logits(encoder, st.student_head, v) / st.student_temp, dim=-1) for v in views]

    total, n_terms = x.new_zeros(()), 0
    for i in range(2):
        for j in range(len(views)):
            if i == j:
                continue
            total = total + (-(t_prob[i] * s_logp[j]).sum(dim=-1).mean())
            n_terms += 1
    loss = total / n_terms

    logs = {"loss": float(loss.detach()), "teacher_temp": tau_t, "n_views": float(len(views)),
            "global_len": float(lens[0]), "local_len": float(lens[-1])}
    with torch.no_grad():
        avg = t_prob.mean(dim=(0, 1))
        ent = float(-(avg * (avg + 1e-8).log()).sum())
        logs["proto_entropy"] = ent
        logs["proto_eff_k"] = math.exp(ent)
    if encoder.training:
        with torch.no_grad():
            st.center.mul_(st.center_momentum).add_(t_out.reshape(-1, t_out.shape[-1]).mean(dim=0, keepdim=True),
                                                    alpha=1.0 - st.center_momentum)
        logs["ema_momentum"] = st.ema_update(encoder)
        st.step += 1
    return loss, logs
