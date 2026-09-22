"""SSL-Wearables multi-task self-supervision (Yuan et al. 2024) on the SHARED ViT1DEncoder.

"Self-supervised learning for human activity recognition using 700,000 person-days of wearable data",
npj Digital Medicine 7:91. Code: github.com/OxWearables/ssl-wearables (mtl.py, sslearning/data/data_transformation.py).

The METHOD, not the released model (their ResNet-V2 trained on UK Biobank is the HarNet row; this file trains our own
backbone on our own corpus, like every other controlled row of IMU-Bench). Their pretext task is a set of binary
classification problems on ONE transformed view of each window: has it been (1) reversed in time ("arrow of time"),
(2) cut into segments that were shuffled ("permutation"), (3) resampled along a random smooth time curve ("time warp")?
Each transformation is applied independently with probability `positive_ratio` (0.5, conf/task/all.yaml), in their
order flip -> permute -> time_warp, and the loss is the mean of the per-task cross-entropies (mtl.py: total_loss /
total_task). Their fourth task, per-axis scaling, is off: the paper's final model uses the three above, and on the
per-window z-scored input of this benchmark a per-channel scale is undetectable anyway.

Transformations follow data_transformation.py exactly (DA_Permutation nPerm=4 / minSegLength=10; DA_TimeWarp with
knot=4 cubic-spline speed curves, sigma=0.2, one independent curve PER CHANNEL, resampled with linear interpolation),
generalised from 3 accelerometer axes to the 6 channels here and vectorised over the batch on the GPU. The
classification heads are one linear layer per task on the mean-pooled tokens, as their per-task fully-connected
heads sit on the pooled ResNet features.

Validation batches use a generator seeded from the batch content, so the validation loss that selects the checkpoint
sees the same transformations at every evaluation.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.utils.helpers import cfg_get

TASKS = ("aot", "permute", "timewarp")


class SSLWearablesHead(nn.Module):
    """One 2-way linear classifier per pretext task on the pooled encoder output."""

    def __init__(self, embed_dim: int, tasks=TASKS):
        super().__init__()
        self.tasks = tuple(tasks)
        self.heads = nn.ModuleDict({t: nn.Linear(embed_dim, 2) for t in self.tasks})

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {t: self.heads[t](feat) for t in self.tasks}


# --------------------------------------------------------------------------------------------------------------------
# transformations (batched; each acts only on the rows selected by `sel`)
# --------------------------------------------------------------------------------------------------------------------
def batch_seed(x: torch.Tensor) -> int:
    """Deterministic seed from the batch content (validation only)."""
    v = x[:, 0, : min(16, x.shape[-1])].float().sum().item()
    return int(abs(v) * 1000.0) % (2 ** 31 - 1)


def flip_time(x: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    return torch.where(sel[:, None, None], x.flip(-1), x)


def permute_segments(x: torch.Tensor, sel: torch.Tensor, rng: np.random.Generator,
                     n_perm: int = 4, min_seg: int = 10) -> torch.Tensor:
    """DA_Permutation: split the time axis into n_perm segments (each > min_seg samples) and shuffle them."""
    B, C, L = x.shape
    rows = torch.nonzero(sel, as_tuple=False).flatten().tolist()
    if not rows:
        return x
    order = np.empty((len(rows), L), dtype=np.int64)
    for r, _ in enumerate(rows):
        idx = rng.permutation(n_perm)
        while True:
            segs = np.zeros(n_perm + 1, dtype=int)
            segs[1:-1] = np.sort(rng.integers(min_seg, L - min_seg, n_perm - 1))
            segs[-1] = L
            if np.min(segs[1:] - segs[:-1]) > min_seg:
                break
        order[r] = np.concatenate([np.arange(segs[i], segs[i + 1]) for i in idx])
    gather = torch.from_numpy(order).to(x.device)[:, None, :].expand(-1, C, -1)
    out = x.clone()
    out[rows] = torch.gather(x[rows], -1, gather)
    return out


def time_warp(x: torch.Tensor, sel: torch.Tensor, rng: np.random.Generator,
              sigma: float = 0.2, knot: int = 4) -> torch.Tensor:
    """DA_TimeWarp: per-channel cubic-spline speed curve -> cumulative time -> resample (np.interp semantics)."""
    from scipy.interpolate import CubicSpline
    B, C, L = x.shape
    rows = torch.nonzero(sel, as_tuple=False).flatten()
    n = int(rows.numel())
    if n == 0:
        return x
    xx = np.arange(0, L, (L - 1) / (knot + 1))[: knot + 2]                     # GenerateRandomCurves knots
    yy = rng.normal(loc=1.0, scale=sigma, size=(knot + 2, n * C))
    curves = CubicSpline(xx, yy)(np.arange(L))                                 # [L, n*C] speed around 1
    curves = np.clip(curves, 1e-3, None)                                        # keep the time axis monotone
    tt = np.cumsum(curves, axis=0)                                              # DistortTimesteps
    tt = tt * ((L - 1) / tt[-1])[None, :]                                       # last sample lands on L-1
    tt = torch.from_numpy(tt.T.reshape(n, C, L)).to(x.device, x.dtype)         # [n, C, L] sample times
    q = torch.arange(L, device=x.device, dtype=x.dtype).view(1, 1, L).expand(n, C, L)
    hi = torch.searchsorted(tt.contiguous(), q.contiguous()).clamp_(1, L - 1)  # np.interp: X[i] + w (X[i+1]-X[i])
    lo = hi - 1
    src = x[rows]
    t_lo, t_hi = torch.gather(tt, -1, lo), torch.gather(tt, -1, hi)
    w = ((q - t_lo) / (t_hi - t_lo).clamp_min(1e-6)).clamp_(0.0, 1.0)
    warped = torch.gather(src, -1, lo) + w * (torch.gather(src, -1, hi) - torch.gather(src, -1, lo))
    out = x.clone()
    out[rows] = warped
    return out


def make_view(x: torch.Tensor, positive_ratio: float, rng: np.random.Generator, tasks=TASKS):
    """One transformed view + {task: 0/1 label}, transformations in the reference order flip -> permute -> warp."""
    B = x.shape[0]
    labels = {t: torch.from_numpy((rng.random(B) < positive_ratio).astype(np.int64)).to(x.device) for t in tasks}
    v = x
    if "aot" in tasks:
        v = flip_time(v, labels["aot"].bool())
    if "permute" in tasks:
        v = permute_segments(v, labels["permute"].bool(), rng)
    if "timewarp" in tasks:
        v = time_warp(v, labels["timewarp"].bool(), rng)
    return v, labels


# --------------------------------------------------------------------------------------------------------------------
def forward_loss(batch, encoder: nn.Module, head: SSLWearablesHead, cfg: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    x = x.float()
    obj = cfg_get(cfg, ["objective"], {}) or {}
    positive_ratio = float(obj.get("positive_ratio", 0.5))
    rng = np.random.default_rng(None if encoder.training else batch_seed(x))
    v, labels = make_view(x, positive_ratio, rng, head.tasks)
    tokens, _ = encoder.forward_tokens(v)                                       # [B, C*Pt, D]
    logits = head(tokens.mean(dim=1))
    losses = {t: F.cross_entropy(logits[t].float(), labels[t]) for t in head.tasks}
    loss = torch.stack(list(losses.values())).mean()                            # mtl.py: total_loss / total_task
    logs = {"loss": float(loss.detach())}
    for t in head.tasks:
        logs[f"loss_{t}"] = float(losses[t].detach())
        logs[f"acc_{t}"] = float((logits[t].argmax(-1) == labels[t]).float().mean())
    return loss, logs
