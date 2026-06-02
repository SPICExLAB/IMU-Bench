"""Authentic FOCAL objective (Liu et al., NeurIPS 2023) on the TransfHAR pipeline.

The FOCAL *loss* (shared/private contrastive + orthogonality + temporal-locality
ranking) and its helpers are vendored verbatim from the original repo
(``imu_lm/models/focal/src/models/loss.py`` + ``FOCALModules.py`` +
``general_utils/tensor_utils.py``), adapted only to take explicit parameters
instead of the repo's ``args.dataset_config`` object.

For a single 6-axis IMU the two natural FOCAL "modalities" are the
accelerometer (channels 0–2) and gyroscope (channels 3–5). Each window
[B, T, C] is split into ``seq_len`` temporal sub-segments; the shared backbone
encodes every (modality, sub-segment) and a per-modality projector maps it to
the factorized latent space. This matches FOCAL's modality × sub-sequence
structure while training on the exact same balanced OCDetect+realdata windows
as the v0 MAE run.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from imu_lm.models.baselines.encoders import build_sequence_encoder, to_namespace


# ── Vendored helpers (general_utils.tensor_utils + FOCALModules) ──────────────

def extract_non_diagonal_matrix(inp: torch.Tensor) -> torch.Tensor:
    """Non-diagonal elements of [b, n, n] → [b, n, n-1]. Verbatim from repo."""
    flat = inp.reshape([-1, inp.shape[-2], inp.shape[-1]])
    b, n, _ = flat.shape
    nd = flat.flatten(start_dim=1)[:, 1:]
    nd = nd.view(b, n - 1, n + 1)[:, :, :-1]
    return nd.reshape([b, n, n - 1])


def split_features(mod_features: Dict[str, torch.Tensor]) -> Dict[str, dict]:
    """Split each modality feature into shared/private halves. Verbatim from repo."""
    out = {}
    for mod in mod_features:
        f = mod_features[mod]
        if f.ndim == 2:
            d = f.shape[1] // 2
            out[mod] = {"shared": f[:, 0:d], "private": f[:, d:]}
        else:
            _, _, dim = f.shape
            d = dim // 2
            out[mod] = {"shared": f[:, :, 0:d], "private": f[:, :, d:2 * d]}
    return out


# ── Authentic FOCAL loss (adapted to explicit params) ─────────────────────────

class FOCALLoss(nn.Module):
    """FOCAL objective: shared + private contrastive + orthogonality + temporal."""

    def __init__(self, device, seq_len, modalities,
                 temperature=0.07, inter_rank_margin=1.0,
                 shared_w=1.0, private_w=1.0, ortho_w=3.0, rank_w=5.0,
                 no_private=False):
        super().__init__()
        self.device      = device
        self.seq_len     = int(seq_len)
        self.modalities  = list(modalities)
        self.temperature = float(temperature)
        self.shared_w, self.private_w = float(shared_w), float(private_w)
        self.ortho_w, self.rank_w     = float(ortho_w), float(rank_w)
        self.no_private  = bool(no_private)

        self.criterion           = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f        = nn.CosineSimilarity(dim=-1)
        self.orthonal_loss_f     = nn.CosineEmbeddingLoss(reduction="mean")
        self.inter_ranking_loss_f = nn.MarginRankingLoss(margin=float(inter_rank_margin),
                                                         reduction="mean")

    def mask_correlated_samples(self, seq_len, batch_size, temporal=False):
        if temporal:
            mask = torch.ones([batch_size, batch_size], dtype=bool, device=self.device)
            mask = mask.fill_diagonal_(0)
            mask = mask.repeat_interleave(seq_len, 0).repeat_interleave(seq_len, 1)
        else:
            N = 2 * batch_size
            diag = torch.eye(batch_size, device=self.device)
            mask = torch.ones((N, N), device=self.device).fill_diagonal_(0)
            mask[0:batch_size, batch_size:2 * batch_size] -= diag
            mask[batch_size:2 * batch_size, 0:batch_size] -= diag
            mask = mask.unsqueeze(0).repeat(seq_len, 1, 1).bool()
        return mask

    def forward_contrastive_loss(self, e1, e2, finegrain=False):
        batch, seq, dim = e1.shape
        if finegrain:
            in1, in2 = e1, e2
            N, dim_parallel, dim_compare = 2 * seq, batch, seq
        else:
            in1, in2 = e1.transpose(0, 1), e2.transpose(0, 1)
            N, dim_parallel, dim_compare = 2 * batch, seq, batch
        z = torch.cat((in1, in2), dim=1)
        sim = self.similarity_f(z.unsqueeze(2), z.unsqueeze(1)) / self.temperature
        sim_i_j = torch.diagonal(sim,  dim_compare, dim1=-2, dim2=-1)
        sim_j_i = torch.diagonal(sim, -dim_compare, dim1=-2, dim2=-1)
        pos = torch.cat((sim_i_j, sim_j_i), dim=1).reshape(dim_parallel, N, 1)
        neg = sim[self.mask_correlated_samples(dim_parallel, dim_compare)].reshape(dim_parallel, N, -1)
        labels = torch.zeros(dim_parallel * N, device=pos.device).long()
        logits = torch.cat((pos, neg), dim=2).reshape(dim_parallel * N, -1)
        return self.criterion(logits, labels)

    def forward_orthogonality_loss(self, e1, e2):
        f1 = e1.reshape(-1, e1.shape[-1])
        f2 = e2.reshape(-1, e2.shape[-1])
        return self.orthonal_loss_f(f1, f2, target=-torch.ones(f1.shape[0], device=e1.device))

    def forward_temporal_inter_ranking_loss(self, emb):
        b, s, dim = emb.shape
        if b <= 1:
            return emb.new_tensor(0.0)
        flat = emb.reshape(b * s, dim)
        dist = torch.cdist(flat, flat, p=2).reshape(b, s, b, s).permute(0, 2, 1, 3)
        mask = torch.ones(b * s, b * s, device=self.device).fill_diagonal_(0)
        mask = mask.reshape(b, s, b, s).permute(0, 2, 1, 3)
        dist = (dist * mask).sum(dim=[2, 3]) / mask.sum(dim=[2, 3]).clamp_min(1.0)
        intra = torch.diagonal(dist, 0, dim1=0, dim2=1).repeat_interleave(b - 1)
        inter = extract_non_diagonal_matrix(dist).flatten()
        return self.inter_ranking_loss_f(intra, inter, -torch.ones_like(intra, device=self.device))

    def forward(self, mod_features1, mod_features2, index=None):
        S = self.seq_len
        r1, r2 = {}, {}
        for mod in self.modalities:
            r1[mod] = mod_features1[mod].reshape(-1, S, mod_features1[mod].shape[-1])
            r2[mod] = mod_features2[mod].reshape(-1, S, mod_features2[mod].shape[-1])
        s1, s2 = split_features(r1), split_features(r2)

        shared = 0.0
        if self.no_private:
            for mf in [r1, r2]:
                for i, m1 in enumerate(self.modalities):
                    for m2 in self.modalities[i + 1:]:
                        shared += self.forward_contrastive_loss(mf[m1], mf[m2])
        else:
            for sf in [s1, s2]:
                for i, m1 in enumerate(self.modalities):
                    for m2 in self.modalities[i + 1:]:
                        shared += self.forward_contrastive_loss(sf[m1]["shared"], sf[m2]["shared"])

        private = 0.0
        for mod in self.modalities:
            private += self.forward_contrastive_loss(s1[mod]["private"], s2[mod]["private"])

        temporal = 0.0
        for mf in [r1, r2]:
            for mod in self.modalities:
                temporal += self.forward_temporal_inter_ranking_loss(mf[mod])

        ortho = 0.0
        for sf in [s1, s2]:
            for i, mod in enumerate(self.modalities):
                ortho += self.forward_orthogonality_loss(sf[mod]["shared"], sf[mod]["private"])
                for m2 in self.modalities[i + 1:]:
                    ortho += self.forward_orthogonality_loss(sf[mod]["private"], sf[m2]["private"])

        loss = (shared * self.shared_w + private * self.private_w
                + ortho * self.ortho_w + temporal * self.rank_w)
        logs = {
            "loss":            float(loss.detach()),
            "loss_shared":     float(shared.detach()) if torch.is_tensor(shared) else 0.0,
            "loss_private":    float(private.detach()) if torch.is_tensor(private) else 0.0,
            "loss_ortho":      float(ortho.detach()) if torch.is_tensor(ortho) else 0.0,
            "loss_temporal":   float(temporal.detach()) if torch.is_tensor(temporal) else 0.0,
        }
        return loss, logs


# ── Backbone: window [B,T,C] → per-modality, per-sub-segment features ──────────

_MODALITY_CHANNELS = {"acc": (0, 3), "gyro": (3, 6)}


class FOCALBackbone(nn.Module):
    """Shared encoder + per-modality projectors over temporal sub-segments."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.seq_len     = int(cfg.get("focal_seq_len", 4))     # sub-segments per window
        self.window_len  = int(cfg.get("seq_len", 128))
        self.seg_len     = self.window_len // self.seq_len
        self.modalities  = list(cfg.get("modalities", ["acc", "gyro"]))
        hidden           = int(cfg.get("hidden", 128))
        emb_dim          = int(cfg.get("emb_dim", 128))
        assert emb_dim % 2 == 0, "emb_dim must be even (FOCAL splits shared/private)"

        # Shared backbone over a sub-segment of one modality (3 channels).
        enc_ns = to_namespace({**cfg, "feature_num": 3, "seq_len": self.seg_len})
        self.encoder = build_sequence_encoder(enc_ns)

        self.projectors = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, emb_dim))
            for m in self.modalities
        })
        self.use_dual_modalities = len(self.modalities) >= 2

    def forward(self, x: torch.Tensor, proj_head: bool = True) -> Dict[str, torch.Tensor]:
        """x [B, T, C] → {modality: [B*seq_len, emb_dim]}."""
        B = x.shape[0]
        seg_len = x.shape[1] // self.seq_len
        xs = x[:, :self.seq_len * seg_len].reshape(B, self.seq_len, seg_len, x.shape[-1])
        out = {}
        for m in self.modalities:
            lo, hi = _MODALITY_CHANNELS[m]
            xm = xs[..., lo:hi].reshape(B * self.seq_len, seg_len, hi - lo)  # [B*S, seg, 3]
            feat = self.encoder(xm).mean(dim=1)                      # [B*S, hidden]
            out[m] = self.projectors[m](feat) if proj_head else feat
        return out


# ── SSL wrapper (uniform interface for run_ssl_baseline.py) ────────────────────

def _augment_view(x: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(x) * 0.05
    scale = 1.0 + torch.randn(x.shape[0], 1, x.shape[2], device=x.device) * 0.1
    out = x * scale + noise
    shift = int(np.random.randint(-5, 6))
    if shift != 0:
        out = torch.roll(out, shifts=shift, dims=1)
    return out


class FOCALSSL(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.backbone = FOCALBackbone(cfg)
        self.seq_len  = self.backbone.seq_len
        self.modalities = self.backbone.modalities
        self._loss_kwargs = dict(
            seq_len=self.seq_len, modalities=self.modalities,
            temperature=float(cfg.get("temperature", 0.07)),
            inter_rank_margin=float(cfg.get("inter_rank_margin", 1.0)),
            shared_w=float(cfg.get("shared_contrastive_loss_weight", 1.0)),
            private_w=float(cfg.get("private_contrastive_loss_weight", 1.0)),
            ortho_w=float(cfg.get("orthogonal_loss_weight", 3.0)),
            rank_w=float(cfg.get("rank_loss_weight", 5.0)),
        )
        self._loss = None  # built lazily once we know the device

    def forward_loss(self, batch, device) -> Tuple[torch.Tensor, Dict]:
        if self._loss is None or self._loss.device != device:
            self._loss = FOCALLoss(device=device, **self._loss_kwargs)
        x = batch.to(device).float()                       # [B, T, C]
        v1, v2 = _augment_view(x), _augment_view(x)
        f1 = self.backbone(v1, proj_head=True)
        f2 = self.backbone(v2, proj_head=True)
        return self._loss(f1, f2)

    @torch.no_grad()
    def encode(self, x) -> torch.Tensor:
        feats = self.backbone(x, proj_head=False)          # {mod: [B*S, hidden]}
        B = x.shape[0]
        # mean over sub-segments, concat modalities
        pooled = [feats[m].reshape(B, self.seq_len, -1).mean(1) for m in self.modalities]
        return torch.cat(pooled, dim=-1)
