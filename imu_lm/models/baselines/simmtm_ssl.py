"""Authentic SimMTM objective (Dong et al., NeurIPS 2023) on the TransfHAR pipeline.

The SimMTM masked-time-series model (1-D CNN encoder + series-wise contrastive
aggregation + point-wise reconstruction) and its loss are vendored from the
official repo (``imu_lm/models/SimMTM/SimMTM_Classification/code/{model,loss}.py``
and ``utils/augmentations.py``). Conv/head dimensions are sized dynamically so
the model fits our [B, C=6, T=128] windows (the repo hardcodes EEG dims).

Trains on the same balanced OCDetect+realdata windows as v0 (the launcher
passes [B, T, C]; we transpose to channels-first for the CNN).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Vendored masking (utils/augmentations.py) ─────────────────────────────────

def geom_noise_mask_single(L, lm, masking_ratio):
    keep = np.ones(L, dtype=bool)
    p_m = 1 / lm
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    p = [p_m, p_u]
    state = int(np.random.rand() > masking_ratio)
    for i in range(L):
        keep[i] = state
        if np.random.rand() < p[state]:
            state = 1 - state
    return keep


def noise_mask(X, masking_ratio=0.5, lm=3, distribution="geometric"):
    if distribution == "geometric":
        m = geom_noise_mask_single(X.shape[0] * X.shape[1] * X.shape[2], lm, masking_ratio)
        m = m.reshape(X.shape[0], X.shape[1], X.shape[2])
    else:
        m = np.random.choice([True, False], size=X.shape, replace=True,
                             p=(1 - masking_ratio, masking_ratio))
    return torch.tensor(m)


def data_transform_masked4cl(sample, masking_ratio, lm, positive_nums=None,
                             distribution="geometric"):
    """sample [B, C, T] → masked [(positive_nums)*B, C, T]. Verbatim from repo."""
    if positive_nums is None:
        positive_nums = math.ceil(1.5 / (1 - masking_ratio))
    sample = sample.permute(0, 2, 1)                       # [B, T, C]
    rep = sample.repeat(positive_nums, 1, 1)
    mask = noise_mask(rep, masking_ratio, lm, distribution=distribution).to(sample.device)
    x_masked = mask * rep
    return x_masked.permute(0, 2, 1), mask.permute(0, 2, 1)


# ── Vendored loss (loss.py) ───────────────────────────────────────────────────

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num, requires_grad=True))

    def forward(self, *x):
        s = 0
        for i, loss in enumerate(x):
            s += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return s


class ContrastiveWeight(nn.Module):
    def __init__(self, temperature=0.2, positive_nums=3):
        super().__init__()
        self.temperature = temperature
        self.positive_nums = positive_nums
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def get_positive_and_negative_mask(self, sim, n):
        diag = torch.eye(n, dtype=torch.bool, device=sim.device)
        oral = n // (self.positive_nums + 1)
        pos = torch.zeros(sim.size(), device=sim.device)
        for i in range(self.positive_nums + 1):
            pos += torch.diag(torch.ones(n - oral * i, device=sim.device),  oral * i)
            pos += torch.diag(torch.ones(n - oral * i, device=sim.device), -oral * i)
        pos[diag] = 0
        neg = 1 - pos
        neg[diag] = 0
        return pos.bool(), neg.bool()

    def forward(self, emb):
        n = emb.shape[0]
        norm = F.normalize(emb, dim=1)
        sim = norm @ norm.transpose(0, 1)
        pos_m, neg_m = self.get_positive_and_negative_mask(sim, n)
        pos = sim[pos_m].view(n, -1)
        neg = sim[neg_m].view(n, -1)
        logits = torch.cat((pos, neg), dim=-1)
        y = torch.cat((torch.ones(n, pos.shape[-1]), torch.zeros(n, neg.shape[-1])),
                      dim=-1).to(emb.device).float()
        loss = self.kl(self.log_softmax(logits / self.temperature), y)
        return loss, sim, logits, pos_m


class AggregationRebuild(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sim, emb):
        shp = emb.shape
        sim = sim / self.temperature
        sim = sim - torch.eye(shp[0], device=sim.device).float() * 1e12
        w = self.softmax(sim)
        flat = emb.reshape(shp[0], -1)
        rebuilt = torch.matmul(w, flat).reshape(shp[0], shp[1], -1)
        return w, rebuilt


# ── Vendored TFC model (model.py), dynamically sized for [B, C, T] ────────────

class TFC(nn.Module):
    def __init__(self, in_channels=6, seq_len=128, kernel_size=8, stride=1,
                 dropout=0.1, final_out_channels=128, proj_dim=128,
                 temperature=0.2, positive_nums=3):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=kernel_size, stride=stride,
                      bias=False, padding=kernel_size // 2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1), nn.Dropout(dropout))
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(final_out_channels), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        # Infer flattened conv-feature size via a dummy forward.
        with torch.no_grad():
            d = self._conv(torch.zeros(1, in_channels, seq_len))
        self._feat_c, self._feat_l = d.shape[1], d.shape[2]
        flat = self._feat_c * self._feat_l
        self.recon_dim = in_channels * seq_len

        self.dense = nn.Sequential(
            nn.Linear(flat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, proj_dim))
        self.head = nn.Linear(flat, self.recon_dim)

        self.awl = AutomaticWeightedLoss(2)
        self.contrastive = ContrastiveWeight(temperature, positive_nums)
        self.aggregation = AggregationRebuild(temperature)
        self.mse = nn.MSELoss()

    def _conv(self, x):
        return self.conv_block3(self.conv_block2(self.conv_block1(x)))   # [B, C', L']

    def forward(self, x_in_t, pretrain=False):
        x = self._conv(x_in_t)                              # [B, C', L']
        if x.shape[-1] != self._feat_l:
            x = F.adaptive_avg_pool1d(x, self._feat_l)
        h = x.reshape(x.shape[0], -1)
        z = self.dense(h)
        if not pretrain:
            return h, z
        loss_cl, sim, _, _ = self.contrastive(z)
        _, agg = self.aggregation(sim, x)
        pred = self.head(agg.reshape(agg.size(0), -1))
        if pred.shape[-1] != x_in_t.shape[1] * x_in_t.shape[2]:
            pred = pred.reshape(pred.shape[0], x_in_t.shape[1], self.recon_dim // x_in_t.shape[1])
            pred = F.interpolate(pred, size=x_in_t.shape[-1], mode="linear", align_corners=False)
        else:
            pred = pred.reshape_as(x_in_t)
        loss_rb = self.mse(pred, x_in_t.detach())
        loss = self.awl(loss_cl, loss_rb)
        return loss, loss_cl, loss_rb


# ── SSL wrapper ───────────────────────────────────────────────────────────────

class SimMTMSSL(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.masking_ratio = float(cfg.get("masking_ratio", 0.5))
        self.lm            = int(cfg.get("lm", 3))
        self.positive_nums = int(cfg.get("positive_nums", 3))
        self.model = TFC(
            in_channels=int(cfg.get("feature_num", 6)),
            seq_len=int(cfg.get("seq_len", 128)),
            kernel_size=int(cfg.get("kernel_size", 8)),
            stride=int(cfg.get("stride", 1)),
            dropout=float(cfg.get("dropout", 0.1)),
            final_out_channels=int(cfg.get("final_out_channels", 128)),
            proj_dim=int(cfg.get("proj_dim", 128)),
            temperature=float(cfg.get("temperature", 0.2)),
            positive_nums=self.positive_nums,
        )

    def forward_loss(self, batch, device) -> Tuple[torch.Tensor, Dict]:
        x = batch.to(device).float().transpose(1, 2)        # [B, T, C] → [B, C, T]
        masked_m, _ = data_transform_masked4cl(
            x, self.masking_ratio, self.lm, self.positive_nums)
        x_om = torch.cat([x, masked_m.to(device)], dim=0)   # [(1+P)*B, C, T]
        loss, loss_cl, loss_rb = self.model(x_om, pretrain=True)
        return loss, {"loss": float(loss.detach()),
                      "loss_cl": float(loss_cl.detach()),
                      "loss_rb": float(loss_rb.detach())}

    @torch.no_grad()
    def encode(self, x) -> torch.Tensor:
        x = x.transpose(1, 2)                               # [B, T, C] → [B, C, T]
        h, _ = self.model(x, pretrain=False)
        return h                                            # [B, C'*L'] flattened features
