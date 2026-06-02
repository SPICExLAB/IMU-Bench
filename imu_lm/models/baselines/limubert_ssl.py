"""Authentic LIMU-BERT objective (Xu et al., SenSys 2021) on the TransfHAR pipeline.

The LIMU-BERT model (parameter-shared Transformer + masked-reconstruction head)
and its span-masking are vendored verbatim from the official repo
(``imu_lm/models/LIMU-BERT-Public/models.py`` + ``utils.py``), made self-contained
(no ``from utils import ...``). Pretraining masks spans of timesteps (zero or
random-replace), encodes, and reconstructs the original values at masked
positions with an MSE loss — exactly as ``pretrain.py``.

Trains on the same balanced OCDetect+realdata windows ([B, T, C]) as v0.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.models.baselines.encoders import to_namespace


# ── Vendored helpers (LIMU-BERT-Public/utils.py + models.py) ──────────────────

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def split_last(x, shape):
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -int(np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    """Verbatim span masking from LIMU-BERT utils.py → list of masked indices."""
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)


# ── Vendored model (LIMU-BERT-Public/models.py) ───────────────────────────────

class LayerNorm(nn.Module):
    def __init__(self, cfg, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden))
        self.beta  = nn.Parameter(torch.zeros(cfg.hidden))
        self.eps   = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        return self.gamma * (x - u) / torch.sqrt(s + self.eps) + self.beta


class Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lin       = nn.Linear(cfg.feature_num, cfg.hidden)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden)
        self.norm      = LayerNorm(cfg)
        self.emb_norm  = bool(getattr(cfg, "emb_norm", True))

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len)
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        self.n_heads = cfg.n_heads

    def forward(self, x):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(t, (self.n_heads, -1)).transpose(1, 2) for t in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = F.softmax(scores, dim=-1)
        h = (scores @ v).transpose(1, 2).contiguous()
        return merge_last(h, 2)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class Transformer(nn.Module):
    """Parameter-shared transformer (LIMU-BERT design): one block applied n_layers times."""

    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.n_layers = cfg.n_layers
        self.attn  = MultiHeadedSelfAttention(cfg)
        self.proj  = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff  = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)

    def forward(self, x):
        h = self.embed(x)
        for _ in range(self.n_layers):
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class LIMUBertModel4Pretrain(nn.Module):
    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = Transformer(cfg)
        self.fc      = nn.Linear(cfg.hidden, cfg.hidden)
        self.linear  = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ   = gelu
        self.norm    = LayerNorm(cfg)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h = self.transformer(input_seqs)
        if self.output_embed:
            return h
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
            h = torch.gather(h, 1, masked_pos)
        h = self.norm(self.activ(self.linear(h)))
        return self.decoder(h)


# ── SSL wrapper ───────────────────────────────────────────────────────────────

class LIMUBertSSL(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.ns = to_namespace(cfg)
        self.model = LIMUBertModel4Pretrain(self.ns)
        self.mask_ratio   = float(cfg.get("mask_ratio", 0.15))
        self.max_gram     = int(cfg.get("max_gram", 10))
        self.mask_prob    = float(cfg.get("mask_prob", 0.8))
        self.replace_prob = float(cfg.get("replace_prob", 0.0))

    def _mask_batch(self, x):
        """x [B, T, C] → (masked_x, masked_pos [B, m], target [B, m, C])."""
        B, T, C = x.shape
        n_pred = max(1, int(round(T * self.mask_ratio)))
        masked_x = x.clone()
        pos_rows = []
        for b in range(B):
            mp = span_mask(T, self.max_gram, goal_num_predict=n_pred)   # list[int]
            r = np.random.rand()
            idx = torch.tensor(mp, device=x.device, dtype=torch.long)
            if r < self.mask_prob:
                masked_x[b, idx, :] = 0.0
            elif r < self.mask_prob + self.replace_prob:
                masked_x[b, idx, :] = torch.rand(len(mp), C, device=x.device)
            # else: keep original values (BERT-style "unchanged" case)
            pos_rows.append(idx)
        m = min(int(p.numel()) for p in pos_rows)
        masked_pos = torch.stack([p[:m] for p in pos_rows], dim=0)       # [B, m]
        target = torch.gather(x, 1, masked_pos.unsqueeze(-1).expand(-1, -1, C))
        return masked_x, masked_pos, target

    def forward_loss(self, batch, device) -> Tuple[torch.Tensor, Dict]:
        x = batch.to(device).float()                       # [B, T, C]
        masked_x, masked_pos, target = self._mask_batch(x)
        recon = self.model(masked_x, masked_pos)           # [B, m, C]
        loss = F.mse_loss(recon, target)
        return loss, {"loss": float(loss.detach()),
                      "mae_loss": float(loss.detach()),
                      "n_masked": float(masked_pos.shape[1])}

    @torch.no_grad()
    def encode(self, x) -> torch.Tensor:
        return self.model.transformer(x).mean(dim=1)       # [B, hidden]
