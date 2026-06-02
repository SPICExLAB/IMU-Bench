"""Shared sequence-encoder factory for the SSL baselines (SimMTM / LIMU-BERT / FOCAL).

All three baseline models call ``build_sequence_encoder(cfg)`` to obtain a
backbone that maps ``[B, T, C] -> [B, T, H]``. Three backbones are available,
selected by ``cfg.encoder_type``:

    "transformer"  : the parameter-shared LIMU-BERT Transformer (default)
    "resnet"       : a 1-D residual CNN
    "cnn"          : a plain 1-D CNN

``cfg`` is an attribute object (e.g. ``types.SimpleNamespace``) carrying
``feature_num, seq_len, hidden, n_layers, n_heads, hidden_ff, emb_norm`` etc.
``to_namespace(dict)`` converts a flat config dict into that object.
"""
from __future__ import annotations

import types

import torch
import torch.nn as nn


# ── Tensor helpers (originally utils.utils, needed by the LIMU-BERT Transformer) ──

def split_last(x, shape):
    """Split the last dimension into the given shape."""
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -torch.prod(torch.tensor(shape)).item())
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    """Merge the last n_dims dimensions into one."""
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


# ── Config adapter ────────────────────────────────────────────────────────────

_DEFAULTS = dict(
    feature_num=6, seq_len=128, hidden=128, n_layers=4, n_heads=4,
    hidden_ff=256, emb_norm=True, encoder_type="transformer", proj_dim=128,
)


def to_namespace(cfg: dict) -> types.SimpleNamespace:
    """Flat dict -> SimpleNamespace with sensible defaults filled in."""
    merged = dict(_DEFAULTS)
    merged.update({k: v for k, v in (cfg or {}).items()})
    return types.SimpleNamespace(**merged)


# ── 1-D CNN / ResNet backbones ────────────────────────────────────────────────

class CNN1DEncoder(nn.Module):
    """Plain 1-D CNN: [B, T, C] -> [B, T, H] (length preserved via padding)."""

    def __init__(self, cfg):
        super().__init__()
        c, h = cfg.feature_num, cfg.hidden
        self.net = nn.Sequential(
            nn.Conv1d(c, h, kernel_size=5, padding=2), nn.GELU(), nn.BatchNorm1d(h),
            nn.Conv1d(h, h, kernel_size=5, padding=2), nn.GELU(), nn.BatchNorm1d(h),
            nn.Conv1d(h, h, kernel_size=3, padding=1), nn.GELU(),
        )

    def forward(self, x):                 # [B, T, C]
        return self.net(x.transpose(1, 2)).transpose(1, 2)   # [B, T, H]


class _ResBlock1D(nn.Module):
    def __init__(self, h, k=3):
        super().__init__()
        self.c1 = nn.Conv1d(h, h, k, padding=k // 2)
        self.c2 = nn.Conv1d(h, h, k, padding=k // 2)
        self.n1 = nn.BatchNorm1d(h)
        self.n2 = nn.BatchNorm1d(h)

    def forward(self, x):
        y = torch.relu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return torch.relu(x + y)


class ResNet1DEncoder(nn.Module):
    """1-D residual CNN: [B, T, C] -> [B, T, H] (length preserved)."""

    def __init__(self, cfg):
        super().__init__()
        c, h = cfg.feature_num, cfg.hidden
        n_blocks = int(getattr(cfg, "n_layers", 4))
        self.stem   = nn.Conv1d(c, h, kernel_size=5, padding=2)
        self.blocks = nn.Sequential(*[_ResBlock1D(h) for _ in range(n_blocks)])

    def forward(self, x):                 # [B, T, C]
        h = self.stem(x.transpose(1, 2))
        return self.blocks(h).transpose(1, 2)                # [B, T, H]


# ── Self-contained transformer backbone ──────────────────────────────────────

class TransformerSeqEncoder(nn.Module):
    """Linear input projection + learned positional embedding + transformer.

    [B, T, C] -> [B, T, H]. Self-contained (no dependency on external model dirs).
    """

    def __init__(self, cfg):
        super().__init__()
        c  = int(cfg.feature_num)
        T  = int(cfg.seq_len)
        h  = int(cfg.hidden)
        nl = int(getattr(cfg, "n_layers", 4))
        nh = int(getattr(cfg, "n_heads", 4))
        ff = int(getattr(cfg, "hidden_ff", 256))
        self.input_proj = nn.Linear(c, h)
        self.pos_embed  = nn.Embedding(T, h)
        self.input_norm = nn.LayerNorm(h)
        layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=nh, dim_feedforward=ff, dropout=0.1,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=nl)

    def forward(self, x):                 # [B, T, C]
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.input_norm(self.input_proj(x) + self.pos_embed(pos))
        return self.transformer(h)        # [B, T, H]


# ── Factory ───────────────────────────────────────────────────────────────────

def build_sequence_encoder(cfg):
    """Return a backbone mapping [B, T, C] -> [B, T, H], per cfg.encoder_type."""
    encoder_type = str(getattr(cfg, "encoder_type", "transformer")).lower()
    if encoder_type == "transformer":
        return TransformerSeqEncoder(cfg)
    if encoder_type == "resnet":
        return ResNet1DEncoder(cfg)
    if encoder_type == "cnn":
        return CNN1DEncoder(cfg)
    raise ValueError(f"Unsupported encoder_type: {encoder_type}")
