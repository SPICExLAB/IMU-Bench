"""Mix-Token (Bonazzi et al., arXiv:2605.04791 §3.3.1) — reimplementation.

No official code is released; this follows the paper's textual description
exactly where it is specific and makes the smallest reasonable choice where
it is not (each such choice is marked UNSPECIFIED below). Layer widths are
tuned so the total lands at the paper's stated ~223K parameters.

Architecture
------------
1. Multi-band convolutional branch
   raw [B, C, W] -> fixed band-pass filterbank (N_BANDS views per channel)
   -> [B, C*N_BANDS, W] -> residual 1D CNN -> global temporal pooling
   -> per-channel embeddings concatenated -> MLP -> y_cnn
2. Statistical Transformer branch
   precomputed feature tokens [B, N_TOK, D_FEAT] -> linear encoder -> + pos
   -> TransformerEncoder -> learnable-query attention pooling -> MLP -> y_attn
3. Fusion head: concat(cnn_emb, attn_emb) -> MLP -> y_fused
4. y = pi0*y_cnn + pi1*y_attn + pi2*y_fused, [pi] = softmax(w), w learnable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt

NATIVE_HZ = 100.0
# UNSPECIFIED: "low-, mid-, and high-frequency" -> 3 bands; edges chosen at
# motion-relevant boundaries within the 0-50Hz Nyquist range.
FILTERBANK = [(0.3, 3.0), (3.0, 10.0), (10.0, 45.0)]


class FixedFilterbank:
    """Numpy band-pass filterbank applied on the CPU side (deterministic,
    no parameters). Returns [W, C*N_BANDS] for a [W, C] window."""

    def __init__(self, bands=FILTERBANK, fs=NATIVE_HZ, order=4):
        self.sos = [butter(order, [lo, hi], btype="band", fs=fs, output="sos") for lo, hi in bands]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        outs = []
        for c in range(X.shape[1]):
            for sos in self.sos:
                outs.append(sosfiltfilt(sos, X[:, c].astype(np.float64)))
        return np.stack(outs, 1).astype(np.float32)

    @property
    def n_bands(self):
        return len(self.sos)


class ResBlock1D(nn.Module):
    def __init__(self, ch, k=5, dropout=0.1, groups=1):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, k, padding=k // 2, groups=groups)
        self.b1 = nn.BatchNorm1d(ch)
        self.c2 = nn.Conv1d(ch, ch, k, padding=k // 2, groups=groups)
        self.b2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.b1(self.c1(x)))
        h = self.drop(h)
        h = self.b2(self.c2(h))
        return F.gelu(x + h)


class ConvBranch(nn.Module):
    """Per-channel residual CNN over the filterbank views: the paper pools
    'a fixed-dimensional embedding per channel, which is then concatenated',
    so every conv is grouped by IMU channel (each channel's N_BANDS views
    are processed by their own filters; no cross-channel mixing inside the
    CNN — that is the fusion head's job). UNSPECIFIED whether the per-channel
    CNNs share weights; we do not share (groups=C)."""

    def __init__(self, n_channels, n_bands, width=32, n_blocks=3, dropout=0.1):
        super().__init__()
        self.C = n_channels
        in_ch = n_channels * n_bands
        self.stem = nn.Conv1d(in_ch, n_channels * width, 7, padding=3, groups=n_channels)
        self.bn = nn.BatchNorm1d(n_channels * width)
        self.blocks = nn.Sequential(*[ResBlock1D(n_channels * width, dropout=dropout, groups=n_channels)
                                      for _ in range(n_blocks)])
        self.out_dim = n_channels * width  # concat of per-channel pooled embeddings

    def forward(self, x):  # x: [B, C*N_BANDS, W]
        h = F.gelu(self.bn(self.stem(x)))
        h = self.blocks(h)
        return h.mean(-1)  # global temporal pooling -> [B, C*width]


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)

    def forward(self, tokens):  # [B, N, d]
        q = self.q.expand(tokens.shape[0], -1, -1)
        out, _ = self.attn(q, tokens, tokens)
        return out[:, 0]


class StatBranch(nn.Module):
    def __init__(self, n_tokens, d_feat, d_model=64, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.enc = nn.Linear(d_feat, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model,
                                           dropout=dropout, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, n_layers)
        self.pool = AttnPool(d_model)
        self.out_dim = d_model

    def forward(self, tok):  # [B, N, d_feat]
        h = self.enc(tok) + self.pos
        h = self.tr(h)
        return self.pool(h)


def _mlp(d_in, d_hidden, n_out, dropout):
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_hidden), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(d_hidden, n_out))


class MixToken(nn.Module):
    # Defaults land at 219.2K params vs the paper's stated 223K (UNSPECIFIED
    # widths; this is the most balanced config within +-5K: 3 residual blocks,
    # a real 1-layer/4-head transformer over 17 tokens, 128-wide MLP heads).
    def __init__(self, n_channels=6, n_bands=3, n_tokens=17, d_feat=21, num_classes=6,
                 cnn_width=24, cnn_blocks=3, d_model=64, tr_layers=1, dropout=0.2):
        super().__init__()
        self.cnn = ConvBranch(n_channels, n_bands, width=cnn_width, n_blocks=cnn_blocks, dropout=0.1)
        self.stat = StatBranch(n_tokens, d_feat, d_model=d_model, n_layers=tr_layers, dropout=0.1)
        self.head_cnn = _mlp(self.cnn.out_dim, 128, num_classes, dropout)
        self.head_attn = _mlp(self.stat.out_dim, 128, num_classes, dropout)
        self.head_fused = _mlp(self.cnn.out_dim + self.stat.out_dim, 128, num_classes, dropout)
        self.fusion_w = nn.Parameter(torch.zeros(3))  # softmax -> [pi0, pi1, pi2], init uniform

    def forward(self, x_fb, tok, return_parts=False):
        e_cnn = self.cnn(x_fb)
        e_attn = self.stat(tok)
        y_cnn = self.head_cnn(e_cnn)
        y_attn = self.head_attn(e_attn)
        y_fused = self.head_fused(torch.cat([e_cnn, e_attn], -1))
        pi = torch.softmax(self.fusion_w, 0)
        y = pi[0] * y_cnn + pi[1] * y_attn + pi[2] * y_fused
        if return_parts:
            return y, (y_cnn, y_attn, y_fused, pi)
        return y

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
