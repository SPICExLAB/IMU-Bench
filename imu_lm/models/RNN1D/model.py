"""RNN1D: Recurrent encoder and decoder for raw 1D IMU/sensor data.

Encoder: [B, C, T] → [B, embed_dim]
- Transpose to [B, T, C]
- Linear input projection: C → hidden_size
- Multi-layer GRU or LSTM
- Mean pooling (or last hidden state) → [B, hidden_size]
- Optional linear projection → [B, embed_dim]

Decoder: [B, embed_dim] → [B, C, T]  (for MAE / autoencoder objectives)
- Project embedding to hidden_size
- Repeat hidden across T steps → [B, T, hidden_size]
- GRU over the sequence
- Per-step linear head → [B, T, C] → [B, C, T]

Architecture-specific components live here; objective is architecture-agnostic.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from imu_lm.utils.helpers import cfg_get


class RNN1DEncoder(nn.Module):
    """Recurrent encoder for raw 1D IMU signals [B, C, T] → [B, embed_dim].

    Architecture:
    1. Transpose to [B, T, C]
    2. Linear projection C → hidden_size
    3. Multi-layer GRU or LSTM
    4. Pooling over time → [B, hidden_size]
    5. Optional output projection → [B, embed_dim]
    """

    def __init__(self, cfg: Any):
        super().__init__()

        rnn_cfg = cfg_get(cfg, ["rnn1d"], {}) or {}
        enc_cfg = rnn_cfg.get("encoder", {}) or {}
        data_cfg = cfg_get(cfg, ["data"], {}) or {}

        sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
        self.in_channels = len(sensor_cols)

        self.hidden_size = int(enc_cfg.get("hidden_size", 256))
        self.num_layers  = int(enc_cfg.get("num_layers",    2))
        self.rnn_type    = str(enc_cfg.get("rnn_type",   "gru")).lower()
        self.dropout     = float(enc_cfg.get("dropout",   0.1))
        self.bidirectional = bool(enc_cfg.get("bidirectional", False))
        self.pooling     = str(enc_cfg.get("pooling",   "mean")).lower()

        # Optional separate embed_dim; defaults to hidden_size (× 2 if bidirectional)
        num_directions = 2 if self.bidirectional else 1
        rnn_out_dim    = self.hidden_size * num_directions
        embed_dim_cfg  = enc_cfg.get("embed_dim")
        if embed_dim_cfg is not None:
            self.embed_dim   = int(embed_dim_cfg)
            self._proj_out   = nn.Linear(rnn_out_dim, self.embed_dim)
        else:
            self.embed_dim   = rnn_out_dim
            self._proj_out   = None

        # Input projection: C → hidden_size
        self.input_proj = nn.Linear(self.in_channels, self.hidden_size)

        # Recurrent core
        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] → [B, embed_dim]"""
        # [B, C, T] → [B, T, C]
        x = x.transpose(1, 2)

        # Input projection: [B, T, C] → [B, T, hidden_size]
        x = self.input_proj(x)

        # RNN: [B, T, hidden_size] → [B, T, rnn_out_dim]
        out, _ = self.rnn(x)

        # Pooling over time
        if self.pooling == "last":
            emb = out[:, -1]
        elif self.pooling == "max":
            emb = out.max(dim=1).values
        else:  # mean
            emb = out.mean(dim=1)

        # Optional projection to embed_dim
        if self._proj_out is not None:
            emb = self._proj_out(emb)

        return self.norm(emb)  # [B, embed_dim]


class RNN1DDecoder(nn.Module):
    """Recurrent decoder for masked reconstruction [B, embed_dim] → [B, C, T].

    Architecture:
    1. Project embedding to hidden_size → use as initial hidden state h_0
    2. Learned start token repeated T times → [B, T, hidden_size]
    3. GRU over sequence conditioned on h_0
    4. Per-step linear head → [B, T, C] → [B, C, T]
    """

    def __init__(self, embed_dim: int, out_channels: int, target_T: int, cfg: Any):
        super().__init__()

        rnn_cfg = cfg_get(cfg, ["rnn1d"], {}) or {}
        dec_cfg = rnn_cfg.get("decoder", {}) or {}

        self.embed_dim   = embed_dim
        self.out_channels = out_channels
        self.target_T    = target_T

        self.hidden_size = int(dec_cfg.get("hidden_size", 256))
        self.num_layers  = int(dec_cfg.get("num_layers",    1))
        self.dropout     = float(dec_cfg.get("dropout",   0.0))

        # Project encoder embedding → initial hidden state for each layer
        self.embed_to_hidden = nn.Linear(embed_dim, self.num_layers * self.hidden_size)

        # Learnable start input token (repeated across time)
        self.start_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        nn.init.trunc_normal_(self.start_token, std=0.02)

        # Decoder GRU
        self.rnn = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Per-step output head: hidden → C
        self.head = nn.Linear(self.hidden_size, out_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """[B, embed_dim] → [B, C, T]"""
        B = z.shape[0]

        # Build initial hidden state h_0: [num_layers, B, hidden_size]
        h0 = self.embed_to_hidden(z)                         # [B, num_layers * hidden_size]
        h0 = h0.view(B, self.num_layers, self.hidden_size)   # [B, num_layers, hidden_size]
        h0 = h0.permute(1, 0, 2).contiguous()               # [num_layers, B, hidden_size]

        # Expand start token to sequence length
        inp = self.start_token.expand(B, self.target_T, -1)  # [B, T, hidden_size]

        # GRU decode
        out, _ = self.rnn(inp, h0)                           # [B, T, hidden_size]

        # Per-step prediction
        out = self.head(out)                                  # [B, T, C]
        return out.transpose(1, 2)                           # [B, C, T]
