"""DeepConvLSTM encoder for raw IMU signals (Ordonez & Roggen 2016).

Architecture from "Deep Convolutional and LSTM Recurrent Neural Networks for
Multimodal Wearable Activity Recognition" (Sensors 2016):
- 4 Conv layers (64 filters, kernel=5, stride=1, no padding), applied along
  the time axis independently per sensor channel (kernel width=1 over the
  channel axis, so channels are never mixed by the conv stack -- matching
  the original paper's per-channel convolution).
- 2-layer LSTM (128 hidden units) over the resulting per-timestep features.
- Output: pooled embedding [B, embed_dim] (embed_dim = lstm hidden size).

This encoder is objective-agnostic; a classification head is attached
externally (see run.py), same pattern as CNN1D/RNN1D in this repo.

Input: raw 1D IMU [B, C, T] (channels x time)
"""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn

from imu_lm.utils.helpers import cfg_get


class DeepConvLSTMEncoder(nn.Module):
    """DeepConvLSTM encoder for raw IMU signals [B, C, T] -> [B, embed_dim]."""

    def __init__(self, cfg: Any):
        super().__init__()

        from imu_lm.data.windowing import compute_T_and_hop

        dclstm_cfg = cfg_get(cfg, ["deepconvlstm"], {}) or {}
        enc_cfg = dclstm_cfg.get("encoder", {}) or {}
        data_cfg = cfg_get(cfg, ["data"], {}) or {}

        sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
        self.in_channels = len(sensor_cols)

        conv_filters: List[int] = enc_cfg.get("conv_filters", [64, 64, 64, 64])
        kernel_size: int = int(enc_cfg.get("kernel_size", 5))
        conv_dropout: float = float(enc_cfg.get("conv_dropout", 0.5))
        lstm_hidden: int = int(enc_cfg.get("lstm_hidden_size", 128))
        lstm_layers: int = int(enc_cfg.get("lstm_num_layers", 2))
        lstm_dropout: float = float(enc_cfg.get("lstm_dropout", 0.5))
        self.pooling: str = str(enc_cfg.get("pooling", "last")).lower()

        # Conv stack: Conv2d(1, filters, kernel=(k,1)) over [B, 1, T, C] --
        # kernel width 1 on the channel axis means channels are convolved
        # independently, exactly like the original per-channel design.
        self.conv_layers = nn.ModuleList()
        prev_ch = 1
        for f in conv_filters:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(prev_ch, f, kernel_size=(kernel_size, 1), stride=1),
                nn.ReLU(inplace=True),
            ))
            prev_ch = f
        self.conv_dropout = nn.Dropout(conv_dropout)

        # Compute post-conv shape via dummy forward
        T, _ = compute_T_and_hop(cfg)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, T, self.in_channels)
            for conv in self.conv_layers:
                dummy = conv(dummy)
            self.T_out = dummy.shape[2]
            lstm_input_size = dummy.shape[1] * dummy.shape[3]  # filters * channels

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        self.embed_dim = lstm_hidden
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] -> [B, embed_dim]."""
        # [B, C, T] -> [B, 1, T, C]
        x = x.transpose(1, 2).unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)
        x = self.conv_dropout(x)  # [B, F, T', C]

        # [B, F, T', C] -> [B, T', F, C] -> [B, T', F*C]
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        out, _ = self.lstm(x)  # [B, T', hidden]

        if self.pooling == "mean":
            emb = out.mean(dim=1)
        elif self.pooling == "max":
            emb = out.max(dim=1).values
        else:  # last
            emb = out[:, -1]

        return emb
