from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.utils.helpers import cfg_get


class PatchEmbed1D(nn.Module):
    """1D Patch Embedding: [B, C, T] -> [B, N, D], N = C * (T // patch_size)."""

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        shared_kernel: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.shared_kernel = shared_kernel

        if shared_kernel:
            self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv1d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] -> [B, C * P, D]."""
        B, C, T = x.shape

        if self.shared_kernel:
            patches = []
            for c in range(C):
                patch_c = self.proj(x[:, c : c + 1, :])   # [B, D, P]
                patches.append(patch_c.transpose(1, 2))   # [B, P, D]
            out = torch.cat(patches, dim=1)               # [B, C*P, D]
        else:
            # Less semantically clean for IMU, but preserved for compatibility
            out = self.proj(x).transpose(1, 2)            # [B, P, D]

        return out


class PositionalEmbedding2D(nn.Module):
    """2D positional embedding: learned time + learned channel."""

    def __init__(
        self,
        max_patches_per_channel: int,
        max_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.time_embed = nn.Embedding(max_patches_per_channel, embed_dim)
        self.channel_embed = nn.Embedding(max_channels, embed_dim)

    def forward(
        self,
        num_patches_per_channel: int,
        num_channels: int,
        device: torch.device,
    ) -> torch.Tensor:
        time_pos = torch.arange(num_patches_per_channel, device=device).repeat(num_channels)
        channel_pos = torch.arange(num_channels, device=device).repeat_interleave(
            num_patches_per_channel
        )
        return self.time_embed(time_pos) + self.channel_embed(channel_pos)


class SinusoidalPositionalEmbedding2D(nn.Module):
    """2D positional embedding: sinusoidal for time, learned for channel."""

    def __init__(
        self,
        max_patches_per_channel: int,
        max_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.channel_embed = nn.Embedding(max_channels, embed_dim)
        self.register_buffer(
            "time_embed_table",
            self._build_sinusoidal(max_patches_per_channel, embed_dim),
        )

    @staticmethod
    def _build_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe

    def forward(
        self,
        num_patches_per_channel: int,
        num_channels: int,
        device: torch.device,
    ) -> torch.Tensor:
        if num_patches_per_channel <= self.time_embed_table.shape[0]:
            time_pe = self.time_embed_table[:num_patches_per_channel]
        else:
            time_pe = self._build_sinusoidal(num_patches_per_channel, self.embed_dim).to(device)

        time_pos_embed = (
            time_pe.unsqueeze(0)
            .expand(num_channels, -1, -1)
            .reshape(num_channels * num_patches_per_channel, self.embed_dim)
        )

        channel_pos = torch.arange(num_channels, device=device)
        ch_embed = self.channel_embed(channel_pos)
        channel_pos_embed = (
            ch_embed.unsqueeze(1)
            .expand(-1, num_patches_per_channel, -1)
            .reshape(num_channels * num_patches_per_channel, self.embed_dim)
        )

        return time_pos_embed + channel_pos_embed


class TemporalDownsample(nn.Module):
    """
    Downsample only along time-patch dimension.

    Input:  [B, C, P, D_in]
    Output: [B, C, P_out, D_out]
    """

    def __init__(self, dim_in: int, dim_out: int, stride: int = 2, mode: str = "avg"):
        super().__init__()
        self.stride = stride
        self.mode = mode
        self.proj = nn.Linear(dim_in, dim_out)

        if mode == "conv":
            self.pool = nn.Conv1d(
                dim_in,
                dim_in,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=dim_in,
            )
        else:
            self.pool = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, P, D]
        B, C, P, D = x.shape

        if self.stride == 1:
            return self.proj(x)

        if self.mode == "avg":
            pad = (self.stride - (P % self.stride)) % self.stride
            if pad > 0:
                # pad P dimension
                x = F.pad(x, (0, 0, 0, pad), mode="replicate")
            B, C, P_pad, D = x.shape
            x = x.view(B, C, P_pad // self.stride, self.stride, D).mean(dim=3)

        elif self.mode == "max":
            x_ = x.permute(0, 1, 3, 2).reshape(B * C, D, P)  # [B*C, D, P]
            x_ = F.max_pool1d(x_, kernel_size=self.stride, stride=self.stride, ceil_mode=True)
            P_out = x_.shape[-1]
            x = x_.reshape(B, C, D, P_out).permute(0, 1, 3, 2)

        elif self.mode == "conv":
            x_ = x.permute(0, 1, 3, 2).reshape(B * C, D, P)  # [B*C, D, P]
            x_ = self.pool(x_)                               # [B*C, D, P_out]
            P_out = x_.shape[-1]
            x = x_.reshape(B, C, D, P_out).permute(0, 1, 3, 2)

        else:
            raise ValueError(f"Unknown downsample mode: {self.mode}")

        x = self.proj(x)
        return x


class HierarchicalTransformerBlock1D(nn.Module):
    """
    One block:
      optional temporal downsample
      -> transformer block on flattened [C * P] tokens
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        downsample_stride: int = 1,
        downsample_mode: str = "avg",
    ):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.downsample_stride = downsample_stride

        self.downsample = None
        if downsample_stride > 1 or dim_in != dim_out:
            self.downsample = TemporalDownsample(
                dim_in=dim_in,
                dim_out=dim_out,
                stride=downsample_stride,
                mode=downsample_mode,
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_out,
            nhead=num_heads,
            dim_feedforward=int(dim_out * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, P, D_in]
        out: [B, C, P_out, D_out]
        """
        if self.downsample is not None:
            x = self.downsample(x)  # [B, C, P_out, D_out]

        B, C, P, D = x.shape
        tokens = x.reshape(B, C * P, D)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        x = tokens.view(B, C, P, D)
        return x


class HierarchicalStage1D(nn.Module):
    """
    Stage with several blocks.
    Only the first block of a stage may downsample / change dim.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        downsample_stride: int = 1,
        downsample_mode: str = "avg",
    ):
        super().__init__()

        blocks = [
            HierarchicalTransformerBlock1D(
                dim_in=dim_in,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                downsample_stride=downsample_stride,
                downsample_mode=downsample_mode,
            )
        ]
        for _ in range(depth - 1):
            blocks.append(
                HierarchicalTransformerBlock1D(
                    dim_in=dim_out,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    downsample_stride=1,
                    downsample_mode=downsample_mode,
                )
            )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class HierarchicalViT1DEncoder(nn.Module):
    """
    Hierarchical ViT-1D Encoder for IMU.

    Key idea:
    - tokenise each channel independently into time patches
    - keep channel identity
    - perform multiscale hierarchy along time only
    - hidden dim grows across stages

    Returns:
      forward(x, mask=None):
        - if mask is None: pooled [B, D_last]
        - if mask provided: Not implemented for hierarchical masked token dropping
                            (raises NotImplementedError)
      forward_tokens(x):
        - [B, C * P_last, D_last], P_last
    """

    def __init__(self, cfg: Any):
        super().__init__()

        vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
        hcfg = vit_cfg.get("hierarchical_encoder", {}) or {}
        data_cfg = cfg_get(cfg, ["data"], {}) or {}

        sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
        self.in_channels = len(sensor_cols)

        self.patch_size = int(vit_cfg.get("patch_size", 16))
        self.pooling = vit_cfg.get("pooling", "mean")
        shared_kernel = bool(vit_cfg.get("shared_kernel", True))

        # initial embedding dim
        embed_dim0 = int(hcfg.get("embed_dim0", 128))

        # hierarchical structure
        dims = list(hcfg.get("dims", [128, 256, 384]))
        depths = list(hcfg.get("depths", [2, 3, 4]))
        heads = list(hcfg.get("heads", [4, 8, 8]))
        downsample_strides = list(hcfg.get("downsample_strides", [1, 2, 2]))
        mlp_ratio = float(hcfg.get("mlp_ratio", 4.0))
        dropout = float(hcfg.get("hidden_dropout_prob", 0.0))
        attn_dropout = float(hcfg.get("attention_probs_dropout_prob", 0.0))
        downsample_mode = str(hcfg.get("downsample_mode", "avg"))

        assert len(dims) == len(depths) == len(heads) == len(downsample_strides), (
            "hierarchical_encoder.{dims,depths,heads,downsample_strides} "
            "must have same length"
        )

        max_patches = int(vit_cfg.get("max_patches_per_channel", 256))
        max_channels = int(vit_cfg.get("max_channels", 32))
        pos_embed_type = str(vit_cfg.get("pos_embed_type", "sinusoidal"))

        self.patch_embed = PatchEmbed1D(
            in_channels=self.in_channels,
            patch_size=self.patch_size,
            embed_dim=embed_dim0,
            shared_kernel=shared_kernel,
        )

        if pos_embed_type == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding2D(
                max_patches_per_channel=max_patches,
                max_channels=max_channels,
                embed_dim=embed_dim0,
            )
        else:
            self.pos_embed = PositionalEmbedding2D(
                max_patches_per_channel=max_patches,
                max_channels=max_channels,
                embed_dim=embed_dim0,
            )

        stages = []
        dim_in = embed_dim0
        for dim_out, depth, nhead, stride in zip(dims, depths, heads, downsample_strides):
            stages.append(
                HierarchicalStage1D(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    depth=depth,
                    num_heads=nhead,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    downsample_stride=stride,
                    downsample_mode=downsample_mode,
                )
            )
            dim_in = dim_out

        self.stages = nn.ModuleList(stages)
        self.embed_dim = dim_in
        self.norm = nn.LayerNorm(self.embed_dim)

    def _tokens_to_grid(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        [B, C, T] -> [B, C, P, D0]
        """
        B, C, T = x.shape
        num_patches_per_channel = T // self.patch_size

        tokens = self.patch_embed(x)  # [B, C*P, D0]
        pos_embed = self.pos_embed(num_patches_per_channel, C, x.device)
        tokens = tokens + pos_embed.unsqueeze(0)

        grid = tokens.view(B, C, num_patches_per_channel, -1)
        return grid, num_patches_per_channel

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
            mask: currently unsupported in this hierarchical version

        Returns:
            [B, D_last] pooled feature
        """
        if mask is not None:
            raise NotImplementedError(
                "MAE-style token dropping before encoder is not implemented for "
                "HierarchicalViT1DEncoder yet. Easiest path is: pretrain without "
                "dropout-removal MAE, or redesign the decoder around multiscale tokens."
            )

        tokens, _ = self.forward_tokens(x)

        if self.pooling == "mean":
            return tokens.mean(dim=1)
        elif self.pooling == "cls":
            return tokens[:, 0]
        else:
            return tokens.mean(dim=1)

    def forward_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Returns:
            tokens: [B, C * P_last, D_last]
            num_patches_per_channel_last: P_last
        """
        grid, _ = self._tokens_to_grid(x)  # [B, C, P0, D0]

        for stage in self.stages:
            grid = stage(grid)

        B, C, P_last, D_last = grid.shape
        tokens = grid.reshape(B, C * P_last, D_last)
        tokens = self.norm(tokens)

        return tokens, P_last
    
    
    
    
    
    
    
    
    
    