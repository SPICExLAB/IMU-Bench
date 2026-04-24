"""ViT-1D encoder and decoder for raw 1D IMU/sensor data (MAE / LSM-2 style).

Encoder: [B, C, T] → [B, D] (probing) or [B, N_vis, D] (MAE training)
- 1D patch embedding with shared kernel across channels
- 2D positional embedding (time + channel)
- Dropout removal: masked patches removed before transformer (MAE training)
- Transformer encoder layers
- Mean pooling over tokens (probing only)

Decoder: visible encoder tokens + mask → [B, C, T]
- Learnable mask token inserted at masked positions
- Lightweight transformer decoder over full sequence
- Per-patch prediction head

Architecture-specific components live here; objective is architecture-agnostic.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from imu_lm.utils.helpers import cfg_get


class PatchEmbed1D(nn.Module):
    """1D Patch Embedding: [B, C, T] → [B, N, D] where N = C * (T // patch_size)."""
    
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
            # Single Conv1d applied to each channel separately (LSM-2 style)
            self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            # Separate projection per channel
            self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] → [B, N, D] where N = C * num_patches_per_channel."""
        B, C, T = x.shape
        num_patches = T // self.patch_size
        
        if self.shared_kernel:
            # Apply same conv to each channel: [B, C, T] → [B, C, num_patches, D]
            patches = []
            for c in range(C):
                # [B, 1, T] → [B, D, num_patches]
                patch_c = self.proj(x[:, c:c+1, :])  # [B, D, num_patches]
                patches.append(patch_c.transpose(1, 2))  # [B, num_patches, D]
            # Stack channels: [B, C * num_patches, D]
            out = torch.cat(patches, dim=1)
        else:
            # [B, C, T] → [B, D, num_patches] → [B, num_patches, D]
            out = self.proj(x).transpose(1, 2)
        
        return out


class PositionalEmbedding2D(nn.Module):
    """2D positional embedding: separate embeddings for time and channel positions."""
    
    def __init__(
        self,
        max_patches_per_channel: int,
        max_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.time_embed = nn.Embedding(max_patches_per_channel, embed_dim)
        self.channel_embed = nn.Embedding(max_channels, embed_dim)
        self.max_patches_per_channel = max_patches_per_channel
        self.max_channels = max_channels
    
    def forward(self, num_patches_per_channel: int, num_channels: int, device: torch.device) -> torch.Tensor:
        """Generate positional embeddings for [N] tokens where N = num_channels * num_patches_per_channel."""
        # Time positions: [0, 1, 2, ..., num_patches-1] repeated for each channel
        time_pos = torch.arange(num_patches_per_channel, device=device)
        time_pos = time_pos.repeat(num_channels)  # [N]
        
        # Channel positions: [0, 0, ..., 1, 1, ..., C-1, C-1, ...]
        channel_pos = torch.arange(num_channels, device=device)
        channel_pos = channel_pos.repeat_interleave(num_patches_per_channel)  # [N]
        
        # Sum embeddings
        pos_embed = self.time_embed(time_pos) + self.channel_embed(channel_pos)  # [N, D]
        return pos_embed


class SinusoidalPositionalEmbedding2D(nn.Module):
    """2D positional embedding: sinusoidal for time, learned for channel.

    The time embedding uses fixed sinusoidal encoding (generalises to any
    sequence length at test time without re-training).  The channel embedding
    stays learned because the number of sensor channels is small and fixed.

    The forward API is identical to PositionalEmbedding2D, so the two classes
    are drop-in replacements for each other.  Select via config key::

        vit1d:
          pos_embed_type: sinusoidal   # default: "learned"
    """

    def __init__(
        self,
        max_patches_per_channel: int,
        max_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_patches_per_channel = max_patches_per_channel

        # Learned channel embedding (bounded by number of sensor channels)
        self.channel_embed = nn.Embedding(max_channels, embed_dim)

        # Pre-compute sinusoidal time table; stored as a non-trainable buffer
        # so it moves with the module when .cuda() / .to(device) is called.
        self.register_buffer(
            "time_embed_table",
            self._build_sinusoidal(max_patches_per_channel, embed_dim),
        )  # [max_patches_per_channel, embed_dim]

    @staticmethod
    def _build_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        """Standard transformer sinusoidal encoding [max_len, d_model]."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [L, 1]
        half_d = d_model // 2
        div_term = torch.exp(
            torch.arange(0, half_d, dtype=torch.float) * (-math.log(10000.0) / half_d)
        )  # [half_d]
        pe[:, 0::2] = torch.sin(position * div_term)                    # even dims
        pe[:, 1::2] = torch.cos(position * div_term[: d_model - half_d])  # odd dims
        return pe

    def forward(
        self, num_patches_per_channel: int, num_channels: int, device: torch.device
    ) -> torch.Tensor:
        """Return [N, D] positional embeddings; N = num_channels * num_patches_per_channel."""
        # -- time embedding -----------------------------------------------
        if num_patches_per_channel <= self.time_embed_table.shape[0]:
            time_pe = self.time_embed_table[:num_patches_per_channel]  # [P, D]
        else:
            # Sequence longer than pre-computed table: compute on-the-fly
            time_pe = self._build_sinusoidal(
                num_patches_per_channel, self.embed_dim
            ).to(device)

        # Expand to channel-major layout [C0_P0, C0_P1, ..., C1_P0, ...]
        time_pos_embed = (
            time_pe.unsqueeze(0)                      # [1, P, D]
            .expand(num_channels, -1, -1)             # [C, P, D]
            .reshape(num_channels * num_patches_per_channel, self.embed_dim)
        )  # [C*P, D]

        # -- channel embedding --------------------------------------------
        channel_pos = torch.arange(num_channels, device=device)  # [C]
        ch_embed = self.channel_embed(channel_pos)                # [C, D]
        channel_pos_embed = (
            ch_embed.unsqueeze(1)                     # [C, 1, D]
            .expand(-1, num_patches_per_channel, -1)  # [C, P, D]
            .reshape(num_channels * num_patches_per_channel, self.embed_dim)
        )  # [C*P, D]

        return time_pos_embed + channel_pos_embed  # [N, D]


class ViT1DEncoder(nn.Module):
    """ViT-1D Encoder: [B, C, T] → [B, embed_dim].
    
    Architecture (LSM-2 style):
    - 1D patch embedding with shared kernel
    - 2D positional embedding (time + channel)
    - Transformer encoder
    - Mean pooling
    """
    
    def __init__(self, cfg: Any):
        super().__init__()
        
        # Get config
        vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
        enc_cfg = vit_cfg.get("encoder", {}) or {}
        data_cfg = cfg_get(cfg, ["data"], {}) or {}
        
        # Input dimensions
        sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
        self.in_channels = len(sensor_cols)
        
        # Patch embedding config (top-level vit1d)
        self.patch_size = int(vit_cfg.get("patch_size", 16))
        shared_kernel = bool(vit_cfg.get("shared_kernel", True))
        
        # Encoder config (nested under vit1d.encoder)
        self.embed_dim = int(enc_cfg.get("hidden_size", 384))
        num_layers = int(enc_cfg.get("num_hidden_layers", 12))
        num_heads = int(enc_cfg.get("num_attention_heads", 6))
        mlp_ratio = float(enc_cfg.get("intermediate_size", 1536)) / self.embed_dim
        dropout = float(enc_cfg.get("hidden_dropout_prob", 0.0))
        attn_dropout = float(enc_cfg.get("attention_probs_dropout_prob", 0.0))
        layer_norm_eps = float(enc_cfg.get("layer_norm_eps", 1e-6))
        
        # Positional embedding config (top-level vit1d)
        max_patches = int(vit_cfg.get("max_patches_per_channel", 256))
        max_channels = int(vit_cfg.get("max_channels", 32))
        
        # Pooling
        self.pooling = vit_cfg.get("pooling", "mean")
        
        # Layers
        self.patch_embed = PatchEmbed1D(
            in_channels=self.in_channels,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            shared_kernel=shared_kernel,
        )
        
        pos_embed_type = str(vit_cfg.get("pos_embed_type", "learned"))
        if pos_embed_type == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding2D(
                max_patches_per_channel=max_patches,
                max_channels=max_channels,
                embed_dim=self.embed_dim,
            )
        else:
            self.pos_embed = PositionalEmbedding2D(
                max_patches_per_channel=max_patches,
                max_channels=max_channels,
                embed_dim=self.embed_dim,
            )

        # Use PyTorch's built-in transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # pre-norm like ViT
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(self.embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
        return_keep_padding_mask: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional MAE masking.
        
        Args:
            x: [B, C, T] raw input signal
            mask: [B, N] boolean, True = masked (removed before transformer).
                  If None, process all tokens and return mean-pooled [B, D].
                  
        Returns:
            If mask/padding_mask is None: [B, D] mean-pooled embedding (for probing/eval)
            If mask or padding_mask given: [B, N_keep_max, D] padded visible-token outputs
        """
        B, C, T = x.shape
        num_patches_per_channel = T // self.patch_size
        
        # Patch embedding: [B, C, T] → [B, N, D]
        tokens = self.patch_embed(x)
        
        # Add positional embedding
        pos_embed = self.pos_embed(num_patches_per_channel, C, x.device)
        tokens = tokens + pos_embed.unsqueeze(0)
        
        keep_padding_mask = None
        if mask is not None or padding_mask is not None:
            token_drop_mask = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=x.device)
            if mask is not None:
                token_drop_mask |= mask
            if padding_mask is not None:
                token_drop_mask |= padding_mask

            ids_shuffle = torch.argsort(token_drop_mask.int(), dim=1)
            keep_counts = (~token_drop_mask).sum(dim=1)
            max_keep = int(keep_counts.max().item())
            if max_keep <= 0:
                raise ValueError("ViT1DEncoder.forward: no visible non-padding tokens in batch")
            ids_keep = ids_shuffle[:, :max_keep]
            tokens = torch.gather(
                tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            )
            keep_padding_mask = (
                torch.arange(max_keep, device=x.device).unsqueeze(0).expand(B, -1)
                >= keep_counts.unsqueeze(1)
            )
        
        # Transformer encoder
        src_key_padding_mask = None
        if keep_padding_mask is not None and bool(keep_padding_mask.any()):
            src_key_padding_mask = keep_padding_mask
        tokens = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        tokens = self.norm(tokens)
        if keep_padding_mask is not None:
            tokens = tokens.masked_fill(keep_padding_mask.unsqueeze(-1), 0.0)
        
        if mask is not None or padding_mask is not None:
            if return_keep_padding_mask:
                return tokens, keep_padding_mask
            return tokens
        
        # Pooling: [B, N, D] → [B, D] for probing
        if self.pooling == "mean":
            out = tokens.mean(dim=1)
        elif self.pooling == "cls":
            out = tokens[:, 0]
        else:
            out = tokens.mean(dim=1)
        
        return out

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return full per-token encoder output (no pooling).

        Used by ViT1DPoseDecoder, which needs the complete token grid rather
        than a single pooled vector.

        Args:
            x: [B, C, T] raw input signal

        Returns:
            tokens: [B, C * num_patches_per_channel, D]
                    Token layout: [C0_P0, C0_P1, ..., C1_P0, ..., CC-1_Pt-1]
            num_patches_per_channel: int  (= T // patch_size)
        """
        B, C, T = x.shape
        num_patches_per_channel = T // self.patch_size

        tokens = self.patch_embed(x)
        pos_embed = self.pos_embed(num_patches_per_channel, C, x.device)
        tokens = tokens + pos_embed.unsqueeze(0)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        return tokens, num_patches_per_channel


class ViT1DDecoder(nn.Module):
    """ViT-1D MAE Decoder: visible encoder tokens + mask → [B, C, T].
    
    Standard MAE decoder (He et al.):
    1. Project visible encoder tokens to decoder dimension
    2. Insert learnable mask tokens at masked positions
    3. Add positional embeddings to full sequence
    4. Lightweight transformer over full sequence
    5. Per-patch linear head → reconstruct raw values
    
    Throwaway after pretraining — only encoder is kept for probing.
    """
    
    def __init__(self, embed_dim: int, out_channels: int, target_T: int, cfg: Any):
        super().__init__()
        
        vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
        dec_cfg = vit_cfg.get("decoder", {}) or {}
        
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.target_T = target_T
        self.patch_size = int(vit_cfg.get("patch_size", 16))
        
        hidden_size = int(dec_cfg.get("hidden_size", 192))
        num_layers = int(dec_cfg.get("num_hidden_layers", 4))
        num_heads = int(dec_cfg.get("num_attention_heads", 4))
        mlp_ratio = float(dec_cfg.get("intermediate_size", 768)) / hidden_size
        
        self.num_patches_per_channel = target_T // self.patch_size
        self.num_patches = out_channels * self.num_patches_per_channel
        self.hidden_size = hidden_size
        
        # Project encoder dim → decoder dim
        self.encoder_to_decoder = nn.Linear(embed_dim, hidden_size)
        
        # Learnable mask token (shared across all masked positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Positional embedding for full sequence
        max_patches = int(vit_cfg.get("max_patches_per_channel", 256))
        max_channels = int(vit_cfg.get("max_channels", 32))
        pos_embed_type = str(vit_cfg.get("pos_embed_type", "learned"))
        if pos_embed_type == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding2D(max_patches, max_channels, hidden_size)
        else:
            self.pos_embed = PositionalEmbedding2D(max_patches, max_channels, hidden_size)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(hidden_size)
        
        # Per-patch prediction: each token → patch_size raw values
        self.head = nn.Linear(hidden_size, self.patch_size)
    
    def forward(
        self,
        visible_tokens: torch.Tensor,
        mask: torch.Tensor,
        num_patches_per_channel: int,
        num_channels: int,
        padding_mask: Optional[torch.Tensor] = None,
        visible_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruct full signal from visible encoder tokens + mask.
        
        Args:
            visible_tokens: [B, N_vis, D_enc] encoder output for visible patches
            mask: [B, N] boolean, True = masked
            num_patches_per_channel: time patches per channel
            num_channels: number of input channels (C)
            
        Returns:
            [B, C, T] reconstructed signal
        """
        B = visible_tokens.shape[0]
        N = mask.shape[1]
        visible_dec = self.encoder_to_decoder(visible_tokens)  # [B, N_vis_max, D_dec]
        full_tokens = self.mask_token.to(dtype=visible_dec.dtype, device=visible_dec.device).expand(B, N, -1).clone()
        keep_mask = ~mask
        if padding_mask is not None:
            keep_mask = keep_mask & ~padding_mask

        for b in range(B):
            keep_positions = torch.nonzero(keep_mask[b], as_tuple=False).squeeze(-1)
            if visible_padding_mask is None:
                num_keep = keep_positions.numel()
            else:
                num_keep = int((~visible_padding_mask[b]).sum().item())
            if num_keep > 0:
                full_tokens[b, keep_positions] = visible_dec[b, :num_keep]
        
        # Add positional embedding
        pos_embed = self.pos_embed(num_patches_per_channel, num_channels, visible_tokens.device)
        pos_embed = pos_embed.to(dtype=full_tokens.dtype, device=full_tokens.device)
        full_tokens = full_tokens + pos_embed.unsqueeze(0)
        
        # Transformer decoder
        decoder_padding_mask = None
        if padding_mask is not None and bool(padding_mask.any()):
            decoder_padding_mask = padding_mask
        full_tokens = self.decoder(full_tokens, src_key_padding_mask=decoder_padding_mask)
        full_tokens = self.norm(full_tokens)
        if padding_mask is not None:
            full_tokens = full_tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        # Predict patch values: [B, N, D_dec] → [B, N, patch_size]
        patch_values = self.head(full_tokens)
        
        # Reshape to [B, C, T]
        # Token ordering: [C0_P0, C0_P1, ..., C1_P0, C1_P1, ...]
        patch_values = patch_values.view(B, num_channels, num_patches_per_channel, self.patch_size)
        out = patch_values.reshape(B, num_channels, num_patches_per_channel * self.patch_size)
        
        if self.target_T > 0 and out.shape[-1] != self.target_T:
            out = torch.nn.functional.interpolate(out, size=self.target_T, mode='linear', align_corners=False)
        
        return out


class ViT1DPoseDecoder(nn.Module):
    """ViT-1D Pose Decoder: encoder token grid → [B, P_t, num_joints * 6].

    Predicts one 6D rotation vector per time patch for each joint in a
    kinematic chain (e.g. the right arm chain with 5 joints → 30 dims).

    Architecture
    ------------
    1. Receive full encoder tokens  [B, C * P_t, D]
    2. Reshape to                   [B, C, P_t, D]
    3. Mean-pool over channels  →   [B, P_t, D]
    4. Lightweight transformer over the P_t time-patch sequence
    5. Per-token linear head     →  [B, P_t, num_joints * 6]

    Token layout assumption (from PatchEmbed1D + ViT1DEncoder):
        index = channel_idx * P_t + time_patch_idx

    This decoder is **throwaway** after pretraining; only the encoder is
    kept for downstream probing.

    Args:
        embed_dim:   Encoder hidden size (D).
        num_patches_per_channel: P_t (= T // patch_size).
        in_channels: C (number of IMU sensor channels).
        num_joints:  Number of joints in the target kinematic chain (default 5).
        cfg:         Config dict (reads vit1d.pose_decoder.* keys).
    """

    def __init__(
        self,
        embed_dim: int,
        num_patches_per_channel: int,
        in_channels: int,
        num_joints: int = 5,
        cfg: Any = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches_per_channel = num_patches_per_channel
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.out_dim = num_joints * 6  # 6D rotation per joint

        # Read decoder hyper-params from cfg (vit1d.pose_decoder.*)
        vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {} if cfg is not None else {}
        dec_cfg = vit_cfg.get("pose_decoder", {}) or {}

        hidden_size   = int(dec_cfg.get("hidden_size",         192))
        num_layers    = int(dec_cfg.get("num_hidden_layers",     2))
        num_heads     = int(dec_cfg.get("num_attention_heads",   4))
        mlp_ratio     = float(dec_cfg.get("mlp_ratio",          4.0))
        dropout       = float(dec_cfg.get("dropout",            0.0))

        self.hidden_size = hidden_size

        # Project encoder dim → decoder dim
        self.encoder_to_decoder = nn.Linear(embed_dim, hidden_size)

        # Learnable positional embeddings over the P_t time patches
        self.time_pos_embed = nn.Embedding(num_patches_per_channel, hidden_size)

        # Lightweight transformer over the P_t time positions
        dec_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

        # Per-patch linear prediction head: hidden → num_joints * 6
        self.head = nn.Linear(hidden_size, self.out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.time_pos_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode encoder token grid to per-patch pose predictions.

        Args:
            tokens: [B, C * P_t, D]  full encoder output from
                    ``ViT1DEncoder.forward_tokens(x)``

        Returns:
            [B, P_t, num_joints * 6]  — one 6D-rotation vector per time patch
                                        per joint (flattened), e.g. [B, 8, 30]
        """
        B, N, D = tokens.shape
        C  = self.in_channels
        Pt = self.num_patches_per_channel
        assert N == C * Pt, (
            f"ViT1DPoseDecoder: expected {C * Pt} tokens (C={C} x Pt={Pt}), "
            f"got {N}"
        )

        # Reshape: [B, C * Pt, D] → [B, C, Pt, D]
        x = tokens.view(B, C, Pt, D)

        # Mean-pool across channels → [B, Pt, D]
        x = x.mean(dim=1)

        # Project to decoder dim
        x = self.encoder_to_decoder(x)  # [B, Pt, hidden_size]

        # Add time positional embedding
        time_idx = torch.arange(Pt, device=tokens.device)
        x = x + self.time_pos_embed(time_idx).unsqueeze(0)  # [B, Pt, H]

        # Transformer over P_t positions
        x = self.transformer(x)      # [B, Pt, H]
        x = self.norm(x)

        # Per-patch prediction → [B, Pt, num_joints * 6]
        out = self.head(x)
        return out


class ViT1DOdomDecoder(nn.Module):
    """ViT-1D Odometry Decoder: encoder token grid → [B, P_t, 3].

    Predicts the root (pelvis) XYZ displacement over each time-patch interval.
    Architecture is identical to ViT1DPoseDecoder but the output head produces
    3-dimensional translations instead of joint rotations.

    Architecture
    ------------
    1. Receive full encoder tokens  [B, C * P_t, D]
    2. Reshape to                   [B, C, P_t, D]
    3. Mean-pool over channels  →   [B, P_t, D]
    4. Lightweight transformer over the P_t time-patch sequence
    5. Per-token linear head     →  [B, P_t, 3]

    Config keys read from ``cfg["vit1d"]["odom_decoder"]``::

        hidden_size           int    192
        num_hidden_layers     int    2
        num_attention_heads   int    4
        mlp_ratio             float  4.0
        dropout               float  0.0
    """

    def __init__(
        self,
        embed_dim: int,
        num_patches_per_channel: int,
        in_channels: int,
        cfg: Any,
    ):
        super().__init__()

        vit_cfg  = cfg_get(cfg, ["vit1d"], {}) or {}
        dec_cfg  = vit_cfg.get("odom_decoder", {}) or {}

        self.num_patches_per_channel = num_patches_per_channel
        self.in_channels             = in_channels
        self.embed_dim               = embed_dim

        hidden_size = int(dec_cfg.get("hidden_size",           192))
        num_layers  = int(dec_cfg.get("num_hidden_layers",       2))
        num_heads   = int(dec_cfg.get("num_attention_heads",     4))
        mlp_ratio   = float(dec_cfg.get("mlp_ratio",           4.0))
        dropout     = float(dec_cfg.get("dropout",             0.0))

        # Project encoder dim → lightweight decoder dim
        self.encoder_to_decoder = nn.Linear(embed_dim, hidden_size)

        # Time positional embedding (one entry per time patch)
        self.time_pos_embed = nn.Embedding(num_patches_per_channel, hidden_size)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

        # Per-patch head → XYZ displacement
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict per-patch root displacement.

        Args:
            tokens: [B, C * P_t, D]  — full encoder token grid.

        Returns:
            [B, P_t, 3]  — XYZ displacement per time patch.
        """
        B  = tokens.shape[0]
        C  = self.in_channels
        Pt = self.num_patches_per_channel

        # Reshape → [B, C, Pt, D]
        x = tokens.view(B, C, Pt, self.embed_dim)

        # Mean-pool channels → [B, Pt, D]
        x = x.mean(dim=1)

        # Project to hidden size
        x = self.encoder_to_decoder(x)  # [B, Pt, hidden_size]

        # Add time positional embedding
        time_idx = torch.arange(Pt, device=tokens.device)
        x = x + self.time_pos_embed(time_idx).unsqueeze(0)  # [B, Pt, H]

        # Lightweight transformer over P_t positions
        x = self.transformer(x)  # [B, Pt, H]
        x = self.norm(x)

        # Per-patch displacement prediction → [B, Pt, 3]
        return self.head(x)
