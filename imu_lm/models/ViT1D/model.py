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
import torch.nn.functional as F

from imu_lm.utils.helpers import cfg_get


class PatchEmbed1D(nn.Module):
    """1D Patch Embedding: [B, C, T] → [B, N, D] where N = C * (T // patch_size).

    shared_kernel modes
    -------------------
    True        : one Conv1d(1→D) shared across all channels (LSM-2 style)
    "acc_gyro"  : two Conv1ds — proj_acc for ch0-2 (accelerometer),
                                proj_gyro for ch3-5 (gyroscope)
    False       : one Conv1d(C→D) — all channels projected jointly
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        shared_kernel=True,   # bool or "acc_gyro"
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.in_channels = in_channels
        self.embed_dim   = embed_dim
        self.shared_kernel = shared_kernel

        if shared_kernel == "acc_gyro":
            # Separate kernels for accelerometer (ch0-2) and gyroscope (ch3-5)
            self.proj_acc  = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.proj_gyro = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif shared_kernel:
            # Single Conv1d applied to each channel independently
            self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            # All channels projected jointly
            self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] → [B, C*num_patches, D]."""
        B, C, T = x.shape

        if self.shared_kernel == "acc_gyro":
            # ch0-2 → proj_acc, ch3-5 → proj_gyro
            n_acc = C // 2   # 3 acc channels, 3 gyro channels
            patches = []
            for c in range(C):
                proj = self.proj_acc if c < n_acc else self.proj_gyro
                patches.append(proj(x[:, c:c+1, :]).transpose(1, 2))  # [B, Pt, D]
            return torch.cat(patches, dim=1)   # [B, C*Pt, D]
        elif self.shared_kernel:
            patches = []
            for c in range(C):
                patches.append(self.proj(x[:, c:c+1, :]).transpose(1, 2))  # [B, Pt, D]
            return torch.cat(patches, dim=1)   # [B, C*Pt, D]
        else:
            return self.proj(x).transpose(1, 2)   # [B, Pt, D]


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


class DropPath(nn.Module):
    """Stochastic depth: drop the entire residual branch with probability p."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # BUG FIX (2026-08-15): this used to be
        #   torch.rand(shape).floor_() + keep_prob
        # i.e. floor(U[0,1)) == 0 always, so mask == keep_prob for every
        # sample and x * keep_prob / keep_prob == x -- DropPath was a silent
        # no-op for every drop_path_rate > 0 ever configured in this repo
        # (18 configs, all standard-ViT1D BEiT/DINO/HAR-finetune runs; the
        # alt-attn encoder never wired drop_path_rate at all until today).
        # Correct timm/MAE form: floor(U[0,1) + keep_prob) is 1 w.p. keep_prob.
        mask = (torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob).floor_()
        return x * mask / keep_prob


class SoftMoEFFN(nn.Module):
    """Soft Mixture-of-Experts FFN: replaces the standard FFN in a Transformer layer.

    K expert FFNs are each applied to every token. A learned router maps each
    token's embedding to K soft weights (via softmax), and the output is the
    weighted sum of all expert outputs — no hard assignment, fully differentiable.

    The K experts conceptually correspond to different frequency regimes:
      Expert 0 → low-freq (0–1 Hz, gravity / DC)
      Expert 1 → cadence  (1–3 Hz, step fundamentals)
      Expert 2 → mid-freq (3–8 Hz, walking harmonics)
      Expert 3 → high-freq (8–25 Hz, fine dynamics)

    The router learns from the token embedding which expert(s) to emphasise for
    each patch, allowing the model to specialise FFN capacity by frequency content.

    Args:
        d_model        : token embedding dimension
        dim_feedforward: FFN hidden width (each expert)
        num_experts    : number of expert FFNs K (default 4, one per freq band)
        dropout        : dropout inside each expert FFN
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        num_experts: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

        # Learned router: token embedding → K unnormalised logits
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.trunc_normal_(self.router.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            [B, N, D]  soft-weighted sum of K expert outputs
        """
        routing_weights = F.softmax(self.router(x), dim=-1)   # [B, N, K]
        # Stack expert outputs along a new dim → [B, N, D, K]
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=-1)
        # Weighted sum over experts
        return (expert_out * routing_weights.unsqueeze(-2)).sum(dim=-1)  # [B, N, D]


class TransformerLayerWithDropPathMoE(nn.Module):
    """Pre-norm transformer layer where the FFN is replaced by a SoftMoEFFN.

    Identical to TransformerLayerWithDropPath in every other respect (pre-norm,
    stochastic depth, MultiheadAttention).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        num_experts: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = SoftMoEFFN(d_model, dim_feedforward,
                                num_experts=num_experts, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        freq_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                key_padding_mask=src_key_padding_mask)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class TransformerLayerWithDropPath(nn.Module):
    """Pre-norm transformer layer (self-attention + FFN) with stochastic depth.

    Replaces nn.TransformerEncoderLayer so that each residual branch can be
    independently dropped at the whole-sample level during training.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        freq_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                key_padding_mask=src_key_padding_mask)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class TransformerWithDropPath(nn.Module):
    """Thin stack wrapper so TransformerLayerWithDropPath layers compose like
    nn.TransformerEncoder (same call signature)."""

    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        freq_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask,
                      freq_tokens=freq_tokens)
        return x


# ── Frequency-Aware Soft MoE ──────────────────────────────────────────────────

FREQ_BANDS_DEFAULT = [(0.0, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 25.0)]


def _bandpass_1d(x: torch.Tensor, lo: float, hi: float, fs: float = 50.0) -> torch.Tensor:
    """Band-pass filter [B, C, T] via FFT zeroing."""
    T     = x.shape[-1]
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs, device=x.device)
    X_f   = torch.fft.rfft(x.float(), dim=-1)
    keep  = (freqs >= lo) & (freqs < hi)
    X_out = torch.zeros_like(X_f)
    X_out[..., keep] = X_f[..., keep]
    return torch.fft.irfft(X_out, n=T, dim=-1).to(x.dtype)


class FreqBandPatchProjection(nn.Module):
    """Projects band-filtered signal patches to token embeddings for MoE experts.

    For each of K frequency bands:
      1. Bandpass-filter the raw signal [B, C, T] to the band using FFT zeroing
      2. Apply a shared-kernel Conv1d patch projection per channel
      3. Produce band-specific tokens [B, C*Pt, D]

    Output: [B, K, C*Pt, D] — K independent band-specific token sets.

    Each expert in FreqAwareSoftMoEFFN receives only the tokens for its own band,
    so expert k sees exclusively the 0–1 Hz / 1–3 Hz / 3–8 Hz / 8–25 Hz content.
    """

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        in_channels: int,
        bands: list = None,
        fs: float = 50.0,
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.embed_dim   = embed_dim
        self.in_channels = in_channels
        self.fs          = fs
        self.bands       = bands or FREQ_BANDS_DEFAULT

        # One independent Conv1d per frequency band (shared_kernel=True style)
        self.projections = nn.ModuleList([
            nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
            for _ in range(len(self.bands))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            [B, K, C*Pt, D]  K band-specific token sets
        """
        B, C, T = x.shape
        band_list = []
        for k, (lo, hi) in enumerate(self.bands):
            x_band = _bandpass_1d(x, lo, hi, self.fs)   # [B, C, T]
            ch_patches = []
            for c in range(C):
                p = self.projections[k](x_band[:, c:c+1, :]).transpose(1, 2)  # [B, Pt, D]
                ch_patches.append(p)
            band_list.append(torch.cat(ch_patches, dim=1))  # [B, C*Pt, D]
        return torch.stack(band_list, dim=1)                 # [B, K, C*Pt, D]


class FreqAwareSoftMoEFFN(nn.Module):
    """Soft MoE FFN where expert k receives band-k-filtered patch tokens as input.

    Router       : full-bandwidth token x → soft weights [B, N, K].
    Expert k     : processes freq_tokens[:, k] (band-k-filtered tokens).
    Shared expert: optional standard FFN that always runs on the full-bandwidth
                   token x and is added to the MoE output (DeepSeek-MoE style).
                   Prevents frequency-partitioned representations by ensuring
                   every layer also captures holistic cross-band features.

    output = shared_expert(x)  +  Σ_k router_weight_k * expert_k(freq_tokens_k)

    If freq_tokens is None, all experts receive x (fallback for probing).
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        num_experts: int = 4,
        dropout: float = 0.0,
        use_shared_expert: bool = False,
    ):
        super().__init__()
        self.num_experts       = num_experts
        self.use_shared_expert = use_shared_expert

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.trunc_normal_(self.router.weight, std=0.02)

        if use_shared_expert:
            self.shared_expert = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        x: torch.Tensor,
        freq_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x           : [B, N, D]      full-bandwidth tokens (routing + shared expert)
            freq_tokens : [B, K, N, D]   band-filtered tokens (freq expert inputs)
        Returns:
            [B, N, D]
        """
        routing_weights = F.softmax(self.router(x), dim=-1)  # [B, N, K]

        if freq_tokens is not None:
            expert_out = torch.stack(
                [self.experts[k](freq_tokens[:, k]) for k in range(self.num_experts)],
                dim=-1,
            )  # [B, N, D, K]
        else:
            expert_out = torch.stack(
                [self.experts[k](x) for k in range(self.num_experts)],
                dim=-1,
            )  # [B, N, D, K]

        moe_out = (expert_out * routing_weights.unsqueeze(-2)).sum(dim=-1)  # [B, N, D]

        if self.use_shared_expert:
            return self.shared_expert(x) + moe_out
        return moe_out


class TransformerLayerWithDropPathFreqMoE(nn.Module):
    """Pre-norm transformer layer with a FreqAwareSoftMoEFFN replacing the FFN.

    Self-attention operates on the full-bandwidth tokens x.
    The FFN receives band-filtered tokens via freq_tokens.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        num_experts: int = 4,
        use_shared_expert: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = FreqAwareSoftMoEFFN(
            d_model, dim_feedforward, num_experts=num_experts,
            dropout=dropout, use_shared_expert=use_shared_expert,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        freq_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                key_padding_mask=src_key_padding_mask)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.ff(self.norm2(x), freq_tokens=freq_tokens))
        return x


class TransformerWithDropPathFreqMoE(nn.Module):
    """Stack wrapper for TransformerLayerWithDropPathFreqMoE."""

    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        freq_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask,
                      freq_tokens=freq_tokens)
        return x


# ── 3D Frequency-Token Architecture (v6) ─────────────────────────────────────
#
# Instead of frequency bands being separate expert inputs (MoE), here each
# frequency band becomes an explicit token in the sequence. A token is indexed
# by a 3-tuple (channel c, time-patch t, frequency band f):
#
#     N = C * Pt * F   tokens   (6 channels × 8 time patches × 4 bands = 192)
#
# Band 0 is the original (full-bandwidth) signal; bands 1..F-1 are FFT-filtered.
# Self-attention now operates across channel, time AND frequency axes jointly.
#
# Positional encoding is the sum of three learned embeddings:
#     pos(c, t, f) = pos_channel(c) + pos_time(t) + pos_freq(f)

FREQ_BANDS_3D_DEFAULT = [None, (0.0, 3.0), (3.0, 8.0), (8.0, 25.0)]  # None = original


def freq_decompose_3d(x: torch.Tensor, bands: list, fs: float = 50.0) -> torch.Tensor:
    """Decompose signal into frequency bands.

    Args:
        x     : [B, C, T]
        bands : list; ``None`` entry = original signal, ``(lo, hi)`` = bandpass
    Returns:
        [B, F, C, T]  stacked frequency views (F = len(bands))
    """
    out = []
    for band in bands:
        if band is None:
            out.append(x)                                   # original / full-bandwidth
        else:
            out.append(_bandpass_1d(x, band[0], band[1], fs))
    return torch.stack(out, dim=1)                          # [B, F, C, T]


class PositionalEmbedding3D(nn.Module):
    """3D positional embedding: pos(c, t, f) = ch_embed(c) + time_embed(t) + freq_embed(f).

    Token layout is freq-major: index n = f*(C*Pt) + c*Pt + t.
    """

    def __init__(self, max_time: int, max_channels: int, max_freq: int, embed_dim: int):
        super().__init__()
        self.time_embed    = nn.Embedding(max_time,     embed_dim)
        self.channel_embed = nn.Embedding(max_channels, embed_dim)
        self.freq_embed    = nn.Embedding(max_freq,     embed_dim)

    def forward(self, num_freq: int, num_channels: int, num_time: int,
                device: torch.device) -> torch.Tensor:
        """Return [F*C*Pt, D] positional embeddings in freq-major order."""
        CT = num_channels * num_time
        f_idx = torch.arange(num_freq,     device=device).repeat_interleave(CT)           # [N]
        c_idx = (torch.arange(num_channels, device=device)
                 .repeat_interleave(num_time).repeat(num_freq))                            # [N]
        t_idx = torch.arange(num_time,     device=device).repeat(num_freq * num_channels) # [N]
        return self.freq_embed(f_idx) + self.channel_embed(c_idx) + self.time_embed(t_idx)


class PatchEmbed3D(nn.Module):
    """3D patch embedding: [B, F, C, T] → [B, F*C*Pt, D].

    One Conv1d per (band, sensor-type) pair — acc/gyro split within each band,
    so band-f accelerometer and band-f gyroscope use different kernels.
    """

    def __init__(self, patch_size: int, embed_dim: int, in_channels: int, num_freq: int):
        super().__init__()
        self.patch_size  = patch_size
        self.embed_dim   = embed_dim
        self.in_channels = in_channels
        self.num_freq    = num_freq
        self.n_acc       = in_channels // 2

        self.proj_acc  = nn.ModuleList([
            nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
            for _ in range(num_freq)
        ])
        self.proj_gyro = nn.ModuleList([
            nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
            for _ in range(num_freq)
        ])

    def forward(self, x_bands: torch.Tensor) -> torch.Tensor:
        """x_bands [B, F, C, T] → [B, F*C*Pt, D] (freq-major token order)."""
        B, F, C, T = x_bands.shape
        out = []
        for f in range(F):
            for c in range(C):
                proj = self.proj_acc[f] if c < self.n_acc else self.proj_gyro[f]
                p = proj(x_bands[:, f, c:c+1, :]).transpose(1, 2)   # [B, Pt, D]
                out.append(p)
        return torch.cat(out, dim=1)                               # [B, F*C*Pt, D]


class ViT1DEncoder3D(nn.Module):
    """ViT-1D encoder with explicit frequency-band tokens + 3D positional encoding.

    Pipeline:
        raw [B, C, T]
          → freq_decompose → [B, F, C, T]
          → PatchEmbed3D    → [B, F*C*Pt, D]
          → + 3D pos embed
          → (optional MAE token-drop masking)
          → Transformer encoder
        Probing: mean-pool all F*C*Pt tokens → [B, D].
    """

    def __init__(self, cfg: Any):
        super().__init__()
        vit_cfg  = cfg_get(cfg, ["vit1d"], {}) or {}
        enc_cfg  = vit_cfg.get("encoder", {}) or {}
        data_cfg = cfg_get(cfg, ["data"], {}) or {}

        sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
        self.in_channels = len(sensor_cols)
        self.patch_size  = int(vit_cfg.get("patch_size", 16))

        self.embed_dim = int(enc_cfg.get("hidden_size", 384))
        num_layers     = int(enc_cfg.get("num_hidden_layers", 12))
        num_heads      = int(enc_cfg.get("num_attention_heads", 6))
        mlp_ratio      = float(enc_cfg.get("intermediate_size", 1536)) / self.embed_dim
        dropout        = float(enc_cfg.get("hidden_dropout_prob", 0.0))
        drop_path_rate = float(enc_cfg.get("drop_path_rate", 0.0))

        # Frequency bands (band 0 = original signal)
        raw_bands = enc_cfg.get("freq_bands_3d", None)
        if raw_bands is None:
            self.bands = FREQ_BANDS_3D_DEFAULT
        else:
            self.bands = [None if (b is None or str(b).lower() in ("none", "original"))
                          else (float(b[0]), float(b[1])) for b in raw_bands]
        self.num_freq_bands = len(self.bands)
        self.fs = float(enc_cfg.get("freq_fs_3d", 50.0))

        max_time     = int(vit_cfg.get("max_patches_per_channel", 256))
        max_channels = int(vit_cfg.get("max_channels", 32))
        max_freq     = int(vit_cfg.get("max_freq_bands", 8))

        self.pooling = vit_cfg.get("pooling", "mean")

        self.patch_embed = PatchEmbed3D(
            patch_size=self.patch_size, embed_dim=self.embed_dim,
            in_channels=self.in_channels, num_freq=self.num_freq_bands,
        )
        self.pos_embed = PositionalEmbedding3D(
            max_time=max_time, max_channels=max_channels,
            max_freq=max_freq, embed_dim=self.embed_dim,
        )

        dpr = [drop_path_rate * i / max(1, num_layers - 1) for i in range(num_layers)]
        self.encoder = TransformerWithDropPath([
            TransformerLayerWithDropPath(
                d_model=self.embed_dim, nhead=num_heads,
                dim_feedforward=int(self.embed_dim * mlp_ratio),
                dropout=dropout, drop_path=dpr[i],
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)

        # Flags read by the loss / run.py
        self._use_bert_mask = False
        self._use_freq_moe  = False
        self._is_3d_freq    = True

    def decompose(self, x: torch.Tensor) -> torch.Tensor:
        """raw [B, C, T] → frequency views [B, F, C, T]."""
        return freq_decompose_3d(x, self.bands, self.fs)

    def _embed(self, x: torch.Tensor):
        """raw [B, C, T] → tokens [B, N, D] with 3D pos embedding."""
        B, C, T = x.shape
        Pt = T // self.patch_size
        x_bands = self.decompose(x)                          # [B, F, C, T]
        tokens  = self.patch_embed(x_bands)                  # [B, F*C*Pt, D]
        pos     = self.pos_embed(self.num_freq_bands, C, Pt, x.device)
        return tokens + pos.unsqueeze(0), Pt

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                padding_mask: torch.Tensor = None,
                return_keep_padding_mask: bool = False):
        """If mask is None → mean-pooled [B, D]. Else → visible tokens [B, N_keep, D]."""
        B, C, T = x.shape
        tokens, Pt = self._embed(x)                          # [B, N, D]

        keep_padding_mask = None
        if mask is not None:
            ids_shuffle = torch.argsort(mask.int(), dim=1)
            keep_counts = (~mask).sum(dim=1)
            max_keep    = int(keep_counts.max().item())
            ids_keep    = ids_shuffle[:, :max_keep]
            tokens = torch.gather(
                tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            )
            keep_padding_mask = (
                torch.arange(max_keep, device=x.device).unsqueeze(0).expand(B, -1)
                >= keep_counts.unsqueeze(1)
            )

        src_kpm = keep_padding_mask if (keep_padding_mask is not None
                                        and bool(keep_padding_mask.any())) else None
        tokens = self.encoder(tokens, src_key_padding_mask=src_kpm)
        tokens = self.norm(tokens)
        if keep_padding_mask is not None:
            tokens = tokens.masked_fill(keep_padding_mask.unsqueeze(-1), 0.0)

        if mask is not None:
            if return_keep_padding_mask:
                return tokens, keep_padding_mask
            return tokens

        return tokens.mean(dim=1)                             # [B, D] probing

    def forward_tokens(self, x: torch.Tensor):
        """Full token grid (no masking). Returns (tokens [B, N, D], Pt)."""
        tokens, Pt = self._embed(x)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens, Pt


class ViT1DDecoder3D(nn.Module):
    """MAE decoder that reconstructs all F frequency bands: tokens → [B, F, C, T]."""

    def __init__(self, embed_dim: int, out_channels: int, target_T: int,
                 num_freq: int, cfg: Any):
        super().__init__()
        vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
        dec_cfg = vit_cfg.get("decoder", {}) or {}

        self.embed_dim    = embed_dim
        self.out_channels = out_channels
        self.target_T     = target_T
        self.num_freq     = num_freq
        self.patch_size   = int(vit_cfg.get("patch_size", 16))
        self.Pt           = target_T // self.patch_size

        hidden_size = int(dec_cfg.get("hidden_size", 192))
        num_layers  = int(dec_cfg.get("num_hidden_layers", 4))
        num_heads   = int(dec_cfg.get("num_attention_heads", 4))
        mlp_ratio   = float(dec_cfg.get("intermediate_size", 768)) / hidden_size
        self.hidden_size = hidden_size

        self.encoder_to_decoder = nn.Linear(embed_dim, hidden_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        max_time     = int(vit_cfg.get("max_patches_per_channel", 256))
        max_channels = int(vit_cfg.get("max_channels", 32))
        max_freq     = int(vit_cfg.get("max_freq_bands", 8))
        self.pos_embed = PositionalEmbedding3D(max_time, max_channels, max_freq, hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, self.patch_size)

    def forward(self, visible_tokens: torch.Tensor, mask: torch.Tensor,
                num_channels: int, visible_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            visible_tokens : [B, N_vis, D_enc]
            mask           : [B, N]  True = masked   (N = F*C*Pt)
        Returns:
            [B, F, C, T]  reconstructed per-band signal
        """
        B = visible_tokens.shape[0]
        N = mask.shape[1]
        F, C, Pt = self.num_freq, num_channels, self.Pt

        visible_dec = self.encoder_to_decoder(visible_tokens)            # [B, N_vis, D_dec]
        full = self.mask_token.to(visible_dec.dtype).expand(B, N, -1).clone()

        keep_mask = ~mask
        for b in range(B):
            keep_pos = torch.nonzero(keep_mask[b], as_tuple=False).squeeze(-1)
            n_keep   = (keep_pos.numel() if visible_padding_mask is None
                        else int((~visible_padding_mask[b]).sum().item()))
            if n_keep > 0:
                full[b, keep_pos] = visible_dec[b, :n_keep]

        pos  = self.pos_embed(F, C, Pt, visible_tokens.device).to(full.dtype)
        full = full + pos.unsqueeze(0)
        full = self.decoder(full)
        full = self.norm(full)

        patch_vals = self.head(full)                                    # [B, N, patch_size]
        # freq-major layout: [B, F, C, Pt, patch_size] → [B, F, C, T]
        out = patch_vals.view(B, F, C, Pt, self.patch_size).reshape(B, F, C, Pt * self.patch_size)
        if self.target_T > 0 and out.shape[-1] != self.target_T:
            out = torch.nn.functional.interpolate(
                out.reshape(B * F, C, -1), size=self.target_T,
                mode="linear", align_corners=False,
            ).reshape(B, F, C, self.target_T)
        return out


# ── Alternating frame/global attention encoder (VGGT-style) ───────────────────

class _PreNormBlock(nn.Module):
    """Standard pre-norm transformer block: x + Attn(LN x), x + FFN(LN x).

    drop_path (2026-08-15): optional stochastic depth on both residual
    branches (no parameters -> state_dicts are unchanged; default 0.0 keeps
    every existing checkpoint's behaviour bit-identical)."""

    def __init__(self, d_model, nhead, dim_ff, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):                 # [*, N, D]
        n = self.norm1(x)
        x = x + self.drop_path(self.attn(n, n, n)[0])
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class ViT1DEncoderAltAttn(nn.Module):
    """ViT-1D encoder with alternating frame-wise / global attention (VGGT-style).

    A "frame" (0.32 s section) is one time-patch position across all C channels:
    its C tokens attend to each other in *frame-wise* attention. *Global*
    attention then lets every token (all sections × channels) plus a learnable
    global CLS token attend to one another. Layers alternate frame → global.
    The CLS token participates only in global attention (excluded from frame
    attention). Uses BERT-style masking so the section grid stays intact.

    Token grid: [B, Pt, C, D]  (Pt = T // patch_size sections, C channels).
    Variable length is supported natively (Pt varies per batch).
    """

    def __init__(self, cfg: Any):
        super().__init__()
        vit_cfg  = cfg_get(cfg, ["vit1d"], {}) or {}
        enc_cfg  = vit_cfg.get("encoder", {}) or {}
        data_cfg = cfg_get(cfg, ["data"], {}) or {}

        sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
        self.in_channels = len(sensor_cols)
        self.patch_size  = int(vit_cfg.get("patch_size", 16))
        _sk = vit_cfg.get("shared_kernel", True)
        shared_kernel = _sk if isinstance(_sk, str) else bool(_sk)

        self.embed_dim = int(enc_cfg.get("hidden_size", 384))
        num_layers     = int(enc_cfg.get("num_hidden_layers", 12))
        num_heads      = int(enc_cfg.get("num_attention_heads", 6))
        dim_ff         = int(enc_cfg.get("intermediate_size", 1536))
        dropout        = float(enc_cfg.get("hidden_dropout_prob", 0.0))
        max_patches    = int(vit_cfg.get("max_patches_per_channel", 256))
        max_channels   = int(vit_cfg.get("max_channels", 32))

        self.patch_embed = PatchEmbed1D(self.in_channels, self.patch_size,
                                        self.embed_dim, shared_kernel=shared_kernel)
        self.pos_embed   = PositionalEmbedding2D(max_patches, max_channels, self.embed_dim)

        # Global CLS token (global attention only) + BERT mask token
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Optional per-section register token (VGGT-style): one learnable register
        # per 0.32 s section, participates in BOTH frame and global attention and is
        # pretrained self-supervised (it aggregates section info that helps the MAE
        # decoder reconstruct masked patches via global attention). Downstream pose
        # heads can read off these pretrained section features.
        self._use_register = bool(enc_cfg.get("use_register", False))
        if self._use_register:
            self.register_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.register_pos   = nn.Embedding(max_patches, self.embed_dim)  # per-section id
            self.register_norm  = nn.LayerNorm(self.embed_dim)
            nn.init.trunc_normal_(self.register_token, std=0.02)

        # Alternating blocks: frame, global, frame, global, ...
        # drop_path_rate (2026-08-15): stochastic depth, linearly increasing
        # from 0 at the first pair to drop_path_rate at the last pair (MAE
        # convention); default 0.0 = off (previously silently ignored here).
        n_pairs = num_layers // 2
        drop_path_rate = float(enc_cfg.get("drop_path_rate", 0.0))
        dpr = [drop_path_rate * i / max(1, n_pairs - 1) for i in range(n_pairs)]
        self.frame_blocks  = nn.ModuleList(
            [_PreNormBlock(self.embed_dim, num_heads, dim_ff, dropout, drop_path=dpr[i]) for i in range(n_pairs)])
        self.global_blocks = nn.ModuleList(
            [_PreNormBlock(self.embed_dim, num_heads, dim_ff, dropout, drop_path=dpr[i]) for i in range(n_pairs)])
        self.n_pairs = n_pairs

        self.norm     = nn.LayerNorm(self.embed_dim)
        self.cls_norm = nn.LayerNorm(self.embed_dim)

        # Flags read by the loss / run.py
        self._use_bert_mask = True
        self._use_freq_moe  = False

    def _backbone(self, x, mask=None):
        """raw [B,C,T] (+ optional BERT mask) → (ch_tokens [B,C*Pt,D], reg [B,Pt,D]|None, cls [B,D], Pt)."""
        B, C, T = x.shape
        Pt = T // self.patch_size
        D  = self.embed_dim

        tok = self.patch_embed(x)                              # [B, C*Pt, D] channel-major
        tok = tok + self.pos_embed(Pt, C, x.device).unsqueeze(0)

        if mask is not None:
            mfloat = mask.float().unsqueeze(-1)                # [B, C*Pt, 1]
            mtok   = self.mask_token.expand(B, C * Pt, -1)
            tok    = tok * (1.0 - mfloat) + mtok * mfloat

        # channel-major [B, C*Pt, D] → section grid [B, Pt, C, D]
        grid = tok.reshape(B, C, Pt, D).permute(0, 2, 1, 3).contiguous()
        cls  = self.cls_token.expand(B, -1, -1)                # [B, 1, D]

        reg = None
        if self._use_register:
            sec = torch.arange(Pt, device=x.device)
            reg = (self.register_token.expand(B, Pt, -1)
                   + self.register_pos(sec).unsqueeze(0)).unsqueeze(2)   # [B, Pt, 1, D]

        for i in range(self.n_pairs):
            # Frame-wise attention: C channel tokens (+ register) within each section
            if self._use_register:
                fin  = torch.cat([grid, reg], dim=2).reshape(B * Pt, C + 1, D)
                fout = self.frame_blocks[i](fin).reshape(B, Pt, C + 1, D)
                grid = fout[:, :, :C]
                reg  = fout[:, :, C:C + 1]
            else:
                gf   = grid.reshape(B * Pt, C, D)
                grid = self.frame_blocks[i](gf).reshape(B, Pt, C, D)
            # Global attention: CLS + all Pt*C channel tokens (+ Pt register tokens)
            flat = grid.reshape(B, Pt * C, D)
            if self._use_register:
                flat_reg = reg.reshape(B, Pt, D)
                g    = self.global_blocks[i](torch.cat([cls, flat, flat_reg], dim=1))
                cls  = g[:, :1]
                grid = g[:, 1:1 + Pt * C].reshape(B, Pt, C, D)
                reg  = g[:, 1 + Pt * C:].reshape(B, Pt, 1, D)
            else:
                g    = self.global_blocks[i](torch.cat([cls, flat], dim=1))
                cls  = g[:, :1]
                grid = g[:, 1:].reshape(B, Pt, C, D)

        # back to channel-major [B, C*Pt, D]
        tok = grid.permute(0, 2, 1, 3).reshape(B, C * Pt, D)
        reg_out = self.register_norm(reg.reshape(B, Pt, D)) if self._use_register else None
        return self.norm(tok), reg_out, self.cls_norm(cls.squeeze(1)), Pt

    def forward(self, x, mask=None, padding_mask=None, return_keep_padding_mask=False):
        tokens, _, cls, _ = self._backbone(x, mask=mask)
        if mask is not None:
            return (tokens, None) if return_keep_padding_mask else tokens
        return cls                                            # [B, D] pooled (CLS) for probing

    def forward_tokens(self, x):
        tokens, _, _, Pt = self._backbone(x, mask=None)
        return tokens, Pt

    def forward_registers(self, x):
        """Per-section pretrained register features [B, Pt, D] (requires use_register)."""
        assert self._use_register, "encoder was not built with use_register: true"
        _, reg, _, Pt = self._backbone(x, mask=None)
        return reg, Pt


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
        _sk = vit_cfg.get("shared_kernel", True)
        shared_kernel = _sk if isinstance(_sk, str) else bool(_sk)
        
        # Encoder config (nested under vit1d.encoder)
        self.embed_dim = int(enc_cfg.get("hidden_size", 384))
        num_layers = int(enc_cfg.get("num_hidden_layers", 12))
        num_heads = int(enc_cfg.get("num_attention_heads", 6))
        mlp_ratio = float(enc_cfg.get("intermediate_size", 1536)) / self.embed_dim
        dropout = float(enc_cfg.get("hidden_dropout_prob", 0.0))
        attn_dropout = float(enc_cfg.get("attention_probs_dropout_prob", 0.0))
        layer_norm_eps = float(enc_cfg.get("layer_norm_eps", 1e-6))
        drop_path_rate = float(enc_cfg.get("drop_path_rate", 0.0))
        
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

        # Linearly spaced drop path rates: 0 at layer 0 → drop_path_rate at last layer
        dpr = [drop_path_rate * i / max(1, num_layers - 1) for i in range(num_layers)]
        num_moe_experts  = int(enc_cfg.get("num_moe_experts",  0))
        freq_moe_experts = int(enc_cfg.get("freq_moe_experts", 0))

        if freq_moe_experts > 0:
            # Frequency-aware Soft MoE: each expert receives band-filtered tokens
            freq_bands = [(b[0], b[1]) for b in enc_cfg.get(
                "freq_moe_bands", FREQ_BANDS_DEFAULT
            )]
            fs = float(enc_cfg.get("freq_moe_fs", 50.0))
            use_shared_expert = bool(enc_cfg.get("freq_moe_shared_expert", False))
            self._use_freq_moe = True
            self.freq_patch_proj = FreqBandPatchProjection(
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                in_channels=self.in_channels,
                bands=freq_bands,
                fs=fs,
            )
            self.encoder = TransformerWithDropPathFreqMoE([
                TransformerLayerWithDropPathFreqMoE(
                    d_model=self.embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(self.embed_dim * mlp_ratio),
                    dropout=dropout,
                    drop_path=dpr[i],
                    num_experts=freq_moe_experts,
                    use_shared_expert=use_shared_expert,
                )
                for i in range(num_layers)
            ])
        elif num_moe_experts > 0:
            # Standard Soft MoE: all experts receive full-bandwidth tokens
            self._use_freq_moe = False
            self.encoder = TransformerWithDropPath([
                TransformerLayerWithDropPathMoE(
                    d_model=self.embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(self.embed_dim * mlp_ratio),
                    dropout=dropout,
                    drop_path=dpr[i],
                    num_experts=num_moe_experts,
                )
                for i in range(num_layers)
            ])
        else:
            self._use_freq_moe = False
            self.encoder = TransformerWithDropPath([
                TransformerLayerWithDropPath(
                    d_model=self.embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(self.embed_dim * mlp_ratio),
                    dropout=dropout,
                    drop_path=dpr[i],
                )
                for i in range(num_layers)
            ])
        
        self.norm = nn.LayerNorm(self.embed_dim)

        # BERT-style masking: learnable mask tokens injected at encoder input
        # instead of dropping masked patches (fixes train/test mismatch and
        # forces encoder to build holistic representations over all positions).
        self._use_bert_mask = bool(enc_cfg.get("bert_masking", False))
        if self._use_bert_mask:
            # One global mask token for the standard (full-bandwidth) token stream
            self.global_mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.trunc_normal_(self.global_mask_token, std=0.02)
            if self._use_freq_moe:
                # Per-band mask tokens: each expert sees a band-specific "unknown" signal
                n_bands = freq_moe_experts  # same count as experts
                self.freq_mask_tokens = nn.Parameter(
                    torch.zeros(n_bands, 1, self.embed_dim)
                )
                nn.init.trunc_normal_(self.freq_mask_tokens, std=0.02)

        # Optional CLS token for DINO-style objectives
        self.use_cls_token = bool(vit_cfg.get("use_cls_token", False))
        if self.use_cls_token:
            self.cls_token  = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.trunc_normal_(self.cls_token,  std=0.02)
            nn.init.trunc_normal_(self.mask_token, std=0.02)

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
        ids_keep = None
        if mask is not None or padding_mask is not None:
            # ── BERT-style masking (inject mask tokens, keep all N positions) ──────
            if self._use_bert_mask:
                bert_mask = torch.zeros(B, tokens.shape[1], dtype=torch.bool, device=x.device)
                if mask is not None:
                    bert_mask = bert_mask | mask
                if padding_mask is not None:
                    bert_mask = bert_mask | padding_mask
                mask_f = bert_mask.float().unsqueeze(-1)   # [B, N, 1]

                # Replace masked standard tokens with global mask token
                gmt = self.global_mask_token.expand(B, tokens.shape[1], -1)
                tokens = tokens * (1.0 - mask_f) + gmt * mask_f

                # Compute freq tokens with per-band mask tokens at masked positions
                freq_tokens = None
                if self._use_freq_moe:
                    raw_freq = self.freq_patch_proj(x)  # [B, K, N, D]
                    raw_freq = raw_freq + pos_embed.unsqueeze(0).unsqueeze(0)
                    for k in range(raw_freq.shape[1]):
                        bmt = self.freq_mask_tokens[k:k+1].expand(B, raw_freq.shape[2], -1)
                        raw_freq[:, k] = raw_freq[:, k] * (1.0 - mask_f) + bmt * mask_f
                    freq_tokens = raw_freq

                tokens = self.encoder(tokens, freq_tokens=freq_tokens)
                tokens = self.norm(tokens)
                if return_keep_padding_mask:
                    return tokens, None   # [B, N, D], no padding needed
                return tokens             # [B, N, D]
            # ── Standard MAE masking (drop masked tokens) ────────────────────────

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

        # Compute frequency-band tokens for FreqMoE (apply same masking as standard tokens)
        freq_tokens = None
        if self._use_freq_moe:
            raw_freq = self.freq_patch_proj(x)  # [B, K, N, D]
            raw_freq = raw_freq + pos_embed.unsqueeze(0).unsqueeze(0)  # add pos embed to each band
            if ids_keep is not None:
                gather_idx = ids_keep.unsqueeze(1).unsqueeze(-1).expand(
                    -1, raw_freq.shape[1], -1, self.embed_dim
                )  # [B, K, max_keep, D]
                freq_tokens = torch.gather(raw_freq, 2, gather_idx)   # [B, K, max_keep, D]
            else:
                freq_tokens = raw_freq

        # Transformer encoder
        src_key_padding_mask = None
        if keep_padding_mask is not None and bool(keep_padding_mask.any()):
            src_key_padding_mask = keep_padding_mask
        tokens = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask,
                              freq_tokens=freq_tokens)
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

        freq_tokens = None
        if self._use_freq_moe:
            freq_tokens = self.freq_patch_proj(x)  # [B, K, N, D]
            freq_tokens = freq_tokens + pos_embed.unsqueeze(0).unsqueeze(0)

        # DINO-pretrained encoders always process tokens with CLS prepended;
        # prepend here so frozen weights see the same input distribution.
        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)      # [B, 1, D]
            tokens = torch.cat([cls, tokens], dim=1)    # [B, 1+N, D]

        tokens = self.encoder(tokens, freq_tokens=freq_tokens)
        tokens = self.norm(tokens)

        if self.use_cls_token:
            tokens = tokens[:, 1:]   # strip CLS; return patch tokens [B, N, D]

        return tokens, num_patches_per_channel

    def forward_dino(
        self,
        x: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DINO/iBOT forward: CLS prepend + optional iBOT-style patch masking.

        Args:
            x          : [B, C, T] raw input signal
            patch_mask : [B, C*Pt] bool, True = masked.  When provided the masked
                         patch embeddings are replaced with ``self.mask_token``
                         (BERT-style) before the transformer — the CLS token still
                         attends to all positions, including masked ones.

        Returns:
            cls_out    : [B, D]        CLS token output
            patch_out  : [B, C*Pt, D]  per-patch token outputs (all positions)

        Requires ``use_cls_token=True`` in the config.
        """
        assert self.use_cls_token, "forward_dino requires vit1d.use_cls_token: true"
        B, C, T = x.shape
        Pt = T // self.patch_size

        tokens = self.patch_embed(x)   # [B, C*Pt, D]

        if patch_mask is not None:
            # Replace masked patch embeddings with the learnable mask token
            mask_exp = patch_mask.unsqueeze(-1).float()                  # [B, N, 1]
            mask_tok = self.mask_token.expand(B, tokens.shape[1], -1)    # [B, N, D]
            tokens   = tokens * (1.0 - mask_exp) + mask_tok * mask_exp  # [B, N, D]

        pos_embed = self.pos_embed(Pt, C, x.device)
        tokens    = tokens + pos_embed.unsqueeze(0)

        # Prepend CLS — no positional embedding on the CLS token (DINOv2 convention)
        cls    = self.cls_token.expand(B, -1, -1)       # [B, 1, D]
        tokens = torch.cat([cls, tokens], dim=1)         # [B, 1+C*Pt, D]

        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        return tokens[:, 0], tokens[:, 1:]               # cls [B,D], patches [B,C*Pt,D]


class ViT1DDINOHead(nn.Module):
    """3-layer MLP projection head for DINO/iBOT.

    Architecture (Caron et al. 2021 / Oquab et al. 2023):
        Linear(D → H) → GELU → Linear(H → H) → GELU → Linear(H → B)
        L2-norm → weight-normalised linear(B → K, no bias)

    The last linear layer uses weight normalisation and a fixed norm (weight_g=1),
    so only the direction of the prototype vectors is learned.  This prevents
    collapse to a single prototype while keeping the vocabulary spread out.

    Args:
        in_dim        : encoder hidden size D
        out_dim       : prototype vocabulary size K (default 4096)
        hidden_dim    : MLP intermediate width H (default 2048)
        bottleneck_dim: bottleneck width B before L2-norm (default 256)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad_(False)  # freeze norm; train direction only

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., D] → [..., K] (any leading batch dims)."""
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)     # L2-normalise before weight-norm projection
        return self.last_layer(x)


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
    """ViT-1D Pose Decoder: encoder token grid → 6D joint rotations.

    Two-stage architecture:

    Stage 1 — 2D cross-channel transformer (all C * P_t tokens):
        1. Project encoder tokens [B, C*P_t, D] → [B, C*P_t, H]
        2. Add 2D positional embedding (time + channel)
        3. Transformer: all C*P_t tokens attend across channels AND time
        4. Mean-pool over channels → [B, P_t, H]

    Stage 2 — 1D temporal transformer → 6D joint rotations:
        5. Add 1D time positional embedding
        6. Transformer over P_t time tokens
        7. Linear head → [B, P_t, K*J*6]  6D rotations

    Config keys (vit1d.pose_decoder.*):
        hidden_size           int    192
        num_hidden_layers     int    4     2D cross-channel transformer layers
        num_temporal_layers   int    2     1D temporal transformer layers
        num_attention_heads   int    4
        mlp_ratio             float  4.0
        dropout               float  0.0   used inside transformer layers AND at:
                                             • after encoder→decoder projection
                                             • after channel mean-pool (stage 1 → 2)
                                             • before the final rotation head
    """

    def __init__(
        self,
        embed_dim: int,
        num_patches_per_channel: int,
        in_channels: int,
        num_joints: int = 5,
        cfg: Any = None,
        sub_patches: int = 1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches_per_channel = num_patches_per_channel
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.sub_patches = sub_patches
        self.out_dim = sub_patches * num_joints * 6

        vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {} if cfg is not None else {}
        dec_cfg = vit_cfg.get("pose_decoder", {}) or {}

        hidden_size    = int(dec_cfg.get("hidden_size",         192))
        num_layers_2d  = int(dec_cfg.get("num_hidden_layers",     4))
        num_layers_1d  = int(dec_cfg.get("num_temporal_layers",   2))
        num_heads      = int(dec_cfg.get("num_attention_heads",   4))
        mlp_ratio      = float(dec_cfg.get("mlp_ratio",          4.0))
        dropout        = float(dec_cfg.get("dropout",            0.0))
        drop_path_rate = float(dec_cfg.get("drop_path_rate",     0.0))

        self.hidden_size = hidden_size

        max_patches  = int(vit_cfg.get("max_patches_per_channel", 256)) if cfg is not None else 256
        max_channels = int(vit_cfg.get("max_channels", 32)) if cfg is not None else 32

        def _make_layers(n):
            """Build n layers with linearly increasing drop path rates."""
            dpr = [drop_path_rate * i / max(1, n - 1) for i in range(n)]
            return TransformerWithDropPath([
                TransformerLayerWithDropPath(
                    d_model=hidden_size, nhead=num_heads,
                    dim_feedforward=int(hidden_size * mlp_ratio),
                    dropout=dropout, drop_path=dpr[i],
                )
                for i in range(n)
            ])

        # ── Stage 1: 2D cross-channel transformer ────────────────────────
        self.encoder_to_decoder = nn.Linear(embed_dim, hidden_size)
        self.proj_drop  = nn.Dropout(dropout)          # after input projection
        self.pos_embed_2d = PositionalEmbedding2D(max_patches, max_channels, hidden_size)
        self.transformer_2d = _make_layers(num_layers_2d)
        self.norm_2d = nn.LayerNorm(hidden_size)

        # ── Stage 2: 1D temporal → 6D joint rotations ────────────────────
        self.pool_drop  = nn.Dropout(dropout)          # after channel mean-pool
        self.time_pos_embed = nn.Embedding(max_patches, hidden_size)
        self.transformer_1d = _make_layers(num_layers_1d)
        self.norm_1d = nn.LayerNorm(hidden_size)
        self.head_drop  = nn.Dropout(dropout)          # before rotation head
        self.head = nn.Linear(hidden_size, self.out_dim)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode encoder tokens to per-patch 6D joint rotations.

        Args:
            tokens: [B, C * P_t, D]  full encoder output

        Returns:
            [B, P_t, K * num_joints * 6]  6D joint rotations
        """
        B, N, D = tokens.shape
        C  = self.in_channels
        Pt = self.num_patches_per_channel
        assert N == C * Pt, (
            f"ViT1DPoseDecoder: expected {C*Pt} tokens (C={C} x Pt={Pt}), got {N}"
        )
        time_idx = torch.arange(Pt, device=tokens.device)

        # ── Stage 1: 2D cross-channel transformer ────────────────────────
        x = self.proj_drop(self.encoder_to_decoder(tokens))         # [B, C*Pt, H]
        x = x + self.pos_embed_2d(Pt, C, tokens.device).to(dtype=x.dtype).unsqueeze(0)
        x = self.transformer_2d(x)
        x = self.norm_2d(x)
        x = self.pool_drop(x.view(B, C, Pt, self.hidden_size).mean(dim=1))  # [B, Pt, H]

        # ── Stage 2: 1D temporal → 6D joint rotations ────────────────────
        x = x + self.time_pos_embed(time_idx).unsqueeze(0)          # [B, Pt, H]
        x = self.transformer_1d(x)
        x = self.norm_1d(x)
        return self.head(self.head_drop(x))                          # [B, Pt, K*J*6]


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
        out_dim: int = 3,
    ):
        super().__init__()

        vit_cfg  = cfg_get(cfg, ["vit1d"], {}) or {}
        dec_cfg  = vit_cfg.get("odom_decoder", {}) or {}

        self.num_patches_per_channel = num_patches_per_channel
        self.in_channels             = in_channels
        self.embed_dim               = embed_dim
        self.out_dim                 = out_dim

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

        # Per-patch head → out_dim values (3 for XYZ displacement, 4 for [dx,dy,dz,dψ])
        self.head = nn.Linear(hidden_size, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict per-patch root displacement.

        Args:
            tokens: [B, C * P_t, D]  — full encoder token grid.

        Returns:
            [B, P_t, out_dim]  — per-patch prediction.
              out_dim=3: XYZ displacement
              out_dim=4: [dx, dy, dz, dψ] local displacement + delta yaw
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
        if self.transformer.layers:
            x = self.transformer(x)  # [B, Pt, H]
        x = self.norm(x)

        # Per-patch displacement prediction → [B, Pt, 3]
        return self.head(x)


# ---------------------------------------------------------------------------
# JEPA Predictor
# ---------------------------------------------------------------------------

class JEPA1DPredictor(nn.Module):
    """Narrow transformer predictor for I-JEPA style SSL on 1D IMU signals.

    Maps context encoder tokens (at unmasked positions) to predicted target
    encoder embeddings (at masked positions), working entirely in latent space.

    Architecture (Assran et al. 2023, adapted):
    1. Project context encoder output D_enc → D_pred
    2. Scatter context projections into full-length sequence;
       fill target positions with a learnable mask token
    3. Add the same 2-D positional embeddings as the encoder
    4. Run narrow transformer over all N positions
    5. LayerNorm → project D_pred → D_enc
    6. Return full [B, N, D_enc] (loss computed only at target positions)

    Args:
        embed_dim : encoder hidden size D_enc
        cfg       : experiment config dict (reads ``vit1d.jepa_predictor.*``)
    """

    def __init__(self, embed_dim: int, cfg: Any):
        super().__init__()
        vit_cfg  = cfg_get(cfg, ["vit1d"], {}) or {}
        pred_cfg = vit_cfg.get("jepa_predictor", {}) or {}

        hidden_size = int(pred_cfg.get("hidden_size", 192))
        num_layers  = int(pred_cfg.get("num_hidden_layers", 4))
        num_heads   = int(pred_cfg.get("num_attention_heads", 4))
        mlp_ratio   = float(pred_cfg.get("mlp_ratio", 4.0))
        max_patches = int(vit_cfg.get("max_patches_per_channel", 256))
        max_channels = int(vit_cfg.get("max_channels", 32))

        self.embed_dim   = embed_dim
        self.hidden_size = hidden_size

        # Project encoder → predictor dim
        self.enc_to_pred = nn.Linear(embed_dim, hidden_size)

        # Learnable query token for target positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embedding shared with encoder convention (2D: time + channel)
        pos_embed_type = str(vit_cfg.get("pos_embed_type", "learned"))
        if pos_embed_type == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding2D(max_patches, max_channels, hidden_size)
        else:
            self.pos_embed = PositionalEmbedding2D(max_patches, max_channels, hidden_size)

        pred_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(pred_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

        # Project predictor → encoder dim (target space)
        self.pred_to_enc = nn.Linear(hidden_size, embed_dim)

    def forward(
        self,
        ctx_tokens: torch.Tensor,
        patch_mask: torch.Tensor,
        num_patches_per_channel: int,
        num_channels: int,
        ctx_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict target encoder embeddings.

        Args:
            ctx_tokens       : [B, N_ctx_max, D_enc]  context encoder output
            patch_mask       : [B, N]  True = target (masked from context encoder)
            num_patches_per_channel : T // patch_size
            num_channels     : C
            ctx_padding_mask : [B, N_ctx_max]  True = padding slot

        Returns:
            [B, N, D_enc]  predictions at all positions; only target positions
            are meaningful for the loss.
        """
        B = ctx_tokens.shape[0]
        N = patch_mask.shape[1]

        # Project context tokens to predictor dim
        ctx_proj = self.enc_to_pred(ctx_tokens)  # [B, N_ctx_max, D_pred]

        # Reconstruct which full-sequence positions correspond to context tokens.
        # Encoder sorts tokens by patch_mask.int() ascending (False < True),
        # so context tokens come first in ascending index order.
        keep_counts  = (~patch_mask).sum(dim=1)          # [B]
        max_keep     = int(keep_counts.max().item())
        ids_shuffle  = torch.argsort(patch_mask.int(), dim=1)
        ids_keep     = ids_shuffle[:, :max_keep]         # [B, max_keep]

        # Build full-length predictor sequence: mask tokens everywhere
        full = self.mask_token.expand(B, N, -1).clone()  # [B, N, D_pred]

        # Scatter context projections back to their original positions
        for b in range(B):
            n = int(keep_counts[b].item())
            if ctx_padding_mask is not None:
                n = int((~ctx_padding_mask[b]).sum().item())
            if n > 0:
                positions = ids_keep[b, :n]              # [n] original indices
                full[b].scatter_(
                    0,
                    positions.unsqueeze(-1).expand(n, self.hidden_size),
                    ctx_proj[b, :n],
                )

        # Add positional embeddings (same 2-D embedding as encoder)
        pos_embed = self.pos_embed(num_patches_per_channel, num_channels, ctx_tokens.device)
        full = full + pos_embed.to(dtype=full.dtype).unsqueeze(0)

        # Transformer
        full = self.transformer(full)
        full = self.norm(full)

        # Project back to encoder embedding dimension
        return self.pred_to_enc(full)   # [B, N, D_enc]

class StatsConcatEncoder(nn.Module):
    """Information-preserving wrapper (diagnostic, 2026-08-18): the loader hands over the
    filtered-but-NOT-z-scored window; this module (i) records the per-window, per-channel
    mean and std of the physical signal (acc scaled by 1/g, gyro in rad/s), (ii) applies the
    usual per-window z-score itself so `base` sees exactly its standard input, and (iii)
    returns [base(x_zscored) ; mean ; std]  ->  embed_dim = base.embed_dim + 2*C.
    Restores the gravity-direction / motion-amplitude cues the z-score deletes without
    touching the pretrained encoder. Config: stats_concat: {enabled: true, eps: 1e-6,
    acc_scale: 9.81, n_acc: 3}; requires preprocess.normalize.enabled: false.
    """

    def __init__(self, base: nn.Module, n_channels: int, eps: float = 1e-6, acc_scale: float = 9.81, n_acc: int = 3,
                 features=None, acc_units: str = "auto"):
        super().__init__()
        self.base = base
        self.n_channels = int(n_channels)
        self.eps = float(eps)
        self.acc_scale = float(acc_scale)
        self.n_acc = int(n_acc)
        self.features = list(features or ["meanstd"])
        self.acc_units = str(acc_units).lower()          # "auto" | "g" | "m/s2"
        n_feat = sum((2 * self.n_channels) if f == "meanstd" else self.FEATURE_DIMS[f] for f in self.features)
        self.n_feat = int(n_feat)
        self.embed_dim = int(base.embed_dim) + self.n_feat
        self.in_channels = getattr(base, "in_channels", self.n_channels)

    # ---- physical feature sets -------------------------------------------------
    # "meanstd"  : per-channel mean and std                                   (2C)
    # "accphys"  : gravity direction (unit LPF-mean acc), |mean acc| in g,
    #              acc dynamic energy log(1+E_a)                              (5)
    # "gyrodyn"  : gyro energy log(1+E_w), mean |w|, per-axis std of w,
    #              dominant frequency and spectral entropy of |w|             (7)
    FEATURE_DIMS = {"meanstd": None, "accphys": 5, "gyrodyn": 7}

    def _unit_gain(self, x: torch.Tensor) -> float:
        """1.0 if acc is already m/s^2, 9.80665 if it is stored in g (|mean acc| ~ 1 vs ~9.8).

        Datasets in this project are mixed: Capture24/NHANES are g, the rest m/s^2. Per-window
        z-scoring hides it, fixed physical scaling does not -- see the 2026-08-20 unit audit.
        """
        if self.acc_units == "m/s2":
            return 1.0
        if self.acc_units == "g":
            return 9.80665
        with torch.no_grad():                                          # auto
            m = x[:, : self.n_acc].mean(-1).norm(dim=1).mean().item()
            sd = x[:, : self.n_acc].std().item()
        return 9.80665 if (m < 3.0 and sd < 1.5) else 1.0

    def _acc_phys(self, x: torch.Tensor) -> torch.Tensor:
        a = x[:, : self.n_acc] * self._unit_gain(x) / self.acc_scale   # [B,3,T] in g
        mu = a.mean(dim=-1)                                            # low-frequency component ~ gravity
        norm = mu.norm(dim=1, keepdim=True).clamp_min(self.eps)
        e = ((a - mu.unsqueeze(-1)) ** 2).mean(dim=(1, 2))              # dynamic energy
        return torch.cat([mu / norm, norm, torch.log1p(e).unsqueeze(1)], dim=1)   # [B,5]

    def _gyro_dyn(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] <= self.n_acc:
            return x.new_zeros(x.shape[0], 8)
        w = x[:, self.n_acc:]                                          # [B,3,T] rad/s
        mag = w.norm(dim=1)                                            # [B,T]
        e = (w ** 2).mean(dim=(1, 2))
        sd = w.std(dim=-1, unbiased=False)                             # [B,3]
        spec = torch.fft.rfft(mag - mag.mean(dim=-1, keepdim=True), dim=-1).abs() ** 2   # [B,F]
        spec = spec[:, 1:] + self.eps
        p = spec / spec.sum(dim=-1, keepdim=True)
        dom = p.argmax(dim=-1).float() / max(1, p.shape[-1])           # dominant frequency (normalised)
        ent = -(p * p.log()).sum(dim=-1) / math.log(p.shape[-1])       # spectral entropy in [0,1]
        return torch.cat([torch.log1p(e).unsqueeze(1), mag.mean(dim=-1, keepdim=True), sd,
                          dom.unsqueeze(1), ent.unsqueeze(1)], dim=1)  # [B,8]

    def stats(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for f in self.features:
            if f == "meanstd":
                mu = x.mean(dim=-1)                       # [B, C]
                sd = x.std(dim=-1, unbiased=False)        # population std, as in preprocess_window
                gain = self._unit_gain(x)
                scale = torch.ones(x.shape[1], device=x.device, dtype=x.dtype)
                scale[: min(self.n_acc, x.shape[1])] = self.acc_scale / gain
                feats.append(torch.cat([mu / scale, sd / scale], dim=1))
            elif f == "accphys":
                feats.append(self._acc_phys(x))
            elif f == "gyrodyn":
                feats.append(self._gyro_dyn(x))
            else:
                raise ValueError(f"unknown stats_concat feature {f!r}")
        return torch.cat(feats, dim=1)

    def zscore(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) / (sd + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.stats(x.float())
        z = self.base(self.zscore(x))
        return torch.cat([z, r.to(z.dtype)], dim=1)

    def forward_tokens(self, x: torch.Tensor):
        return self.base.forward_tokens(self.zscore(x))

    def load_pretrained_state(self, enc_state):
        from imu_lm.models.ViT1D.run import _load_encoder_state_dict   # lazy: run.py imports this module
        _load_encoder_state_dict(self.base, enc_state)


def maybe_stats_wrap(encoder: nn.Module, cfg) -> nn.Module:
    """Apply StatsConcatEncoder when cfg.stats_concat.enabled (see class docstring)."""
    from imu_lm.utils.helpers import cfg_get
    sc = cfg_get(cfg, ["stats_concat"], None) or {}
    if not sc.get("enabled", False) or isinstance(encoder, StatsConcatEncoder):
        return encoder
    if bool(cfg_get(cfg, ["preprocess", "normalize", "enabled"], True)):
        raise ValueError("stats_concat requires preprocess.normalize.enabled: false (the wrapper z-scores itself)")
    cols = cfg_get(cfg, ["data", "sensor_columns"], None) or ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    n_ch = 3 if bool(cfg_get(cfg, ["data", "acc_only"], False)) else len(cols)
    return StatsConcatEncoder(encoder, n_channels=n_ch, eps=float(sc.get("eps", 1e-6)),
                              acc_scale=float(sc.get("acc_scale", 9.81)), n_acc=int(sc.get("n_acc", 3)),
                              features=sc.get("features", ["meanstd"]),
                              acc_units=sc.get("acc_units", "auto"))

class DualStreamGatedEncoder(nn.Module):
    """Two pretrained stems + learned gated fusion (2026-08-18 ablation).

        z_a = f_acc(a)            acc-only MAE stem, sees the 3 accelerometer channels
        z_f = f_full(x)           6-channel MAE stem, sees acc+gyro
        alpha = sigmoid(W [z_a ; z_f])            (per-dimension gate, one scalar per unit)
        z = alpha * z_a' + (1 - alpha) * z_f      (z_a' = linear projection when dims differ)

    Tests "separate stems + adaptive fusion" WITHOUT re-pretraining: MHEALTH/HHAR can lean on the
    accelerometer/gravity stem while PAMAP2/UTD lean on the full acc+gyro stem. Both stems are loaded
    from existing runs and frozen for `probe`; the gate (and optional stats) stay trainable.
    Config:
      dual_stream: {enabled: true, acc_run: vit1d_ssl_mae_varlen_acconly_multidata,
                    full_run: vit1d_ssl_mae_varlen_altattn_multidata, freeze: true, concat: false}
    """

    def __init__(self, acc_stem: nn.Module, full_stem: nn.Module, n_acc: int = 3, concat: bool = False):
        super().__init__()
        self.acc_stem, self.full_stem, self.n_acc, self.concat = acc_stem, full_stem, int(n_acc), bool(concat)
        da, df = int(acc_stem.embed_dim), int(full_stem.embed_dim)
        self.proj = nn.Identity() if da == df else nn.Linear(da, df)
        self.embed_dim = (da + df) if concat else df
        self.gate = nn.Sequential(nn.Linear(da + df, df), nn.Sigmoid())
        self.in_channels = getattr(full_stem, "in_channels", 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        za = self.acc_stem(x[:, : self.n_acc])
        zf = self.full_stem(x)
        if self.concat:
            return torch.cat([za, zf], dim=1)
        a = self.gate(torch.cat([za, zf], dim=1))
        return a * self.proj(za) + (1.0 - a) * zf

    def load_pretrained_state(self, enc_state) -> None:
        """No-op: both stems are loaded from their own runs in build_dual_stream()."""
        return

    def gate_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Mean gate value per window: >0.5 = leaning on the accelerometer stem."""
        with torch.no_grad():
            za, zf = self.acc_stem(x[:, : self.n_acc]), self.full_stem(x)
            return self.gate(torch.cat([za, zf], dim=1)).mean(dim=1)


def build_dual_stream(cfg) -> nn.Module:
    """Build DualStreamGatedEncoder from cfg.dual_stream (loads both pretrained stems)."""
    import os
    import torch as _t
    from imu_lm.utils.helpers import cfg_get
    from copy import deepcopy
    ds = cfg_get(cfg, ["dual_stream"], None) or {}
    runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")

    def _stem(run, acc_only):
        ck = _t.load(os.path.join(runs_root, run, "checkpoints", "best.pt"), map_location="cpu", weights_only=False)
        c = deepcopy(ck["cfg"])
        enc = (ViT1DEncoderAltAttn if bool(cfg_get(c, ["vit1d", "encoder", "alt_attention"], False)) else ViT1DEncoder)(c)
        from imu_lm.models.ViT1D.run import _load_encoder_state_dict
        _load_encoder_state_dict(enc, ck)
        if ds.get("freeze", True):
            enc.requires_grad_(False)
            enc.eval()
        return enc

    acc = _stem(ds["acc_run"], True)
    full = _stem(ds["full_run"], False)
    return DualStreamGatedEncoder(acc, full, n_acc=int(ds.get("n_acc", 3)), concat=bool(ds.get("concat", False)))
