"""Discrete IMU tokenizer with frequency-progressive Residual Vector Quantization (RVQ).

Architecture
------------
  Encoder (ViT1D, no CLS token)
    → [B, N, D] patch embeddings
    → projection: Linear(D, code_dim)  [if D != code_dim]

  ResidualVectorQuantizer (K levels, each 512 codes × code_dim):
    Level 1 residual → q1 → target: lowpass 0–2 Hz cumulative prediction
    Level 2 residual → q2 → target: lowpass 0–5 Hz cumulative prediction
    Level 3 residual → q3 → target: lowpass 0–12 Hz cumulative prediction
    Level 4 residual → q4 → target: full signal

  PatchDecoder[k]: code_dim → hidden → patch_size  (one MLP per RVQ level)

Loss = sum_k( lambda_k * MSE(pred_k, target_k) )
     + commitment_weight * sum_k( ||z - q_k.detach()||^2 )
     + lambda_spec * SpectralLoss (final level only)

Notes
-----
  - Codebook updated via EMA (not gradients).
  - Straight-through estimator for encoder gradients through VQ.
  - Targets are the *cumulative* decoded signal so each level sees
    an increasingly high-frequency residual.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Butterworth low-pass filter applied on CPU via scipy
# ---------------------------------------------------------------------------

def butter_lowpass_filter(
    signal: np.ndarray,   # [T] or [B, C, T]
    cutoff_hz: float,
    fs: float = 50.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter (scipy.signal.filtfilt).

    cutoff_hz >= fs/2 returns the signal unchanged (already band-limited).
    """
    if cutoff_hz >= fs / 2.0:
        return signal.copy()

    from scipy.signal import butter, filtfilt  # lazy import; not available on all nodes

    nyq = fs / 2.0
    Wn  = cutoff_hz / nyq
    b, a = butter(order, Wn, btype="low")

    if signal.ndim == 1:
        return filtfilt(b, a, signal).astype(signal.dtype)

    # Apply along last axis (time) for [B, C, T]
    out = np.empty_like(signal)
    for bi in range(signal.shape[0]):
        for ci in range(signal.shape[1]):
            out[bi, ci] = filtfilt(b, a, signal[bi, ci]).astype(signal.dtype)
    return out


def get_frequency_targets(
    x: torch.Tensor,           # [B, C, T]  float32 on any device
    cutoff_hz_list: List[float],
    fs: float = 50.0,
) -> List[torch.Tensor]:
    """Return list of frequency-band targets [B, C, T] for each cutoff.

    The last cutoff is treated as "full signal" — no filtering applied.
    Filtering is done on CPU via scipy; returned tensors are on same device as x.
    """
    device = x.device
    x_np   = x.detach().cpu().float().numpy()   # [B, C, T]
    targets = []
    for hz in cutoff_hz_list:
        if hz >= fs / 2.0:
            targets.append(x)
        else:
            filtered = butter_lowpass_filter(x_np, hz, fs=fs)
            targets.append(torch.from_numpy(filtered).to(device=device, dtype=x.dtype))
    return targets


# ---------------------------------------------------------------------------
# EMA-updated Vector Quantizer (single codebook level)
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """Straight-through EMA vector quantizer.

    Codebook updated via exponential moving average (Oord et al. 2017 Appendix A).
    Encoder gradients pass through the quantization via straight-through estimator.
    """

    def __init__(self, num_codes: int, code_dim: int, ema_decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_codes   = num_codes
        self.code_dim    = code_dim
        self.ema_decay   = ema_decay
        self.eps         = eps

        # Codebook
        embed = torch.randn(num_codes, code_dim)
        nn.init.uniform_(embed, -1.0 / num_codes, 1.0 / num_codes)
        self.register_buffer("codebook",       embed.clone())
        self.register_buffer("ema_cluster_n",  torch.ones(num_codes))
        self.register_buffer("ema_embed_sum",  embed.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize z.

        Args:
            z: [*, code_dim]  continuous encoder output

        Returns:
            q       : [*, code_dim]  quantized (straight-through on z)
            indices : [*]  int64 nearest-code indices
        """
        flat = z.reshape(-1, self.code_dim)   # [M, D]

        # Pairwise squared distance: ||z - e||^2 = ||z||^2 - 2*z@e^T + ||e||^2
        dist = (
            (flat ** 2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.codebook.T
            + (self.codebook ** 2).sum(dim=1)
        )
        indices_flat = dist.argmin(dim=1)           # [M]
        q_flat = self.codebook[indices_flat]         # [M, D]

        if self.training:
            self._ema_update(flat.detach(), indices_flat)

        # Straight-through
        q_flat = z.reshape(-1, self.code_dim) + (q_flat - z.reshape(-1, self.code_dim)).detach()
        q = q_flat.reshape(z.shape)
        indices = indices_flat.reshape(z.shape[:-1])
        return q, indices

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        one_hot = F.one_hot(indices, self.num_codes).float()  # [M, K]
        self.ema_cluster_n.mul_(self.ema_decay).add_(one_hot.sum(0), alpha=1.0 - self.ema_decay)
        self.ema_embed_sum.mul_(self.ema_decay).add_(one_hot.T @ flat, alpha=1.0 - self.ema_decay)
        n = (self.ema_cluster_n + self.eps).unsqueeze(1)
        self.codebook.copy_(self.ema_embed_sum / n)


# ---------------------------------------------------------------------------
# Residual Vector Quantizer
# ---------------------------------------------------------------------------

class ResidualVectorQuantizer(nn.Module):
    """K-level RVQ: each level quantizes residual of the previous cumulative sum."""

    def __init__(
        self,
        num_levels:      int   = 4,
        num_codes:       int   = 512,
        code_dim:        int   = 256,
        ema_decay:       float = 0.99,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.num_levels        = num_levels
        self.commitment_weight = commitment_weight
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_codes, code_dim, ema_decay) for _ in range(num_levels)
        ])

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Multi-level quantization.

        Args:
            z: [B, N, code_dim]  projected encoder output

        Returns:
            quantized_list  : list[K] of [B, N, code_dim] cumulative quantized vectors
            indices_list    : list[K] of [B, N] int64 code indices per level
            commitment_loss : scalar commitment loss (sum over levels)
        """
        residual     = z
        cumulative_q = torch.zeros_like(z)
        quantized_list   : List[torch.Tensor] = []
        indices_list     : List[torch.Tensor] = []
        commitment_loss  = z.new_zeros(())

        for vq in self.quantizers:
            q, idx = vq(residual)
            cumulative_q = cumulative_q + q
            quantized_list.append(cumulative_q.clone())
            indices_list.append(idx)

            # Commitment loss: push encoder output toward codebook entry
            commitment_loss = commitment_loss + F.mse_loss(residual, q.detach())
            residual = residual - q.detach()

        return quantized_list, indices_list, self.commitment_weight * commitment_loss


# ---------------------------------------------------------------------------
# Per-level patch decoder MLP
# ---------------------------------------------------------------------------

class PatchDecoder(nn.Module):
    """code_dim → hidden → out_dim  (one instance per RVQ level).

    out_dim = patch_size           when tokens are per (channel, patch)
    out_dim = num_channels * patch_size  when tokens are pooled across channels
    """

    def __init__(self, code_dim: int, out_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(code_dim),
            nn.Linear(code_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q: [B, N, code_dim] → pred: [B, N, out_dim]."""
        return self.net(q)


# ---------------------------------------------------------------------------
# Spectral loss (STFT on reconstructed vs target signal)
# ---------------------------------------------------------------------------

def spectral_loss(
    pred:   torch.Tensor,   # [B, C, T]
    target: torch.Tensor,   # [B, C, T]
    n_fft:  int = 64,
    hop_length: int = 16,
) -> torch.Tensor:
    """Log-magnitude STFT MSE loss."""
    B, C, T = pred.shape
    p = pred.reshape(B * C, T)
    t = target.reshape(B * C, T)

    window = torch.hann_window(n_fft, device=pred.device)
    P = torch.stft(p, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                   window=window, return_complex=True)
    T_ = torch.stft(t, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                    window=window, return_complex=True)

    log_P = torch.log1p(P.abs())
    log_T = torch.log1p(T_.abs())
    return F.mse_loss(log_P, log_T)


# ---------------------------------------------------------------------------
# Helpers: reshape decoder predictions back to [B, C, T]
# ---------------------------------------------------------------------------

def _patches_to_signal(
    pred_patches: torch.Tensor,   # [B, C*Pt, patch_size]  per-(channel,patch) tokens
    B: int, C: int, Pt: int, patch_size: int,
) -> torch.Tensor:
    """Per-(channel, patch) tokens → [B, C, T].  Token layout: channel-major."""
    return pred_patches.view(B, C, Pt, patch_size).reshape(B, C, Pt * patch_size)


def _temporal_preds_to_signal(
    pred_flat: torch.Tensor,   # [B, Pt, C*patch_size]  per-temporal-position tokens
    B: int, C: int, Pt: int, patch_size: int,
) -> torch.Tensor:
    """Channel-pooled temporal tokens → [B, C, T]."""
    return pred_flat.view(B, Pt, C, patch_size).permute(0, 2, 1, 3).reshape(B, C, Pt * patch_size)


# ---------------------------------------------------------------------------
# Forward loss
# ---------------------------------------------------------------------------

def forward_loss(
    batch:        object,
    encoder:      nn.Module,
    projection:   Optional[nn.Module],
    rvq:          ResidualVectorQuantizer,
    decoders:     nn.ModuleList,
    cfg:          dict,
    cutoff_hz_list: List[float],
    lambda_levels:  List[float],
    lambda_spec:    float = 0.1,
    fs:             float = 50.0,
    pool_channels:  bool  = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """RVQ tokenizer forward pass.

    Args:
        batch          : (x [B,C,T], *any_labels) — single view, no masking
        encoder        : ViT1D encoder (use_cls_token=False)
        projection     : optional Linear(D, code_dim) projection layer or None
        rvq            : ResidualVectorQuantizer
        decoders       : nn.ModuleList of K PatchDecoder instances
        cfg            : merged experiment config dict
        cutoff_hz_list : K cutoff frequencies in Hz (last should be >= fs/2)
        lambda_levels  : K per-level loss weights
        lambda_spec    : spectral loss weight on final level
        fs             : signal sampling rate
        pool_channels  : if True (default), mean-pool the C channel tokens for each
                         temporal position before quantization, producing Pt tokens
                         per window (e.g. 8 × 4 levels = 32 indices for a 128-sample
                         window with patch_size=16).  If False, each (channel, patch)
                         pair is a separate token: C*Pt tokens per window.
    """
    x = batch[0] if isinstance(batch, (list, tuple)) else batch  # [B, C, T]
    B, C, T = x.shape

    # ── Encoder → patch tokens [B, C*Pt, D] ─────────────────────────────────
    tokens, Pt = encoder.forward_tokens(x)   # [B, C*Pt, D]
    D_enc = tokens.shape[-1]

    if pool_channels:
        # Mean-pool across channels for each temporal patch → [B, Pt, D]
        # Token layout from forward_tokens is channel-major: [C0P0, C0P1, ..., C1P0, ...]
        tokens_q = tokens.view(B, C, Pt, D_enc).mean(dim=1)   # [B, Pt, D]
    else:
        tokens_q = tokens   # [B, C*Pt, D]

    N_q = tokens_q.shape[1]   # Pt if pool_channels else C*Pt

    if projection is not None:
        z = projection(tokens_q)   # [B, N_q, code_dim]
    else:
        z = tokens_q

    # Normalise before quantization to prevent encoder scale drift.
    # Without this, ||z − q||² grows unboundedly as the encoder trains, causing
    # commitment_loss to dominate and the codebook to never fully converge.
    z = F.normalize(z, dim=-1)

    # ── RVQ quantization ─────────────────────────────────────────────────────
    quantized_list, indices_list, commitment_loss = rvq(z)
    # quantized_list[k]: [B, N_q, code_dim]  cumulative up to level k

    # ── Frequency-band targets ───────────────────────────────────────────────
    freq_targets = get_frequency_targets(x, cutoff_hz_list, fs=fs)
    # freq_targets[k]: [B, C, T]

    # ── Per-level reconstruction loss ────────────────────────────────────────
    patch_size = encoder.patch_size
    recon_loss_total = x.new_zeros(())
    level_logs: Dict[str, float] = {}

    pred_final_signal = None

    for k, (q_k, dec_k, target_k, lam_k) in enumerate(
        zip(quantized_list, decoders, freq_targets, lambda_levels)
    ):
        pred_flat = dec_k(q_k)   # [B, N_q, out_dim]

        if pool_channels:
            # [B, Pt, C*patch_size] → [B, C, T]
            pred_signal_k = _temporal_preds_to_signal(pred_flat, B, C, Pt, patch_size)
        else:
            # [B, C*Pt, patch_size] → [B, C, T]
            pred_signal_k = _patches_to_signal(pred_flat, B, C, Pt, patch_size)

        t_len = min(pred_signal_k.shape[-1], target_k.shape[-1])
        recon_k = F.mse_loss(pred_signal_k[..., :t_len], target_k[..., :t_len])
        recon_loss_total = recon_loss_total + lam_k * recon_k
        level_logs[f"recon_loss_l{k+1}"] = float(recon_k.detach())

        if k == len(decoders) - 1:
            pred_final_signal = pred_signal_k

    # ── Spectral loss on final level ─────────────────────────────────────────
    spec_loss = x.new_zeros(())
    if lambda_spec > 0.0 and pred_final_signal is not None:
        t_len = min(pred_final_signal.shape[-1], x.shape[-1])
        spec_loss = spectral_loss(pred_final_signal[..., :t_len], x[..., :t_len])

    total = recon_loss_total + commitment_loss + lambda_spec * spec_loss

    # ── Code utilization: fraction of codebook slots used in this batch ──────
    with torch.no_grad():
        util_list = []
        for k, (idx_k, vq_k) in enumerate(zip(indices_list, rvq.quantizers)):
            unique_k = float(idx_k.unique().numel()) / vq_k.num_codes
            util_list.append(unique_k)
            level_logs[f"code_util_l{k+1}"] = unique_k

    logs: Dict[str, float] = {
        "loss":            float(total.detach()),
        "recon_loss":      float(recon_loss_total.detach()),
        "commitment_loss": float(commitment_loss.detach()),
        "spec_loss":       float(spec_loss.detach()),
        "avg_code_util":   float(sum(util_list) / max(len(util_list), 1)),
        "n_token_indices": N_q * rvq.num_levels,   # 8*4=32 when pool_channels=True
    }
    logs.update(level_logs)

    return total, logs
