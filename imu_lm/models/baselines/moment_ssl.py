"""MOMENT reconstruction SSL wrapper for the unified baseline launcher.

This uses the vendored MOMENT implementation under ``imu_lm/models/moment`` and
trains it with MOMENT's masked reconstruction objective on TransfHAR windows.
The wrapper accepts [B, T, C] batches and forwards [B, C, T] to MOMENT.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_MOMENT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "moment",
)
if _MOMENT_ROOT not in sys.path:
    sys.path.insert(0, _MOMENT_ROOT)


def _default_t5_config(d_model: int, n_layers: int, n_heads: int, d_ff: int) -> dict:
    return {
        "vocab_size": 1,
        "d_model": d_model,
        "d_kv": d_model // n_heads,
        "d_ff": d_ff,
        "num_layers": n_layers,
        "num_decoder_layers": n_layers,
        "num_heads": n_heads,
        "relative_attention_num_buckets": 32,
        "dropout_rate": 0.1,
        "layer_norm_epsilon": 1.0e-6,
        "feed_forward_proj": "gated-gelu",
        "is_encoder_decoder": True,
        "use_cache": False,
    }


class MOMENTSSL(nn.Module):
    """MOMENT masked reconstruction objective with the baseline launcher API."""

    def __init__(self, cfg: dict):
        super().__init__()
        try:
            from momentfm import MOMENT
        except ImportError as exc:
            raise ImportError(
                "MOMENT dependencies are missing. Install the vendored package, e.g. "
                "`pip install -e imu_lm/models/moment`, before running --model moment."
            ) from exc

        d_model = int(cfg.get("d_model", 512))
        n_layers = int(cfg.get("n_layers", 12))
        n_heads = int(cfg.get("n_heads", 8))
        d_ff = int(cfg.get("d_ff", 1536))
        patch_len = int(cfg.get("patch_len", 8))
        seq_len = int(cfg.get("seq_len", 512))

        moment_cfg = {
            "task_name": "reconstruction",
            "seq_len": seq_len,
            "patch_len": patch_len,
            "patch_stride_len": int(cfg.get("patch_stride_len", patch_len)),
            "d_model": d_model,
            "transformer_backbone": str(cfg.get("transformer_backbone", "google/flan-t5-small")),
            "transformer_type": "encoder_only",
            "randomly_initialize_backbone": bool(cfg.get("randomly_initialize_backbone", True)),
            "freeze_embedder": bool(cfg.get("freeze_embedder", False)),
            "freeze_encoder": bool(cfg.get("freeze_encoder", False)),
            "freeze_head": bool(cfg.get("freeze_head", False)),
            "mask_ratio": float(cfg.get("mask_ratio", 0.3)),
            "patch_dropout": float(cfg.get("patch_dropout", 0.1)),
            "head_dropout": float(cfg.get("head_dropout", 0.1)),
            "add_positional_embedding": bool(cfg.get("add_positional_embedding", True)),
            "value_embedding_bias": bool(cfg.get("value_embedding_bias", False)),
            "revin_affine": bool(cfg.get("revin_affine", False)),
            "enable_gradient_checkpointing": bool(cfg.get("enable_gradient_checkpointing", True)),
            "t5_config": cfg.get("t5_config") or _default_t5_config(
                d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff),
        }
        self.model = MOMENT(moment_cfg)

    def forward_loss(self, batch, device) -> Tuple[torch.Tensor, Dict]:
        x = batch.to(device).float().transpose(1, 2)  # [B, T, C] -> [B, C, T]
        input_mask = torch.ones(x.shape[0], x.shape[-1], device=device, dtype=x.dtype)
        out = self.model(x_enc=x, input_mask=input_mask)
        recon = out.reconstruction
        mask = out.pretrain_mask.to(device).float()
        hidden = (1.0 - mask) * input_mask
        denom = hidden.sum().clamp_min(1.0) * x.shape[1]
        loss = (((recon - x) ** 2) * hidden.unsqueeze(1)).sum() / denom
        return loss, {"loss": float(loss.detach()),
                      "masked_mse": float(loss.detach()),
                      "mask_ratio": float(hidden.mean().detach())}

    @torch.no_grad()
    def encode(self, x) -> torch.Tensor:
        x = x.transpose(1, 2)
        input_mask = torch.ones(x.shape[0], x.shape[-1], device=x.device, dtype=x.dtype)
        out = self.model.embed(x_enc=x, input_mask=input_mask, reduction="mean")
        return out.embeddings
