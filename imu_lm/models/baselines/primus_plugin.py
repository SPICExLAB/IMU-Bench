"""PRIMUS (Das et al., ICASSP 2025, Nokia Bell Labs) released IMU encoder as an IMU-Bench plugin encoder.

Weights: Zenodo record 15147513, `best_model.ckpt` (CC BY 4.0), a Lightning checkpoint whose `imu_encoder.backbone.*`
keys are the IMU encoder (1.4 M parameters). The backbone below re-implements `MW2StackRNNPoolingMultihead.backbone`
from github.com/nokia-bell-labs/pretrained-imu-encoders (lib/imu_models.py, BSD-3-Clause-Clear) layer for layer, so
the released keys load strictly; it avoids the repository's pytorch_lightning dependency.

Pretraining: head-worn Aria IMU from Ego-Exo4D (6 channels, m/s^2 with gravity and rad/s), aligned with video and
narration text plus self-supervision; the only normalisation is the in-network GroupNorm.

The encoder is a WINDOW encoder: its GRU runs over the 32 convolution channels, not over time (the released code feeds
[B, 32 channels, 32 pooled steps] to a batch-first GRU), so it has no per-time-step output. As for HarNet, the token
sequence is built by encoding native-length crops slid across the window, one per token slot; windows shorter than a
crop are encoded whole. Pooled = the whole window encoded once.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, cin, cout, k, pool="max", emb=32):
        super().__init__()
        pool_fn = nn.MaxPool1d(kernel_size=3) if pool == "max" else nn.AdaptiveAvgPool1d(output_size=emb)
        self.net = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=k, dilation=2, bias=False), pool_fn)

    def forward(self, x):
        return self.net(x)


class PRIMUSEncoder(nn.Module):
    def __init__(self, size_embeddings: int = 512, input_dim: int = 32, crop: int = 250):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.GroupNorm(2, 6),
            _Block(6, input_dim, 10),
            _Block(input_dim, input_dim, 5),
            _Block(input_dim, input_dim, 5, pool="adaptive", emb=32),
            nn.GroupNorm(4, input_dim),
            nn.GRU(batch_first=True, input_size=input_dim, hidden_size=size_embeddings),
        )
        self.embed_dim = size_embeddings
        self.crop = int(crop)
        self.min_len = 123                       # shortest input the conv/pool stack accepts (measured)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < self.min_len:           # replicate-pad very short windows up to the minimum
            x = nn.functional.pad(x, (0, self.min_len - x.shape[-1]), mode="replicate")
        return self.backbone(x.float())[1][0]    # last GRU hidden state, [N, D]

    def forward(self, x: torch.Tensor):
        B, C, T = x.shape
        pooled = self.encode(x)
        n = max(1, T // 16)
        L = min(self.crop, T)
        if L == T or n == 1:
            return {"tokens": pooled.unsqueeze(1), "pooled": pooled}
        starts = torch.linspace(0, T - L, n, device=x.device).round().long()
        idx = starts.view(n, 1) + torch.arange(L, device=x.device).view(1, L)
        crops = x[:, :, idx].permute(0, 2, 1, 3).reshape(B * n, C, L)
        return {"tokens": self.encode(crops).view(B, n, -1), "pooled": pooled}


def extract_release(ckpt_path: str, out_path: str) -> None:
    """One-off: copy the IMU encoder keys out of the Lightning checkpoint (read without unpickling objects)."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    enc = {k[len("imu_encoder."):]: v for k, v in sd.items() if k.startswith("imu_encoder.backbone.")}
    if not enc:
        raise RuntimeError("no imu_encoder.backbone.* keys in the checkpoint")
    torch.save(enc, out_path)


def build(weights: Optional[str] = None, crop: int = 250, **_) -> nn.Module:
    m = PRIMUSEncoder(crop=crop)
    if weights:
        m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=False)
        missing = [k for k in m.state_dict() if k not in torch.load(weights, map_location="cpu", weights_only=True)]
        if missing:
            raise RuntimeError(f"PRIMUS weights missing {missing[:6]}")
    return m
