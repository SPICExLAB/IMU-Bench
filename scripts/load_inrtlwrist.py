#!/usr/bin/env python
"""Load the released InrtlWrist checkpoint (https://huggingface.co/YiquanLi/InrtlWrist) into the ViT1D encoder.

    from scripts.load_inrtlwrist import load_inrtlwrist, zscore
    enc = load_inrtlwrist("inrtlwrist_mae_5src_step144k.pt")        # eval mode, CPU
    x = zscore(torch.randn(2, 6, 512))   # [B, 6, T]: acc x/y/z (m/s^2) then gyro x/y/z (rad/s), one wrist, FLU frame,
                                          # 50 Hz, T a multiple of 16 up to 512 (0.32 s patches, 10.24 s max)
    z = enc(x)                            # [B, 384] pooled (CLS) embedding, what the benchmark's frozen probes read
    tokens, patches = enc.forward_tokens(x)   # [B, 6 * T/16, 384] per-channel patch tokens, used by the kinematic decoders

Run this file directly to check the checkpoint loads and to print the output shapes:
    python scripts/load_inrtlwrist.py path/to/inrtlwrist_mae_5src_step144k.pt
"""
import sys
import torch

from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn

SENSOR_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-channel, per-window standardisation, the input normalisation used in pretraining and in every task."""
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + eps)


def load_inrtlwrist(path: str, device: str = "cpu") -> ViT1DEncoderAltAttn:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = {"vit1d": ck["config"]["vit1d"], "data": {"sensor_columns": SENSOR_COLUMNS}}
    enc = ViT1DEncoderAltAttn(cfg)
    enc.load_state_dict(ck["model"], strict=True)
    return enc.to(device).eval()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "inrtlwrist_mae_5src_step144k.pt"
    enc = load_inrtlwrist(path)
    n = sum(p.numel() for p in enc.parameters())
    print(f"loaded {path}: {n / 1e6:.1f} M parameters")
    with torch.no_grad():
        for T in (512, 128):
            x = zscore(torch.randn(2, 6, T))
            z = enc(x)
            tok, pt = enc.forward_tokens(x)
            print(f"T={T}: pooled {tuple(z.shape)}, tokens {tuple(tok.shape)}, patches per channel {pt}")
