"""InrtlWrist through the plugin interface: a template for submitting any encoder to the open track.

The harness imports `build` (named in the encoder card as `entry: examples.plugin.inrtlwrist_plugin:build`),
calls it once per job, and wraps the returned module. Copy this file, replace the body of `build` and
`forward` with your model, and keep the contract:

    build(weights=<local path or None>, **kwargs) -> nn.Module with an integer `embed_dim`
    module(x)  x: [B, C, T] float, acc x/y/z (m/s^2) then gyro x/y/z (rad/s), wrist FLU frame, card's model_hz
               -> [B, N, D] tokens over time, [B, D] pooled, or {"tokens": [B, N, D], "pooled": [B, D]}
"""
import torch
import torch.nn as nn

from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn

SENSOR_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


class InrtlWristPlugin(nn.Module):
    def __init__(self, ckpt: dict):
        super().__init__()
        self.enc = ViT1DEncoderAltAttn({"vit1d": ckpt["config"]["vit1d"], "data": {"sensor_columns": SENSOR_COLUMNS}})
        self.enc.load_state_dict(ckpt["model"], strict=True)
        self.embed_dim = self.enc.embed_dim

    def forward(self, x: torch.Tensor):
        tokens, pt = self.enc.forward_tokens(x)                 # [B, C * Pt, D], channel-major
        B, _, D = tokens.shape
        tokens = tokens.view(B, -1, pt, D).mean(dim=1)          # one token per 0.32 s: mean over channels
        return {"tokens": tokens, "pooled": self.enc(x)}        # pooled = the CLS embedding the probes read


def build(weights=None, **kwargs) -> nn.Module:
    if weights is None:
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download("YiquanLi/InrtlWrist", "inrtlwrist_mae_5src_step144k.pt")
    return InrtlWristPlugin(torch.load(weights, map_location="cpu", weights_only=False))
