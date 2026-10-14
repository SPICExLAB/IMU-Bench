"""ImageBind's IMU branch (Girdhar et al., CVPR 2023) as an IMU-Bench plugin encoder.

The released `imagebind_huge.pth` holds all six modalities; only the IMU branch (19.6 M parameters, keys
`modality_*.imu.*`) is used, extracted once into `imagebind_huge_imu_only.pt`. The modules are built with
ImageBind's own classes from the official repository (github.com/facebookresearch/ImageBind, CC BY-NC-SA 4.0),
cloned under baselines/SSL/ImageBind/ImageBind.

Native input: [B, 6, 2000] = 10 s at 200 Hz from the Ego4D head-mounted camera IMU, accelerometer then gyroscope.
The preprocessing (units, normalisation) is not documented by the authors; the Ego4D IMU export is m/s^2 with
gravity and rad/s, which is what the harness delivers with `normalize_method: raw`, so no rescaling is applied.

Adaptation (stated, not hidden):
  * the harness resamples to 200 Hz (card `model_hz: 200`);
  * windows are cut into 8-sample patches as in pretraining, and the learned position embedding (250 patches)
    is linearly interpolated to the window's patch count (a 10.24 s window has 256, a 2.56 s window 64);
  * drop-path is 0 downstream (pretraining used 0.7, a training-time regulariser);
  * tokens = the trunk's patch tokens after the head LayerNorm (512-d, 40 ms each, pooled onto the 0.32 s grid
    by the plugin wrapper); pooled = the head LayerNorm's CLS token, i.e. the input of the projection into the
    CLIP space (512-d, so tokens and pooled share one width).
"""
from __future__ import annotations

import sys
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.models.baselines._vendor import baselines_dir

REPO = baselines_dir("SSL", "ImageBind", "ImageBind")
EMBED, BLOCKS, HEADS, KERNEL, NATIVE_PATCHES = 512, 6, 8, 8, 250


def _stub_packages() -> None:
    """Import imagebind.models.* without running imagebind/__init__.py, which pulls in the vision and audio
    data stack (pytorchvideo, torchaudio, pkg_resources) that the IMU branch does not use."""
    import os
    import types
    for name, sub in (("imagebind", ""), ("imagebind.models", "models")):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [os.path.join(REPO, "imagebind", sub) if sub else os.path.join(REPO, "imagebind")]
            sys.modules[name] = mod


class ImageBindIMU(nn.Module):
    def __init__(self, drop_path: float = 0.0):
        super().__init__()
        _stub_packages()
        from imagebind.models.helpers import EinOpsRearrange
        from imagebind.models.multimodal_preprocessors import (IMUPreprocessor, PatchEmbedGeneric,
                                                               SpatioTemporalPosEmbeddingHelper)
        from imagebind.models.transformer import MultiheadAttention, SimpleTransformer

        stem = PatchEmbedGeneric([nn.Linear(in_features=48, out_features=EMBED, bias=False)],
                                 norm_layer=nn.LayerNorm(normalized_shape=EMBED))
        self.pre = IMUPreprocessor(img_size=[6, 2000], num_cls_tokens=1, kernel_size=KERNEL, embed_dim=EMBED,
                                   pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
                                   imu_stem=stem)
        self.trunk = SimpleTransformer(
            embed_dim=EMBED, num_blocks=BLOCKS, ffn_dropout_rate=0.0, drop_path_rate=drop_path,
            attn_target=partial(MultiheadAttention, embed_dim=EMBED, num_heads=HEADS, bias=True, add_bias_kv=True),
            pre_transformer_layer=nn.Sequential(nn.Identity(), EinOpsRearrange("b l d -> l b d")),
            post_transformer_layer=EinOpsRearrange("l b d -> b l d"))
        self.head_ln = nn.LayerNorm(normalized_shape=EMBED, eps=1e-6)
        self.embed_dim = EMBED

    def load_release(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        own = {}
        for k, v in sd.items():
            if k.startswith("modality_preprocessors.imu."):
                own["pre." + k[len("modality_preprocessors.imu."):]] = v
            elif k.startswith("modality_trunks.imu."):
                own["trunk." + k[len("modality_trunks.imu."):]] = v
            elif k.startswith("modality_heads.imu.0."):
                own["head_ln." + k[len("modality_heads.imu.0."):]] = v
        res = self.load_state_dict(own, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"ImageBindIMU: missing {res.missing_keys[:6]}")

    def forward(self, x: torch.Tensor):
        T = (x.shape[-1] // KERNEL) * KERNEL
        x = x[..., :T]
        p = x.unfold(-1, KERNEL, KERNEL).permute(0, 2, 1, 3)              # [B, P, 6, 8], as IMUPreprocessor
        p = p.reshape(p.size(0), p.size(1), -1)
        stem = self.pre.imu_stem
        tok = stem.norm_layer(stem.proj(p))                              # [B, P, 512]
        pos = self.pre.pos_embed                                          # [1, 1 + 250, 512]
        pos_p = pos[:, 1:]
        if tok.shape[1] != NATIVE_PATCHES:
            pos_p = F.interpolate(pos_p.transpose(1, 2), size=tok.shape[1], mode="linear",
                                  align_corners=False).transpose(1, 2)
        cls = self.pre.cls_token.expand(tok.shape[0], -1, -1) + pos[:, :1]
        out = self.trunk(torch.cat([cls, tok + pos_p], dim=1))
        out = self.head_ln(out)
        return {"tokens": out[:, 1:], "pooled": out[:, 0]}


def build(weights: Optional[str] = None, drop_path: float = 0.0, **_) -> nn.Module:
    m = ImageBindIMU(drop_path=drop_path)
    if weights:
        m.load_release(weights)
    return m
