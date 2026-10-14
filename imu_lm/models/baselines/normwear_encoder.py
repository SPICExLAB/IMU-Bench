"""NormWear (Luo et al., 2024) as a drop-in ViT1DEncoder look-alike for this repo's downstream harness.

"Toward Foundation Models for Multivariate Wearable Sensing of Physiological Signals"
  code     https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear  (Apache-2.0), vendored clone at
           baselines/SSL/NormWear (commit 07517fc)
  weights  the authors' release v1.0.0-alpha, `normwear_pretrain_ckpt.pth` (weights only, 545 MB), loaded with
           strict=True into their own `NormWear` backbone -- nothing is re-implemented here.

What the model is: a 12-layer, 768-wide ViT over the continuous-wavelet scalogram of EACH sensor channel (the
signal and its first two differences, 65 Ricker scales, 9x5 patches), with a [CLS]-based fusion across channels
in every layer. It was pretrained by masked reconstruction on ~2.5 M segments of heterogeneous wearable signals
(PPG, ECG, EEG, GSR and IMU), so unlike every controlled row of IMU-Bench it shares neither our backbone nor our
corpus: it is an EXTERNAL pretrained reference, like HarNet, and is reported as such.

How it is wired (mirrors harnet_encoder.py):
  * `forward_tokens(x)`: x [B, C, T] -> scalogram (their torch `cwt_wrap`, differentiable) -> their
    `get_signal_embedding` -> [B, C, 1 + Lp*13, 768]. The 13 frequency patches of one time slot are averaged (their
    own `combine_freq_patch` sums them), giving one token per 9 samples per channel; the time axis is then linearly
    interpolated onto the harness grid Pt = T // patch_size, exactly as the Inertia-1 adapter does. Tokens are
    returned channel-major, [B, C*Pt, 768], the layout every task decoder expects (in_channels = C).
  * long windows: attention is quadratic in Lp*13 tokens, so a window longer than `chunk_samples` (default 256,
    5.12 s at 50 Hz; pretraining used ~6 s) is encoded in consecutive chunks whose tokens are concatenated in time.
  * `forward(x)`: mean over all tokens -> [B, 768] (their README's mean aggregation).
  * the sampling rate: their `get_embedding` resamples only inputs above 256 Hz, so a 50 Hz signal is consumed as it
    is, as in their own examples; we do the same.
  * input scaling: the harness' per-window z-score is kept (it is the benchmark's input for every encoder and
    makes the scalogram independent of sensor units); set `preprocess.normalize.enabled: false` to feed raw units.

Config block (``cfg["normwear"]``):
    weights:        path to normwear_pretrain_ckpt.pth   (required unless resuming from a harness checkpoint)
    chunk_samples:  max samples encoded in one attention pass (default 256)
    freeze:         true -> backbone parameters do not require grad (probe runs freeze through the objective anyway)
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.models.baselines._vendor import baselines_dir
VENDOR_PARENT = baselines_dir("SSL")
N_FREQ_PATCH = 13          # 65 scales / patch height 5 -- fixed by the released architecture


class NormWearEncoder(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        ncfg: Dict[str, Any] = dict((cfg or {}).get("normwear", {}) or {})
        if VENDOR_PARENT not in sys.path:
            sys.path.insert(0, VENDOR_PARENT)
        from NormWear.modules.normwear import NormWear, cwt_wrap          # vendored verbatim

        self._cwt_wrap = cwt_wrap
        # exactly the constructor call of NormWear/main_model.py::NormWearModel
        self.backbone = NormWear(img_size=(387, 65), patch_size=(9, 5), mask_scheme="random", mask_prob=0.8,
                                 use_cwt=True, nvar=4, comb_freq=False)
        self.embed_dim = 768
        data = (cfg or {}).get("data", {}) or {}
        self.in_channels = len(data.get("sensor_columns") or [0] * 6)
        vit_cfg = (cfg or {}).get("vit1d", {}) or {}
        self.patch_size = int(vit_cfg.get("patch_size", 16))
        self.use_cls_token = False
        self.num_freq_bands = 1
        self.chunk = int(ncfg.get("chunk_samples", 256))
        if ncfg.get("weights"):
            self.load_release_weights(str(ncfg["weights"]))
        if bool(ncfg.get("freeze", False)):
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def load_release_weights(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        self.backbone.load_state_dict(sd, strict=True)          # the release file is the backbone's own state dict

    # ------------------------------------------------------------------------------------------------------
    def _encode_chunk(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, L] -> [B, C, Lp, 768]: one token per 9-sample time slot and channel."""
        B, C, L = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):      # the wavelet transform is not fp16-safe
            spec = self._cwt_wrap(x.reshape(B * C, L).float(), 0.1, 64)     # [B*C, 3, L-2, 65]
        spec = spec.view(B, C, spec.shape[1], spec.shape[2], spec.shape[3])
        out = self.backbone.get_signal_embedding(spec, hidden_out=False, device=x.device)   # [B, C, 1 + Lp*13, E]
        tok = out[:, :, 1:, :]
        Lp = tok.shape[2] // N_FREQ_PATCH
        return tok[:, :, :Lp * N_FREQ_PATCH].reshape(B, C, Lp, N_FREQ_PATCH, self.embed_dim).mean(dim=3)

    def forward_tokens(self, x: torch.Tensor):
        """(tokens [B, C*Pt, 768] channel-major, Pt) on the harness grid Pt = T // patch_size."""
        if x.dim() != 3:
            raise ValueError(f"NormWearEncoder expects [B, C, T], got {tuple(x.shape)}")
        B, C, T = x.shape
        Pt = max(1, T // self.patch_size)
        chunks = [x[:, :, s:s + self.chunk] for s in range(0, T, self.chunk)]
        if len(chunks) > 1 and chunks[-1].shape[-1] < 32:                   # a stub too short to patch: merge it
            chunks[-2] = torch.cat([chunks[-2], chunks[-1]], dim=-1); chunks.pop()
        tok = torch.cat([self._encode_chunk(c) for c in chunks], dim=2)     # [B, C, Lp_total, E]
        if tok.shape[2] != Pt:
            tok = F.interpolate(tok.reshape(B * C, tok.shape[2], self.embed_dim).transpose(1, 2), size=Pt,
                                mode="linear", align_corners=False).transpose(1, 2).reshape(B, C, Pt, self.embed_dim)
        return tok.reshape(B, C * Pt, self.embed_dim).contiguous(), Pt

    def forward(self, x: torch.Tensor, mask=None, padding_mask=None, return_keep_padding_mask: bool = False):
        if mask is not None or padding_mask is not None:
            raise NotImplementedError("NormWearEncoder: masked / padded forward is not supported (probe/finetune only)")
        return self.forward_tokens(x)[0].mean(dim=1)                        # [B, 768]

    def load_pretrained_state(self, enc_state: Dict[str, Any], strict: bool = True) -> None:
        """A checkpoint written BY this harness (keys prefixed `backbone.`), else the release layout."""
        sd = enc_state["model"] if isinstance(enc_state, dict) and "model" in enc_state else enc_state
        if any(k.startswith("backbone.") for k in sd):
            sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        res = self.backbone.load_state_dict(sd, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"NormWearEncoder: checkpoint missing keys {res.missing_keys[:8]}")


def is_normwear_cfg(cfg: Any) -> bool:
    return isinstance(cfg, dict) and bool(cfg.get("normwear"))


def normwear_downstream_config(cfg: Dict[str, Any], weights: str, *, chunk_samples: int = 256) -> Dict[str, Any]:
    """Rewrite a ViT1D downstream template so that it probes / fine-tunes NormWear instead. Only the encoder side
    changes: the task recipe, protocol, decoder class and scorer stay the harness' own, all six channels are kept and
    the per-window z-score stays on. Decoders read 768-d tokens instead of 384-d ones; their own width is unchanged."""
    cfg = dict(cfg)
    vit = dict(cfg.get("vit1d", {}) or {})
    for k in ("encoder", "shared_kernel", "pos_embed_type", "use_cls_token", "max_channels"):
        vit.pop(k, None)
    vit.setdefault("patch_size", 16)
    vit.setdefault("max_patches_per_channel", 256)
    vit["pooling"] = "mean"
    cfg["vit1d"] = vit
    cfg["normwear"] = {"weights": weights, "chunk_samples": int(chunk_samples)}
    return cfg
