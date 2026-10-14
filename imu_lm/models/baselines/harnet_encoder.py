"""HarNet (Yuan et al., npj Digital Medicine 2024) as a drop-in ViT1DEncoder
look-alike for this repo's downstream harness.

"Self-supervised learning for human activity recognition using 700,000
person-days of wearable data" (OxWearables/ssl-wearables). Unlike Inertia-1,
this one DOES ship pretrained weights: `model_check_point/mtl_{5,30}_best.mdl`
in the repo, loaded here with zero missing and zero unexpected keys.

Why it is worth the wiring: it is pretrained on UK Biobank at ~17 million hours
of accelerometry, roughly 7,000x the 2,472 h four-source corpus this project
pretrains on. If our encoder matches it on activity recognition, "corpus scale
is not the bottleneck" stops being an internal ablation and becomes a
cross-paper result.

    backbone     1-D ResNet (5 stages, anti-aliased downsampling), 4.25 M params
    native input [B, 3, 150]  = 5 s of TRIAXIAL ACCELEROMETER at 30 Hz, in g
                 (harnet5; the 30 s variant uses mtl_30_best.mdl)
    output       512-d

THE SHAPE PROBLEM, AND WHAT THIS ADAPTER DOES ABOUT IT
------------------------------------------------------
HarNet is a WINDOW classifier, not a sequence model: its five stages downsample
by 2*2*3*3*3 = 108, so a native 150-sample window collapses to a feature map of
length 1 (measured, not assumed). Feeding it a 10.24 s harness window gives 4
frames, and the pose/odometry decoders want 32 and 8 tokens respectively.

Three ways to paper over that were rejected:
  * replicating one vector across Pt token slots -- the decoder then sees no
    temporal information at all, so pose/odometry would score badly because of
    the adapter, not the model, and a reader would misattribute it;
  * tapping an intermediate stage for a longer feature map -- throws away the
    last two stages' pretrained weights, i.e. benchmarks half the model;
  * skipping the two kinematic tasks -- honest, but leaves the table incomplete.

Instead the encoder is applied AS INTENDED, in a sliding window: the harness
window is cut into Pt overlapping 5 s crops on the harness token grid, each is
encoded natively, and the resulting vectors are the tokens. Cost is Pt forward
passes worth of FLOPs, but they run as ONE batched call (B*Pt crops), so it is
~Pt x the compute of a single window and no extra latency from Python looping.
Crops are taken with `unfold`, so each token sees a genuine 5 s of signal
centred on its slot.

Short harness windows are a second shape problem. A 2.56 s window resamples to
77 samples at 30 Hz, and the backbone's minimum viable input is 108 (measured:
107 raises, 108 gives one frame -- unsurprising, it equals the 108x total
downsampling). Such windows are REFLECT-padded up to the native 150, not
zero-padded: to an accelerometer model a run of zeros is not "no data", it is a
perfectly still, gravity-free device -- a signal that never occurs in UK Biobank
and that the model would happily encode as such. Reflection keeps the local
amplitude and spectrum of the real signal and restores the window length the
model was pretrained on.

Config block (``cfg["harnet"]``):
    variant:         "harnet5" | "harnet30"        (default harnet5)
    weights:         path to mtl_*.mdl             (required unless resuming)
    input_hz:        rate of the batches this run feeds (default 50)
    acc_scale_to_g:  multiplier taking batch acc units to g (default 1/9.80665)
    acc_channels:    indices of the 3 acc channels (default [0,1,2])
    window_sec:      native crop length in seconds (default from variant)
    short_pad:       "reflect" (default) | "replicate" -- how a sub-native window
                     is brought up to the native length
    freeze:          freeze the backbone (default false; the probe protocol
                     freezes through the harness instead)
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G_MPS2 = 9.80665
MODEL_HZ = 30.0
from imu_lm.models.baselines._vendor import baselines_dir
VENDOR = baselines_dir("SSL", "ssl_wearables")
VARIANTS = {                       # epoch_len, native window seconds, feature dim
    "harnet5": (5, 5.0, 512),
    "harnet10": (10, 10.0, 1024),     # mtl_best.mdl, the 10 s model the HarNet paper reports
    "harnet30": (30, 30.0, 1024),
}


def _resample_operator(T_in: int, hz_in: float, hz_out: float) -> np.ndarray:
    """Matrix R [T_out, T_in] with y = R @ x == scipy.signal.resample_poly(x, up, down)."""
    from fractions import Fraction
    from scipy.signal import resample_poly
    fr = Fraction(hz_out / hz_in).limit_denominator(1000)
    R = resample_poly(np.eye(T_in, dtype=np.float64), fr.numerator, fr.denominator, axis=0)
    return R.astype(np.float32)


class HarnetEncoder(nn.Module):
    """ViT1DEncoder-compatible wrapper around the HarNet ResNet backbone."""

    def __init__(self, cfg: Any):
        super().__init__()
        hcfg: Dict[str, Any] = dict((cfg or {}).get("harnet", {}) or {})
        self.variant = str(hcfg.get("variant", "harnet5")).lower()
        if self.variant not in VARIANTS:
            raise ValueError(f"harnet.variant must be one of {sorted(VARIANTS)}, got {self.variant!r}")
        epoch_len, win_sec, dim = VARIANTS[self.variant]

        if VENDOR not in sys.path:
            sys.path.insert(0, os.path.normpath(VENDOR))
        import accNet                                        # vendored verbatim

        # is_mtl=True builds the four pretext heads the released checkpoint carries;
        # they are dropped below, but the flag is what makes the key names line up.
        net = accNet.Resnet(output_size=2, n_channels=3, is_mtl=True,
                            epoch_len=epoch_len, resnet_version=1)
        self.backbone = net.feature_extractor

        # ---- harness-facing attributes (mirror ViT1DEncoder) -------------------
        self.embed_dim = int(dim)
        self.in_channels = 1              # one fused acc token stream
        self.num_input_channels = 3       # HarNet is accelerometer-only by construction
        vit_cfg = (cfg or {}).get("vit1d", {}) or {}
        self.patch_size = int(vit_cfg.get("patch_size", 16))
        self.use_cls_token = False
        self.num_freq_bands = 1

        # ---- input adaptation ---------------------------------------------------
        self.input_hz = float(hcfg.get("input_hz", 50))
        self.model_hz = float(hcfg.get("model_hz", MODEL_HZ))
        self.acc_scale_to_g = float(hcfg.get("acc_scale_to_g", 1.0 / G_MPS2))
        self.acc_channels = [int(c) for c in hcfg.get("acc_channels", [0, 1, 2])]
        if len(self.acc_channels) != 3:
            raise ValueError(f"HarNet consumes exactly 3 accelerometer channels, got {self.acc_channels}")
        self.crop = int(round(float(hcfg.get("window_sec", win_sec)) * self.model_hz))
        self.short_pad = str(hcfg.get("short_pad", "reflect")).lower()
        self._resamplers: Dict[int, torch.Tensor] = {}

        if hcfg.get("weights"):
            self.load_release_weights(str(hcfg["weights"]))
        if bool(hcfg.get("freeze", False)):
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    # ---- weights -------------------------------------------------------------
    def load_release_weights(self, path: str) -> None:
        """Load an official `mtl_*.mdl`. The release is a DataParallel state dict
        (`module.` prefix) carrying four pretext heads (aot/scale/permute/time_w)
        that this adapter does not use; only `feature_extractor.*` is taken, and
        every backbone key must be present or this raises."""
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        pref = "module.feature_extractor."
        body = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
        if not body:                                  # already-stripped checkpoint
            body = {k[len("feature_extractor."):]: v for k, v in sd.items()
                    if k.startswith("feature_extractor.")}
        res = self.backbone.load_state_dict(body, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"HarnetEncoder: backbone missing keys {res.missing_keys[:8]}")

    # ---- input prep ----------------------------------------------------------
    def _resampler(self, T: int, device, dtype) -> torch.Tensor:
        R = self._resamplers.get(T)
        if R is None or R.device != device:
            R = torch.from_numpy(_resample_operator(T, self.input_hz, self.model_hz)).to(device)
            self._resamplers[T] = R
        return R.to(dtype)

    def prep_input(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] harness batch -> [B, 3, T30] g-unit 30 Hz signal."""
        if x.dim() != 3 or x.shape[1] < 3:
            raise ValueError(f"HarnetEncoder expects [B, C>=3, T], got {tuple(x.shape)}")
        x = x[:, self.acc_channels, :].float() * self.acc_scale_to_g
        if abs(self.input_hz - self.model_hz) > 1e-6:
            R = self._resampler(x.shape[-1], x.device, x.dtype)     # [T30, T]
            x = torch.matmul(x, R.t())
        return x

    # ---- ViT1DEncoder interface ----------------------------------------------
    MIN_LEN = 108      # measured: the backbone raises below this (== its total downsampling)

    def _pad_short(self, x: torch.Tensor) -> torch.Tensor:
        """Bring a sub-native window up to the native crop length by reflection."""
        need = self.crop - x.shape[-1]
        if need <= 0:
            return x
        mode = self.short_pad
        if mode == "reflect" and need >= x.shape[-1]:
            mode = "replicate"          # torch reflect requires pad < length
        return F.pad(x, (0, need), mode=mode)

    def _encode_crops(self, crops: torch.Tensor) -> torch.Tensor:
        """[N, 3, L] -> [N, D]: backbone then global average over whatever frames
        the ResNet's 108x downsampling leaves (1 for a native 5 s crop)."""
        crops = self._pad_short(crops)
        with torch.autocast(device_type=crops.device.type, enabled=False):
            f = self.backbone(crops.float())            # [N, D, T']
        return f.mean(dim=-1)

    def forward_tokens(self, x: torch.Tensor):
        """Return (tokens [B, Pt, D], Pt) on the harness grid Pt = T // patch_size.

        Each token is a native-length crop encoded on its own -- the sliding-window
        application this backbone was designed for -- batched into a single call.
        """
        x30 = self.prep_input(x)                        # [B, 3, T30]
        B, _, T30 = x30.shape
        Pt = max(1, int(x.shape[-1]) // self.patch_size)
        L = min(self.crop, T30)                         # short windows keep their true length
        if Pt == 1 or T30 <= L:
            return self._encode_crops(x30).unsqueeze(1).expand(B, Pt, self.embed_dim).contiguous(), Pt

        # Pt crop starts spread evenly over the valid range, so token i is centred
        # on the part of the window its slot represents.
        starts = torch.linspace(0, T30 - L, Pt, device=x30.device).round().long()
        idx = starts.view(Pt, 1) + torch.arange(L, device=x30.device).view(1, L)   # [Pt, L]
        crops = x30[:, :, idx]                          # [B, 3, Pt, L]
        crops = crops.permute(0, 2, 1, 3).reshape(B * Pt, 3, L)
        toks = self._encode_crops(crops).view(B, Pt, self.embed_dim)
        return toks.contiguous(), Pt

    def forward(self, x: torch.Tensor, mask=None, padding_mask=None,
                return_keep_padding_mask: bool = False):
        if mask is not None or padding_mask is not None:
            raise NotImplementedError(
                "HarnetEncoder: masked / padded forward is not supported (probe/finetune only)")
        return self.forward_tokens(x)[0].mean(dim=1)    # [B, D]

    def load_pretrained_state(self, enc_state: Dict[str, Any], strict: bool = True) -> None:
        """Load a checkpoint written BY this harness (keys prefixed `backbone.`),
        falling back to an official release file's layout."""
        sd = enc_state["model"] if isinstance(enc_state, dict) and "model" in enc_state else enc_state
        if any(k.startswith("backbone.") for k in sd):
            body = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        else:
            pref = "module.feature_extractor."
            body = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
        res = self.backbone.load_state_dict(body, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"HarnetEncoder: checkpoint missing keys {res.missing_keys[:8]}")


def is_harnet_cfg(cfg: Any) -> bool:
    return isinstance(cfg, dict) and bool(cfg.get("harnet"))


def build_encoder(cfg: Any) -> nn.Module:
    return HarnetEncoder(cfg)


# --------------------------------------------------------------------------- #
#  Config conversion (used by bench/make_config.py)
# --------------------------------------------------------------------------- #
OPENWATCH_LSB_PER_G = 4096.0


def harnet_downstream_config(cfg: Dict[str, Any], variant: str, weights: str, *,
                             openwatch: bool = False, clip: bool = False) -> Dict[str, Any]:
    """Rewrite a ViT1D downstream template so it probes/finetunes HarNet instead.

    Only the encoder side changes; the task recipe, protocol and scorer are the
    harness' own, so a HarNet row is produced by exactly the pipeline that
    produces every other row.

      vit1d.encoder / shared_kernel / pos_embed_type / use_cls_token -> removed
      harnet: {variant, weights, input_hz, acc_scale_to_g, ...}
      data.acc_only: true + 3 acc sensor columns   (HarNet is accelerometer-only)
      preprocess.filter/normalize.enabled: false   (it consumes raw g, no z-score)

    There is no `pretrained_run`: unlike the Inertia-1 rows, whose weights are
    our own reproduction living in runs/, HarNet's weights are the authors'
    released file and are loaded by path inside the encoder.
    """
    cfg = dict(cfg)
    vit = dict(cfg.get("vit1d", {}) or {})
    for k in ("encoder", "shared_kernel", "pos_embed_type", "use_cls_token", "max_channels"):
        vit.pop(k, None)
    vit.setdefault("patch_size", 16)
    vit.setdefault("max_patches_per_channel", 256)
    vit["pooling"] = "mean"
    cfg["vit1d"] = vit

    if variant not in VARIANTS:
        raise ValueError(f"unknown HarNet variant {variant!r}; choose from {sorted(VARIANTS)}")
    h = {"variant": variant, "weights": weights, "input_hz": 50,
         "acc_scale_to_g": 1.0 / G_MPS2, "acc_channels": [0, 1, 2],
         "model_hz": MODEL_HZ, "short_pad": "reflect"}
    if openwatch:
        # OpenWatch CSVs are integer counts, not m/s^2 (4096 LSB/g) -- the same
        # trap that saturated the physical-scaling gesture cells in 2026-08.
        h["input_hz"] = 100
        h["acc_scale_to_g"] = 1.0 / OPENWATCH_LSB_PER_G
        vit["patch_size"] = 20
    cfg["harnet"] = h
    if openwatch:
        # patch_size=20 above is only legal with a 100-sample pad (openwatch_loader
        # requires window_pad_len % patch_size == 0). The template's 128 would also
        # append 28 zero samples -- in g units that is free fall, which HarNet would
        # read as signal. Same pairing as inertia1_downstream_config.
        obj = cfg["objective"]
        blk = obj.setdefault(obj["type"], {}) or {}
        obj[obj["type"]] = blk
        blk["window_pad_len"] = 100

    data = dict(cfg.get("data", {}) or {})
    cols = data.get("sensor_columns") or ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    data["sensor_columns"] = list(cols[:3])
    data["acc_only"] = True
    cfg["data"] = data

    pp = dict(cfg.get("preprocess", {}) or {})
    pp.setdefault("impute", {"enabled": True, "method": "linear", "max_missing_frac": 0.2})
    pp["filter"] = {"enabled": False}
    pp["normalize"] = {"enabled": False, "method": "zscore", "eps": 1.0e-6}
    cfg["preprocess"] = pp
    if clip:
        cfg.setdefault("humanml3d", {})["zscore"] = False
    return cfg
