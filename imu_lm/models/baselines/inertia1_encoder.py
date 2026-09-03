"""Inertia-1 (Xu et al. 2026, arXiv:2607.06617) encoders as drop-in ViT1DEncoder
look-alikes for this repo's downstream harness.

Inertia-1 (github.com/yang-ai-lab/Inertia-1) ships code but NO pretrained
checkpoints (issue #1 there is an open weights request), so the checkpoints
consumed here are our own reproductions of their two "robust" objectives
(paper Sec. 4.1/4.3), pretrained on NHANES with their recipe by
baselines/inertia1/scripts/pretrain_inertia1.py:

    * ``ar_transformer`` -- ART: linear patch projection (patch_len=10 samples =
      0.5 s @ 20 Hz), sinusoidal PE, causal nn.TransformerEncoder (pre-norm),
      next-patch MSE.  Downstream representation = NON-causal pass, mean over
      patch tokens (exactly their ``ARTransformerModule.encode``).
    * ``patchtst``       -- PatchTST_Accel (x/y/z treated as one 3-vector per
      patch), masked patch reconstruction (mask 0.4).  Downstream = mean of the
      backbone tokens (their ``PatchTSTModule.encode``).

Both use their "small" preset (d=512, L=4, H=4, d_ff=1024; ~8.5M params) and
their default input: 30 s windows of 20 Hz TRIAXIAL ACCELEROMETER in **g**
units, **no normalisation** (no RevIN / z-score anywhere in their model or
downstream loader).

The two nn.Modules below (ARTransformerNet / PatchTSTNet) keep the exact
parameter names of the corresponding Lightning modules in their repo
(``patch_proj``, ``pos_encoder.pe``, ``transformer.layers.*``, ``head`` /
``model.backbone.*``, ``model.head.*``) so an official ``.ckpt`` state_dict
would load into them unchanged if one is ever released.

Adaptation to this harness (all explicit config, see ``Inertia1Encoder``):
    input  [B, C, T] at ``inertia1.input_hz`` (repo canon 50 Hz; OpenWatch 100)
      -> take the 3 accelerometer channels (``inertia1.acc_channels``)
      -> multiply by ``inertia1.acc_scale_to_g`` (m/s^2 -> g, LSB -> g, ...)
      -> anti-aliased polyphase resample to 20 Hz (scipy resample_poly, applied
         as a cached linear operator so it runs on GPU)
      -> Inertia-1 backbone -> native tokens [B, N, 512], N = T20 // 10
    forward(x)        -> [B, 512]   mean over native tokens (their encode())
    forward_tokens(x) -> ([B, Pt, 512], Pt) with Pt = T // vit1d.patch_size:
                         native tokens linearly interpolated onto the harness'
                         token grid so the existing pose / odometry decoders run
                         unchanged (in_channels = 1: one fused token stream).
Windows shorter than the 30 s pretraining window use the first N positions of
PatchTST's learned positional table (ART's sinusoidal PE is length-agnostic).
Configs must turn OFF the harness' per-window z-score / band-pass
(``preprocess.normalize.enabled: false``, ``preprocess.filter.enabled: false``)
so the model sees raw g-scale signal as in its pretraining.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_INERTIA1_REPO = os.path.join(_REPO_ROOT, "baselines", "inertia1", "repo")
if _INERTIA1_REPO not in sys.path:
    sys.path.insert(0, _INERTIA1_REPO)

from inertia1.models.backbones.patchtst.patchTST_accel import PatchTST_Accel  # noqa: E402  (pure torch)
from inertia1.augmentations.patchtst import create_patch, random_masking  # noqa: E402

MODEL_HZ = 20
G_MPS2 = 9.80665

SMALL_PRESET = dict(d_model=512, n_layers=4, n_heads=4, d_ff=1024)
MEDIUM_PRESET = dict(d_model=768, n_layers=6, n_heads=8, d_ff=2048)
LARGE_PRESET = dict(d_model=1024, n_layers=8, n_heads=16, d_ff=4096)
PRESETS = {"small": SMALL_PRESET, "medium": MEDIUM_PRESET, "large": LARGE_PRESET}


# --------------------------------------------------------------------------- #
#  ART  (verbatim port of inertia1/methods/ar_transformer_module.py, no Lightning)
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (buffer name ``pe`` as in their code)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class ARTransformerNet(nn.Module):
    """Inertia-1 AR-Transformer.  x: [B, C, T] -> next-patch prediction."""

    def __init__(self, channels: int = 3, patch_len: int = 10, d_model: int = 512, n_heads: int = 4,
                 n_layers: int = 4, d_ff: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.patch_len = int(patch_len)
        self.channels = int(channels)
        patch_dim = self.channels * self.patch_len
        self.d_model = int(d_model)
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, patch_dim)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        n = T // self.patch_len
        x = x[:, :, : n * self.patch_len].reshape(B, C, n, self.patch_len)
        return x.permute(0, 2, 1, 3).reshape(B, n, C * self.patch_len)

    @staticmethod
    def _causal_mask(sz: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._to_patches(x)
        h = self.pos_encoder(self.patch_proj(patches[:, :-1, :]))
        h = self.transformer(h, mask=self._causal_mask(h.size(1), h.device), is_causal=True)
        return self.head(h)

    def pretrain_loss(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._to_patches(x)
        target = patches[:, 1:, :]
        y_hat = self.forward(x)
        return torch.mean((y_hat - target) ** 2)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Their ``encode()`` before the mean: NON-causal pass over all patches."""
        h = self.pos_encoder(self.patch_proj(self._to_patches(x)))
        return self.transformer(h)  # [B, N, d_model]


# --------------------------------------------------------------------------- #
#  PatchTST  (mirror of inertia1/methods/patchtst_module.py, no Lightning)
# --------------------------------------------------------------------------- #
class PatchTSTNet(nn.Module):
    """Inertia-1 PatchTST_Accel with masked-patch pretraining.  x: [B, 3, T]."""

    def __init__(self, patch_len: int = 10, stride: int = 10, mask_ratio: float = 0.4, num_patch: int = 60,
                 n_layers: int = 4, d_model: int = 512, n_heads: int = 4, d_ff: int = 1024,
                 norm: str = "LayerNorm", attn_dropout: float = 0.0, dropout: float = 0.0, act: str = "gelu",
                 res_attention: bool = True, pre_norm: bool = True, store_attn: bool = False,
                 pe: str = "zeros", learn_pe: bool = True, head_dropout: float = 0.0):
        super().__init__()
        self.patch_len, self.stride, self.mask_ratio = int(patch_len), int(stride), float(mask_ratio)
        self.num_patch = int(num_patch)
        self.d_model = int(d_model)
        self.model = PatchTST_Accel(
            patch_len=self.patch_len, stride=self.stride, num_patch=self.num_patch,
            n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
            attn_dropout=attn_dropout, dropout=dropout, act=act, res_attention=res_attention,
            pre_norm=pre_norm, store_attn=store_attn, pe=pe, learn_pe=learn_pe,
            head_dropout=head_dropout, head_type="pretrain", target_dim=None, verbose=False,
        )

    def build_W_P6(self) -> None:
        """Widen the patch projection from 3*patch_len to 6*patch_len (acc+gyro variant).

        Built eagerly at construction so the module's key set is identical whether or not a
        forward pass has run -- a lazily-created layer breaks strict state_dict loading.
        """
        import torch.nn as _nn
        bb = self.model.backbone
        w = bb.W_P.weight                                       # [d_model, 3*L]
        W6 = _nn.Linear(2 * w.shape[1], w.shape[0], bias=bb.W_P.bias is not None).to(w.device, w.dtype)
        with torch.no_grad():
            W6.weight.copy_(torch.cat([w, w], dim=1) * 0.5)     # halved so acc keeps its init scale
            if bb.W_P.bias is not None:
                W6.bias.copy_(bb.W_P.bias)
        self.W_P6 = W6

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        xb = x.permute(0, 2, 1)  # [B, T, 3]
        xb_patch, _ = create_patch(xb, self.patch_len, self.stride)  # [B, N, 3, patch_len]
        return xb_patch

    def _backbone(self, xb_patch: torch.Tensor) -> torch.Tensor:
        """AccelPatchTSTEncoder.forward with the positional table sliced to the
        actual patch count (identical to theirs when N == num_patch)."""
        bb = self.model.backbone
        B, N, C, L = xb_patch.shape
        assert C in (3, 6), f"expected 3 (acc) or 6 (acc+gyro) channels, got {C}"
        if N > bb.W_pos.shape[0]:
            raise ValueError(f"window has {N} patches > pretraining table {bb.W_pos.shape[0]}")
        flat = xb_patch.reshape(B, N, -1)
        if C == 6:
            # 6-channel variant for the architecture-vs-modality 2x2: their W_P projects
            # 3*patch_len -> d_model, so widen it to 6*patch_len by tiling the acc weights across
            # the gyro half (halved so the acc pathway keeps its original scale at init). Built
            # once, then trained normally -- the backbone is otherwise untouched.
            return bb.encoder(bb.dropout(self.W_P6(flat) + bb.W_pos[:N]))
        z = bb.W_P(flat)
        z = bb.dropout(z + bb.W_pos[:N])
        return bb.encoder(z)  # [B, N, d_model]

    def pretrain_loss(self, x: torch.Tensor) -> torch.Tensor:
        xb_patch = self._patches(x)
        xb_masked, _, mask, _ = random_masking(xb_patch, self.mask_ratio)
        preds = self.model.head(self._backbone(xb_masked))
        loss = ((preds - xb_patch) ** 2).mean(dim=-1)  # [B, N, 3]
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(self._patches(x))  # [B, N, d_model]


# --------------------------------------------------------------------------- #
#  Harness adapter
# --------------------------------------------------------------------------- #
def _resample_operator(T_in: int, hz_in: float, hz_out: float) -> np.ndarray:
    """Matrix R [T_out, T_in] with y = R @ x == scipy.signal.resample_poly(x, up, down)."""
    from fractions import Fraction
    from scipy.signal import resample_poly
    fr = Fraction(hz_out / hz_in).limit_denominator(1000)
    up, down = fr.numerator, fr.denominator
    R = resample_poly(np.eye(T_in, dtype=np.float64), up, down, axis=0)
    return R.astype(np.float32)


class Inertia1Encoder(nn.Module):
    """ViT1DEncoder-compatible wrapper around an Inertia-1 backbone.

    Config block (``cfg["inertia1"]``):
        variant:            "patchtst" | "ar_transformer"
        preset:             "small" | "medium" | "large"   (default small; or set d_model/n_layers/n_heads/d_ff)
        patch_len:          10   stride: 10   mask_ratio: 0.4   pretrain_window_sec: 30
        input_hz:           sampling rate of the batches this run feeds (default 50)
        acc_scale_to_g:     multiplier taking the batch's acc units to g (default 1/9.80665 for m/s^2)
        acc_channels:       indices of the 3 acc channels in the batch (default [0,1,2])
        native_input:       true when batches are ALREADY 20 Hz g-unit windows (pretraining) -> no prep
    """

    def __init__(self, cfg: Any):
        super().__init__()
        icfg: Dict[str, Any] = dict((cfg or {}).get("inertia1", {}) or {})
        self.variant = str(icfg.get("variant", "patchtst")).lower()
        dims = dict(PRESETS[str(icfg.get("preset", "small")).lower()])
        for k in ("d_model", "n_layers", "n_heads", "d_ff"):
            if k in icfg:
                dims[k] = int(icfg[k])
        patch_len = int(icfg.get("patch_len", 10))
        pre_win = float(icfg.get("pretrain_window_sec", 30))
        self.model_hz = float(icfg.get("model_hz", MODEL_HZ))
        num_patch = 1 + (int(round(pre_win * self.model_hz)) - patch_len) // int(icfg.get("stride", patch_len))

        if self.variant in ("ar_transformer", "art"):
            self.variant = "ar_transformer"
            self.net = ARTransformerNet(channels=3, patch_len=patch_len, dropout=float(icfg.get("dropout", 0.0)), **dims)
        elif self.variant == "patchtst":
            self.net = PatchTSTNet(
                patch_len=patch_len, stride=int(icfg.get("stride", patch_len)),
                mask_ratio=float(icfg.get("mask_ratio", 0.4)), num_patch=num_patch,
                dropout=float(icfg.get("dropout", 0.0)), attn_dropout=float(icfg.get("attn_dropout", 0.0)),
                res_attention=bool(icfg.get("res_attention", True)), pre_norm=bool(icfg.get("pre_norm", True)),
                pe=str(icfg.get("pe", "zeros")), learn_pe=bool(icfg.get("learn_pe", True)), **dims)
        else:
            raise ValueError(f"inertia1.variant must be patchtst|ar_transformer, got {self.variant!r}")

        # ---- harness-facing attributes (mirror ViT1DEncoder) -------------------
        self.embed_dim = int(dims["d_model"])
        self.in_channels = 1                       # one fused acc token stream (token-grid semantics for decoders)
        self.num_input_channels = 3                # raw signal channels consumed (acc x/y/z) -- for input-truncation code
        vit_cfg = (cfg or {}).get("vit1d", {}) or {}
        self.patch_size = int(vit_cfg.get("patch_size", 16))   # harness token grid only
        self.use_cls_token = False
        self.num_freq_bands = 1

        # ---- input adaptation ----------------------------------------------------
        self.native_input = bool(icfg.get("native_input", False))
        self.input_hz = float(icfg.get("input_hz", 50))
        self.acc_scale_to_g = float(icfg.get("acc_scale_to_g", 1.0 / G_MPS2))
        # `sensor_channels` generalises `acc_channels`: the Inertia-1 backbone is accelerometer-only
        # by construction (PatchTST over a fused triaxial stream), but for the architecture-vs-modality
        # 2x2 we also need an acc+gyro variant. Passing 6 indices keeps the same backbone and simply
        # feeds it 6 channels; only the accelerometer indices get the g-unit rescale.
        self.acc_channels = [int(c) for c in icfg.get("sensor_channels", icfg.get("acc_channels", [0, 1, 2]))]
        assert len(self.acc_channels) in (3, 6), f"expected 3 or 6 channels, got {len(self.acc_channels)}"
        self.n_acc_in = 3
        self._six_channel = len(self.acc_channels) == 6
        if self._six_channel:
            if not hasattr(self.net, "build_W_P6"):
                raise ValueError("6-channel mode is only implemented for the patchtst variant")
            self.net.build_W_P6()
        self.token_patch_len = patch_len
        self._resamplers: Dict[int, torch.Tensor] = {}

    # ---- input prep ---------------------------------------------------------------
    def _resampler(self, T: int, device, dtype) -> torch.Tensor:
        R = self._resamplers.get(T)
        if R is None or R.device != device:
            R = torch.from_numpy(_resample_operator(T, self.input_hz, self.model_hz)).to(device)
            self._resamplers[T] = R
        return R.to(dtype)

    def prep_input(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] harness batch -> [B, 3, T20] g-unit 20 Hz signal."""
        if self.native_input:
            return x.float()
        if x.dim() != 3 or x.shape[1] < 3:
            raise ValueError(f"Inertia1Encoder expects [B, C>=3, T], got {tuple(x.shape)}")
        x = x[:, self.acc_channels, :].float()
        scale = torch.ones(x.shape[1], device=x.device, dtype=x.dtype)
        scale[: self.n_acc_in] = self.acc_scale_to_g        # gyro (if present) keeps rad/s
        x = x * scale.view(1, -1, 1)
        if abs(self.input_hz - self.model_hz) > 1e-6:
            R = self._resampler(x.shape[-1], x.device, x.dtype)   # [T20, T]
            x = torch.matmul(x, R.t())                            # [B, 3, T20]
        return x

    # ---- ViT1DEncoder interface -----------------------------------------------------
    def native_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] -> [B, N, D] Inertia-1 patch tokens (N = T20 // patch_len)."""
        x20 = self.prep_input(x)
        if x20.shape[-1] < self.token_patch_len:
            raise ValueError(f"window too short for one Inertia-1 patch: T20={x20.shape[-1]}")
        # Their modules run in fp32; keep the backbone in fp32 under harness autocast
        # (their probe code notes ART attention is bf16-sensitive).
        with torch.autocast(device_type=x20.device.type, enabled=False):
            return self.net.encode_tokens(x20.float())

    def forward(self, x: torch.Tensor, mask=None, padding_mask=None, return_keep_padding_mask: bool = False):
        if mask is not None or padding_mask is not None:
            raise NotImplementedError("Inertia1Encoder: masked / padded forward is not supported (probe/finetune only)")
        return self.native_tokens(x).mean(dim=1)  # [B, D]  == their encode()

    def forward_tokens(self, x: torch.Tensor):
        """Return (tokens [B, Pt, D], Pt) on the harness grid Pt = T // vit1d.patch_size."""
        toks = self.native_tokens(x)                       # [B, N, D]
        Pt = max(1, int(x.shape[-1]) // self.patch_size)
        if toks.shape[1] != Pt:
            toks = F.interpolate(toks.transpose(1, 2), size=Pt, mode="linear", align_corners=False).transpose(1, 2)
        return toks.contiguous(), Pt

    # ---- checkpoint loading (bypasses run.py's ViT1D key remap) --------------------
    def load_pretrained_state(self, enc_state: Dict[str, Any], strict: bool = True) -> None:
        sd = enc_state["model"] if isinstance(enc_state, dict) and "model" in enc_state else enc_state
        if any(k.startswith("net.") for k in sd):
            target, sd_use = self, {k: v for k, v in sd.items() if k.startswith("net.")}
        else:
            # raw Inertia-1 Lightning state_dict (patch_proj.* / model.backbone.* ...)
            target, sd_use = self.net, {k: v for k, v in sd.items()
                                        if not k.startswith(("channel_adapter",))}
        res = target.load_state_dict(sd_use, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"Inertia1Encoder: checkpoint missing keys {res.missing_keys}")
        if res.unexpected_keys:
            import logging
            logging.getLogger(__name__).warning("[inertia1] ignoring unexpected checkpoint keys: %s", res.unexpected_keys)


def is_inertia1_cfg(cfg: Any) -> bool:
    return isinstance(cfg, dict) and bool(cfg.get("inertia1"))


def build_encoder(cfg: Any) -> nn.Module:
    """Encoder factory used by run.py and the eval scripts: Inertia-1 adapter when
    the config carries an ``inertia1:`` block, else the ViT1D encoders exactly as
    before (alt-attn variant when vit1d.encoder.alt_attention is set)."""
    if is_inertia1_cfg(cfg):
        return Inertia1Encoder(cfg)
    from imu_lm.utils.helpers import cfg_get
    from imu_lm.models.ViT1D.model import build_dual_stream, maybe_stats_wrap
    if (cfg_get(cfg, ["dual_stream"], None) or {}).get("enabled", False):
        return maybe_stats_wrap(build_dual_stream(cfg), cfg)
    if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
        from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
        return maybe_stats_wrap(ViT1DEncoderAltAttn(cfg), cfg)
    from imu_lm.models.ViT1D.model import ViT1DEncoder
    return maybe_stats_wrap(ViT1DEncoder(cfg), cfg)


# --------------------------------------------------------------------------- #
#  Config conversion (shared by baselines/inertia1/scripts/make_configs.py and bench/make_config.py)
# --------------------------------------------------------------------------- #
OPENWATCH_LSB_PER_G = 4096.0


def inertia1_downstream_config(cfg: Dict[str, Any], variant: str, pretrained_run: str, *,
                               openwatch: bool = False, clip: bool = False,
                               pretrained_ckpt: str = "best", preset: str = "small") -> Dict[str, Any]:
    """Rewrite a ViT1D downstream template so that it trains/probes an Inertia-1
    encoder instead: ONLY the encoder side changes.

      vit1d.encoder / shared_kernel / pos_embed_type / use_cls_token / max_channels -> removed
      inertia1: {variant, preset (small|medium|large), input_hz, acc_scale_to_g, ...}
      objective.<type>.pretrained_run/ckpt -> the Inertia-1 pretraining run
      data.acc_only: true + acc sensor_columns    (Inertia-1 is accelerometer-only)
      preprocess.filter/normalize.enabled: false  (raw g-scale input as in Inertia-1)
      openwatch: input_hz 100, 4096 LSB/g, window_pad_len 100 (no zero padding), patch_size 20
      clip:      humanml3d.zscore false (var-len loader's own z-score switch)
    """
    import copy as _copy
    cfg = _copy.deepcopy(cfg)
    vit = cfg.setdefault("vit1d", {}) or {}
    cfg["vit1d"] = vit
    for k in ("encoder", "shared_kernel", "pos_embed_type", "use_cls_token", "max_channels"):
        vit.pop(k, None)
    vit.setdefault("patch_size", 16)
    vit.setdefault("max_patches_per_channel", 256)
    vit["pooling"] = "mean"

    variant = "ar_transformer" if variant in ("art", "ar_transformer") else variant
    if str(preset).lower() not in PRESETS:
        raise ValueError(f"unknown Inertia-1 preset {preset!r}; choose from {sorted(PRESETS)}")
    inertia = {"variant": variant, "preset": str(preset).lower(), "input_hz": 50, "acc_scale_to_g": 1.0 / G_MPS2,
               "acc_channels": [0, 1, 2], "patch_len": 10, "stride": 10, "mask_ratio": 0.4,
               "pretrain_window_sec": 30, "model_hz": MODEL_HZ}
    if openwatch:
        inertia["input_hz"] = 100
        inertia["acc_scale_to_g"] = 1.0 / OPENWATCH_LSB_PER_G
        vit["patch_size"] = 20   # only constrains window_pad_len (multiple of patch); token grid unused for classification
    cfg["inertia1"] = inertia

    obj = cfg["objective"]
    blk = obj.setdefault(obj["type"], {}) or {}
    obj[obj["type"]] = blk
    blk["pretrained_run"] = pretrained_run
    blk["pretrained_ckpt"] = pretrained_ckpt
    if openwatch:
        blk["window_pad_len"] = 100   # 100 samples @100 Hz -> 20 @20 Hz -> 2 Inertia-1 patches, no zeros

    data = cfg.setdefault("data", {}) or {}
    cfg["data"] = data
    cols = data.get("sensor_columns") or ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    data["sensor_columns"] = list(cols[:3])
    data["acc_only"] = True

    pp = cfg.setdefault("preprocess", {}) or {}
    cfg["preprocess"] = pp
    pp.setdefault("impute", {"enabled": True, "method": "linear", "max_missing_frac": 0.2})
    pp["filter"] = {"enabled": False}
    pp["normalize"] = {"enabled": False, "method": "zscore", "eps": 1.0e-6}
    if clip:
        cfg.setdefault("humanml3d", {})["zscore"] = False
    return cfg
