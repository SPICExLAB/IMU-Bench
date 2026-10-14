"""Any PyTorch model as an IMU-Bench encoder (the open track's entry point).

An encoder card with ``external_encoder: plugin`` names a builder function; the harness wraps the model
it returns so that every task card (decoders, recipes, splits, scorers) runs on it unchanged.

Encoder card (bench/registry/encoders.yaml, or one file in bench/registry/encoders.d/)::

    my_encoder:
      name: My encoder (Doe et al., 2026)
      external_encoder: plugin
      entry: my_package.imubench:build           # callable(weights=<path or None>, **kwargs) -> nn.Module
      kwargs: {size: base}                       # optional, passed to the builder
      weights: hf://my-org/my-model/encoder.pt   # local path, hf://<repo>/<file>, or "builtin"
      pythonpath: /path/to/my/code               # optional, prepended to sys.path
      channels: 6                                # 6 = acc + gyro, 3 = accelerometer only
      model_hz: 50                               # rate the model expects; the harness resamples
      acc_units: m/s2                            # m/s2 or g; gyro is always rad/s
      normalize_method: zscore                   # what the harness applies first: zscore | physical | raw
      params_m: 21
      pretrain_data: ...                         # declared corpus, audited for evaluation overlap
      track: open

Model contract: ``model(x)`` with ``x`` a float tensor ``[B, C, T]`` (C = channels; acc x/y/z then gyro x/y/z
in a forward-left-up wrist frame, at ``model_hz``) returns ``[B, N, D]``, N tokens evenly spaced over the
window, or ``[B, D]``, one pooled vector, or a dict with both (``{"tokens": [B, N, D], "pooled": [B, D]}``,
for a model whose summary vector is not the token mean, e.g. a CLS token); ``model.embed_dim`` is D. Token outputs are interpolated
onto the harness grid of one token per 0.32 s, which the pose and odometry decoders read; classification
and retrieval read the mean token (or the pooled vector). A pooled-only model gets that vector repeated
on every token, so its kinematic results carry no within-window temporal information.

``weights: builtin`` means the builder loads its own weights (torch.hub, a package download). The harness
still needs a checkpoint file to point the training branches at; make_config writes an empty marker.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

G_MPS2 = 9.80665
BUILTIN_MARK = "imubench_plugin_builtin"


def _import_entry(entry: str, pythonpath: Optional[str] = None):
    if pythonpath:
        for p in str(pythonpath).split(os.pathsep):
            if p and p not in sys.path:
                sys.path.insert(0, p)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if root not in sys.path:                      # entries that live in the repository (examples/...)
        sys.path.append(root)
    if ":" not in entry:
        raise ValueError(f"plugin entry must look like 'package.module:function', got {entry!r}")
    mod, fn = entry.split(":", 1)
    return getattr(importlib.import_module(mod), fn)


def resolve_weights(spec: Optional[str], cache_dir: Optional[str] = None) -> Optional[str]:
    """Local path for a card's `weights`: a path, hf://<org>/<repo>/<file>, or None/"builtin"."""
    if spec in (None, "", "builtin"):
        return None
    spec = str(spec)
    if spec.startswith("hf://"):
        from huggingface_hub import hf_hub_download
        parts = spec[len("hf://"):].split("/")
        if len(parts) < 3:
            raise ValueError(f"hf weights must be hf://<org>/<repo>/<file>, got {spec!r}")
        return hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]), cache_dir=cache_dir)
    return os.path.abspath(os.path.expanduser(spec))


class PluginEncoder(nn.Module):
    """ViT1DEncoder-compatible wrapper around a user model (one fused token stream)."""

    def __init__(self, cfg: Any):
        super().__init__()
        pcfg: Dict[str, Any] = dict((cfg or {}).get("plugin", {}) or {})
        build = _import_entry(pcfg["entry"], pcfg.get("pythonpath"))
        self.model = build(weights=pcfg.get("weights_path"), **(pcfg.get("kwargs") or {}))
        if not hasattr(self.model, "embed_dim"):
            raise AttributeError(f"plugin {pcfg['entry']}: the returned model needs an integer `embed_dim`")

        # ---- harness-facing attributes (mirror ViT1DEncoder) -----------------------
        self.embed_dim = int(self.model.embed_dim)
        self.in_channels = 1
        self.num_input_channels = int(pcfg.get("channels", 6))
        self.patch_size = int(((cfg or {}).get("vit1d", {}) or {}).get("patch_size", 16))
        self.use_cls_token = False
        self.num_freq_bands = 1

        # ---- input adaptation -------------------------------------------------------
        self.input_hz = float(pcfg.get("input_hz", 50))
        self.model_hz = float(pcfg.get("model_hz", self.input_hz))
        self.acc_gain = 1.0 / G_MPS2 if str(pcfg.get("acc_units", "m/s2")).lower() == "g" else 1.0
        self._resamplers: Dict[int, torch.Tensor] = {}

    def prep_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, : self.num_input_channels].float()
        if self.acc_gain != 1.0:
            x = torch.cat([x[:, :3] * self.acc_gain, x[:, 3:]], dim=1)
        if self.model_hz < self.input_hz - 1e-6:          # downsample: anti-aliased polyphase filter
            from imu_lm.models.baselines.harnet_encoder import _resample_operator
            key = int(x.shape[-1])
            R = self._resamplers.get(key)
            if R is None or R.device != x.device:
                R = torch.from_numpy(_resample_operator(key, self.input_hz, self.model_hz)).to(x.device)
                self._resamplers[key] = R
            x = torch.matmul(x, R.t().to(x.dtype))
        elif self.model_hz > self.input_hz + 1e-6:        # upsample: linear interpolation
            T = max(1, int(round(x.shape[-1] * self.model_hz / self.input_hz)))
            x = F.interpolate(x, size=T, mode="linear", align_corners=False)
        return x

    def _call(self, x: torch.Tensor):
        """-> (tokens [B, N, D] or None, pooled [B, D] or None) from whatever the model returns."""
        out = self.model(self.prep_input(x))
        if isinstance(out, dict):
            return out.get("tokens"), out.get("pooled")
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() == 2:
            return None, out
        if out.dim() == 3:
            return out, None
        raise ValueError(f"plugin model must return [B, D], [B, N, D] or a dict, got {tuple(out.shape)}")

    def forward_tokens(self, x: torch.Tensor):
        """(tokens [B, Pt, D], Pt) on the harness grid Pt = T // patch_size."""
        Pt = max(1, int(x.shape[-1]) // self.patch_size)
        toks, pooled = self._call(x)
        out = toks if toks is not None else pooled
        if out.dim() == 2:
            return out.unsqueeze(1).expand(out.shape[0], Pt, out.shape[1]).contiguous(), Pt
        if out.shape[1] != Pt:
            out = F.interpolate(out.transpose(1, 2), size=Pt, mode="linear", align_corners=False).transpose(1, 2)
        return out.contiguous(), Pt

    def forward(self, x: torch.Tensor, mask=None, padding_mask=None, return_keep_padding_mask: bool = False):
        if mask is not None or padding_mask is not None:
            raise NotImplementedError("PluginEncoder: masked / padded forward is not supported (probe/finetune only)")
        toks, pooled = self._call(x)
        return pooled if pooled is not None else toks.mean(dim=1)

    def load_pretrained_state(self, enc_state: Dict[str, Any], strict: bool = True) -> None:
        """Called by the training branches with the file `pretrained_ckpt` names. For a plugin that is the
        builtin marker or the card's `weights` file, which the builder has already loaded in its own
        format; it is loaded again here only if its keys are this model's keys (e.g. a plain state dict)."""
        if isinstance(enc_state, dict) and enc_state.get(BUILTIN_MARK):
            return
        sd = enc_state
        for k in ("model", "state_dict", "encoder"):
            if isinstance(sd, dict) and isinstance(sd.get(k), dict):
                sd = sd[k]
                break
        if not isinstance(sd, dict):
            return
        own = self.model.state_dict()
        if any(k.startswith("model.") for k in sd) and not any(k.startswith("model.") for k in own):
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        hit = [k for k in sd if k in own]
        if not hit:
            return                       # the builder's own format; it loaded these weights already
        missing = [k for k in own if k not in sd]
        if missing and strict:
            raise RuntimeError(f"PluginEncoder: the weights file matches the model only partly; {len(missing)} "
                               f"key(s) missing, e.g. {missing[:6]}")
        self.model.load_state_dict({k: sd[k] for k in hit}, strict=False)


def is_plugin_cfg(cfg: Any) -> bool:
    return isinstance(cfg, dict) and bool(cfg.get("plugin"))


def plugin_downstream_config(cfg: Dict[str, Any], card: Dict[str, Any], weights_path: Optional[str], *,
                             openwatch: bool = False, clip: bool = False) -> Dict[str, Any]:
    """Rewrite a ViT1D downstream template so it probes / fine-tunes a plugin encoder.
    Only the encoder side changes; task recipe, protocol and scorer stay the harness' own."""
    cfg = dict(cfg)
    vit = dict(cfg.get("vit1d", {}) or {})
    for k in ("encoder", "shared_kernel", "pos_embed_type", "use_cls_token", "max_channels"):
        vit.pop(k, None)
    vit.setdefault("patch_size", 16)
    vit.setdefault("max_patches_per_channel", 256)
    vit["pooling"] = "mean"
    cfg["vit1d"] = vit
    ch = int(card.get("channels", 6))
    cfg["plugin"] = {
        "entry": card["entry"], "kwargs": dict(card.get("kwargs") or {}), "weights_path": weights_path,
        "pythonpath": card.get("pythonpath"), "channels": ch,
        "input_hz": 100 if openwatch else 50, "model_hz": float(card.get("model_hz", 100 if openwatch else 50)),
        "acc_units": str(card.get("acc_units", "m/s2")),
    }
    if ch == 3:
        data = dict(cfg.get("data", {}) or {})
        cols = data.get("sensor_columns") or ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        data["sensor_columns"] = list(cols[:3])
        data["acc_only"] = True
        cfg["data"] = data
    if str(card.get("normalize_method", "zscore")) == "raw":
        # raw = the signal as the loader delivers it (m/s^2 with gravity, rad/s): no band filter either,
        # which would remove gravity (make_config switches normalisation off for `raw` cards).
        pp = dict(cfg.get("preprocess", {}) or {})
        pp["filter"] = {"enabled": False}
        cfg["preprocess"] = pp
    if clip and str(card.get("normalize_method", "zscore")) != "zscore":
        cfg.setdefault("humanml3d", {})["zscore"] = False
    return cfg
