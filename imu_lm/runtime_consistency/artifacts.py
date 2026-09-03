"""Save/load encoder artifacts and metadata."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import torch


def artifact_paths(run_dir: str) -> Dict[str, str]:
    art_dir = os.path.join(run_dir, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    return {
        "dir": art_dir,
        "encoder": os.path.join(art_dir, "encoder.pt"),
        "meta": os.path.join(art_dir, "encoder_meta.json"),
    }


def save_meta(meta: Dict[str, Any], run_dir: str):
    """Save encoder metadata JSON (architecture, input spec). Call once at run start."""
    paths = artifact_paths(run_dir)
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[artifact] saved encoder_meta to {paths['meta']}")


def load_meta(run_dir: str) -> Dict[str, Any]:
    paths = artifact_paths(run_dir)
    with open(paths["meta"], "r") as f:
        return json.load(f)


def save_label_map(label_map: Dict[str, Any], run_dir: str):
    """Save label_map.json to artifacts/. Call right after build_label_map()."""
    paths = artifact_paths(run_dir)
    label_map_path = os.path.join(paths["dir"], "label_map.json")
    serialisable = {k: v for k, v in label_map.items() if k != "unknown_id" or v is not None}
    with open(label_map_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[artifact] saved label_map to {label_map_path}")


def save_encoder(encoder: torch.nn.Module, meta: Dict[str, Any], run_dir: str):
    """Save full pickled encoder + meta. Legacy; prefer save_meta + checkpoints."""
    paths = artifact_paths(run_dir)
    torch.save(encoder, paths["encoder"])
    save_meta(meta, run_dir)
    print(f"[artifact] saved encoder to {paths['encoder']}")


def _build_encoder_from_cfg(backbone: str, cfg: Any):
    """Reconstruct an encoder from backbone name + config."""
    if backbone == "vit1d":
        from imu_lm.models.ViT1D.model import ViT1DEncoder
        return ViT1DEncoder(cfg)
    elif backbone == "inertia1":
        from imu_lm.models.baselines.inertia1_encoder import Inertia1Encoder
        return Inertia1Encoder(cfg)
    elif backbone in ("vit", "vit2d"):
        from imu_lm.models.ViT2D.model import ViTEncoder
        return ViTEncoder(cfg)
    elif backbone == "cnn1d":
        from imu_lm.models.CNN1D.model import CNN1DEncoder
        return CNN1DEncoder(cfg)
    elif backbone == "tstransformer1d":
        from imu_lm.models.TSTransformer1D.model import TSTransformer1DEncoder
        return TSTransformer1DEncoder(cfg)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def load_encoder(run_dir: str, ckpt_name: str = "best", map_location=None):
    """Load encoder from run directory.

    Prefers reconstruction from meta + checkpoint state_dict.
    Falls back to legacy pickled encoder.pt if meta is missing.
    """
    paths = artifact_paths(run_dir)

    if os.path.exists(paths["meta"]):
        meta = load_meta(run_dir)
        backbone = meta.get("backbone")
        ckpt_path = os.path.join(run_dir, "checkpoints", f"{ckpt_name}.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(run_dir, "checkpoints", "latest.pt")
        state = torch.load(ckpt_path, map_location=map_location or "cpu", weights_only=False)
        cfg = state["cfg"]
        encoder = _build_encoder_from_cfg(backbone, cfg)
        encoder.load_state_dict(state["model"], strict=True)
        print(f"[artifact] reconstructed {backbone} encoder from {ckpt_path}")
        return encoder

    if os.path.exists(paths["encoder"]):
        encoder = torch.load(paths["encoder"], map_location=map_location, weights_only=False)
        print(f"[artifact] loaded legacy pickled encoder from {paths['encoder']}")
        return encoder

    raise FileNotFoundError(f"No encoder meta or encoder.pt found in {run_dir}/artifacts/")


def save_supervised_model(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    meta: Dict[str, Any],
    run_dir: str,
    label_map: Dict[str, Any] | None = None,
):
    """Save encoder + classification head + label_map for fully supervised training."""
    paths = artifact_paths(run_dir)
    torch.save(encoder, paths["encoder"])
    head_path = os.path.join(paths["dir"], "head.pt")
    torch.save(head, head_path)
    model_path = os.path.join(paths["dir"], "model.pt")
    torch.save({"encoder": encoder, "head": head}, model_path)
    if label_map is not None:
        label_map_path = os.path.join(paths["dir"], "label_map.json")
        with open(label_map_path, "w") as f:
            json.dump({k: v for k, v in label_map.items() if k != "unknown_id" or v is not None}, f, indent=2)
        meta["label_map_path"] = "label_map.json"
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[artifact] saved encoder to {paths['encoder']}")
    print(f"[artifact] saved head to {head_path}")
    print(f"[artifact] saved combined model to {model_path}")
    print(f"[artifact] saved meta to {paths['meta']}")
