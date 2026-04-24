"""
run_pretrain.py
----------------
Stage A entrypoint: load/merge configs, create run dir, route to model run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import logging
from typing import Any, Dict

from imu_lm.models.ViT2D import run as vit_run
from imu_lm.models.ViT1D import run as vit1d_run
from imu_lm.models.CNN1D import run as cnn_run
from imu_lm.models.TSTransformer1D import run as tstransformer1d_run
from imu_lm.models.RNN1D import run as rnn1d_run
from imu_lm.utils.helpers import deep_update, load_yaml
try:
    import wandb
except ImportError:
    wandb = None


def _resolve_run_dir(cfg: Dict[str, Any], run_name: str | None) -> str:
    runs_root = cfg.get("paths", {}).get("runs_root", "runs")
    if run_name is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"run-{ts}"
    run_dir = os.path.join(runs_root, run_name)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=True)
    return run_dir


def _log_resolved(cfg: Dict[str, Any], run_dir: str):
    log_path = os.path.join(run_dir, "logs", "stdout.log")
    with open(log_path, "a") as f:
        f.write("# Resolved config\n")
        f.write(json.dumps(cfg, indent=2))
        f.write("\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Base config YAML")
    ap.add_argument("--model-config", required=True, help="Model config YAML")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None, help="Checkpoint path or name (latest/best) to resume from")
    args = ap.parse_args()

    base_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    cfg = deep_update(base_cfg, model_cfg)

    run_dir = _resolve_run_dir(cfg, args.run_name)
    _log_resolved(cfg, run_dir)

    # Minimal wandb init (uses wandb.project / wandb.entity from config)
    if wandb is not None:
        wb_cfg = cfg.get("wandb", {}) or {}
        try:
            wandb.init(
                project=wb_cfg.get("project", "imu-lm"),
                entity=wb_cfg.get("entity", None),
                name=args.run_name or os.path.basename(run_dir),
                config=cfg,
                dir=run_dir,
            )
        except Exception as e:
            logging.getLogger(__name__).warning("wandb init failed: %s", e)

    # Detect model type by config keys
    if "cnn1d" in cfg:
        cnn_run.main(cfg, run_dir, resume_ckpt=args.resume)
    elif "rnn1d" in cfg:
        rnn1d_run.main(cfg, run_dir, resume_ckpt=args.resume)
    elif "vit1d" in cfg:
        vit1d_run.main(cfg, run_dir, resume_ckpt=args.resume)
    elif "tstransformer1d" in cfg:
        tstransformer1d_run.main(cfg, run_dir, resume_ckpt=args.resume)
    elif "vit" in cfg:
        vit_run.main(cfg, run_dir, resume_ckpt=args.resume)
    else:
        raise ValueError("No supported model config found (expected 'vit', 'vit1d', 'tstransformer1d', 'rnn1d', or 'cnn1d' section)")


if __name__ == "__main__":
    main()
