"""DeepConvLSTM Stage A wiring: loaders -> encoder + head -> objective -> trainer -> artifacts.

Supervised only: this repo has no self-supervised pretraining variant of
DeepConvLSTM, so it is used purely as a from-scratch supervised HAR baseline
(same role as the ViT1D "Scratch" columns, but with the classic
conv+LSTM architecture from Ordonez & Roggen 2016 instead).
"""

from __future__ import annotations

from typing import Any, Optional

from imu_lm.data.loaders import make_loaders
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.DeepConvLSTM.model import DeepConvLSTMEncoder
from imu_lm.objectives import supervised as supervised_obj
from imu_lm.probe.head import LinearHead
from imu_lm.probe.eval import eval_head
from imu_lm.runtime_consistency.artifacts import save_label_map, save_meta, save_supervised_model
from imu_lm.runtime_consistency.trainer import Trainer
from imu_lm.utils.helpers import cfg_get
from imu_lm.utils.metrics import format_metrics_summary
from imu_lm.utils.training import (
    resolve_resume_path,
    build_optimizer_from_params,
    build_scheduler,
    build_label_map,
    load_checkpoint,
)


def main(cfg: Any, run_dir: str, resume_ckpt: Optional[str] = None):
    T, _ = compute_T_and_hop(cfg)

    objective_type = cfg_get(cfg, ["objective", "type"], "supervised").lower()
    if objective_type != "supervised":
        raise ValueError(
            f"DeepConvLSTM only supports objective.type=supervised, got {objective_type!r}"
        )

    encoder = DeepConvLSTMEncoder(cfg)

    data_cfg = cfg_get(cfg, ["data"], {}) or {}
    sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
    meta = {
        "embedding_dim": encoder.embed_dim,
        "encoding": "raw_1d_imu",
        "objective": objective_type,
        "backbone": "deepconvlstm",
        "input_spec": {
            "channels": len(sensor_cols),
            "time_steps": T,
            "format": "[B, C, T]",
        },
        "architecture": {
            "conv_filters": encoder.conv_filters,
            "kernel_size": encoder.kernel_size,
            "lstm_hidden_size": encoder.embed_dim,
        },
        "normalization": cfg_get(cfg, ["preprocess", "normalize", "method"], None),
    }
    save_meta(meta, run_dir)

    loaders = make_loaders(cfg)
    label_map = build_label_map(loaders["train_loader"], cfg)
    save_label_map(label_map, run_dir)
    raw_to_idx = label_map["raw_to_idx"]
    num_classes = label_map["num_classes"]
    head = LinearHead(encoder.embed_dim, num_classes)

    def objective_fn(batch, model, cfg):
        return supervised_obj.forward_loss(batch, encoder, head, cfg, raw_to_idx=raw_to_idx)

    all_params = list(encoder.parameters()) + list(head.parameters())

    train_loader = loaders.get("train_loader")
    val_loader = loaders.get("val_loader")

    optimizer = build_optimizer_from_params(all_params, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    extra_modules = {"head": head}

    resume_path = resolve_resume_path(run_dir, resume_ckpt)
    start_step = load_checkpoint(resume_path, encoder, optimizer, scheduler, extra_modules=extra_modules)

    trainer = Trainer(cfg, run_dir)
    trainer.fit(encoder, objective_fn, train_loader, val_loader, optimizer, scheduler, start_step=start_step, extra_modules=extra_modules)

    meta["num_classes"] = num_classes
    save_supervised_model(encoder, head, meta, run_dir, label_map=label_map)

    import json, os, torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if os.path.exists(best_path):
        best_state = torch.load(best_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(best_state["model"])
        if "head" in best_state:
            head.load_state_dict(best_state["head"])
        print(f"[eval] loaded best checkpoint from step {best_state.get('step', '?')}")

    encoder.to(device)
    head.to(device)

    eval_results = {}
    for split_name, loader in [("val", val_loader), ("test", loaders.get("test_loader"))]:
        if loader is None:
            continue
        metrics = eval_head(encoder, head, loader, label_map, device)
        eval_results[split_name] = metrics
        print(f"[eval] {split_name}: {format_metrics_summary(metrics)}")

    summary_path = os.path.join(run_dir, "summary.json")
    summary = {
        "objective": objective_type,
        "num_classes": num_classes,
    }
    for split_name, metrics in eval_results.items():
        summary[split_name] = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] summary written to {summary_path}")


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_pretrain.py to launch training.")
