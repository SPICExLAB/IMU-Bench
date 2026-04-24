"""RNN1D Stage A wiring: loaders → encoder + head/decoder → objective → trainer → artifacts.

Supports two objectives:
- MAE (masked): encoder + decoder, self-supervised reconstruction
- Supervised: encoder + LinearHead, fully-supervised classification
"""

from __future__ import annotations

from typing import Any, Optional

from imu_lm.data.loaders import make_loaders
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.RNN1D.model import RNN1DEncoder, RNN1DDecoder
from imu_lm.objectives import masked_1d as masked_1d_obj
from imu_lm.objectives import supervised as supervised_obj
from imu_lm.probe.head import LinearHead
from imu_lm.runtime_consistency.artifacts import save_encoder, save_label_map, save_meta, save_supervised_model
from imu_lm.runtime_consistency.trainer import Trainer
from imu_lm.utils.helpers import cfg_get
from imu_lm.utils.training import (
    resolve_resume_path,
    build_optimizer_from_params,
    build_scheduler,
    build_label_map,
    load_checkpoint,
)


def _make_mae_loaders(cfg: Any, objective_type: str):
    """Build train/val DataLoaders for RNN1D MAE.

    For 'mae_realdata_nymeria_real_wrist_flu': ConcatDataset of
    Nymeria real-wrist FLU pkl sessions and real-6D parquet windows.
    Otherwise: standard make_loaders(cfg) using the parquet dataset.
    """
    if objective_type != "mae_realdata_nymeria_real_wrist_flu":
        return make_loaders(cfg)

    import glob as _glob
    import logging as _logging
    import os as _os
    import torch as _torch
    from imu_lm.data.nymeria_real_wrist_loader_FLU import NymeriaRealWristDatasetFLU
    from imu_lm.data.loaders import WindowDataset, build_session_index, make_splits
    from torch.utils.data import ConcatDataset, DataLoader as _DL
    _log = _logging.getLogger(__name__)

    rw_cfg         = cfg_get(cfg, ["objective", "mae_realdata_nymeria_real_wrist_flu"], {}) or {}
    train_pkl_root = rw_cfg.get("train_pkl_root")
    val_pkl_root   = rw_cfg.get("val_pkl_root") or train_pkl_root

    def _expand(root, files_cfg):
        if files_cfg is None or files_cfg == "all":
            return sorted(_glob.glob(_os.path.join(root, "*.pkl")))
        if isinstance(files_cfg, list):
            return [f if _os.path.isabs(f) else _os.path.join(root, f) for f in files_cfg]
        return [files_cfg if _os.path.isabs(files_cfg) else _os.path.join(root, files_cfg)]

    train_pkl_files = _expand(train_pkl_root, rw_cfg.get("train_files"))
    val_pkl_files   = _expand(val_pkl_root,   rw_cfg.get("val_files"))
    _add_gravity    = bool(rw_cfg.get("add_gravity", True))

    class _XOnly(_torch.utils.data.Dataset):
        def __init__(self, ds):
            self._ds = ds
        def __len__(self):
            return len(self._ds)
        def __getitem__(self, idx):
            item = self._ds[idx]
            if item is None:
                return None
            return item[0], _torch.empty(0)

    def _collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return _torch.stack([b[0] for b in batch], 0), _torch.empty(0)

    rw_train_ds = _XOnly(NymeriaRealWristDatasetFLU(
        pkl_paths=train_pkl_files, cfg=cfg, split_name="train", add_gravity=_add_gravity
    ))
    rw_val_ds = _XOnly(NymeriaRealWristDatasetFLU(
        pkl_paths=val_pkl_files, cfg=cfg, split_name="val", add_gravity=_add_gravity
    ))

    parquet_path   = cfg_get(cfg, ["paths", "dataset_path"])
    train_on       = cfg_get(cfg, ["data", "train_on_datasets"], None)
    parquet_si     = build_session_index(parquet_path, cfg,
                                         dataset_filter=list(train_on) if train_on else None)
    parquet_splits = make_splits(parquet_si, cfg)
    parq_train_ds  = _XOnly(WindowDataset(
        parquet_path, parquet_si, parquet_splits["train_keys"], cfg, split_name="train"
    ))
    parq_val_ds    = _XOnly(WindowDataset(
        parquet_path, parquet_si, parquet_splits["val_keys"], cfg, split_name="val"
    ))

    combined_train = ConcatDataset([rw_train_ds, parq_train_ds])
    combined_val   = ConcatDataset([rw_val_ds,   parq_val_ds])
    _log.info(
        "[rnn1d mae_realdata_nymeria_real_wrist_flu] train=%d val=%d "
        "(rw_train=%d parq_train=%d rw_val=%d parq_val=%d)",
        len(combined_train), len(combined_val),
        len(rw_train_ds), len(parq_train_ds), len(rw_val_ds), len(parq_val_ds),
    )

    nw  = int(cfg_get(cfg, ["data", "num_workers"], 0))
    pin = bool(cfg_get(cfg, ["data", "pin_memory"], False))
    bs  = int(cfg_get(cfg, ["data", "batch_size"], 256))
    ebs = int(cfg_get(cfg, ["data", "eval_batch_size"], bs))
    per = nw > 0
    return {
        "train_loader": _DL(combined_train, batch_size=bs,  shuffle=True,  num_workers=nw,
                            pin_memory=pin, drop_last=True,  persistent_workers=per, collate_fn=_collate),
        "val_loader":   _DL(combined_val,   batch_size=ebs, shuffle=False, num_workers=nw,
                            pin_memory=pin, drop_last=False, persistent_workers=per, collate_fn=_collate),
    }


def main(cfg: Any, run_dir: str, resume_ckpt: Optional[str] = None):
    # Compute input temporal dimension from windowing config
    T, _ = compute_T_and_hop(cfg)

    # Get objective type: "mae" or "supervised"
    objective_type = cfg_get(cfg, ["objective", "type"], "mae").lower()

    # Create encoder (always needed)
    encoder = RNN1DEncoder(cfg)

    # Save architecture metadata at run start
    data_cfg = cfg_get(cfg, ["data"], {}) or {}
    sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
    rnn_cfg = cfg_get(cfg, ["rnn1d"], {}) or {}
    enc_cfg = rnn_cfg.get("encoder", {}) or {}
    meta = {
        "embedding_dim": encoder.embed_dim,
        "encoding": "raw_1d_imu",
        "objective": objective_type,
        "backbone": "rnn1d",
        "input_spec": {
            "channels": len(sensor_cols),
            "time_steps": T,
            "format": "[B, C, T]",
        },
        "architecture": {
            "hidden_size": encoder.hidden_size,
            "num_layers": encoder.num_layers,
            "rnn_type": encoder.rnn_type,
            "bidirectional": encoder.bidirectional,
            "pooling": encoder.pooling,
        },
        "normalization": cfg_get(cfg, ["preprocess", "normalize", "method"], None),
    }
    save_meta(meta, run_dir)

    if objective_type == "supervised":
        # Supervised: encoder + LinearHead, auto-discover classes from data
        loaders = make_loaders(cfg)
        label_map = build_label_map(loaders["train_loader"], cfg)
        save_label_map(label_map, run_dir)
        raw_to_idx = label_map["raw_to_idx"]
        num_classes = label_map["num_classes"]
        head = LinearHead(encoder.embed_dim, num_classes)

        def objective_fn(batch, model, cfg):
            return supervised_obj.forward_loss(batch, encoder, head, cfg, raw_to_idx=raw_to_idx)

        all_params = list(encoder.parameters()) + list(head.parameters())
    else:
        # MAE: encoder + decoder, self-supervised
        decoder = RNN1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )

        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)

        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders = _make_mae_loaders(cfg, objective_type)

    train_loader = loaders.get("train_loader")
    val_loader   = loaders.get("val_loader")

    optimizer = build_optimizer_from_params(all_params, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    if objective_type == "supervised":
        extra_modules = {"head": head}
    else:
        extra_modules = {"decoder": decoder}

    resume_path = resolve_resume_path(run_dir, resume_ckpt)
    start_step  = load_checkpoint(resume_path, encoder, optimizer, scheduler, extra_modules=extra_modules)

    trainer = Trainer(cfg, run_dir)
    trainer.fit(encoder, objective_fn, train_loader, val_loader, optimizer, scheduler, start_step=start_step, extra_modules=extra_modules)

    # Save final artifacts
    meta["embedding_dim"] = encoder.embed_dim

    if objective_type == "supervised":
        meta["num_classes"] = num_classes
        save_label_map(label_map, run_dir)
        save_supervised_model(encoder, head, meta, run_dir)
    else:
        save_encoder(encoder, meta, run_dir)
