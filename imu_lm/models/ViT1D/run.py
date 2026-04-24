"""ViT1D Stage A wiring: loaders → encoder + head/decoder → objective → trainer → artifacts.

Supports nine objectives:
- MAE (masked): encoder + ViT1DDecoder, self-supervised reconstruction
- Supervised: encoder + LinearHead, fully-supervised classification
- Pose: encoder + ViT1DPoseDecoder, per-patch 6D joint rotation regression
- Odom (odom_nymeria): encoder + ViT1DOdomDecoder, per-patch root displacement regression
- Pose probe (pose_probe): frozen pretrained encoder + trainable ViT1DPoseDecoder
- Odom probe (odom_probe): frozen pretrained encoder + trainable ViT1DOdomDecoder
- Odom finetune (odom_finetune): warm-started encoder (trainable) + trainable ViT1DOdomDecoder
- MAE var-len (mae_nymeria_var_len): MAE with sinusoidal pos-embed and random T per batch
- IMU-text CLIP (imu_text_clip): UniMTS-style contrastive pretraining on HumanML3D
"""

from __future__ import annotations

import os
import torch
from typing import Any, Optional

from imu_lm.data.loaders import make_loaders
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DDecoder, ViT1DPoseDecoder, ViT1DOdomDecoder
from imu_lm.objectives import masked_1d as masked_1d_obj
from imu_lm.objectives import supervised as supervised_obj
from imu_lm.objectives import pose_1d as pose_1d_obj
from imu_lm.objectives import odom_1d as odom_1d_obj
from imu_lm.probe.head import LinearHead
from imu_lm.runtime_consistency.artifacts import save_label_map, save_meta
from imu_lm.runtime_consistency.trainer import Trainer
from imu_lm.utils.helpers import cfg_get
from imu_lm.utils.training import (
    resolve_resume_path,
    build_optimizer_from_params,
    build_scheduler,
    build_label_map,
    load_checkpoint,
)


def main(cfg: Any, run_dir: str, resume_ckpt: Optional[str] = None):
    # Compute input temporal dimension from windowing config
    T, _ = compute_T_and_hop(cfg)
    
    # Get objective type: "mae", "supervised", or "pose"
    objective_type = cfg_get(cfg, ["objective", "type"], "mae").lower()
    
    # Create encoder (always needed)
    encoder = ViT1DEncoder(cfg)
    
    # Save architecture metadata at run start (static — doesn't change during training).
    # Weights are saved in checkpoints/best.pt and latest.pt by the trainer.
    data_cfg = cfg_get(cfg, ["data"], {}) or {}
    sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
    vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
    meta = {
        "embedding_dim": encoder.embed_dim,
        "encoding": "raw_1d_imu",
        "objective": objective_type,
        "backbone": "vit1d",
        "input_spec": {
            "channels": len(sensor_cols),
            "time_steps": T,
            "patch_size": encoder.patch_size,
            "format": "[B, C, T]",
        },
        "architecture": {
            "hidden_size": encoder.embed_dim,
            "num_layers": int(vit_cfg.get("encoder", {}).get("num_hidden_layers", 12)),
            "num_heads": int(vit_cfg.get("encoder", {}).get("num_attention_heads", 6)),
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
    elif objective_type == "pose":
        # Pose regression: encoder + ViT1DPoseDecoder, [B, P_t, J*6]
        # Use full-body loader (24 joints, 144 dims) when num_joints == 24,
        # otherwise use right-arm-chain loader (5 joints, 30 dims).
        vit_cfg  = cfg_get(cfg, ["vit1d"], {}) or {}
        pose_cfg = cfg_get(cfg, ["objective", "pose"], {}) or {}
        patch_size              = int(vit_cfg.get("patch_size", 16))
        num_patches_per_channel = T // patch_size
        num_joints              = int(pose_cfg.get("num_joints", 5))

        _loader_type = str(pose_cfg.get("loader", "")).lower()
        if _loader_type == "flu":
            from imu_lm.data.nymeria_wrist_loader_FLU import make_nymeria_loaders_flu as _make_pose_loaders
        elif _loader_type == "nymeria_real_wrist_flu":
            from imu_lm.data.nymeria_real_wrist_loader_FLU import make_nymeria_real_wrist_loaders_flu as _make_pose_loaders
        elif _loader_type == "imuposer_flu":
            from imu_lm.data.imuposer_wrist_loader_FLU import make_imuposer_wrist_flu_loaders as _make_pose_loaders
        elif _loader_type == "imuposer_flu_16hz_to_50hz":
            from imu_lm.data.imuposer_wrist_16hz_to_50hz_loader_FLU import make_imuposer_wrist_16hz_to_50hz_flu_loaders as _make_pose_loaders
        elif _loader_type == "imuposer_full_body_flu":
            from imu_lm.data.imuposer_wrist_full_body_loader_FLU import make_imuposer_wrist_full_body_flu_loaders as _make_pose_loaders
        elif num_joints == 24:
            from imu_lm.data.nymeria_wrist_full_body_loader import make_nymeria_loaders as _make_pose_loaders
        else:
            from imu_lm.data.nymeria_wrist_loader import make_nymeria_loaders as _make_pose_loaders

        # Support train_pkl_root / val_pkl_root (separate dirs) and
        # the sentinel value "all" (load every .pkl in the directory).
        import glob as _glob
        train_pkl_root = pose_cfg.get("train_pkl_root") or pose_cfg.get("pkl_root")
        val_pkl_root   = pose_cfg.get("val_pkl_root")   or train_pkl_root

        def _expand_files(root, files_cfg):
            """Return a list of *absolute* .pkl paths."""
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f)
                        for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg)
                    else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, pose_cfg.get("train_files"))
        val_files   = _expand_files(val_pkl_root,   pose_cfg.get("val_files"))

        # pkl_root is unused when files are absolute, but required by the API
        pkl_root = "/"
        pose_decoder = ViT1DPoseDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            num_joints=num_joints,
            cfg=cfg,
        )

        # Optional: warm-start the encoder from a pretrained run (encoder stays trainable)
        pretrained_run  = pose_cfg.get("pretrained_run")
        pretrained_ckpt = pose_cfg.get("pretrained_ckpt", "best")
        if pretrained_run:
            runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
            if pretrained_ckpt in ("best", "latest"):
                enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                             "checkpoints", f"{pretrained_ckpt}.pt")
            else:
                enc_ckpt_path = pretrained_ckpt
            if not os.path.exists(enc_ckpt_path):
                raise FileNotFoundError(
                    f"[pose] pretrained encoder checkpoint not found: {enc_ckpt_path}"
                )
            import logging as _logging
            _log = _logging.getLogger(__name__)
            _log.info("[pose] warm-starting encoder from %s (encoder stays trainable)", enc_ckpt_path)
            enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
            encoder.load_state_dict(enc_state["model"], strict=True)

        def objective_fn(batch, model, cfg):
            return pose_1d_obj.forward_loss(batch, encoder, pose_decoder, cfg)
        all_params = list(encoder.parameters()) + list(pose_decoder.parameters())
        if _loader_type in ("imuposer_flu", "imuposer_flu_16hz_to_50hz", "imuposer_full_body_flu"):
            loaders = _make_pose_loaders(
                cfg=cfg,
                train_files=train_files,
                val_files=val_files,
                add_gravity=bool(pose_cfg.get("add_gravity", False)),
            )
        else:
            _flu_gyro_w = float(pose_cfg.get("gyro_white_noise_std", 0.0))
            _flu_gyro_b = float(pose_cfg.get("gyro_bias_drift_std",  0.0))
            loaders = _make_pose_loaders(
                cfg=cfg,
                pkl_root=pkl_root,          # ignored; files are absolute
                train_files=train_files,
                val_files=val_files,
                gyro_white_noise_std=_flu_gyro_w,
                gyro_bias_drift_std =_flu_gyro_b,
            )
    elif objective_type == "pose_probe":
        # Frozen pretrained encoder + trainable ViT1DPoseDecoder.
        # Encoder weights are loaded from a prior MAE (or pose) run and frozen.
        # Only the pose decoder parameters are updated during training.
        import glob as _glob
        probe_cfg = cfg_get(cfg, ["objective", "pose_probe"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size
        num_joints              = int(probe_cfg.get("num_joints", 5))

        # ---- load pretrained encoder ----------------------------------------
        pretrained_run  = probe_cfg.get("pretrained_run")
        pretrained_ckpt = probe_cfg.get("pretrained_ckpt", "best")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt   # treat as absolute path

        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[pose_probe] pretrained encoder checkpoint not found: {enc_ckpt_path}"
            )

        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info("[pose_probe] loading pretrained encoder from %s", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(enc_state["model"], strict=True)
        _log.info("[pose_probe] encoder loaded — freezing all encoder parameters")

        # Freeze encoder completely
        encoder.requires_grad_(False)
        encoder.eval()

        _probe_loader_type = str(probe_cfg.get("loader", "")).lower()
        if _probe_loader_type == "flu":
            from imu_lm.data.nymeria_wrist_loader_FLU import make_nymeria_loaders_flu as _make_loaders
        elif _probe_loader_type == "nymeria_real_wrist_flu":
            from imu_lm.data.nymeria_real_wrist_loader_FLU import make_nymeria_real_wrist_loaders_flu as _make_loaders
        elif _probe_loader_type == "imuposer_flu":
            from imu_lm.data.imuposer_wrist_loader_FLU import make_imuposer_wrist_flu_loaders as _make_loaders
        elif _probe_loader_type == "imuposer_flu_16hz_to_50hz":
            from imu_lm.data.imuposer_wrist_16hz_to_50hz_loader_FLU import make_imuposer_wrist_16hz_to_50hz_flu_loaders as _make_loaders
        elif _probe_loader_type == "imuposer_full_body_flu":
            from imu_lm.data.imuposer_wrist_full_body_loader_FLU import make_imuposer_wrist_full_body_flu_loaders as _make_loaders
        elif num_joints == 24:
            from imu_lm.data.nymeria_wrist_full_body_loader import make_nymeria_loaders as _make_loaders
        else:
            from imu_lm.data.nymeria_wrist_loader import make_nymeria_loaders as _make_loaders

        train_pkl_root = probe_cfg.get("train_pkl_root")
        val_pkl_root   = probe_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, probe_cfg.get("train_files"))
        val_files   = _expand_files(val_pkl_root,   probe_cfg.get("val_files"))

        pose_decoder = ViT1DPoseDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            num_joints=num_joints,
            cfg=cfg,
        )

        def objective_fn(batch, model, cfg):
            # Ensure encoder stays in eval / no-grad mode each step
            encoder.eval()
            with torch.no_grad():
                tokens, _ = encoder.forward_tokens(batch[0].float())
            # Swap batch so pose_1d sees pre-computed tokens as if a fresh forward
            # Instead, call decoder directly and compute MSE
            import torch.nn.functional as _F
            y_pose = batch[1].float()
            pred   = pose_decoder(tokens)
            loss   = _F.mse_loss(pred, y_pose)
            return loss, {"loss": float(loss.detach()), "pose_mse": float(loss.detach())}

        # Only decoder parameters are trainable
        all_params = list(pose_decoder.parameters())
        if _probe_loader_type in ("imuposer_flu", "imuposer_flu_16hz_to_50hz", "imuposer_full_body_flu"):
            loaders = _make_loaders(
                cfg=cfg,
                train_files=train_files,
                val_files=val_files,
                add_gravity=bool(probe_cfg.get("add_gravity", False)),
            )
        else:
            _flu_gyro_w = float(probe_cfg.get("gyro_white_noise_std", 0.0))
            _flu_gyro_b = float(probe_cfg.get("gyro_bias_drift_std",  0.0))
            loaders = _make_loaders(
                cfg=cfg,
                pkl_root="/",
                train_files=train_files,
                val_files=val_files,
                gyro_white_noise_std=_flu_gyro_w,
                gyro_bias_drift_std =_flu_gyro_b,
            )

    elif objective_type == "odom_probe":
        # Frozen pretrained encoder + trainable ViT1DOdomDecoder.
        # Mirrors pose_probe but targets root displacement instead of joint rotations.
        import glob as _glob
        odom_probe_cfg = cfg_get(cfg, ["objective", "odom_probe"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        # ---- load & freeze pretrained encoder --------------------------------
        pretrained_run  = odom_probe_cfg.get("pretrained_run")
        pretrained_ckpt = odom_probe_cfg.get("pretrained_ckpt", "best")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt

        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[odom_probe] pretrained encoder checkpoint not found: {enc_ckpt_path}"
            )

        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info("[odom_probe] loading pretrained encoder from %s", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(enc_state["model"], strict=True)
        _log.info("[odom_probe] encoder loaded — freezing all encoder parameters")
        encoder.requires_grad_(False)
        encoder.eval()

        from imu_lm.data.nymeria_wrist_odom_loader import make_nymeria_odom_loaders

        train_pkl_root = odom_probe_cfg.get("train_pkl_root")
        val_pkl_root   = odom_probe_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, odom_probe_cfg.get("train_files"))
        val_files   = _expand_files(val_pkl_root,   odom_probe_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
        )

        def objective_fn(batch, model, cfg):
            encoder.eval()
            x_b, y_b = batch
            with torch.no_grad():
                tokens, _ = encoder.forward_tokens(x_b.float())
            pred = odom_decoder(tokens)
            loss = torch.nn.functional.mse_loss(pred, y_b.float())
            with torch.no_grad():
                mae = (pred.detach() - y_b.float()).abs().mean()
            return loss, {"loss": float(loss.detach()),
                          "odom_mse": float(loss.detach()),
                          "odom_mae": float(mae)}

        all_params = list(odom_decoder.parameters())
        loaders = make_nymeria_odom_loaders(
            cfg=cfg,
            pkl_root="/",
            train_files=train_files,
            val_files=val_files,
        )

    elif objective_type == "odom_finetune":
        # Warm-started encoder (stays trainable) + trainable ViT1DOdomDecoder.
        # Mirrors odom_probe but the encoder is NOT frozen — both encoder and decoder
        # are updated jointly, starting from a pretrained MAE checkpoint.
        import glob as _glob
        odom_ft_cfg = cfg_get(cfg, ["objective", "odom_finetune"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        # ---- warm-start encoder (stays trainable) ----------------------------
        pretrained_run  = odom_ft_cfg.get("pretrained_run")
        pretrained_ckpt = odom_ft_cfg.get("pretrained_ckpt", "best")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt

        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[odom_finetune] pretrained encoder checkpoint not found: {enc_ckpt_path}"
            )

        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info("[odom_finetune] warm-starting encoder from %s (encoder stays trainable)", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(enc_state["model"], strict=True)

        from imu_lm.data.nymeria_wrist_odom_loader import make_nymeria_odom_loaders

        train_pkl_root = odom_ft_cfg.get("train_pkl_root")
        val_pkl_root   = odom_ft_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, odom_ft_cfg.get("train_files"))
        val_files   = _expand_files(val_pkl_root,   odom_ft_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
        )

        def objective_fn(batch, model, cfg):
            return odom_1d_obj.forward_loss(batch, encoder, odom_decoder, cfg)

        all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
        loaders = make_nymeria_odom_loaders(
            cfg=cfg,
            pkl_root="/",
            train_files=train_files,
            val_files=val_files,
        )

    elif objective_type == "odom_probe_imuposer_flu":
        # Frozen pretrained encoder + trainable ViT1DOdomDecoder on IMUPoser FLU data.
        # Real right-wrist MEMS IMU in FLU body frame → per-patch pelvis root displacement [3].
        import glob as _glob
        odom_probe_ip_cfg = cfg_get(cfg, ["objective", "odom_probe_imuposer_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        # ---- load & freeze pretrained encoder --------------------------------
        pretrained_run  = odom_probe_ip_cfg.get("pretrained_run")
        pretrained_ckpt = odom_probe_ip_cfg.get("pretrained_ckpt", "best")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt

        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[odom_probe_imuposer_flu] pretrained encoder checkpoint not found: {enc_ckpt_path}"
            )

        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info("[odom_probe_imuposer_flu] loading pretrained encoder from %s", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(enc_state["model"], strict=True)
        _log.info("[odom_probe_imuposer_flu] encoder loaded — freezing all encoder parameters")
        encoder.requires_grad_(False)
        encoder.eval()

        from imu_lm.data.imuposer_wrist_odom_loader_FLU import make_imuposer_odom_flu_loaders

        train_pkl_root = odom_probe_ip_cfg.get("train_pkl_root")
        val_pkl_root   = odom_probe_ip_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, odom_probe_ip_cfg.get("train_files"))
        val_files   = _expand_files(val_pkl_root,   odom_probe_ip_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
        )

        def objective_fn(batch, model, cfg):
            encoder.eval()
            x_b, y_b = batch
            with torch.no_grad():
                tokens, _ = encoder.forward_tokens(x_b.float())
            pred = odom_decoder(tokens)
            loss = torch.nn.functional.mse_loss(pred, y_b.float())
            with torch.no_grad():
                mae = (pred.detach() - y_b.float()).abs().mean()
            return loss, {"loss": float(loss.detach()),
                          "odom_mse": float(loss.detach()),
                          "odom_mae": float(mae)}

        all_params = list(odom_decoder.parameters())
        loaders = make_imuposer_odom_flu_loaders(
            cfg=cfg,
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(odom_probe_ip_cfg.get("add_gravity", False)),
        )

    elif objective_type == "odom_finetune_imuposer_flu":
        # Warm-started encoder (stays trainable) + trainable ViT1DOdomDecoder on IMUPoser FLU data.
        # Both encoder and decoder are updated jointly from a pretrained MAE checkpoint.
        import glob as _glob
        odom_ft_ip_cfg = cfg_get(cfg, ["objective", "odom_finetune_imuposer_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        # ---- warm-start encoder (stays trainable) ----------------------------
        pretrained_run  = odom_ft_ip_cfg.get("pretrained_run")
        pretrained_ckpt = odom_ft_ip_cfg.get("pretrained_ckpt", "best")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt

        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[odom_finetune_imuposer_flu] pretrained encoder checkpoint not found: {enc_ckpt_path}"
            )

        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info("[odom_finetune_imuposer_flu] warm-starting encoder from %s (encoder stays trainable)", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(enc_state["model"], strict=True)

        from imu_lm.data.imuposer_wrist_odom_loader_FLU import make_imuposer_odom_flu_loaders

        train_pkl_root = odom_ft_ip_cfg.get("train_pkl_root")
        val_pkl_root   = odom_ft_ip_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, odom_ft_ip_cfg.get("train_files"))
        val_files   = _expand_files(val_pkl_root,   odom_ft_ip_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
        )

        def objective_fn(batch, model, cfg):
            return odom_1d_obj.forward_loss(batch, encoder, odom_decoder, cfg)

        all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
        loaders = make_imuposer_odom_flu_loaders(
            cfg=cfg,
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(odom_ft_ip_cfg.get("add_gravity", False)),
        )

    elif objective_type == "odom_nymeria":
        # Odometry regression: sensor-frame wrist IMU → per-patch root displacement [B, Pt, 3]
        import glob as _glob
        odom_cfg = cfg_get(cfg, ["objective", "odom_nymeria"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        from imu_lm.data.nymeria_wrist_odom_loader import make_nymeria_odom_loaders

        train_pkl_root = odom_cfg.get("train_pkl_root")
        val_pkl_root   = odom_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, odom_cfg.get("train_files"))
        val_files   = _expand_files(val_pkl_root,   odom_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
        )

        def objective_fn(batch, model, cfg):
            return odom_1d_obj.forward_loss(batch, encoder, odom_decoder, cfg)

        all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
        loaders = make_nymeria_odom_loaders(
            cfg=cfg,
            pkl_root="/",
            train_files=train_files,
            val_files=val_files,
        )

    elif objective_type == "mae_imuposer_flu":
        # MAE pretraining on IMUPoser pkl files in the right-wrist FLU body frame.
        # Loader yields (x [6,T], y [N_patches,30]); MAE uses only x.
        from imu_lm.data.imuposer_wrist_loader_FLU import make_imuposer_wrist_flu_loaders
        import glob as _glob
        mae_ip_cfg     = cfg_get(cfg, ["objective", "mae_imuposer_flu"], {}) or {}
        train_pkl_root = mae_ip_cfg.get("train_pkl_root")
        val_pkl_root   = mae_ip_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, mae_ip_cfg.get("train_files"))
        val_files   = _expand_files(val_pkl_root,   mae_ip_cfg.get("val_files"))

        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )
        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders = make_imuposer_wrist_flu_loaders(
            cfg=cfg,
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(mae_ip_cfg.get("add_gravity", False)),
        )

    elif objective_type == "odom_imuposer_flu":
        # Odometry regression on IMUPoser pkl files in the right-wrist FLU body frame.
        # Loader yields (x [6,T], y [N_patches,3]); supervised displacement regression.
        from imu_lm.data.imuposer_wrist_odom_loader_FLU import make_imuposer_odom_flu_loaders
        import glob as _glob
        odom_ip_cfg = cfg_get(cfg, ["objective", "odom_imuposer_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        train_pkl_root = odom_ip_cfg.get("train_pkl_root")
        val_pkl_root   = odom_ip_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, odom_ip_cfg.get("train_files"))
        val_files   = _expand_files(val_pkl_root,   odom_ip_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
        )

        def objective_fn(batch, model, cfg):
            return odom_1d_obj.forward_loss(batch, encoder, odom_decoder, cfg)

        all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
        loaders = make_imuposer_odom_flu_loaders(
            cfg=cfg,
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(odom_ip_cfg.get("add_gravity", False)),
        )

    elif objective_type == "mae_nymeria":
        # but data comes from pkl files via NymeriaWristDataset.
        # The Nymeria loader returns (x [B,6,T], y [...]) and the MAE objective
        # does `x, _ = batch`, so pose labels are simply discarded.
        from imu_lm.data.nymeria_wrist_loader import make_nymeria_loaders
        import glob as _glob
        mae_nymeria_cfg = cfg_get(cfg, ["objective", "mae_nymeria"], {}) or {}
        train_pkl_root = mae_nymeria_cfg.get("train_pkl_root")
        val_pkl_root   = mae_nymeria_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, mae_nymeria_cfg.get("train_files"))
        val_files   = _expand_files(val_pkl_root,   mae_nymeria_cfg.get("val_files"))

        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )
        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders = make_nymeria_loaders(
            cfg=cfg,
            pkl_root="/",           # ignored — files are absolute
            train_files=train_files,
            val_files=val_files,
        )

    elif objective_type == "mae_nymeria_flu":
        # MAE pretraining on Nymeria FLU synthetic IMU (right-wrist, FLU body frame).
        # Uses NymeriaWristDatasetFLU loader (linear acc, no gravity component).
        # Pose labels from the loader are discarded — only x is used.
        from imu_lm.data.nymeria_wrist_loader_FLU import make_nymeria_loaders_flu
        import glob as _glob

        mae_flu_cfg    = cfg_get(cfg, ["objective", "mae_nymeria_flu"], {}) or {}
        train_pkl_root = mae_flu_cfg.get("train_pkl_root")
        val_pkl_root   = mae_flu_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, mae_flu_cfg.get("train_files"))
        val_files   = _expand_files(val_pkl_root,   mae_flu_cfg.get("val_files"))

        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )
        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        _mae_flu_gyro_w = float(mae_flu_cfg.get("gyro_white_noise_std", 0.0))
        _mae_flu_gyro_b = float(mae_flu_cfg.get("gyro_bias_drift_std",  0.0))
        loaders = make_nymeria_loaders_flu(
            cfg=cfg,
            pkl_root="/",           # ignored — files are absolute
            train_files=train_files,
            val_files=val_files,
            gyro_white_noise_std=_mae_flu_gyro_w,
            gyro_bias_drift_std =_mae_flu_gyro_b,
        )

    elif objective_type == "mae_nymeria_real_wrist_flu":
        # MAE pretraining on Nymeria REAL wrist IMU data (FLU frame).
        # Pipeline: wrist IMU local CPF → global CPF (world) → local FLU.
        # Uses NymeriaRealWristDatasetFLU — requires has_wrist_imu=True pkl files.
        # Pose labels from the loader are discarded — only x is used.
        from imu_lm.data.nymeria_real_wrist_loader_FLU import make_nymeria_real_wrist_loaders_flu
        import glob as _glob

        mae_rw_cfg     = cfg_get(cfg, ["objective", "mae_nymeria_real_wrist_flu"], {}) or {}
        train_pkl_root = mae_rw_cfg.get("train_pkl_root")
        val_pkl_root   = mae_rw_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, mae_rw_cfg.get("train_files"))
        val_files   = _expand_files(val_pkl_root,   mae_rw_cfg.get("val_files"))
        _add_gravity = bool(mae_rw_cfg.get("add_gravity", True))

        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )
        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders = make_nymeria_real_wrist_loaders_flu(
            cfg=cfg,
            pkl_root="/",           # ignored — files are absolute
            train_files=train_files,
            val_files=val_files,
            add_gravity=_add_gravity,
        )

    elif objective_type == "mae_realdata_nymeria_real_wrist_flu":
        # MAE pretraining on the union of:
        #   (a) Nymeria REAL wrist IMU windows (pkl, FLU frame)   [real sensor data]
        #   (b) Real 6D parquet windows (pamap2/recofit/…)        [~154 h labelled]
        # Both datasets yield (x [6, T], y) 2-tuples; y shapes differ so we strip
        # labels with XOnlyDataset before ConcatDataset, same as mae_combined_flu.
        from imu_lm.data.nymeria_real_wrist_loader_FLU import (
            NymeriaRealWristDatasetFLU,
        )
        from imu_lm.data.loaders import (
            WindowDataset, collate_skip_none as parquet_collate_skip_none,
            build_session_index, make_splits,
        )
        from torch.utils.data import ConcatDataset, DataLoader as _DL
        import glob as _glob
        import logging as _logging
        _log = _logging.getLogger(__name__)

        rw_cfl_cfg     = cfg_get(cfg, ["objective", "mae_realdata_nymeria_real_wrist_flu"], {}) or {}
        train_pkl_root = rw_cfl_cfg.get("train_pkl_root")
        val_pkl_root   = rw_cfl_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_rw_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_pkl_files = _expand_rw_files(train_pkl_root, rw_cfl_cfg.get("train_files"))
        val_pkl_files   = _expand_rw_files(val_pkl_root,   rw_cfl_cfg.get("val_files"))
        _rw_add_gravity = bool(rw_cfl_cfg.get("add_gravity", True))

        # Thin wrapper: strips pose label y, returns uniform (x, dummy) tuples
        # so ConcatDataset items can be stacked regardless of source.
        class _XOnlyDataset(torch.utils.data.Dataset):
            def __init__(self, ds):
                self._ds = ds
            def __len__(self):
                return len(self._ds)
            def __getitem__(self, idx):
                item = self._ds[idx]
                if item is None:
                    return None
                return item[0], torch.empty(0)   # (x [6,T], dummy)

        def _collate_rw_combined(batch):
            batch = [b for b in batch if b is not None]
            if not batch:
                return None
            xs = torch.stack([b[0] for b in batch], 0)
            return xs, torch.empty(0)   # MAE only uses xs

        # Build Nymeria real-wrist datasets
        rw_train_ds = _XOnlyDataset(NymeriaRealWristDatasetFLU(
            pkl_paths=train_pkl_files, cfg=cfg, split_name="train",
            add_gravity=_rw_add_gravity,
        ))
        rw_val_ds = _XOnlyDataset(NymeriaRealWristDatasetFLU(
            pkl_paths=val_pkl_files, cfg=cfg, split_name="val",
            add_gravity=_rw_add_gravity,
        ))

        # Build parquet datasets
        parquet_path  = cfg_get(cfg, ["paths", "dataset_path"])
        train_on      = cfg_get(cfg, ["data", "train_on_datasets"], None)
        parquet_si    = build_session_index(parquet_path, cfg,
                                            dataset_filter=list(train_on) if train_on else None)
        parquet_splits = make_splits(parquet_si, cfg)
        parq_train_ds = _XOnlyDataset(WindowDataset(
            parquet_path, parquet_si, parquet_splits["train_keys"], cfg, split_name="train"
        ))
        parq_val_ds = _XOnlyDataset(WindowDataset(
            parquet_path, parquet_si, parquet_splits["val_keys"], cfg, split_name="val"
        ))

        combined_train = ConcatDataset([rw_train_ds, parq_train_ds])
        combined_val   = ConcatDataset([rw_val_ds,   parq_val_ds])

        _log.info(
            "[mae_realdata_nymeria_real_wrist_flu] train=%d val=%d "
            "(rw_train=%d parq_train=%d rw_val=%d parq_val=%d)",
            len(combined_train), len(combined_val),
            len(rw_train_ds), len(parq_train_ds),
            len(rw_val_ds),   len(parq_val_ds),
        )

        num_workers_  = int(cfg_get(cfg, ["data", "num_workers"], 0))
        pin_memory_   = bool(cfg_get(cfg, ["data", "pin_memory"], False))
        batch_size_   = int(cfg_get(cfg, ["data", "batch_size"], 256))
        eval_bs_      = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size_))
        persistent_   = num_workers_ > 0

        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )
        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders = {
            "train_loader": _DL(combined_train, batch_size=batch_size_,  shuffle=True,
                                num_workers=num_workers_, pin_memory=pin_memory_,
                                drop_last=True,  persistent_workers=persistent_,
                                collate_fn=_collate_rw_combined),
            "val_loader":   _DL(combined_val,   batch_size=eval_bs_,     shuffle=False,
                                num_workers=num_workers_, pin_memory=pin_memory_,
                                drop_last=False, persistent_workers=persistent_,
                                collate_fn=_collate_rw_combined),
        }

    elif objective_type == "mae_nymeria_var_len":
        # Variable-length MAE on Nymeria pkl files.
        # Window length is sampled uniformly from [T_min, T_max] each batch
        # (configured via windowing.min_window_seconds / max_window_seconds).
        # Uses sinusoidal positional embeddings (set pos_embed_type: sinusoidal
        # in the config) to generalise to lengths unseen at inference time.
        from imu_lm.data.nymeria_wrist_var_len_loader import make_nymeria_var_len_loaders
        import glob as _glob

        mae_var_cfg    = cfg_get(cfg, ["objective", "mae_nymeria_var_len"], {}) or {}
        train_pkl_root = mae_var_cfg.get("train_pkl_root")
        val_pkl_root   = mae_var_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, mae_var_cfg.get("train_files"))
        val_files   = _expand_files(val_pkl_root,   mae_var_cfg.get("val_files"))

        # Decoder target_T=0 disables fixed-T interpolation in ViT1DDecoder.forward;
        # num_patches_per_channel is passed dynamically at each forward call.
        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=0,   # variable length: skip interpolation guard
            cfg=cfg,
        )

        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)

        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders    = make_nymeria_var_len_loaders(
            cfg=cfg,
            train_files=train_files,
            val_files=val_files,
        )

    elif objective_type == "mae_combined_flu":
        # MAE pretraining on the union of:
        #   (a) Nymeria FLU synthetic IMU windows (pkl files)  [~1k h unlabelled]
        #   (b) Real 6D parquet windows (pamap2/recofit/…)     [~154 h labelled]
        # Both datasets yield (x [6, T], y) 2-tuples.  The MAE objective only
        # uses x (does `x, _ = batch`), but the y shapes differ between the two
        # sources, so we wrap each dataset in XOnlyDataset before concatenation.
        from imu_lm.data.nymeria_wrist_loader_FLU import (
            NymeriaWristDatasetFLU,
            collate_skip_none as flu_collate_skip_none,
        )
        from imu_lm.data.loaders import (
            WindowDataset, collate_skip_none as parquet_collate_skip_none,
            build_session_index, make_splits,
        )
        from imu_lm.data.splits import SessionKey
        from torch.utils.data import ConcatDataset, DataLoader as _DL
        import glob as _glob

        cfl_cfg        = cfg_get(cfg, ["objective", "mae_combined_flu"], {}) or {}
        train_pkl_root = cfl_cfg.get("train_pkl_root")
        val_pkl_root   = cfl_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_pkl_files = _expand_files(train_pkl_root, cfl_cfg.get("train_files"))
        val_pkl_files   = _expand_files(val_pkl_root,   cfl_cfg.get("val_files"))

        # Thin wrapper: strips label / pose y, returns uniform (x, dummy) tuples
        # so ConcatDataset items can be stacked regardless of source.
        class XOnlyDataset(torch.utils.data.Dataset):
            def __init__(self, ds):
                self._ds = ds
            def __len__(self):
                return len(self._ds)
            def __getitem__(self, idx):
                item = self._ds[idx]
                if item is None:
                    return None
                return item[0], torch.empty(0)   # (x [6,T], dummy)

        def _collate_combined(batch):
            batch = [b for b in batch if b is not None]
            if not batch:
                return None
            xs = torch.stack([b[0] for b in batch], 0)
            return xs, torch.empty(0)   # MAE only uses xs

        # Build FLU Nymeria datasets
        flu_train_ds = XOnlyDataset(NymeriaWristDatasetFLU(
            pkl_paths=train_pkl_files, cfg=cfg, split_name="train"
        ))
        flu_val_ds = XOnlyDataset(NymeriaWristDatasetFLU(
            pkl_paths=val_pkl_files, cfg=cfg, split_name="val"
        ))

        # Build parquet datasets
        parquet_path  = cfg_get(cfg, ["paths", "dataset_path"])
        train_on      = cfg_get(cfg, ["data", "train_on_datasets"], None)
        parquet_si    = build_session_index(parquet_path, cfg,
                                            dataset_filter=list(train_on) if train_on else None)
        parquet_splits = make_splits(parquet_si, cfg)
        parq_train_ds = XOnlyDataset(WindowDataset(
            parquet_path, parquet_si, parquet_splits["train_keys"], cfg, split_name="train"
        ))
        parq_val_ds = XOnlyDataset(WindowDataset(
            parquet_path, parquet_si, parquet_splits["val_keys"], cfg, split_name="val"
        ))

        combined_train = ConcatDataset([flu_train_ds, parq_train_ds])
        combined_val   = ConcatDataset([flu_val_ds,   parq_val_ds])

        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info("[mae_combined_flu] train=%d  val=%d  (flu_train=%d parq_train=%d flu_val=%d parq_val=%d)",
                  len(combined_train), len(combined_val),
                  len(flu_train_ds), len(parq_train_ds),
                  len(flu_val_ds),   len(parq_val_ds))

        num_workers_  = int(cfg_get(cfg, ["data", "num_workers"], 0))
        pin_memory_   = bool(cfg_get(cfg, ["data", "pin_memory"], False))
        batch_size_   = int(cfg_get(cfg, ["data", "batch_size"], 256))
        eval_bs_      = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size_))
        persistent_   = num_workers_ > 0

        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )
        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders = {
            "train_loader": _DL(combined_train, batch_size=batch_size_,  shuffle=True,
                                num_workers=num_workers_, pin_memory=pin_memory_,
                                drop_last=True,  persistent_workers=persistent_,
                                collate_fn=_collate_combined),
            "val_loader":   _DL(combined_val,   batch_size=eval_bs_,     shuffle=False,
                                num_workers=num_workers_, pin_memory=pin_memory_,
                                drop_last=False, persistent_workers=persistent_,
                                collate_fn=_collate_combined),
        }

    elif objective_type == "mae_combined_flu_var_len":
        from imu_lm.data.combined_flu_var_len_loader import make_combined_flu_var_len_loaders
        import glob as _glob

        cfl_cfg = cfg_get(cfg, ["objective", "mae_combined_flu_var_len"], {}) or {}
        train_pkl_root = cfl_cfg.get("train_pkl_root")
        val_pkl_root = cfl_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_pkl_files = _expand_files(train_pkl_root, cfl_cfg.get("train_files"))
        val_pkl_files = _expand_files(val_pkl_root, cfl_cfg.get("val_files"))

        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )

        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)

        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders = make_combined_flu_var_len_loaders(
            cfg=cfg,
            train_pkl_files=train_pkl_files,
            val_pkl_files=val_pkl_files,
        )

    elif objective_type == "imu_text_clip":
        # CLIP-style contrastive pretraining: IMU ↔ text on HumanML3D.
        # The ViT1D encoder (sinusoidal pos-embed) is warm-started from a var-len
        # MAE run and then jointly fine-tuned with a linear CLIP projection head
        # against a frozen CLIP text encoder.
        from imu_lm.objectives.contrastive_clip_1d import (
            ContrastiveCLIPHead, CLIPTextEncoder
        )
        import imu_lm.objectives.contrastive_clip_1d as clip_obj
        from imu_lm.data.humanml3d_dataset import make_humanml3d_imu_loaders

        clip_cfg       = cfg_get(cfg, ["objective", "imu_text_clip"], {}) or {}
        runs_root      = cfg_get(cfg, ["paths", "runs_root"], "runs")
        pretrained_run = clip_cfg.get("pretrained_run")
        pretrained_ckpt = clip_cfg.get("pretrained_ckpt", "best")
        humanml3d_root = clip_cfg.get("humanml3d_root")
        clip_model     = clip_cfg.get("clip_model", "openai/clip-vit-base-patch32")
        clip_embed_dim = int(clip_cfg.get("clip_embed_dim", 512))
        freeze_imu     = bool(clip_cfg.get("freeze_imu_encoder", False))
        freeze_text    = bool(clip_cfg.get("freeze_text_encoder", True))
        text_cache_path = clip_cfg.get("text_cache_path", None)

        if not humanml3d_root:
            raise ValueError("[imu_text_clip] 'humanml3d_root' must be set in objective.imu_text_clip")

        # -- load pre-cached text embeddings (optional) ----------------------
        text_emb_cache = None
        if text_cache_path:
            import logging as _logging2
            _log2 = _logging2.getLogger(__name__)
            if not os.path.exists(text_cache_path):
                raise FileNotFoundError(
                    f"[imu_text_clip] text_cache_path not found: {text_cache_path}\n"
                    "Run:  python scripts/cache_clip_text_embeddings.py "
                    f"--humanml3d-root {humanml3d_root} --out {text_cache_path}"
                )
            _log2.info("[imu_text_clip] loading pre-cached text embeddings from %s", text_cache_path)
            text_emb_cache = torch.load(text_cache_path, map_location="cpu", weights_only=False)
            _log2.info("[imu_text_clip] cache loaded: %d captions", len(text_emb_cache))

        # -- warm-start IMU encoder ------------------------------------------
        if pretrained_run:
            if pretrained_ckpt in ("best", "latest"):
                enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                             "checkpoints", f"{pretrained_ckpt}.pt")
            else:
                enc_ckpt_path = pretrained_ckpt
            if not os.path.exists(enc_ckpt_path):
                raise FileNotFoundError(
                    f"[imu_text_clip] pretrained encoder checkpoint not found: {enc_ckpt_path}"
                )
            import logging as _logging
            _log = _logging.getLogger(__name__)
            _log.info("[imu_text_clip] warm-starting encoder from %s", enc_ckpt_path)
            enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
            encoder.load_state_dict(enc_state["model"], strict=True)

        if freeze_imu:
            encoder.requires_grad_(False)
            encoder.eval()

        # -- CLIP projection head (trained) -----------------------------------
        clip_head = ContrastiveCLIPHead(
            imu_embed_dim  = encoder.embed_dim,
            clip_embed_dim = clip_embed_dim,
        )

        # -- CLIP text encoder (frozen by default) ----------------------------
        # Skip loading the CLIP text model entirely when a pre-cached embedding
        # file is available AND the text encoder is frozen (the most common case).
        # This saves ~250 MB of GPU memory and eliminates per-batch tokenisation.
        local_files = bool(clip_cfg.get("local_files_only", False))
        if text_emb_cache is not None and freeze_text:
            text_encoder = None   # cache fully replaces the online encoder
        else:
            text_encoder = CLIPTextEncoder(
                model_name       = clip_model,
                freeze           = freeze_text,
                local_files_only = local_files,
            )

        def objective_fn(batch, model, cfg):
            return clip_obj.forward_loss(batch, encoder, clip_head, text_encoder, cfg)

        if freeze_imu:
            all_params = list(clip_head.parameters())
        else:
            all_params = list(encoder.parameters()) + list(clip_head.parameters())
        if not freeze_text and text_encoder is not None:
            all_params += list(text_encoder.model.parameters())

        loaders = make_humanml3d_imu_loaders(
            root           = humanml3d_root,
            cfg            = cfg,
            text_emb_cache = text_emb_cache,
        )

    elif objective_type == "imu_text_clip_flu":
        # CLIP-style contrastive pretraining: FLU-frame IMU ↔ text on HumanML3D.
        # Uses humanml3d_dataset_FLU.py which produces right-wrist IMU in the FLU
        # body frame (linear acc, no gravity) — matching the Nymeria FLU pipeline.
        # Encoder warm-started from a pretrained checkpoint (e.g. vit1d_realdata_pretrain).
        from imu_lm.objectives.contrastive_clip_1d import (
            ContrastiveCLIPHead, CLIPTextEncoder
        )
        import imu_lm.objectives.contrastive_clip_1d as clip_obj
        from imu_lm.data.humanml3d_dataset_FLU import make_humanml3d_imu_loaders_flu

        clip_cfg        = cfg_get(cfg, ["objective", "imu_text_clip_flu"], {}) or {}
        runs_root       = cfg_get(cfg, ["paths", "runs_root"], "runs")
        pretrained_run  = clip_cfg.get("pretrained_run")
        pretrained_ckpt = clip_cfg.get("pretrained_ckpt", "best")
        humanml3d_root  = clip_cfg.get("humanml3d_root")
        clip_model      = clip_cfg.get("clip_model", "openai/clip-vit-base-patch32")
        clip_embed_dim  = int(clip_cfg.get("clip_embed_dim", 512))
        freeze_imu      = bool(clip_cfg.get("freeze_imu_encoder", False))
        freeze_text     = bool(clip_cfg.get("freeze_text_encoder", True))
        text_cache_path = clip_cfg.get("text_cache_path", None)

        if not humanml3d_root:
            raise ValueError("[imu_text_clip_flu] 'humanml3d_root' must be set in objective.imu_text_clip_flu")

        # -- load pre-cached text embeddings (optional) ----------------------
        text_emb_cache = None
        if text_cache_path:
            import logging as _logging2
            _log2 = _logging2.getLogger(__name__)
            if not os.path.exists(text_cache_path):
                raise FileNotFoundError(
                    f"[imu_text_clip_flu] text_cache_path not found: {text_cache_path}\n"
                    "Run:  python scripts/cache_clip_text_embeddings.py "
                    f"--humanml3d-root {humanml3d_root} --out {text_cache_path}"
                )
            _log2.info("[imu_text_clip_flu] loading pre-cached text embeddings from %s", text_cache_path)
            text_emb_cache = torch.load(text_cache_path, map_location="cpu", weights_only=False)
            _log2.info("[imu_text_clip_flu] cache loaded: %d captions", len(text_emb_cache))

        # -- warm-start IMU encoder ------------------------------------------
        if pretrained_run:
            if pretrained_ckpt in ("best", "latest"):
                enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                             "checkpoints", f"{pretrained_ckpt}.pt")
            else:
                enc_ckpt_path = pretrained_ckpt
            if not os.path.exists(enc_ckpt_path):
                raise FileNotFoundError(
                    f"[imu_text_clip_flu] pretrained encoder checkpoint not found: {enc_ckpt_path}"
                )
            import logging as _logging
            _log = _logging.getLogger(__name__)
            _log.info("[imu_text_clip_flu] warm-starting encoder from %s", enc_ckpt_path)
            enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
            encoder.load_state_dict(enc_state["model"], strict=True)

        if freeze_imu:
            encoder.requires_grad_(False)
            encoder.eval()

        # -- CLIP projection head (trained) -----------------------------------
        clip_head = ContrastiveCLIPHead(
            imu_embed_dim  = encoder.embed_dim,
            clip_embed_dim = clip_embed_dim,
        )

        # -- CLIP text encoder (frozen by default) ----------------------------
        local_files = bool(clip_cfg.get("local_files_only", False))
        if text_emb_cache is not None and freeze_text:
            text_encoder = None   # cache fully replaces the online encoder
        else:
            text_encoder = CLIPTextEncoder(
                model_name       = clip_model,
                freeze           = freeze_text,
                local_files_only = local_files,
            )

        def objective_fn(batch, model, cfg):
            return clip_obj.forward_loss(batch, encoder, clip_head, text_encoder, cfg)

        if freeze_imu:
            all_params = list(clip_head.parameters())
        else:
            all_params = list(encoder.parameters()) + list(clip_head.parameters())
        if not freeze_text and text_encoder is not None:
            all_params += list(text_encoder.model.parameters())

        loaders = make_humanml3d_imu_loaders_flu(
            root           = humanml3d_root,
            cfg            = cfg,
            text_emb_cache = text_emb_cache,
        )

    elif objective_type == "mae_realdata_16hz_to_50hz":
        # MAE on real 6D parquet data with 50→16Hz→50Hz frequency degradation applied
        # to every session at load time (downsample + band-limited upsample).
        from imu_lm.data.realdata_16hz_to_50hz_loader import make_realdata_16hz_to_50hz_loaders
        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )
        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders = make_realdata_16hz_to_50hz_loaders(cfg)

    elif objective_type == "mae_realdata_30hz_to_50hz":
        # MAE on real 6D parquet data with 50→30Hz→50Hz frequency degradation applied
        # to every session at load time (downsample + band-limited upsample).
        from imu_lm.data.realdata_30hz_to_50hz_loader import make_realdata_30hz_to_50hz_loaders
        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )
        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders = make_realdata_30hz_to_50hz_loaders(cfg)

    elif objective_type == "watchhand_gesture_supervised":
        # Fully-supervised gesture classification on WatchHand IMU data.
        # Encoder (ViT1D) + LinearHead trained end-to-end (or warm-started from a pretrained run).
        # Supports three pose groups: simple (5 classes), asl (10 classes), wrist (3 classes).
        from imu_lm.data.watchhand_loader import (
            WatchHandDataset,
            make_watchhand_loaders,
            SIMPLE_POSE_IDS,
            ASL_POSE_IDS,
            WRIST_POSE_IDS,
        )
        import logging as _logging
        _wh_log = _logging.getLogger(__name__)

        wh_cfg = cfg_get(cfg, ["objective", "watchhand_gesture_supervised"], {}) or {}

        # Select pose group
        pose_group = str(wh_cfg.get("pose_group", "all")).lower()
        if pose_group == "simple":
            pose_ids = SIMPLE_POSE_IDS
        elif pose_group == "asl":
            pose_ids = ASL_POSE_IDS
        elif pose_group == "wrist":
            pose_ids = WRIST_POSE_IDS
        else:
            pose_ids = None  # all 18 classes

        watchhand_root = wh_cfg.get("watchhand_root", "WatchHand")
        train_subjects = wh_cfg.get("train_subjects", None)
        val_subjects   = wh_cfg.get("val_subjects",   None)
        studies        = wh_cfg.get("studies",        None)   # None → default ["Study-1"]
        target_hz      = int(wh_cfg.get("target_hz",  50))

        # window_size is driven by T (from windowing.window_seconds @ 50 Hz)
        window_size = T

        train_ds = WatchHandDataset(
            watchhand_root=watchhand_root,
            window_size=window_size,
            subjects=train_subjects,
            studies=studies,
            pose_ids=pose_ids,
            target_hz=target_hz,
        )
        val_ds = WatchHandDataset(
            watchhand_root=watchhand_root,
            window_size=window_size,
            subjects=val_subjects,
            studies=studies,
            pose_ids=pose_ids,
            target_hz=target_hz,
        ) if val_subjects else None

        batch_size_  = int(cfg_get(cfg, ["data", "batch_size"],       64))
        eval_bs_     = int(cfg_get(cfg, ["data", "eval_batch_size"],  batch_size_))
        n_workers_   = int(cfg_get(cfg, ["data", "num_workers"],       4))
        pin_mem_     = bool(cfg_get(cfg, ["data", "pin_memory"],    True))

        from torch.utils.data import DataLoader as _DL
        _train_loader_wh = _DL(train_ds, batch_size=batch_size_, shuffle=True,
                                num_workers=n_workers_, pin_memory=pin_mem_, drop_last=True)
        _val_loader_wh   = (_DL(val_ds, batch_size=eval_bs_, shuffle=False,
                                num_workers=n_workers_, pin_memory=pin_mem_)
                            if val_ds is not None else None)
        loaders = {"train_loader": _train_loader_wh, "val_loader": _val_loader_wh}

        num_classes = train_ds.num_classes
        _wh_log.info("[watchhand_gesture_supervised] pose_group=%s  num_classes=%d  train=%d  val=%s",
                     pose_group, num_classes, len(train_ds),
                     len(val_ds) if val_ds else "none")

        head = LinearHead(encoder.embed_dim, num_classes)

        # Optional: warm-start encoder from a pretrained MAE/supervised run
        pretrained_run  = wh_cfg.get("pretrained_run",  None)
        pretrained_ckpt = wh_cfg.get("pretrained_ckpt", "best")
        if pretrained_run:
            runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
            if pretrained_ckpt in ("best", "latest"):
                enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                             "checkpoints", f"{pretrained_ckpt}.pt")
            else:
                enc_ckpt_path = pretrained_ckpt
            if not os.path.exists(enc_ckpt_path):
                raise FileNotFoundError(
                    f"[watchhand_gesture_supervised] pretrained encoder not found: {enc_ckpt_path}"
                )
            _wh_log.info("[watchhand_gesture_supervised] warm-starting encoder from %s", enc_ckpt_path)
            enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
            encoder.load_state_dict(enc_state["model"], strict=True)

        # Labels are already remapped to 0-based by WatchHandDataset; no raw_to_idx needed
        def objective_fn(batch, model, cfg):
            return supervised_obj.forward_loss(batch, encoder, head, cfg, raw_to_idx=None)

        all_params = list(encoder.parameters()) + list(head.parameters())

    elif objective_type == "watchhand_gesture_supervised_within_subject":
        # Fully-supervised gesture classification — within-subject split.
        # Both train and val come from the SAME subjects, split by session block index.
        # Default: blocks 0–7 = train, blocks 8–9 = val (Study-1 has 10 blocks per subject).
        from imu_lm.data.watchhand_loader import (
            WatchHandDataset,
            SIMPLE_POSE_IDS,
            ASL_POSE_IDS,
            WRIST_POSE_IDS,
            WITHIN_SUBJECT_TRAIN_BLOCKS,
            WITHIN_SUBJECT_VAL_BLOCKS,
        )
        import logging as _logging
        _wh_log = _logging.getLogger(__name__)

        wh_cfg = cfg_get(cfg, ["objective", "watchhand_gesture_supervised_within_subject"], {}) or {}

        # Select pose group
        pose_group = str(wh_cfg.get("pose_group", "all")).lower()
        if pose_group == "simple":
            pose_ids = SIMPLE_POSE_IDS
        elif pose_group == "asl":
            pose_ids = ASL_POSE_IDS
        elif pose_group == "wrist":
            pose_ids = WRIST_POSE_IDS
        else:
            pose_ids = None  # all 18 classes

        watchhand_root  = wh_cfg.get("watchhand_root", "WatchHand")
        subjects        = wh_cfg.get("subjects",    None)   # None → all subjects
        studies         = wh_cfg.get("studies",     None)   # None → default ["Study-1"]
        target_hz       = int(wh_cfg.get("target_hz", 50))
        train_block_nums = wh_cfg.get("train_block_nums", WITHIN_SUBJECT_TRAIN_BLOCKS)
        val_block_nums   = wh_cfg.get("val_block_nums",   WITHIN_SUBJECT_VAL_BLOCKS)

        window_size = T

        train_ds = WatchHandDataset(
            watchhand_root=watchhand_root,
            window_size=window_size,
            subjects=subjects,
            studies=studies,
            pose_ids=pose_ids,
            target_hz=target_hz,
            block_nums=train_block_nums,
        )
        val_ds = WatchHandDataset(
            watchhand_root=watchhand_root,
            window_size=window_size,
            subjects=subjects,
            studies=studies,
            pose_ids=pose_ids,
            target_hz=target_hz,
            block_nums=val_block_nums,
        )

        batch_size_  = int(cfg_get(cfg, ["data", "batch_size"],       64))
        eval_bs_     = int(cfg_get(cfg, ["data", "eval_batch_size"],  batch_size_))
        n_workers_   = int(cfg_get(cfg, ["data", "num_workers"],       4))
        pin_mem_     = bool(cfg_get(cfg, ["data", "pin_memory"],    True))

        from torch.utils.data import DataLoader as _DL
        _train_loader_wh = _DL(train_ds, batch_size=batch_size_, shuffle=True,
                                num_workers=n_workers_, pin_memory=pin_mem_, drop_last=True)
        _val_loader_wh   = _DL(val_ds,   batch_size=eval_bs_,   shuffle=False,
                                num_workers=n_workers_, pin_memory=pin_mem_)
        loaders = {"train_loader": _train_loader_wh, "val_loader": _val_loader_wh}

        num_classes = train_ds.num_classes
        _wh_log.info(
            "[watchhand_gesture_supervised_within_subject] pose_group=%s  num_classes=%d  "
            "train=%d (blocks %s)  val=%d (blocks %s)",
            pose_group, num_classes, len(train_ds), train_block_nums, len(val_ds), val_block_nums,
        )

        head = LinearHead(encoder.embed_dim, num_classes)

        # Optional: warm-start encoder from a pretrained run
        pretrained_run  = wh_cfg.get("pretrained_run",  None)
        pretrained_ckpt = wh_cfg.get("pretrained_ckpt", "best")
        if pretrained_run:
            runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
            if pretrained_ckpt in ("best", "latest"):
                enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                             "checkpoints", f"{pretrained_ckpt}.pt")
            else:
                enc_ckpt_path = pretrained_ckpt
            if not os.path.exists(enc_ckpt_path):
                raise FileNotFoundError(
                    f"[watchhand_gesture_supervised_within_subject] pretrained encoder not found: {enc_ckpt_path}"
                )
            _wh_log.info("[watchhand_gesture_supervised_within_subject] warm-starting encoder from %s", enc_ckpt_path)
            enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
            encoder.load_state_dict(enc_state["model"], strict=True)

        def objective_fn(batch, model, cfg):
            return supervised_obj.forward_loss(batch, encoder, head, cfg, raw_to_idx=None)

        all_params = list(encoder.parameters()) + list(head.parameters())

    else:
        # MAE: encoder + decoder, self-supervised (parquet data)
        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )
        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders = make_loaders(cfg)

    train_loader = loaders.get("train_loader")
    val_loader = loaders.get("val_loader")
    
    optimizer = build_optimizer_from_params(all_params, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    
    # Build extra_modules dict for checkpointing head or decoder
    if objective_type in ("supervised", "watchhand_gesture_supervised",
                          "watchhand_gesture_supervised_within_subject"):
        extra_modules = {"head": head}
    elif objective_type in ("pose", "pose_probe"):
        extra_modules = {"pose_decoder": pose_decoder}
    elif objective_type in ("odom_nymeria", "odom_probe", "odom_finetune", "odom_imuposer_flu",
                            "odom_probe_imuposer_flu", "odom_finetune_imuposer_flu"):
        extra_modules = {"odom_decoder": odom_decoder}
    elif objective_type in ("imu_text_clip", "imu_text_clip_flu"):
        # Save projection head + logit_scale; text encoder can be reloaded from HuggingFace
        extra_modules = {"clip_head": clip_head}
    else:
        extra_modules = {"decoder": decoder}
    
    # Load checkpoint if resuming
    resume_path = resolve_resume_path(run_dir, resume_ckpt)
    start_step = load_checkpoint(resume_path, encoder, optimizer, scheduler, extra_modules=extra_modules)
    
    trainer = Trainer(cfg, run_dir)
    trainer.fit(encoder, objective_fn, train_loader, val_loader, optimizer, scheduler, start_step=start_step, extra_modules=extra_modules)


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_pretrain.py to launch training.")
