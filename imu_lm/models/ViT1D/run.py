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
        elif _loader_type == "nymeria_real_wrist_full_body_flu":
            from imu_lm.data.nymeria_real_wrist_full_body_loader_FLU import make_nymeria_real_wrist_full_body_loaders_flu as _make_pose_loaders
        elif _loader_type == "nymeria_real_wrist_full_body_global":
            from imu_lm.data.nymeria_real_wrist_full_body_loader_global import make_nymeria_real_wrist_full_body_loaders_global as _make_pose_loaders
        elif _loader_type == "nymeria_syn_wrist_full_body_global":
            from imu_lm.data.nymeria_syn_wrist_full_body_loader_global import make_nymeria_syn_wrist_full_body_loaders_global as _make_pose_loaders
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
        _frac = pose_cfg.get("train_fraction")
        if _frac is not None:
            train_files = train_files[:max(1, int(len(train_files) * float(_frac)))]
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

        _pose_num_joints = num_joints  # capture for closure
        def objective_fn(batch, model, cfg):
            x, y_full = batch
            # If num_joints < full label (e.g. 23 vs 24), skip root joint (first 6 dims)
            label_joints = y_full.shape[-1] // 6
            if _pose_num_joints < label_joints:
                skip = (label_joints - _pose_num_joints) * 6
                y_pose = y_full[:, :, skip:]
            else:
                y_pose = y_full
            return pose_1d_obj.forward_loss((x, y_pose), encoder, pose_decoder, cfg)
        all_params = list(encoder.parameters()) + list(pose_decoder.parameters())
        if _loader_type == "totalcapture_real_wrist_flu":
            from imu_lm.data.totalcapture_wrist_loader_FLU import make_totalcapture_real_wrist_loaders_flu as _make_tc
            _tc_root = pose_cfg.get("pkl_root") or train_pkl_root or "/mnt/nas2/DIP_recalculate"
            def _expand_tc_subjects(subjects):
                return [f for s in (subjects or [])
                        for f in sorted(_glob.glob(os.path.join(_tc_root, f"{s}_*.pkl")))]
            _tc_train = (train_files if pose_cfg.get("train_files")
                         else _expand_tc_subjects(pose_cfg.get("train_subjects")))
            _tc_val   = (val_files   if pose_cfg.get("val_files")
                         else _expand_tc_subjects(pose_cfg.get("val_subjects")))
            loaders = _make_tc(
                cfg=cfg,
                pkl_root=_tc_root,
                train_files=_tc_train,
                val_files=_tc_val,
                target_hz=int(pose_cfg.get("target_hz", 60)),
                add_gravity=bool(pose_cfg.get("add_gravity", True)),
                use_gt_ori=bool(pose_cfg.get("use_gt_ori", False)),
                num_joints=num_joints,
            )
        elif _loader_type == "totalcapture_syn_wrist_flu":
            from imu_lm.data.totalcapture_wrist_loader_FLU_syn import make_totalcapture_syn_wrist_loaders_flu as _make_tc_syn
            _tc_root = pose_cfg.get("pkl_root") or train_pkl_root or "/mnt/nas2/DIP_recalculate"
            def _expand_tc_subjects(subjects):
                return [f for s in (subjects or [])
                        for f in sorted(_glob.glob(os.path.join(_tc_root, f"{s}_*.pkl")))]
            _tc_train = (train_files if pose_cfg.get("train_files")
                         else _expand_tc_subjects(pose_cfg.get("train_subjects")))
            _tc_val   = (val_files   if pose_cfg.get("val_files")
                         else _expand_tc_subjects(pose_cfg.get("val_subjects")))
            loaders = _make_tc_syn(
                cfg=cfg,
                pkl_root=_tc_root,
                train_files=_tc_train,
                val_files=_tc_val,
                target_hz=int(pose_cfg.get("target_hz", 60)),
                add_gravity=bool(pose_cfg.get("add_gravity", True)),
                official_dir=pose_cfg.get("official_dir"),
                gyro_white_noise_std=float(pose_cfg.get("gyro_white_noise_std", 0.0)),
                gyro_bias_drift_std=float(pose_cfg.get("gyro_bias_drift_std", 0.0)),
                num_joints=num_joints,
            )
        elif _loader_type in ("imuposer_flu", "imuposer_flu_16hz_to_50hz", "imuposer_full_body_flu"):
            loaders = _make_pose_loaders(
                cfg=cfg,
                train_files=train_files,
                val_files=val_files,
                add_gravity=bool(pose_cfg.get("add_gravity", False)),
            )
        elif _loader_type in ("nymeria_real_wrist_flu", "nymeria_real_wrist_full_body_flu",
                               "nymeria_real_wrist_full_body_global",
                               "nymeria_syn_wrist_full_body_global"):
            loaders = _make_pose_loaders(
                cfg=cfg,
                pkl_root=pkl_root,
                train_files=train_files,
                val_files=val_files,
                add_gravity=bool(pose_cfg.get("add_gravity", True)),
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
        elif _probe_loader_type == "nymeria_real_wrist_full_body_flu":
            from imu_lm.data.nymeria_real_wrist_full_body_loader_FLU import make_nymeria_real_wrist_full_body_loaders_flu as _make_loaders
        elif _probe_loader_type == "nymeria_real_wrist_full_body_global":
            from imu_lm.data.nymeria_real_wrist_full_body_loader_global import make_nymeria_real_wrist_full_body_loaders_global as _make_loaders
        elif _probe_loader_type == "nymeria_syn_wrist_full_body_global":
            from imu_lm.data.nymeria_syn_wrist_full_body_loader_global import make_nymeria_syn_wrist_full_body_loaders_global as _make_loaders
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

        train_pkl_root = probe_cfg.get("train_pkl_root") or probe_cfg.get("pkl_root")
        val_pkl_root   = probe_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, probe_cfg.get("train_files"))
        _frac = probe_cfg.get("train_fraction")
        if _frac is not None:
            train_files = train_files[:max(1, int(len(train_files) * float(_frac)))]
        val_files   = _expand_files(val_pkl_root,   probe_cfg.get("val_files"))

        pose_decoder = ViT1DPoseDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            num_joints=num_joints,
            cfg=cfg,
        )

        _probe_num_joints = num_joints  # capture for closure
        def objective_fn(batch, model, cfg):
            # Ensure encoder stays in eval / no-grad mode each step
            encoder.eval()
            with torch.no_grad():
                tokens, _ = encoder.forward_tokens(batch[0].float())
            import torch.nn.functional as _F
            y_full = batch[1].float()
            # If num_joints < full label (e.g. 23 vs 24), skip root joint (first 6 dims)
            label_joints = y_full.shape[-1] // 6
            if _probe_num_joints < label_joints:
                skip = (label_joints - _probe_num_joints) * 6
                y_pose = y_full[:, :, skip:]
            else:
                y_pose = y_full
            pred = pose_decoder(tokens)
            loss = _F.mse_loss(pred, y_pose)
            return loss, {"loss": float(loss.detach()), "pose_mse": float(loss.detach())}

        # Only decoder parameters are trainable
        all_params = list(pose_decoder.parameters())
        if _probe_loader_type == "totalcapture_real_wrist_flu":
            from imu_lm.data.totalcapture_wrist_loader_FLU import make_totalcapture_real_wrist_loaders_flu as _make_tc_p
            _tc_root_p = probe_cfg.get("pkl_root") or train_pkl_root or "/mnt/nas2/DIP_recalculate"
            def _expand_tc_subjects_p(subjects):
                return [f for s in (subjects or [])
                        for f in sorted(_glob.glob(os.path.join(_tc_root_p, f"{s}_*.pkl")))]
            _tc_train_p = (train_files if probe_cfg.get("train_files")
                           else _expand_tc_subjects_p(probe_cfg.get("train_subjects")))
            _tc_val_p   = (val_files   if probe_cfg.get("val_files")
                           else _expand_tc_subjects_p(probe_cfg.get("val_subjects")))
            loaders = _make_tc_p(
                cfg=cfg,
                pkl_root=_tc_root_p,
                train_files=_tc_train_p,
                val_files=_tc_val_p,
                target_hz=int(probe_cfg.get("target_hz", 60)),
                add_gravity=bool(probe_cfg.get("add_gravity", True)),
                use_gt_ori=bool(probe_cfg.get("use_gt_ori", False)),
                num_joints=num_joints,
            )
        elif _probe_loader_type == "totalcapture_syn_wrist_flu":
            from imu_lm.data.totalcapture_wrist_loader_FLU_syn import make_totalcapture_syn_wrist_loaders_flu as _make_tc_syn_p
            _tc_root_p = probe_cfg.get("pkl_root") or train_pkl_root or "/mnt/nas2/DIP_recalculate"
            def _expand_tc_subjects_p(subjects):
                return [f for s in (subjects or [])
                        for f in sorted(_glob.glob(os.path.join(_tc_root_p, f"{s}_*.pkl")))]
            _tc_train_p = (train_files if probe_cfg.get("train_files")
                           else _expand_tc_subjects_p(probe_cfg.get("train_subjects")))
            _tc_val_p   = (val_files   if probe_cfg.get("val_files")
                           else _expand_tc_subjects_p(probe_cfg.get("val_subjects")))
            loaders = _make_tc_syn_p(
                cfg=cfg,
                pkl_root=_tc_root_p,
                train_files=_tc_train_p,
                val_files=_tc_val_p,
                target_hz=int(probe_cfg.get("target_hz", 60)),
                add_gravity=bool(probe_cfg.get("add_gravity", True)),
                official_dir=probe_cfg.get("official_dir"),
                gyro_white_noise_std=float(probe_cfg.get("gyro_white_noise_std", 0.0)),
                gyro_bias_drift_std=float(probe_cfg.get("gyro_bias_drift_std", 0.0)),
                num_joints=num_joints,
            )
        elif _probe_loader_type in ("imuposer_flu", "imuposer_flu_16hz_to_50hz", "imuposer_full_body_flu"):
            loaders = _make_loaders(
                cfg=cfg,
                train_files=train_files,
                val_files=val_files,
                add_gravity=bool(probe_cfg.get("add_gravity", False)),
            )
        elif _probe_loader_type in ("nymeria_real_wrist_flu", "nymeria_real_wrist_full_body_flu",
                                     "nymeria_real_wrist_full_body_global",
                                     "nymeria_syn_wrist_full_body_global"):
            loaders = _make_loaders(
                cfg=cfg,
                pkl_root="/",
                train_files=train_files,
                val_files=val_files,
                add_gravity=bool(probe_cfg.get("add_gravity", True)),
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

    elif objective_type == "odom_nymeria_real_wrist_flu":
        # Odometry regression using REAL wrist IMU (FLU frame) → per-patch FLU velocity [B, Pt, 4, 3]
        # Label is flattened to [B, Pt, 12] before MSE (4 sub-patches × 3-D velocity).
        # Encoder + odom decoder both trained from scratch (fully supervised).
        import glob as _glob
        odom_rw_cfg = cfg_get(cfg, ["objective", "odom_nymeria_real_wrist_flu"], {}) or {}
        vit_cfg_    = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        from imu_lm.data.nymeria_real_wrist_odom_loader_FLU import make_nymeria_real_wrist_odom_loaders_flu

        train_pkl_root = odom_rw_cfg.get("train_pkl_root")
        val_pkl_root   = odom_rw_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, odom_rw_cfg.get("train_files"))
        _frac = odom_rw_cfg.get("train_fraction")
        if _frac is not None:
            train_files = train_files[:max(1, int(len(train_files) * float(_frac)))]
        val_files   = _expand_files(val_pkl_root,   odom_rw_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
            out_dim=12,   # 4 sub-patches × 3D FLU velocity
        )

        def objective_fn(batch, model, cfg):
            x, y = batch
            # y: [B, N_patches, 4, 3] → [B, N_patches, 12]
            y_flat = y.reshape(y.shape[0], y.shape[1], -1)
            return odom_1d_obj.forward_loss((x, y_flat), encoder, odom_decoder, cfg)

        all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
        loaders = make_nymeria_real_wrist_odom_loaders_flu(
            cfg=cfg,
            pkl_root="/",
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(odom_rw_cfg.get("add_gravity", True)),
        )

    elif objective_type == "odom_probe_nymeria_real_wrist_flu":
        # Frozen pretrained encoder + trainable ViT1DOdomDecoder on Nymeria real wrist FLU data.
        # Encoder loaded from a pretrained run (e.g. full-body pose supervised); all encoder params frozen.
        import glob as _glob
        odom_probe_rw_cfg = cfg_get(cfg, ["objective", "odom_probe_nymeria_real_wrist_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        # ---- load & freeze pretrained encoder --------------------------------
        pretrained_run  = odom_probe_rw_cfg.get("pretrained_run")
        pretrained_ckpt = odom_probe_rw_cfg.get("pretrained_ckpt", "best")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt

        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[odom_probe_nymeria_real_wrist_flu] pretrained encoder checkpoint not found: {enc_ckpt_path}"
            )

        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info("[odom_probe_nymeria_real_wrist_flu] loading pretrained encoder from %s", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(enc_state["model"], strict=True)
        _log.info("[odom_probe_nymeria_real_wrist_flu] encoder loaded — freezing all encoder parameters")
        encoder.requires_grad_(False)
        encoder.eval()

        from imu_lm.data.nymeria_real_wrist_odom_loader_FLU import make_nymeria_real_wrist_odom_loaders_flu

        train_pkl_root = odom_probe_rw_cfg.get("train_pkl_root")
        val_pkl_root   = odom_probe_rw_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, odom_probe_rw_cfg.get("train_files"))
        _frac = odom_probe_rw_cfg.get("train_fraction")
        if _frac is not None:
            train_files = train_files[:max(1, int(len(train_files) * float(_frac)))]
        val_files   = _expand_files(val_pkl_root,   odom_probe_rw_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
            out_dim=12,   # 4 sub-patches × 3D FLU velocity
        )

        def objective_fn(batch, model, cfg):
            encoder.eval()
            x_b, y_b = batch
            # y_b: [B, N_patches, 4, 3] → [B, N_patches, 12]
            y_flat = y_b.reshape(y_b.shape[0], y_b.shape[1], -1).float()
            with torch.no_grad():
                tokens, _ = encoder.forward_tokens(x_b.float())
            pred = odom_decoder(tokens)
            loss = torch.nn.functional.mse_loss(pred, y_flat)
            with torch.no_grad():
                mae = (pred.detach() - y_flat).abs().mean()
            return loss, {"loss": float(loss.detach()),
                          "odom_mse": float(loss.detach()),
                          "odom_mae": float(mae)}

        all_params = list(odom_decoder.parameters())
        loaders = make_nymeria_real_wrist_odom_loaders_flu(
            cfg=cfg,
            pkl_root="/",
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(odom_probe_rw_cfg.get("add_gravity", True)),
        )

    elif objective_type == "odom_finetune_nymeria_real_wrist_flu":
        # Warm-started encoder (stays trainable) + trainable ViT1DOdomDecoder on Nymeria real wrist FLU data.
        # Encoder loaded from a pretrained run (e.g. full-body pose supervised); encoder remains trainable.
        import glob as _glob
        odom_ft_rw_cfg = cfg_get(cfg, ["objective", "odom_finetune_nymeria_real_wrist_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        # ---- warm-start encoder (stays trainable) ----------------------------
        pretrained_run  = odom_ft_rw_cfg.get("pretrained_run")
        pretrained_ckpt = odom_ft_rw_cfg.get("pretrained_ckpt", "best")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt

        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[odom_finetune_nymeria_real_wrist_flu] pretrained encoder checkpoint not found: {enc_ckpt_path}"
            )

        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info("[odom_finetune_nymeria_real_wrist_flu] warm-starting encoder from %s (encoder stays trainable)", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(enc_state["model"], strict=True)

        from imu_lm.data.nymeria_real_wrist_odom_loader_FLU import make_nymeria_real_wrist_odom_loaders_flu

        train_pkl_root = odom_ft_rw_cfg.get("train_pkl_root")
        val_pkl_root   = odom_ft_rw_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, odom_ft_rw_cfg.get("train_files"))
        _frac = odom_ft_rw_cfg.get("train_fraction")
        if _frac is not None:
            train_files = train_files[:max(1, int(len(train_files) * float(_frac)))]
        val_files   = _expand_files(val_pkl_root,   odom_ft_rw_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
            out_dim=12,   # 4 sub-patches × 3D FLU velocity
        )

        def objective_fn(batch, model, cfg):
            x, y = batch
            # y: [B, N_patches, 4, 3] → [B, N_patches, 12]
            y_flat = y.reshape(y.shape[0], y.shape[1], -1)
            return odom_1d_obj.forward_loss((x, y_flat), encoder, odom_decoder, cfg)

        all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
        loaders = make_nymeria_real_wrist_odom_loaders_flu(
            cfg=cfg,
            pkl_root="/",
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(odom_ft_rw_cfg.get("add_gravity", True)),
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

    elif objective_type == "mae_totalcapture_flu":
        # MAE pretraining on TotalCapture DIP pkl files in the right-wrist FLU body frame.
        # Loader yields (x [6,T], y [N_joints,N_patches,...]); MAE uses only x.
        # Supports subject-based expansion (train_subjects / val_subjects) or explicit
        # file lists (train_files / val_files) relative to pkl_root.
        from imu_lm.data.totalcapture_wrist_loader_FLU import make_totalcapture_real_wrist_loaders_flu
        import glob as _glob
        mae_tc_cfg   = cfg_get(cfg, ["objective", "mae_totalcapture_flu"], {}) or {}
        _tc_pkl_root = mae_tc_cfg.get("pkl_root") or "/mnt/nas2/DIP_recalculate"

        def _expand_tc_subjects_mae(subjects, root):
            paths = []
            for subj in (subjects or []):
                paths.extend(sorted(_glob.glob(os.path.join(root, f"{subj}_*.pkl"))))
            return paths

        def _abs_tc_mae(files):
            return [f if os.path.isabs(f) else os.path.join(_tc_pkl_root, f)
                    for f in (files or [])]

        _train_subjs = mae_tc_cfg.get("train_subjects")
        _val_subjs   = mae_tc_cfg.get("val_subjects")
        _train_files = mae_tc_cfg.get("train_files")
        _val_files   = mae_tc_cfg.get("val_files")

        train_files = _abs_tc_mae(_train_files) if _train_files else _expand_tc_subjects_mae(_train_subjs, _tc_pkl_root)
        val_files   = _abs_tc_mae(_val_files)   if _val_files   else _expand_tc_subjects_mae(_val_subjs,   _tc_pkl_root)

        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )
        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders = make_totalcapture_real_wrist_loaders_flu(
            cfg=cfg,
            pkl_root=_tc_pkl_root,
            train_files=train_files,
            val_files=val_files,
            target_hz=int(mae_tc_cfg.get("target_hz", 50)),
            add_gravity=bool(mae_tc_cfg.get("add_gravity", True)),
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

    elif objective_type == "local_disp_imuposer_flu":
        # Local-frame displacement + delta yaw regression on IMUPoser pkls.
        # Input : right-wrist FLU acc+gyro  [B, 6, T]
        # Output: [B, N_patches, 4]  — [dx, dy, dz, dψ] per patch
        #   [:, :, :3] = pelvis-body-frame displacement (metres, Y-up SMPL)
        #   [:, :,  3] = delta yaw around world-up axis (radians, Z-up world)
        # This avoids the global-yaw-unobservable problem of world-frame XYZ labels.
        from imu_lm.data.imuposer_wrist_local_odom_loader_FLU import make_imuposer_local_disp_flu_loaders
        import glob as _glob
        ld_ip_cfg = cfg_get(cfg, ["objective", "local_disp_imuposer_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        train_pkl_root = ld_ip_cfg.get("train_pkl_root")
        val_pkl_root   = ld_ip_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_ld_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_ld_files(train_pkl_root, ld_ip_cfg.get("train_files"))
        val_files   = _expand_ld_files(val_pkl_root,   ld_ip_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
            out_dim=4,   # [dx, dy, dz, dψ]
        )

        def objective_fn(batch, model, cfg):
            return odom_1d_obj.forward_loss(batch, encoder, odom_decoder, cfg)

        all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
        loaders = make_imuposer_local_disp_flu_loaders(
            cfg=cfg,
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(ld_ip_cfg.get("add_gravity", False)),
        )

    elif objective_type == "local_vel_imuposer_flu":
        # Fully-supervised FLU-velocity regression on IMUPoser pkls.
        # Input : right-wrist FLU acc+gyro  [B, 6, T]
        # Output: per-patch 4×3-D pelvis velocity in wrist FLU frame [B, N_patches, 4, 3]
        #   4 sub-patches per patch, each Q = patch_size // 4 frames (0.08 s @ 50 Hz).
        #   v_flu[p, k] = R_X_from_world[t_mid] @ v_world   (m/s)
        # Decoder produces [B, N_patches, 12]; y is flattened to the same shape for MSE.
        from imu_lm.data.imuposer_wrist_local_odom_loader_FLU import make_imuposer_local_disp_flu_loaders
        import glob as _glob
        lv_ip_cfg = cfg_get(cfg, ["objective", "local_vel_imuposer_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        train_pkl_root = lv_ip_cfg.get("train_pkl_root")
        val_pkl_root   = lv_ip_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_lv_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_lv_files(train_pkl_root, lv_ip_cfg.get("train_files"))
        val_files   = _expand_lv_files(val_pkl_root,   lv_ip_cfg.get("val_files"))

        # Decoder: [B, N_patches, 12]  (12 = 4 sub-patches × 3 velocity components)
        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
            out_dim=12,   # 4 sub-patches × 3D FLU velocity
        )

        def objective_fn(batch, model, cfg):
            x, y = batch
            # y: [B, N_patches, 4, 3] → [B, N_patches, 12] for MSE
            y_flat = y.reshape(y.shape[0], y.shape[1], -1)
            return odom_1d_obj.forward_loss((x, y_flat), encoder, odom_decoder, cfg)

        all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
        loaders = make_imuposer_local_disp_flu_loaders(
            cfg=cfg,
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(lv_ip_cfg.get("add_gravity", False)),
        )

    elif objective_type == "local_vel_probe_imuposer_flu":
        # Frozen pretrained encoder + trainable ViT1DOdomDecoder (out_dim=12).
        # Input : right-wrist FLU acc+gyro  [B, 6, T]
        # Output: per-patch 4×3-D pelvis velocity in wrist FLU frame [B, N_patches, 12]
        from imu_lm.data.imuposer_wrist_local_odom_loader_FLU import make_imuposer_local_disp_flu_loaders
        import glob as _glob
        import logging as _logging
        lv_probe_cfg = cfg_get(cfg, ["objective", "local_vel_probe_imuposer_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        # ---- load & freeze pretrained encoder --------------------------------
        pretrained_run  = lv_probe_cfg.get("pretrained_run")
        pretrained_ckpt = lv_probe_cfg.get("pretrained_ckpt", "best")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt

        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[local_vel_probe_imuposer_flu] pretrained encoder checkpoint not found: {enc_ckpt_path}"
            )

        _log = _logging.getLogger(__name__)
        _log.info("[local_vel_probe_imuposer_flu] loading pretrained encoder from %s", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(enc_state["model"], strict=True)
        _log.info("[local_vel_probe_imuposer_flu] encoder loaded — freezing all encoder parameters")
        encoder.requires_grad_(False)
        encoder.eval()

        train_pkl_root = lv_probe_cfg.get("train_pkl_root")
        val_pkl_root   = lv_probe_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_lvp_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_lvp_files(train_pkl_root, lv_probe_cfg.get("train_files"))
        val_files   = _expand_lvp_files(val_pkl_root,   lv_probe_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
            out_dim=12,   # 4 sub-patches × 3D FLU velocity
        )

        def objective_fn(batch, model, cfg):
            encoder.eval()
            x, y = batch
            y_flat = y.reshape(y.shape[0], y.shape[1], -1)
            with torch.no_grad():
                tokens, _ = encoder.forward_tokens(x.float())
            pred = odom_decoder(tokens)
            loss = torch.nn.functional.mse_loss(pred, y_flat.float())
            with torch.no_grad():
                mae = (pred.detach() - y_flat.float()).abs().mean()
            return loss, {"loss": float(loss.detach()),
                          "odom_mse": float(loss.detach()),
                          "odom_mae": float(mae)}

        all_params = list(odom_decoder.parameters())
        loaders = make_imuposer_local_disp_flu_loaders(
            cfg=cfg,
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(lv_probe_cfg.get("add_gravity", False)),
        )

    elif objective_type == "local_vel_finetune_imuposer_flu":
        # Warm-started encoder (stays trainable) + trainable ViT1DOdomDecoder (out_dim=12).
        # Both encoder and decoder are updated jointly from a pretrained MAE checkpoint.
        from imu_lm.data.imuposer_wrist_local_odom_loader_FLU import make_imuposer_local_disp_flu_loaders
        import glob as _glob
        import logging as _logging
        lv_ft_cfg = cfg_get(cfg, ["objective", "local_vel_finetune_imuposer_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        # ---- warm-start encoder (stays trainable) ----------------------------
        pretrained_run  = lv_ft_cfg.get("pretrained_run")
        pretrained_ckpt = lv_ft_cfg.get("pretrained_ckpt", "best")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt

        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[local_vel_finetune_imuposer_flu] pretrained encoder checkpoint not found: {enc_ckpt_path}"
            )

        _log = _logging.getLogger(__name__)
        _log.info("[local_vel_finetune_imuposer_flu] warm-starting encoder from %s (encoder stays trainable)", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(enc_state["model"], strict=True)

        train_pkl_root = lv_ft_cfg.get("train_pkl_root")
        val_pkl_root   = lv_ft_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_lvft_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_lvft_files(train_pkl_root, lv_ft_cfg.get("train_files"))
        val_files   = _expand_lvft_files(val_pkl_root,   lv_ft_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
            out_dim=12,   # 4 sub-patches × 3D FLU velocity
        )

        def objective_fn(batch, model, cfg):
            x, y = batch
            y_flat = y.reshape(y.shape[0], y.shape[1], -1)
            return odom_1d_obj.forward_loss((x, y_flat), encoder, odom_decoder, cfg)

        all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
        loaders = make_imuposer_local_disp_flu_loaders(
            cfg=cfg,
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(lv_ft_cfg.get("add_gravity", False)),
        )

    elif objective_type == "odom_totalcapture_flu":
        # Supervised odometry on TotalCapture DIP pkls, right-wrist FLU frame.
        # No pretrained encoder — trained from scratch end-to-end.
        # Per-patch root displacement is derived by double-integrating world-frame acc.
        import glob as _glob
        from imu_lm.data.totalcapture_wrist_odom_loader_FLU import make_totalcapture_odom_flu_loaders
        odom_tc_cfg = cfg_get(cfg, ["objective", "odom_totalcapture_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        _tc_pkl_root = odom_tc_cfg.get("pkl_root") or "/mnt/nas2/DIP_recalculate"

        def _expand_tc_subjects(subjects, root):
            """Expand subject IDs (e.g. ['s1','s2']) → sorted list of pkl paths."""
            import re
            paths = []
            for subj in (subjects or []):
                paths.extend(sorted(_glob.glob(os.path.join(root, f"{subj}_*.pkl"))))
            return paths

        _train_subjs  = odom_tc_cfg.get("train_subjects")
        _val_subjs    = odom_tc_cfg.get("val_subjects")
        _train_files  = odom_tc_cfg.get("train_files")
        _val_files    = odom_tc_cfg.get("val_files")

        def _abs_tc(files):
            return [f if os.path.isabs(f) else os.path.join(_tc_pkl_root, f)
                    for f in (files or [])]

        train_files = _abs_tc(_train_files) if _train_files else _expand_tc_subjects(_train_subjs, _tc_pkl_root)
        val_files   = _abs_tc(_val_files)   if _val_files   else _expand_tc_subjects(_val_subjs,   _tc_pkl_root)

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
        )

        def objective_fn(batch, model, cfg):
            return odom_1d_obj.forward_loss(batch, encoder, odom_decoder, cfg)

        all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
        loaders = make_totalcapture_odom_flu_loaders(
            cfg=cfg,
            pkl_root=_tc_pkl_root,
            train_files=train_files,
            val_files=val_files,
            target_hz=int(odom_tc_cfg.get("target_hz", 50)),
            add_gravity=bool(odom_tc_cfg.get("add_gravity", True)),
            use_gt_ori=bool(odom_tc_cfg.get("use_gt_ori", False)),
        )

    elif objective_type == "odom_probe_totalcapture_flu":
        # Frozen pretrained encoder + trainable ViT1DOdomDecoder on TotalCapture FLU data.
        import glob as _glob
        from imu_lm.data.totalcapture_wrist_odom_loader_FLU import make_totalcapture_odom_flu_loaders
        odom_probe_tc_cfg = cfg_get(cfg, ["objective", "odom_probe_totalcapture_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        # ---- load & freeze pretrained encoder --------------------------------
        pretrained_run  = odom_probe_tc_cfg.get("pretrained_run")
        pretrained_ckpt = odom_probe_tc_cfg.get("pretrained_ckpt", "best")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt

        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[odom_probe_totalcapture_flu] pretrained encoder checkpoint not found: {enc_ckpt_path}"
            )

        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info("[odom_probe_totalcapture_flu] loading pretrained encoder from %s", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(enc_state["model"], strict=True)
        _log.info("[odom_probe_totalcapture_flu] encoder loaded — freezing all encoder parameters")
        encoder.requires_grad_(False)
        encoder.eval()

        _tc_pkl_root = odom_probe_tc_cfg.get("pkl_root") or "/mnt/nas2/DIP_recalculate"

        def _expand_tc_subjects_probe(subjects, root):
            paths = []
            for subj in (subjects or []):
                paths.extend(sorted(_glob.glob(os.path.join(root, f"{subj}_*.pkl"))))
            return paths

        _train_subjs  = odom_probe_tc_cfg.get("train_subjects")
        _val_subjs    = odom_probe_tc_cfg.get("val_subjects")
        _train_files  = odom_probe_tc_cfg.get("train_files")
        _val_files    = odom_probe_tc_cfg.get("val_files")

        def _abs_tc_probe(files):
            return [f if os.path.isabs(f) else os.path.join(_tc_pkl_root, f)
                    for f in (files or [])]

        train_files = _abs_tc_probe(_train_files) if _train_files else _expand_tc_subjects_probe(_train_subjs, _tc_pkl_root)
        val_files   = _abs_tc_probe(_val_files)   if _val_files   else _expand_tc_subjects_probe(_val_subjs,   _tc_pkl_root)

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
        loaders = make_totalcapture_odom_flu_loaders(
            cfg=cfg,
            pkl_root=_tc_pkl_root,
            train_files=train_files,
            val_files=val_files,
            target_hz=int(odom_probe_tc_cfg.get("target_hz", 50)),
            add_gravity=bool(odom_probe_tc_cfg.get("add_gravity", True)),
            use_gt_ori=bool(odom_probe_tc_cfg.get("use_gt_ori", False)),
        )

    elif objective_type == "odom_finetune_totalcapture_flu":
        # Warm-started encoder (stays trainable) + ViT1DOdomDecoder on TotalCapture FLU data.
        import glob as _glob
        from imu_lm.data.totalcapture_wrist_odom_loader_FLU import make_totalcapture_odom_flu_loaders
        odom_ft_tc_cfg = cfg_get(cfg, ["objective", "odom_finetune_totalcapture_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        # ---- warm-start encoder (stays trainable) ----------------------------
        pretrained_run  = odom_ft_tc_cfg.get("pretrained_run")
        pretrained_ckpt = odom_ft_tc_cfg.get("pretrained_ckpt", "best")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt

        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[odom_finetune_totalcapture_flu] pretrained encoder checkpoint not found: {enc_ckpt_path}"
            )

        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info("[odom_finetune_totalcapture_flu] warm-starting encoder from %s (encoder stays trainable)", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(enc_state["model"], strict=True)

        _tc_pkl_root = odom_ft_tc_cfg.get("pkl_root") or "/mnt/nas2/DIP_recalculate"

        def _expand_tc_subjects_ft(subjects, root):
            paths = []
            for subj in (subjects or []):
                paths.extend(sorted(_glob.glob(os.path.join(root, f"{subj}_*.pkl"))))
            return paths

        _train_subjs  = odom_ft_tc_cfg.get("train_subjects")
        _val_subjs    = odom_ft_tc_cfg.get("val_subjects")
        _train_files  = odom_ft_tc_cfg.get("train_files")
        _val_files    = odom_ft_tc_cfg.get("val_files")

        def _abs_tc_ft(files):
            return [f if os.path.isabs(f) else os.path.join(_tc_pkl_root, f)
                    for f in (files or [])]

        train_files = _abs_tc_ft(_train_files) if _train_files else _expand_tc_subjects_ft(_train_subjs, _tc_pkl_root)
        val_files   = _abs_tc_ft(_val_files)   if _val_files   else _expand_tc_subjects_ft(_val_subjs,   _tc_pkl_root)

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
        )

        def objective_fn(batch, model, cfg):
            return odom_1d_obj.forward_loss(batch, encoder, odom_decoder, cfg)

        all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
        loaders = make_totalcapture_odom_flu_loaders(
            cfg=cfg,
            pkl_root=_tc_pkl_root,
            train_files=train_files,
            val_files=val_files,
            target_hz=int(odom_ft_tc_cfg.get("target_hz", 50)),
            add_gravity=bool(odom_ft_tc_cfg.get("add_gravity", True)),
            use_gt_ori=bool(odom_ft_tc_cfg.get("use_gt_ori", False)),
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
            ContrastiveCLIPHead, CLIPTextEncoder, TextProjectionMLP
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
        use_text_mlp    = bool(clip_cfg.get("text_projection_mlp", False))
        text_mlp_hidden = clip_cfg.get("text_mlp_hidden_dim", None)

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

        # -- text-side MLP projection (trainable, optional) -----------------
        text_mlp = None
        if use_text_mlp:
            text_mlp = TextProjectionMLP(
                embed_dim  = clip_embed_dim,
                hidden_dim = text_mlp_hidden,
            )

        def objective_fn(batch, model, cfg):
            return clip_obj.forward_loss(batch, encoder, clip_head, text_encoder, cfg,
                                         text_mlp=text_mlp)

        if freeze_imu:
            all_params = list(clip_head.parameters())
        else:
            all_params = list(encoder.parameters()) + list(clip_head.parameters())
        if not freeze_text and text_encoder is not None:
            all_params += list(text_encoder.model.parameters())
        if text_mlp is not None:
            all_params += list(text_mlp.parameters())

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

    elif objective_type == "mae_ocdetect_flu":
        # MAE pretraining on OCDetect wrist-IMU data (FLU frame, 50 Hz, 2.56 s windows).
        # 678 recordings, 22 OCD subjects; data converted Android → FLU in the loader.
        # Labels (compulsive) are discarded — only x is used for reconstruction.
        from imu_lm.data.OCDetect_dataloader import (
            make_ocdetect_loaders_random,
            DATA_DIR_DEFAULT as _OCDetect_DATA_DIR,
        )
        ocdetect_cfg = cfg_get(cfg, ["objective", "mae_ocdetect_flu"], {}) or {}
        data_dir_    = ocdetect_cfg.get("data_dir", _OCDetect_DATA_DIR)
        splits_      = tuple(ocdetect_cfg.get("splits", [0.8, 0.1, 0.1]))
        num_workers_ = int(cfg_get(cfg, ["data", "num_workers"], 0))
        pin_memory_  = bool(cfg_get(cfg, ["data", "pin_memory"], False))
        batch_size_  = int(cfg_get(cfg, ["data", "batch_size"], 256))
        eval_bs_     = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size_))

        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )
        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders = make_ocdetect_loaders_random(
            data_dir=data_dir_,
            splits=splits_,
            batch_size=batch_size_,
            eval_batch_size=eval_bs_,
            num_workers=num_workers_,
            pin_memory=pin_memory_,
            cfg=cfg,
        )

    elif objective_type == "surfacexr_gesture_supervised":
        # Fully-supervised 9-class gesture classification on SurfaceXR smartwatch IMU.
        # Encoder (ViT1D, in_channels=6) + LinearHead trained end-to-end.
        # Data: Google Pixel Watch 3, raw acc/gyro downsampled 200→50 Hz, 2.56 s windows
        #       (128 samples), 6 FLU channels — identical format to all other 50 Hz datasets.
        # Labels: 0=no_gesture, 1=SINGLE_TAP, 2=DOUBLE_TAP, 3=PINCH_IN, 4=PINCH_OUT,
        #          5=SWIPE_UP, 6=SWIPE_DOWN, 7=SWIPE_LEFT, 8=SWIPE_RIGHT
        # Split: LOPO (leave-one-person-out) by test_user.
        from imu_lm.data.surfaceXR_dataloader import (
            make_surfaceXR_loaders_lopo_200hz,
            make_surfaceXR_loaders_random_200hz,
            make_surfaceXR_loaders_lopo_50hz,
            make_surfaceXR_loaders_random_50hz,
            ALL_USERS as _SXR_ALL_USERS,
            RAW_IMU_PKL_DEFAULT as _SXR_RAW_PKL,
        )
        import logging as _logging
        _sxr_log = _logging.getLogger(__name__)

        sxr_cfg = cfg_get(cfg, ["objective", "surfaceXR_gesture_supervised"], {}) or {}  # key in YAML is mixed-case
        pkl_path    = sxr_cfg.get("pkl_path", _SXR_RAW_PKL)
        test_user   = sxr_cfg.get("test_user", None)
        num_classes = int(sxr_cfg.get("num_classes", 9))
        apply_flu   = bool(sxr_cfg.get("apply_flu", True))
        split_type  = str(sxr_cfg.get("split", "lopo")).lower()
        sample_rate_hz = int(sxr_cfg.get("sample_rate_hz", 50))

        batch_size_  = int(cfg_get(cfg, ["data", "batch_size"],       256))
        eval_bs_     = int(cfg_get(cfg, ["data", "eval_batch_size"],  batch_size_))
        n_workers_   = int(cfg_get(cfg, ["data", "num_workers"],       0))
        pin_mem_     = bool(cfg_get(cfg, ["data", "pin_memory"],    False))

        if sample_rate_hz == 50:
            make_lopo = make_surfaceXR_loaders_lopo_50hz
            make_random = make_surfaceXR_loaders_random_50hz
        elif sample_rate_hz == 200:
            make_lopo = make_surfaceXR_loaders_lopo_200hz
            make_random = make_surfaceXR_loaders_random_200hz
        else:
            raise ValueError(
                f"[surfaceXR_gesture_supervised] unsupported sample_rate_hz={sample_rate_hz}; expected 50 or 200"
            )

        if split_type == "lopo":
            if test_user is None:
                raise ValueError(
                    "[surfaceXR_gesture_supervised] 'test_user' must be set for LOPO split"
                )
            loaders = make_lopo(
                pkl_path=pkl_path,
                test_user=test_user,
                label_type="gesture",
                batch_size=batch_size_,
                eval_batch_size=eval_bs_,
                num_workers=n_workers_,
                pin_memory=pin_mem_,
                apply_flu=apply_flu,
            )
        else:
            loaders = make_random(
                pkl_path=pkl_path,
                label_type="gesture",
                batch_size=batch_size_,
                eval_batch_size=eval_bs_,
                num_workers=n_workers_,
                pin_memory=pin_mem_,
                apply_flu=apply_flu,
            )

        head = LinearHead(encoder.embed_dim, num_classes)

        # Optional: warm-start encoder from a pretrained run
        pretrained_run  = sxr_cfg.get("pretrained_run",  None)
        pretrained_ckpt = sxr_cfg.get("pretrained_ckpt", "best")
        if pretrained_run:
            runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
            if pretrained_ckpt in ("best", "latest"):
                enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                             "checkpoints", f"{pretrained_ckpt}.pt")
            else:
                enc_ckpt_path = pretrained_ckpt
            if not os.path.exists(enc_ckpt_path):
                raise FileNotFoundError(
                    f"[surfaceXR_gesture_supervised] pretrained encoder not found: {enc_ckpt_path}"
                )
            _sxr_log.info("[surfaceXR_gesture_supervised] warm-starting encoder from %s", enc_ckpt_path)
            enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
            encoder.load_state_dict(enc_state["model"], strict=True)

        _sxr_log.info(
            "[surfaceXR_gesture_supervised] split=%s test_user=%s num_classes=%d sample_rate_hz=%d",
            split_type, test_user, num_classes, sample_rate_hz,
        )

        def objective_fn(batch, model, cfg):
            return supervised_obj.forward_loss(batch, encoder, head, cfg, raw_to_idx=None)

        all_params = list(encoder.parameters()) + list(head.parameters())

    elif objective_type == "surfacexr_gesture_probe":
        # Linear probe: frozen pretrained encoder + trainable LinearHead.
        # Same 50 Hz / 6-channel / 128-sample data as surfaceXR_gesture_supervised.
        from imu_lm.data.surfaceXR_dataloader import (
            make_surfaceXR_loaders_lopo_50hz,
            make_surfaceXR_loaders_random_50hz,
            RAW_IMU_PKL_DEFAULT as _SXR_RAW_PKL_PROBE,
        )
        import logging as _logging
        _sxr_log = _logging.getLogger(__name__)

        sxr_probe_cfg = cfg_get(cfg, ["objective", "surfaceXR_gesture_probe"], {}) or {}
        pkl_path    = sxr_probe_cfg.get("pkl_path", _SXR_RAW_PKL_PROBE)
        test_user   = sxr_probe_cfg.get("test_user", None)
        num_classes = int(sxr_probe_cfg.get("num_classes", 9))
        apply_flu   = bool(sxr_probe_cfg.get("apply_flu", True))
        split_type  = str(sxr_probe_cfg.get("split", "lopo")).lower()

        # Load & freeze pretrained encoder
        pretrained_run  = sxr_probe_cfg.get("pretrained_run")
        pretrained_ckpt = sxr_probe_cfg.get("pretrained_ckpt", "best")
        if not pretrained_run:
            raise ValueError(
                "[surfaceXR_gesture_probe] 'pretrained_run' must be set for probing"
            )
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt
        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[surfaceXR_gesture_probe] pretrained encoder not found: {enc_ckpt_path}"
            )
        _sxr_log.info("[surfaceXR_gesture_probe] loading pretrained encoder from %s", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(enc_state["model"], strict=True)
        _sxr_log.info("[surfaceXR_gesture_probe] encoder loaded — freezing all encoder parameters")
        encoder.requires_grad_(False)
        encoder.eval()

        batch_size_  = int(cfg_get(cfg, ["data", "batch_size"],       256))
        eval_bs_     = int(cfg_get(cfg, ["data", "eval_batch_size"],  batch_size_))
        n_workers_   = int(cfg_get(cfg, ["data", "num_workers"],       0))
        pin_mem_     = bool(cfg_get(cfg, ["data", "pin_memory"],    False))

        if split_type == "lopo":
            if test_user is None:
                raise ValueError(
                    "[surfaceXR_gesture_probe] 'test_user' must be set for LOPO split"
                )
            loaders = make_surfaceXR_loaders_lopo_50hz(
                pkl_path=pkl_path,
                test_user=test_user,
                label_type="gesture",
                batch_size=batch_size_,
                eval_batch_size=eval_bs_,
                num_workers=n_workers_,
                pin_memory=pin_mem_,
                apply_flu=apply_flu,
            )
        else:
            loaders = make_surfaceXR_loaders_random_50hz(
                pkl_path=pkl_path,
                label_type="gesture",
                batch_size=batch_size_,
                eval_batch_size=eval_bs_,
                num_workers=n_workers_,
                pin_memory=pin_mem_,
                apply_flu=apply_flu,
            )

        head = LinearHead(encoder.embed_dim, num_classes)

        _sxr_log.info(
            "[surfaceXR_gesture_probe] split=%s test_user=%s num_classes=%d pretrained_run=%s",
            split_type, test_user, num_classes, pretrained_run,
        )

        def objective_fn(batch, model, cfg):
            encoder.eval()
            x_b, y_b = batch
            with torch.no_grad():
                z = encoder(x_b.float())
            logits = head(z)
            loss = torch.nn.functional.cross_entropy(logits, y_b.long())
            preds = logits.argmax(dim=-1)
            acc = float((preds == y_b).float().mean())
            return loss, {"loss": float(loss.detach()), "acc": acc}

        all_params = list(head.parameters())

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
                          "watchhand_gesture_supervised_within_subject",
                          "surfacexr_gesture_supervised", "surfacexr_gesture_probe"):
        extra_modules = {"head": head}
    elif objective_type in ("pose", "pose_probe"):
        extra_modules = {"pose_decoder": pose_decoder}
    elif objective_type in ("odom_nymeria", "odom_nymeria_real_wrist_flu",
                            "odom_probe_nymeria_real_wrist_flu",
                            "odom_finetune_nymeria_real_wrist_flu",
                            "odom_probe", "odom_finetune", "odom_imuposer_flu",
                            "odom_probe_imuposer_flu", "odom_finetune_imuposer_flu",
                            "odom_totalcapture_flu", "odom_probe_totalcapture_flu",
                            "odom_finetune_totalcapture_flu", "local_disp_imuposer_flu",
                            "local_vel_imuposer_flu",
                            "local_vel_probe_imuposer_flu", "local_vel_finetune_imuposer_flu"):
        extra_modules = {"odom_decoder": odom_decoder}
    elif objective_type in ("imu_text_clip", "imu_text_clip_flu"):
        # Save projection head + logit_scale; text encoder can be reloaded from HuggingFace
        extra_modules = {"clip_head": clip_head}
        if objective_type == "imu_text_clip_flu" and text_mlp is not None:
            extra_modules["text_mlp"] = text_mlp
    else:
        extra_modules = {"decoder": decoder}
    
    # Load checkpoint if resuming
    resume_path = resolve_resume_path(run_dir, resume_ckpt)
    start_step = load_checkpoint(resume_path, encoder, optimizer, scheduler, extra_modules=extra_modules)
    
    trainer = Trainer(cfg, run_dir)
    trainer.fit(encoder, objective_fn, train_loader, val_loader, optimizer, scheduler, start_step=start_step, extra_modules=extra_modules)


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_pretrain.py to launch training.")
