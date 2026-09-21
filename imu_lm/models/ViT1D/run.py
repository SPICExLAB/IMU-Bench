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
from imu_lm.models.ViT1D.model import ViT1DEncoder, maybe_stats_wrap, ViT1DDecoder, ViT1DPoseDecoder, ViT1DOdomDecoder
from imu_lm.objectives import masked_1d as masked_1d_obj
from imu_lm.objectives import supervised as supervised_obj
from imu_lm.objectives import pose_1d as pose_1d_obj
from imu_lm.objectives import odom_1d as odom_1d_obj
from imu_lm.probe.head import LinearHead
from imu_lm.utils.metrics import compute_metrics
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


def _remap_encoder_state_dict(state_dict: dict) -> dict:
    """Remap old nn.TransformerEncoderLayer key names to TransformerLayerWithDropPath names.

    Old (nn.TransformerEncoderLayer): self_attn.*, linear1.*, linear2.*
    New (TransformerLayerWithDropPath): attn.*, ff.0.*, ff.3.*
    """
    remapped = {}
    for k, v in state_dict.items():
        k = k.replace(".self_attn.", ".attn.")
        k = k.replace(".linear1.", ".ff.0.")
        k = k.replace(".linear2.", ".ff.3.")
        remapped[k] = v
    return remapped


def _load_encoder_state_dict(encoder: "torch.nn.Module", enc_state: dict) -> None:
    """Load a pretrained encoder checkpoint, tolerating checkpoint params that
    belong to an optional sub-module this downstream config doesn't activate
    (e.g. bert_masking's global_mask_token/freq_mask_tokens, or a cls_token,
    when the pretraining run enabled a flag this run leaves off). Those
    params are simply never referenced by forward() when the corresponding
    flag is off here, so dropping them is safe.

    Still fails loudly on *missing* keys (params this encoder needs but the
    checkpoint doesn't have) -- that indicates a genuine architecture
    mismatch, not an inactive optional module, and should not be silenced.
    """
    if hasattr(encoder, "load_pretrained_state"):
        # External encoders (e.g. Inertia1Encoder) own their key layout; the
        # ViT1D self_attn/linear1/linear2 remap below would corrupt it.
        encoder.load_pretrained_state(enc_state)
        return
    remapped = _remap_encoder_state_dict(enc_state["model"])
    result = encoder.load_state_dict(remapped, strict=False)
    if result.missing_keys:
        raise RuntimeError(
            f"Encoder checkpoint is missing required key(s): {result.missing_keys}"
        )
    if result.unexpected_keys:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[encoder-load] ignoring %d checkpoint key(s) not used by this "
            "downstream architecture (likely an optional module inactive here, "
            "e.g. bert_masking off): %s",
            len(result.unexpected_keys), result.unexpected_keys,
        )


def _openwatch_ce_weight(loaders, cfg_block, num_classes):
    """Inverse-frequency class weights over the TRAIN loader's labels when
    `class_weighted_ce: true` is set in the objective block (mirrors the
    OpenWatch paper's §3.3.3 recipe for the §5f fair comparison); else None
    (plain CE, this repo's default). Labels are read from the dataset's
    window/record index so no data pass is needed."""
    if not bool(cfg_block.get("class_weighted_ce", False)):
        return None
    ds = loaders["train_loader"].dataset
    if getattr(ds, "windowed", False):
        ys = [int(ds.records[ri].label) for ri, _s in ds._window_index]
    else:
        ys = [int(r.label) for r in ds.records]
    counts = torch.bincount(torch.tensor(ys), minlength=num_classes).float()
    w = counts.sum() / (num_classes * counts.clamp(min=1.0))
    return w


def main(cfg: Any, run_dir: str, resume_ckpt: Optional[str] = None):
    # Compute input temporal dimension from windowing config
    T, _ = compute_T_and_hop(cfg)
    
    # Get objective type: "mae", "supervised", or "pose"
    objective_type = cfg_get(cfg, ["objective", "type"], "mae").lower()
    
    # Create encoder (always needed). An ``inertia1:`` block selects the
    # Inertia-1 external-baseline adapter (imu_lm/models/baselines/inertia1_encoder.py);
    # otherwise the plain ViT1DEncoder as before (objective branches below may
    # still swap in ViT1DEncoderAltAttn when vit1d.encoder.alt_attention is set).
    if cfg_get(cfg, ["inertia1"], None):
        from imu_lm.models.baselines.inertia1_encoder import Inertia1Encoder
        encoder = Inertia1Encoder(cfg)
    elif cfg_get(cfg, ["normwear"], None):
        # NormWear (Luo et al. 2024) external baseline, the authors' released checkpoint
        from imu_lm.models.baselines.normwear_encoder import NormWearEncoder
        encoder = NormWearEncoder(cfg)
    elif cfg_get(cfg, ["harnet"], None):
        # HarNet (Yuan et al. 2024) external baseline, released UK Biobank weights
        from imu_lm.models.baselines.harnet_encoder import HarnetEncoder
        encoder = HarnetEncoder(cfg)
    else:
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
        "backbone": "inertia1" if cfg_get(cfg, ["inertia1"], None) else "vit1d",
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
        if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
            import logging as _logging
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _logging.getLogger(__name__).info(
                "[supervised] using alternating frame/global attention encoder (VGGT-style)")
        loaders = make_loaders(cfg)
        label_map = build_label_map(loaders["train_loader"], cfg)
        save_label_map(label_map, run_dir)
        raw_to_idx = label_map["raw_to_idx"]
        num_classes = label_map["num_classes"]
        if (cfg_get(cfg, ['dual_stream'], None) or {}).get('enabled', False):
            from imu_lm.models.ViT1D.model import build_dual_stream
            encoder = build_dual_stream(cfg)      # two pretrained stems + gated fusion
        encoder = maybe_stats_wrap(encoder, cfg)   # stats_concat diagnostic (no-op unless enabled)
        head = LinearHead(encoder.embed_dim, num_classes)
        _sup_cw = None
        if bool(cfg_get(cfg, ["objective", "supervised", "class_weighted_ce"], False)):
            _c = torch.tensor(label_map["class_counts"], dtype=torch.float32)
            _sup_cw = _c.sum() / (num_classes * _c.clamp(min=1)); _sup_cw = _sup_cw / _sup_cw.sum() * num_classes
        def objective_fn(batch, model, cfg):
            return supervised_obj.forward_loss(batch, encoder, head, cfg, raw_to_idx=raw_to_idx, class_weights=_sup_cw)
        all_params = list(encoder.parameters()) + list(head.parameters())
    elif objective_type == "pose":
        # Pose regression: encoder + ViT1DPoseDecoder, [B, P_t, J*6]
        # Use full-body loader (24 joints, 144 dims) when num_joints == 24,
        # otherwise use right-arm-chain loader (5 joints, 30 dims).
        vit_cfg  = cfg_get(cfg, ["vit1d"], {}) or {}
        pose_cfg = cfg_get(cfg, ["objective", "pose"], {}) or {}
        dec_cfg  = (vit_cfg.get("pose_decoder") or {})
        patch_size              = int(vit_cfg.get("patch_size", 16))
        num_patches_per_channel = T // patch_size
        num_joints              = int(pose_cfg.get("num_joints", 5))
        sub_patches             = int(dec_cfg.get("sub_patches", 1))

        if bool(vit_cfg.get("encoder", {}).get("alt_attention", False)):
            import logging as _logging
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _logging.getLogger(__name__).info(
                "[pose] using alternating frame/global attention encoder (VGGT-style)")

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
        elif _loader_type == "nymeria_syn_wrist_full_body_flu":
            from imu_lm.data.nymeria_syn_wrist_full_body_loader_FLU import make_nymeria_syn_wrist_full_body_loaders_flu as _make_pose_loaders
        elif _loader_type == "bones_seed_wrist_full_body_flu":
            from imu_lm.data.bones_seed_wrist_full_body_loader_FLU import make_bones_seed_wrist_full_body_loaders_flu as _make_pose_loaders
        elif _loader_type == "imuposer_flu":
            from imu_lm.data.imuposer_wrist_loader_FLU import make_imuposer_wrist_flu_loaders as _make_pose_loaders
        elif _loader_type == "imuposer_flu_16hz_to_50hz":
            from imu_lm.data.imuposer_wrist_16hz_to_50hz_loader_FLU import make_imuposer_wrist_16hz_to_50hz_flu_loaders as _make_pose_loaders
        elif _loader_type == "imuposer_full_body_flu":
            from imu_lm.data.imuposer_wrist_full_body_loader_FLU import make_imuposer_wrist_full_body_flu_loaders as _make_pose_loaders
        elif _loader_type == "imuposer_full_body_23_flu":
            from imu_lm.data.imuposer_wrist_full_body_loader_FLU import make_imuposer_wrist_full_body_23_flu_loaders as _make_pose_loaders
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
            if _loader_type == "bones_seed_wrist_full_body_flu":
                return files_cfg if files_cfg is not None else "all"
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f)
                        for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg)
                    else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, pose_cfg.get("train_files"))
        _frac = pose_cfg.get("train_fraction")
        if _frac is not None and _loader_type != "bones_seed_wrist_full_body_flu":
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
            sub_patches=sub_patches,
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
            _load_encoder_state_dict(encoder, enc_state)

        # Load SMPL body model for val MPJPE (fails silently if unavailable)
        _pose_body_model = None
        try:
            import core.articulate as _art
            from core.paths import Paths as _Paths
            _bm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _pose_body_model = _art.model.ParametricModel(_Paths.SMPL_FILE, device=_bm_device)
        except Exception:
            pass

        def objective_fn(batch, model, cfg):
            x, y_full = batch[0], batch[1]
            expected  = pose_decoder.out_dim
            if y_full.shape[-1] > expected:
                _B, _Pt, _ = y_full.shape
                _J_loader = y_full.shape[-1] // (sub_patches * 6)
                y_pose = (
                    y_full.reshape(_B, _Pt, sub_patches, _J_loader, 6)
                    [:, :, :, _J_loader - num_joints:, :]
                    .reshape(_B, _Pt, expected)
                )
            else:
                y_pose = y_full
            _jerk_w    = float(pose_cfg.get("jerk_weight", pose_cfg.get("jitter_weight", 0.0)))
            _jp_weight = float(pose_cfg.get("joint_pos_weight", 0.0))
            _gt_pos = batch[2] if isinstance(batch, (tuple, list)) and len(batch) >= 3 else None
            return pose_1d_obj.forward_loss(
                (x, y_pose), encoder, pose_decoder, cfg, jerk_weight=_jerk_w,
                joint_pos_weight=_jp_weight,
                body_model=_pose_body_model, y_full_rot=y_full.float(), gt_pos=_gt_pos,
            )
        all_params = list(encoder.parameters()) + list(pose_decoder.parameters())
        if _loader_type == "bones_seed_wrist_full_body_flu":
            loaders = _make_pose_loaders(
                cfg=cfg,
                bvh_root=pose_cfg.get("bvh_root") or pose_cfg.get("train_bvh_root") or train_pkl_root,
                train_files=pose_cfg.get("train_files", "all"),
                val_files=pose_cfg.get("val_files"),
                bones_seed_root=pose_cfg.get("bones_seed_root", "/projects/b53017/vbc1591/bones-seed"),
                metadata_csv=pose_cfg.get("metadata_csv"),
                target_hz=int(pose_cfg.get("target_hz", 50)),
                add_gravity=bool(pose_cfg.get("add_gravity", True)),
                wrist_joint=pose_cfg.get("wrist_joint"),
                body_joints=pose_cfg.get("body_joints"),
                bvh_scale=float(pose_cfg.get("bvh_scale", 0.01)),
                tar_fallback=bool(pose_cfg.get("tar_fallback", True)),
                max_train_sessions=pose_cfg.get("max_train_sessions"),
                max_val_sessions=pose_cfg.get("max_val_sessions"),
            )
        elif _loader_type == "totalcapture_real_wrist_flu":
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
        elif _loader_type in ("imuposer_flu", "imuposer_flu_16hz_to_50hz", "imuposer_full_body_flu", "imuposer_full_body_23_flu"):
            loaders = _make_pose_loaders(
                cfg=cfg,
                train_files=train_files,
                val_files=val_files,
                add_gravity=bool(pose_cfg.get("add_gravity", False)),
            )
        elif _loader_type in ("nymeria_real_wrist_flu", "nymeria_real_wrist_full_body_flu",
                               "nymeria_real_wrist_full_body_global",
                               "nymeria_syn_wrist_full_body_global",
                               "nymeria_syn_wrist_full_body_flu"):
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
        dec_cfg_p = (vit_cfg_.get("pose_decoder") or {})
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size
        num_joints              = int(probe_cfg.get("num_joints", 5))
        sub_patches             = int(dec_cfg_p.get("sub_patches", 1))

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
        if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _log.info("[pose_probe] using alternating frame/global attention encoder (VGGT-style)")
        _log.info("[pose_probe] loading pretrained encoder from %s", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        _load_encoder_state_dict(encoder, enc_state)
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
        elif _probe_loader_type == "nymeria_syn_wrist_full_body_flu":
            from imu_lm.data.nymeria_syn_wrist_full_body_loader_FLU import make_nymeria_syn_wrist_full_body_loaders_flu as _make_loaders
        elif _probe_loader_type == "bones_seed_wrist_full_body_flu":
            from imu_lm.data.bones_seed_wrist_full_body_loader_FLU import make_bones_seed_wrist_full_body_loaders_flu as _make_loaders
        elif _probe_loader_type == "imuposer_flu":
            from imu_lm.data.imuposer_wrist_loader_FLU import make_imuposer_wrist_flu_loaders as _make_loaders
        elif _probe_loader_type == "imuposer_flu_16hz_to_50hz":
            from imu_lm.data.imuposer_wrist_16hz_to_50hz_loader_FLU import make_imuposer_wrist_16hz_to_50hz_flu_loaders as _make_loaders
        elif _probe_loader_type == "imuposer_full_body_flu":
            from imu_lm.data.imuposer_wrist_full_body_loader_FLU import make_imuposer_wrist_full_body_flu_loaders as _make_loaders
        elif _probe_loader_type == "imuposer_full_body_23_flu":
            from imu_lm.data.imuposer_wrist_full_body_loader_FLU import make_imuposer_wrist_full_body_23_flu_loaders as _make_loaders
        elif num_joints == 24:
            from imu_lm.data.nymeria_wrist_full_body_loader import make_nymeria_loaders as _make_loaders
        else:
            from imu_lm.data.nymeria_wrist_loader import make_nymeria_loaders as _make_loaders

        train_pkl_root = probe_cfg.get("train_pkl_root") or probe_cfg.get("pkl_root")
        val_pkl_root   = probe_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if _probe_loader_type == "bones_seed_wrist_full_body_flu":
                return files_cfg if files_cfg is not None else "all"
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, probe_cfg.get("train_files"))
        _frac = probe_cfg.get("train_fraction")
        if _frac is not None and _probe_loader_type != "bones_seed_wrist_full_body_flu":
            train_files = train_files[:max(1, int(len(train_files) * float(_frac)))]
        val_files   = _expand_files(val_pkl_root,   probe_cfg.get("val_files"))

        pose_decoder = ViT1DPoseDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            num_joints=num_joints,
            cfg=cfg,
            sub_patches=sub_patches,
        )

        _probe_body_model = None
        try:
            import core.articulate as _art
            from core.paths import Paths as _Paths
            _bm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _probe_body_model = _art.model.ParametricModel(_Paths.SMPL_FILE, device=_bm_device)
        except Exception:
            pass

        def objective_fn(batch, model, cfg):
            encoder.eval()
            with torch.no_grad():
                tokens, _ = encoder.forward_tokens(batch[0].float())
            y_full   = batch[1].float()
            expected = pose_decoder.out_dim
            if y_full.shape[-1] > expected:
                _B, _Pt, _ = y_full.shape
                _J_loader = y_full.shape[-1] // (sub_patches * 6)
                y_pose = (
                    y_full.reshape(_B, _Pt, sub_patches, _J_loader, 6)
                    [:, :, :, _J_loader - num_joints:, :]
                    .reshape(_B, _Pt, expected)
                )
            else:
                y_pose = y_full
            _jerk_w    = float(probe_cfg.get("jerk_weight", probe_cfg.get("jitter_weight", 0.0)))
            _jp_weight = float(probe_cfg.get("joint_pos_weight", 0.0))
            _gt_pos = batch[2] if isinstance(batch, (tuple, list)) and len(batch) >= 3 else None
            return pose_1d_obj.forward_loss(
                (batch[0].float(), y_pose), encoder, pose_decoder, cfg, jerk_weight=_jerk_w,
                joint_pos_weight=_jp_weight,
                body_model=_probe_body_model, y_full_rot=y_full, gt_pos=_gt_pos,
            )

        # Only decoder parameters are trainable
        all_params = list(pose_decoder.parameters())
        if _probe_loader_type == "bones_seed_wrist_full_body_flu":
            loaders = _make_loaders(
                cfg=cfg,
                bvh_root=probe_cfg.get("bvh_root") or probe_cfg.get("train_bvh_root") or train_pkl_root,
                train_files=probe_cfg.get("train_files", "all"),
                val_files=probe_cfg.get("val_files"),
                bones_seed_root=probe_cfg.get("bones_seed_root", "/projects/b53017/vbc1591/bones-seed"),
                metadata_csv=probe_cfg.get("metadata_csv"),
                target_hz=int(probe_cfg.get("target_hz", 50)),
                add_gravity=bool(probe_cfg.get("add_gravity", True)),
                wrist_joint=probe_cfg.get("wrist_joint"),
                body_joints=probe_cfg.get("body_joints"),
                bvh_scale=float(probe_cfg.get("bvh_scale", 0.01)),
                tar_fallback=bool(probe_cfg.get("tar_fallback", True)),
                max_train_sessions=probe_cfg.get("max_train_sessions"),
                max_val_sessions=probe_cfg.get("max_val_sessions"),
            )
        elif _probe_loader_type == "totalcapture_real_wrist_flu":
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
        elif _probe_loader_type in ("imuposer_flu", "imuposer_flu_16hz_to_50hz", "imuposer_full_body_flu", "imuposer_full_body_23_flu"):
            loaders = _make_loaders(
                cfg=cfg,
                train_files=train_files,
                val_files=val_files,
                add_gravity=bool(probe_cfg.get("add_gravity", False)),
            )
        elif _probe_loader_type in ("nymeria_real_wrist_flu", "nymeria_real_wrist_full_body_flu",
                                     "nymeria_real_wrist_full_body_global",
                                     "nymeria_syn_wrist_full_body_global",
                                     "nymeria_syn_wrist_full_body_flu"):
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
        _load_encoder_state_dict(encoder, enc_state)
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
        _load_encoder_state_dict(encoder, enc_state)

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
        _load_encoder_state_dict(encoder, enc_state)
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
        _load_encoder_state_dict(encoder, enc_state)

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

        if bool(vit_cfg_.get("encoder", {}).get("alt_attention", False)):
            import logging as _logging
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _logging.getLogger(__name__).info(
                "[odom_nymeria_real_wrist_flu] using alternating frame/global "
                "attention encoder (VGGT-style)")

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

    elif objective_type == "odom_nymeria_real_wrist_worldframe_flu":
        # Odometry regression using REAL wrist IMU (FLU frame) -> per-patch
        # CANONICAL WORLD-FRAME velocity [B, Pt, 4, 3]. See
        # nymeria_real_wrist_odom_worldframe_loader_FLU.py: each window is
        # rotated once about the vertical axis so its own net displacement
        # points along +X (heading canonicalization derived from movement
        # direction, since pelvis body orientation isn't available -- only
        # position). Unlike odom_nymeria_real_wrist_flu (continuously
        # re-referenced to the wrist's own rotating orientation, needing gyro
        # integration to reconstruct), this frame is fixed per window, so
        # reconstruction at eval time is a plain velocity cumulative sum.
        # Label flattened to [B, Pt, 12]. Encoder + odom decoder both trained
        # from scratch (fully supervised).
        import glob as _glob
        odom_wf_cfg = cfg_get(cfg, ["objective", "odom_nymeria_real_wrist_worldframe_flu"], {}) or {}
        vit_cfg_    = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        if bool(vit_cfg_.get("encoder", {}).get("alt_attention", False)):
            import logging as _logging
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _logging.getLogger(__name__).info(
                "[odom_nymeria_real_wrist_worldframe_flu] using alternating frame/global "
                "attention encoder (VGGT-style)")

        from imu_lm.data.nymeria_real_wrist_odom_worldframe_loader_FLU import (
            make_nymeria_real_wrist_odom_worldframe_loaders_flu,
        )

        train_pkl_root = odom_wf_cfg.get("train_pkl_root")
        val_pkl_root   = odom_wf_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, odom_wf_cfg.get("train_files"))
        _frac = odom_wf_cfg.get("train_fraction")
        if _frac is not None:
            train_files = train_files[:max(1, int(len(train_files) * float(_frac)))]
        val_files   = _expand_files(val_pkl_root,   odom_wf_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
            out_dim=12,   # 4 sub-patches × 3D canonical-world-frame velocity
        )

        def objective_fn(batch, model, cfg):
            x, y = batch[0], batch[1]   # batch[2] (R_canon) unused for training loss
            y_flat = y.reshape(y.shape[0], y.shape[1], -1)
            return odom_1d_obj.forward_loss((x, y_flat), encoder, odom_decoder, cfg)

        all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
        loaders = make_nymeria_real_wrist_odom_worldframe_loaders_flu(
            cfg=cfg,
            pkl_root="/",
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(odom_wf_cfg.get("add_gravity", True)),
        )

    elif objective_type == "odom_finetune_nymeria_real_wrist_worldframe_flu":
        # Warm-started encoder (stays trainable) + trainable ViT1DOdomDecoder on
        # Nymeria pelvis motion, CANONICAL WORLD-FRAME target. Sibling to
        # odom_finetune_nymeria_real_wrist_flu (same warm-start pattern), but uses
        # the world-frame loader/label convention of
        # odom_nymeria_real_wrist_worldframe_flu (heading canonicalized per-window
        # from movement direction; no gyro integration needed at eval time).
        import glob as _glob
        odom_wf_ft_cfg = cfg_get(cfg, ["objective", "odom_finetune_nymeria_real_wrist_worldframe_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        if bool(vit_cfg_.get("encoder", {}).get("alt_attention", False)):
            import logging as _logging
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _logging.getLogger(__name__).info(
                "[odom_finetune_nymeria_real_wrist_worldframe_flu] using alternating "
                "frame/global attention encoder (VGGT-style)")

        # ---- warm-start encoder (stays trainable) ----------------------------
        pretrained_run  = odom_wf_ft_cfg.get("pretrained_run")
        pretrained_ckpt = odom_wf_ft_cfg.get("pretrained_ckpt", "best")
        if not pretrained_run:
            raise ValueError(
                "[odom_finetune_nymeria_real_wrist_worldframe_flu] 'pretrained_run' must be set"
            )
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt
        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[odom_finetune_nymeria_real_wrist_worldframe_flu] pretrained encoder "
                f"checkpoint not found: {enc_ckpt_path}"
            )
        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info("[odom_finetune_nymeria_real_wrist_worldframe_flu] warm-starting encoder "
                   "from %s (encoder stays trainable)", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        _load_encoder_state_dict(encoder, enc_state)
        # IMU-Bench: `freeze_encoder: true` turns this branch into the frozen
        # world-frame odometry PROBE (same loader / decoder / label convention
        # as the finetune, only the encoder is frozen) so probe and finetune
        # rows share one code path and one window length.
        _wf_freeze = bool(odom_wf_ft_cfg.get("freeze_encoder", False))
        if _wf_freeze:
            _log.info("[odom_finetune_nymeria_real_wrist_worldframe_flu] freeze_encoder=true -> "
                      "encoder frozen (probe mode)")
            encoder.requires_grad_(False)
            encoder.eval()

        from imu_lm.data.nymeria_real_wrist_odom_worldframe_loader_FLU import (
            make_nymeria_real_wrist_odom_worldframe_loaders_flu,
        )

        train_pkl_root = odom_wf_ft_cfg.get("train_pkl_root")
        val_pkl_root   = odom_wf_ft_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, odom_wf_ft_cfg.get("train_files"))
        _frac = odom_wf_ft_cfg.get("train_fraction")
        if _frac is not None:
            train_files = train_files[:max(1, int(len(train_files) * float(_frac)))]
        val_files   = _expand_files(val_pkl_root,   odom_wf_ft_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
            out_dim=12,   # 4 sub-patches × 3D canonical-world-frame velocity
        )

        if _wf_freeze:
            def objective_fn(batch, model, cfg):
                encoder.eval()
                x, y = batch[0], batch[1]
                y_flat = y.reshape(y.shape[0], y.shape[1], -1).float()
                with torch.no_grad():
                    tokens, _ = encoder.forward_tokens(x.float())
                pred = odom_decoder(tokens)
                loss = torch.nn.functional.mse_loss(pred, y_flat)
                with torch.no_grad():
                    mae = (pred.detach() - y_flat).abs().mean()
                return loss, {"loss": float(loss.detach()),
                              "odom_mse": float(loss.detach()),
                              "odom_mae": float(mae)}
            all_params = list(odom_decoder.parameters())
        else:
            def objective_fn(batch, model, cfg):
                x, y = batch[0], batch[1]   # batch[2] (R_canon) unused for training loss
                y_flat = y.reshape(y.shape[0], y.shape[1], -1)
                return odom_1d_obj.forward_loss((x, y_flat), encoder, odom_decoder, cfg)
            all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
        loaders = make_nymeria_real_wrist_odom_worldframe_loaders_flu(
            cfg=cfg,
            pkl_root="/",
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(odom_wf_ft_cfg.get("add_gravity", True)),
        )

    elif objective_type == "odom_probe_nymeria_real_wrist_flu":
        # Frozen pretrained encoder + trainable ViT1DOdomDecoder on Nymeria real wrist FLU data.
        # Encoder loaded from a pretrained run (e.g. full-body pose supervised); all encoder params frozen.
        import glob as _glob
        odom_probe_rw_cfg = cfg_get(cfg, ["objective", "odom_probe_nymeria_real_wrist_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        if bool(vit_cfg_.get("encoder", {}).get("alt_attention", False)):
            import logging as _logging
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _logging.getLogger(__name__).info(
                "[odom_probe_nymeria_real_wrist_flu] using alternating frame/global "
                "attention encoder (VGGT-style)")

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
        _load_encoder_state_dict(encoder, enc_state)
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

        if bool(vit_cfg_.get("encoder", {}).get("alt_attention", False)):
            import logging as _logging
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _logging.getLogger(__name__).info(
                "[odom_finetune_nymeria_real_wrist_flu] using alternating frame/global "
                "attention encoder (VGGT-style)")

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
        _load_encoder_state_dict(encoder, enc_state)
        # IMU-Bench (2026-09-18, local-frame odometry replaces the world-frame task):
        # `freeze_encoder: true` turns this branch into the frozen local-frame PROBE,
        # mirroring odom_finetune_nymeria_real_wrist_worldframe_flu, so probe and
        # finetune rows share one loader / decoder / label convention.
        _lf_freeze = bool(odom_ft_rw_cfg.get("freeze_encoder", False))
        if _lf_freeze:
            _log.info("[odom_finetune_nymeria_real_wrist_flu] freeze_encoder=true -> "
                      "encoder frozen (probe mode)")
            encoder.requires_grad_(False)
            encoder.eval()

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

        if _lf_freeze:
            def objective_fn(batch, model, cfg):
                encoder.eval()
                x, y = batch[0], batch[1]
                y_flat = y.reshape(y.shape[0], y.shape[1], -1).float()
                with torch.no_grad():
                    tokens, _ = encoder.forward_tokens(x.float())
                pred = odom_decoder(tokens)
                loss = torch.nn.functional.mse_loss(pred, y_flat)
                with torch.no_grad():
                    mae = (pred.detach() - y_flat).abs().mean()
                return loss, {"loss": float(loss.detach()),
                              "odom_mse": float(loss.detach()),
                              "odom_mae": float(mae)}
            all_params = list(odom_decoder.parameters())
        else:
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

    elif objective_type == "odom_nymeria_syn_wrist_flu":
        # Odometry on Nymeria SYNTHETIC wrist IMU (FLU frame) → per-patch FLU velocity.
        # Encoder + odom decoder both trained from scratch (fully supervised).
        from imu_lm.data.nymeria_syn_wrist_odom_loader_FLU import make_nymeria_syn_wrist_odom_loaders_flu
        import glob as _glob
        odom_syn_cfg = cfg_get(cfg, ["objective", "odom_nymeria_syn_wrist_flu"], {}) or {}
        vit_cfg_     = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        train_pkl_root = odom_syn_cfg.get("train_pkl_root")
        val_pkl_root   = odom_syn_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, odom_syn_cfg.get("train_files"))
        _frac = odom_syn_cfg.get("train_fraction")
        if _frac is not None:
            train_files = train_files[:max(1, int(len(train_files) * float(_frac)))]
        val_files = _expand_files(val_pkl_root, odom_syn_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
            out_dim=12,
        )

        def objective_fn(batch, model, cfg):
            x, y = batch
            y_flat = y.reshape(y.shape[0], y.shape[1], -1)
            return odom_1d_obj.forward_loss((x, y_flat), encoder, odom_decoder, cfg)

        all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
        loaders = make_nymeria_syn_wrist_odom_loaders_flu(
            cfg=cfg,
            pkl_root="/",
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(odom_syn_cfg.get("add_gravity", True)),
        )

    elif objective_type == "odom_probe_nymeria_syn_wrist_flu":
        # Frozen pretrained encoder + trainable ViT1DOdomDecoder on Nymeria synthetic wrist FLU.
        from imu_lm.data.nymeria_syn_wrist_odom_loader_FLU import make_nymeria_syn_wrist_odom_loaders_flu
        import glob as _glob
        odom_probe_syn_cfg = cfg_get(cfg, ["objective", "odom_probe_nymeria_syn_wrist_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        pretrained_run  = odom_probe_syn_cfg.get("pretrained_run")
        pretrained_ckpt = odom_probe_syn_cfg.get("pretrained_ckpt", "best")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        enc_ckpt_path = (
            os.path.join(runs_root, pretrained_run, "checkpoints", f"{pretrained_ckpt}.pt")
            if pretrained_ckpt in ("best", "latest") else pretrained_ckpt
        )
        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[odom_probe_nymeria_syn_wrist_flu] checkpoint not found: {enc_ckpt_path}"
            )
        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info("[odom_probe_nymeria_syn_wrist_flu] loading encoder from %s", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        _load_encoder_state_dict(encoder, enc_state)
        _log.info("[odom_probe_nymeria_syn_wrist_flu] freezing encoder")
        encoder.requires_grad_(False)
        encoder.eval()

        train_pkl_root = odom_probe_syn_cfg.get("train_pkl_root")
        val_pkl_root   = odom_probe_syn_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, odom_probe_syn_cfg.get("train_files"))
        _frac = odom_probe_syn_cfg.get("train_fraction")
        if _frac is not None:
            train_files = train_files[:max(1, int(len(train_files) * float(_frac)))]
        val_files = _expand_files(val_pkl_root, odom_probe_syn_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
            out_dim=12,
        )

        def objective_fn(batch, model, cfg):
            encoder.eval()
            x_b, y_b = batch
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
        loaders = make_nymeria_syn_wrist_odom_loaders_flu(
            cfg=cfg,
            pkl_root="/",
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(odom_probe_syn_cfg.get("add_gravity", True)),
        )

    elif objective_type == "odom_finetune_nymeria_syn_wrist_flu":
        # Warm-started encoder (trainable) + trainable ViT1DOdomDecoder on Nymeria synthetic wrist FLU.
        from imu_lm.data.nymeria_syn_wrist_odom_loader_FLU import make_nymeria_syn_wrist_odom_loaders_flu
        import glob as _glob
        odom_ft_syn_cfg = cfg_get(cfg, ["objective", "odom_finetune_nymeria_syn_wrist_flu"], {}) or {}
        vit_cfg_  = cfg_get(cfg, ["vit1d"], {}) or {}
        patch_size              = int(vit_cfg_.get("patch_size", 16))
        num_patches_per_channel = T // patch_size

        pretrained_run  = odom_ft_syn_cfg.get("pretrained_run")
        pretrained_ckpt = odom_ft_syn_cfg.get("pretrained_ckpt", "best")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        enc_ckpt_path = (
            os.path.join(runs_root, pretrained_run, "checkpoints", f"{pretrained_ckpt}.pt")
            if pretrained_ckpt in ("best", "latest") else pretrained_ckpt
        )
        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[odom_finetune_nymeria_syn_wrist_flu] checkpoint not found: {enc_ckpt_path}"
            )
        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info("[odom_finetune_nymeria_syn_wrist_flu] warm-starting encoder from %s", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        _load_encoder_state_dict(encoder, enc_state)

        train_pkl_root = odom_ft_syn_cfg.get("train_pkl_root")
        val_pkl_root   = odom_ft_syn_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files = _expand_files(train_pkl_root, odom_ft_syn_cfg.get("train_files"))
        _frac = odom_ft_syn_cfg.get("train_fraction")
        if _frac is not None:
            train_files = train_files[:max(1, int(len(train_files) * float(_frac)))]
        val_files = _expand_files(val_pkl_root, odom_ft_syn_cfg.get("val_files"))

        odom_decoder = ViT1DOdomDecoder(
            embed_dim=encoder.embed_dim,
            num_patches_per_channel=num_patches_per_channel,
            in_channels=encoder.in_channels,
            cfg=cfg,
            out_dim=12,
        )

        def objective_fn(batch, model, cfg):
            x, y = batch
            y_flat = y.reshape(y.shape[0], y.shape[1], -1)
            return odom_1d_obj.forward_loss((x, y_flat), encoder, odom_decoder, cfg)

        all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
        loaders = make_nymeria_syn_wrist_odom_loaders_flu(
            cfg=cfg,
            pkl_root="/",
            train_files=train_files,
            val_files=val_files,
            add_gravity=bool(odom_ft_syn_cfg.get("add_gravity", True)),
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
        _load_encoder_state_dict(encoder, enc_state)
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
        _load_encoder_state_dict(encoder, enc_state)

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
        _load_encoder_state_dict(encoder, enc_state)
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
        _load_encoder_state_dict(encoder, enc_state)

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

        if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
            import logging as _logging
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _logging.getLogger(__name__).info(
                "[mae_nymeria_real_wrist_flu] using alternating frame/global "
                "attention encoder (VGGT-style)")

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

    elif objective_type == "mae_nymeria_syn_wrist_flu":
        # MAE pretraining on Nymeria SYNTHETIC wrist IMU in FLU local frame.
        # Synthesizes acc+gyro from xsens_pose (joint 9, right wrist) via finite
        # differences, then rotates from world frame into FLU body frame using the
        # wrist joint orientation. Works with any pkl that has gt_data['xsens_pose'].
        from imu_lm.data.nymeria_syn_wrist_loader_FLU import make_nymeria_syn_wrist_loaders_flu
        import glob as _glob

        mae_syn_flu_cfg = cfg_get(cfg, ["objective", "mae_nymeria_syn_wrist_flu"], {}) or {}
        train_pkl_root  = mae_syn_flu_cfg.get("train_pkl_root")
        val_pkl_root    = mae_syn_flu_cfg.get("val_pkl_root") or train_pkl_root

        def _expand_files(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files  = _expand_files(train_pkl_root, mae_syn_flu_cfg.get("train_files"))
        val_files    = _expand_files(val_pkl_root,   mae_syn_flu_cfg.get("val_files"))
        _add_gravity = bool(mae_syn_flu_cfg.get("add_gravity", True))

        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )
        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders = make_nymeria_syn_wrist_loaders_flu(
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
            _load_encoder_state_dict(encoder, enc_state)

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
        train_split_name = clip_cfg.get("train_split_name", "train")

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
        import logging as _logging
        _log = _logging.getLogger(__name__)
        if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _log.info("[imu_text_clip_flu] using alternating frame/global attention "
                      "encoder (VGGT-style)")
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
            _log.info("[imu_text_clip_flu] warm-starting encoder from %s", enc_ckpt_path)
            enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
            _load_encoder_state_dict(encoder, enc_state)

        if freeze_imu:
            encoder.requires_grad_(False)
            encoder.eval()

        # -- CLIP projection head (trained) -----------------------------------
        # init_temperature / freeze_logit_scale (2026-08-15, retrieval-tuning
        # sweep): optional override of the learnable temperature. Default
        # (None / False) keeps CLIP's tau=0.07 init and learnable scale, i.e.
        # every pre-existing config is unchanged.
        _init_tau = clip_cfg.get("init_temperature", None)
        _head_kwargs = {}
        if _init_tau is not None:
            import math as _math
            _head_kwargs["init_logit_scale"] = float(_math.log(1.0 / float(_init_tau)))
        clip_head = ContrastiveCLIPHead(
            imu_embed_dim  = encoder.embed_dim,
            clip_embed_dim = clip_embed_dim,
            **_head_kwargs,
        )
        if bool(clip_cfg.get("freeze_logit_scale", False)):
            clip_head.logit_scale.requires_grad_(False)

        # -- CLIP text encoder (frozen by default) ----------------------------
        local_files = bool(clip_cfg.get("local_files_only", False))
        if text_emb_cache is not None and freeze_text:
            text_encoder = None   # cache fully replaces the online encoder
        else:
            text_encoder = CLIPTextEncoder(
                model_name       = clip_model,
                freeze           = freeze_text,
                local_files_only = local_files,
                # text_projection (2026-07-28): apply CLIP's learned text
                # projection (official CLIP / UniMTS convention) instead of
                # the legacy bare pooler_output -- see CLIPTextEncoder.
                # Default False so every pre-existing config/checkpoint
                # keeps its original (pooler_output-space) behavior.
                use_projection   = bool(clip_cfg.get("text_projection", False)),
            )
        # text_train_last_n_layers (2026-08-15): with freeze_text_encoder:
        # false, optionally keep only the LAST n transformer layers (+ final
        # LN + text_projection) of the CLIP text tower trainable and freeze
        # embeddings + the remaining layers. None/absent = whole tower
        # trainable (unchanged behaviour).
        _text_last_n = clip_cfg.get("text_train_last_n_layers", None)
        if (not freeze_text) and text_encoder is not None and _text_last_n is not None:
            _text_last_n = int(_text_last_n)
            _tm = text_encoder.model
            _n_layers = len(_tm.text_model.encoder.layers)
            for _p in _tm.parameters():
                _p.requires_grad_(False)
            for _lyr in _tm.text_model.encoder.layers[max(0, _n_layers - _text_last_n):]:
                for _p in _lyr.parameters():
                    _p.requires_grad_(True)
            for _p in _tm.text_model.final_layer_norm.parameters():
                _p.requires_grad_(True)
            if getattr(_tm, "text_projection", None) is not None:
                for _p in _tm.text_projection.parameters():
                    _p.requires_grad_(True)
            _log.info("[imu_text_clip_flu] text tower: training last %d/%d layers "
                      "(+final LN, text_projection); %d/%d params trainable",
                      _text_last_n, _n_layers,
                      sum(p.numel() for p in _tm.parameters() if p.requires_grad),
                      sum(p.numel() for p in _tm.parameters()))

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

        # Optimizer parameter groups (2026-08-15, retrieval-tuning sweep).
        # Optional keys under objective.imu_text_clip_flu -- ALL default to
        # the legacy single-group behaviour when absent:
        #   text_lr_mult   : LR multiplier for the CLIP text tower (default 1.0)
        #   head_lr_mult   : LR multiplier for clip_head (+text_mlp) (default 1.0)
        #   layer_decay    : layer-wise LR decay for the IMU encoder
        #                    (BEiT/MAE-style; block depth d in 1..D gets
        #                    lr * layer_decay**(D+1-d), embeddings get
        #                    lr * layer_decay**(D+1); default 1.0 = off)
        #   no_decay_norm_bias : exclude 1-D params (norms, biases, tokens,
        #                    logit_scale) from weight decay (default false)
        _text_lr_mult = float(clip_cfg.get("text_lr_mult", 1.0))
        _head_lr_mult = float(clip_cfg.get("head_lr_mult", 1.0))
        _layer_decay  = float(clip_cfg.get("layer_decay", 1.0))
        _no_decay_1d  = bool(clip_cfg.get("no_decay_norm_bias", False))
        _use_groups = (_text_lr_mult != 1.0 or _head_lr_mult != 1.0
                       or _layer_decay != 1.0 or _no_decay_1d)
        if not _use_groups:
            if freeze_imu:
                all_params = list(clip_head.parameters())
            else:
                all_params = list(encoder.parameters()) + list(clip_head.parameters())
            if not freeze_text and text_encoder is not None:
                all_params += list(text_encoder.model.parameters())
            if text_mlp is not None:
                all_params += list(text_mlp.parameters())
        else:
            import re as _re
            _base_lr = float(cfg_get(cfg, ["trainer", "optim", "lr"], cfg_get(cfg, ["trainer", "lr"], 1e-4)))
            _base_wd = float(cfg_get(cfg, ["trainer", "optim", "weight_decay"], cfg_get(cfg, ["trainer", "weight_decay"], 0.0)))
            _groups = {}
            def _add(name, p, lr_mult):
                if not p.requires_grad:
                    return
                wd = 0.0 if (_no_decay_1d and p.ndim <= 1) else _base_wd
                key = (round(lr_mult, 8), wd)
                g = _groups.setdefault(key, {"params": [], "lr": _base_lr * lr_mult,
                                             "weight_decay": wd, "lr_mult": lr_mult})
                g["params"].append(p)
            if not freeze_imu:
                _depths = [int(m.group(2)) for n, _ in encoder.named_parameters()
                           for m in [_re.match(r"(frame_blocks|global_blocks|blocks|layers)\.(\d+)\.", n)] if m]
                _D = (max(_depths) + 1) if _depths else 0
                for n, p in encoder.named_parameters():
                    m = _re.match(r"(frame_blocks|global_blocks|blocks|layers)\.(\d+)\.", n)
                    if _layer_decay == 1.0 or _D == 0:
                        mult = 1.0
                    elif m:
                        d = int(m.group(2)) + 1                     # 1..D
                        mult = _layer_decay ** (_D + 1 - d)
                    elif n.startswith(("norm.", "cls_norm.")):
                        mult = 1.0                                   # top of the stack
                    else:
                        mult = _layer_decay ** (_D + 1)              # embeddings / tokens
                    _add(n, p, mult)
            for n, p in clip_head.named_parameters():
                _add("clip_head." + n, p, _head_lr_mult)
            if not freeze_text and text_encoder is not None:
                for n, p in text_encoder.model.named_parameters():
                    _add("text." + n, p, _text_lr_mult)
            if text_mlp is not None:
                for n, p in text_mlp.named_parameters():
                    _add("text_mlp." + n, p, _head_lr_mult)
            all_params = list(_groups.values())
            _log.info("[imu_text_clip_flu] optimizer param groups: %s",
                      "; ".join(f"lr={g['lr']:.2e} wd={g['weight_decay']:.3g} n={sum(p.numel() for p in g['params'])}"
                                for g in all_params))

        loaders = make_humanml3d_imu_loaders_flu(
            root             = humanml3d_root,
            cfg              = cfg,
            text_emb_cache   = text_emb_cache,
            train_split_name = train_split_name,
        )

    elif objective_type == "imu_text_clip_nymeria_flu":
        # CLIP-style contrastive pretraining: real-wrist FLU IMU ↔ text on
        # Nymeria's own motion-narration captions (downloaded via
        # scripts/download_nymeria_captions.py). Same machinery as
        # imu_text_clip_flu (HumanML3D) — ContrastiveCLIPHead/CLIPTextEncoder/
        # TextProjectionMLP/info_nce_loss — swapping in
        # nymeria_caption_dataset_FLU's real-wrist-IMU-plus-caption-segment
        # loader instead of HumanML3D's synthetic-IMU-plus-whole-clip-caption
        # loader. Unlike HumanML3D, captions are real human narrations of a
        # real recorded IMU signal (no AMASS/SMPL synthesis involved).
        from imu_lm.objectives.contrastive_clip_1d import (
            ContrastiveCLIPHead, CLIPTextEncoder, TextProjectionMLP
        )
        import imu_lm.objectives.contrastive_clip_1d as clip_obj
        from imu_lm.data.nymeria_caption_dataset_FLU import make_nymeria_caption_imu_loaders_flu

        clip_cfg        = cfg_get(cfg, ["objective", "imu_text_clip_nymeria_flu"], {}) or {}
        runs_root       = cfg_get(cfg, ["paths", "runs_root"], "runs")
        pretrained_run  = clip_cfg.get("pretrained_run")
        pretrained_ckpt = clip_cfg.get("pretrained_ckpt", "best")
        captions_root   = clip_cfg.get("captions_root")
        train_pkl_root  = clip_cfg.get("train_pkl_root")
        val_pkl_root    = clip_cfg.get("val_pkl_root")
        clip_model      = clip_cfg.get("clip_model", "openai/clip-vit-base-patch32")
        clip_embed_dim  = int(clip_cfg.get("clip_embed_dim", 512))
        freeze_imu      = bool(clip_cfg.get("freeze_imu_encoder", False))
        freeze_text     = bool(clip_cfg.get("freeze_text_encoder", True))
        text_cache_path = clip_cfg.get("text_cache_path", None)
        use_text_mlp    = bool(clip_cfg.get("text_projection_mlp", False))
        text_mlp_hidden = clip_cfg.get("text_mlp_hidden_dim", None)
        caption_source  = clip_cfg.get("caption_source", "motion_narration")
        imu_source      = clip_cfg.get("imu_source", "real")

        if not captions_root or not train_pkl_root or not val_pkl_root:
            raise ValueError(
                "[imu_text_clip_nymeria_flu] 'captions_root', 'train_pkl_root', and "
                "'val_pkl_root' must all be set in objective.imu_text_clip_nymeria_flu"
            )

        # -- load pre-cached text embeddings (optional) ----------------------
        text_emb_cache = None
        if text_cache_path:
            import logging as _logging2n
            _log2n = _logging2n.getLogger(__name__)
            if not os.path.exists(text_cache_path):
                raise FileNotFoundError(
                    f"[imu_text_clip_nymeria_flu] text_cache_path not found: {text_cache_path}\n"
                    "Run:  python scripts/cache_clip_text_embeddings_nymeria.py "
                    f"--captions-root {captions_root} --out {text_cache_path}"
                )
            _log2n.info("[imu_text_clip_nymeria_flu] loading pre-cached text embeddings from %s", text_cache_path)
            text_emb_cache = torch.load(text_cache_path, map_location="cpu", weights_only=False)
            _log2n.info("[imu_text_clip_nymeria_flu] cache loaded: %d captions", len(text_emb_cache))

        # -- warm-start IMU encoder ------------------------------------------
        import logging as _logging_n
        _log_n = _logging_n.getLogger(__name__)
        if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _log_n.info("[imu_text_clip_nymeria_flu] using alternating frame/global attention "
                        "encoder (VGGT-style)")
        if pretrained_run:
            if pretrained_ckpt in ("best", "latest"):
                enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                             "checkpoints", f"{pretrained_ckpt}.pt")
            else:
                enc_ckpt_path = pretrained_ckpt
            if not os.path.exists(enc_ckpt_path):
                raise FileNotFoundError(
                    f"[imu_text_clip_nymeria_flu] pretrained encoder checkpoint not found: {enc_ckpt_path}"
                )
            _log_n.info("[imu_text_clip_nymeria_flu] warm-starting encoder from %s", enc_ckpt_path)
            enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
            _load_encoder_state_dict(encoder, enc_state)

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
            text_encoder = None
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

        loaders = make_nymeria_caption_imu_loaders_flu(
            captions_root   = captions_root,
            train_pkl_root  = train_pkl_root,
            val_pkl_root    = val_pkl_root,
            cfg             = cfg,
            text_emb_cache  = text_emb_cache,
            caption_source  = caption_source,
            imu_source      = imu_source,
        )

    elif objective_type == "imu_text_clip_realdata":
        # CLIP-style contrastive pretraining: IMU ↔ text on real labeled parquet data.
        # Text labels are generated as sentence_template.format(dataset_activity_label),
        # e.g. "The person is walking".  Trained from scratch (no warm-start).
        #
        # CLIP text embeddings for all unique activity sentences are pre-computed once
        # at startup using the frozen CLIP text encoder, then cached in a dict for
        # zero-overhead lookup during training (no CLIP model stays in GPU memory).
        from imu_lm.objectives.contrastive_clip_1d import (
            ContrastiveCLIPHead, CLIPTextEncoder, TextProjectionMLP,
        )
        import imu_lm.objectives.contrastive_clip_1d as clip_obj
        from imu_lm.data.realdata_clip_loader import make_realdata_clip_loaders
        import logging as _logging_rdc
        _log_rdc = _logging_rdc.getLogger(__name__)

        rdc_cfg          = cfg_get(cfg, ["objective", "imu_text_clip_realdata"], {}) or {}
        clip_model       = rdc_cfg.get("clip_model",        "openai/clip-vit-base-patch32")
        clip_embed_dim   = int(rdc_cfg.get("clip_embed_dim",   512))
        freeze_imu       = bool(rdc_cfg.get("freeze_imu_encoder", False))
        local_files      = bool(rdc_cfg.get("local_files_only",   False))
        sentence_tmpl    = rdc_cfg.get("sentence_template", "The person is {}")
        use_text_mlp     = bool(rdc_cfg.get("text_projection_mlp", False))
        text_mlp_hidden  = rdc_cfg.get("text_mlp_hidden_dim", None)
        pretrained_run   = rdc_cfg.get("pretrained_run", None)
        pretrained_ckpt  = rdc_cfg.get("pretrained_ckpt", "best")

        # Bug fix (2026-07-24): this objective never checked alt_attention when
        # constructing its own encoder, silently keeping the plain ViT1DEncoder
        # assigned at the top of main() -- any warm-start from an
        # alt_attention:true checkpoint (every checkpoint that matters in this
        # benchmark) landed 100% of its weights in missing/unexpected instead
        # of raising, so the "warm-started" encoder was actually
        # near-randomly-initialized. Mirrors every other objective's own
        # alt_attention check (e.g. utd_mhad_finetune, imu_text_clip_nymeria_flu).
        if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _log_rdc.info("[imu_text_clip_realdata] using alternating frame/global "
                          "attention encoder (VGGT-style)")

        # Optional: warm-start IMU encoder from a prior MAE/CLIP run
        if pretrained_run:
            runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
            if pretrained_ckpt in ("best", "latest"):
                _enc_ckpt = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
            else:
                _enc_ckpt = pretrained_ckpt
            if not os.path.exists(_enc_ckpt):
                raise FileNotFoundError(
                    f"[imu_text_clip_realdata] pretrained encoder not found: {_enc_ckpt}"
                )
            _log_rdc.info("[imu_text_clip_realdata] warm-starting encoder from %s", _enc_ckpt)
            _enc_state = torch.load(_enc_ckpt, map_location="cpu", weights_only=False)
            # Routed through the shared helper (2026-07-28): this was the last
            # warm-start path still doing a bare strict=False load with only a
            # warning, and without _remap_encoder_state_dict. A key-name
            # mismatch (e.g. an older checkpoint using self_attn.* names) would
            # therefore leave the encoder essentially randomly initialized while
            # the run logged "warm-starting encoder from ..." and continued --
            # exactly the silent failure the helper was written to make loud.
            _load_encoder_state_dict(encoder, _enc_state)

        if freeze_imu:
            encoder.requires_grad_(False)
            encoder.eval()

        # CLIP projection head (always trained)
        clip_head = ContrastiveCLIPHead(
            imu_embed_dim  = encoder.embed_dim,
            clip_embed_dim = clip_embed_dim,
        )

        # Bug fix (2026-07-25): pretrained_run only ever warm-started the
        # encoder, never the paired clip_head it was jointly trained with in
        # Stage 2 -- clip_head was always freshly (randomly) initialized here,
        # then fit solely on the midtrain mix's narrow templated-caption
        # vocabulary. Root-caused via a UTD-MHAD zero-shot collapse (model
        # predicts the same class for ~100% of windows) that persisted even
        # with freeze_imu_encoder:true, which ruled out encoder drift as the
        # cause -- the fresh, narrowly-trained clip_head was the actual
        # culprit. Warm-start it too when the checkpoint has one saved.
        if pretrained_run and "clip_head" in _enc_state:
            clip_head.load_state_dict(_enc_state["clip_head"], strict=True)
            _log_rdc.info("[imu_text_clip_realdata] warm-starting clip_head from %s", _enc_ckpt)

        # Build data loaders (also constructs the id→text label map internally)
        loaders = make_realdata_clip_loaders(cfg, sentence_template=sentence_tmpl)

        freeze_text_enc = bool(rdc_cfg.get("freeze_text_encoder", True))

        train_ds     = loaders["train_loader"].dataset
        unique_texts = sorted(set(train_ds.id_to_text.values()))

        # Optional trainable MLP on top of frozen text embeddings
        text_mlp_rdc = None
        if use_text_mlp:
            text_mlp_rdc = TextProjectionMLP(
                embed_dim  = clip_embed_dim,
                hidden_dim = text_mlp_hidden,
            )
            _log_rdc.info("[imu_text_clip_realdata] TextProjectionMLP enabled (hidden=%s)", text_mlp_hidden)

        if freeze_text_enc:
            # Pre-compute text embeddings once; discard the CLIP model to save GPU memory.
            _log_rdc.info(
                "[imu_text_clip_realdata] pre-computing CLIP embeddings for %d unique activities",
                len(unique_texts),
            )
            _text_enc_tmp = CLIPTextEncoder(
                model_name       = clip_model,
                freeze           = True,
                local_files_only = local_files,
            )
            _emb_chunks: list = []
            with torch.no_grad():
                for _i in range(0, len(unique_texts), 64):
                    _batch_txt = unique_texts[_i : _i + 64]
                    _emb_chunks.append(
                        _text_enc_tmp(_batch_txt, device=torch.device("cpu")).cpu()
                    )
            # {sentence → L2-normalised CPU tensor [D_clip]}
            _activity_emb_map: dict = {
                txt: emb
                for txt, emb in zip(
                    unique_texts, torch.cat(_emb_chunks, dim=0)
                )
            }
            del _text_enc_tmp, _emb_chunks   # free ~250 MB
            text_enc = None

            def objective_fn(batch, model, cfg):
                x, texts = batch
                text_embs = torch.stack(
                    [_activity_emb_map[t] for t in texts], dim=0
                )   # [B, D_clip] on CPU — forward_loss moves it to x.device
                return clip_obj.forward_loss(
                    (x, text_embs), encoder, clip_head, None, cfg,
                    text_mlp=text_mlp_rdc,
                )

            if freeze_imu:
                all_params = list(clip_head.parameters())
            else:
                all_params = list(encoder.parameters()) + list(clip_head.parameters())
            if text_mlp_rdc is not None:
                all_params += list(text_mlp_rdc.parameters())

        else:
            # Trainable text encoder: encode on-the-fly each step, gradients flow through.
            _log_rdc.info(
                "[imu_text_clip_realdata] text encoder is TRAINABLE (%d unique activities)",
                len(unique_texts),
            )
            text_enc = CLIPTextEncoder(
                model_name       = clip_model,
                freeze           = False,
                local_files_only = local_files,
            )
            # Lazy GPU migration: move text_enc to training device on first forward.
            # .to(device) moves parameters in-place so the optimizer references stay valid.
            _text_enc_moved = [False]

            def objective_fn(batch, model, cfg):
                x, texts = batch
                if not _text_enc_moved[0]:
                    text_enc.to(x.device)
                    _text_enc_moved[0] = True
                text_embs = text_enc(texts, device=x.device)   # [B, D_clip]
                return clip_obj.forward_loss(
                    (x, text_embs), encoder, clip_head, None, cfg,
                    text_mlp=text_mlp_rdc,
                )

            if freeze_imu:
                all_params = list(clip_head.parameters()) + list(text_enc.parameters())
            else:
                all_params = (
                    list(encoder.parameters())
                    + list(clip_head.parameters())
                    + list(text_enc.parameters())
                )
            if text_mlp_rdc is not None:
                all_params += list(text_mlp_rdc.parameters())

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
            apply_flu_transform=True,
            cfg=cfg,
        )

    elif objective_type == "mae_realdata_ocdetect_flu":
        # MAE pretraining combining:
        #   (a) Real parquet datasets (pamap2/recofit/…) — FLU z-norm, no rotation aug
        #   (b) OCDetect wrist-IMU — FLU transform applied, z-norm
        # Balanced via WeightedRandomSampler (50/50 realdata vs OCDetect per batch).
        # OCDetect capped at max_ocdetect_hours hours to prevent domination.
        # Loss: frequency-band-weighted MAE (lower bands get higher weights).
        import logging as _logging
        import glob as _glob
        import numpy as np
        from torch.utils.data import (
            ConcatDataset, WeightedRandomSampler, DataLoader as _DL,
        )
        from imu_lm.data.OCDetect_dataloader import (
            _collect_windows, _list_files, select_ocd_files,
            DATA_DIR_DEFAULT as _OCDetect_DATA_DIR,
            OCDetectDataset,
        )
        from imu_lm.data.loaders import (
            WindowDataset, collate_skip_none as parquet_collate_skip_none,
            build_session_index, make_splits,
        )
        from imu_lm.objectives import masked_1d_freq as freq_mae_obj

        _log = _logging.getLogger(__name__)

        rdo_cfg      = cfg_get(cfg, ["objective", "mae_realdata_ocdetect_flu"], {}) or {}
        ocd_data_dir = rdo_cfg.get("ocdetect_data_dir", _OCDetect_DATA_DIR)
        ocd_splits   = tuple(rdo_cfg.get("ocdetect_splits", [0.8, 0.1, 0.1]))
        max_ocd_h    = float(rdo_cfg.get("max_ocdetect_hours", 500.0))
        max_ocd_files = rdo_cfg.get("max_ocdetect_files", None)
        num_workers_ = int(cfg_get(cfg, ["data", "num_workers"], 4))
        pin_memory_  = bool(cfg_get(cfg, ["data", "pin_memory"], True))
        batch_size_  = int(cfg_get(cfg, ["data", "batch_size"], 256))
        eval_bs_     = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size_))

        # ── Variable-length pretraining config ───────────────────────────────
        # When enabled, windows are cut at the MAX length and each batch is
        # randomly cropped to a length L ∈ [min_samples, max_samples] (multiple
        # of patch_size). Token count then ranges 24–192 (6 ch × 4–32 patches).
        _vl_cfg   = cfg_get(cfg, ["data", "variable_length"], {}) or {}
        _var_len  = bool(_vl_cfg.get("enabled", False))
        _vl_min   = int(_vl_cfg.get("min_samples", 64))
        _vl_max   = int(_vl_cfg.get("max_samples", 512))
        _patch_sz = int((cfg_get(cfg, ["vit1d"], {}) or {}).get("patch_size", 16))

        win_sec      = float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56))
        _ocd_win_samples = _vl_max if _var_len else 128

        # ── OCDetect windows ─────────────────────────────────────────────────
        # Pre-cap files (deterministic) BEFORE windowing to bound RAM, then
        # window-level cap to exactly max_ocdetect_hours.
        _ocd_files = select_ocd_files(ocd_data_dir,
                                      max_files=int(max_ocd_files) if max_ocd_files else None)
        X_ocd, y_ocd = _collect_windows(
            _ocd_files, apply_flu_transform=True,
            window_samples=_ocd_win_samples, hop_samples=_ocd_win_samples // 2,
        )
        # Limit to max_ocdetect_hours
        max_ocd_wins = int(max_ocd_h * 3600.0 / win_sec)
        if len(X_ocd) > max_ocd_wins:
            _rng_ocd = np.random.default_rng(42)
            _idx     = _rng_ocd.choice(len(X_ocd), size=max_ocd_wins, replace=False)
            X_ocd, y_ocd = X_ocd[_idx], y_ocd[_idx]
        elif max_ocd_files:
            _log.warning("[mae_realdata_ocdetect_flu] only %d windows collected "
                         "from %d files (< %d-window cap) — raise max_ocdetect_files",
                         len(X_ocd), len(_ocd_files), max_ocd_wins)
        _log.info("[mae_realdata_ocdetect_flu] OCDetect: %d windows (%.1f h cap, %d files)",
                  len(X_ocd), max_ocd_h, len(_ocd_files))

        # Split OCDetect into train / val
        _n_ocd       = len(X_ocd)
        _n_ocd_train = int(ocd_splits[0] * _n_ocd)
        _n_ocd_val   = int(ocd_splits[1] * _n_ocd)
        X_ocd_train, y_ocd_train = X_ocd[:_n_ocd_train],               y_ocd[:_n_ocd_train]
        X_ocd_val,   y_ocd_val   = X_ocd[_n_ocd_train:_n_ocd_train+_n_ocd_val], \
                                    y_ocd[_n_ocd_train:_n_ocd_train+_n_ocd_val]

        ocd_train_ds = OCDetectDataset(X_ocd_train, y_ocd_train, cfg=cfg)
        ocd_val_ds   = OCDetectDataset(X_ocd_val,   y_ocd_val,   cfg=cfg)

        # ── Realdata (parquet) windows ───────────────────────────────────────
        parquet_path   = cfg_get(cfg, ["paths", "dataset_path"])
        train_on       = cfg_get(cfg, ["data", "train_on_datasets"], None)
        parquet_si     = build_session_index(parquet_path, cfg,
                                             dataset_filter=list(train_on) if train_on else None)
        parquet_splits_ = make_splits(parquet_si, cfg)

        class _XOnlyDataset(torch.utils.data.Dataset):
            """Strip label, return (x [C,T], dummy) so ConcatDataset stacks cleanly."""
            def __init__(self, ds): self._ds = ds
            def __len__(self):      return len(self._ds)
            def __getitem__(self, i):
                item = self._ds[i]
                if item is None:
                    return None
                return item[0], torch.empty(0)

        parq_train_ds = _XOnlyDataset(WindowDataset(
            parquet_path, parquet_si, parquet_splits_["train_keys"], cfg, split_name="train"
        ))
        parq_val_ds = _XOnlyDataset(WindowDataset(
            parquet_path, parquet_si, parquet_splits_["val_keys"], cfg, split_name="val"
        ))

        class _OCDXOnly(torch.utils.data.Dataset):
            """OCDetect wrapper that returns (x, dummy) matching parquet format."""
            def __init__(self, ds): self._ds = ds
            def __len__(self):      return len(self._ds)
            def __getitem__(self, i):
                item = self._ds[i]
                if item is None:
                    return None
                x = item[0] if isinstance(item, (tuple, list)) else item
                return x, torch.empty(0)

        ocd_train_xonly = _OCDXOnly(ocd_train_ds)
        ocd_val_xonly   = _OCDXOnly(ocd_val_ds)

        combined_train = ConcatDataset([parq_train_ds, ocd_train_xonly])
        combined_val   = ConcatDataset([parq_val_ds,   ocd_val_xonly])

        _log.info(
            "[mae_realdata_ocdetect_flu] train=%d (parq=%d ocd=%d) val=%d (parq=%d ocd=%d)",
            len(combined_train), len(parq_train_ds), len(ocd_train_xonly),
            len(combined_val),   len(parq_val_ds),   len(ocd_val_xonly),
        )

        # ── Sampler ───────────────────────────────────────────────────────────
        # balanced_sampling: true  → 50/50 realdata vs OCDetect per batch
        #                    false → natural ratio (e.g. ~38:1 with full OCDetect):
        #                            plain shuffle over the full combined dataset.
        _balanced = bool(rdo_cfg.get("balanced_sampling", True))
        _n_parq = len(parq_train_ds)
        _n_ocd2 = len(ocd_train_xonly)
        if _balanced:
            _w_parq = 1.0 / max(_n_parq, 1)
            _w_ocd  = 1.0 / max(_n_ocd2, 1)
            _sample_weights = [_w_parq] * _n_parq + [_w_ocd] * _n_ocd2
            _num_samples = 2 * min(_n_parq, _n_ocd2)   # one epoch = 2×smaller source
            balanced_sampler = WeightedRandomSampler(
                weights=_sample_weights, num_samples=_num_samples, replacement=True)
            _train_sampler, _train_shuffle = balanced_sampler, False
            _log.info("[mae_realdata_ocdetect_flu] sampler=balanced 50/50 "
                      "(epoch=%d windows)", _num_samples)
        else:
            # Natural ratio: each window seen once per epoch, OCDetect dominates.
            _train_sampler, _train_shuffle = None, True
            _log.info("[mae_realdata_ocdetect_flu] sampler=natural ratio "
                      "(parq=%d : ocd=%d ≈ 1:%.1f, epoch=%d windows)",
                      _n_parq, _n_ocd2, _n_ocd2 / max(_n_parq, 1), _n_parq + _n_ocd2)

        def _collate_x(batch):
            batch = [b for b in batch if b is not None]
            if not batch:
                return None
            xs = torch.stack([b[0] for b in batch], 0)
            return xs, torch.empty(0)

        if _var_len:
            import random as _random
            _vl_lengths = list(range(_vl_min, _vl_max + 1, _patch_sz))   # multiples of patch_size

            def _collate_x_train(batch):
                batch = [b for b in batch if b is not None]
                if not batch:
                    return None
                xs = torch.stack([b[0] for b in batch], 0)        # [B, C, Tmax]
                L  = _random.choice(_vl_lengths)                  # same L for whole batch
                Tmax = xs.shape[-1]
                if L < Tmax:
                    off = _random.randint(0, Tmax - L)            # random temporal offset
                    xs = xs[:, :, off:off + L]
                return xs.contiguous(), torch.empty(0)

            def _collate_x_val(batch):
                # Fixed length at val so val_loss is comparable across steps.
                batch = [b for b in batch if b is not None]
                if not batch:
                    return None
                xs = torch.stack([b[0] for b in batch], 0)[:, :, :_vl_max]
                return xs.contiguous(), torch.empty(0)

            _train_collate, _val_collate = _collate_x_train, _collate_x_val
            _log.info("[mae_realdata_ocdetect_flu] variable-length: L ∈ [%d, %d] "
                      "samples (%d–%d tokens)",
                      _vl_min, _vl_max,
                      6 * (_vl_min // _patch_sz), 6 * (_vl_max // _patch_sz))
        else:
            _train_collate = _val_collate = _collate_x

        # ── Decoder and objective ─────────────────────────────────────────────
        # ssl_method selects which SSL objective runs on the SAME ViT1DEncoder,
        # for a controlled "which SSL objective is best?" comparison:
        #   mae (default) / limu_bert / simmtm / focal
        _ssl_method  = str(rdo_cfg.get("ssl_method", "mae")).lower()
        _use_3d_freq = bool(cfg_get(cfg, ["vit1d", "encoder", "use_3d_freq"], False))

        if _use_3d_freq:
            # v6: 3D frequency-token architecture (channel × time × freq tokens)
            from imu_lm.models.ViT1D.model import ViT1DEncoder3D, ViT1DDecoder3D
            from imu_lm.objectives import masked_3d_freq as freq3d_obj
            encoder = ViT1DEncoder3D(cfg)            # replace the 2D encoder
            decoder = ViT1DDecoder3D(
                embed_dim=encoder.embed_dim, out_channels=encoder.in_channels,
                target_T=T, num_freq=encoder.num_freq_bands, cfg=cfg,
            )
            def objective_fn(batch, model, cfg):
                return freq3d_obj.forward_loss(batch, encoder, decoder, cfg)
            all_params = list(encoder.parameters()) + list(decoder.parameters())

        elif _ssl_method in ("limu_bert", "simmtm", "focal"):
            from imu_lm.objectives import ssl_vit_baselines as _sslb
            _Pt = T // encoder.patch_size
            _N  = encoder.in_channels * _Pt
            obj_blk = cfg_get(cfg, ["objective"], {}) or {}
            # target_T=0 → decoder outputs num_patches*patch_size (length-agnostic)
            _dec_T = 0 if _var_len else T
            _log.info("[mae_realdata_ocdetect_flu] SSL objective on shared ViT1DEncoder: %s%s",
                      _ssl_method, " (variable length)" if _var_len else "")
            if _ssl_method == "limu_bert":
                decoder = ViT1DDecoder(embed_dim=encoder.embed_dim,
                                       out_channels=encoder.in_channels, target_T=_dec_T, cfg=cfg)
                def objective_fn(batch, model, cfg):
                    return _sslb.limubert_forward_loss(batch, encoder, decoder, cfg)
                all_params = list(encoder.parameters()) + list(decoder.parameters())
            elif _ssl_method == "simmtm":
                if _var_len:
                    # length-agnostic head (pool for contrastive, per-patch recon)
                    decoder = _sslb.SimMTMHeadVarLen(
                        embed_dim=encoder.embed_dim, patch_size=encoder.patch_size,
                        proj_dim=int(obj_blk.get("proj_dim", 128)),
                        temperature=float(obj_blk.get("temperature", 0.2)),
                        positive_nums=int(obj_blk.get("positive_nums", 3)))
                    def objective_fn(batch, model, cfg):
                        return _sslb.simmtm_forward_loss_varlen(batch, encoder, decoder, cfg)
                else:
                    decoder = _sslb.SimMTMHead(
                        embed_dim=encoder.embed_dim, num_tokens=_N,
                        recon_dim=encoder.in_channels * T,
                        proj_dim=int(obj_blk.get("proj_dim", 128)),
                        temperature=float(obj_blk.get("temperature", 0.2)),
                        positive_nums=int(obj_blk.get("positive_nums", 3)))
                    def objective_fn(batch, model, cfg):
                        return _sslb.simmtm_forward_loss(batch, encoder, decoder, cfg)
                all_params = list(encoder.parameters()) + list(decoder.parameters())
            else:  # focal — per-token projectors are already length-agnostic
                decoder = _sslb.FOCALHead(
                    embed_dim=encoder.embed_dim,
                    modalities=list(obj_blk.get("modalities", ["acc", "gyro"])),
                    emb_dim=int(obj_blk.get("emb_dim", 128)))
                def objective_fn(batch, model, cfg):
                    return _sslb.focal_forward_loss(batch, encoder, decoder, cfg)
                all_params = list(encoder.parameters()) + list(decoder.parameters())

        else:
            # MAE. Optionally swap in the alternating frame/global attention
            # encoder (VGGT-style; BERT masking, CLS token, frame+global attn).
            if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
                from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
                encoder = ViT1DEncoderAltAttn(cfg)
                _log.info("[mae_realdata_ocdetect_flu] using alternating frame/global "
                          "attention encoder (VGGT-style)%s",
                          " (variable length)" if _var_len else "")
            decoder = ViT1DDecoder(
                embed_dim=encoder.embed_dim, out_channels=encoder.in_channels,
                target_T=(0 if _var_len else T), cfg=cfg,
            )
            def objective_fn(batch, model, cfg):
                return freq_mae_obj.forward_loss(batch, encoder, decoder, cfg)
            all_params = list(encoder.parameters()) + list(decoder.parameters())

        loaders = {
            "train_loader": _DL(
                combined_train,
                batch_size=batch_size_,
                sampler=_train_sampler,
                shuffle=_train_shuffle,
                num_workers=num_workers_,
                pin_memory=pin_memory_,
                drop_last=True,
                persistent_workers=num_workers_ > 0,
                collate_fn=_train_collate,
            ),
            "val_loader": _DL(
                combined_val,
                batch_size=eval_bs_,
                shuffle=False,
                num_workers=num_workers_,
                pin_memory=pin_memory_,
                drop_last=False,
                persistent_workers=num_workers_ > 0,
                collate_fn=_val_collate,
            ),
        }

    elif objective_type == "mae_multidata_varlen_altattn_flu":
        # Pretraining combining every IMU source available in this deployment:
        #   realdata (parquet) + OCDetect + Nymeria real wrist + TotalCapture/DIP-IMU +
        #   IMUPoser + EatingSpeed + Motion-X++ (synth) + AMASS/HumanML3D (synth) +
        #   SONIC/bones-seed (synth) + Capture24 (acc-only).
        # See imu_lm/data/combined_mae_multidata_loader.py for per-source details,
        # on-disk paths, and the balanced/natural sampling + varlen-crop machinery
        # (mirrors the two-source mae_realdata_ocdetect_flu objective above, generalised
        # to N sources with per-source WeightedRandomSampler weighting).
        #
        # objective.mae_multidata_varlen_altattn_flu.ssl_method selects the
        # pretext task on the SAME combined dataset (mirrors
        # mae_realdata_ocdetect_flu's ssl_method switch):
        #   mae (default) — masked_1d_freq_channel_mask, optionally alt-attn encoder.
        #   simmtm / focal — baselines on the plain (non-alt-attn) ViT1DEncoder,
        #     same encoder capacity as the "mae" arm, for a fair, controlled
        #     "which SSL objective is best?" comparison (identical data + backbone
        #     size; only the pretext task differs). NOTE: these vendored baseline
        #     losses only read batch[0] (x) — they do not consume
        #     channel_valid_mask, so capture24's zero-padded "gyro" channels are
        #     treated as real signal for simmtm's reconstruction loss and focal's
        #     gyro-modality contrastive term. This is a minor, accepted skew
        #     (capture24 is a small, downweighted slice of the combined data),
        #     not fixed here to avoid rewriting the vendored loss math.
        import logging as _logging
        from imu_lm.data.combined_mae_multidata_loader import make_combined_mae_multidata_loaders

        _log = _logging.getLogger(__name__)
        _var_len = bool(cfg_get(cfg, ["data", "variable_length", "enabled"], False))
        _mm_cfg = cfg_get(cfg, ["objective", "mae_multidata_varlen_altattn_flu"], {}) or {}
        _ssl_method = str(_mm_cfg.get("ssl_method", "mae")).lower()

        if _ssl_method in ("simmtm", "focal", "relcon"):
            from imu_lm.objectives import ssl_vit_baselines as _sslb
            obj_blk = cfg_get(cfg, ["objective"], {}) or {}
            _log.info("[mae_multidata_varlen_altattn_flu] SSL baseline on shared ViT1DEncoder: %s%s",
                      _ssl_method, " (variable length)" if _var_len else "")
            if _ssl_method == "simmtm":
                decoder = _sslb.SimMTMHeadVarLen(
                    embed_dim=encoder.embed_dim, patch_size=encoder.patch_size,
                    proj_dim=int(obj_blk.get("proj_dim", 128)),
                    temperature=float(obj_blk.get("temperature", 0.2)),
                    positive_nums=int(obj_blk.get("positive_nums", 3)))
                def objective_fn(batch, model, cfg):
                    return _sslb.simmtm_forward_loss_varlen(batch, encoder, decoder, cfg)
                all_params = list(encoder.parameters()) + list(decoder.parameters())
            elif _ssl_method == "focal":
                decoder = _sslb.FOCALHead(
                    embed_dim=encoder.embed_dim,
                    modalities=list(obj_blk.get("modalities", ["acc", "gyro"])),
                    emb_dim=int(obj_blk.get("emb_dim", 128)))
                def objective_fn(batch, model, cfg):
                    return _sslb.focal_forward_loss(batch, encoder, decoder, cfg)
                all_params = list(encoder.parameters()) + list(decoder.parameters())
            else:  # relcon
                # Stage 2 of the RelCon baseline (imu_lm/objectives/relcon_ssl.py) --
                # trains the SAME shared ViT1DEncoder via RelCon's relative
                # contrastive loss, using a FROZEN Stage-1 MotifDist distance
                # network (trained separately via
                # scripts/train_relcon_motifdist.py) to rank within-batch
                # candidates. No learnable head beyond the encoder itself --
                # `decoder` is a no-op placeholder only so the checkpointing
                # dispatch below (which expects a `decoder` variable for this
                # objective_type) has something harmless to save.
                import torch.nn as nn
                from imu_lm.objectives.relcon_ssl import MotifDistNet, relcon_forward_loss
                motifdist_run = obj_blk.get("motifdist_run")
                if not motifdist_run:
                    raise ValueError("[mae_multidata_varlen_altattn_flu/relcon] "
                                      "'objective.motifdist_run' must be set (read from the TOP-LEVEL "
                                      "objective block, not objective.relcon.*)")
                runs_root_ = cfg_get(cfg, ["paths", "runs_root"], "runs")
                motifdist_ckpt_path = os.path.join(runs_root_, motifdist_run, "checkpoints", "best.pt")
                motifdist_ckpt = torch.load(motifdist_ckpt_path, map_location="cpu", weights_only=False)
                motifdist_net = MotifDistNet(
                    in_channels=int(motifdist_ckpt.get("in_channels", 6)),
                    embed_dim=int(motifdist_ckpt.get("embed_dim", 64)),
                    kernel_size=int(motifdist_ckpt.get("kernel_size", 15)),
                )
                motifdist_net.load_state_dict(motifdist_ckpt["model"])
                motifdist_net.requires_grad_(False)
                motifdist_net.eval()
                # NOTE: encoder is not moved to its final device until
                # Trainer.fit() runs (after this dispatch block returns), so
                # next(encoder.parameters()).device would be wrong here
                # (still cpu) -- match Trainer's own device selection instead
                # (imu_lm/runtime_consistency/trainer.py: self.device).
                motifdist_net = motifdist_net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                _log.info("[mae_multidata_varlen_altattn_flu/relcon] loaded frozen MotifDist "
                          "from %s (step=%s)", motifdist_ckpt_path, motifdist_ckpt.get("step"))
                decoder = nn.Identity()
                def objective_fn(batch, model, cfg):
                    return relcon_forward_loss(batch, encoder, motifdist_net, cfg)
                all_params = list(encoder.parameters())

            loaders = make_combined_mae_multidata_loaders(cfg)

        else:
            # MAE (default). Capture24 is accelerometer-only: its batches carry a
            # per-sample channel_valid_mask (all channels valid except capture24's
            # 3 fake gyro channels), consumed by masked_1d_freq_channel_mask (NOT
            # the plain masked_1d_freq used by every other objective in this file)
            # to force those channels to always be masked/hidden from the encoder
            # and always excluded from the reconstruction loss.
            from imu_lm.objectives import masked_1d_freq_channel_mask as freq_mae_obj

            if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
                from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
                encoder = ViT1DEncoderAltAttn(cfg)
                _log.info("[mae_multidata_varlen_altattn_flu] using alternating frame/global "
                          "attention encoder (VGGT-style)%s",
                          " (variable length)" if _var_len else "")

            decoder = ViT1DDecoder(
                embed_dim=encoder.embed_dim, out_channels=encoder.in_channels,
                target_T=(0 if _var_len else T), cfg=cfg,
            )

            # ---- optional warm-start from an earlier MAE stage (added 2026-07-28) ----
            # For MULTI-STAGE MAE pretraining, e.g. stage 1 on a large acc-only
            # pool then stage 2 on the smaller acc+gyro pool. Distinct from
            # `--resume`: resume restores optimizer/scheduler/step to continue
            # ONE run, whereas a new stage wants the learned weights but a
            # FRESH optimizer, scheduler and step counter (new data, new LR
            # schedule). Loads the encoder always, and the decoder too by
            # default -- continuing the same MAE objective, so the stage-1
            # decoder is a strictly better starting point than re-initializing
            # it (set load_decoder: false to reinitialize instead).
            _stage_run  = _mm_cfg.get("pretrained_run")
            _stage_ckpt = _mm_cfg.get("pretrained_ckpt", "best")
            if _stage_run:
                _runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
                if _stage_ckpt in ("best", "latest"):
                    _stage_path = os.path.join(_runs_root, _stage_run,
                                                "checkpoints", f"{_stage_ckpt}.pt")
                else:
                    _stage_path = _stage_ckpt
                if not os.path.exists(_stage_path):
                    raise FileNotFoundError(
                        f"[mae_multidata_varlen_altattn_flu] stage warm-start checkpoint "
                        f"not found: {_stage_path}"
                    )
                _log.info("[mae_multidata_varlen_altattn_flu] stage warm-start from %s "
                          "(fresh optimizer/scheduler/step -- NOT a --resume)", _stage_path)
                _stage_state = torch.load(_stage_path, map_location="cpu", weights_only=False)
                _load_encoder_state_dict(encoder, _stage_state)
                if bool(_mm_cfg.get("load_decoder", True)):
                    if "decoder" not in _stage_state:
                        raise KeyError(
                            f"[mae_multidata_varlen_altattn_flu] load_decoder is true but "
                            f"{_stage_path} has no 'decoder' state (keys: "
                            f"{sorted(k for k in _stage_state if k != 'cfg')}). Set "
                            f"load_decoder: false to start stage 2 with a fresh decoder."
                        )
                    decoder.load_state_dict(_stage_state["decoder"], strict=True)
                    _log.info("[mae_multidata_varlen_altattn_flu] stage warm-start: decoder loaded too")
                else:
                    _log.info("[mae_multidata_varlen_altattn_flu] stage warm-start: encoder only, "
                              "decoder re-initialized (load_decoder: false)")

            # IMU-Bench transfer study, "masked reconstruction" as a TARGET task: `freeze_encoder: true`
            # keeps the (warm-started) encoder fixed and trains only the MAE decoder, the frozen-probe
            # counterpart of every other target; without it this branch is the trainable transfer.
            _mae_freeze = bool(_mm_cfg.get("freeze_encoder", False))
            if _mae_freeze:
                _log.info("[mae_multidata_varlen_altattn_flu] freeze_encoder=true -> encoder frozen, "
                          "only the MAE decoder is trained")
                encoder.requires_grad_(False)
                encoder.eval()

            def objective_fn(batch, model, cfg):
                if _mae_freeze:
                    encoder.eval()
                return freq_mae_obj.forward_loss(batch, encoder, decoder, cfg)

            all_params = (list(decoder.parameters()) if _mae_freeze
                          else list(encoder.parameters()) + list(decoder.parameters()))
            loaders = make_combined_mae_multidata_loaders(cfg)

    elif objective_type == "mae_ocdetect_anyrot":
        # MAE pretraining on OCDetect in raw Android frame + uniform SO(3) random rotation.
        # Skipping FLU transform means orientation is arbitrary; random_rotation augmentation
        # (enabled in the YAML config) then sees every possible orientation during training,
        # producing an orientation-invariant representation.
        from imu_lm.data.OCDetect_dataloader import (
            make_ocdetect_loaders_random,
            DATA_DIR_DEFAULT as _OCDetect_DATA_DIR,
        )
        ocdetect_cfg = cfg_get(cfg, ["objective", "mae_ocdetect_anyrot"], {}) or {}
        data_dir_    = ocdetect_cfg.get("data_dir", _OCDetect_DATA_DIR)
        splits_      = tuple(ocdetect_cfg.get("splits", [0.8, 0.1, 0.1]))
        num_workers_ = int(cfg_get(cfg, ["data", "num_workers"], 0))
        pin_memory_  = bool(cfg_get(cfg, ["data", "pin_memory"], False))
        batch_size_  = int(cfg_get(cfg, ["data", "batch_size"], 256))
        eval_bs_     = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size_))
        apply_flu_   = bool(ocdetect_cfg.get("apply_flu_transform", False))

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
            apply_flu_transform=apply_flu_,
            cfg=cfg,
        )

    elif objective_type == "jepa_ocdetect_flu":
        # I-JEPA style SSL on OCDetect wrist-IMU data (FLU frame, 50 Hz, 2.56 s windows).
        # context_encoder (= encoder, trainable) sees only unmasked context patches.
        # target_encoder_ema (EMA copy, no grad) sees the full unmasked input.
        # predictor (narrow transformer) maps context tokens → predicted target embeddings.
        # Loss: smooth-L1 between predictor output and layer-normed target encoder output
        #        at masked positions.
        import copy as _copy
        from imu_lm.objectives import jepa_1d as jepa_1d_obj
        from imu_lm.models.ViT1D.model import JEPA1DPredictor
        from imu_lm.data.OCDetect_dataloader import (
            make_ocdetect_loaders_random,
            DATA_DIR_DEFAULT as _OCDetect_DATA_DIR,
        )

        jepa_cfg_   = cfg_get(cfg, ["objective", "jepa_ocdetect_flu"], {}) or {}
        data_dir_   = jepa_cfg_.get("data_dir", _OCDetect_DATA_DIR)
        splits_     = tuple(jepa_cfg_.get("splits", [0.8, 0.1, 0.1]))
        ema_momentum = float(jepa_cfg_.get("ema_momentum", 0.996))
        num_workers_ = int(cfg_get(cfg, ["data", "num_workers"], 0))
        pin_memory_  = bool(cfg_get(cfg, ["data", "pin_memory"], False))
        batch_size_  = int(cfg_get(cfg, ["data", "batch_size"], 256))
        eval_bs_     = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size_))

        # EMA target encoder: deep copy of context encoder, no grad, no optimizer
        target_encoder_ema = _copy.deepcopy(encoder)
        for _p in target_encoder_ema.parameters():
            _p.requires_grad_(False)
        target_encoder_ema.eval()

        # Narrow predictor transformer
        predictor = JEPA1DPredictor(embed_dim=encoder.embed_dim, cfg=cfg)

        # Capture in closure; EMA updated inside objective_fn (one-step delay is fine)
        _context_encoder    = encoder
        _target_encoder_ema = target_encoder_ema
        _predictor          = predictor
        _ema_momentum       = ema_momentum

        def objective_fn(batch, model, cfg):
            loss, logs = jepa_1d_obj.forward_loss(
                batch, _context_encoder, _target_encoder_ema, _predictor, cfg,
            )
            # EMA update: only during training (not val/eval)
            if _context_encoder.training:
                jepa_1d_obj.update_ema(_context_encoder, _target_encoder_ema, _ema_momentum)
            return loss, logs

        # Only context encoder + predictor go into optimizer; target is EMA-only
        all_params = list(encoder.parameters()) + list(predictor.parameters())

        loaders = make_ocdetect_loaders_random(
            data_dir=data_dir_,
            splits=splits_,
            batch_size=batch_size_,
            eval_batch_size=eval_bs_,
            num_workers=num_workers_,
            pin_memory=pin_memory_,
            cfg=cfg,
        )

    elif objective_type == "dino_ocdetect_anyrot":
        # DINOv2-style SSL on OCDetect wrist-IMU (any orientation, 50 Hz, 2.56 s windows).
        #
        # Each window produces TWO independently-augmented views (multi_view=True):
        #   View 1 → student encoder (+ iBOT patch masking inside forward_dino)
        #   View 2 → teacher encoder EMA (full, unmasked)
        #
        # Two losses summed:
        #   CLS  loss (DINO)  — cross-entropy on global CLS tokens
        #   Patch loss (iBOT) — cross-entropy at masked patch positions only
        #
        # Augmentations (both views, independent random draws):
        #   random_rotation: uniform SO(3) on acc+gyro → orientation-invariant repr
        #   gaussian_noise / jitter: applied via standard apply_augment pipeline
        #
        # FLU transform is skipped (apply_flu_transform=false); random rotation already
        # covers all orientations so starting coordinate frame is irrelevant.
        import copy as _copy
        from imu_lm.objectives import dino_1d as dino_1d_obj
        from imu_lm.objectives.dino_1d import DINOLoss as _DINOLoss
        from imu_lm.models.ViT1D.model import ViT1DDINOHead
        from imu_lm.data.OCDetect_dataloader import (
            make_ocdetect_loaders_random,
            DATA_DIR_DEFAULT as _OCDetect_DATA_DIR,
        )

        dino_cfg_         = cfg_get(cfg, ["objective", "dino_ocdetect_anyrot"], {}) or {}
        data_dir_         = dino_cfg_.get("data_dir", _OCDetect_DATA_DIR)
        splits_           = tuple(dino_cfg_.get("splits", [0.8, 0.1, 0.1]))
        ema_momentum_end  = float(dino_cfg_.get("ema_momentum",       0.996))
        ema_momentum_start= float(dino_cfg_.get("ema_momentum_start", 0.9))
        apply_flu_        = bool(dino_cfg_.get("apply_flu_transform", False))
        train_frac_   = float(dino_cfg_.get("train_fraction", 1.0))
        max_files_    = dino_cfg_.get("max_files", None)
        if max_files_ is not None:
            max_files_ = int(max_files_)

        # Window / hop from windowing config (supports 2.56 s default and 10.24 s etc.)
        from imu_lm.data.windowing import compute_T_and_hop as _ctah
        _win_T, _win_hop = _ctah(cfg)

        num_workers_ = int(cfg_get(cfg, ["data", "num_workers"], 0))
        pin_memory_  = bool(cfg_get(cfg, ["data", "pin_memory"], False))
        batch_size_  = int(cfg_get(cfg, ["data", "batch_size"], 256))
        eval_bs_     = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size_))

        # DINO head config
        dino_head_cfg  = (cfg_get(cfg, ["vit1d"], {}) or {}).get("dino_head", {}) or {}
        out_dim_       = int(dino_head_cfg.get("out_dim",        4096))
        hidden_dim_    = int(dino_head_cfg.get("hidden_dim",     2048))
        bottleneck_    = int(dino_head_cfg.get("bottleneck_dim",  256))

        # Loss config
        loss_cfg_               = dino_cfg_.get("loss", {}) or {}
        teacher_temp_           = float(loss_cfg_.get("teacher_temp",             0.07))
        warmup_teacher_temp_    = float(loss_cfg_.get("warmup_teacher_temp",      0.04))
        warmup_teacher_temp_steps_ = int(loss_cfg_.get("warmup_teacher_temp_steps", 0))
        student_temp_           = float(loss_cfg_.get("student_temp",             0.10))
        center_mom_             = float(loss_cfg_.get("center_momentum",          0.90))
        lambda_patch_           = float(loss_cfg_.get("lambda_patch",             1.0))

        # Steps to freeze prototype directions (last_layer.weight_v) after init.
        # Mirrors the original DINO "freeze_last_layer" trick: prototype vectors stay
        # fixed while the encoder + centering buffer stabilise, preventing the student
        # from rotating prototypes to a collapse shortcut in the first few hundred steps.
        # Default: same as LR warmup_steps so it covers the most vulnerable period.
        _warmup_steps_default = int(cfg_get(cfg, ["trainer", "warmup_steps"], 1000))
        _freeze_last_layer_steps = int(
            dino_cfg_.get("freeze_last_layer_steps", _warmup_steps_default)
        )

        # Teacher encoder: EMA copy, frozen
        teacher_encoder_ema = _copy.deepcopy(encoder)
        for _p in teacher_encoder_ema.parameters():
            _p.requires_grad_(False)
        teacher_encoder_ema.eval()

        # Student DINO head (trainable)
        dino_head = ViT1DDINOHead(
            in_dim=encoder.embed_dim,
            out_dim=out_dim_,
            hidden_dim=hidden_dim_,
            bottleneck_dim=bottleneck_,
        )
        dino_head = dino_head.to(next(encoder.parameters()).device)

        # Teacher DINO head: EMA copy of student head, frozen.
        # deepcopy fails on weight_norm modules, so create fresh and copy weights manually.
        teacher_dino_head = ViT1DDINOHead(
            in_dim=encoder.embed_dim,
            out_dim=out_dim_,
            hidden_dim=hidden_dim_,
            bottleneck_dim=bottleneck_,
        )
        with torch.no_grad():
            for p_s, p_t in zip(dino_head.parameters(), teacher_dino_head.parameters()):
                p_t.data.copy_(p_s.data)
        for _p in teacher_dino_head.parameters():
            _p.requires_grad_(False)
        teacher_dino_head.eval()
        teacher_dino_head = teacher_dino_head.to(next(encoder.parameters()).device)

        # Loss module (holds centering buffers)
        dino_loss_module = _DINOLoss(
            out_dim=out_dim_,
            teacher_temp=teacher_temp_,
            student_temp=student_temp_,
            center_momentum=center_mom_,
            lambda_patch=lambda_patch_,
        )
        dino_loss_module = dino_loss_module.to(next(encoder.parameters()).device)

        # Capture in closure
        _student_enc    = encoder
        _teacher_enc    = teacher_encoder_ema
        _student_head   = dino_head
        _teacher_head   = teacher_dino_head
        _dino_loss      = dino_loss_module
        _max_steps      = int(cfg_get(cfg, ["trainer", "max_steps"], 1000000))
        # Mutable state for cosine-annealed EMA momentum and teacher temp warmup.
        # _dino_state["step"] is synced with start_step after checkpoint load below.
        _dino_state     = {
            "step":      0,
            "ema_start": ema_momentum_start,
            "ema_end":   ema_momentum_end,
            # teacher temperature warmup bookkeeping
            "t_temp_start": warmup_teacher_temp_,
            "t_temp_end":   teacher_temp_,
            "t_temp_steps": warmup_teacher_temp_steps_,
        }

        # Register gradient hook to freeze prototype directions for the first N steps.
        # weight_g (norm) is permanently frozen; this also freezes weight_v (direction).
        # The hook captures _dino_state by reference so it checks the live step counter.
        def _last_layer_freeze_hook(grad):
            if _dino_state["step"] < _freeze_last_layer_steps:
                return torch.zeros_like(grad)
            return grad
        dino_head.last_layer.weight_v.register_hook(_last_layer_freeze_hook)

        def objective_fn(batch, model, cfg):
            import math as _math
            state = _dino_state
            step  = state["step"]

            # ── EMA momentum: cosine anneal ema_start → ema_end over max_steps ──
            t        = min(step, _max_steps)
            cos_val  = _math.cos(_math.pi * t / _max_steps)
            momentum = state["ema_end"] - (state["ema_end"] - state["ema_start"]) * (1.0 + cos_val) / 2.0

            # ── Teacher temperature: linear warmup t_temp_start → t_temp_end ───
            # Matches DINOv2: start sharp (0.04), warm to softer (0.07) over N steps.
            # After warmup, stays at t_temp_end.
            warm_steps = state["t_temp_steps"]
            if warm_steps > 0 and step < warm_steps:
                alpha = step / warm_steps
                cur_teacher_temp = state["t_temp_start"] + alpha * (state["t_temp_end"] - state["t_temp_start"])
            else:
                cur_teacher_temp = state["t_temp_end"]
            _dino_loss.teacher_temp = cur_teacher_temp

            loss, logs = dino_1d_obj.forward_loss(
                batch, _student_enc, _teacher_enc,
                _student_head, _teacher_head,
                _dino_loss, cfg,
            )
            if _student_enc.training:
                dino_1d_obj.update_ema(_student_enc, _teacher_enc, momentum)
                dino_1d_obj.update_ema(_student_head, _teacher_head, momentum)
                state["step"] += 1
                logs["ema_momentum"]   = momentum
                logs["teacher_temp"]   = cur_teacher_temp
            return loss, logs

        # Optimizer covers student encoder + DINO head; teacher is EMA-only
        all_params = list(encoder.parameters()) + list(dino_head.parameters())

        loaders = make_ocdetect_loaders_random(
            data_dir=data_dir_,
            splits=splits_,
            batch_size=batch_size_,
            eval_batch_size=eval_bs_,
            num_workers=num_workers_,
            pin_memory=pin_memory_,
            apply_flu_transform=apply_flu_,
            multi_view=True,
            window_samples=_win_T,
            hop_samples=_win_hop,
            train_fraction=train_frac_,
            max_files=max_files_,
            cfg=cfg,
        )

    elif objective_type == "mae_dino_ocdetect_anyrot":
        # MAE + DINOv1 CLS combined SSL on OCDetect wrist-IMU (any orientation).
        #
        # Two independently-augmented views per window (multi_view=True):
        #   View 1 → student encoder (masked, BERT-style [MASK] tokens)
        #            → patch tokens at masked positions → reconstruction head → MAE MSE loss
        #            → CLS token → DINO head → student prototype distribution
        #   View 2 → teacher encoder EMA (full, no masking)
        #            → CLS token → DINO head EMA → teacher prototype distribution
        #            → DINO CLS cross-entropy loss
        #
        # No iBOT patch distillation — simpler and more stable than DINOv2.
        import copy as _copy
        from imu_lm.objectives import mae_dino_1d as mae_dino_obj
        from imu_lm.objectives.mae_dino_1d import DINOCLSLoss as _DINOCLSLoss
        from imu_lm.objectives.mae_dino_1d import PatchReconHead as _PatchReconHead
        from imu_lm.models.ViT1D.model import ViT1DDINOHead
        from imu_lm.data.OCDetect_dataloader import (
            make_ocdetect_loaders_random,
            DATA_DIR_DEFAULT as _OCDetect_DATA_DIR,
        )

        md_cfg_           = cfg_get(cfg, ["objective", "mae_dino_ocdetect_anyrot"], {}) or {}
        data_dir_         = md_cfg_.get("data_dir", _OCDetect_DATA_DIR)
        splits_           = tuple(md_cfg_.get("splits", [0.8, 0.1, 0.1]))
        ema_momentum_end  = float(md_cfg_.get("ema_momentum",       0.996))
        ema_momentum_start= float(md_cfg_.get("ema_momentum_start", 0.994))
        apply_flu_        = bool(md_cfg_.get("apply_flu_transform", False))

        from imu_lm.data.windowing import compute_T_and_hop as _ctah
        _win_T, _win_hop = _ctah(cfg)

        num_workers_ = int(cfg_get(cfg, ["data", "num_workers"], 0))
        pin_memory_  = bool(cfg_get(cfg, ["data", "pin_memory"], False))
        batch_size_  = int(cfg_get(cfg, ["data", "batch_size"], 256))
        eval_bs_     = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size_))

        # DINO head config
        dino_head_cfg_ = (cfg_get(cfg, ["vit1d"], {}) or {}).get("dino_head", {}) or {}
        out_dim_       = int(dino_head_cfg_.get("out_dim",        4096))
        hidden_dim_    = int(dino_head_cfg_.get("hidden_dim",     2048))
        bottleneck_    = int(dino_head_cfg_.get("bottleneck_dim",  256))

        # Loss config
        loss_cfg_                  = md_cfg_.get("loss", {}) or {}
        teacher_temp_              = float(loss_cfg_.get("teacher_temp",              0.07))
        warmup_teacher_temp_       = float(loss_cfg_.get("warmup_teacher_temp",       0.04))
        warmup_teacher_temp_steps_ = int(loss_cfg_.get("warmup_teacher_temp_steps",   10000))
        student_temp_              = float(loss_cfg_.get("student_temp",              0.10))
        center_mom_                = float(loss_cfg_.get("center_momentum",           0.9))
        lambda_koleo_              = float(loss_cfg_.get("lambda_koleo",              0.1))
        lambda_mae_                = float(loss_cfg_.get("lambda_mae",                1.0))
        lambda_cls_                = float(loss_cfg_.get("lambda_cls",                1.0))

        _max_steps_md = int(cfg_get(cfg, ["trainer", "max_steps"], 1000000))

        # Teacher encoder: EMA copy, frozen
        teacher_encoder_ema = _copy.deepcopy(encoder)
        for _p in teacher_encoder_ema.parameters():
            _p.requires_grad_(False)
        teacher_encoder_ema.eval()

        # Student DINO head
        dino_head = ViT1DDINOHead(
            in_dim=encoder.embed_dim,
            out_dim=out_dim_,
            hidden_dim=hidden_dim_,
            bottleneck_dim=bottleneck_,
        ).to(next(encoder.parameters()).device)

        # Teacher DINO head: EMA copy (avoid deepcopy — weight_norm incompatible)
        teacher_dino_head = ViT1DDINOHead(
            in_dim=encoder.embed_dim,
            out_dim=out_dim_,
            hidden_dim=hidden_dim_,
            bottleneck_dim=bottleneck_,
        )
        with torch.no_grad():
            for p_s, p_t in zip(dino_head.parameters(), teacher_dino_head.parameters()):
                p_t.data.copy_(p_s.data)
        for _p in teacher_dino_head.parameters():
            _p.requires_grad_(False)
        teacher_dino_head.eval()
        teacher_dino_head = teacher_dino_head.to(next(encoder.parameters()).device)

        # Reconstruction head: patch_embedding → patch_size signal values
        recon_head = _PatchReconHead(
            embed_dim=encoder.embed_dim,
            patch_size=encoder.patch_size,
        ).to(next(encoder.parameters()).device)

        # CLS-only DINO loss module
        dino_cls_loss_module = _DINOCLSLoss(
            out_dim=out_dim_,
            teacher_temp=teacher_temp_,
            student_temp=student_temp_,
            center_momentum=center_mom_,
        ).to(next(encoder.parameters()).device)

        # freeze_last_layer for first N steps (same DINO stability trick)
        _warmup_steps_default = int(cfg_get(cfg, ["trainer", "sched", "warmup_steps"], 1000))
        _freeze_ll_steps = int(md_cfg_.get("freeze_last_layer_steps", _warmup_steps_default))

        _md_state = {
            "step":         0,
            "ema_start":    ema_momentum_start,
            "ema_end":      ema_momentum_end,
            "t_temp_start": warmup_teacher_temp_,
            "t_temp_end":   teacher_temp_,
            "t_temp_steps": warmup_teacher_temp_steps_,
        }

        def _ll_freeze_hook(grad):
            if _md_state["step"] < _freeze_ll_steps:
                return torch.zeros_like(grad)
            return grad
        dino_head.last_layer.weight_v.register_hook(_ll_freeze_hook)

        _md_enc_s  = encoder
        _md_enc_t  = teacher_encoder_ema
        _md_head_s = dino_head
        _md_head_t = teacher_dino_head
        _md_recon  = recon_head
        _md_loss   = dino_cls_loss_module

        def objective_fn(batch, model, cfg):
            import math as _math
            state = _md_state
            step  = state["step"]

            # EMA momentum: cosine anneal
            t        = min(step, _max_steps_md)
            cos_val  = _math.cos(_math.pi * t / _max_steps_md)
            momentum = state["ema_end"] - (state["ema_end"] - state["ema_start"]) * (1.0 + cos_val) / 2.0

            # Teacher temperature: linear warmup
            warm_steps = state["t_temp_steps"]
            if warm_steps > 0 and step < warm_steps:
                alpha = step / warm_steps
                cur_t_temp = state["t_temp_start"] + alpha * (state["t_temp_end"] - state["t_temp_start"])
            else:
                cur_t_temp = state["t_temp_end"]
            _md_loss.teacher_temp = cur_t_temp

            loss, logs = mae_dino_obj.forward_loss(
                batch, _md_enc_s, _md_enc_t,
                _md_head_s, _md_head_t,
                _md_recon, _md_loss, cfg,
                lambda_koleo=lambda_koleo_,
            )
            if _md_enc_s.training:
                mae_dino_obj.update_ema(_md_enc_s, _md_enc_t, momentum)
                mae_dino_obj.update_ema(_md_head_s, _md_head_t, momentum)
                state["step"] += 1
                logs["ema_momentum"] = momentum
                logs["teacher_temp"] = cur_t_temp
            return loss, logs

        # Optimizer covers encoder + DINO head + reconstruction head
        all_params = (list(encoder.parameters())
                      + list(dino_head.parameters())
                      + list(recon_head.parameters()))

        loaders = make_ocdetect_loaders_random(
            data_dir=data_dir_,
            splits=splits_,
            batch_size=batch_size_,
            eval_batch_size=eval_bs_,
            num_workers=num_workers_,
            pin_memory=pin_memory_,
            apply_flu_transform=apply_flu_,
            multi_view=True,
            window_samples=_win_T,
            hop_samples=_win_hop,
            cfg=cfg,
        )

    elif objective_type == "rvq_tokenizer_ocdetect_anyrot":
        # Discrete IMU tokenizer with frequency-progressive Residual Vector Quantization.
        #
        # Encoder (ViT1D, no CLS token) → projection (optional) → K-level RVQ
        # Each RVQ level predicts a frequency band of the original signal:
        #   Level 1:  0 – 2 Hz (low-frequency motion trend)
        #   Level 2:  0 – 5 Hz (stride-level periodicity)
        #   Level 3:  0 – 12 Hz (fast limb movements)
        #   Level 4:  full signal (all frequencies)
        # Per-level PatchDecoder MLP: code_dim → 512 → patch_size
        # Codebooks updated via EMA (not gradients).
        import torch.nn as nn
        import copy as _copy
        from imu_lm.objectives import rvq_tokenizer as rvq_obj
        from imu_lm.objectives.rvq_tokenizer import (
            ResidualVectorQuantizer as _RVQ,
            PatchDecoder as _PatchDecoder,
        )
        from imu_lm.data.OCDetect_dataloader import (
            make_ocdetect_loaders_random,
            DATA_DIR_DEFAULT as _OCDetect_DATA_DIR,
        )

        rvq_cfg_          = cfg_get(cfg, ["objective", "rvq_tokenizer_ocdetect_anyrot"], {}) or {}
        data_dir_         = rvq_cfg_.get("data_dir", _OCDetect_DATA_DIR)
        splits_           = tuple(rvq_cfg_.get("splits", [0.8, 0.1, 0.1]))
        apply_flu_        = bool(rvq_cfg_.get("apply_flu_transform", False))

        from imu_lm.data.windowing import compute_T_and_hop as _ctah
        _win_T, _win_hop = _ctah(cfg)

        num_workers_ = int(cfg_get(cfg, ["data", "num_workers"], 0))
        pin_memory_  = bool(cfg_get(cfg, ["data", "pin_memory"], False))
        batch_size_  = int(cfg_get(cfg, ["data", "batch_size"], 256))
        eval_bs_     = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size_))

        # RVQ config
        num_levels_         = int(rvq_cfg_.get("num_levels",          4))
        num_codes_          = int(rvq_cfg_.get("num_codes",           512))
        code_dim_           = int(rvq_cfg_.get("code_dim",            256))
        ema_decay_          = float(rvq_cfg_.get("ema_decay",         0.99))
        commitment_weight_  = float(rvq_cfg_.get("commitment_weight", 0.25))
        decoder_hidden_dim_ = int(rvq_cfg_.get("decoder_hidden_dim",  512))
        cutoff_hz_list_     = list(rvq_cfg_.get("cutoff_hz", [2.0, 5.0, 12.0, 999.0]))
        lambda_levels_      = list(rvq_cfg_.get("lambda_levels", [1.0, 1.0, 1.0, 1.0]))
        lambda_spec_        = float(rvq_cfg_.get("lambda_spec", 0.1))
        fs_                 = float(rvq_cfg_.get("fs", 50.0))
        pool_channels_      = bool(rvq_cfg_.get("pool_channels", True))
        num_channels_in_    = int(rvq_cfg_.get("num_channels",    6))

        # When pool_channels=True each temporal token decodes all C channels at once:
        # decoder output dim = C * patch_size instead of patch_size
        dec_out_dim_ = (num_channels_in_ * encoder.patch_size) if pool_channels_ else encoder.patch_size

        # Projection layer (optional): map encoder_embed_dim → code_dim
        _dev = next(encoder.parameters()).device
        if encoder.embed_dim != code_dim_:
            rvq_proj = nn.Linear(encoder.embed_dim, code_dim_).to(_dev)
        else:
            rvq_proj = None

        rvq_module = _RVQ(
            num_levels=num_levels_,
            num_codes=num_codes_,
            code_dim=code_dim_,
            ema_decay=ema_decay_,
            commitment_weight=commitment_weight_,
        ).to(_dev)

        rvq_decoders = nn.ModuleList([
            _PatchDecoder(code_dim_, dec_out_dim_, decoder_hidden_dim_)
            for _ in range(num_levels_)
        ]).to(_dev)

        _rvq_proj         = rvq_proj
        _rvq_module       = rvq_module
        _rvq_dec          = rvq_decoders
        _rvq_cutoffs      = cutoff_hz_list_
        _rvq_lams         = lambda_levels_
        _rvq_spec         = lambda_spec_
        _rvq_fs           = fs_
        _rvq_pool_ch      = pool_channels_

        def objective_fn(batch, model, cfg):
            return rvq_obj.forward_loss(
                batch, model, _rvq_proj, _rvq_module, _rvq_dec,
                cfg,
                cutoff_hz_list=_rvq_cutoffs,
                lambda_levels=_rvq_lams,
                lambda_spec=_rvq_spec,
                fs=_rvq_fs,
                pool_channels=_rvq_pool_ch,
            )

        # Optimizer covers encoder + projection (if any) + decoders
        # RVQ codebooks are updated via EMA — not included in optimizer
        all_params = list(encoder.parameters()) + list(rvq_decoders.parameters())
        if rvq_proj is not None:
            all_params += list(rvq_proj.parameters())

        loaders = make_ocdetect_loaders_random(
            data_dir=data_dir_,
            splits=splits_,
            batch_size=batch_size_,
            eval_batch_size=eval_bs_,
            num_workers=num_workers_,
            pin_memory=pin_memory_,
            apply_flu_transform=apply_flu_,
            multi_view=False,
            window_samples=_win_T,
            hop_samples=_win_hop,
            cfg=cfg,
        )

    elif objective_type == "rvq_tokenizer_nymeria_real_wrist_flu":
        # Discrete IMU tokenizer with frequency-progressive RVQ on Nymeria real wrist IMU.
        # Same RVQ architecture as rvq_tokenizer_ocdetect_anyrot but uses the Nymeria
        # real wrist FLU loader (pkl files) instead of OCDetect CSV files.
        import torch.nn as nn
        import glob as _glob
        from imu_lm.objectives import rvq_tokenizer as rvq_obj
        from imu_lm.objectives.rvq_tokenizer import (
            ResidualVectorQuantizer as _RVQ,
            PatchDecoder as _PatchDecoder,
        )
        from imu_lm.data.nymeria_real_wrist_loader_FLU import make_nymeria_real_wrist_loaders_flu

        rvq_nyr_cfg_       = cfg_get(cfg, ["objective", "rvq_tokenizer_nymeria_real_wrist_flu"], {}) or {}
        train_pkl_root_    = rvq_nyr_cfg_.get("train_pkl_root")
        val_pkl_root_      = rvq_nyr_cfg_.get("val_pkl_root") or train_pkl_root_
        add_gravity_       = bool(rvq_nyr_cfg_.get("add_gravity", True))

        def _expand_files_nyr(root, files_cfg):
            if files_cfg is None or files_cfg == "all":
                return sorted(_glob.glob(os.path.join(root, "*.pkl")))
            if isinstance(files_cfg, list):
                return [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
            return [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]

        train_files_nyr_ = _expand_files_nyr(train_pkl_root_, rvq_nyr_cfg_.get("train_files"))
        val_files_nyr_   = _expand_files_nyr(val_pkl_root_,   rvq_nyr_cfg_.get("val_files"))

        # RVQ config (same keys as rvq_tokenizer_ocdetect_anyrot)
        num_levels_nyr_        = int(rvq_nyr_cfg_.get("num_levels",          4))
        num_codes_nyr_         = int(rvq_nyr_cfg_.get("num_codes",           512))
        code_dim_nyr_          = int(rvq_nyr_cfg_.get("code_dim",            256))
        ema_decay_nyr_         = float(rvq_nyr_cfg_.get("ema_decay",         0.95))
        commitment_weight_nyr_ = float(rvq_nyr_cfg_.get("commitment_weight", 0.05))
        decoder_hidden_nyr_    = int(rvq_nyr_cfg_.get("decoder_hidden_dim",  512))
        cutoff_hz_nyr_         = list(rvq_nyr_cfg_.get("cutoff_hz", [2.0, 5.0, 12.0, 999.0]))
        lambda_levels_nyr_     = list(rvq_nyr_cfg_.get("lambda_levels", [1.0, 1.0, 1.0, 1.0]))
        lambda_spec_nyr_       = float(rvq_nyr_cfg_.get("lambda_spec", 0.1))
        fs_nyr_                = float(rvq_nyr_cfg_.get("fs", 50.0))
        pool_channels_nyr_     = bool(rvq_nyr_cfg_.get("pool_channels", True))
        num_channels_nyr_      = int(rvq_nyr_cfg_.get("num_channels",   6))

        dec_out_dim_nyr_ = (num_channels_nyr_ * encoder.patch_size) if pool_channels_nyr_ else encoder.patch_size

        _dev_nyr = next(encoder.parameters()).device
        if encoder.embed_dim != code_dim_nyr_:
            rvq_proj = nn.Linear(encoder.embed_dim, code_dim_nyr_).to(_dev_nyr)
        else:
            rvq_proj = None

        rvq_module = _RVQ(
            num_levels=num_levels_nyr_,
            num_codes=num_codes_nyr_,
            code_dim=code_dim_nyr_,
            ema_decay=ema_decay_nyr_,
            commitment_weight=commitment_weight_nyr_,
        ).to(_dev_nyr)

        rvq_decoders = nn.ModuleList([
            _PatchDecoder(code_dim_nyr_, dec_out_dim_nyr_, decoder_hidden_nyr_)
            for _ in range(num_levels_nyr_)
        ]).to(_dev_nyr)

        _rvq_proj_nyr    = rvq_proj
        _rvq_module_nyr  = rvq_module
        _rvq_dec_nyr     = rvq_decoders
        _rvq_cutoffs_nyr = cutoff_hz_nyr_
        _rvq_lams_nyr    = lambda_levels_nyr_
        _rvq_spec_nyr    = lambda_spec_nyr_
        _rvq_fs_nyr      = fs_nyr_
        _rvq_pool_nyr    = pool_channels_nyr_

        def objective_fn(batch, model, cfg):
            return rvq_obj.forward_loss(
                batch, model, _rvq_proj_nyr, _rvq_module_nyr, _rvq_dec_nyr,
                cfg,
                cutoff_hz_list=_rvq_cutoffs_nyr,
                lambda_levels=_rvq_lams_nyr,
                lambda_spec=_rvq_spec_nyr,
                fs=_rvq_fs_nyr,
                pool_channels=_rvq_pool_nyr,
            )

        all_params = list(encoder.parameters()) + list(rvq_decoders.parameters())
        if rvq_proj is not None:
            all_params += list(rvq_proj.parameters())

        loaders = make_nymeria_real_wrist_loaders_flu(
            cfg=cfg,
            pkl_root="/",
            train_files=train_files_nyr_,
            val_files=val_files_nyr_,
            add_gravity=add_gravity_,
        )

    elif objective_type in ("rvq_tokenizer_realdata_nopooling_8lvl",
                             "rvq_tokenizer_realdata_anyrot_nopooling_8lvl"):
        # Discrete IMU tokenizer with frequency-progressive RVQ on master parquet (realdata).
        # rvq_tokenizer_realdata_anyrot_nopooling_8lvl adds z-norm + random_rotation to
        # reduce dataset bias (NMI 0.51 → target <0.15).
        import torch.nn as nn
        from imu_lm.objectives import rvq_tokenizer as rvq_obj
        from imu_lm.objectives.rvq_tokenizer import (
            ResidualVectorQuantizer as _RVQ,
            PatchDecoder as _PatchDecoder,
        )

        rvq_rd_cfg_ = cfg_get(cfg, ["objective", objective_type], {}) or {}

        num_levels_rd_        = int(rvq_rd_cfg_.get("num_levels",          8))
        num_codes_rd_         = int(rvq_rd_cfg_.get("num_codes",           1024))
        code_dim_rd_          = int(rvq_rd_cfg_.get("code_dim",            256))
        ema_decay_rd_         = float(rvq_rd_cfg_.get("ema_decay",         0.95))
        commitment_weight_rd_ = float(rvq_rd_cfg_.get("commitment_weight", 0.05))
        decoder_hidden_rd_    = int(rvq_rd_cfg_.get("decoder_hidden_dim",  512))
        cutoff_hz_rd_         = list(rvq_rd_cfg_.get("cutoff_hz", [1.0, 2.0, 4.0, 6.0, 10.0, 15.0, 20.0, 999.0]))
        lambda_levels_rd_     = list(rvq_rd_cfg_.get("lambda_levels", [1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 3.0, 3.0]))
        lambda_spec_rd_       = float(rvq_rd_cfg_.get("lambda_spec",  0.1))
        fs_rd_                = float(rvq_rd_cfg_.get("fs",           50.0))
        pool_channels_rd_     = bool(rvq_rd_cfg_.get("pool_channels", False))
        num_channels_rd_      = int(rvq_rd_cfg_.get("num_channels",   6))

        dec_out_dim_rd_ = (num_channels_rd_ * encoder.patch_size) if pool_channels_rd_ else encoder.patch_size

        _dev_rd = next(encoder.parameters()).device
        if encoder.embed_dim != code_dim_rd_:
            rvq_proj = nn.Linear(encoder.embed_dim, code_dim_rd_).to(_dev_rd)
        else:
            rvq_proj = None

        rvq_module = _RVQ(
            num_levels=num_levels_rd_,
            num_codes=num_codes_rd_,
            code_dim=code_dim_rd_,
            ema_decay=ema_decay_rd_,
            commitment_weight=commitment_weight_rd_,
        ).to(_dev_rd)

        rvq_decoders = nn.ModuleList([
            _PatchDecoder(code_dim_rd_, dec_out_dim_rd_, decoder_hidden_rd_)
            for _ in range(num_levels_rd_)
        ]).to(_dev_rd)

        _rvq_proj_rd    = rvq_proj
        _rvq_module_rd  = rvq_module
        _rvq_dec_rd     = rvq_decoders
        _rvq_cutoffs_rd = cutoff_hz_rd_
        _rvq_lams_rd    = lambda_levels_rd_
        _rvq_spec_rd    = lambda_spec_rd_
        _rvq_fs_rd      = fs_rd_
        _rvq_pool_rd    = pool_channels_rd_

        def objective_fn(batch, model, cfg):
            return rvq_obj.forward_loss(
                batch, model, _rvq_proj_rd, _rvq_module_rd, _rvq_dec_rd,
                cfg,
                cutoff_hz_list=_rvq_cutoffs_rd,
                lambda_levels=_rvq_lams_rd,
                lambda_spec=_rvq_spec_rd,
                fs=_rvq_fs_rd,
                pool_channels=_rvq_pool_rd,
            )

        all_params = list(encoder.parameters()) + list(rvq_decoders.parameters())
        if rvq_proj is not None:
            all_params += list(rvq_proj.parameters())

        loaders = make_loaders(cfg)

    elif objective_type == "beit_ocdetect_anyrot":
        # BEiT v2 style pretraining on OCDetect wrist-IMU (any orientation).
        #
        # A frozen RVQ tokenizer (from vit1d_rvq_tokenizer_ocdetect_anyrot) converts
        # each temporal patch position into K discrete token indices.  The BEiT encoder
        # (ViT1D with CLS token) sees all positions — visible patches + [MASK] tokens +
        # CLS — and K linear heads predict the token index for each masked position.
        # Cross-entropy loss on masked patches only; CLS token is trained implicitly.
        import torch.nn as nn
        from imu_lm.objectives import beit_1d as beit_obj
        from imu_lm.objectives.beit_1d import (
            FrozenRVQTokenizer,
            load_frozen_tokenizer,
        )
        from imu_lm.data.OCDetect_dataloader import (
            make_ocdetect_loaders_random,
            DATA_DIR_DEFAULT as _OCDetect_DATA_DIR_BEiT,
        )
        from imu_lm.data.windowing import compute_T_and_hop as _ctah_beit

        assert encoder.use_cls_token, (
            "[beit_ocdetect_anyrot] encoder requires use_cls_token: true in vit1d config"
        )

        _dev_beit = next(encoder.parameters()).device
        beit_cfg_ = cfg_get(cfg, ["objective", "beit_ocdetect_anyrot"], {}) or {}

        # Data loaders
        data_dir_beit_  = beit_cfg_.get("data_dir",  _OCDetect_DATA_DIR_BEiT)
        splits_beit_    = tuple(beit_cfg_.get("splits",   [0.8, 0.1, 0.1]))
        apply_flu_beit_ = bool(beit_cfg_.get("apply_flu_transform", False))
        batch_size_beit_  = int(cfg_get(cfg, ["data", "batch_size"],      256))
        eval_bs_beit_     = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size_beit_))
        num_workers_beit_ = int(cfg_get(cfg, ["data", "num_workers"],     0))
        pin_mem_beit_     = bool(cfg_get(cfg, ["data", "pin_memory"],     False))
        _win_T_beit, _win_hop_beit = _ctah_beit(cfg)

        # Load frozen tokenizer
        runs_root_beit_ = cfg_get(cfg, ["paths", "runs_root"], "runs")
        tok_run_beit_   = beit_cfg_.get("tokenizer_run",
                                        "runs/vit1d_rvq_tokenizer_ocdetect_anyrot")
        # Resolve relative paths against runs_root
        if not os.path.isabs(tok_run_beit_):
            tok_run_beit_abs_ = tok_run_beit_   # already relative to cwd
        else:
            tok_run_beit_abs_ = tok_run_beit_

        _frozen_tokenizer = load_frozen_tokenizer(
            run_dir=tok_run_beit_abs_,
            device=_dev_beit,
            cfg_key="rvq_tokenizer_ocdetect_anyrot",
        )

        # BEiT prediction heads: one Linear(D, num_codes) per RVQ level
        num_levels_beit_ = int(beit_cfg_.get("num_levels", 4))
        num_codes_beit_  = int(beit_cfg_.get("num_codes",  512))
        beit_heads = nn.ModuleList([
            nn.Linear(encoder.embed_dim, num_codes_beit_)
            for _ in range(num_levels_beit_)
        ]).to(_dev_beit)

        _beit_tokenizer = _frozen_tokenizer
        _beit_heads     = beit_heads

        def objective_fn(batch, model, cfg):
            return beit_obj.forward_loss(
                batch, model, _beit_heads, cfg, _beit_tokenizer
            )

        # Only BEiT encoder + prediction heads are optimised; tokenizer is frozen
        all_params = list(encoder.parameters()) + list(beit_heads.parameters())

        loaders = make_ocdetect_loaders_random(
            data_dir=data_dir_beit_,
            splits=splits_beit_,
            batch_size=batch_size_beit_,
            eval_batch_size=eval_bs_beit_,
            num_workers=num_workers_beit_,
            pin_memory=pin_mem_beit_,
            apply_flu_transform=apply_flu_beit_,
            multi_view=False,
            window_samples=_win_T_beit,
            hop_samples=_win_hop_beit,
            cfg=cfg,
        )

    elif objective_type == "beit_ocdetect_anyrot_nopooling_8lvl":
        # BEiT v2 pretraining with 8-level no-pooling RVQ tokenizer.
        # Tokenizer: vit1d_rvq_tokenizer_ocdetect_anyrot_nopooling_8lvl
        #   pool_channels=False → one code per (channel, patch), 384 targets per window
        #   8 levels × 1024 codes covering frequency bands 1–20 Hz progressively
        # Encoder: ViT-Small (12 layers, 384 dim) with CLS token
        # Heads: 8 × Linear(384, 1024) — one per RVQ level
        import torch.nn as nn
        from imu_lm.objectives import beit_1d as beit_obj
        from imu_lm.objectives.beit_1d import load_frozen_tokenizer
        from imu_lm.data.OCDetect_dataloader import (
            make_ocdetect_loaders_random,
            DATA_DIR_DEFAULT as _OCDetect_DATA_DIR_BEiT8,
        )
        from imu_lm.data.windowing import compute_T_and_hop as _ctah_beit8

        assert encoder.use_cls_token, (
            "[beit_ocdetect_anyrot_nopooling_8lvl] encoder requires use_cls_token: true"
        )

        _dev_beit8 = next(encoder.parameters()).device
        beit8_cfg_ = cfg_get(cfg, ["objective", "beit_ocdetect_anyrot_nopooling_8lvl"], {}) or {}

        data_dir_beit8_   = beit8_cfg_.get("data_dir",  _OCDetect_DATA_DIR_BEiT8)
        splits_beit8_     = tuple(beit8_cfg_.get("splits",   [0.8, 0.1, 0.1]))
        apply_flu_beit8_  = bool(beit8_cfg_.get("apply_flu_transform", False))
        batch_size_beit8_ = int(cfg_get(cfg, ["data", "batch_size"],      256))
        eval_bs_beit8_    = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size_beit8_))
        num_workers_beit8_= int(cfg_get(cfg, ["data", "num_workers"],     0))
        pin_mem_beit8_    = bool(cfg_get(cfg, ["data", "pin_memory"],     False))
        _win_T_beit8, _win_hop_beit8 = _ctah_beit8(cfg)

        tok_run_beit8_ = beit8_cfg_.get(
            "tokenizer_run",
            "runs/vit1d_rvq_tokenizer_ocdetect_anyrot_nopooling_8lvl",
        )

        _frozen_tokenizer8 = load_frozen_tokenizer(
            run_dir=tok_run_beit8_,
            device=_dev_beit8,
            cfg_key="rvq_tokenizer_ocdetect_anyrot",
        )

        num_levels_beit8_ = int(beit8_cfg_.get("num_levels", 8))
        num_codes_beit8_  = int(beit8_cfg_.get("num_codes",  1024))
        beit8_heads = nn.ModuleList([
            nn.Linear(encoder.embed_dim, num_codes_beit8_)
            for _ in range(num_levels_beit8_)
        ]).to(_dev_beit8)

        _beit8_tokenizer = _frozen_tokenizer8
        _beit8_heads     = beit8_heads

        def objective_fn(batch, model, cfg):
            return beit_obj.forward_loss(
                batch, model, _beit8_heads, cfg, _beit8_tokenizer
            )

        all_params = list(encoder.parameters()) + list(beit8_heads.parameters())

        loaders = make_ocdetect_loaders_random(
            data_dir=data_dir_beit8_,
            splits=splits_beit8_,
            batch_size=batch_size_beit8_,
            eval_batch_size=eval_bs_beit8_,
            num_workers=num_workers_beit8_,
            pin_memory=pin_mem_beit8_,
            apply_flu_transform=apply_flu_beit8_,
            multi_view=False,
            window_samples=_win_T_beit8,
            hop_samples=_win_hop_beit8,
            cfg=cfg,
        )

    elif objective_type == "beit_realdata_nopooling_8lvl":
        # BEiT v2 pretraining with 8-level no-pooling RVQ tokenizer trained on realdata.
        # Tokenizer: vit1d_rvq_tokenizer_realdata_nopooling_8lvl
        #   pool_channels=False, no anyrot, orientation-aware
        # Encoder: ViT-Small (12 layers, 384 dim) with CLS token
        # Data: master parquet via make_loaders(cfg)
        import torch.nn as nn
        from imu_lm.objectives import beit_1d as beit_obj
        from imu_lm.objectives.beit_1d import load_frozen_tokenizer

        assert encoder.use_cls_token, (
            "[beit_realdata_nopooling_8lvl] encoder requires use_cls_token: true"
        )

        _dev_beit_rd = next(encoder.parameters()).device
        beit_rd_cfg_ = cfg_get(cfg, ["objective", "beit_realdata_nopooling_8lvl"], {}) or {}

        tok_run_beit_rd_ = beit_rd_cfg_.get(
            "tokenizer_run",
            "runs/vit1d_rvq_tokenizer_realdata_nopooling_8lvl",
        )

        _frozen_tokenizer_rd = load_frozen_tokenizer(
            run_dir=tok_run_beit_rd_,
            device=_dev_beit_rd,
            cfg_key="rvq_tokenizer_realdata_nopooling_8lvl",
        )

        num_levels_beit_rd_ = int(beit_rd_cfg_.get("num_levels", 8))
        num_codes_beit_rd_  = int(beit_rd_cfg_.get("num_codes",  1024))
        beit_rd_heads = nn.ModuleList([
            nn.Linear(encoder.embed_dim, num_codes_beit_rd_)
            for _ in range(num_levels_beit_rd_)
        ]).to(_dev_beit_rd)

        _beit_rd_tokenizer = _frozen_tokenizer_rd
        _beit_rd_heads     = beit_rd_heads

        def objective_fn(batch, model, cfg):
            return beit_obj.forward_loss(
                batch, model, _beit_rd_heads, cfg, _beit_rd_tokenizer
            )

        all_params = list(encoder.parameters()) + list(beit_rd_heads.parameters())

        loaders = make_loaders(cfg)

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
            _load_encoder_state_dict(encoder, enc_state)

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
        _load_encoder_state_dict(encoder, enc_state)
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

    elif objective_type == "openwatch_gesture_probe":
        # Linear probe: frozen pretrained encoder + trainable LinearHead over
        # OpenWatch's per-CSV micro-gesture recordings (59 classes by default —
        # see label_map.json under the dataset root). Mirrors surfaceXR_gesture_probe
        # above; data comes from imu_lm.data.openwatch_loader (already used by the
        # combined MAE multidata pretrain to build this same source).
        from imu_lm.data.openwatch_loader import make_openwatch_loaders_from_cfg
        import logging as _logging
        _ow_log = _logging.getLogger(__name__)

        ow_probe_cfg = cfg_get(cfg, ["objective", "openwatch_gesture_probe"], {}) or {}
        num_classes = int(ow_probe_cfg.get("num_classes", 59))

        # Load & freeze pretrained encoder
        pretrained_run  = ow_probe_cfg.get("pretrained_run")
        pretrained_ckpt = ow_probe_cfg.get("pretrained_ckpt", "best")
        if not pretrained_run:
            raise ValueError(
                "[openwatch_gesture_probe] 'pretrained_run' must be set for probing"
            )
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt
        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[openwatch_gesture_probe] pretrained encoder not found: {enc_ckpt_path}"
            )
        if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _ow_log.info("[openwatch_gesture_probe] using alternating frame/global "
                         "attention encoder (VGGT-style)")
        _ow_log.info("[openwatch_gesture_probe] loading pretrained encoder from %s", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        _load_encoder_state_dict(encoder, enc_state)
        _ow_log.info("[openwatch_gesture_probe] encoder loaded — freezing all encoder parameters")
        encoder.requires_grad_(False)
        encoder.eval()

        # objective.openwatch_gesture_probe also doubles as the per-source config
        # block consumed by make_openwatch_loaders_from_cfg (root/train_subjects/
        # val_subjects/positions/gestures/label_ids/include_ppg/coordinate_frame).
        loaders = make_openwatch_loaders_from_cfg(cfg, objective_key="openwatch_gesture_probe")
        head = LinearHead(encoder.embed_dim, num_classes)
        _ce_w = _openwatch_ce_weight(loaders, ow_probe_cfg, num_classes)

        _ow_log.info(
            "[openwatch_gesture_probe] num_classes=%d pretrained_run=%s class_weighted_ce=%s",
            num_classes, pretrained_run, _ce_w is not None,
        )

        def objective_fn(batch, model, cfg):
            encoder.eval()
            x_b, y_b = batch
            with torch.no_grad():
                z = encoder(x_b.float())
            logits = head(z)
            _w = _ce_w.to(logits.device) if _ce_w is not None else None
            loss = torch.nn.functional.cross_entropy(logits, y_b.long(), weight=_w)
            preds = logits.argmax(dim=-1)
            acc = float((preds == y_b).float().mean())
            y_true = y_b.detach().cpu().tolist()
            y_pred = preds.detach().cpu().tolist()
            metrics = compute_metrics(y_true, y_pred)
            return loss, {
                "loss": float(loss.detach()), "acc": acc,
                "bal_acc": metrics["bal_acc"], "macro_f1": metrics["macro_f1"],
                "macro_precision": metrics["macro_precision"], "macro_recall": metrics["macro_recall"],
                "_y_true": y_true, "_y_pred": y_pred,
            }

        all_params = list(head.parameters())

    elif objective_type == "openwatch_gesture_finetune":
        # Finetune: warm-started encoder (stays trainable) + trainable LinearHead
        # over OpenWatch micro-gesture recordings. Sibling to openwatch_gesture_probe
        # (same loader/data config convention) but the encoder isn't frozen.
        from imu_lm.data.openwatch_loader import make_openwatch_loaders_from_cfg
        import logging as _logging
        _ow_log = _logging.getLogger(__name__)

        ow_ft_cfg = cfg_get(cfg, ["objective", "openwatch_gesture_finetune"], {}) or {}
        num_classes = int(ow_ft_cfg.get("num_classes", 59))

        pretrained_run  = ow_ft_cfg.get("pretrained_run")
        pretrained_ckpt = ow_ft_cfg.get("pretrained_ckpt", "best")
        if not pretrained_run:
            raise ValueError(
                "[openwatch_gesture_finetune] 'pretrained_run' must be set"
            )
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt
        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[openwatch_gesture_finetune] pretrained encoder not found: {enc_ckpt_path}"
            )
        if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _ow_log.info("[openwatch_gesture_finetune] using alternating frame/global "
                         "attention encoder (VGGT-style)")
        _ow_log.info("[openwatch_gesture_finetune] warm-starting encoder from %s "
                     "(encoder stays trainable)", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        _load_encoder_state_dict(encoder, enc_state)

        # objective.openwatch_gesture_finetune also doubles as the per-source config
        # block consumed by make_openwatch_loaders_from_cfg (root/train_subjects/
        # val_subjects/positions/gestures/label_ids/include_ppg/coordinate_frame).
        loaders = make_openwatch_loaders_from_cfg(cfg, objective_key="openwatch_gesture_finetune")
        head = LinearHead(encoder.embed_dim, num_classes)
        _ce_w = _openwatch_ce_weight(loaders, ow_ft_cfg, num_classes)

        _ow_log.info(
            "[openwatch_gesture_finetune] num_classes=%d pretrained_run=%s", num_classes, pretrained_run,
        )

        def objective_fn(batch, model, cfg):
            x_b, y_b = batch
            z = encoder(x_b.float())
            logits = head(z)
            _w = _ce_w.to(logits.device) if _ce_w is not None else None
            loss = torch.nn.functional.cross_entropy(logits, y_b.long(), weight=_w)
            preds = logits.argmax(dim=-1)
            acc = float((preds == y_b).float().mean())
            y_true = y_b.detach().cpu().tolist()
            y_pred = preds.detach().cpu().tolist()
            metrics = compute_metrics(y_true, y_pred)
            return loss, {
                "loss": float(loss.detach()), "acc": acc,
                "bal_acc": metrics["bal_acc"], "macro_f1": metrics["macro_f1"],
                "macro_precision": metrics["macro_precision"], "macro_recall": metrics["macro_recall"],
                "_y_true": y_true, "_y_pred": y_pred,
            }

        all_params = list(encoder.parameters()) + list(head.parameters())

    elif objective_type == "openwatch_gesture_supervised":
        # Fully supervised OpenWatch micro-gesture classification, trained FROM
        # SCRATCH: encoder + LinearHead trained end-to-end from random init (no
        # frozen/pretrained weights). Mirrors openwatch_gesture_probe's data
        # config (same objective_key convention) so the two are directly
        # comparable — this is the "no self-supervised pretraining" baseline.
        from imu_lm.data.openwatch_loader import make_openwatch_loaders_from_cfg
        import logging as _logging
        _ows_log = _logging.getLogger(__name__)

        if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _ows_log.info("[openwatch_gesture_supervised] using alternating frame/global "
                          "attention encoder (VGGT-style)")

        ows_cfg = cfg_get(cfg, ["objective", "openwatch_gesture_supervised"], {}) or {}
        num_classes = int(ows_cfg.get("num_classes", 59))

        loaders = make_openwatch_loaders_from_cfg(cfg, objective_key="openwatch_gesture_supervised")
        head = LinearHead(encoder.embed_dim, num_classes)
        _ce_w = _openwatch_ce_weight(loaders, ows_cfg, num_classes)

        _ows_log.info(
            "[openwatch_gesture_supervised] num_classes=%d (trained from scratch)", num_classes,
        )

        def objective_fn(batch, model, cfg):
            x_b, y_b = batch
            z = encoder(x_b.float())
            logits = head(z)
            _w = _ce_w.to(logits.device) if _ce_w is not None else None
            loss = torch.nn.functional.cross_entropy(logits, y_b.long(), weight=_w)
            preds = logits.argmax(dim=-1)
            acc = float((preds == y_b).float().mean())
            y_true = y_b.detach().cpu().tolist()
            y_pred = preds.detach().cpu().tolist()
            metrics = compute_metrics(y_true, y_pred)
            return loss, {
                "loss": float(loss.detach()), "acc": acc,
                "bal_acc": metrics["bal_acc"], "macro_f1": metrics["macro_f1"],
                "macro_precision": metrics["macro_precision"], "macro_recall": metrics["macro_recall"],
                "_y_true": y_true, "_y_pred": y_pred,
            }

        all_params = list(encoder.parameters()) + list(head.parameters())

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
            _load_encoder_state_dict(encoder, enc_state)

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
            _load_encoder_state_dict(encoder, enc_state)

        def objective_fn(batch, model, cfg):
            return supervised_obj.forward_loss(batch, encoder, head, cfg, raw_to_idx=None)

        all_params = list(encoder.parameters()) + list(head.parameters())

    elif objective_type == "har_probe":
        # Generic linear probe: frozen pretrained encoder + trainable LinearHead.
        # Works with any dataset in the master parquet via data.train_on_datasets.
        import logging as _logging
        _har_log = _logging.getLogger(__name__)

        har_cfg = cfg_get(cfg, ["objective", "har_probe"], {}) or {}
        pretrained_run  = har_cfg.get("pretrained_run")
        pretrained_ckpt = har_cfg.get("pretrained_ckpt", "best")
        if not pretrained_run:
            raise ValueError("[har_probe] 'pretrained_run' must be set")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt
        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(f"[har_probe] pretrained encoder not found: {enc_ckpt_path}")
        _har_log.info("[har_probe] loading pretrained encoder from %s", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        _load_encoder_state_dict(encoder, enc_state)
        _har_log.info("[har_probe] encoder loaded — freezing all encoder parameters")
        encoder.requires_grad_(False)
        encoder.eval()

        loaders    = make_loaders(cfg)
        label_map  = build_label_map(loaders["train_loader"], cfg)
        save_label_map(label_map, run_dir)
        raw_to_idx  = label_map["raw_to_idx"]
        num_classes = label_map["num_classes"]
        head = LinearHead(encoder.embed_dim, num_classes)

        # Inverse-frequency class weights — critical for imbalanced datasets
        counts  = torch.tensor(label_map["class_counts"], dtype=torch.float32)
        total   = counts.sum()
        _har_cw = total / (num_classes * counts.clamp(min=1))
        _har_cw = _har_cw / _har_cw.sum() * num_classes   # normalise so mean weight = 1

        imbalance_ratio = float(counts.max() / counts.clamp(min=1).min())
        _har_log.info(
            "[har_probe] num_classes=%d imbalance_ratio=%.1f pretrained_run=%s",
            num_classes, imbalance_ratio, pretrained_run,
        )

        def objective_fn(batch, model, cfg):
            encoder.eval()
            return supervised_obj.forward_loss(
                batch, encoder, head, cfg,
                raw_to_idx=raw_to_idx, class_weights=_har_cw,
            )

        all_params = list(head.parameters())

    elif objective_type == "utd_mhad_probe":
        # Linear probe: frozen pretrained encoder + trainable LinearHead.
        # Data comes from the master parquet (dataset == "utd_mhad").
        # Set train_on_datasets: ["utd_mhad"] in the model config to restrict to that dataset.
        import logging as _logging
        _utd_log = _logging.getLogger(__name__)

        utd_cfg = cfg_get(cfg, ["objective", "utd_mhad_probe"], {}) or {}

        # Load & freeze pretrained encoder
        pretrained_run  = utd_cfg.get("pretrained_run")
        pretrained_ckpt = utd_cfg.get("pretrained_ckpt", "best")
        if not pretrained_run:
            raise ValueError("[utd_mhad_probe] 'pretrained_run' must be set for probing")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt
        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[utd_mhad_probe] pretrained encoder not found: {enc_ckpt_path}"
            )
        if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _utd_log.info("[utd_mhad_probe] using alternating frame/global attention encoder (VGGT-style)")
        _utd_log.info("[utd_mhad_probe] loading pretrained encoder from %s", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        if (cfg_get(cfg, ['dual_stream'], None) or {}).get('enabled', False):
            from imu_lm.models.ViT1D.model import build_dual_stream
            encoder = build_dual_stream(cfg)      # two pretrained stems + gated fusion
        encoder = maybe_stats_wrap(encoder, cfg)   # stats_concat diagnostic (no-op unless enabled)
        _load_encoder_state_dict(encoder, enc_state)
        _utd_log.info("[utd_mhad_probe] encoder loaded — freezing all encoder parameters")
        encoder.requires_grad_(False)
        encoder.eval()

        loaders    = make_loaders(cfg)
        label_map  = build_label_map(loaders["train_loader"], cfg)
        save_label_map(label_map, run_dir)
        raw_to_idx = label_map["raw_to_idx"]
        num_classes = label_map["num_classes"]
        head = LinearHead(encoder.embed_dim, num_classes)

        # optional inverse-frequency class-weighted CE (Inertia-1 paper protocol; off by default)
        _utd_cw = None
        if bool(utd_cfg.get("class_weighted_ce", False)):
            _c = torch.tensor(label_map["class_counts"], dtype=torch.float32)
            _utd_cw = _c.sum() / (num_classes * _c.clamp(min=1))
            _utd_cw = _utd_cw / _utd_cw.sum() * num_classes
            _utd_log.info("[utd_mhad_probe] class_weighted_ce on: imbalance=%.1f", float(_c.max() / _c.clamp(min=1).min()))
        _utd_log.info(
            "[utd_mhad_probe] num_classes=%d pretrained_run=%s", num_classes, pretrained_run,
        )

        def objective_fn(batch, model, cfg):
            encoder.eval()
            return supervised_obj.forward_loss(batch, encoder, head, cfg, raw_to_idx=raw_to_idx, class_weights=_utd_cw)

        all_params = list(head.parameters())

    elif objective_type == "ocdetect_probe":
        # Binary linear probe: frozen pretrained encoder + trainable LinearHead(D, 2).
        # Labels: 0=background, 1=compulsive episode (window contains ≥1 labeled sample).
        # Data: OCDetect CSV files via make_ocdetect_loaders_random.
        # Class weights applied — heavily imbalanced (~5-10% positive windows).
        import logging as _logging
        from imu_lm.data.OCDetect_dataloader import (
            make_ocdetect_loaders_random,
            DATA_DIR_DEFAULT as _OCD_DATA_DIR,
        )
        from imu_lm.data.windowing import compute_T_and_hop as _ctah_ocd
        _ocd_log = _logging.getLogger(__name__)

        ocd_cfg = cfg_get(cfg, ["objective", "ocdetect_probe"], {}) or {}

        pretrained_run  = ocd_cfg.get("pretrained_run")
        pretrained_ckpt = ocd_cfg.get("pretrained_ckpt", "latest")
        if not pretrained_run:
            raise ValueError("[ocdetect_probe] 'pretrained_run' must be set")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt
        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[ocdetect_probe] pretrained encoder not found: {enc_ckpt_path}"
            )
        _ocd_log.info("[ocdetect_probe] loading pretrained encoder from %s", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        _load_encoder_state_dict(encoder, enc_state)
        _ocd_log.info("[ocdetect_probe] encoder loaded — freezing")
        encoder.requires_grad_(False)
        encoder.eval()

        _win_T_ocd, _win_hop_ocd = _ctah_ocd(cfg)
        batch_size_ocd_  = int(cfg_get(cfg, ["data", "batch_size"],      256))
        eval_bs_ocd_     = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size_ocd_))
        num_workers_ocd_ = int(cfg_get(cfg, ["data", "num_workers"],     0))
        pin_mem_ocd_     = bool(cfg_get(cfg, ["data", "pin_memory"],     False))
        splits_ocd_      = tuple(ocd_cfg.get("splits", [0.8, 0.1, 0.1]))
        apply_flu_ocd_   = bool(ocd_cfg.get("apply_flu_transform", False))

        loaders = make_ocdetect_loaders_random(
            data_dir=ocd_cfg.get("data_dir", _OCD_DATA_DIR),
            splits=splits_ocd_,
            batch_size=batch_size_ocd_,
            eval_batch_size=eval_bs_ocd_,
            num_workers=num_workers_ocd_,
            pin_memory=pin_mem_ocd_,
            apply_flu_transform=apply_flu_ocd_,
            multi_view=False,
            window_samples=_win_T_ocd,
            hop_samples=_win_hop_ocd,
            cfg=cfg,
        )

        num_classes_ocd = 2   # binary: background / compulsive
        head = LinearHead(encoder.embed_dim, num_classes_ocd)

        # Inverse-frequency class weights for heavy imbalance
        y_train = loaders["train_loader"].dataset._y.numpy()
        counts  = torch.tensor(
            [(y_train == c).sum() for c in range(num_classes_ocd)], dtype=torch.float32
        )
        total   = counts.sum()
        _ocd_cw = total / (num_classes_ocd * counts.clamp(min=1))
        _ocd_cw = _ocd_cw / _ocd_cw.sum() * num_classes_ocd
        _ocd_log.info(
            "[ocdetect_probe] train windows=%d  pos=%d (%.1f%%)  class_weights=%s",
            int(total), int(counts[1]), 100.0 * counts[1] / total,
            _ocd_cw.tolist(),
        )

        def objective_fn(batch, model, cfg):
            encoder.eval()
            return supervised_obj.forward_loss(
                batch, encoder, head, cfg,
                raw_to_idx=None, class_weights=_ocd_cw,
            )

        all_params = list(head.parameters())

    elif objective_type == "utd_mhad_finetune":
        # Finetune: warm-start encoder from pretrained MAE, train encoder + head end-to-end.
        # Data comes from the master parquet (dataset == "utd_mhad").
        import logging as _logging
        _utd_log = _logging.getLogger(__name__)

        utd_cfg = cfg_get(cfg, ["objective", "utd_mhad_finetune"], {}) or {}

        if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
            from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
            encoder = ViT1DEncoderAltAttn(cfg)
            _utd_log.info("[utd_mhad_finetune] using alternating frame/global "
                          "attention encoder (VGGT-style)")

        pretrained_run  = utd_cfg.get("pretrained_run")
        pretrained_ckpt = utd_cfg.get("pretrained_ckpt", "best")
        if not pretrained_run:
            raise ValueError("[utd_mhad_finetune] 'pretrained_run' must be set")
        runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
        if pretrained_ckpt in ("best", "latest"):
            enc_ckpt_path = os.path.join(runs_root, pretrained_run,
                                         "checkpoints", f"{pretrained_ckpt}.pt")
        else:
            enc_ckpt_path = pretrained_ckpt
        if not os.path.exists(enc_ckpt_path):
            raise FileNotFoundError(
                f"[utd_mhad_finetune] pretrained encoder not found: {enc_ckpt_path}"
            )
        _utd_log.info("[utd_mhad_finetune] warm-starting encoder from %s", enc_ckpt_path)
        enc_state = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
        if (cfg_get(cfg, ['dual_stream'], None) or {}).get('enabled', False):
            from imu_lm.models.ViT1D.model import build_dual_stream
            encoder = build_dual_stream(cfg)      # two pretrained stems + gated fusion
        encoder = maybe_stats_wrap(encoder, cfg)   # stats_concat diagnostic (no-op unless enabled)
        _load_encoder_state_dict(encoder, enc_state)

        loaders    = make_loaders(cfg)
        label_map  = build_label_map(loaders["train_loader"], cfg)
        save_label_map(label_map, run_dir)
        raw_to_idx = label_map["raw_to_idx"]
        num_classes = label_map["num_classes"]
        head = LinearHead(encoder.embed_dim, num_classes)

        _utd_cw = None
        if bool(utd_cfg.get("class_weighted_ce", False)):
            _c = torch.tensor(label_map["class_counts"], dtype=torch.float32)
            _utd_cw = _c.sum() / (num_classes * _c.clamp(min=1))
            _utd_cw = _utd_cw / _utd_cw.sum() * num_classes
            _utd_log.info("[utd_mhad_finetune] class_weighted_ce on: imbalance=%.1f", float(_c.max() / _c.clamp(min=1).min()))
        _utd_log.info(
            "[utd_mhad_finetune] num_classes=%d pretrained_run=%s", num_classes, pretrained_run,
        )

        def objective_fn(batch, model, cfg):
            return supervised_obj.forward_loss(batch, encoder, head, cfg, raw_to_idx=raw_to_idx, class_weights=_utd_cw)

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
                          "surfacexr_gesture_supervised", "surfacexr_gesture_probe",
                          "har_probe", "utd_mhad_probe", "utd_mhad_finetune",
                          "ocdetect_probe",
                          "openwatch_gesture_probe", "openwatch_gesture_supervised",
                          "openwatch_gesture_finetune"):
        extra_modules = {"head": head}
    elif objective_type in ("pose", "pose_probe"):
        extra_modules = {"pose_decoder": pose_decoder}
    elif objective_type in ("odom_nymeria", "odom_nymeria_real_wrist_flu",
                            "odom_nymeria_real_wrist_worldframe_flu",
                            "odom_probe_nymeria_real_wrist_flu",
                            "odom_finetune_nymeria_real_wrist_flu",
                            "odom_finetune_nymeria_real_wrist_worldframe_flu",
                            "odom_probe", "odom_finetune", "odom_imuposer_flu",
                            "odom_probe_imuposer_flu", "odom_finetune_imuposer_flu",
                            "odom_totalcapture_flu", "odom_probe_totalcapture_flu",
                            "odom_finetune_totalcapture_flu", "local_disp_imuposer_flu",
                            "local_vel_imuposer_flu",
                            "local_vel_probe_imuposer_flu", "local_vel_finetune_imuposer_flu",
                            "odom_nymeria_syn_wrist_flu",
                            "odom_probe_nymeria_syn_wrist_flu",
                            "odom_finetune_nymeria_syn_wrist_flu"):
        extra_modules = {"odom_decoder": odom_decoder}
    elif objective_type in ("imu_text_clip", "imu_text_clip_flu", "imu_text_clip_nymeria_flu", "imu_text_clip_realdata"):
        # Save projection head + logit_scale; text encoder can be reloaded from HuggingFace
        # UNLESS it was trained (freeze_text_encoder: false) -- in that case its
        # fine-tuned weights must be checkpointed too, or they're silently lost
        # and any eval script re-fetching the plain pretrained HF weights would
        # discard everything the joint text-tower training did.
        extra_modules = {"clip_head": clip_head}
        if objective_type in ("imu_text_clip", "imu_text_clip_flu", "imu_text_clip_nymeria_flu") \
                and not freeze_text and text_encoder is not None:
            extra_modules["text_enc"] = text_encoder
        if objective_type in ("imu_text_clip_flu", "imu_text_clip_nymeria_flu") and text_mlp is not None:
            extra_modules["text_mlp"] = text_mlp
        if objective_type == "imu_text_clip_realdata":
            if text_enc is not None:
                extra_modules["text_enc"] = text_enc
            if text_mlp_rdc is not None:
                extra_modules["text_mlp"] = text_mlp_rdc
    elif objective_type == "jepa_ocdetect_flu":
        extra_modules = {
            "predictor":        predictor,
            "target_encoder":   target_encoder_ema,
        }
    elif objective_type == "dino_ocdetect_anyrot":
        extra_modules = {
            "dino_head":        dino_head,
            "teacher_head":     teacher_dino_head,
            "teacher_encoder":  teacher_encoder_ema,
            "dino_loss":        dino_loss_module,
        }
    elif objective_type == "mae_dino_ocdetect_anyrot":
        extra_modules = {
            "dino_head":        dino_head,
            "teacher_head":     teacher_dino_head,
            "teacher_encoder":  teacher_encoder_ema,
            "recon_head":       recon_head,
            "dino_loss":        dino_cls_loss_module,
        }
    elif objective_type in ("rvq_tokenizer_ocdetect_anyrot",
                            "rvq_tokenizer_nymeria_real_wrist_flu",
                            "rvq_tokenizer_realdata_nopooling_8lvl",
                            "rvq_tokenizer_realdata_anyrot_nopooling_8lvl"):
        extra_modules = {"rvq": rvq_module, "decoders": rvq_decoders}
        if rvq_proj is not None:
            extra_modules["rvq_proj"] = rvq_proj
    elif objective_type == "beit_ocdetect_anyrot":
        extra_modules = {"beit_heads": beit_heads}
    elif objective_type == "beit_ocdetect_anyrot_nopooling_8lvl":
        extra_modules = {"beit_heads": beit8_heads}
    elif objective_type == "beit_realdata_nopooling_8lvl":
        extra_modules = {"beit_heads": beit_rd_heads}
    else:
        extra_modules = {"decoder": decoder}

    # Load checkpoint if resuming
    resume_path = resolve_resume_path(run_dir, resume_ckpt)
    start_step = load_checkpoint(resume_path, encoder, optimizer, scheduler, extra_modules=extra_modules)

    # Sync EMA step counter with resume step so momentum annealing continues correctly
    if objective_type == "dino_ocdetect_anyrot":
        _dino_state["step"] = start_step
    elif objective_type == "mae_dino_ocdetect_anyrot":
        _md_state["step"] = start_step
    
    trainer = Trainer(cfg, run_dir)
    trainer.fit(encoder, objective_fn, train_loader, val_loader, optimizer, scheduler, start_step=start_step, extra_modules=extra_modules)


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_pretrain.py to launch training.")
