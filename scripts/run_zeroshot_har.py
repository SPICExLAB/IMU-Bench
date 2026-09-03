#!/usr/bin/env python3
"""Zero-shot HAR evaluation: CLIP IMU encoder ranked against text class embeddings.

Loads a pretrained IMU-CLIP checkpoint and evaluates zero-shot on any dataset
in the parquet (default: wisdm).  For each IMU window the predicted class is:

    pred = argmax( clip_head(encoder(x))  @  CLIP_text(class_sentences).T )

No labelled data from the target dataset is used — only the activity name strings.

Usage
-----
    python scripts/run_zeroshot_har.py \\
        --run   runs/vit1d_clip_humanml3d_flu_finetune_10s \\
        --config configs/base.yaml \\
        --dataset wisdm

    # Multiple datasets at once:
    python scripts/run_zeroshot_har.py \\
        --run runs/vit1d_clip_humanml3d_flu_finetune_10s \\
        --config configs/base.yaml \\
        --dataset pamap2,wisdm,shoaib
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.dataset as _pa_ds
from scipy.stats import mode as _scipy_mode

from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.objectives.contrastive_clip_1d import ContrastiveCLIPHead, CLIPTextEncoder, TextProjectionMLP
from imu_lm.data.realdata_clip_loader import (
    _build_label_map,
    RealdataCLIPDataset,
    _collate_clip,
    _resample_roundtrip,
)
from imu_lm.data.splits import build_session_index, make_splits
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.data.augmentations.preprocess import preprocess_window, PreprocessStats
from imu_lm.utils.helpers import load_yaml, deep_update, cfg_get


# ────────────────────────────────────────────────────────────────────────────
# Full-session dataset (no windowing — whole sequence wrap-padded to T_model)
# ────────────────────────────────────────────────────────────────────────────

class FullSessionCLIPDataset(torch.utils.data.Dataset):
    """One item per session: load the entire signal, preprocess, wrap-pad to T_model.

    Compared to windowed loading this:
    - uses ALL available signal from each short sequence (no truncation)
    - wrap-pads from the true sequence length, not from an arbitrary short window
    - produces exactly one (x, text) pair per action trial
    """

    def __init__(self, parquet_path, session_keys, cfg, id_to_text,
                 T_model, sentence_template="The person is {}",
                 roundtrip_low_hz=None, native_hz=50.0):
        self._T_model    = T_model
        self._id_to_text = id_to_text
        self._template   = sentence_template
        self._stats      = PreprocessStats()
        self._roundtrip_low_hz = roundtrip_low_hz
        self._native_hz  = native_hz

        sensor_cols  = cfg_get(cfg, ["data", "sensor_columns"], [])
        label_col    = cfg_get(cfg, ["data", "label_column"],   "dataset_activity_id")
        dataset_col  = cfg_get(cfg, ["data", "dataset_column"], "dataset")
        subject_col  = cfg_get(cfg, ["data", "subject_column"], "subject_id")
        session_col  = cfg_get(cfg, ["data", "session_column"], "session_id")
        load_cols    = sensor_cols + [label_col, dataset_col, subject_col, session_col]

        pa_dataset = _pa_ds.dataset(parquet_path, format="parquet")

        self._items: list = []
        for key in session_keys:
            filt = (
                (_pa_ds.field(dataset_col) == key.dataset)
                & (_pa_ds.field(subject_col) == key.subject_id)
                & (_pa_ds.field(session_col) == key.session_id)
            )
            df = pa_dataset.to_table(columns=load_cols, filter=filt).to_pandas()
            if df.empty:
                continue
            X = df[sensor_cols].to_numpy(dtype=np.float32)   # [T_sess, C]
            y = df[label_col].to_numpy()
            # Majority label across the whole session
            label = int(_scipy_mode(y, keepdims=False).mode)
            text  = id_to_text.get(label, sentence_template.format(str(label)))
            self._items.append((X, text, cfg))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        X, text, cfg = self._items[idx]

        if self._roundtrip_low_hz is not None:
            X = _resample_roundtrip(X, self._native_hz, self._roundtrip_low_hz)

        # Preprocess (filter + normalise): input [T, C], output [C, T] or None
        Xp = preprocess_window(X, cfg, self._stats)
        if Xp is None:
            Xp = np.transpose(X, (1, 0))   # fallback: raw data, still [C, T]

        x = torch.from_numpy(Xp).float()   # [C, T_sess]

        # Wrap-pad to T_model: tile along time then truncate
        T = x.shape[-1]
        if T < self._T_model:
            n = -(-self._T_model // T)          # ceil(T_model / T)
            x = x.repeat(1, n)[:, :self._T_model]

        return x, text


# ────────────────────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────────────────────

def _load_model(run_dir: str, ckpt_name: str, device: torch.device):
    """Return (encoder, clip_head, text_mlp, ckpt_cfg, text_enc_state) from a
    CLIP run checkpoint.

    text_mlp is None when the checkpoint was trained without a text projection MLP.
    text_enc_state is None unless the checkpoint was trained with
    freeze_text_encoder: false (see imu_lm/models/ViT1D/run.py's extra_modules
    logic for imu_text_clip_flu) -- in that case it's the fine-tuned CLIP text
    tower's state_dict, and the caller must load it into a CLIPTextEncoder
    before encoding class labels, or the whole point of training the text
    tower jointly is silently lost (falling back to vanilla pretrained CLIP
    text weights that never saw this checkpoint's captions).
    """
    ckpt_path = os.path.join(run_dir, "checkpoints", f"{ckpt_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]

    # Encoder -- must match the architecture the checkpoint was actually
    # trained with. Every alt_attention:true checkpoint in this benchmark
    # (VGGT-style alternating frame/global attention) uses a DIFFERENT class
    # (ViT1DEncoderAltAttn) than the plain ViT1DEncoder -- loading with the
    # wrong class silently drops 100% of the encoder weights (all keys land
    # in missing/unexpected) rather than raising, so this must be checked
    # explicitly, matching imu_lm.models.ViT1D.run.py's own selection logic.
    if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
        from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
        encoder = ViT1DEncoderAltAttn(cfg)
    else:
        encoder = ViT1DEncoder(cfg)
    missing, unexpected = encoder.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        logging.warning(
            "Encoder partial load — missing: %s  unexpected: %s", missing, unexpected
        )
    encoder.eval().to(device)

    # CLIP projection head
    clip_head_state = (
        ckpt.get("clip_head")
        or ckpt.get("extra_modules", {}).get("clip_head")
    )
    if clip_head_state is None:
        raise KeyError(
            f"clip_head not found in checkpoint. Top-level keys: {list(ckpt.keys())}"
        )
    proj_w    = clip_head_state["projection.weight"]   # [D_clip, D_enc]
    clip_head = ContrastiveCLIPHead(
        imu_embed_dim  = proj_w.shape[1],
        clip_embed_dim = proj_w.shape[0],
    )
    clip_head.load_state_dict(clip_head_state)
    clip_head.eval().to(device)

    # Optional text-side MLP (present when trained with text_projection_mlp: true)
    text_mlp = None
    text_mlp_state = (
        ckpt.get("text_mlp")
        or ckpt.get("extra_modules", {}).get("text_mlp")
    )
    if text_mlp_state is not None:
        embed_dim  = proj_w.shape[0]
        hidden_dim = text_mlp_state["net.1.weight"].shape[0]
        text_mlp   = TextProjectionMLP(embed_dim=embed_dim, hidden_dim=hidden_dim)
        text_mlp.load_state_dict(text_mlp_state)
        text_mlp.eval().to(device)
        logging.info("Loaded TextProjectionMLP (hidden_dim=%d)", hidden_dim)

    # Fine-tuned text encoder (present only when trained with
    # freeze_text_encoder: false -- see docstring above)
    text_enc_state = (
        ckpt.get("text_enc")
        or ckpt.get("extra_modules", {}).get("text_enc")
    )

    return encoder, clip_head, text_mlp, cfg, text_enc_state


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    log = logging.getLogger(__name__)

    ap = argparse.ArgumentParser(
        description="Zero-shot HAR evaluation with a pretrained CLIP IMU model"
    )
    ap.add_argument("--run",     required=True,
                    help="Run directory or name under runs/ (e.g. runs/vit1d_clip_...)")
    ap.add_argument("--config",  default="configs/base.yaml",
                    help="Base config YAML (supplies dataset_path and paths)")
    ap.add_argument("--dataset", default="wisdm",
                    help="Comma-separated dataset name(s) to evaluate on (e.g. wisdm,pamap2)")
    ap.add_argument("--ckpt",    default="best",
                    help="Checkpoint to load: 'best' or 'latest'")
    ap.add_argument("--template", default="The person is {}",
                    help="Sentence template used to build class label strings")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32",
                    help="HuggingFace CLIP text model name")
    ap.add_argument("--local-files-only", action="store_true",
                    help="Do not fetch CLIP weights from the HuggingFace hub")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5],
                    help="Report top-k accuracies")
    ap.add_argument("--window-seconds", type=float, default=None,
                    help="Override the checkpoint's window_seconds (useful for short-sequence "
                         "datasets like utd_mhad whose actions are shorter than the training "
                         "window, e.g. --window-seconds 2.56)")
    ap.add_argument("--full-session", action="store_true",
                    help="Load each session as one item (no windowing). The full signal is "
                         "wrap-padded to T_model. Best for short-sequence datasets like "
                         "utd_mhad where each trial is one continuous action.")
    ap.add_argument("--val-only", action="store_true",
                    help="Evaluate only on the held-out val split, reusing the SPLIT CONFIG "
                         "SAVED IN THE CHECKPOINT (splits.seed/group_key/shuffle/val_ratio as "
                         "the finetune itself ran with), instead of the full dataset. Required "
                         "for a leakage-free classification-accuracy number on a checkpoint that "
                         "was FINETUNED on this dataset (the default train_keys-only, "
                         "val_ratio=0 behavior is only valid for a checkpoint that never saw "
                         "this dataset at all, i.e. true zero-shot). Do NOT hardcode split "
                         "params here: they must match the finetune exactly or the 'held-out' "
                         "set silently contains training subjects (see the 2026-07-28 fix note "
                         "at the --val-only handling site).")
    ap.add_argument("--max-gaps-per-session", type=int, default=None,
                    help="Override windowing.max_gaps_per_session (PAMAP2 needs 700 -- the "
                         "repo-wide default of 100 rejects every PAMAP2 val session due to "
                         "real-world sensor dropout, see EXPERIMENTS.md §10).")
    ap.add_argument("--roundtrip-20hz", action="store_true",
                    help="Downsample every target-dataset window to 20 Hz then upsample it "
                         "back to --native-hz before feeding it to the model. Matches the "
                         "effective bandwidth cap HumanML3D-derived training data has (its "
                         "real content is capped by AMASS mocap's native 20 Hz frame rate "
                         "even though the working pipeline upsamples it to native_hz before "
                         "deriving acc/gyro) -- without this, zero-shot eval feeds the model "
                         "genuinely higher-bandwidth real sensor data than anything it was "
                         "ever trained on.")
    ap.add_argument("--native-hz", type=float, default=50.0,
                    help="Sample rate the target dataset's windows are stored at in the "
                         "parquet (default 50.0 -- this benchmark's canonical rate for every "
                         "real dataset). Only used by --roundtrip-20hz.")
    ap.add_argument("--zscore-input", action="store_true",
                    help="Force per-window z-score normalization (preprocess.normalize: "
                         "enabled=true, method=zscore) on the target dataset's windows before "
                         "encoding, overriding whatever the checkpoint's own saved config says. "
                         "Debugging aid (2026-07-25): HumanML3DIMUVarLenDatasetFLU -- the loader "
                         "Stage-2 HumanML3D-CLIP training actually uses -- unconditionally "
                         "per-window z-scores every training sample (hardcoded, not gated by "
                         "any config flag -- see imu_lm/data/humanml3d_dataset_FLU.py __iter__). "
                         "But this eval script's own real-dataset loading path goes through the "
                         "generic WindowDataset/preprocess_window pipeline, which IS gated by "
                         "preprocess.normalize.enabled -- and every HumanML3D-CLIP checkpoint's "
                         "own saved config has no preprocess block at all (defaults to "
                         "disabled), so real target-dataset windows were being fed to the "
                         "encoder RAW while HumanML3D training windows were always z-scored -- a "
                         "genuine train/eval distribution mismatch. This flag tests the fix "
                         "without retraining: it z-scores the target dataset the same way "
                         "HumanML3D windows always were.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Resolve run directory
    run_dir = args.run if os.path.isdir(args.run) else os.path.join("runs", args.run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run directory not found: {args.run}")

    # ── 1. Load model ───────────────────────────────────────────────────────
    log.info("Loading checkpoint '%s' from %s ...", args.ckpt, run_dir)
    encoder, clip_head, text_mlp, ckpt_cfg, text_enc_state = _load_model(run_dir, args.ckpt, device)
    D_enc  = encoder.embed_dim
    D_clip = clip_head.projection.weight.shape[0]
    log.info("Encoder embed_dim=%d  →  CLIP dim=%d  text_mlp=%s  fine-tuned text_enc=%s",
             D_enc, D_clip, text_mlp is not None, text_enc_state is not None)

    # Merge: base config supplies paths; checkpoint config supplies architecture
    # and windowing (window_seconds, patch_size, preprocessing).
    base_cfg = load_yaml(args.config)
    cfg = deep_update(ckpt_cfg, {"paths": base_cfg.get("paths", {})})
    if args.zscore_input:
        cfg = deep_update(cfg, {"preprocess": {"normalize": {"enabled": True, "method": "zscore"}}})
        log.info("[--zscore-input] forcing preprocess.normalize.enabled=true (zscore) on target dataset windows")
    if args.roundtrip_20hz:
        log.info("[--roundtrip-20hz] downsampling target dataset windows to 20Hz then "
                 "upsampling back to %.1fHz before encoding", args.native_hz)

    # T_model: the sample length the encoder was trained with.
    # Windows shorter than this are zero-padded on the right so the model
    # always receives its expected input shape.
    #
    # Bug fix (2026-07-26): compute_T_and_hop reads the SINGULAR
    # windowing.window_seconds key, defaulting to 2.56s if absent -- but
    # every HumanML3D-CLIP Stage-2 config in this benchmark instead sets
    # windowing.min_window_seconds/max_window_seconds (a different key pair
    # used by make_humanml3d_imu_loaders_flu for variable-length training),
    # so T_model was SILENTLY falling back to the 2.56s default for every
    # checkpoint this whole investigation, regardless of the checkpoint's
    # actual training window. This was harmless for standard windowing
    # (--window-seconds only controls extraction length there, and T_model
    # is only used as a "pad up if shorter" floor that never triggers when
    # the extracted window is already >= T_model) but broke --full-session
    # mode outright, which wrap-pads directly to T_model with no other
    # override path -- surfaced as a batch-collation crash on UTD-MHAD
    # (sessions ~3.3-3.5s long, each ending up at its own native length
    # instead of a consistent padded T_model=128, since 128 < every
    # session's real length so the "pad if shorter" branch never fired).
    # Now: when --window-seconds is explicitly passed, it also overrides
    # T_model directly, not just the extraction length, so the CLI flag's
    # own documented contract ("Override the checkpoint's window_seconds")
    # actually holds for every mode including --full-session.
    if args.window_seconds is not None:
        T_model, _ = compute_T_and_hop(deep_update(ckpt_cfg, {"windowing": {"window_seconds": args.window_seconds}}))
    else:
        T_model, _ = compute_T_and_hop(ckpt_cfg)
    log.info("Model trained with T=%d samples (%.2fs)",
             T_model, ckpt_cfg.get("windowing", {}).get("window_seconds", "?"))

    parquet_path   = cfg["paths"]["dataset_path"]
    dataset_filter = [d.strip() for d in args.dataset.split(",")]
    log.info("Parquet : %s", parquet_path)
    log.info("Datasets: %s", dataset_filter)

    # ── 2. Build class text embeddings ──────────────────────────────────────
    label_col      = cfg.get("data", {}).get("label_column",      "dataset_activity_id")
    label_name_col = cfg.get("data", {}).get("label_name_column", "dataset_activity_label")
    dataset_col    = cfg.get("data", {}).get("dataset_column",    "dataset")

    id_to_text = _build_label_map(
        parquet_path, label_col, label_name_col,
        args.template, dataset_filter, dataset_col,
    )
    if not id_to_text:
        raise SystemExit(f"No activity labels found for dataset(s): {dataset_filter}")

    class_ids   = sorted(id_to_text)
    class_texts = [id_to_text[cid] for cid in class_ids]

    log.info("%d activity classes:", len(class_ids))
    for cid, txt in zip(class_ids, class_texts):
        log.info("  [%3d] %s", cid, txt)

    log.info("Encoding class labels with %s ...", args.clip_model)
    # text_projection convention MUST follow the checkpoint's own training
    # config (2026-07-28): checkpoints trained with objective.<type>.
    # text_projection: true aligned their clip_head to CLIP's projected text
    # space (official CLIP / UniMTS convention); older checkpoints aligned
    # to the bare pooler_output space. Encoding class labels in the wrong
    # space silently breaks the eval (measured cosine between the two spaces
    # is only 0.01-0.04).
    _obj_cfg = ckpt_cfg.get("objective", {}) or {}
    _use_proj = bool(
        (_obj_cfg.get(_obj_cfg.get("type", ""), {}) or {}).get("text_projection", False)
    )
    if _use_proj:
        log.info("Checkpoint was trained with text_projection: true -- encoding "
                 "class labels in CLIP's projected text space")
    text_enc = CLIPTextEncoder(
        model_name       = args.clip_model,
        freeze           = True,
        local_files_only = args.local_files_only,
        use_projection   = _use_proj,
    )
    if text_enc_state is not None:
        text_enc.load_state_dict(text_enc_state)
        log.info("Loaded FINE-TUNED text encoder weights from checkpoint "
                 "(this checkpoint was trained with freeze_text_encoder: false)")
    text_enc.eval().to(device)
    with torch.no_grad():
        z_classes = text_enc(class_texts, device=device)   # [N_cls, D_clip]
        if text_mlp is not None:
            z_classes = text_mlp(z_classes)                # apply learned text projection
    del text_enc   # free ~250 MB
    log.info("Class embedding shape: %s", tuple(z_classes.shape))

    # ── 3. Build evaluation data loader ─────────────────────────────────────
    # Set val/test ratios to 0 so every session ends up in train_keys —
    # we are doing inference, not training, so we want the full dataset.
    windowing_override: dict = {}
    if args.window_seconds is not None:
        trained_ws = cfg.get("windowing", {}).get("window_seconds", "?")
        log.info(
            "Overriding window_seconds: %.2f  (checkpoint trained with %.2f) — "
            "needed for datasets with short sequences",
            args.window_seconds, trained_ws,
        )
        windowing_override = {"window_seconds": args.window_seconds}

    if args.max_gaps_per_session is not None:
        windowing_override["max_gaps_per_session"] = args.max_gaps_per_session

    if args.val_only:
        # Leakage fix (2026-07-28): this used to hardcode seed=42/group_key=
        # subject_id/val_ratio=0.2, claiming to match "the convention this
        # benchmark's own HAR finetune configs use". It did not. Those configs
        # set ONLY splits.val_ratio; splits.seed comes from configs/base.yaml,
        # where it is 0 (trainer.seed=42 is an unrelated key -- splits.py:206
        # reads splits.seed). make_splits shuffles subject groups with
        # RandomState(seed), so seed=42 at eval time produced a DIFFERENT val
        # subject set than seed=0 at finetune time: measured overlap was
        # UTD-MHAD 0/2 (i.e. both "held-out" eval subjects were finetuning
        # TRAIN subjects), PAMAP2 1/2, WISDM 3/7, RecoFit 2/19 -- silently
        # inflating every --val-only accuracy on a finetuned checkpoint.
        #
        # `cfg` is derived from ckpt_cfg (the config saved inside the
        # checkpoint at training time), so the checkpoint's own splits block
        # IS the split the finetune actually held out. Inherit it verbatim
        # rather than re-deriving it, which makes the two agree by
        # construction instead of by convention.
        splits_override = {}
        _inherited = cfg_get(cfg, ["splits"], {}) or {}
        if not _inherited:
            raise SystemExit(
                "[--val-only] checkpoint config has no 'splits' block, so the finetune's "
                "held-out val split cannot be reproduced. Refusing to guess: a wrong split "
                "silently reports training subjects as held-out."
            )
        log.info(
            "[--val-only] inheriting the checkpoint's own split: seed=%s group_key=%s "
            "shuffle=%s val_ratio=%s test_ratio=%s (reproduces the finetune's held-out set)",
            _inherited.get("seed", 0), _inherited.get("group_key"),
            _inherited.get("shuffle"), _inherited.get("val_ratio"),
            _inherited.get("test_ratio"),
        )
        if not float(_inherited.get("val_ratio", 0.0)) > 0:
            raise SystemExit(
                f"[--val-only] checkpoint's splits.val_ratio is {_inherited.get('val_ratio')}, "
                "so it has no val split to evaluate on. This checkpoint was not finetuned "
                "with a held-out val split; --val-only is meaningless for it."
            )
    else:
        # Set val/test ratios to 0 so every session ends up in train_keys —
        # appropriate only for a checkpoint that never trained on this
        # dataset (true zero-shot); see --val-only for the finetuned case.
        splits_override = {"val_ratio": 0.0, "test_ratio": 0.0}

    eval_cfg = deep_update(cfg, {
        "data": {
            "batch_size":      args.batch_size,
            "eval_batch_size": args.batch_size,
            "num_workers":     args.num_workers,
            "pin_memory":      True,
        },
        "windowing": windowing_override,
        "splits": splits_override,
    })

    log.info("Building session index ...")
    session_index = build_session_index(
        parquet_path, eval_cfg, dataset_filter=dataset_filter
    )
    splits    = make_splits(session_index, eval_cfg)
    eval_keys = splits.get("val_keys", []) if args.val_only else splits.get("train_keys", [])
    log.info("Sessions: %d (%s)", len(eval_keys), "val-only" if args.val_only else "full dataset")

    if args.full_session:
        log.info("Mode: full-session (one item per trial, wrap-padded to T=%d)", T_model)
        dataset = FullSessionCLIPDataset(
            parquet_path      = parquet_path,
            session_keys      = eval_keys,
            cfg               = eval_cfg,
            id_to_text        = id_to_text,
            T_model           = T_model,
            sentence_template = args.template,
            roundtrip_low_hz  = 20.0 if args.roundtrip_20hz else None,
            native_hz         = args.native_hz,
        )
        # FullSessionCLIPDataset already pads to T_model, no wrap needed in loop
        collate_fn = _collate_clip
    else:
        dataset = RealdataCLIPDataset(
            parquet_path      = parquet_path,
            session_index     = session_index,
            session_keys      = eval_keys,
            cfg               = eval_cfg,
            split_name        = "val",       # disables augmentation
            id_to_text        = id_to_text,
            sentence_template = args.template,
            roundtrip_low_hz  = 20.0 if args.roundtrip_20hz else None,
            native_hz         = args.native_hz,
        )
        collate_fn = _collate_clip

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 0 if args.full_session else args.num_workers,
        pin_memory  = True,
        drop_last   = False,
        collate_fn  = collate_fn,
    )
    log.info("Evaluation items: %d", len(dataset))

    # ── 4. Zero-shot inference ──────────────────────────────────────────────
    log.info("Running inference ...")
    all_logits: list[torch.Tensor] = []
    all_gts:    list[int]          = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            x, texts = batch
            x = x.to(device)                            # [B, C, T]
            if x.shape[-1] < T_model:
                # Wrap-pad: tile the signal until it fills T_model
                # e.g. [a,b,c] → [a,b,c,a,b,c,a,b,...] truncated to T_model
                n = -(-T_model // x.shape[-1])          # ceil division
                x = x.repeat(1, 1, n)[..., :T_model]   # [B, C, T_model]
            z_imu  = clip_head(encoder(x))              # [B, D_clip] L2-normed
            logits = z_imu @ z_classes.T                # [B, N_cls]

            all_logits.append(logits.cpu())
            for txt in texts:
                gt = next((j for j, ct in enumerate(class_texts) if ct == txt), -1)
                all_gts.append(gt)

            if (i + 1) % 50 == 0:
                log.info("  %d / %d batches ...", i + 1, len(loader))

    all_logits = torch.cat(all_logits, dim=0)   # [N, N_cls]
    all_gts    = np.array(all_gts)

    valid      = all_gts >= 0
    all_logits = all_logits[valid]
    all_gts    = all_gts[valid]
    N          = len(all_gts)
    log.info("Valid windows: %d  (skipped: %d)", N, (~valid).sum())

    # ── 5. Report metrics ───────────────────────────────────────────────────
    preds = all_logits.argmax(dim=1).numpy()

    print()
    print("=" * 68)
    print(f"  ZERO-SHOT HAR  |  model: {os.path.basename(run_dir)}")
    print(f"  dataset: {args.dataset}  |  windows: {N}  |  classes: {len(class_ids)}")
    print("=" * 68)

    for k in sorted(set(args.topk)):
        if k > len(class_ids):
            continue
        topk_preds = all_logits.topk(k, dim=1).indices.numpy()   # [N, k]
        hits = (topk_preds == all_gts[:, None]).any(axis=1)
        print(f"  Top-{k:2d} accuracy : {hits.mean() * 100:6.2f}%")

    # Standard classification accuracy/macro-F1, directly comparable to
    # UniMTS's own "acc: X, prec: Y, rec: Z, f1: W" convention
    # (evaluate_custom.py/finetune_custom.py) -- same metric type, computed
    # the same way (sklearn, pooled over the whole eval pass).
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc  = accuracy_score(all_gts, preds)
    prec = precision_score(all_gts, preds, average="macro", zero_division=0)
    rec  = recall_score(all_gts, preds, average="macro", zero_division=0)
    f1   = f1_score(all_gts, preds, average="macro", zero_division=0)
    print(f"  acc: {acc}, prec: {prec}, rec: {rec}, f1: {f1}")

    print()
    print(f"  Per-class top-1 accuracy:")
    for j, (cid, txt) in enumerate(zip(class_ids, class_texts)):
        mask = all_gts == j
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == j).mean() * 100
        print(f"    {txt:<52s}  {acc:5.1f}%  (n={int(mask.sum()):5d})")

    try:
        from sklearn.metrics import classification_report
        gt_names   = [class_texts[g] for g in all_gts]
        pred_names = [class_texts[p] for p in preds]
        print()
        print("  Classification report:")
        report = classification_report(gt_names, pred_names, zero_division=0)
        for line in report.splitlines():
            print("    " + line)
    except ImportError:
        log.warning("sklearn not available — skipping classification report")

    print("=" * 68)


if __name__ == "__main__":
    main()
