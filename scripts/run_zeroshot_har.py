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
                 T_model, sentence_template="The person is {}"):
        self._T_model    = T_model
        self._id_to_text = id_to_text
        self._template   = sentence_template
        self._stats      = PreprocessStats()

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
    """Return (encoder, clip_head, text_mlp, ckpt_cfg) from a CLIP run checkpoint.

    text_mlp is None when the checkpoint was trained without a text projection MLP.
    """
    ckpt_path = os.path.join(run_dir, "checkpoints", f"{ckpt_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]

    # Encoder
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

    return encoder, clip_head, text_mlp, cfg


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
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Resolve run directory
    run_dir = args.run if os.path.isdir(args.run) else os.path.join("runs", args.run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run directory not found: {args.run}")

    # ── 1. Load model ───────────────────────────────────────────────────────
    log.info("Loading checkpoint '%s' from %s ...", args.ckpt, run_dir)
    encoder, clip_head, text_mlp, ckpt_cfg = _load_model(run_dir, args.ckpt, device)
    D_enc  = encoder.embed_dim
    D_clip = clip_head.projection.weight.shape[0]
    log.info("Encoder embed_dim=%d  →  CLIP dim=%d  text_mlp=%s",
             D_enc, D_clip, text_mlp is not None)

    # Merge: base config supplies paths; checkpoint config supplies architecture
    # and windowing (window_seconds, patch_size, preprocessing).
    base_cfg = load_yaml(args.config)
    cfg = deep_update(ckpt_cfg, {"paths": base_cfg.get("paths", {})})

    # T_model: the sample length the encoder was trained with.
    # Windows shorter than this are zero-padded on the right so the model
    # always receives its expected input shape.
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
    text_enc = CLIPTextEncoder(
        model_name       = args.clip_model,
        freeze           = True,
        local_files_only = args.local_files_only,
    )
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

    eval_cfg = deep_update(cfg, {
        "data": {
            "batch_size":      args.batch_size,
            "eval_batch_size": args.batch_size,
            "num_workers":     args.num_workers,
            "pin_memory":      True,
        },
        "windowing": windowing_override,
        "splits": {
            "val_ratio":  0.0,
            "test_ratio": 0.0,
        },
    })

    log.info("Building session index ...")
    session_index = build_session_index(
        parquet_path, eval_cfg, dataset_filter=dataset_filter
    )
    splits    = make_splits(session_index, eval_cfg)
    eval_keys = splits.get("train_keys", [])
    log.info("Sessions: %d", len(eval_keys))

    if args.full_session:
        log.info("Mode: full-session (one item per trial, wrap-padded to T=%d)", T_model)
        dataset = FullSessionCLIPDataset(
            parquet_path      = parquet_path,
            session_keys      = eval_keys,
            cfg               = eval_cfg,
            id_to_text        = id_to_text,
            T_model           = T_model,
            sentence_template = args.template,
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
