#!/usr/bin/env python3
"""Complete bidirectional retrieval evaluation for IMU<->text alignment models.

Motivation (2026-08-15): every language-alignment number in this benchmark
so far is a single figure -- in-batch (gallery = batch size 64) top-1
retrieval accuracy averaged over randomly-drawn val batches (random clip
window, random caption per clip). That is (a) one number, (b) tied to a
gallery size that is an accident of the training batch size, and (c) noisy
(random windows/captions). Standard text-motion retrieval work (TMR) and
recent IMU-language work (AnyMo) instead report Recall@{1,5,10}, mean
reciprocal rank and median rank in BOTH directions, and AnyMo separately
reports a fixed-size (100-sample) gallery and the full gallery because
retrieval difficulty changes drastically with gallery size.

This script computes exactly that, once per checkpoint, on the SAME
held-out item set for every model:

  * IMU -> Text : query = one IMU window per val clip/segment; gallery = all
                  UNIQUE caption strings of the val split. A query is correct
                  at rank k if ANY of its own captions is in the top-k
                  (MS-COCO 5-caption convention: rank of the best positive).
  * Text -> IMU : query = each unique caption; gallery = all IMU windows.
                  Positives = every clip that carries that exact caption
                  string (Nymeria captions repeat heavily; the legacy in-batch
                  metric already treated identical captions as positives via
                  the cos>0.999 rule, so this is the same convention scaled
                  up to the full gallery).
  * Metrics      : R@1, R@5, R@10, MRR (all in %), MedR (median rank), MeanR.
  * Galleries    : full (every val item), plus fixed-size galleries of 64
                  (= the legacy batch size, so R@1@64 is the closest
                  deterministic analogue of the old acc_imu2text/acc_text2imu
                  numbers) and 100 (AnyMo's convention). Fixed-size numbers
                  pool the per-query ranks over --n-subsets random subsets.
  * caption_mode : 'all' (main, above) and 'first' (one caption per clip --
                  the deterministic first line, matching UniMTS's own
                  ValDataset convention) are both written to the JSON.

Deterministic windows (no random crop / caption):
  * HumanML3D ("Ours" ViT1D): replicates HumanML3DIMUVarLenDatasetFLU's
    per-item pipeline (clip loader by data_source, lowpass filter, per-window
    z-score, wrap/zero pad) with T = the config's fixed window when
    min==max window, else the clip's own snapped length. crop_mode 'first'
    is honoured; 'random' becomes a centre crop.
  * Nymeria ("Ours"): replicates NymeriaCaptionVarLenDatasetFLU per segment;
    variable-window configs use the segment's own snapped length clipped to
    [T_min, T_max] (centre crop above T_max, pad below T_min).
  * UniMTS: the exact input pipeline of train_unimts_userdata.py /
    train_unimts_nymeria.py (wrap-pad/truncate to 200 samples @ 20 Hz, NaN->0,
    optional wrist-only masking, gyro dropped by to_model_input).
  * Both model families on Nymeria enumerate segments with the SAME rule
    (>= 16 frames @ 50 Hz, the ViT1D patch size), so the galleries are
    identical (UniMTS's own scripts used >= 10 frames; the 10-15-frame
    segments are ~0.2-0.3 s and near-degenerate anyway).

Usage
-----
  # Ours (ViT1D CLIP-align run; objective type is read from the checkpoint cfg)
  python scripts/eval_retrieval_alignment.py --model ours \
      --run runs/vit1d_clip_humanml3d_finetune_mae_unimtssim_texttrainable_lr5e-5_i2t \
      --out analysis/retrieval_eval/humanml3d__ours_mae_finetune_i2t.json

  # UniMTS on HumanML3D UniMTS_data val (wrist-only input-matched eval)
  python scripts/eval_retrieval_alignment.py --model unimts \
      --checkpoint baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_jointmask_100ep/last.pth \
      --dataset humanml3d --wrist-only 1 --out analysis/retrieval_eval/humanml3d__unimts_scratch_jointmask.json

  # UniMTS on Nymeria (synthetic wrist IMU + atomic_action captions)
  python scripts/eval_retrieval_alignment.py --model unimts \
      --checkpoint baselines/language_alignment/UniMTS/checkpoint/nymeria_scratch_synthetic/best_val_loss.pth \
      --dataset nymeria --imu-source synthetic --caption-source atomic_action \
      --out analysis/retrieval_eval/nymeria__unimts_scratch_synthetic.json

Then: python scripts/summarize_retrieval_alignment.py analysis/retrieval_eval
"""
from __future__ import annotations

from imu_lm.utils.data_root import data_path as _data_path
import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _REPO_ROOT)

from imu_lm.utils.helpers import cfg_get  # noqa: E402

log = logging.getLogger("eval_retrieval")

_TARGET_FPS = 50
_MIN_SEG_FRAMES = 16   # ViT1D patch size; shared segment-validity rule for both model families


# ─────────────────────────────────────────────────────────────────────────────
# Item containers
# ─────────────────────────────────────────────────────────────────────────────

class Item:
    """One retrieval unit: a val clip (HumanML3D) or caption segment (Nymeria)."""
    __slots__ = ("key", "imu", "caps")

    def __init__(self, key: str, imu: np.ndarray, caps: List[str]):
        self.key = key
        self.imu = imu        # raw [T, 6] float32 (50 Hz) -- model-specific windowing applied later
        self.caps = caps      # list[str], stripped, non-empty, order preserved (caps[0] = 'first')


def _clean_caps(caps: Sequence[str]) -> List[str]:
    out: List[str] = []
    for c in caps:
        c = str(c).strip()
        if c:
            out.append(c)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dataset enumeration (shared by both model families)
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_humanml3d_val(root: str, *, patch_size: int, clip_min_T: int, T_max: int,
                            gravity_mps2: float, resample_method: str, joints_subpath: str,
                            rotation_source: str, captions_subpath: str, data_source: str,
                            joints_num: int = 22, split: str = "val") -> List[Item]:
    """Every clip of <root>/HumanML3D/<split>.txt that the training-time loader
    would have considered valid, via the very same clip loader
    (``_load_clip_imu_flu`` -> ``_load_clip_imu_unimts_sim`` for
    data_source='unimts_sim'). Returns raw 50 Hz [T,6] IMU + all captions."""
    from imu_lm.data.humanml3d_dataset_FLU import _load_clip_imu_flu
    from os.path import join as pjoin

    split_file = pjoin(root, "HumanML3D", f"{split}.txt")
    with open(split_file) as fh:
        ids = [l.strip() for l in fh if l.strip()]

    items: List[Item] = []
    n_skip = 0
    for cid in ids:
        clip = _load_clip_imu_flu(
            root, cid, joints_num, patch_size, clip_min_T, T_max,
            gravity_mps2=gravity_mps2, resample_method=resample_method,
            joints_subpath=joints_subpath, rotation_source=rotation_source,
            captions_subpath=captions_subpath, data_source=data_source,
        )
        if clip is None:
            n_skip += 1
            continue
        caps = _clean_caps(clip["caps"])
        if not caps:
            n_skip += 1
            continue
        items.append(Item(cid, np.asarray(clip["imu"], dtype=np.float32), caps))
    log.info("[humanml3d/%s] %d listed, %d loaded, %d skipped (data_source=%s, captions=%s)",
             split, len(ids), len(items), n_skip, data_source, captions_subpath)
    return items


def enumerate_humanml3d_val_unimts_raw(data_path: str, split_file: str) -> List[Tuple[str, np.ndarray, List[str]]]:
    """UniMTS-native items: (clip_id, sim [T,22,6] @20Hz NaN->0, caps) for
    split ∩ sim ∩ aug_texts -- the exact enumeration of
    train_unimts_userdata.load_userdata_split."""
    with open(split_file) as fh:
        ids = [l.strip() for l in fh if l.strip()]
    out = []
    n_skip = 0
    for cid in ids:
        sp = os.path.join(data_path, "sim", f"{cid}.npy")
        tp = os.path.join(data_path, "aug_texts", f"{cid}.txt")
        if not (os.path.exists(sp) and os.path.exists(tp)):
            n_skip += 1
            continue
        imu = np.load(sp).astype(np.float32)
        imu[np.isnan(imu)] = 0
        if imu.ndim != 3 or imu.shape[0] < 1:
            n_skip += 1
            continue
        caps = []
        with open(tp, encoding="utf-8") as fh:
            for line in fh:
                c = line.strip().split("#")[0].strip()
                if c:
                    caps.append(c)
        if not caps:
            n_skip += 1
            continue
        out.append((cid, imu, caps))
    log.info("[humanml3d/unimts_data] %d listed, %d loaded, %d skipped", len(ids), len(out), n_skip)
    return out


def enumerate_nymeria_val(captions_root: str, pkl_root: str, caption_source: str, imu_source: str,
                          add_gravity: bool = True, min_frames: int = _MIN_SEG_FRAMES,
                          target_hz: int = _TARGET_FPS, limit_sessions: Optional[int] = None) -> List[Item]:
    """Every (session, caption-row) segment of the val split with >= min_frames
    frames -- the same enumeration/time-slicing as NymeriaCaptionVarLenDatasetFLU
    (and train_unimts_nymeria.load_nymeria_split), raw 50 Hz [seg_T,6] IMU."""
    from imu_lm.data.nymeria_caption_dataset_FLU import (
        NYMERIA_CAPTION_SOURCES, _discover_sequences, _parse_caption_csv,
        _load_nymeria_session_imu_flu, _load_nymeria_session_imu_flu_synthetic,
    )
    from os.path import join as pjoin

    caption_filename = NYMERIA_CAPTION_SOURCES[caption_source]["filename"]
    caption_columns = NYMERIA_CAPTION_SOURCES[caption_source]["columns"]
    loader = (_load_nymeria_session_imu_flu_synthetic if imu_source == "synthetic"
              else _load_nymeria_session_imu_flu)
    seq_to_pkl = _discover_sequences(captions_root, pkl_root, caption_filename)

    items: List[Item] = []
    n_sess_skip = n_seg_skip = 0
    t0 = time.time()
    sids = sorted(seq_to_pkl.keys())
    if limit_sessions:
        sids = sids[:limit_sessions]
    for k, sid in enumerate(sids):
        session = loader(seq_to_pkl[sid], target_hz, add_gravity)
        if session is None:
            n_sess_skip += 1
            continue
        rows = _parse_caption_csv(pjoin(captions_root, sid, caption_filename), columns=caption_columns)
        for ri, (start_abs, end_abs, caps) in enumerate(rows):
            start_rel = start_abs - session["reference_time"]
            end_rel = min(end_abs - session["reference_time"], session["duration_seconds"])
            if start_rel < 0 or end_rel <= start_rel:
                n_seg_skip += 1
                continue
            start_idx = int(round(start_rel * target_hz))
            end_idx = min(int(round(end_rel * target_hz)), session["n_frames"])
            if end_idx - start_idx < min_frames:
                n_seg_skip += 1
                continue
            caps = _clean_caps(caps)
            if not caps:
                n_seg_skip += 1
                continue
            seg = np.asarray(session["imu"][start_idx:end_idx], dtype=np.float32)
            items.append(Item(f"{sid}:{ri}", seg, caps))
        if (k + 1) % 10 == 0:
            log.info("  ... %d/%d sessions, %d segments (%.0fs)", k + 1, len(seq_to_pkl), len(items), time.time() - t0)
    log.info("[nymeria/%s] %d sessions (%d skipped), %d segments (%d skipped), caption_source=%s imu_source=%s",
             os.path.basename(pkl_root), len(seq_to_pkl), n_sess_skip, len(items), n_seg_skip,
             caption_source, imu_source)
    return items


# ─────────────────────────────────────────────────────────────────────────────
# "Ours" ViT1D model: loading + windowing + encoding
# ─────────────────────────────────────────────────────────────────────────────

def load_ours(run_dir: str, ckpt_name: str, device: torch.device):
    """Return (encoder, clip_head, text_mlp, text_encoder_or_None, text_cache_or_None, cfg, ckpt_step)."""
    from imu_lm.models.ViT1D.model import ViT1DEncoder
    from imu_lm.objectives.contrastive_clip_1d import (
        ContrastiveCLIPHead, CLIPTextEncoder, TextProjectionMLP,
    )
    ckpt_path = os.path.join(run_dir, "checkpoints", f"{ckpt_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]

    from imu_lm.models.baselines.inertia1_encoder import build_encoder as _build_encoder
    encoder = _build_encoder(cfg)   # ViT1D / alt-attn / Inertia-1 adapter per the ckpt's cfg
    missing, unexpected = encoder.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        log.warning("encoder partial load: missing=%s unexpected=%s", missing, unexpected)
    encoder.eval().to(device)

    extra = ckpt.get("extra_modules", {}) or {}
    head_state = ckpt.get("clip_head") or extra.get("clip_head")
    if head_state is None:
        raise KeyError(f"clip_head missing from checkpoint; keys={list(ckpt.keys())}")
    proj_w = head_state["projection.weight"]
    clip_head = ContrastiveCLIPHead(imu_embed_dim=proj_w.shape[1], clip_embed_dim=proj_w.shape[0])
    clip_head.load_state_dict(head_state)
    clip_head.eval().to(device)

    text_mlp = None
    mlp_state = ckpt.get("text_mlp") or extra.get("text_mlp")
    if mlp_state is not None:
        text_mlp = TextProjectionMLP(embed_dim=proj_w.shape[0], hidden_dim=mlp_state["net.1.weight"].shape[0])
        text_mlp.load_state_dict(mlp_state)
        text_mlp.eval().to(device)

    obj_type = cfg_get(cfg, ["objective", "type"], "")
    ocfg = cfg_get(cfg, ["objective", obj_type], {}) or {}
    freeze_text = bool(ocfg.get("freeze_text_encoder", True))
    text_cache_path = ocfg.get("text_cache_path", None)
    text_cache = None
    if freeze_text and text_cache_path and os.path.exists(text_cache_path):
        text_cache = torch.load(text_cache_path, map_location="cpu", weights_only=False)
        log.info("text cache loaded: %d captions (%s)", len(text_cache), text_cache_path)

    text_enc = CLIPTextEncoder(
        model_name=ocfg.get("clip_model", "openai/clip-vit-base-patch32"),
        freeze=True,
        local_files_only=bool(ocfg.get("local_files_only", False)),
        use_projection=bool(ocfg.get("text_projection", False)),
    )
    te_state = ckpt.get("text_enc") or extra.get("text_enc")
    if te_state is not None:
        text_enc.load_state_dict(te_state)
        log.info("loaded FINE-TUNED text tower from checkpoint (freeze_text_encoder=false run)")
    elif not freeze_text:
        log.warning("cfg says freeze_text_encoder=false but no text_enc state in checkpoint -- "
                    "falling back to vanilla CLIP text weights (this is what the run itself "
                    "would have used only if text_cache_path silently no-op'd training).")
    text_enc.eval().to(device)
    return encoder, clip_head, text_mlp, text_enc, text_cache, cfg, int(ckpt.get("step", -1))


def _zscore(w: np.ndarray) -> np.ndarray:
    m = w.mean(axis=0, keepdims=True)
    s = w.std(axis=0, keepdims=True) + 1e-8
    return (w - m) / s


def _pad(w: np.ndarray, T: int, pad_mode: str) -> np.ndarray:
    n = w.shape[0]
    if n >= T:
        return w[:T]
    if pad_mode == "wrap":
        return np.pad(w, ((0, T - n), (0, 0)), mode="wrap").astype(np.float32)
    out = np.zeros((T, w.shape[1]), dtype=np.float32)
    out[:n] = w
    return out


def _crop(w: np.ndarray, T: int, mode: str) -> np.ndarray:
    n = w.shape[0]
    if n <= T:
        return w
    if mode == "first":
        s = 0
    else:                       # 'random' at train time -> deterministic centre crop at eval
        s = (n - T) // 2
    return w[s:s + T]


def window_ours_humanml3d(imu: np.ndarray, cfg: Any, *, T_min: int, T_max: int, patch: int,
                          pad_mode: str, crop_mode: str, zscore: bool) -> np.ndarray:
    """Mirror of HumanML3DIMUVarLenDatasetFLU.__iter__'s per-item path (no aug)."""
    from imu_lm.data.augmentations.preprocess import filter_window
    clip_T = imu.shape[0]
    T = T_max if T_min == T_max else clip_T          # clip_T is already snapped by the loader
    if clip_T >= T:
        w = _crop(imu, T, crop_mode)
        w = filter_window(w, cfg)
        if zscore:
            w = _zscore(w)
    else:
        w = filter_window(imu, cfg)
        if zscore:
            w = _zscore(w)
        w = _pad(w, T, pad_mode)
    return w.astype(np.float32)


def window_ours_nymeria(seg: np.ndarray, *, T_min: int, T_max: int, patch: int, pad_mode: str) -> np.ndarray:
    """Mirror of NymeriaCaptionVarLenDatasetFLU.__iter__'s per-item path (no aug, no filter):
    variable-window configs -> segment's own snapped length clipped to [T_min, T_max]."""
    seg_T = seg.shape[0]
    if T_min == T_max:
        T = T_max
    else:
        T = min(max((seg_T // patch) * patch, T_min), T_max)
    if seg_T >= T:
        w = _crop(seg, T, "center")
        w = _zscore(w)
    else:
        w = _zscore(seg)
        w = _pad(w, T, pad_mode)
    return w.astype(np.float32)


@torch.no_grad()
def encode_ours(items: List[Item], windows: List[np.ndarray], encoder, clip_head, device, batch_size: int,
                use_amp: bool) -> torch.Tensor:
    """Group by T, batch, encoder -> clip_head -> L2-normed [N, D] (float32, cpu)."""
    by_T: Dict[int, List[int]] = defaultdict(list)
    for i, w in enumerate(windows):
        by_T[int(w.shape[0])].append(i)
    D = clip_head.projection.weight.shape[0]
    z = torch.zeros(len(items), D, dtype=torch.float32)
    for T, idxs in sorted(by_T.items()):
        for b0 in range(0, len(idxs), batch_size):
            bi = idxs[b0:b0 + batch_size]
            x = torch.from_numpy(np.stack([windows[i].T for i in bi], axis=0)).float().to(device)  # [B, 6, T]
            # acc-only ablation (Trainer._acc_only_slice mirror). External encoders whose
            # in_channels means "token streams" (Inertia-1: 1) expose num_input_channels.
            _cin = int(getattr(encoder, "num_input_channels", getattr(encoder, "in_channels", x.shape[1])))
            if x.shape[1] > _cin:
                x = x[:, :_cin]
            with torch.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                zb = clip_head(encoder(x))
            z[bi] = F.normalize(zb.float(), dim=-1).cpu()
    return z


@torch.no_grad()
def encode_text_ours(captions: List[str], text_enc, text_mlp, text_cache, device, batch_size: int = 512) -> torch.Tensor:
    z = torch.zeros(len(captions), text_enc_dim(text_enc, text_mlp, text_cache), dtype=torch.float32)
    todo: List[int] = []
    n_cache = 0
    if text_cache is not None:
        for i, c in enumerate(captions):
            e = text_cache.get(c)
            if e is None:
                todo.append(i)
            else:
                z[i] = torch.as_tensor(np.asarray(e), dtype=torch.float32)
                n_cache += 1
    else:
        todo = list(range(len(captions)))
    log.info("text: %d from cache, %d encoded online", n_cache, len(todo))
    for b0 in range(0, len(todo), batch_size):
        bi = todo[b0:b0 + batch_size]
        z[bi] = text_enc([captions[i] for i in bi], device=device).float().cpu()
    if text_mlp is not None:
        outs = []
        for b0 in range(0, len(captions), batch_size):
            outs.append(text_mlp(z[b0:b0 + batch_size].to(device)).float().cpu())
        z = torch.cat(outs, dim=0)
    return F.normalize(z, dim=-1)


def text_enc_dim(text_enc, text_mlp, text_cache) -> int:
    if text_cache is not None:
        return int(np.asarray(next(iter(text_cache.values()))).shape[0])
    return int(text_enc.model.config.projection_dim if text_enc.use_projection else text_enc.model.config.hidden_size)


# ─────────────────────────────────────────────────────────────────────────────
# UniMTS model
# ─────────────────────────────────────────────────────────────────────────────

_UNIMTS_DIR = os.path.join(_REPO_ROOT, "baselines", "language_alignment", "UniMTS")


def load_unimts(checkpoint: str, device: torch.device):
    sys.path.insert(0, _UNIMTS_DIR)
    from types import SimpleNamespace
    from contrastive import ContrastiveModule  # noqa
    model = ContrastiveModule(SimpleNamespace(gyro=0, stft=0, stage="evaluation")).to(device)
    sd = torch.load(checkpoint, map_location="cpu")
    model.model.load_state_dict(sd)
    model.eval()
    log.info("UniMTS loaded: %s", checkpoint)
    return model


def unimts_flat_from_grid(imu_grid: np.ndarray, wrist_only: bool) -> np.ndarray:
    """[T,22,6] @20Hz -> [200,132] exactly as train_unimts_userdata.load_userdata_split."""
    from train_unimts_userdata import PADDING_SIZE, NUM_JOINTS, RIGHT_WRIST_JOINT  # noqa
    imu = imu_grid
    if wrist_only:
        keep = imu[:, RIGHT_WRIST_JOINT, :].copy()
        imu = np.zeros_like(imu)
        imu[:, RIGHT_WRIST_JOINT, :] = keep
    if len(imu) < PADDING_SIZE:
        imu = np.pad(imu, ((0, PADDING_SIZE - len(imu)), (0, 0), (0, 0)), mode="wrap")
    return imu[:PADDING_SIZE].reshape(PADDING_SIZE, NUM_JOINTS * 6).astype(np.float32)


def unimts_flat_from_wrist_50hz(seg: np.ndarray) -> np.ndarray:
    """raw 50 Hz [T,6] wrist -> [200,132] exactly as train_unimts_nymeria.load_nymeria_split."""
    from eval_unimts_nymeria import NUM_JOINTS, PADDING_SIZE, RIGHT_WRIST_JOINT, resample_and_pad  # noqa
    acc_gyro = resample_and_pad(seg)
    grid = np.zeros((PADDING_SIZE, NUM_JOINTS, 6), dtype=np.float32)
    grid[:, RIGHT_WRIST_JOINT, :] = acc_gyro
    return grid.reshape(PADDING_SIZE, NUM_JOINTS * 6)


@torch.no_grad()
def encode_unimts(model, flats: List[np.ndarray], captions: List[str], device, batch_size: int):
    from train_unimts_humanml3d import to_model_input  # noqa
    import clip  # noqa
    zi = []
    for b0 in range(0, len(flats), batch_size):
        flat = torch.from_numpy(np.stack(flats[b0:b0 + batch_size]))
        x = to_model_input(flat, aug=False, device=device)
        zi.append(F.normalize(model.encode_image(x).float(), dim=-1).cpu())
    zt = []
    for b0 in range(0, len(captions), batch_size):
        tok = clip.tokenize(captions[b0:b0 + batch_size], truncate=True).to(device)
        zt.append(F.normalize(model.encode_text(tok).float(), dim=-1).cpu())
    return (torch.cat(zi, 0) if zi else None), (torch.cat(zt, 0) if zt else None)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval metrics
# ─────────────────────────────────────────────────────────────────────────────

def _best_positive_scores(sim: torch.Tensor, positives: List[np.ndarray]) -> torch.Tensor:
    best = torch.empty(sim.shape[0], dtype=sim.dtype, device=sim.device)
    for q, pos in enumerate(positives):
        best[q] = sim[q, torch.as_tensor(pos, device=sim.device)].max()
    return best


def _ranks(sim: torch.Tensor, positives: List[np.ndarray], chunk: int = 2048) -> np.ndarray:
    """Expected rank of q's best positive under random tie-breaking:
    1 + #{items scoring strictly higher} + 0.5 * #{non-positive items tied with it}.

    Fixed 2026-09-19: the previous rule counted only strictly higher items, so a collapsed encoder
    (all embeddings equal -> every score tied) was credited with rank 1 for every query. Exact ties
    do not occur for non-degenerate embeddings, so ordinary results are unchanged.
    """
    best = _best_positive_scores(sim, positives)
    ranks = torch.empty(sim.shape[0], dtype=torch.float64, device=sim.device)
    for s in range(0, sim.shape[0], chunk):
        e = min(s + chunk, sim.shape[0])
        blk, b = sim[s:e], best[s:e, None]
        higher = (blk > b).sum(dim=1).double()
        tied = (blk == b).sum(dim=1).double()
        for q in range(s, e):                      # positives tied with the best one are not competitors
            pos = torch.as_tensor(positives[q], device=sim.device)
            tied[q - s] -= (sim[q, pos] == best[q]).sum().double()
        ranks[s:e] = 1.0 + higher + 0.5 * tied
    return ranks.cpu().numpy()


def summarize_ranks(ranks: np.ndarray, ks=(1, 5, 10)) -> Dict[str, float]:
    r = ranks.astype(np.float64)
    out = {f"R@{k}": float((r <= k).mean() * 100.0) for k in ks}
    out["MRR"] = float((1.0 / r).mean() * 100.0)
    out["MedR"] = float(np.median(r))
    out["MeanR"] = float(r.mean())
    out["n_queries"] = int(len(r))
    return out


def build_caption_index(items: List[Item], caption_mode: str):
    """Unique-caption gallery + per-item positive caption ids + per-caption positive item ids."""
    cap2id: Dict[str, int] = {}
    item_caps: List[np.ndarray] = []
    for it in items:
        caps = it.caps if caption_mode == "all" else it.caps[:1]
        ids = []
        for c in caps:
            j = cap2id.setdefault(c, len(cap2id))
            ids.append(j)
        item_caps.append(np.unique(np.asarray(ids, dtype=np.int64)))
    captions = [None] * len(cap2id)
    for c, j in cap2id.items():
        captions[j] = c
    cap_items: List[List[int]] = [[] for _ in captions]
    for i, ids in enumerate(item_caps):
        for j in ids:
            cap_items[j].append(i)
    cap_items = [np.asarray(v, dtype=np.int64) for v in cap_items]
    return captions, item_caps, cap_items


def eval_full(z_imu: torch.Tensor, z_txt: torch.Tensor, item_caps, cap_items, device) -> Dict[str, Any]:
    sim = (z_imu.to(device) @ z_txt.to(device).T)              # [N, M]
    r_i2t = _ranks(sim, item_caps)
    r_t2i = _ranks(sim.T.contiguous(), cap_items)
    return {
        "gallery": "full",
        "n_imu": int(z_imu.shape[0]), "n_text": int(z_txt.shape[0]),
        "imu2text": summarize_ranks(r_i2t),
        "text2imu": summarize_ranks(r_t2i),
    }


def eval_fixed(z_imu: torch.Tensor, z_txt: torch.Tensor, item_caps, G: int, n_subsets: int, seed: int,
               device, collect: Optional[Dict[str, list]] = None) -> Dict[str, Any]:
    """Pool per-query ranks over n_subsets random galleries of G items (text gallery =
    the unique captions of those G items).

    collect: optional dict; if given, the per-query records are appended to it
    (i2t_subset/i2t_item/i2t_rank, t2i_subset/t2i_cap/t2i_rank) -- the rng stream
    and every reported number are unchanged."""
    N = z_imu.shape[0]
    if N < G:
        return {"gallery": G, "skipped": f"only {N} items"}
    rng = np.random.default_rng(seed)
    zi, zt = z_imu.to(device), z_txt.to(device)
    all_i2t, all_t2i = [], []
    r1_i2t, r1_t2i = [], []
    text_gallery_sizes = []
    for _ in range(n_subsets):
        idx = np.sort(rng.choice(N, G, replace=False))
        cap_ids = np.unique(np.concatenate([item_caps[i] for i in idx]))
        text_gallery_sizes.append(len(cap_ids))
        remap = {int(c): k for k, c in enumerate(cap_ids)}
        sim = zi[torch.as_tensor(idx, device=device)] @ zt[torch.as_tensor(cap_ids, device=device)].T  # [G, M]
        pos_i2t = [np.asarray([remap[int(c)] for c in item_caps[i]], dtype=np.int64) for i in idx]
        pos_t2i: List[List[int]] = [[] for _ in cap_ids]
        for k, i in enumerate(idx):
            for c in item_caps[i]:
                pos_t2i[remap[int(c)]].append(k)
        pos_t2i = [np.asarray(v, dtype=np.int64) for v in pos_t2i]
        ri = _ranks(sim, pos_i2t)
        rt = _ranks(sim.T.contiguous(), pos_t2i)
        all_i2t.append(ri)
        all_t2i.append(rt)
        if collect is not None:
            s_id = len(all_i2t) - 1
            collect.setdefault("i2t_subset", []).append(np.full(len(idx), s_id, dtype=np.int32))
            collect.setdefault("i2t_item", []).append(np.asarray(idx, dtype=np.int64))
            collect.setdefault("i2t_rank", []).append(np.asarray(ri, dtype=np.int64))
            collect.setdefault("t2i_subset", []).append(np.full(len(cap_ids), s_id, dtype=np.int32))
            collect.setdefault("t2i_cap", []).append(np.asarray(cap_ids, dtype=np.int64))
            collect.setdefault("t2i_rank", []).append(np.asarray(rt, dtype=np.int64))
        r1_i2t.append(float((ri <= 1).mean() * 100))
        r1_t2i.append(float((rt <= 1).mean() * 100))
    out = {
        "gallery": G, "n_subsets": n_subsets,
        "imu2text": summarize_ranks(np.concatenate(all_i2t)),
        "text2imu": summarize_ranks(np.concatenate(all_t2i)),
    }
    out["imu2text"]["R@1_subset_std"] = float(np.std(r1_i2t))
    out["text2imu"]["R@1_subset_std"] = float(np.std(r1_t2i))
    out["mean_text_gallery"] = float(np.mean(text_gallery_sizes))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["ours", "unimts"])
    ap.add_argument("--run", help="[ours] run dir (or name under runs/)")
    ap.add_argument("--ckpt", default="best", help="[ours] checkpoint name (best|latest)")
    ap.add_argument("--checkpoint", help="[unimts] state-dict .pth path")
    ap.add_argument("--dataset", choices=["humanml3d", "nymeria"], help="[unimts] which val set")
    ap.add_argument("--wrist-only", type=int, default=1, help="[unimts/humanml3d] zero all joints but the right wrist")
    ap.add_argument("--imu-source", default="synthetic", choices=["real", "synthetic"], help="[unimts/nymeria]")
    ap.add_argument("--caption-source", default="atomic_action", help="[unimts/nymeria]")
    ap.add_argument("--add-gravity", type=int, default=1, help="[unimts/nymeria]")
    ap.add_argument("--humanml3d-root", default=_data_path("/projects/b53017/vbc1591/HumanML3D"))
    ap.add_argument("--nymeria-captions-root", default=_data_path("/projects/b53017/vbc1591/nymeria_captions"))
    ap.add_argument("--nymeria-val-pkl-root", default=_data_path("/projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist"))
    ap.add_argument("--split", default="val", help="[humanml3d] which HumanML3D/<split>.txt to evaluate (val|test); Section-13 tables use val")
    ap.add_argument("--gallery-sizes", default="64,100")
    ap.add_argument("--n-subsets", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--label", default=None, help="human-readable row label stored in the JSON")
    ap.add_argument("--limit-items", type=int, default=None, help="debug: cap #items after enumeration")
    ap.add_argument("--limit-sessions", type=int, default=None, help="debug: cap #Nymeria val sessions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-units", default=None,
                    help="optional .npz with the per-query ranks of every fixed-gallery protocol "
                         "(per caption mode and G) plus the item/caption index, for bootstrap CIs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gallery_sizes = [int(g) for g in args.gallery_sizes.split(",") if g.strip()]
    t0 = time.time()
    meta: Dict[str, Any] = {"model": args.model, "label": args.label, "seed": args.seed,
                            "n_subsets": args.n_subsets, "device": str(device), "split": args.split}

    # ── build items + embeddings ─────────────────────────────────────────────
    if args.model == "ours":
        run_dir = args.run if os.path.isdir(args.run) else os.path.join(_REPO_ROOT, "runs", args.run)
        encoder, clip_head, text_mlp, text_enc, text_cache, cfg, step = load_ours(run_dir, args.ckpt, device)
        obj_type = cfg_get(cfg, ["objective", "type"], "")
        ocfg = cfg_get(cfg, ["objective", obj_type], {}) or {}
        if bool(cfg_get(cfg, ["data", "acc_only"], False)):
            # acc-only ablation: the trainer truncated every batch to the first
            # in_channels (=3) channels; encode_ours mirrors that below.
            log.info("data.acc_only checkpoint: IMU windows will be truncated to %d channels",
                     int(getattr(encoder, "in_channels", 3)))
        patch = int(cfg_get(cfg, ["vit1d", "patch_size"], 16))
        win = cfg_get(cfg, ["windowing"], {}) or {}
        T_min = int(round(float(win.get("min_window_seconds", 1.28)) * _TARGET_FPS / patch)) * patch
        T_max = int(round(float(win.get("max_window_seconds", 10.24)) * _TARGET_FPS / patch)) * patch
        pad_mode = str(win.get("pad_mode", "zero"))
        use_amp = bool(cfg_get(cfg, ["trainer", "amp"], False))
        meta.update({"run": os.path.basename(run_dir.rstrip("/")), "ckpt": args.ckpt, "ckpt_step": step,
                     "objective": obj_type, "T_min": T_min, "T_max": T_max, "pad_mode": pad_mode})

        if obj_type == "imu_text_clip_flu":
            dataset = "humanml3d"
            crop_mode = cfg_get(cfg, ["humanml3d", "crop_mode"], "random")
            zscore = bool(cfg_get(cfg, ["humanml3d", "zscore"], True))
            items = enumerate_humanml3d_val(
                ocfg["humanml3d_root"], patch_size=patch,
                clip_min_T=(patch if pad_mode == "wrap" else T_min), T_max=T_max,
                gravity_mps2=float(cfg_get(cfg, ["humanml3d", "gravity_mps2"], 0.0)),
                resample_method=cfg_get(cfg, ["humanml3d", "resample_method"], "linear"),
                joints_subpath=cfg_get(cfg, ["humanml3d", "joints_subpath"], "joints"),
                rotation_source=cfg_get(cfg, ["humanml3d", "rotation_source"], "amass"),
                captions_subpath=cfg_get(cfg, ["humanml3d", "captions_subpath"], "texts"),
                data_source=cfg_get(cfg, ["humanml3d", "data_source"], "derived"),
                split=args.split,
            )
            if args.limit_items:
                items = items[:args.limit_items]
            windows = [window_ours_humanml3d(it.imu, cfg, T_min=T_min, T_max=T_max, patch=patch,
                                             pad_mode=pad_mode, crop_mode=crop_mode, zscore=zscore)
                       for it in items]
            meta.update({"data_source": cfg_get(cfg, ["humanml3d", "data_source"], "derived"),
                         "captions_subpath": cfg_get(cfg, ["humanml3d", "captions_subpath"], "texts"),
                         "crop_mode": crop_mode})
        elif obj_type == "imu_text_clip_nymeria_flu":
            dataset = "nymeria"
            items = enumerate_nymeria_val(
                ocfg["captions_root"], ocfg["val_pkl_root"],
                caption_source=ocfg.get("caption_source", "motion_narration"),
                imu_source=ocfg.get("imu_source", "real"),
                add_gravity=bool(ocfg.get("add_gravity", True)),
                limit_sessions=args.limit_sessions,
            )
            if args.limit_items:
                items = items[:args.limit_items]
            windows = [window_ours_nymeria(it.imu, T_min=T_min, T_max=T_max, patch=patch, pad_mode=pad_mode)
                       for it in items]
            meta.update({"caption_source": ocfg.get("caption_source", "motion_narration"),
                         "imu_source": ocfg.get("imu_source", "real")})
        else:
            raise ValueError(f"unsupported objective type {obj_type!r}")

        log.info("encoding %d IMU windows ...", len(items))
        z_imu = encode_ours(items, windows, encoder, clip_head, device, args.batch_size, use_amp)
        del windows

        def text_embed_fn(captions: List[str]) -> torch.Tensor:
            return encode_text_ours(captions, text_enc, text_mlp, text_cache, device)

    else:  # unimts
        if not args.checkpoint or not args.dataset:
            ap.error("--model unimts requires --checkpoint and --dataset")
        model = load_unimts(args.checkpoint, device)
        dataset = args.dataset
        meta.update({"checkpoint": os.path.relpath(args.checkpoint, _REPO_ROOT)})
        if dataset == "humanml3d":
            raw = enumerate_humanml3d_val_unimts_raw(
                os.path.join(args.humanml3d_root, "UniMTS_data", "data"),
                os.path.join(args.humanml3d_root, "HumanML3D", f"{args.split}.txt"))
            if args.limit_items:
                raw = raw[:args.limit_items]
            items = [Item(cid, np.zeros((0, 6), np.float32), _clean_caps(caps)) for cid, _, caps in raw]
            flats = [unimts_flat_from_grid(g, bool(args.wrist_only)) for _, g, _ in raw]
            del raw
            meta.update({"wrist_only": bool(args.wrist_only), "data_source": "unimts_sim", "captions_subpath": "aug_texts"})
        else:
            items = enumerate_nymeria_val(
                args.nymeria_captions_root, args.nymeria_val_pkl_root,
                caption_source=args.caption_source, imu_source=args.imu_source,
                add_gravity=bool(args.add_gravity), limit_sessions=args.limit_sessions,
            )
            if args.limit_items:
                items = items[:args.limit_items]
            flats = [unimts_flat_from_wrist_50hz(it.imu) for it in items]
            meta.update({"caption_source": args.caption_source, "imu_source": args.imu_source})

        log.info("encoding %d IMU items ...", len(items))
        z_imu, _ = encode_unimts(model, flats, [], device, args.batch_size)

        def text_embed_fn(captions: List[str]) -> torch.Tensor:
            _, zt = encode_unimts(model, [], captions, device, args.batch_size)
            return zt

    for it in items:      # free raw signal
        it.imu = None

    # ── metrics for both caption modes ───────────────────────────────────────
    results: Dict[str, Any] = {"meta": meta, "dataset": dataset, "n_items": len(items), "protocols": {}}
    n_pairs = int(sum(len(it.caps) for it in items))
    results["n_caption_pairs"] = n_pairs
    units: Dict[str, np.ndarray] = {}
    if args.dump_units:
        units["item_key"] = np.asarray([str(it.key) for it in items])
    for caption_mode in ("all", "first"):
        captions, item_caps, cap_items = build_caption_index(items, caption_mode)
        if args.dump_units and caption_mode == "first":
            # one caption per item under 'first': the item's caption id (items can share a caption)
            units[f"{caption_mode}__item_cap"] = np.asarray([int(c[0]) for c in item_caps], dtype=np.int64)
            units[f"{caption_mode}__n_captions"] = np.asarray(len(captions))
        log.info("[caption_mode=%s] %d unique captions over %d items -> encoding text ...",
                 caption_mode, len(captions), len(items))
        z_txt = text_embed_fn(captions)
        proto: Dict[str, Any] = {
            "n_unique_captions": len(captions),
            "full": eval_full(z_imu, z_txt, item_caps, cap_items, device),
            "fixed": {},
        }
        for G in gallery_sizes:
            col: Optional[Dict[str, list]] = {} if args.dump_units else None
            proto["fixed"][str(G)] = eval_fixed(z_imu, z_txt, item_caps, G, args.n_subsets, args.seed, device,
                                                collect=col)
            for k, v in (col or {}).items():
                units[f"{caption_mode}__g{G}__{k}"] = np.concatenate(v)
        results["protocols"][caption_mode] = proto
        f = proto["full"]
        log.info("[%s] FULL  i2t R@1/5/10=%.2f/%.2f/%.2f MRR=%.2f MedR=%.0f | t2i R@1/5/10=%.2f/%.2f/%.2f MRR=%.2f MedR=%.0f",
                 caption_mode, f["imu2text"]["R@1"], f["imu2text"]["R@5"], f["imu2text"]["R@10"], f["imu2text"]["MRR"],
                 f["imu2text"]["MedR"], f["text2imu"]["R@1"], f["text2imu"]["R@5"], f["text2imu"]["R@10"],
                 f["text2imu"]["MRR"], f["text2imu"]["MedR"])
        for G, fx in proto["fixed"].items():
            if "skipped" in fx:
                continue
            log.info("[%s] G=%s  i2t R@1/5/10=%.2f/%.2f/%.2f MRR=%.2f | t2i R@1/5/10=%.2f/%.2f/%.2f MRR=%.2f",
                     caption_mode, G, fx["imu2text"]["R@1"], fx["imu2text"]["R@5"], fx["imu2text"]["R@10"],
                     fx["imu2text"]["MRR"], fx["text2imu"]["R@1"], fx["text2imu"]["R@5"], fx["text2imu"]["R@10"],
                     fx["text2imu"]["MRR"])

    if args.dump_units:
        units["meta_json"] = np.asarray(json.dumps(meta))
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_units)), exist_ok=True)
        np.savez_compressed(args.dump_units, **units)
        log.info("wrote per-query units %s", args.dump_units)
    results["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    log.info("wrote %s (%.0fs)", args.out, results["elapsed_s"])


if __name__ == "__main__":
    main()
