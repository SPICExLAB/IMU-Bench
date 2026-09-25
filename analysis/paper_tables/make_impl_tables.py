#!/usr/bin/env python3
"""Appendix implementation tables, read from the configs/checkpoints that produced the paper's numbers.

    python analysis/paper_tables/make_impl_tables.py

       <paper>/Sections/tab_downstream_protocols.tex (one row per task: window, decoders, LR, selection)
Downstream settings are read from the generated bench configs of the inrtlWrist cells (identical for every
encoder of a task apart from the encoder block).
"""
import os
import re
import sys

import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAPER = "/gpfs/home/vbc1591/paper_iclr/Sections"
sys.path.insert(0, REPO)
from bench.leaderboard_paper import COLS  # noqa: E402

PRE = [  # label, config, run, objective description
    ("inrtlWrist (Ours)", "configs/vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic_flu.yaml",
     "vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic", "masked patch reconstruction, 75\\,\\% mask"),
    ("LIMU-BERT", "configs/pretrain5src/limubert_5src.yaml", "vit1d_ssl_limubert_5src",
     "15\\,\\% temporal span mask, MSE, 4$\\times$192 decoder"),
    ("SimMTM", "configs/pretrain5src/simmtm_5src.yaml", "vit1d_ssl_simmtm_5src",
     "50\\,\\% mask, 3 positives, $\\tau{=}0.2$"),
    ("FOCAL", "configs/pretrain5src/focal_5src.yaml", "vit1d_ssl_focal_5src",
     "acc/gyro factorised contrastive, $\\tau{=}0.07$"),
    ("RelCon", "configs/pretrain5src/relcon_5src.yaml", "vit1d_ssl_relcon_5src",
     "relative contrastive, learned motif distance, $\\tau{=}0.1$"),
    ("SSL-Wearables", "configs/pretrain5src/sslwear_5src.yaml", "vit1d_ssl_sslwear_5src",
     "arrow of time + permutation + time warp, $p{=}0.5$ each"),
    ("DINO", "configs/pretrain5src/dino_5src.yaml", "vit1d_ssl_dino_5src",
     "2 global + 4 local crops, EMA teacher, 4,096 prototypes"),
    ("I-JEPA", "configs/pretrain5src/jepa_5src.yaml", "vit1d_ssl_jepa_5src",
     "4 target blocks, EMA target encoder, 4$\\times$192 predictor"),
    ("inrtlWrist, plain attention", "configs/pretrain5src/mae_noaltattn_5src.yaml", "vit1d_ssl_mae_varlen_noaltattn_5src",
     "as inrtlWrist"),
    ("inrtlWrist, accelerometer only", "configs/pretrain5src/mae_acconly_5src.yaml", "vit1d_ssl_mae_varlen_acconly_5src",
     "as inrtlWrist, 3 input channels"),
]


# An arm whose pretraining has not produced a checkpoint yet is left out of the table (it returns automatically).
PRE = [r for r in PRE if os.path.exists(os.path.join(REPO, "runs", r[2], "checkpoints", "best.pt"))]


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def best_step(run):
    p = os.path.join(REPO, "runs", run, "checkpoints", "best.pt")
    if not os.path.exists(p):
        return None
    return int(torch.load(p, map_location="cpu", weights_only=False).get("step"))


pre_csv = []                      # the released CSV; the printed table is gone (see below)
for lab, cfgp, run, desc in PRE:
    c = yaml.safe_load(open(os.path.join(REPO, cfgp)))
    enc = g(c, "vit1d", "encoder", default={})
    tr = c.get("trainer", {})
    ms = tr.get("max_steps")
    step = best_step(run)
    pre_csv.append([lab, desc.replace("\\,", " ").replace("\\%", "%").replace("$\\times$", "x").replace("$\\tau{=}", "tau=").replace("$", ""),
                    "alternating" if enc.get("alt_attention") else "plain", ms if ms and ms <= 10**6 else None, step, run, cfgp])
# The self-supervised pretraining table (tab_pretrain_details.tex, formerly Table 6) was removed
# from the paper on 2026-09-24 (user decision). The per-encoder settings it printed -- objective,
# attention pattern, step cap and selected step -- stay in the released CSV written at the end of
# this file, and Appendix Pretraining states the shared backbone, corpus and optimiser in prose.

# ---- downstream protocols ------------------------------------------------------------------------
TASKS = yaml.safe_load(open(os.path.join(REPO, "bench/registry/tasks.yaml")))["tasks"]
NAMES = {t: lab for t, lab, _ in COLS}


def gen_cfg(task, proto):
    p = os.path.join(REPO, "bench/generated/configs", f"bench_{task}__mae_5src__{proto}__s42.yaml")
    return yaml.safe_load(open(p)) if os.path.exists(p) else None


# Class defaults of ViT1DPoseDecoder / ViT1DOdomDecoder: a key a task card leaves out is NOT zero.
# Reading them from the classes keeps this table honest if the defaults ever change.
def _defaults():
    import inspect
    from imu_lm.models.ViT1D import model as _m
    out = {}
    for cls, key in ((_m.ViT1DPoseDecoder, "pose_decoder"), (_m.ViT1DOdomDecoder, "odom_decoder")):
        src = inspect.getsource(cls.__init__)
        out[key] = {n: int(v) for n, v in
                    re.findall(r'dec_cfg\.get\(\s*"(num_hidden_layers|num_temporal_layers|hidden_size)"\s*,\s*(\d+)', src)}
    return out


DEC_DEFAULTS = _defaults()


def decoder(c):
    v = c.get("vit1d", {}) if c else {}
    for k in ("pose_decoder", "odom_decoder"):
        d = v.get(k)
        if not isinstance(d, dict):
            continue
        dflt = DEC_DEFAULTS.get(k, {})
        n = int(d.get("num_hidden_layers", dflt.get("num_hidden_layers", 0)))
        t = d.get("num_temporal_layers", dflt.get("num_temporal_layers"))
        t = 0 if t is None else int(t)
        depth = f"{n}+{t}" if t else f"{n}"
        return f"{depth} layer{'s' if (n + t) != 1 else ''}, {d.get('hidden_size', dflt.get('hidden_size'))}"
    return "linear"


drows, dp_csv = [], []
for t, lab, _ in COLS:
    card = TASKS[t]
    cp, cf = gen_cfg(t, "probe"), gen_cfg(t, "finetune")
    win = card.get("input", {}).get("window_seconds")
    lr = lambda c: "--" if c is None else f"{g(c, 'trainer', 'optim', 'lr'):g}"
    sel = card.get("selection", {})
    SEL = {"mpjpe_mm": "MPJPE", "val_loss": "val.\\ loss", "macro_f1": "macro-F1", "bal_acc": "bal.\\ acc."}
    head_p, head_f = decoder(cp), decoder(cf)
    if t == "language_humanml3d":
        head_p = head_f = "projection to text space"
    # Selection metric, eval split and learning rate were dropped from the PRINTED table on
    # 2026-09-24 (user
    # decision). The caption still gives the early-stopping rule and says which tasks report a
    # test split, and the per-task values stay in the released CSV below.
    dp_csv.append([t, lab, win, head_p, head_f, lr(cp), lr(cf), sel.get("monitor", "fixed 8k steps"),
                   card.get("eval_split", "val")])
    drows.append(f"{lab} & {win:g}\\,s & {head_p} & {head_f} \\\\")
dl = [r"% Generated by IMU-SSL/analysis/paper_tables/make_impl_tables.py -- do not edit by hand.",
      r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
      r"\caption{Downstream protocols, shared by every encoder of a task. Decoder depth is given as "
      r"cross-channel(+temporal) attention layers and width; activity, gesture and Parkinson's heads are linear. "
      r"Runs use early stopping (patience 20 evaluations) on the selection metric "
      r"of the validation split, except IMU--text retrieval, which aligns IMU embeddings with the CLIP ViT-B/32 "
      r"text space (InfoNCE, $\tau{=}0.1$, text tower trainable) for a fixed 8,000 steps. Gesture and IMU--text "
      r"report their held-out test split; the other tasks have no test split and report the validation split "
      r"used for model selection.}",
      r"\label{tab:downstream_protocols}",
      r"\begin{tabular}{@{}llll@{}}", r"\toprule",
      r"Task & Window & Probe decoder & Fine-tune decoder \\", r"\midrule"] + drows + \
     [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
open(os.path.join(PAPER, "tab_downstream_protocols.tex"), "w").write("\n".join(dl))
print("\n".join(drows))

sys.path.insert(0, os.path.join(REPO, "paper_results"))
from paper_io import write_csv  # noqa: E402
write_csv("appendix_pretrain_details", "pretrain_details.csv",
          ["encoder", "objective", "attention", "step_cap", "selected_step", "run", "config"], pre_csv)
write_csv("appendix_downstream_protocols", "downstream_protocols.csv",
          ["task", "task_label", "window_s", "probe_decoder", "finetune_decoder", "lr_probe", "lr_finetune",
           "selection_metric", "eval_split"], dp_csv)
