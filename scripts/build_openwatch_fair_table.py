#!/usr/bin/env python3
"""Assemble the §5f OpenWatch fair-comparison table from the per-model result
JSONs. Every row below was produced on the IDENTICAL contamination-safe
posture-stratified split, IDENTICAL 6 restricted-sub-interval 1s windows per
clip, and IDENTICAL window/clip(k=3) metric code — so the columns are
directly comparable, unlike Table 6's old cited-number row.

Inputs (any missing -> row marked pending):
  results_openwatch_paperprotocol.json          (ViT1D MAE, 4 rows x val/test)
  results_openwatch_mixtoken_{noaug,aug}.json
  results_openwatch_normwear_protocol_{noaug,aug}.json

Writes analysis/openwatch_fair_comparison.md (markdown + LaTeX) and prints it.
"""

from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "analysis", "openwatch_fair_comparison.md")

# Cited numbers, Bonazzi et al. Table 2 (their split, their run) — context row only.
CITED = {
    "aug": [("Mix-Token (cited, Bonazzi et al.)", "223K", 0.768, 0.908, 0.903, 0.955),
            ("Apple-CNN (cited)", "51K", 0.576, 0.774, 0.756, 0.879),
            ("NormWear-Base (cited; possibly +PPG)", "136M", 0.430, 0.682, 0.565, 0.788)],
    "noaug": [("NormWear-Base* (cited, no-aug IMU-only)", "136M", 0.340, 0.370, 0.620, 0.690)],
}


def _load(p):
    p = os.path.join(REPO, p)
    return json.load(open(p)) if os.path.exists(p) else None


def rows_for(condition: str):
    rows = []  # (name, params, split, win_f1, win_acc, clip_f1, clip_acc)
    tag = "win/aug" if condition == "aug" else "win/noaug"
    # main ViT1D rows: paper recipe (inverse-freq weighted CE, select on val macro-F1)
    for r in _load("results_openwatch_paperprotocol.json") or []:
        if r["name"].endswith(tag):
            nm = "Ours (MAE) probe" if "probe" in r["name"] else "Ours (MAE) finetune"
            rows.append((nm, "~21M", r["split"], r["window"]["macro_f1"], r["window"]["acc"],
                         r["clip"]["macro_f1"], r["clip"]["acc"]))
    # ablation rows: repo-default recipe (unweighted CE, select on val_loss) — kept
    # to show how much of any gap is the head recipe rather than the encoder
    for r in _load("results_openwatch_paperprotocol_unweighted_valloss.json") or []:
        if r["name"].endswith(tag):
            nm = ("Ours (MAE) probe" if "probe" in r["name"] else "Ours (MAE) finetune") + " [unweighted CE, val_loss sel.]"
            rows.append((nm, "~21M", r["split"], r["window"]["macro_f1"], r["window"]["acc"],
                         r["clip"]["macro_f1"], r["clip"]["acc"]))
    # Inertia-1 (Xu et al. 2026) external baseline -- our NHANES reproductions run
    # through the harness (baselines/inertia1/, EXPERIMENTS.md §16); same split /
    # windows / head recipe (weighted CE, macro-F1 selection) as the ViT1D rows.
    for v, nm0 in [("patchtst", "Inertia-1 PatchTST-small (NHANES repro)"), ("art", "Inertia-1 ART-small (NHANES repro)")]:
        for r in _load(f"results_openwatch_inertia1_{v}.json") or []:
            if r["name"].endswith(tag):
                nm = nm0 + (" probe" if "probe" in r["name"] else " finetune")
                rows.append((nm, "~8.5M", r["split"], r["window"]["macro_f1"], r["window"]["acc"],
                             r["clip"]["macro_f1"], r["clip"]["acc"]))
    for fname, nm in [(f"results_openwatch_mixtoken_{condition}.json", "Mix-Token (reimpl.)"),
                      (f"results_openwatch_normwear_protocol_{condition}.json", "NormWear-Base (first-party)")]:
        r = _load(fname)
        if r is None:
            rows.append((nm, "?", "pending", None, None, None, None))
            continue
        pstr = f"{r['params']/1e3:.0f}K" if r["params"] < 1e6 else f"{r['params']/1e6:.0f}M"
        for split in ["val", "test"]:
            m = r[split]
            rows.append((nm, pstr, split, m["window"]["macro_f1"], m["window"]["acc"],
                         m["clip"]["macro_f1"], m["clip"]["acc"]))
    return rows


def fmt(v):
    return "pending" if v is None else f"{v:.3f}"


def md_table(condition):
    title = {"aug": "Trained on augmented release, evaluated on real recordings (vs. paper headline rows)",
             "noaug": "Trained on real recordings only (vs. paper NormWear-Base* row)"}[condition]
    lines = [f"### {title}", "",
             "| Model | Params | Split | Window F1 | Window Acc | Clip F1 (k=3) | Clip Acc (k=3) |",
             "|---|---|---|---|---|---|---|"]
    for name, p, split, wf, wa, cf, ca in rows_for(condition):
        lines.append(f"| {name} | {p} | {split} | {fmt(wf)} | {fmt(wa)} | {fmt(cf)} | {fmt(ca)} |")
    lines += ["", "*Cited from Bonazzi et al. Table 2 — their split, their run; context only, NOT the same split:*", "",
              "| Model (cited) | Params | Window F1 | Window Acc | Clip F1 | Clip Acc |", "|---|---|---|---|---|---|"]
    for name, p, wf, wa, cf, ca in CITED[condition]:
        lines.append(f"| {name} | {p} | {wf:.3f} | {wa:.3f} | {cf:.3f} | {ca:.3f} |")
    return "\n".join(lines)


def latex_table():
    def block(condition, label):
        out = []
        for name, p, split, wf, wa, cf, ca in rows_for(condition):
            if split != "test":
                continue
            out.append(f"{name} & {p} & {fmt(wf)} & {fmt(wa)} & {fmt(cf)} & {fmt(ca)} \\\\")
        return f"\\multicolumn{{6}}{{l}}{{\\emph{{{label}}}}} \\\\\n" + "\n".join(out)
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{OpenWatch microgesture recognition under the OpenWatch protocol "
        r"(1\,s windows, stride 20, 6 windows/clip, $k{=}3$ clip voting), IMU-only, test split. "
        r"All rows: identical contamination-safe posture-stratified subject split, identical windows, "
        r"identical metric code. Mix-Token is our reimplementation (no official code); NormWear-Base is the "
        r"official checkpoint with the paper's head recipe.}",
        r"\label{tab:openwatch-fair}", r"\begin{tabular}{llcccc}", r"\toprule",
        r"Model & Params & Win.\ F1 & Win.\ Acc & Clip F1 & Clip Acc \\", r"\midrule",
        block("aug", "Trained with OpenWatch augmentation, evaluated on real recordings"),
        r"\midrule",
        block("noaug", "Trained on real recordings only"),
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])


def main():
    md = ["# OpenWatch fair comparison (§5f protocol)", "",
          "Identical split (contamination-safe, posture-stratified, canonical identities: 33/5/5; "
          "val=102 clips/612 windows, test=87 clips/522 windows), identical 6 windows/clip "
          "(1s, stride 20, centered 2.0s sub-interval), identical k=3 clip voting and metric code. "
          "Head recipe now IDENTICAL across all rows: inverse-frequency class-weighted CE, model selection "
          "on val window macro-F1 (paper §3.3.3). ViT1D rows tagged [unweighted CE, val_loss sel.] are the "
          "repo-default recipe kept as an ablation. **Test is the paper-comparable split.**", "",
          md_table("aug"), "", md_table("noaug"), "", "## LaTeX", "", "```latex", latex_table(), "```", ""]
    text = "\n".join(md)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(text)
    print(text)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
