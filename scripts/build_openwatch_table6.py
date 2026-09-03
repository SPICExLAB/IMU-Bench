#!/usr/bin/env python3
"""Build the paper's Table 6 ("Microgesture recognition results on OpenWatch.
Higher is better.  Model | Acc. | Bal. Acc. | Macro-F1") in the NEW
configuration = the §5f OpenWatch paper protocol, from the per-model result
JSONs. Every row: identical contamination-safe posture-stratified split,
identical 6 restricted-sub-interval 1s windows/clip, k=3 clip voting,
identical metric code, inverse-frequency class-weighted CE + selection on
val window macro-F1 (paper §3.3.3). Test split; the paper's Table 2 reports
"all test windows".

Emits BOTH aggregation levels — the paper reports both:
  * clip-level (k=3): the paper's headline numbers -> Table 6 main
  * window-level: the paper's "Window-Level" columns -> Table 6b

Inputs:
  results_openwatch_paperprotocol.json    (Ours MAE probe/finetune x cond x split)
  results_openwatch_table6_rows.json      (FOCAL/SimMTM/RelCon/no-alt-attn/acc-only/Scratch)
  results_openwatch_mixtoken_{cond}.json, results_openwatch_normwear_protocol_{cond}.json

Writes analysis/table6_openwatch_new.md (markdown + LaTeX) and prints it.
Usage: python scripts/build_openwatch_table6.py [--split test|val] [--condition aug|noaug|both]
"""

from __future__ import annotations

import argparse
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "analysis", "table6_openwatch_new.md")

# Table 6 row order (paper's), + the two protocol baselines from the OpenWatch paper.
ROW_ORDER = [
    "FOCAL", "SimMTM", "RelCon", "Scratch",
    "Ours (MAE)", "Ours (MAE), finetuned", "Ours (MAE), no alt-attn", "Ours (MAE), acc-only",
    "Mix-Token (reimpl.)", "NormWear-Base (first-party)",
]
PARAMS = {"Mix-Token (reimpl.)": "219K", "NormWear-Base (first-party)": "136M"}


def _load(p):
    p = os.path.join(REPO, p)
    return json.load(open(p)) if os.path.exists(p) else None


def collect(condition: str, split: str):
    """-> {row_name: {"clip": metrics, "window": metrics}} for one condition/split."""
    tag = f"win/{condition}"
    out = {}
    for r in (_load("results_openwatch_paperprotocol.json") or []) + (_load("results_openwatch_table6_rows.json") or []):
        if r.get("split") != split or not r["name"].endswith(tag):
            continue
        base = r["name"][: -len(" " + tag)]
        name = {"Ours (MAE) probe": "Ours (MAE)", "Ours (MAE) finetune": "Ours (MAE), finetuned"}.get(base, base)
        out[name] = {"clip": r["clip"], "window": r["window"]}
    for fname, name in [(f"results_openwatch_mixtoken_{condition}.json", "Mix-Token (reimpl.)"),
                        (f"results_openwatch_normwear_protocol_{condition}.json", "NormWear-Base (first-party)")]:
        r = _load(fname)
        if r is not None and split in r:
            out[name] = {"clip": r[split]["clip"], "window": r[split]["window"]}
    return out


def pct(v):
    return "–" if v is None else f"{100*v:.2f}%"


def f3(v):
    return "–" if v is None else f"{v:.3f}"


def _get(m, level, key):
    try:
        return m[level][key]
    except (KeyError, TypeError):
        return None


def md_block(condition, split, level):
    rows = collect(condition, split)
    lvl = {"clip": "clip-level (k=3)", "window": "window-level"}[level]
    lines = [f"**{lvl}, {split} split, {'trained with OpenWatch augmentation' if condition=='aug' else 'trained on real recordings only'}**", "",
             "| Model | Acc. ↑ | Bal. Acc. ↑ | Macro-F1 ↑ |", "|---|---|---|---|"]
    best = {}
    for k in ["acc", "bal_acc", "macro_f1"]:
        vals = [(_get(rows.get(n), level, k), n) for n in ROW_ORDER]
        vals = [(v, n) for v, n in vals if v is not None]
        best[k] = max(vals)[1] if vals else None
    for n in ROW_ORDER:
        m = rows.get(n)
        cells = []
        for k, fmt in [("acc", pct), ("bal_acc", pct), ("macro_f1", f3)]:
            s = fmt(_get(m, level, k))
            cells.append(f"**{s}**" if best[k] == n and m is not None else s)
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latex(condition, split, level):
    rows = collect(condition, split)
    lvl = {"clip": "clip-level, $k{=}3$", "window": "window-level"}[level]
    cond = "trained with OpenWatch augmentation" if condition == "aug" else "trained on real recordings only"
    lab = f"tab:openwatch-{condition}-{level}"
    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{Microgesture recognition on OpenWatch under the OpenWatch protocol ({lvl}; {cond}; "
         r"IMU-only; test split). All rows share one contamination-safe posture-stratified subject split, "
         r"the same 1\,s/stride-20 windows (6 per clip), class-weighted CE and val macro-F1 model selection. "
         r"Mix-Token is our reimplementation (no official code); NormWear-Base is the official checkpoint "
         r"with the paper's head recipe. Higher is better.}",
         rf"\label{{{lab}}}", r"\begin{tabular}{lccc}", r"\toprule",
         r"Model & Acc.\ $\uparrow$ & Bal.\ Acc.\ $\uparrow$ & Macro-F1 $\uparrow$ \\", r"\midrule"]
    for n in ROW_ORDER:
        if n == "Ours (MAE)":
            L.append(r"\midrule")
        if n == "Mix-Token (reimpl.)":
            L.append(r"\midrule")
        m = rows.get(n)
        a, b, f = _get(m, level, "acc"), _get(m, level, "bal_acc"), _get(m, level, "macro_f1")
        cells = [("--" if v is None else (f"{100*v:.2f}\\%" if k != "f1" else f"{v:.3f}")) for v, k in [(a, "a"), (b, "b"), (f, "f1")]]
        L.append(f"{n} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--condition", default="both", choices=["aug", "noaug", "both"])
    args = ap.parse_args()
    conds = ["aug", "noaug"] if args.condition == "both" else [args.condition]

    md = ["# Table 6 (new configuration): Microgesture recognition on OpenWatch", "",
          "OpenWatch paper protocol (arXiv:2605.04791): 1s/100-sample windows, stride 20, 6 windows per clip "
          "(centered 2.0s sub-interval), clip label = k=3 consecutive-run else majority. One contamination-safe "
          "posture-stratified split for every row (33/5/5 canonical identities; test = 87 clips / 522 windows). "
          "IMU-only. Inverse-frequency class-weighted CE; model selection on val window macro-F1. "
          "Bold = column best. `–` = run not finished (see job table in EXPERIMENTS.md §5f).", ""]
    for c in conds:
        md += [f"## {'Trained with OpenWatch augmentation (paper headline condition)' if c=='aug' else 'Trained on real recordings only'}", ""]
        for level in ["clip", "window"]:
            md += [md_block(c, args.split, level), ""]
    md += ["## LaTeX", ""]
    for c in conds:
        for level in ["clip", "window"]:
            md += ["```latex", latex(c, args.split, level), "```", ""]
    md += ["## Cited (Bonazzi et al. Table 2 — their split, their run; context only)", "",
           "| Model | Window Acc | Window F1 | Clip Acc (k=3) | Clip F1 |", "|---|---|---|---|---|",
           "| Mix-Token (aug) | 90.8% | 0.768 | 95.5% | 0.903 |",
           "| Apple-CNN (aug) | 77.4% | 0.576 | 87.9% | 0.756 |",
           "| NormWear-Base (aug; possibly +PPG) | 68.2% | 0.430 | 78.8% | 0.565 |",
           "| NormWear-Base* (no-aug, IMU-only) | 37.0% | 0.340 | 69.0% | 0.620 |", ""]
    text = "\n".join(md)
    open(OUT, "w").write(text)
    print(text)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
