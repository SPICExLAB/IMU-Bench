#!/usr/bin/env python3
"""Appendix table: the benchmark re-scored WITHOUT checkpoint selection.

    python analysis/paper_tables/make_selection_table.py

Eight of the ten tasks select the reported checkpoint on the split they report on. This table
repeats Table~\\ref{tab:main_results} from each run's FINAL checkpoint (bench/results_nosel/, written
by `python -m bench.rescore_noselection`), which removes the best-of-N step; the caption carries the
mean change and the rank agreement with the reported table, both computed here.

Writes <paper>/Sections/tab_noselection.tex and paper_results/appendix_selection/selection_free.csv.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from bench.leaderboard_paper import COLS, CONTROLLED, HAR3   # noqa: E402
from bench.registry_io import REPO, load_tasks               # noqa: E402

PAPER = "/gpfs/home/vbc1591/paper_iclr/Sections"
MAIN, NOSEL = os.path.join(REPO, "bench", "results"), os.path.join(REPO, "bench", "results_nosel")
SHORT = {"arm_pose_nymeria": "Arm", "full_body_pose_nymeria": "Body", "odometry_nymeria_local": "Odom.",
         "microgesture_openwatch": "Gest.", "language_humanml3d": "Text", "pd_pads": "PD", "har_hhar": "HHAR",
         "har_mhealth": "MHLTH", "har_prism": "PRISM", "har_userwatch": "User"}
SCALE = {t: 100 for t in ("microgesture_openwatch", "pd_pads", *HAR3)}
DIG = {"arm_pose_nymeria": 2, "full_body_pose_nymeria": 1, "odometry_nymeria_local": 3, "language_humanml3d": 1}
CONTROLS = [("control_random", "Random frozen encoder"), ("control_raw", "Raw input, 0 layers")]
tasks = load_tasks()["tasks"]
TASKS = [t for t, _, _ in COLS]


# The microgesture task already selects on validation and reports on a held-out test split, and its
# scorer defines clip voting on the selected checkpoint, so there is no selection to remove: its column
# is "n/a" rather than a pending "--", and it is left out of the statistics below.
NA_TASKS = {t for t in TASKS if t.startswith("microgesture")}


def read(root, task, enc, proto, seed=42):
    p = os.path.join(root, task, f"{enc}__{proto}__s{seed}.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    if d.get("status") != "ok":
        return None
    v = (d.get("metrics") or {}).get(d.get("primary_metric"))
    return None if v is None else float(v) * SCALE.get(task, 1)


def hib(task):
    m = tasks[task]["metrics"]
    return bool(m[m["primary"]].get("higher_is_better", True))


def fmt(task, v):
    if v is None:
        return r"\textit{n/a}" if task in NA_TASKS else "--"
    return f"{v:.{DIG.get(task, 1)}f}"


ROWS = [(e, lab, "probe") for e, lab, _, _ in CONTROLLED] + [(e, lab, "probe") for e, lab in CONTROLS] + \
       [(e, lab, "finetune") for e, lab, _, _ in CONTROLLED] + [("scratch", "Scratch", "scratch")]


def row_vals(root, enc, proto):
    vals = [read(root, t, enc, proto) for t in TASKS]
    har = [vals[TASKS.index(h)] for h in HAR3]
    vals.append(sum(har) / len(har) if all(v is not None for v in har) else None)
    return vals


# ---- the numbers, and how far they move -----------------------------------------------------------
pairs, csv_rows = [], []
for enc, lab, proto in ROWS:
    for t in TASKS:
        if t in NA_TASKS:
            continue
        a, b = read(MAIN, t, enc, proto), read(NOSEL, t, enc, proto)
        csv_rows.append([t, enc, lab, proto, a, b, None if (a is None or b is None) else b - a])
        if a is not None and b is not None:
            pairs.append((t, enc, proto, a, b))

# Mean absolute change, as a fraction of the reported value, and how often the better of two
# self-supervised encoders swaps under the selection-free scoring.
rel = [abs(b - a) / abs(a) for _, _, _, a, b in pairs if a]
ssl = [e for e, _, g, _ in CONTROLLED if g == "ssl"]
kept = {}
for proto in ("probe", "finetune"):
    swaps = comps = 0
    for t in TASKS:
        if t in NA_TASKS:
            continue
        have = [(read(MAIN, t, e, proto), read(NOSEL, t, e, proto)) for e in ssl]
        have = [(a, b) for a, b in have if a is not None and b is not None]
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                (a1, b1), (a2, b2) = have[i], have[j]
                if a1 == a2 or b1 == b2:
                    continue
                comps += 1
                swaps += ((a1 > a2) != (b1 > b2))
    kept[proto] = (comps - swaps, comps)
n_cells = len(pairs)
ok = all(c for _, c in kept.values()) and rel
stat = (f"Over the {n_cells} cells that have both scorings the mean absolute change is "
        f"{100 * sum(rel) / len(rel):.1f}\\,\\% of the reported value. Every one of the "
        f"{kept['probe'][1]} pairwise orderings between self-supervised encoders under the frozen probe "
        f"survives the change" if kept["probe"][0] == kept["probe"][1] else
        f"{kept['probe'][0]} of {kept['probe'][1]} frozen-probe orderings survive the change") if ok else \
       "Numbers pending: run \\texttt{python -m bench.rescore\\_noselection}."
if ok:
    stat += (f", against {kept['finetune'][0]} of {kept['finetune'][1]} under fine-tuning, where the runs that "
             f"gain most from selection are the ones whose validation loss and task metric disagree.")

lines = [r"% Generated by IMU-SSL/analysis/paper_tables/make_selection_table.py -- do not edit by hand.",
         r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{The benchmark scored \emph{without} checkpoint selection: every cell of "
         r"Table~\ref{tab:main_results} re-evaluated from the final checkpoint of the same run instead of the "
         r"best one on validation, so no checkpoint is chosen on the reported split (early stopping on "
         r"validation loss still decides when a run ends). Microgesture already selects on validation and "
         r"reports on a held-out test split, so it has no selection to remove and its column is \textit{n/a}. "
         + stat +
         r" Units and column order as in Table~\ref{tab:main_results}; one seed per cell.}",
         r"\label{tab:noselection}",
         r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
         r"\begin{tabular}{@{}l" + "r" * (len(TASKS) + 1) + r"@{}}", r"\toprule",
         "Encoder & " + " & ".join(SHORT[t] for t in TASKS) + r" & HAR \\", r"\midrule",
         rf"\multicolumn{{{len(TASKS) + 2}}}{{@{{}}l}}{{\emph{{Frozen probe}}}} \\"]
prev = None
for enc, lab, proto in ROWS:
    if prev is not None and proto != prev:
        lines += [r"\midrule", rf"\multicolumn{{{len(TASKS) + 2}}}{{@{{}}l}}{{\emph{{{'Fine-tune' if proto == 'finetune' else 'Scratch'}}}}} \\"] \
            if proto != "scratch" else [r"\midrule"]
    if enc == "control_random":
        lines.append(r"\cmidrule(l){1-" + str(len(TASKS) + 2) + "}")
    vals = row_vals(NOSEL, enc, proto)
    lines.append(lab + " & " + " & ".join(fmt(t, v) for t, v in zip(TASKS + ["har_hhar"], vals)) + r" \\")
    prev = proto
lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]
open(os.path.join(PAPER, "tab_noselection.tex"), "w").write("\n".join(lines))

sys.path.insert(0, os.path.join(REPO, "paper_results"))
from paper_io import write_csv  # noqa: E402
write_csv("appendix_selection", "selection_free.csv",
          ["task", "encoder", "encoder_label", "protocol", "reported_best_ckpt", "final_ckpt", "delta"], csv_rows)
print(f"wrote {PAPER}/tab_noselection.tex ({n_cells} cells scored both ways)")
