#!/usr/bin/env python3
"""One compact body table: our encoder against the system built for each task.

    python analysis/paper_tables/make_specialists_table.py

Writes <paper>/Sections/tab_specialists.tex. Tasks run across the columns, as in
Table~\\ref{tab:main_results}, with the same short names, units and rounding. Rows: training from
scratch, inrtlWrist frozen and fine-tuned, and the best-scoring published system for each task
retrained under the benchmark's protocol, with a name row underneath that spans the tasks a system
covers. The full per-task tables (every metric, every system) are in the appendix; this table exists
so the body can make the comparison without them.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import analysis.paper_tables.make_task_tables as T   # noqa: E402
from analysis.paper_tables.make_main_table import SHORT, DIGITS, SCALE  # noqa: E402  (same names, units, rounding)
from bench.leaderboard_paper import COLS             # noqa: E402

OURS = "mae_5src"
PAPER = "/gpfs/home/vbc1591/paper_iclr/Sections"
# Column groups, in COLS order: (title, number of tasks). A one-task group prints no rule.
GROUPS = [("Kinematic", 3), ("Semantic", 2), ("Health", 1), ("Activity recognition", 4)]
# Short display names for the system row; the full attributions go in the caption.
SYS_SHORT = {"mobileposer": "MobilePoser", "imuposer": "IMUPoser", "tlio": "TLIO", "airio": "Air-IO",
             "mixtoken": "Mix-Token", "unimts": "UniMTS", "unimts_scratch": "UniMTS", "pads_boss": "BOSS",
             "pads_boss_svm": "BOSS + SVM", "pads_features": "PADS spectral features",
             "deepconvlstm": "DeepConvLSTM", "normwear": "NormWear-Base"}
ALIAS, SHOW, NAMES = T.EXPERT_ALIAS, T.EXPERT_SHOW, dict(T.EXPERTS)


def experts_for(task):
    """Same rule as the per-task tables: a task's own external cells, and the sibling-protocol rows
    only while it has none of its own."""
    encs = SHOW.get(task, [e for e, _ in T.EXPERTS])
    ok = lambda t, e: bool(T.cells.get((t, e, "external"))) and T.cells[(t, e, "external")].ok
    own = [(e, task) for e in encs if ok(task, e)]
    if own or task not in ALIAS:
        return own
    sib, aliased, _ = ALIAS[task]
    return [(e, sib) for e in aliased if ok(sib, e)]


def value(task, enc, proto, scored_on=None):
    k = T.tasks[task]["metrics"]["primary"]
    c = T.cells.get(((scored_on or task), enc, proto))
    if not c or not c.ok:
        return None
    vs = [r.metrics.get(k) for r in c.ok if r.metrics.get(k) is not None]
    return sum(vs) / len(vs) * SCALE.get(task, 1) if vs else None


def best_expert(task):
    hi = T.tasks[task]["metrics"][T.tasks[task]["metrics"]["primary"]].get("higher_is_better", True)
    got = [(v, e) for e, on in experts_for(task) if (v := value(task, e, "external", on)) is not None]
    if not got:
        return None, None
    v, e = (max if hi else min)(got, key=lambda x: x[0])
    return v, SYS_SHORT.get(e, T.tex(NAMES.get(e, e)))


def fmt(task, v):
    return "--" if v is None else f"{v:.{DIGITS.get(task, 1)}f}"


TASKS = [t for t, _, _ in COLS]
n = len(TASKS)
rows = [("Trained from scratch", [value(t, "scratch", "scratch") for t in TASKS], True),
        ("inrtlWrist (Ours), frozen probe", [value(t, OURS, "probe") for t in TASKS], True),
        ("inrtlWrist (Ours), fine-tuned", [value(t, OURS, "finetune") for t in TASKS], True)]
expert = [best_expert(t) for t in TASKS]
rows.append(("Task-specific system", [v for v, _ in expert], False))
missing = [t for t, (v, _) in zip(TASKS, expert) if v is None]

# Bold the best of the three controlled rows per column; the task-specific row is not ranked.
best = []
for j, t in enumerate(TASKS):
    hi = T.tasks[t]["metrics"][T.tasks[t]["metrics"]["primary"]].get("higher_is_better", True)
    vs = [r[1][j] for r in rows if r[2] and r[1][j] is not None]
    best.append((max(vs) if hi else min(vs)) if vs else None)


def line(lab, vals, bold):
    cs = [fmt(t, v) for t, v in zip(TASKS, vals)]
    if bold:
        cs = [r"\textbf{" + c + "}" if v is not None and v == best[j] else c
              for j, (c, v) in enumerate(zip(cs, vals))]
    return lab + " & " + " & ".join(cs) + r" \\"


# The name row spans the consecutive tasks a system covers (MobilePoser over both pose tasks,
# DeepConvLSTM over the four activity sets), which is what makes the row readable at this width.
name_cells, j = [], 0
while j < n:
    k = j
    while k + 1 < n and expert[k + 1][1] == expert[j][1]:
        k += 1
    nm = expert[j][1] or "--"
    span = k - j + 1
    name_cells.append(r"\multicolumn{%d}{c}{\emph{%s}}" % (span, nm) if span > 1 else
                      r"\multicolumn{1}{c}{\emph{%s}}" % nm)
    j = k + 1

grp_top, grp_rule, col = [], [], 2
for title, k in GROUPS:
    grp_top.append(r"\multicolumn{%d}{c}{%s}" % (k, title) if k > 1 else title)
    grp_rule.append(r"\cmidrule(lr){%d-%d}" % (col, col + k - 1))
    col += k
assert sum(k for _, k in GROUPS) == n, "GROUPS must cover every column"

caption = (
    r"\textbf{inrtlWrist against the system built for each task.} Every row is scored on the task's primary metric "
    r"under this benchmark's protocol, with the same short names, units and rounding as Table~\ref{tab:main_results}; "
    r"bold marks the best of the three controlled rows. The task-specific row is the better of the published systems "
    r"available for that task, retrained on our splits: MobilePoser and IMUPoser~\citep{mollyn2023imuposer} for pose, "
    r"TLIO and Air-IO for odometry, Mix-Token for gestures, UniMTS for IMU--text, the PADS authors' BOSS classifier "
    r"\citep{varghese2024pads} and DeepConvLSTM~\citep{deepconvlstm}. These keep their own inputs, recipes and "
    r"auxiliary components, are untuned here and are listed for orientation, not ranked: the two pose systems also "
    r"read wrist orientation, which the controlled encoders never see, while our arm-pose metric supplies the "
    r"ancestor joints they would otherwise predict; TLIO and Air-IO report the epoch with the lowest evaluation ATE; "
    r"and the UniMTS row is a wrist-only reimplementation trained from scratch, whose text-to-IMU direction is at "
    r"chance because of hubness. Parkinson's is ranked on the per-subject metric, the clinically meaningful unit; "
    r"its window metrics are in the task table. Appendix~\ref{app:tasks} gives every metric and "
    r"every system per task (Tables~\ref{tab:task_arm_pose_nymeria}--\ref{tab:task_har}).")

lines = ["% Generated by IMU-SSL/analysis/paper_tables/make_specialists_table.py -- do not edit by hand.",
         r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4.5pt}",
         r"\caption{" + caption + "}", r"\label{tab:specialists}",
         r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
         r"\begin{tabular}{@{}l" + "r" * n + r"@{}}", r"\toprule",
         " & " + " & ".join(grp_top) + r" \\", "".join(grp_rule),
         " & " + " & ".join(SHORT[t][0] for t in TASKS) + r" \\",
         " & " + " & ".join(SHORT[t][1] + (r"$\uparrow$" if T.tasks[t]["metrics"][T.tasks[t]["metrics"]["primary"]]
                                           .get("higher_is_better", True) else r"$\downarrow$") for t in TASKS) + r" \\",
         r"\midrule"]
lines += [line(*r) for r in rows[:3]]
lines += [r"\cmidrule(l){1-%d}" % (n + 1), line(*rows[3]),
          r"\addlinespace[1pt]", r"\quad\emph{system} & " + " & ".join(name_cells) + r" \\",
          r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"]
out = os.path.join(PAPER, "tab_specialists.tex")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", out, f"({n} task columns)")
if missing:
    print("no task-specific system for:", ", ".join(missing))
