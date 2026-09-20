#!/usr/bin/env python3
"""One results table per IMU-Bench task, with every metric that task reports.

    python analysis/paper_tables/make_task_tables.py

Writes <paper>/Sections/tab_task_<id>.tex and the numbers behind them to
paper_results/tab2_main_results/per_task_all_metrics.csv. Rows: the self-supervised encoders and the
MAE ablations under both protocols, the two decoder controls (frozen probe) and training from
scratch. Columns: the task card's metrics, primary first. Pending cells print "--".

The four activity-recognition sets share one metric set, so they share one table (tab:task_har) with
a column group per dataset and the mean macro-F1 that Section~\\ref{sec:results} quotes; every other
task gets its own. Replaces the single wide results table (make_main_table.py still writes it for
the appendix summary and the leaderboard CSV).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from bench.leaderboard import collect                       # noqa: E402
from bench.leaderboard_paper import COLS, CONTROLLED, HAR3  # noqa: E402
from bench.registry_io import load_encoders, load_tasks     # noqa: E402

PAPER = "/gpfs/home/vbc1591/paper_iclr/Sections"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONTROLS = [("control_random", "Random frozen encoder"), ("control_raw", "Raw input, 0 layers")]
PCT = {"microgesture_openwatch", "pd_pads", "har_hhar", "har_mhealth", "har_prism", "har_userwatch"}
SKIP = {"n_classes_predicted"}
HAR_TASKS = [t for t, _, _ in COLS if t.startswith("har_")]
HAR_SHORT = {"har_hhar": "HHAR", "har_mhealth": "MHEALTH", "har_prism": "PRISM", "har_userwatch": "User study"}
# Tasks whose metric labels repeat a qualifier (gallery size, direction) get a grouped header instead,
# so the column heads stay narrow enough for the text width: (group title or None, [keys], [short labels]).
GROUPED = {"pd_pads": [
    ("Per subject", ["subject_macro_f1", "subject_bal_acc"], ["macro-F1", "bal-acc"]),
    ("Per window", ["window_bal_acc", "window_macro_f1", "window_acc"], ["bal-acc", "macro-F1", "acc"])],
    "language_humanml3d": [
    (None, ["r1_g100"], ["R@1 mean"]),
    ("Gallery of 100", ["r1_i2t_g100", "r1_t2i_g100", "mrr_i2t_g100", "mrr_t2i_g100"],
     ["i2t R@1", "t2i R@1", "i2t MRR", "t2i MRR"]),
    ("Full gallery", ["r1_i2t_full", "r1_t2i_full", "mrr_i2t_full", "mrr_t2i_full"],
     ["i2t R@1", "t2i R@1", "i2t MRR", "t2i MRR"])]}
tasks = load_tasks()["tasks"]
encoders = load_encoders()
cells = collect()
# Published systems built for one task. They bring their own inputs, output rates, auxiliary
# components and training recipes, so they go in a separate block and are never bolded: the bold
# marks the best controlled encoder, and these are not controlled.
# tex(): a published system's name can carry "&" or "_", which would otherwise break the row.
EXPERTS = [(e, c.get("name", e)) for e, c in encoders.items() if (c or {}).get("group") == "external"]
# Published systems evaluated under a sibling task's protocol: (sibling task, encoders, row suffix).
# The wrist-frame odometry task has no external systems of its own; TLIO and Air-IO were scored under
# the world-frame protocol on the same 73,515 windows and 33-point trajectories.
EXPERT_ALIAS = {"odometry_nymeria_local": ("odometry_nymeria_worldframe", ["tlio", "airio"], " (world-frame protocol)")}
# Caption note once a task's systems have been run under the task's own protocol (replaces EXPERT_NOTE).
EXPERT_DIRECT_NOTE = {
    "odometry_nymeria_local": "TLIO and Air-IO are retrained on the benchmark's Nymeria training windows for this "
                              "protocol and scored by the same wrist-frame reconstruction on the same 73,515 windows: "
                              "Air-IO predicts per-step velocity in the sensor frame, its native target, and TLIO, whose "
                              "network emits one vector per window, the window-mean wrist-frame velocity.",
}
# Caveats that belong in the caption of the task they apply to.
EXPERT_NOTE = {
    "arm_pose_nymeria": "Our arm-pose metric supplies ground-truth ancestor joints, which a full-body "
                        "system such as MobilePoser must predict itself.",
    "odometry_nymeria_local": "TLIO and Air-IO predict a world-frame trajectory and were scored under the "
                              "benchmark's world-frame protocol on the same 73,515 windows and 33-point "
                              "trajectories; that protocol gives systematically lower error than the wrist-frame "
                              "one (the same frozen MAE probe scores 0.148\\,m under it against 0.200\\,m here), "
                              "so their rows are a reference point, not a ranking.",
    "language_humanml3d": "The official UniMTS checkpoint is left out because it was pretrained on "
                          "HumanML3D; the row is its wrist-only reimplementation trained from scratch.",
}


def tex(t):
    """Unicode that appears in task subtitles -> LaTeX."""
    for a, b in (("↔", "$\\leftrightarrow$"), ("→", "$\\rightarrow$"), ("°", "$^\\circ$"),
                 ("×", "$\\times$"), ("%", "\\%"), ("&", "\\&"), ("_", "\\_")):
        t = t.replace(a, b)
    return t


def metric_keys(task):
    m = tasks[task]["metrics"]
    return [m["primary"]] + [k for k in m if k not in ("primary", m["primary"]) and not k.startswith("val_")
                             and k not in SKIP]


def pct(task, k):
    return task in PCT and ("f1" in k or "acc" in k)


def label(task, k, short=False):
    m = tasks[task]["metrics"][k]
    lab = m.get("label", k.replace("_", " "))
    lab = lab.replace("%", "\\%").replace("°", "$^\\circ$")
    if pct(task, k) and not short:
        lab += " (\\%)"
    return lab + (r"$\uparrow$" if m.get("higher_is_better", True) else r"$\downarrow$")


def value(task, enc, proto, k):
    c = cells.get((task, enc, proto))
    if not c or not c.ok:
        return None
    vals = [r.metrics.get(k) for r in c.ok if r.metrics.get(k) is not None]
    if not vals:
        return None
    v = sum(vals) / len(vals)
    return v * 100 if pct(task, k) else v


def mean_value(enc, proto, k):
    """Mean over the four activity sets, as Section 4 reports it; None unless all four are in."""
    vs = [value(t, enc, proto, k) for t in HAR_TASKS]
    return sum(vs) / len(vs) if all(v is not None for v in vs) else None


def fmt(task, k, v):
    if v is None:
        return "--"
    return ("{:.1f}" if pct(task, k) else tasks[task]["metrics"][k].get("fmt", "{:.3f}")).format(v)


def block(columns, rows, bold_group=None):
    """One row per (encoder, label, protocol); `columns` is a list of (task, metric, getter)."""
    best = {}
    if bold_group:
        for i, (task, k, get) in enumerate(columns):
            vs = [get(e, p) for e, _, p in rows if e in bold_group]
            vs = [v for v in vs if v is not None]
            if vs:
                best[i] = max(vs) if tasks[task]["metrics"][k].get("higher_is_better", True) else min(vs)
    out = []
    for enc, lab, proto in rows:
        cs = []
        for i, (task, k, get) in enumerate(columns):
            v = value(enc[1], enc[2], "external", k) if isinstance(enc, tuple) else get(enc, proto)
            s = fmt(task, k, v)
            if bold_group and enc in bold_group and v is not None and best.get(i) == v:
                s = r"\textbf{" + s + "}"
            cs.append(s)
        out.append(lab + " & " + " & ".join(cs) + r" \\")
    return out


def write_table(table_id, columns, header_lines, caption, wide):
    ssl = {e for e, _, g, _ in CONTROLLED if g == "ssl"}
    probe = [(e, lab, "probe") for e, lab, _, _ in CONTROLLED]
    ft = [(e, lab, "finetune") for e, lab, _, _ in CONTROLLED]
    ctrl = [(e, lab, "probe") for e, lab in CONTROLS]
    n = len(columns)
    env, size = ("table*", r"\scriptsize") if wide else ("table", r"\small")
    # Shrink-only-if-needed on every table, not just the wide ones: a long row label (a published
    # system's name) can push a narrow table past the text block too.
    fit = r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%"
    lines = ["% Generated by IMU-SSL/analysis/paper_tables/make_task_tables.py -- do not edit by hand.",
             rf"\begin{{{env}}}[t]", r"\centering", size,
             r"\setlength{\tabcolsep}{" + ("2.5pt" if n > 8 else "4pt") + "}",
             rf"\caption{{{caption}}}", rf"\label{{tab:task_{table_id}}}"] + ([fit] if fit else []) + \
            [r"\begin{tabular}{@{}l" + "r" * n + r"@{}}", r"\toprule"] + header_lines + \
            [r"\midrule", rf"\multicolumn{{{n + 1}}}{{@{{}}l}}{{\emph{{Frozen probe}}}} \\"]
    lines += block(columns, probe, ssl)
    lines += [r"\cmidrule(l){1-" + str(n + 1) + "}"] + block(columns, ctrl)
    lines += [r"\midrule", rf"\multicolumn{{{n + 1}}}{{@{{}}l}}{{\emph{{Fine-tune}}}} \\"]
    lines += block(columns, ft, ssl)
    lines += [r"\midrule"] + block(columns, [("scratch", "Scratch", "scratch")])
    expert = [(e, tex(lab), "external") for e, lab in EXPERTS
              if any(get(e, "external") is not None for _, _, get in columns)]
    if table_id in EXPERT_ALIAS and not expert:      # sibling-protocol rows only while the task has none of its own
        sib, encs, suffix = EXPERT_ALIAS[table_id]
        names = dict(EXPERTS)
        for e in encs:
            if cells.get((sib, e, "external")) and cells[(sib, e, "external")].ok:
                expert.append((("alias", sib, e), tex(names.get(e, e)) + suffix, "external"))
    if expert:
        lines += [r"\midrule",
                  rf"\multicolumn{{{n + 1}}}{{@{{}}l}}{{\emph{{Task-specific systems}} (not ranked)}} \\"]
        lines += block(columns, expert)          # no bold_group: these are not controlled
    lines += [r"\bottomrule", r"\end{tabular}"] + (["}"] if fit else []) + [rf"\end{{{env}}}", ""]
    open(os.path.join(PAPER, f"tab_task_{table_id}.tex"), "w").write("\n".join(lines))
    return probe + ctrl + ft + [("scratch", "Scratch", "scratch")] + expert


COMMON = (r"all encoders share the backbone, the pretraining corpus and this task's decoder. Bold: best "
          r"self-supervised value per metric and protocol; one training seed per cell. Task-specific "
          r"systems are published models for this task alone, with their own inputs, output rates and "
          r"training recipes; they are listed for orientation and are not ranked against the controlled "
          r"encoders (Appendix~\ref{app:tasks}).")
csv_rows, written = [], []

# ---- the six single-task tables --------------------------------------------------------------------
for task, short, _ in COLS:
    if task in HAR_TASKS:
        continue
    keys = metric_keys(task)
    columns = [(task, k, (lambda k: lambda e, p: value(task, e, p, k))(k)) for k in keys]
    card = tasks[task]
    if task in GROUPED:
        groups = GROUPED[task]
        assert [k for _, ks, _ in groups for k in ks] == keys, f"{task}: GROUPED does not cover its metrics"
        top, rule, col = [], [], 1
        for title, ks, _ in groups:
            top.append(r"\multicolumn{" + str(len(ks)) + r"}{c}{" + (title or "") + "}")
            if title:
                rule.append(r"\cmidrule(lr){" + f"{col + 1}-{col + len(ks)}" + "}")
            col += len(ks)
        up = r"$\uparrow$" if tasks[task]["metrics"][keys[0]].get("higher_is_better", True) else r"$\downarrow$"
        header = [" & " + " & ".join(top) + r" \\", "".join(rule),
                  "Encoder & " + " & ".join(lab + up for _, _, labs in groups for lab in labs) + r" \\"]
    else:
        header = ["Encoder & " + " & ".join(label(task, k) for k in keys) + r" \\"]
    # Grouped headers drop the per-column "(%)", so say it once in the caption instead.
    unit = " All values in \\%." if (task in GROUPED and any(pct(task, k) for k in keys)) else ""
    note = EXPERT_NOTE.get(task, "")
    if task in EXPERT_DIRECT_NOTE and any(cells.get((task, e, "external")) and cells[(task, e, "external")].ok for e, _ in EXPERTS):
        note = EXPERT_DIRECT_NOTE[task]           # the task now has systems run under its own protocol
    caption = (f"{short}: {tex(card['subtitle'].strip())}.{unit} Evaluation on the "
               f"{card.get('eval_split', 'val')} split; {COMMON}" + (" " + note if note else ""))
    rows = write_table(task, columns, header, caption, wide=len(keys) >= 6)
    written.append(task)
    for enc, lab, proto in rows:
        if isinstance(enc, tuple):
            csv_rows.append([task, f"{enc[2]}@{enc[1]}", lab, proto] + [value(enc[1], enc[2], "external", k) for k in keys] + [";".join(keys)])
        else:
            csv_rows.append([task, enc, lab, proto] + [value(task, enc, proto, k) for k in keys] + [";".join(keys)])

# ---- the four activity sets in one table -----------------------------------------------------------
KEYS = metric_keys(HAR_TASKS[0])
assert all(metric_keys(t) == KEYS for t in HAR_TASKS), "activity sets no longer share a metric set"
columns = [(t, k, (lambda t, k: lambda e, p: value(t, e, p, k))(t, k)) for t in HAR_TASKS for k in KEYS]
columns.append((HAR_TASKS[0], KEYS[0], (lambda e, p: mean_value(e, p, KEYS[0]))))
header = [" & " + " & ".join(r"\multicolumn{" + str(len(KEYS)) + r"}{c}{" + HAR_SHORT[t] + "}" for t in HAR_TASKS)
          + r" & Mean \\",
          "".join(r"\cmidrule(lr){" + f"{2 + i * len(KEYS)}-{1 + (i + 1) * len(KEYS)}" + "}"
                  for i in range(len(HAR_TASKS))),
          "Encoder & " + " & ".join(label(t, k, short=True) for t in HAR_TASKS for k in KEYS)
          + " & " + label(HAR_TASKS[0], KEYS[0], short=True) + r" \\"]
caption = ("Activity recognition, all four sets (\\%). HHAR (6 classes) and MHEALTH (12) are subject-disjoint "
           "with two held-out subjects each, PRISM (5 global ADL classes) with six; the user-study set (60) is "
           "split in time, so it "
           "measures recognition later in the same recording rather than generalisation to new people. "
           "\\emph{Mean} is the mean macro-F1 over the four sets quoted in Section~\\ref{sec:results}. "
           "Evaluation on the validation split; " + COMMON)
rows = write_table("har", columns, header, caption, wide=True)
written.append("har")
for enc, lab, proto in rows:
    for t in HAR_TASKS:
        csv_rows.append([t, enc, lab, proto] + [value(t, enc, proto, k) for k in KEYS] + [";".join(KEYS)])
    csv_rows.append(["har_mean", enc, lab, proto, mean_value(enc, proto, KEYS[0])] + [None] * (len(KEYS) - 1)
                    + [KEYS[0]])

sys.path.insert(0, os.path.join(REPO, "paper_results"))
from paper_io import write_csv  # noqa: E402
maxk = max(len(metric_keys(t)) for t, _, _ in COLS)
write_csv("tab2_main_results", "per_task_all_metrics.csv",
          ["task", "encoder", "encoder_label", "protocol"] + [f"m{i + 1}" for i in range(maxk)] + ["metric_keys"],
          [r[:4] + r[4:-1] + [None] * (maxk - (len(r) - 5)) + [r[-1]] for r in csv_rows])
print(f"wrote {len(written)} tables to {PAPER}/tab_task_*.tex: {', '.join(written)}")
