#!/usr/bin/env python3
"""Paper Table 2: the IMU-Bench main results, generated from bench/results (never hand-typed).

    python analysis/paper_tables/make_main_table.py [out.tex]

Two panels (frozen probe, fine-tuning), ten tasks + the activity-recognition mean, one row per
self-supervised objective and MAE ablation, the training-from-scratch reference in both panels and,
in the probe panel, two controls that separate the encoder from the probe decoder:
a random-initialised frozen encoder and a 0-layer encoder (raw input + frozen random patch
projection), both read by the same decoders as every other row.
Best self-supervised value per column is bold. Pending cells print as "--" and are listed on stderr.
Column set and encoder ids mirror bench/leaderboard_paper.py so paper and leaderboard cannot diverge.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from bench.leaderboard import collect                       # noqa: E402
from bench.leaderboard_paper import COLS, CONTROLLED, EXTRA_COLS, HAR3  # noqa: E402
from bench.registry_io import load_tasks                    # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "/gpfs/home/vbc1591/paper_iclr/Sections/tab_main_results.tex"
SHORT = {"arm_pose_nymeria": ("Arm", "cm"), "full_body_pose_nymeria": ("Body", "mm"),
         "odometry_nymeria_local": ("Odom.", "m"), "microgesture_openwatch": ("Gest.", "F1"),
         "language_humanml3d": ("Text", "R@1"), "pd_pads": ("PD", "win F1"), "har_hhar": ("HHAR", "F1"),
         "har_mhealth": ("MHLTH", "F1"), "har_prism": ("PRISM", "F1"), "har_userwatch": ("TrHAR", "F1")}
CONTROLS = [("control_random", "Random frozen encoder"), ("control_raw", "Raw input, 0 layers")]
# Released pretrained models run through the benchmark unchanged. They share neither our backbone size nor our corpus,
# so they are printed below the controlled block, marked with a dagger, and never compete for the bold.
PRETRAINED_EXT = [("normwear_base", r"NormWear$^{\ddagger}$")]
SCALE = {"microgesture_openwatch": 100, "pd_pads": 100, "har_hhar": 100, "har_mhealth": 100, "har_prism": 100,
         "har_userwatch": 100}                                  # F1 shown in %
DIGITS = {"arm_pose_nymeria": 2, "full_body_pose_nymeria": 1, "odometry_nymeria_local": 3, "language_humanml3d": 1}

tasks = load_tasks()["tasks"]
cells = collect()
missing = []


# Column = (task, metric key or None for the task's primary). EXTRA_COLS adds a second column
# right after the task it belongs to.
CSPEC = []
for _t, _s, _m in COLS:
    CSPEC.append((_t, None, SHORT[_t][0], SHORT[_t][1]))
    if _t in EXTRA_COLS:
        _k, _short, _lab = EXTRA_COLS[_t]
        CSPEC.append((_t, _k, _short, _lab))


def val(task, enc, proto, key=None):
    c = cells.get((task, enc, proto))
    if c is None or not c.ok:
        missing.append(f"{task}/{enc}/{proto}")
        return None
    if key is None:
        v = c.primary()[0]
    else:
        vs = [r.metrics.get(key) for r in c.ok if r.metrics.get(key) is not None]
        v = sum(vs) / len(vs) if vs else None
    if v is None:
        missing.append(f"{task}/{enc}/{proto}:{key}")
        return None
    return v * SCALE.get(task, 1)


def hib(task, key=None):
    m = tasks[task]["metrics"]
    return bool(m[key or m["primary"]].get("higher_is_better", True))


def fmt(task, v):
    return "--" if v is None else f"{v:.{DIGITS.get(task, 1)}f}"


def row_values(enc, proto):
    vals = [val(t, enc, proto, k) for t, k, _, _ in CSPEC]
    idx = {t: i for i, (t, k, _, _) in enumerate(CSPEC) if k is None}
    har = [vals[idx[h]] for h in HAR3]
    vals.append(sum(har) / len(har) if all(v is not None for v in har) else None)
    return vals


cols = [(t, k) for t, k, _, _ in CSPEC] + [("har_mean", None)]
tasks["har_mean"] = {"metrics": {"primary": "m", "m": {"higher_is_better": True}}}
DIGITS["har_mean"] = 1


def panel(proto, title):
    rows = [(lab, grp, row_values(enc, proto)) for enc, lab, grp, _ in CONTROLLED]
    best = {}
    for j, (t, k) in enumerate(cols):
        # Only the task's PRIMARY metric is ranked. A secondary column (k is not None) is
        # context, and bolding a best in it invites exactly the misreading that made us move
        # PD off its subject metric: two columns of the same task each crowning a different
        # encoder, with no statement of which one the benchmark actually ranks on.
        if k is not None:
            continue
        vs = [r[2][j] for r in rows if r[1] == "ssl" and r[2][j] is not None]
        if vs:
            best[j] = max(vs) if hib(t, k) else min(vs)
    out = [rf"\multicolumn{{{len(cols) + 1}}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]
    prev = None
    for lab, grp, vs in rows:
        if prev == "ssl" and grp != "ssl":
            out.append(r"\cmidrule(l){1-" + str(len(cols) + 1) + "}")
        cells_ = [(r"\textbf{" + fmt(cols[j][0], v) + "}") if (grp == "ssl" and v is not None and best.get(j) == v)
                  else fmt(cols[j][0], v) for j, v in enumerate(vs)]
        out.append(lab + " & " + " & ".join(cells_) + r" \\")
        prev = grp
    if PRETRAINED_EXT:
        out.append(r"\cmidrule(l){1-" + str(len(cols) + 1) + "}")
        for enc, lab in PRETRAINED_EXT:
            out.append(lab + " & " + " & ".join(fmt(cols[j][0], v) for j, v in enumerate(row_values(enc, proto))) + r" \\")
    if proto == "probe":
        out.append(r"\cmidrule(l){1-" + str(len(cols) + 1) + "}")
        for enc, lab in CONTROLS:
            out.append(lab + " & " + " & ".join(fmt(cols[j][0], v) for j, v in enumerate(row_values(enc, "probe"))) + r" \\")
    return out


lines = [r"% Generated by IMU-SSL/analysis/paper_tables/make_main_table.py from bench/results -- do not edit by hand.",
         r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3.2pt}",
         r"\caption{IMU-Bench results: the primary metric of every task under the frozen-probe and fine-tuning protocols, with "
         r"training from scratch and two controls (a never-trained frozen encoder, and raw input through a frozen random patch "
         r"projection). The rows above the rule share the backbone size and the pretraining corpus, and the plain-attention MAE row "
         r"isolates our attention pattern; $\ddagger$ marks a released pretrained model run through the benchmark unchanged, which "
         # A fairness disclosure, not a footnote: NormWear is far slower per step, so three of its cells hit a
         # 15,000-step ceiling instead of the benchmark's early stopping, and on both odometry cells its best
         # checkpoint is at step 14,000 of 15,000 -- it was still improving when the budget ran out.
         r"shares neither (NormWear: 136\,M parameters and its own corpus). Three NormWear cells --- full-body "
         r"fine-tuning and both odometry cells --- ran under a 15{,}000-step cap rather than the benchmark's "
         r"early-stopping rule, because that model is far slower per step; on both odometry cells its best "
         r"checkpoint falls at step 14{,}000, so those two values are not converged. "
         r"\citep{bradshaw2026transfhar}. Bold: best value among the six pretraining objectives; the two MAE ablation rows "
         r"and the $\ddagger$ row are listed for reference and do not compete for it, and each beats the bolded value in a "
         r"few columns. One training seed per cell; Appendix~\ref{app:ci} describes how evaluation uncertainty is "
         r"quantified and quotes the intervals the claims rest on.}",
         r"\label{tab:main_results}",
         r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
         r"\begin{tabular}{@{}l" + "r" * len(cols) + r"@{}}", r"\toprule",
         " & " + " & ".join(sh for _, _, sh, _ in CSPEC) + r" & HAR \\",
         " & " + " & ".join(ml + (r"$\uparrow$" if hib(t, k) else r"$\downarrow$") for t, k, _, ml in CSPEC)
         + r" & mean$\uparrow$ \\", r"\midrule"]
lines += panel("probe", "Frozen probe")
lines.append(r"\midrule")
lines += panel("finetune", "Fine-tune")
lines.append(r"\midrule")
lines.append("Scratch & " + " & ".join(fmt(cols[j][0], v) for j, v in enumerate(row_values("scratch", "scratch"))) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]
open(OUT, "w").write("\n".join(lines))

# ---- tidy export for paper_results/tab2_main_results -------------------------------------------
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "paper_results"))
from paper_io import write_csv  # noqa: E402

REPO_ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
long_rows = []
groups = [(e, lab, grp, p) for e, lab, grp, _ in CONTROLLED for p in ("probe", "finetune")] + \
         [(e, lab, "control", "probe") for e, lab in CONTROLS] + [("scratch", "Scratch", "scratch", "scratch")]
for enc, lab, grp, proto in groups:
    for t, tlab, mlab in COLS:
        c = cells.get((t, enc, proto))
        raw = c.primary()[0] if c else None
        f_ = os.path.join(REPO_, "bench", "results", t, f"{enc}__{proto}__s42.json")
        m = tasks[t]["metrics"]
        long_rows.append([t, tlab, enc, lab, grp, proto, m["primary"], mlab, raw,
                          int(bool(m[m["primary"]].get("higher_is_better", True))),
                          "ok" if raw is not None else "pending",
                          os.path.relpath(f_, REPO_) if os.path.exists(f_) else ""])
write_csv("tab2_main_results", "main_results_long.csv",
          ["task", "task_label", "encoder", "encoder_label", "group", "protocol", "metric", "metric_label",
           "value", "higher_is_better", "status", "result_file"], long_rows)
for proto in ("probe", "finetune", "scratch"):
    encs = [(e, l) for e, l, g, p in groups if p == proto]
    wide = []
    for e, l in encs:
        vals = [next((r[8] for r in long_rows if r[0] == t and r[2] == e and r[5] == proto), None) for t, _, _ in COLS]
        har = [vals[[t for t, _, _ in COLS].index(h)] for h in HAR3]
        wide.append([e, l] + vals + [sum(har) / len(har) if all(v is not None for v in har) else None])
    write_csv("tab2_main_results", f"main_results_{proto}_wide.csv",
              ["encoder", "encoder_label"] + [t for t, _, _ in COLS] + ["har_mean"], wide)
print(f"wrote {OUT}; {len(set(missing))} pending cells", file=sys.stderr)
for m in sorted(set(missing)):
    print("   pending:", m, file=sys.stderr)
