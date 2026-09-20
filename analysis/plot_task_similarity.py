#!/usr/bin/env python3
"""Figure: representational task similarity (CKNNA) between the MAE encoder's task-adapted copies.

    python analysis/plot_task_similarity.py [--outdir <paper>/Pictures]

Left  -- behavioural: rows are the task the encoder was fine-tuned on (source), columns the task it
         is then frozen and probed on (target); a cell is the relative change of that probe against the
         unadapted MAE probe on the same target, so 0 means "no better than generic pretraining".
         Diverging colour: blue helps, red hurts, grey is zero. Diagonal is the task itself (blank).
Right -- representational: CKNNA between the pooled embeddings of two adapted encoders on one shared
         set of Nymeria wrist-IMU windows, never looking at task performance. Sequential blue.
Data: paper_results/tab3_task_similarity/task_similarity.csv (make_task_similarity.py) and
analysis/task_similarity/cknna_results.csv (compute_cknna_scratch_encoders.py --encoders ...).
Writes Pictures/task_similarity.{png,pdf} and paper_results/tab3_task_similarity/figure_matrices.csv.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS_CSV = os.path.join(REPO, "paper_results", "tab3_task_similarity", "task_similarity.csv")
CKNNA_CSV = os.path.join(REPO, "analysis", "task_similarity", "cknna_results.csv")
SCRATCH_CSV = os.path.join(REPO, "analysis", "task_similarity_scratch", "cknna_results.csv")   # optional

# Palette (dataviz reference instance): sequential blue ramp, diverging blue<->red with grey midpoint.
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RED = ["#fbe2e2", "#f4b3b3", "#ec8484", "#e34948", "#c23131", "#961f1f"]
GREY_MID = "#f0efec"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

# Display order and name mapping across the two data files.
TASKS = ["Full-body pose", "Odometry", "Microgesture", "Parkinson's", "PRISM activity", "IMU--text"]
SHORT = {"Full-body pose": "Full-body\npose", "Odometry": "Odometry", "Microgesture": "Micro-\ngesture",
         "Parkinson's": "Parkinson's", "PRISM activity": "PRISM\nactivity", "IMU--text": "IMU–text"}
CK_NAME = {"Full-body pose": "Full-body pose", "Odometry": "Odometry", "Microgesture": "Microgesture",
           "Parkinson's": "Parkinson's", "PRISM activity": "PRISM", "IMU--text": "IMU-text"}
CK_EXTRA = [("Arm pose", "Arm pose"), ("MAE (no task)", "MAE\n(no task)")]   # CKNNA-only rows/cols


def read_pairs():
    """Directional transfer gains: gain[src][tgt] = rel(src-adapted encoder probed on tgt)."""
    gain = {}
    with open(PAIRS_CSV) as f:
        for r in csv.DictReader(f):
            a, b = r["task_a"], r["task_b"]
            if r["rel_a_to_b"] not in ("", "None"):
                gain[(a, b)] = 100 * float(r["rel_a_to_b"])
            if r["rel_b_to_a"] not in ("", "None"):
                gain[(b, a)] = 100 * float(r["rel_b_to_a"])
    return gain


def read_cknna(path=CKNNA_CSV):
    ck = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            ck[frozenset((r["task_a"], r["task_b"]))] = float(r["cknna"])
    return ck


def draw_matrix(ax, M, labels, cmap, norm, fmt, title, pending=None, cbar_label=""):
    n = len(labels)
    ax.set_facecolor("white")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, aspect="equal")
    # cell borders as a 2px surface gap, diagonal and pending cells hatched
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc="white", ec=GRID, lw=1.2, hatch="////",
                                           zorder=3))
                continue
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc="white", ec=GRID, lw=1.2, zorder=3))
                ax.text(j, i, "pending" if (pending and (i, j) in pending) else "–", ha="center", va="center",
                        fontsize=7, color=MUTED, zorder=4)
                continue
            rgba = cmap(norm(v))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8,
                    color="white" if lum < 0.45 else INK, zorder=4)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([l.replace("-\n", "").replace("\n", " ") for l in labels], fontsize=8, color=INK2,
                       rotation=32, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels, fontsize=8, color=INK2)
    ax.tick_params(length=0, pad=4)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True); ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7, color=GRID, labelcolor=INK2, length=2)
    cb.set_label(cbar_label, fontsize=8, color=INK2)
    ax.set_anchor("N"); cb.ax.set_anchor("N")     # both panels hang from the top so their titles align
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="/gpfs/home/vbc1591/paper_iclr/Pictures")
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()

    gain, ck = read_pairs(), read_cknna()

    # ---- left: directional transfer (source rows, target columns) ---------------------------------
    n = len(TASKS)
    T = np.full((n, n), np.nan)
    pending = set()
    for i, s in enumerate(TASKS):
        for j, t in enumerate(TASKS):
            if i == j:
                continue
            if (s, t) in gain:
                T[i, j] = gain[(s, t)]
            else:
                pending.add((i, j))
    lim = np.nanmax(np.abs(T)) if np.isfinite(T).any() else 30
    lim = float(np.ceil(lim / 10) * 10)
    div = LinearSegmentedColormap.from_list("div", list(reversed(RED[:5])) + [GREY_MID] + BLUE[1:6])
    dnorm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

    # ---- right: CKNNA over adapted encoders (+ arm pose, + unadapted MAE) ---------------------------
    ck_labels = [(t, SHORT[t], CK_NAME[t]) for t in TASKS] + [(k, lab, k) for k, lab in CK_EXTRA]
    m = len(ck_labels)
    C = np.full((m, m), np.nan)
    for i, (_, _, ci) in enumerate(ck_labels):
        for j, (_, _, cj) in enumerate(ck_labels):
            if i != j and frozenset((ci, cj)) in ck:
                C[i, j] = ck[frozenset((ci, cj))]
    seq = LinearSegmentedColormap.from_list("seq", BLUE)

    # Right panel (when its run exists): encoders trained FROM SCRATCH on each task -- no shared
    # pretraining, so any structure here is what the tasks alone impose on the same architecture.
    SC_ORDER = ["Full-body pose", "Odometry", "Microgesture", "Parkinson's", "PRISM", "IMU-text", "Arm pose",
                "HHAR", "MHEALTH", "User study"]
    SC_SHORT = {"Full-body pose": "Full-body\npose", "Microgesture": "Micro-\ngesture", "PRISM": "PRISM\nactivity",
                "IMU-text": "IMU–text", "User study": "User\nstudy"}
    S = None
    if os.path.exists(SCRATCH_CSV):
        cks = read_cknna(SCRATCH_CSV)
        present = [t for t in SC_ORDER if any(t in k for k in cks)]
        S = np.full((len(present), len(present)), np.nan)
        for i, ti in enumerate(present):
            for j, tj in enumerate(present):
                if i != j and frozenset((ti, tj)) in cks:
                    S[i, j] = cks[frozenset((ti, tj))]
    vmax = float(np.nanmax([np.nanmax(C)] + ([np.nanmax(S)] if S is not None else [])))
    snorm = plt.Normalize(vmin=0, vmax=vmax)

    if S is None:
        fig, axR = plt.subplots(1, 1, figsize=(6.4, 5.6))
        axes = [axR]
    else:
        fig, (axR, axS) = plt.subplots(1, 2, figsize=(12.6, 5.6),
                                       gridspec_kw=dict(width_ratios=[m, len(present)], wspace=0.5))
        axes = [axR, axS]
    fig.patch.set_facecolor("white")
    draw_matrix(axR, C, [lab for _, lab, _ in ck_labels], seq, snorm, lambda v: f"{v:.2f}",
                "(a) Encoders fine-tuned from the MAE", cbar_label="CKNNA (shared held-out Nymeria windows)")
    if S is not None:
        draw_matrix(axS, S, [SC_SHORT.get(t, t) for t in present], seq, snorm, lambda v: f"{v:.2f}",
                    "(b) Encoders trained from scratch per task", cbar_label="CKNNA (same windows)")

    os.makedirs(a.outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.outdir, f"task_similarity.{ext}"), dpi=a.dpi, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)

    # ---- numbers behind the figure -----------------------------------------------------------------
    import sys
    sys.path.insert(0, os.path.join(REPO, "paper_results"))
    from paper_io import write_csv  # noqa: E402
    rows = [["transfer_pct", s, t, None if np.isnan(T[i, j]) else T[i, j]]     # kept for the record
            for i, s in enumerate(TASKS) for j, t in enumerate(TASKS) if i != j]
    rows += [["cknna", ck_labels[i][0], ck_labels[j][0], None if np.isnan(C[i, j]) else C[i, j]]
             for i in range(m) for j in range(m) if i < j]
    if S is not None:
        rows += [["cknna_scratch", present[i], present[j], None if np.isnan(S[i, j]) else S[i, j]]
                 for i in range(len(present)) for j in range(len(present)) if i < j]
    write_csv("tab3_task_similarity", "figure_matrices.csv", ["panel", "row", "col", "value"], rows)
    print(f"wrote {a.outdir}/task_similarity.png  transfer cells={int(np.isfinite(T).sum())}/{n*(n-1)}  "
          f"cknna pairs={int(np.isfinite(C).sum())//2}/{m*(m-1)//2}")


if __name__ == "__main__":
    main()
