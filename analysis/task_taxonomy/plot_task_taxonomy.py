"""Nested-donut ("pie") inventory of every dataset + downstream task in IMU-SSL.

Inner ring  = the 6 granularity tiers, ordered clockwise from the top seam:
              coarse (one label per session) -> fine (per-frame joint kinematics).
Outer ring  = the 22 individual dataset-task evaluations.
Side panel  = the tier key, plus the 11-source SSL pretraining corpus.

Two layouts are rendered, each in a light and a dark variant:

  v1  tier names written along the inner ring, COARSEST/FINEST markers at the
      seam, single-column pretraining corpus, taller canvas.
  v2  tier numbers only in the ring (names live in the key), two-column corpus,
      more compact canvas.

Outer-ring label direction is set per tier, not per geometric half: tiers 1-2
read outward from the ring, tiers 3-6 read inward, so no label within a tier is
upside down relative to its neighbours.

Sources of truth
----------------
* downstream tasks   : EXPERIMENTS.md sections 1-15, Experiments_table.md,
                       launch_slurm_{pd,pdhc,pddd}_*_pads_*.sh, configs/base_nightbeat.yaml
* pretraining corpus : analysis/mae_scaling_diagnostics/out/01_composition/window_counts.json
                       (train-window budget at the 100% recipe, per-source caps applied)
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrowPatch, Rectangle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
WINDOW_COUNTS = os.path.join(
    REPO, "analysis", "mae_scaling_diagnostics", "out", "01_composition", "window_counts.json"
)

# --------------------------------------------------------------------------
# Theme (validated palette; see dataviz references/palette.md)
# --------------------------------------------------------------------------

THEMES = {
    "light": dict(
        surface="#ffffff",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        hair="#e1e0d9",
        series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
        tint=0.58,   # fraction of surface mixed into the outer-ring fill
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        hair="#2c2c2a",
        series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
        tint=0.45,
    ),
}

# v1 keeps a wider seam so the COARSEST/FINEST markers have room to sit in it.
# "square" is the stripped paper figure: the pie plus a compact tier key, no
# title block and no pretraining-corpus panel.
LAYOUTS = {
    "v1": dict(seam_gap=16.0, ring_label="name", seam_markers=True,
               corpus_cols=1, figsize=(16.8, 10.4), titles=True, key="full",
               ax_box=[-0.030, 0.015, 0.655, 0.885],
               side_box=[0.625, 0.015, 0.365, 0.885],
               xlim=(-1.66, 1.66), ylim=(-1.40, 1.46), hero_fs=44.0),
    "v2": dict(seam_gap=10.0, ring_label="number", seam_markers=False,
               corpus_cols=2, figsize=(16.4, 9.8), titles=True, key="full",
               ax_box=[-0.030, 0.015, 0.655, 0.885],
               side_box=[0.625, 0.015, 0.365, 0.885],
               xlim=(-1.66, 1.66), ylim=(-1.40, 1.46), hero_fs=44.0),
    # Limits are cropped to where labels actually reach (the left-hand tier-6
    # labels run longest), so the axes box carries no dead margin and the saved
    # figure comes out close to square.
    "square": dict(seam_gap=10.0, ring_label="number", seam_markers=False,
                   corpus_cols=0, figsize=(10.9, 8.7), titles=False, key="compact",
                   ax_box=[0.000, 0.012, 0.775, 0.975],
                   side_box=[0.775, 0.130, 0.215, 0.740],
                   xlim=(-1.50, 1.38), ylim=(-1.34, 1.50), hero_fs=34.0),
    # Nothing beside the pie: tier names ride in the ring, and the coarse/fine
    # ends of the axis are marked at the seam they meet across.
    # A wider tier ring buys the arc length tier 1 needs to hold its full name.
    "solo": dict(seam_gap=16.0, ring_label="name", seam_markers=True,
                 corpus_cols=0, figsize=(10.6, 10.0), titles=False, key="none",
                 ax_box=[0.000, 0.000, 1.000, 1.000], side_box=None,
                 xlim=(-1.54, 1.46), ylim=(-1.42, 1.58), hero_fs=36.0,
                 radii=(0.50, 0.74, 0.77, 1.00)),
    # Same pie, but the ring carries no text at all: a curved arrow inside the
    # hole shows the 1 -> 6 direction, and a straight axis under the pie spells
    # the six tiers out left-to-right.
    "arrow": dict(seam_gap=10.0, ring_label="number", seam_markers=False,
                  corpus_cols=0, figsize=(10.0, 10.9), titles=False, key="none",
                  ax_box=[0.000, 0.112, 1.000, 0.885], side_box=None,
                  xlim=(-1.54, 1.46), ylim=(-1.42, 1.46), hero_fs=36.0,
                  radii=(0.50, 0.74, 0.77, 1.00),
                  ring_arrow=True, bottom_axis=[0.030, 0.004, 0.940, 0.104]),
}


def mix(hex_a: str, hex_b: str, t: float) -> tuple:
    """Blend hex_a toward hex_b by t (0 -> a, 1 -> b)."""
    a = np.array(matplotlib.colors.to_rgb(hex_a))
    b = np.array(matplotlib.colors.to_rgb(hex_b))
    return tuple((1.0 - t) * a + t * b)


# --------------------------------------------------------------------------
# The taxonomy: coarse -> fine, by how much motion detail the output demands
# --------------------------------------------------------------------------

TIERS = [
    dict(
        name="Subject-level state",
        rule="one binary label per session",
        detail="A clinical trait, not a movement.",
        tasks=[
            ("PADS", "PD vs. all"),
            ("PADS", "PD vs. healthy ctrl"),
            ("PADS", "PD vs. differential dx"),
        ],
    ),
    dict(
        name="Activity class",
        rule="1-of-K over a 5-10 s window",
        detail="Closed-set HAR: locomotion, household actions.",
        tasks=[
            ("WISDM", "6-class HAR"),
            ("PAMAP2", "HAR"),
            ("UTD-MHAD", "HAR"),
            ("RecoFit", "exercise HAR"),
            ("WHARF", "HAR"),
            ("HHAR", "HAR"),
            ("MHEALTH", "HAR"),
        ],
    ),
    dict(
        name="Open-vocab semantics",
        rule="free text, unbounded label set",
        detail="IMU$\\leftrightarrow$text CLIP: retrieval + zero-shot transfer.",
        tasks=[
            ("HumanML3D", "IMU$\\leftrightarrow$text CLIP"),
            ("Nymeria", "motion narration"),
            ("Nymeria", "atomic action"),
            ("WISDM/UTD/PAMAP2", "zero-shot HAR"),
        ],
    ),
    dict(
        name="Micro-gesture",
        rule="1-of-6 over ~1 s",
        detail="Sub-second discrete wrist gestures.",
        tasks=[
            ("OpenWatch", "6-class microgesture"),
        ],
    ),
    dict(
        name="Physiological signal",
        rule="continuous scalar per window",
        detail="Heart rate from the wrist ballistocardiogram.",
        tasks=[
            ("Nightbeat-DB", "heart rate (bpm)"),
        ],
    ),
    dict(
        name="Dense kinematics",
        rule="3-72 DOF at every 20 ms frame",
        detail="Joint rotation, position, and body trajectory.",
        tasks=[
            ("Nymeria", "arm pose, 5 joints"),
            ("TotalCapture/DIP", "arm pose, 5 joints"),
            ("Nymeria", "full-body pose, 23 j."),
            ("TotalCapture/DIP", "full-body pose, 24 j."),
            ("Nymeria", "odometry, local vel."),
            ("Nymeria", "odometry, world ATE"),
        ],
    ),
]

# Tiers 1-2 label outward; tiers 3-6 label inward. Snapping the flip to this
# tier boundary (~164 deg) rather than the geometric 180 deg keeps every label
# inside a tier reading the same way.
FLIP_FROM_TIER = 2

# Display names + modality for the 11 pretraining sources (from
# analysis/mae_scaling_diagnostics/out/09_dataset_characterization_v3_100pct).
PRETRAIN_META = {
    "nymeria":      ("Nymeria",             "real"),
    "capture24":    ("Capture-24",          "real, acc-only"),
    "realdata":     ("HAR master parquet*", "real"),
    "eatingspeed":  ("EatingSpeed",         "real"),
    "sonic":        ("SONIC / bones-seed",  "synthetic"),
    "ocdetect":     ("OCDetect",            "real"),
    "motionx":      ("Motion-X++",          "synthetic"),
    "humanml3d":    ("HumanML3D",           "synthetic"),
    "openwatch":    ("OpenWatch",           "real"),
    "totalcapture": ("TotalCapture / DIP",  "real"),
    "imuposer":     ("IMUPoser",            "real"),
}


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

TIER_GAP = 2.6       # degrees between tiers
TASK_GAP = 0.75      # degrees between tasks inside a tier

R_HOLE, R_TIER = 0.44, 0.66           # inner ring (tiers)
R_TASK_IN, R_TASK_OUT = 0.69, 1.00    # outer ring (individual evaluations)
R_LABEL = 1.05                        # leader text starts here


def wedge_angles(seam_gap: float):
    """(tier_spans, task_spans) in degrees travelled clockwise from 12 o'clock."""
    n_tasks = sum(len(t["tasks"]) for t in TIERS)
    n_task_gaps = n_tasks - len(TIERS)
    n_tier_gaps = len(TIERS) - 1
    usable = 360.0 - seam_gap - n_tier_gaps * TIER_GAP - n_task_gaps * TASK_GAP
    per = usable / n_tasks

    tier_spans, task_spans = [], []
    cursor = seam_gap / 2.0
    for ti, tier in enumerate(TIERS):
        tier_start = cursor
        for k in range(len(tier["tasks"])):
            task_spans.append((cursor, cursor + per))
            cursor += per
            if k < len(tier["tasks"]) - 1:
                cursor += TASK_GAP
        tier_spans.append((tier_start, cursor))
        if ti < len(TIERS) - 1:
            cursor += TIER_GAP
    return tier_spans, task_spans


def to_mpl(span):
    """Clockwise-from-top degrees -> matplotlib Wedge (theta1, theta2)."""
    s, e = span
    return 90.0 - e, 90.0 - s


def polar(deg_from_top, r):
    a = np.deg2rad(90.0 - deg_from_top)
    return r * np.cos(a), r * np.sin(a)


# --------------------------------------------------------------------------
# Draw
# --------------------------------------------------------------------------

def _arc_fit_fontsize(text: str, span_deg: float, r: float, pts_per_unit: float,
                      cap: float = 8.6, floor: float = 7.0):
    """Largest font size at which `text` still fits inside an arc, or None if
    even `floor` overruns it. Width is estimated at 0.60 em/char, which tracks
    DejaVu Sans Bold closely enough for a fit/no-fit decision."""
    arc_pts = np.deg2rad(span_deg) * r * pts_per_unit * 0.90   # 10% side margin
    fs = arc_pts / (max(len(text), 1) * 0.60)
    if fs < floor:
        return None
    return min(cap, fs)


def _donut(ax, th, surface, lay, pts_per_unit):
    ink, ink2, muted, hair = th["ink"], th["ink2"], th["muted"], th["hair"]
    r_hole, r_tier, r_task_in, r_task_out = lay.get(
        "radii", (R_HOLE, R_TIER, R_TASK_IN, R_TASK_OUT))
    tier_spans, task_spans = wedge_angles(lay["seam_gap"])
    r_mid = (r_hole + r_tier) / 2.0

    tier_of_task = []
    for ti, tier in enumerate(TIERS):
        tier_of_task.extend([ti] * len(tier["tasks"]))

    # Size every ring name the same: fit each independently, then set them all
    # to the smallest that fits. Tiers whose arc can't hold a name at the floor
    # (the single-task ones) show their number instead — their one outer label
    # already says what the tier is.
    ring_fs, named = None, set()
    if lay["ring_label"] == "name":
        fits = {}
        for ti, span in enumerate(tier_spans):
            f = _arc_fit_fontsize(f"{ti + 1} · {TIERS[ti]['name']}",
                                  span[1] - span[0], r_mid, pts_per_unit)
            if f is not None:
                fits[ti] = f
        if fits:
            ring_fs, named = min(fits.values()), set(fits)

    # ---- inner ring: tiers ---------------------------------------------
    for ti, span in enumerate(tier_spans):
        t1, t2 = to_mpl(span)
        ax.add_patch(Wedge((0, 0), r_tier, t1, t2, width=r_tier - r_hole,
                           facecolor=th["series"][ti], edgecolor=surface,
                           linewidth=2.0, zorder=3))
        mid = (span[0] + span[1]) / 2.0
        x, y = polar(mid, r_mid)

        label = f"{ti + 1} · {TIERS[ti]['name']}" if ti in named else None
        fs = ring_fs if ti in named else None
        if fs is not None:
            rot = -mid                       # run along the arc, not radially
            if 90.0 < mid < 270.0:
                rot += 180.0
            ax.text(x, y, label, rotation=rot, rotation_mode="anchor",
                    ha="center", va="center", fontsize=fs, fontweight="bold",
                    color="#ffffff", zorder=6)
        elif lay["ring_label"] != "none":
            ax.text(x, y, str(ti + 1), ha="center", va="center", fontsize=13.5,
                    fontweight="bold", color="#ffffff", zorder=6)

    # ---- outer ring: individual evaluations, each directly labelled -----
    for k, span in enumerate(task_spans):
        ti = tier_of_task[k]
        t1, t2 = to_mpl(span)
        ax.add_patch(Wedge((0, 0), r_task_out, t1, t2, width=r_task_out - r_task_in,
                           facecolor=mix(th["series"][ti], surface, th["tint"]),
                           edgecolor=surface, linewidth=2.0, zorder=3))

        ds, task = TIERS[ti]["tasks"][k - sum(len(t["tasks"]) for t in TIERS[:ti])]
        mid = (span[0] + span[1]) / 2.0

        flip = ti >= FLIP_FROM_TIER
        rot = 90.0 - mid + (180.0 if flip else 0.0)
        ha = "right" if flip else "left"

        # two-tone stacked label: dataset (bold ink) over task (muted),
        # offset perpendicular to the text run
        nx, ny = np.cos(np.deg2rad(rot + 90.0)), np.sin(np.deg2rad(rot + 90.0))
        bx, by = polar(mid, R_LABEL)
        ax.text(bx + 0.030 * nx, by + 0.030 * ny, ds, rotation=rot,
                rotation_mode="anchor", ha=ha, va="center", fontsize=8.8,
                fontweight="bold", color=ink, zorder=5)
        ax.text(bx - 0.030 * nx, by - 0.030 * ny, task, rotation=rot,
                rotation_mode="anchor", ha=ha, va="center", fontsize=8.2,
                color=ink2, zorder=5)

        x0, y0 = polar(mid, r_task_out + 0.004)
        x1, y1 = polar(mid, R_LABEL - 0.012)
        ax.plot([x0, x1], [y0, y1], lw=0.8, color=hair, zorder=2)

    # ---- the seam: where "finest" wraps back around to "coarsest" -------
    ax.plot([0, 0], [r_hole - 0.05, r_task_out + 0.05], lw=1.1, color=hair,
            zorder=4, solid_capstyle="round")
    if lay["seam_markers"]:
        # set vertically inside the widened seam gap, clear of every label
        ax.text(0.062, 1.20, "C O A R S E S T", rotation=90.0, ha="center",
                va="center", fontsize=8.0, fontweight="bold", color=muted, zorder=5)
        ax.text(-0.062, 1.20, "F I N E S T", rotation=90.0, ha="center",
                va="center", fontsize=8.0, fontweight="bold", color=muted, zorder=5)

    # ---- the 1 -> 6 direction, traced inside the hole -------------------
    if lay.get("ring_arrow"):
        r_arrow = r_hole - 0.065
        ts = np.linspace(tier_spans[0][0], tier_spans[-1][1], 400)
        pts = np.array([polar(t, r_arrow) for t in ts])
        ax.plot(pts[:-1, 0], pts[:-1, 1], lw=1.6, color=muted, zorder=5,
                solid_capstyle="round")
        ax.add_patch(FancyArrowPatch(
            posA=tuple(pts[-2]), posB=tuple(pts[-1]),
            arrowstyle="-|>,head_length=9,head_width=4.2", lw=1.6,
            color=muted, shrinkA=0, shrinkB=0, zorder=5))

    n_tasks = sum(len(t["tasks"]) for t in TIERS)
    hero = lay["hero_fs"]
    ax.text(0, 0.115, str(n_tasks), ha="center", va="center", fontsize=hero,
            fontweight="bold", color=ink, zorder=6)
    ax.text(0, -0.055, "downstream\nevaluations", ha="center", va="center",
            fontsize=10.5 * hero / 44.0, color=ink2, linespacing=1.3, zorder=6)
    ax.text(0, -0.190, "13 datasets · 6 tiers", ha="center", va="center",
            fontsize=8.4 * hero / 44.0, color=muted, zorder=6)


def _wrap(text: str, width: int = 14) -> str:
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines)


def _bottom_axis(bax, th, surface):
    """The same coarse -> fine ordering as the ring, straightened out: one
    left-to-right arrow with the six tiers marked along it."""
    ink, muted = th["ink"], th["muted"]
    y = 0.74

    bax.add_patch(FancyArrowPatch(
        (0.012, y), (0.995, y),
        arrowstyle="-|>,head_length=8,head_width=3.6", lw=1.4, color=muted,
        shrinkA=0, shrinkB=0, clip_on=False, zorder=2))
    bax.text(0.012, y + 0.115, "coarse", fontsize=8.6, color=muted,
             ha="left", va="bottom")
    bax.text(0.995, y + 0.115, "fine-grained", fontsize=8.6, color=muted,
             ha="right", va="bottom")

    for ti, x in enumerate(np.linspace(0.085, 0.915, len(TIERS))):
        bax.plot([x], [y], marker="s", markersize=15, color=th["series"][ti],
                 markeredgecolor=surface, markeredgewidth=1.6, zorder=3,
                 clip_on=False)
        bax.text(x, y, str(ti + 1), fontsize=7.6, fontweight="bold",
                 color="#ffffff", ha="center", va="center", zorder=4)
        bax.text(x, y - 0.175, _wrap(TIERS[ti]["name"]), fontsize=8.4,
                 color=ink, ha="center", va="top", linespacing=1.35, zorder=3)


def _compact_key(side, th):
    """Just the six tiers and the coarse -> fine axis they sit on."""
    ink, muted = th["ink"], th["muted"]
    top, pitch = 0.925, 0.158

    for ti, tier in enumerate(TIERS):
        ry = top - ti * pitch
        side.add_patch(Rectangle((0.105, ry - 0.026), 0.100, 0.042,
                                 facecolor=th["series"][ti], edgecolor="none",
                                 clip_on=False))
        side.text(0.155, ry - 0.005, str(ti + 1), fontsize=8.6,
                  fontweight="bold", color="#ffffff", ha="center", va="center")
        side.text(0.255, ry - 0.005, tier["name"], fontsize=9.4,
                  fontweight="bold", color=ink, va="center")

    y_top, y_bot = top + 0.028, top - 5 * pitch - 0.032
    side.add_patch(FancyArrowPatch(
        (0.040, y_top), (0.040, y_bot),
        arrowstyle="-|>,head_length=6,head_width=3", lw=1.3, color=muted,
        shrinkA=0, shrinkB=0, clip_on=False, zorder=2))
    # left-aligned so the words stay inside the panel instead of bleeding into
    # the gap between the pie and the key
    side.text(0.0, y_top + 0.020, "coarse", fontsize=8.4, color=muted,
              ha="left", va="bottom")
    side.text(0.0, y_bot - 0.024, "fine-grained", fontsize=8.4, color=muted,
              ha="left", va="top")


def _side_panel(side, th, surface, lay, counts):
    ink, ink2, muted, hair = th["ink"], th["ink2"], th["muted"], th["hair"]
    total_windows = sum(counts.values())
    pretrain_rows = sorted(counts.items(), key=lambda kv: -kv[1])

    y = 0.995
    side.text(0.0, y, "G R A N U L A R I T Y   T I E R S", fontsize=9.0,
              fontweight="bold", color=muted, va="top")
    y -= 0.038
    side.plot([0.0, 1.0], [y, y], lw=1.0, color=hair)

    tier_top = y - 0.026
    row_h = 0.098
    for ti, tier in enumerate(TIERS):
        ry = tier_top - ti * row_h
        side.add_patch(Rectangle((0.075, ry - 0.030), 0.028, 0.030,
                                 facecolor=th["series"][ti], edgecolor="none",
                                 clip_on=False))
        side.text(0.093, ry - 0.015, str(ti + 1), fontsize=8.0,
                  fontweight="bold", color="#ffffff", ha="center", va="center")
        side.text(0.125, ry - 0.001, tier["name"], fontsize=10.4,
                  fontweight="bold", color=ink, va="top")
        side.text(0.125, ry - 0.032, tier["rule"], fontsize=8.5, color=ink2,
                  va="top", style="italic")
        side.text(0.125, ry - 0.060, tier["detail"], fontsize=8.3, color=muted,
                  va="top")

    # the coarse -> fine axis, drawn once alongside the six tiers
    y_top, y_bot = tier_top - 0.004, tier_top - 5 * row_h - 0.034
    side.add_patch(FancyArrowPatch(
        (0.020, y_top), (0.020, y_bot),
        arrowstyle="-|>,head_length=6,head_width=3", lw=1.3, color=muted,
        shrinkA=0, shrinkB=0, clip_on=False, zorder=2))
    side.text(0.020, y_top + 0.012, "coarse", fontsize=8.0, color=muted,
              ha="center", va="bottom")
    side.text(0.020, y_bot - 0.014, "fine-grained", fontsize=8.0, color=muted,
              ha="center", va="top")

    # ---- pretraining corpus ---------------------------------------------
    y = y_bot - 0.058
    side.text(0.0, y, "S H A R E D   S S L   P R E T R A I N I N G   C O R P U S",
              fontsize=9.0, fontweight="bold", color=muted, va="top")
    y -= 0.038
    side.plot([0.0, 1.0], [y, y], lw=1.0, color=hair)
    y -= 0.028
    side.text(0.0, y, f"11 sources · {total_windows:,} train windows · "
                      f"masked-autoencoder objective",
              fontsize=8.6, color=ink2, va="top")
    y -= 0.042

    biggest = pretrain_rows[0][1]
    two_col = lay["corpus_cols"] == 2
    n_left = 6 if two_col else len(pretrain_rows)
    bar_w = 0.145 if two_col else 0.300
    bar_x = 0.245 if two_col else 0.420
    pct_x = 0.400 if two_col else 0.740
    pitch = 0.034 if two_col else 0.0345

    for i, (key, n) in enumerate(pretrain_rows):
        col, row = (0, i) if i < n_left else (1, i - n_left)
        cx = 0.0 + col * 0.515
        cy = y - row * pitch
        name, kind = PRETRAIN_META[key]
        pct = 100.0 * n / total_windows
        side.text(cx, cy, name, fontsize=8.3, color=ink, va="center")
        if two_col:
            side.text(cx, cy - 0.0175, kind, fontsize=7.0, color=muted, va="center")
        else:
            side.text(cx + 0.255, cy, kind, fontsize=7.4, color=muted, va="center")
        side.add_patch(Rectangle((cx + bar_x, cy - 0.006),
                                 max(bar_w * n / biggest, 0.006), 0.012,
                                 facecolor=mix(th["series"][0], surface, 0.30),
                                 edgecolor="none", clip_on=False))
        side.text(cx + pct_x, cy, f"{pct:.1f}%" if pct >= 0.1 else "<0.1%",
                  fontsize=7.8, color=ink2, va="center", ha="left")

    y -= n_left * pitch + 0.020
    side.text(0.0, y,
              "* HAR master parquet bundles PAMAP2, RecoFit, SAMoSA, Shoaib, UT-Watch,\n"
              "  UTD-MHAD and WISDM — tier 2's own datasets also appear upstream, so check\n"
              "  pretrain/eval overlap before reading tier 2 as fully held-out.",
              fontsize=7.5, color=muted, va="top", linespacing=1.5)


def draw(layout_name: str, theme_name: str, out_stem: str) -> None:
    th = THEMES[theme_name]
    lay = LAYOUTS[layout_name]
    surface, ink, ink2 = th["surface"], th["ink"], th["ink2"]

    with open(WINDOW_COUNTS) as f:
        counts = json.load(f)["full_source_sizes"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
    })

    fig = plt.figure(figsize=lay["figsize"], facecolor=surface)

    ax_box = lay["ax_box"]
    xlim, ylim = lay["xlim"], lay["ylim"]
    ax = fig.add_axes(ax_box)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_facecolor(surface)

    # equal aspect => the tighter of the two axis scales sets points-per-unit
    fw, fh = lay["figsize"]
    pts_per_unit = 72.0 * min(fw * ax_box[2] / (xlim[1] - xlim[0]),
                              fh * ax_box[3] / (ylim[1] - ylim[0]))
    _donut(ax, th, surface, lay, pts_per_unit)

    if lay.get("bottom_axis"):
        bax = fig.add_axes(lay["bottom_axis"])
        bax.set_xlim(0, 1)
        bax.set_ylim(0, 1)
        bax.axis("off")
        bax.set_facecolor(surface)
        _bottom_axis(bax, th, surface)

    if lay["key"] != "none":
        side = fig.add_axes(lay["side_box"])
        side.set_xlim(0, 1)
        side.set_ylim(0, 1)
        side.axis("off")
        side.set_facecolor(surface)
        if lay["key"] == "compact":
            _compact_key(side, th)
        else:
            _side_panel(side, th, surface, lay, counts)

    if lay["titles"]:
        fig.text(0.012, 0.982,
                 "IMU-SSL: one pretraining corpus, 22 downstream evaluations",
                 fontsize=17.5, fontweight="bold", color=ink, va="top")
        fig.text(0.012, 0.941,
                 "Outer ring: individual dataset · task.   Inner ring: granularity tier, running clockwise from the seam at 12 o'clock — "
                 "from one label per session (1) to per-frame joint kinematics (6).",
                 fontsize=9.6, color=ink2, va="top")

    for ext in ("png", "pdf"):
        path = f"{out_stem}.{ext}"
        fig.savefig(path, dpi=300, facecolor=surface, bbox_inches="tight",
                    pad_inches=0.26)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    for layout in ("v1", "v2", "square", "solo", "arrow"):
        for theme in ("light", "dark"):
            draw(layout, theme, os.path.join(HERE, f"task_taxonomy_{layout}_{theme}"))
