#!/usr/bin/env python3
"""Render cross-task transferability matrices as bare PNG table figures.

Rows are target tasks, columns are the source the encoder was initialised from.
Colour encodes *transferability*: the relative gain over that table's reference
protocol, sign-corrected so positive always means "better".

    higher-is-better metric:   (x - ref) / ref
    lower-is-better  metric:   (ref - x) / ref

Darker = stronger transfer. Cells worse than the reference (negative transfer)
are drawn on the neutral midpoint with a red hatched ring, so the sign survives
greyscale printing and colour-vision deficiency.

The PADS Parkinson's-disease row and column are read from
``analysis/pd_transfer_values.json`` (written by ``harvest_pd_transfer.py``);
cells whose job has not landed yet render as a dashed "pending" placeholder.

The figures carry no title, caption or footnote — they are meant to drop into a
paper under their own LaTeX caption.

Usage
-----
    python analysis/plot_transferability.py --table scratch
    python analysis/plot_transferability.py --table finetune
    python analysis/plot_transferability.py --table all          # both (default)
    python analysis/plot_transferability.py --table all --scale row
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------- palette ----
# Sequential blue ramp, steps 100 -> 700, light -> dark (documented palette).
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SURFACE = "#fcfcfb"      # chart surface
PLANE = "#f9f9f7"        # page plane (reference / n-a cells)
INK = "#0b0b0b"          # primary ink
INK_2 = "#52514e"        # secondary ink
MUTED = "#898781"        # axis / labels
GRIDLINE = "#e1e0d9"     # hairline
NEUTRAL_MID = "#f0efec"  # diverging neutral midpoint
CRITICAL = "#d03b3b"     # status: critical (negative transfer)

CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

# ------------------------------------------------------------------- data ----
# Source keys are ordered to match the target rows, so the "source == target"
# cells fall on a visible diagonal.
#
# A cell is one of:
#   number   the measured value
#   None     source == target, drawn as the diagonal "—"
#   PENDING  the job is still queued/running, drawn as a dashed placeholder
PENDING = "pending"

TABLES = {
    # Frozen / from-scratch protocol: the reference is training the target from
    # scratch, and the MAE probe is one of the transfer sources.
    "scratch": dict(
        out="transferability_scratch.png",
        ref_label="From scratch",
        ref_sub="(baseline)",
        cbar="Gain over from-scratch (%)",
        sources=[
            ("mae", "MAE probe", "self-supervised"),
            ("wisdm", "WISDM", "HAR"),
            ("pose", "Pose", "Nymeria"),
            ("gesture", "Gesture", "OpenWatch"),
            ("odom", "Odometry", "Nymeria"),
            ("pd", "Parkinson's", "PADS"),
        ],
        targets=[
            dict(
                name="WISDM", metric="Accuracy", arrow="↑", higher_better=True,
                fmt="{:.2f}", ref=60.89,
                values=dict(mae=65.50, wisdm=None, pose=64.09, gesture=64.78, odom=62.31,
                            pd=PENDING),
            ),
            dict(
                name="Nymeria full-body pose", metric="MPJPE (mm)", arrow="↓",
                higher_better=False, fmt="{:.2f}", ref=92.56,
                values=dict(mae=97.63, wisdm=100.08, pose=None, gesture=99.70, odom=95.93,
                            pd=PENDING),
            ),
            dict(
                name="OpenWatch gesture", metric="Accuracy", arrow="↑",
                higher_better=True, fmt="{:.2f}", ref=44.90,
                values=dict(mae=67.35, wisdm=64.29, pose=62.24, gesture=None, odom=56.12,
                            pd=PENDING),
            ),
            dict(
                name="Nymeria world-frame odometry", metric="ATE (m)", arrow="↓",
                higher_better=False, fmt="{:.4f}", ref=0.1419,
                values=dict(mae=0.0787, wisdm=0.0783, pose=0.0753, gesture=0.0790, odom=None,
                            pd=PENDING),
            ),
            # PADS 3-class differential diagnosis. `acc` is uninformative here
            # (the PD-heavy majority sits at ~62% of val windows and every
            # column lands within 1.5pp of it), so the row is scored on
            # balanced accuracy, where chance is 33.33% -- see
            # Experiments_table.md §14.
            dict(
                name="PADS Parkinson's", metric="Balanced acc.", arrow="↑",
                higher_better=True, fmt="{:.2f}", ref=35.99,
                values=dict(mae=38.94, wisdm=PENDING, pose=PENDING, gesture=PENDING,
                            odom=PENDING, pd=None),
            ),
        ],
    ),
    # Table 12 — trainable cross-task transfer: the source-finetuned encoder
    # initialises the target model, which is then optimised end to end. The
    # "MAE pretrain" column is the direct MAE finetune. Laid out like the table
    # above: from-scratch is the leftmost baseline and the reference for colour,
    # so the two figures share one scale and read the same way.
    "finetune": dict(
        out="transferability_finetune.png",
        ref_label="From scratch",
        ref_sub="(baseline)",
        cbar="Gain over from-scratch (%)",
        sources=[
            ("mae", "MAE pretrain", "direct finetune"),
            ("wisdm", "WISDM", "HAR"),
            ("pose", "Pose", "Nymeria"),
            ("gesture", "Gesture", "OpenWatch"),
            ("odom", "Odometry", "Nymeria"),
            ("pd", "Parkinson's", "PADS"),
        ],
        targets=[
            dict(
                name="WISDM", metric="Accuracy", arrow="↑", higher_better=True,
                fmt="{:.2f}", ref=60.89,
                values=dict(mae=71.27, wisdm=None, pose=71.80, gesture=71.18, odom=71.29,
                            pd=PENDING),
            ),
            dict(
                name="Nymeria full-body pose", metric="MPJPE (mm)", arrow="↓",
                higher_better=False, fmt="{:.2f}", ref=92.56,
                values=dict(mae=92.39, wisdm=92.16, pose=None, gesture=92.24, odom=91.49,
                            pd=PENDING),
            ),
            dict(
                name="OpenWatch gesture", metric="Accuracy", arrow="↑",
                higher_better=True, fmt="{:.2f}", ref=44.90,
                values=dict(mae=66.33, wisdm=63.27, pose=68.37, gesture=None, odom=67.35,
                            pd=PENDING),
            ),
            dict(
                name="Nymeria world-frame odometry", metric="ATE (m)", arrow="↓",
                higher_better=False, fmt="{:.4f}", ref=0.1419,
                values=dict(mae=0.1293, wisdm=0.1278, pose=0.1259, gesture=0.1323, odom=None,
                            pd=PENDING),
            ),
            dict(
                name="PADS Parkinson's", metric="Balanced acc.", arrow="↑",
                higher_better=True, fmt="{:.2f}", ref=35.99,
                values=dict(mae=39.19, wisdm=PENDING, pose=PENDING, gesture=PENDING,
                            odom=PENDING, pd=None),
            ),
        ],
    ),
}

# --------------------------------------------------------------- harvesting --
# The PD row/column are measured by analysis/harvest_pd_transfer.py rather than
# typed in above; whatever it has managed to read wins over the placeholders.
PD_VALUES_JSON = Path(__file__).parent / "pd_transfer_values.json"
PD_TARGET = "PADS Parkinson's"


def apply_pd_harvest(path=PD_VALUES_JSON):
    """Overwrite the PD cells from the harvested JSON, if it exists."""
    if not path.exists():
        return
    data = json.loads(path.read_text())
    for table, spec in TABLES.items():
        by_name = {t["name"]: t for t in spec["targets"]}
        pd_row = by_name[PD_TARGET]
        if data.get("pd_row_ref") is not None:
            pd_row["ref"] = data["pd_row_ref"]
        for src, value in data.get("pd_row", {}).get(table, {}).items():
            if value is not None:
                pd_row["values"][src] = value
        for target_name, value in data.get("pd_column", {}).get(table, {}).items():
            if value is not None:
                by_name[target_name]["values"]["pd"] = value


def transfer_gain(value, ref, higher_better):
    """Relative gain over the reference protocol, in %, positive = better."""
    if value is None or value is PENDING:
        return np.nan
    delta = (value - ref) if higher_better else (ref - value)
    return 100.0 * delta / abs(ref)


def nice_ceiling(x):
    """Round up to a readable axis maximum (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8 x 10^n)."""
    if not np.isfinite(x) or x <= 0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(x))
    for m in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
        if x <= m * mag:
            return m * mag
    return 10.0 * mag


# -------------------------------------------------------------------- ink -----
def _srgb_to_lin(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb):
    r, g, b = (_srgb_to_lin(255 * c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(rgb_a, hex_b):
    la = relative_luminance(rgb_a)
    lb = relative_luminance(matplotlib.colors.to_rgb(hex_b))
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def ink_on(fill_rgb):
    """Pick whichever ink wins WCAG contrast against the cell fill."""
    return "#ffffff" if contrast(fill_rgb, "#ffffff") >= contrast(fill_rgb, INK) else INK


# ------------------------------------------------------------------- draw -----
CELL_GAP = 0.05      # surface gap between fills, in cell units
RADIUS = 0.045       # rounded corners, in cell units

# figure geometry, in inches
COL_GAP = 0.42       # gap between reference column and the matrix
CELL_W = 1.42
CELL_H = 1.02
HEAD_H = 0.60        # column-header band
BAR_W = 1.02         # colourbar + its label
PAD = 0.10
GUTTER = 0.30        # breathing room between a label and what it labels


def text_width_in(s, fontsize, weight="normal", dpi=100):
    """Measured rendered width of a string, in inches — labels are sized, not guessed."""
    fig = plt.figure(figsize=(1, 1), dpi=dpi)
    t = fig.text(0, 0, s, fontsize=fontsize, fontweight=weight)
    fig.canvas.draw()
    w = t.get_window_extent(fig.canvas.get_renderer()).width / dpi
    plt.close(fig)
    return w


def rounded_cell(ax, x, y, w, h, **kw):
    pad = CELL_GAP / 2
    patch = FancyBboxPatch(
        (x + pad, y + pad), w - 2 * pad, h - 2 * pad,
        boxstyle=f"round,pad=0,rounding_size={RADIUS}", **kw,
    )
    ax.add_patch(patch)
    return patch


def global_gain_max():
    """Largest gain over every table, so all figures can share one colour scale."""
    best = 0.0
    for spec in TABLES.values():
        for t in spec["targets"]:
            for key, _, _ in spec["sources"]:
                g = transfer_gain(t["values"][key], t["ref"], t["higher_better"])
                if np.isfinite(g):
                    best = max(best, g)
    return best


def render(spec, scale, outdir, dpi, vmax=None):
    sources, targets = spec["sources"], spec["targets"]
    n_rows, n_cols = len(targets), len(sources)

    gains = np.full((n_rows, n_cols), np.nan)
    for i, t in enumerate(targets):
        for j, (key, _, _) in enumerate(sources):
            gains[i, j] = transfer_gain(t["values"][key], t["ref"], t["higher_better"])

    if scale == "row":
        cvals = np.full_like(gains, np.nan)
        for i in range(n_rows):
            row = gains[i]
            lo, hi = np.nanmin(row), np.nanmax(row)
            cvals[i] = (row - lo) / (hi - lo) if hi > lo else 0.5
        norm = Normalize(vmin=0.0, vmax=1.0)
        cbar_label = "Transferability within target (0 = weakest, 1 = strongest)"
    else:
        cvals = gains
        norm = Normalize(vmin=0.0, vmax=vmax or nice_ceiling(np.nanmax(gains)))
        cbar_label = spec["cbar"]

    # ---- geometry -----------------------------------------------------------
    label_w = GUTTER + max(
        max(text_width_in(t["name"], 10.5, "bold") for t in targets),
        max(text_width_in(f"{t['metric']} {t['arrow']}", 9) for t in targets),
    )
    ref_w_in = max(
        1.55,
        text_width_in(spec["ref_label"], 10.5, "bold") + 0.16,
    )
    grid_w = n_cols * CELL_W
    fig_w = PAD + label_w + ref_w_in + COL_GAP + grid_w + BAR_W + PAD
    fig_h = PAD + HEAD_H + n_rows * CELL_H + PAD

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=SURFACE)
    ax = fig.add_axes([
        (PAD + label_w) / fig_w,
        PAD / fig_h,
        (ref_w_in + COL_GAP + grid_w) / fig_w,
        (n_rows * CELL_H) / fig_h,
    ])
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # data units: one cell = 1.0 wide; the reference column is scaled to match
    ref_w = ref_w_in / CELL_W
    gap_w = COL_GAP / CELL_W
    x0 = ref_w + gap_w
    ax.set_xlim(-0.03, x0 + n_cols + 0.03)
    ax.set_ylim(n_rows + 0.03, -0.03)

    # header offsets in data units (HEAD_H inches above the top of the grid)
    h_lo = -(0.16 * HEAD_H / CELL_H)   # sub-label baseline
    h_hi = -(0.52 * HEAD_H / CELL_H)   # bold label baseline

    # ---- reference column ---------------------------------------------------
    ax.text(ref_w / 2, h_hi, spec["ref_label"], ha="center", va="bottom",
            fontsize=10.5, color=INK_2, fontweight="bold")
    ax.text(ref_w / 2, h_lo, spec["ref_sub"], ha="center", va="bottom",
            fontsize=8.5, color=MUTED)
    for i, t in enumerate(targets):
        rounded_cell(ax, 0.0, i, ref_w, 1.0, facecolor=PLANE,
                     edgecolor=GRIDLINE, linewidth=1.0)
        ax.text(ref_w / 2, i + 0.5, t["fmt"].format(t["ref"]), ha="center",
                va="center", fontsize=11.5, color=INK_2)

    # ---- column headers -----------------------------------------------------
    for j, (_, label, sub) in enumerate(sources):
        cx = x0 + j + 0.5
        ax.text(cx, h_hi, label, ha="center", va="bottom", fontsize=10.5,
                color=INK, fontweight="bold")
        ax.text(cx, h_lo, sub, ha="center", va="bottom", fontsize=8.5, color=MUTED)

    # ---- matrix -------------------------------------------------------------
    for i, t in enumerate(targets):
        ax.text(-0.20, i + 0.42, t["name"], ha="right", va="bottom",
                fontsize=10.5, color=INK, fontweight="bold", clip_on=False)
        ax.text(-0.20, i + 0.58, f"{t['metric']} {t['arrow']}", ha="right",
                va="top", fontsize=9, color=MUTED, clip_on=False)

        for j, (key, _, _) in enumerate(sources):
            x, value = x0 + j, t["values"][key]

            if value is None:  # source == target
                rounded_cell(ax, x, i, 1.0, 1.0, facecolor=PLANE,
                             edgecolor=GRIDLINE, linewidth=1.0)
                ax.text(x + 0.5, i + 0.5, "—", ha="center", va="center",
                        fontsize=15, color=MUTED, alpha=0.45)
                continue

            if value is PENDING:  # job submitted, result not in yet
                rounded_cell(ax, x, i, 1.0, 1.0, facecolor=PLANE,
                             edgecolor=MUTED, linewidth=1.0, linestyle=(0, (3, 3)))
                ax.text(x + 0.5, i + 0.5, "pending", ha="center", va="center",
                        fontsize=9.5, color=MUTED, style="italic")
                continue

            gain = gains[i, j]
            bbox = None
            if gain < 0:  # negative transfer
                rounded_cell(ax, x, i, 1.0, 1.0, facecolor=NEUTRAL_MID,
                             edgecolor=CRITICAL, linewidth=1.6, hatch="//")
                val_ink, delta_ink = INK, CRITICAL
                # a plate behind the text keeps it legible over the hatch
                bbox = dict(facecolor=NEUTRAL_MID, edgecolor="none",
                            boxstyle="square,pad=0.32")
            else:
                fill = CMAP(norm(cvals[i, j]))
                rounded_cell(ax, x, i, 1.0, 1.0, facecolor=fill,
                             edgecolor="none", linewidth=0)
                val_ink = delta_ink = ink_on(fill)

            ax.text(x + 0.5, i + 0.44, t["fmt"].format(value), ha="center",
                    va="bottom", fontsize=12, color=val_ink, fontweight="bold",
                    bbox=bbox)
            ax.text(x + 0.5, i + 0.57, f"{gain:+.2f}%", ha="center", va="top",
                    fontsize=9, color=delta_ink, bbox=bbox)

    # ---- colourbar ----------------------------------------------------------
    cax = fig.add_axes([
        (PAD + label_w + ref_w_in + COL_GAP + grid_w + 0.20) / fig_w,
        PAD / fig_h,
        0.15 / fig_w,
        (n_rows * CELL_H) / fig_h,
    ])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, colors=MUTED, length=0, pad=5)
    cb.set_label(cbar_label, fontsize=9, color=INK_2, labelpad=11)

    out = Path(outdir) / spec["out"].replace(
        ".png", "_rownorm.png" if scale == "row" else ".png"
    )
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")

    head = f"{'target':<30}" + "".join(f"{s[1]:>12}" for s in sources)
    print(head)
    for i, t in enumerate(targets):
        cells = ""
        for j, (key, _, _) in enumerate(sources):
            if t["values"][key] is PENDING:
                cells += f"{'pending':>12}"
            elif np.isnan(gains[i, j]):
                cells += f"{'n/a':>12}"
            else:
                cells += f"{gains[i, j]:>+11.2f}%"
        print(f"{t['name']:<30}{cells}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", choices=["scratch", "finetune", "all"], default="all")
    ap.add_argument(
        "--scale", choices=["absolute", "row"], default="absolute",
        help="absolute: one gain scale across all targets (default). "
             "row: min-max normalise within each target row.",
    )
    ap.add_argument(
        "--per-table-scale", action="store_true",
        help="give each table its own colour range instead of one shared range "
             "(shared is the default, so darkness means the same in every figure).",
    )
    ap.add_argument("--outdir", default=str(Path(__file__).parent))
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument(
        "--no-harvest", action="store_true",
        help="ignore analysis/pd_transfer_values.json and keep the placeholders",
    )
    args = ap.parse_args()

    if not args.no_harvest:
        apply_pd_harvest()

    vmax = None if args.per_table_scale else nice_ceiling(global_gain_max())
    keys = ["scratch", "finetune"] if args.table == "all" else [args.table]
    for k in keys:
        render(TABLES[k], args.scale, args.outdir, args.dpi, vmax)


if __name__ == "__main__":
    main()
