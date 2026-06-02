#!/usr/bin/env python3
"""
Aggregate per-dataset supervised-training CSVs into one combined comparison.

Reads data_vis/supervised_<dataset>/supervised_<dataset>.csv for each dataset
and produces a single grouped bar chart + summary table comparing v0 vs v6
(or whatever labels were used) across all datasets.

Usage:
    python scripts/aggregate_supervised_results.py \
        --datasets pamap2 recofit samosa shoaib ut_watch utd_mhad wisdm \
        --root data_vis \
        --out-dir data_vis/supervised_summary/
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+",
                   default=["pamap2", "recofit", "samosa", "shoaib",
                            "ut_watch", "utd_mhad", "wisdm"])
    p.add_argument("--root",    default="data_vis",
                   help="Root dir containing supervised_<dataset>/ folders")
    p.add_argument("--out-dir", default="data_vis/supervised_summary")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Collect results: {dataset: {label: bal_acc}}
    rows = []
    found_datasets = []
    for ds in args.datasets:
        csv = os.path.join(args.root, f"supervised_{ds}", f"supervised_{ds}.csv")
        if not os.path.exists(csv):
            print(f"  [skip] {csv} not found")
            continue
        df = pd.read_csv(csv)
        rec = {"dataset": ds}
        for _, r in df.iterrows():
            rec[r["model"]] = r["bal_acc"]
        rows.append(rec)
        found_datasets.append(ds)

    if not rows:
        print("No result CSVs found. Run supervised_train_samosa.py per dataset first.")
        return

    summary = pd.DataFrame(rows).set_index("dataset")
    labels  = [c for c in summary.columns]   # model labels (e.g. v0, v6)

    # ── Print table ────────────────────────────────────────────────────────────
    print("\n" + "=" * (14 + 12 * len(labels) + 12))
    header = f"{'Dataset':<14}" + "".join(f"{l:>12}" for l in labels) + f"{'Δ(last-first)':>14}"
    print(header)
    print("-" * len(header))
    for ds in found_datasets:
        vals = [summary.loc[ds, l] for l in labels]
        delta = vals[-1] - vals[0]
        row = f"{ds:<14}" + "".join(f"{v:>12.4f}" for v in vals) + f"{delta:>+14.4f}"
        print(row)
    print("-" * len(header))
    # Mean row
    means = [summary[l].mean() for l in labels]
    mrow = f"{'MEAN':<14}" + "".join(f"{m:>12.4f}" for m in means) + \
           f"{means[-1]-means[0]:>+14.4f}"
    print(mrow)
    print("=" * len(header))

    # ── Grouped bar chart ──────────────────────────────────────────────────────
    x      = np.arange(len(found_datasets))
    width  = 0.8 / len(labels)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(max(9, len(found_datasets) * 1.6), 5.5))
    for li, label in enumerate(labels):
        vals = [summary.loc[ds, label] for ds in found_datasets]
        bars = ax.bar(x + li * width, vals, width * 0.9,
                      label=label, color=colors[li % len(colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(found_datasets, fontsize=10)
    ax.set_ylabel("Balanced Accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Supervised Training from Scratch — All Datasets\n"
                 "(full-parameter, no pretraining)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "supervised_all_datasets.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_png}")

    # Save combined CSV
    summary["mean_row"] = False
    out_csv = os.path.join(args.out_dir, "supervised_all_datasets.csv")
    summary.drop(columns="mean_row").to_csv(out_csv)
    print(f"Saved → {out_csv}")


if __name__ == "__main__":
    main()
