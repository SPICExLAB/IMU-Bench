#!/usr/bin/env python3
"""Harvest the Parkinson's-disease (PADS) row and column of the cross-task
transfer matrix into analysis/pd_transfer_values.json.

The other 20 cells of the matrix are the published §8 numbers and stay
hard-coded in plot_transferability.py; only the PD cells added on 2026-08-18
are read from disk here, so the figure can be regenerated as jobs land.

    python analysis/harvest_pd_transfer.py            # write the JSON
    python analysis/harvest_pd_transfer.py --print    # just show what is ready

Metric per target, matching that target's row in the figure:
    WISDM / OpenWatch gesture  accuracy %          logs/metrics.txt  (acc)
    Nymeria full-body pose     MPJPE mm            logs/metrics.txt  (mpjpe_mm)
    Nymeria w-frame odometry   ATE m               data_vis/<dir>/*.csv (ate)
    PADS Parkinson's           balanced accuracy % logs/metrics.txt  (bal_acc)

Values are read at the best.pt checkpoint's step (repo convention: metrics at
the best-val-loss checkpoint, not the last row).
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root -> `bench`

from bench.score.common import (
    load_ckpt_meta,
    parse_metrics_txt,
    resolve_ckpt,
    run_dir_of,
    val_row_at_step,
)

HERE = Path(__file__).parent
REPO = HERE.parent
OUT_JSON = HERE / "pd_transfer_values.json"

# ---------------------------------------------------------------- the cells --
# (table, target, source) -> how to read it.  `run` reads logs/metrics.txt at
# the best checkpoint; `csv_dir` reads the `ate` column of a data_vis CSV.
PD_ROW = {  # PD as the transfer TARGET -- the new row
    "scratch": {
        "mae": dict(run="vit1d_pd_probe_pads_mae"),
        "wisdm": dict(run="vit1d_transfer_pd_from_wisdm"),
        "pose": dict(run="vit1d_transfer_pd_from_fullbodypose"),
        "gesture": dict(run="vit1d_transfer_pd_from_gesture"),
        "odom": dict(run="vit1d_transfer_pd_from_odomworldframe"),
    },
    "finetune": {
        "mae": dict(run="vit1d_pd_finetune_pads_mae"),
        "wisdm": dict(run="vit1d_transfer_finetune_pd_from_wisdm"),
        "pose": dict(run="vit1d_transfer_finetune_pd_from_fullbodypose"),
        "gesture": dict(run="vit1d_transfer_finetune_pd_from_gesture"),
        "odom": dict(run="vit1d_transfer_finetune_pd_from_odomworldframe"),
    },
}
PD_ROW_REF = dict(run="vit1d_pd_supervised_pads_scratch")   # from-scratch baseline

PD_COLUMN = {  # PD as the transfer SOURCE -- the new column
    "scratch": {
        "WISDM": dict(run="vit1d_transfer_wisdm_from_pd", key="acc", pct=True),
        "Nymeria full-body pose": dict(run="vit1d_transfer_fullbodypose_from_pd", key="mpjpe_mm", pct=False),
        "OpenWatch gesture": dict(run="vit1d_transfer_gesture_from_pd", key="acc", pct=True),
        "Nymeria world-frame odometry": dict(
            csv_dir="data_vis/nymeria_odom_worldframe_transfer_from_pd"),
    },
    "finetune": {
        "WISDM": dict(run="vit1d_transfer_finetune_wisdm_from_pd", key="acc", pct=True),
        "Nymeria full-body pose": dict(run="vit1d_transfer_finetune_fullbodypose_from_pd",
                                       key="mpjpe_mm", pct=False),
        "OpenWatch gesture": dict(run="vit1d_transfer_finetune_gesture_from_pd", key="acc",
                                  pct=True),
        "Nymeria world-frame odometry": dict(
            csv_dir="data_vis/nymeria_odom_worldframe_transfer_finetune_from_pd"),
    },
}


def read_metric(run, key="bal_acc", pct=True):
    """Value of `key` in the val row at the best checkpoint's step, or None."""
    rd = run_dir_of(run)
    # best.pt appears at the first val improvement, so its existence alone does
    # NOT mean the job is finished -- the launch scripts drop .transfer_done at
    # the end, and a transfer cell is only harvestable once that lands.
    if run.startswith("vit1d_transfer_") and not os.path.exists(
            os.path.join(rd, ".transfer_done")):
        return None
    ckpt = resolve_ckpt(rd)
    if ckpt is None:
        return None
    step = load_ckpt_meta(ckpt)["step"]
    row = val_row_at_step(parse_metrics_txt(rd, "val"), step)
    if row is None or key not in row:
        return None
    return 100.0 * row[key] if pct else row[key]


def read_ate(csv_dir):
    """ATE (m) from the single-row CSV the odometry eval scripts write."""
    hits = sorted(glob.glob(os.path.join(REPO, csv_dir, "*.csv")))
    for path in hits:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if rows and rows[-1].get("ate"):
            return float(rows[-1]["ate"])
    return None


def read_cell(spec):
    if "csv_dir" in spec:
        return read_ate(spec["csv_dir"])
    return read_metric(spec["run"], spec.get("key", "bal_acc"), spec.get("pct", True))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print", dest="show", action="store_true",
                    help="report readiness without writing the JSON")
    args = ap.parse_args()

    out = {"pd_row_ref": read_cell(PD_ROW_REF), "pd_row": {}, "pd_column": {}}
    for table in ("scratch", "finetune"):
        out["pd_row"][table] = {src: read_cell(spec) for src, spec in PD_ROW[table].items()}
        out["pd_column"][table] = {tgt: read_cell(spec) for tgt, spec in PD_COLUMN[table].items()}

    ready = total = 0
    print(f"{'cell':<62}{'value':>12}")
    ref = out["pd_row_ref"]
    ref_s = "pending" if ref is None else f"{ref:.4f}"
    print(f"{'PD from-scratch reference (bal_acc %)':<62}{ref_s:>12}")
    for table in ("scratch", "finetune"):
        for src, v in out["pd_row"][table].items():
            total += 1
            ready += v is not None
            print(f"{f'[{table}] PD  <- {src}':<62}{'pending' if v is None else f'{v:.4f}':>12}")
        for tgt, v in out["pd_column"][table].items():
            total += 1
            ready += v is not None
            print(f"{f'[{table}] {tgt}  <- PD':<62}{'pending' if v is None else f'{v:.4f}':>12}")
    print(f"\n{ready}/{total} PD cells ready")

    if not args.show:
        OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
