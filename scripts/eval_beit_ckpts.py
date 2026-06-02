#!/usr/bin/env python
"""
Linear-probe HAR accuracy at every saved checkpoint of a BEiT pretraining run.

For each step_*.pt in runs/<beit-run>/checkpoints/, and for each probe dataset,
trains a frozen linear head on the BEiT CLS token and reports best val balanced accuracy.

Results are written to runs/<beit-run>/har_probe_results.txt and printed as a table.
Each probe run is saved under runs/<beit-run>_<dataset>_probe_<step>/ and skipped
on re-runs unless --force is passed.

Usage:
    # Evaluate all step checkpoints on all datasets:
    python scripts/eval_beit_ckpts.py \\
        --config    configs/base.yaml \\
        --beit-run  vit1d_beit_ocdetect_anyrot_nopooling_8lvl

    # Evaluate only specific steps:
    python scripts/eval_beit_ckpts.py ... --steps 5000 50000 100000 200000

    # Evaluate only specific datasets:
    python scripts/eval_beit_ckpts.py ... --datasets utd_mhad samosa

    # Evaluate a single checkpoint file directly:
    python scripts/eval_beit_ckpts.py ... --ckpt /abs/path/to/latest.pt

    # Re-train even if probe already exists:
    python scripts/eval_beit_ckpts.py ... --force
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys

import imu_lm.models.ViT1D.run as vit_run
from imu_lm.utils.helpers import deep_update, load_yaml


DATASET_CONFIGS = {
    "utd_mhad": (
        "configs/vit1d_har_probe_utd_mhad_beit.yaml",
        "utd_mhad_probe",
    ),
    "samosa": (
        "configs/vit1d_har_probe_samosa_beit.yaml",
        "har_probe",
    ),
    "prism": (
        "configs/vit1d_har_probe_prism_beit.yaml",
        "har_probe",
    ),
}
ALL_DATASETS = list(DATASET_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_checkpoints(ckpt_dir: str, steps: list[int] | None) -> list[tuple[int, str]]:
    """Return sorted [(step, abs_path)] for all step_*.pt files."""
    results = []
    for fname in os.listdir(ckpt_dir):
        m = re.fullmatch(r"step_(\d+)\.pt", fname)
        if not m:
            continue
        step = int(m.group(1))
        if steps is not None and step not in steps:
            continue
        results.append((step, os.path.join(ckpt_dir, fname)))
    results.sort(key=lambda x: x[0])
    return results


def _best_val_metrics(metrics_path: str) -> dict:
    """Parse metrics.txt and return the val row with highest bal_acc."""
    best = {"probe_step": None, "bal_acc": 0.0, "acc": 0.0, "macro_f1": 0.0}
    if not os.path.exists(metrics_path):
        return best
    with open(metrics_path) as f:
        for line in f:
            if "split=val" not in line:
                continue
            m_step = re.search(r"step=(\d+)", line)
            m_bal  = re.search(r"bal_acc=([\d.eE+\-]+)", line)
            m_acc  = re.search(r"(?<!\w)acc=([\d.eE+\-]+)", line)
            m_f1   = re.search(r"macro_f1=([\d.eE+\-]+)", line)
            if not (m_step and m_bal):
                continue
            bal_acc = float(m_bal.group(1))
            if bal_acc > best["bal_acc"]:
                best = {
                    "probe_step": int(m_step.group(1)),
                    "bal_acc":    bal_acc,
                    "acc":        float(m_acc.group(1)) if m_acc else 0.0,
                    "macro_f1":   float(m_f1.group(1))  if m_f1  else 0.0,
                }
    return best


def _run_probe(cfg_base: dict, beit_run: str, ckpt_path: str,
               step_tag: str, dataset: str, probe_config_path: str,
               obj_key: str, runs_root: str, force: bool) -> dict:
    """Train one probe and return best val metrics."""
    probe_run_name = f"{beit_run}_{dataset}_probe_{step_tag}"
    probe_run_dir  = os.path.join(runs_root, probe_run_name)
    metrics_path   = os.path.join(probe_run_dir, "logs", "metrics.txt")

    if not force:
        existing = _best_val_metrics(metrics_path)
        if existing["probe_step"] is not None:
            print(f"  [{dataset}] skip — already done (bal_acc={existing['bal_acc']:.4f})")
            return existing
    else:
        import shutil
        logs_dir = os.path.join(probe_run_dir, "logs")
        if os.path.isdir(logs_dir):
            shutil.rmtree(logs_dir)

    os.makedirs(probe_run_dir, exist_ok=True)

    probe_cfg = load_yaml(probe_config_path)
    cfg = deep_update(copy.deepcopy(cfg_base), probe_cfg)
    obj_cfg = cfg.setdefault("objective", {}).setdefault(obj_key, {})
    obj_cfg["pretrained_run"]  = beit_run
    obj_cfg["pretrained_ckpt"] = ckpt_path

    print(f"  [{dataset}] training → {probe_run_name}")
    try:
        vit_run.main(cfg, probe_run_dir)
    except Exception as exc:
        print(f"  [{dataset}] ERROR: {exc}")
        return {"probe_step": None, "bal_acc": float("nan"),
                "acc": float("nan"), "macro_f1": float("nan")}

    best = _best_val_metrics(metrics_path)
    print(f"  [{dataset}] done — bal_acc={best['bal_acc']:.4f}")
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",   required=True, help="Base config YAML (configs/base.yaml)")
    ap.add_argument("--beit-run", required=True, help="BEiT run name under runs/")
    ap.add_argument("--steps", nargs="*", type=int, default=None,
                    help="Only eval specific steps (e.g. --steps 5000 50000 100000 200000)")
    ap.add_argument("--ckpt", default=None,
                    help="Evaluate a single checkpoint by absolute path (any filename)")
    ap.add_argument("--datasets", nargs="*", default=ALL_DATASETS,
                    choices=ALL_DATASETS, metavar="DATASET",
                    help=f"Datasets to probe (default: all). Choices: {ALL_DATASETS}")
    ap.add_argument("--force", action="store_true",
                    help="Re-train probe even if results already exist")
    args = ap.parse_args()

    base_cfg  = load_yaml(args.config)
    runs_root = base_cfg.get("paths", {}).get("runs_root", "runs")

    if args.ckpt:
        ckpt_path   = os.path.abspath(args.ckpt)
        if not os.path.exists(ckpt_path):
            sys.exit(f"[error] checkpoint not found: {ckpt_path}")
        label       = os.path.splitext(os.path.basename(ckpt_path))[0]
        checkpoints = [(label, ckpt_path)]
    else:
        beit_ckpt_dir = os.path.join(runs_root, args.beit_run, "checkpoints")
        if not os.path.isdir(beit_ckpt_dir):
            sys.exit(f"[error] checkpoint dir not found: {beit_ckpt_dir}")
        checkpoints = _find_checkpoints(beit_ckpt_dir, args.steps)
        if not checkpoints:
            sys.exit(f"[error] no step_*.pt checkpoints found in {beit_ckpt_dir}")

    datasets = args.datasets
    print(f"\nBEiT run   : {args.beit_run}")
    print(f"Datasets   : {datasets}")
    print(f"Checkpoints: {len(checkpoints)}\n")

    all_results: dict[str, dict[str, dict]] = {}

    for beit_step, ckpt_path in checkpoints:
        step_tag = f"step_{beit_step:07d}" if isinstance(beit_step, int) else beit_step
        print(f"── checkpoint {step_tag} ──────────────────────────────")
        all_results[step_tag] = {}
        for dataset in datasets:
            probe_cfg_path, obj_key = DATASET_CONFIGS[dataset]
            m = _run_probe(base_cfg, args.beit_run, ckpt_path,
                           step_tag, dataset, probe_cfg_path,
                           obj_key, runs_root, args.force)
            all_results[step_tag][dataset] = m
        print()

    # ── Summary table ──────────────────────────────────────────────────────
    col_w = 11
    ds_header = "  ".join(f"{d:^{col_w}}" for d in datasets)
    header = f"{'BEiT ckpt':>14}  {ds_header}"
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print(f"Linear probe bal_acc — {args.beit_run}")
    print(sep)
    print(header)
    sub_header = " " * 16 + "  ".join(f"{'bal_acc':^{col_w}}" for _ in datasets)
    print(sub_header)
    print(sep)
    for step_tag, ds_metrics in all_results.items():
        row = f"{step_tag:>14}"
        for dataset in datasets:
            m = ds_metrics.get(dataset, {})
            v = m.get("bal_acc", float("nan"))
            row += f"  {v:^{col_w}.4f}"
        print(row)
    print("=" * len(header))

    # ── Write to file ──────────────────────────────────────────────────────
    out_path = os.path.join(runs_root, args.beit_run, "har_probe_results.txt")
    with open(out_path, "w") as f:
        f.write(f"# HAR linear probe results — {args.beit_run}\n")
        f.write(f"# datasets: {datasets}\n\n")
        f.write(f"{'beit_ckpt':>14}")
        for d in datasets:
            f.write(f"  {d+'_bal_acc':>13}  {d+'_acc':>10}  {d+'_f1':>8}")
        f.write("\n" + sep + "\n")
        for step_tag, ds_metrics in all_results.items():
            f.write(f"{step_tag:>14}")
            for dataset in datasets:
                m = ds_metrics.get(dataset, {})
                f.write(f"  {m.get('bal_acc', float('nan')):>13.4f}"
                        f"  {m.get('acc',     float('nan')):>10.4f}"
                        f"  {m.get('macro_f1',float('nan')):>8.4f}")
            f.write("\n")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
