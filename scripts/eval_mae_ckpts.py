#!/usr/bin/env python
"""
Linear-probe HAR accuracy for a MAE pretraining run.

Works with any MAE run that saves step_*.pt or latest.pt checkpoints.
For each checkpoint and dataset, trains a frozen linear head (mean-pooled
encoder output) and reports best val balanced accuracy.

Results are written to runs/<mae-run>/har_probe_results.txt.
Each probe run is saved under runs/<mae-run>_<dataset>_probe_<step>/ and
skipped on re-runs unless --force is passed.

Usage:
    # Probe latest.pt only:
    python scripts/eval_mae_ckpts.py \\
        --config   configs/base.yaml \\
        --mae-run  vit1d_mae_ocdetect_flu_no_norm_aug \\
        --datasets utd_mhad samosa prism

    # Probe specific step checkpoints:
    python scripts/eval_mae_ckpts.py ... --steps 100000 200000 300000

    # Probe a single checkpoint file directly:
    python scripts/eval_mae_ckpts.py ... --ckpt /abs/path/to/latest.pt

    # Use a custom probe config variant (default: no_norm):
    python scripts/eval_mae_ckpts.py ... --probe-variant flu

    # Re-train even if results already exist:
    python scripts/eval_mae_ckpts.py ... --force
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys

import imu_lm.models.ViT1D.run as vit_run
from imu_lm.utils.helpers import deep_update, load_yaml


# Map dataset → (probe_config_path, objective_key)
# probe_config_path uses {variant} placeholder filled by --probe-variant
DATASET_CONFIGS = {
    "utd_mhad": (
        "configs/vit1d_har_probe_utd_mhad_mae_{variant}.yaml",
        "utd_mhad_probe",
    ),
    "samosa": (
        "configs/vit1d_har_probe_samosa_mae_{variant}.yaml",
        "har_probe",
    ),
    "prism": (
        "configs/vit1d_har_probe_prism_mae_{variant}.yaml",
        "har_probe",
    ),
    "ocdetect": (
        "configs/vit1d_har_probe_ocdetect_mae_{variant}.yaml",
        "ocdetect_probe",
    ),
}
ALL_DATASETS = list(DATASET_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_checkpoints(ckpt_dir: str, steps: list[int] | None) -> list[tuple[int, str]]:
    """Return sorted [(step, abs_path)] for step_*.pt files, plus latest.pt."""
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

    # If no step checkpoints matched (or none exist), fall back to latest.pt
    if not results:
        latest = os.path.join(ckpt_dir, "latest.pt")
        if os.path.exists(latest):
            results.append(("latest", latest))

    return results


def _best_val_metrics(metrics_path: str) -> dict:
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


def _run_probe(cfg_base: dict, mae_run: str, ckpt_path: str,
               step_tag: str, dataset: str, probe_config_path: str,
               obj_key: str, runs_root: str, force: bool) -> dict:
    probe_run_name = f"{mae_run}_{dataset}_probe_{step_tag}"
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
    obj_cfg["pretrained_run"]  = mae_run
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
    ap.add_argument("--mae-run",  required=True, help="MAE run name under runs/")
    ap.add_argument("--steps", nargs="*", type=int, default=None,
                    help="Only eval specific steps; omit to use latest.pt")
    ap.add_argument("--ckpt", default=None,
                    help="Evaluate a single checkpoint by absolute path")
    ap.add_argument("--datasets", nargs="*", default=ALL_DATASETS,
                    choices=ALL_DATASETS, metavar="DATASET",
                    help=f"Datasets to probe (default: all). Choices: {ALL_DATASETS}")
    ap.add_argument("--probe-variant", default="no_norm",
                    help="Probe config variant suffix: 'no_norm' or 'flu' (default: no_norm)")
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
        mae_ckpt_dir = os.path.join(runs_root, args.mae_run, "checkpoints")
        if not os.path.isdir(mae_ckpt_dir):
            sys.exit(f"[error] checkpoint dir not found: {mae_ckpt_dir}")
        checkpoints = _find_checkpoints(mae_ckpt_dir, args.steps)
        if not checkpoints:
            sys.exit(f"[error] no checkpoints found in {mae_ckpt_dir}")

    # Resolve probe config paths for this variant
    dataset_cfgs = {}
    for ds, (cfg_template, obj_key) in DATASET_CONFIGS.items():
        cfg_path = cfg_template.format(variant=args.probe_variant)
        if not os.path.exists(cfg_path):
            print(f"[warning] probe config not found, skipping {ds}: {cfg_path}")
            continue
        dataset_cfgs[ds] = (cfg_path, obj_key)

    datasets = [d for d in args.datasets if d in dataset_cfgs]
    if not datasets:
        sys.exit("[error] no valid dataset configs found")

    print(f"\nMAE run      : {args.mae_run}")
    print(f"Probe variant: {args.probe_variant}")
    print(f"Datasets     : {datasets}")
    print(f"Checkpoints  : {len(checkpoints)}\n")

    all_results: dict[str, dict[str, dict]] = {}

    for mae_step, ckpt_path in checkpoints:
        step_tag = f"step_{mae_step:07d}" if isinstance(mae_step, int) else mae_step
        print(f"── checkpoint {step_tag} ──────────────────────────────")
        all_results[step_tag] = {}
        for dataset in datasets:
            probe_cfg_path, obj_key = dataset_cfgs[dataset]
            m = _run_probe(base_cfg, args.mae_run, ckpt_path,
                           step_tag, dataset, probe_cfg_path,
                           obj_key, runs_root, args.force)
            all_results[step_tag][dataset] = m
        print()

    # ── Summary table ──────────────────────────────────────────────────────
    col_w = 11
    ds_header = "  ".join(f"{d:^{col_w}}" for d in datasets)
    header = f"{'MAE ckpt':>14}  {ds_header}"
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print(f"Linear probe bal_acc — {args.mae_run}")
    print(sep)
    print(header)
    print(" " * 16 + "  ".join(f"{'bal_acc':^{col_w}}" for _ in datasets))
    print(sep)
    for step_tag, ds_metrics in all_results.items():
        row = f"{step_tag:>14}"
        for dataset in datasets:
            v = ds_metrics.get(dataset, {}).get("bal_acc", float("nan"))
            row += f"  {v:^{col_w}.4f}"
        print(row)
    print("=" * len(header))

    # ── Write to file ──────────────────────────────────────────────────────
    out_path = os.path.join(runs_root, args.mae_run, "har_probe_results.txt")
    with open(out_path, "w") as f:
        f.write(f"# HAR linear probe results — {args.mae_run}\n")
        f.write(f"# datasets: {datasets}\n\n")
        f.write(f"{'mae_ckpt':>14}")
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
