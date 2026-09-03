#!/usr/bin/env python3
"""Materialise a training config for one benchmark cell.

    python -m bench.make_config --task arm_pose_nymeria --encoder focal --protocol finetune [--seed 42]
    python -m bench.make_config --all            # every cell of the controlled grid, seed 42

The generated yaml is the task's template config for that protocol with the
encoder-specific keys rewritten (alt_attention, shared_kernel, channels,
pretrained_run/ckpt) and the task's controlled `recipe` overrides applied.
Nothing else in the template changes -- so within a task, every encoder is
trained by a byte-identical recipe. Output: bench/generated/configs/<run>.yaml
"""
from __future__ import annotations

import argparse
import copy
import os
from typing import Any, Dict, Tuple

import yaml

from bench.registry_io import (REPO, ddel, deep_update, dget, dset, grid_cells, load_encoders,
                               load_protocols, load_tasks, run_name)

GEN_DIR = os.path.join(REPO, "bench", "generated", "configs")


def build_config(task_id: str, encoder_id: str, protocol: str, seed: int = 42,
                 tasks=None, encoders=None) -> Tuple[str, Dict[str, Any]]:
    tasks = tasks or load_tasks()
    encoders = encoders or load_encoders()
    task = tasks["tasks"][task_id]
    enc = encoders[encoder_id]
    protos = load_protocols()["protocols"]
    if protocol not in protos or protocol == "external":
        raise ValueError(f"bad protocol {protocol}")
    if enc.get("external") and not enc.get("external_encoder"):
        raise ValueError(f"encoder {encoder_id} is external; cannot be trained through the harness")
    if protocol == "scratch" and enc.get("run"):
        raise ValueError("protocol=scratch only makes sense with the 'scratch' encoder")
    if protocol != "scratch" and not enc.get("run"):
        raise ValueError(f"protocol={protocol} needs a pretrained encoder; {encoder_id} has none")

    tpl = task["templates"][protocol]
    with open(os.path.join(REPO, tpl["config"])) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)

    obj_type = tpl["objective_type"]
    dset(cfg, "objective.type", obj_type)
    if obj_type not in cfg["objective"] or cfg["objective"][obj_type] is None:
        cfg["objective"][obj_type] = {}
    ob = cfg["objective"][obj_type]

    # ---- encoder-specific keys ----------------------------------------------
    if enc.get("external_encoder") == "inertia1":
        # Foundation-model baseline run THROUGH the harness via an adapter
        # (imu_lm/models/baselines/inertia1_encoder.py): swaps the ViT1D encoder
        # block for an `inertia1:` block, acc-only input, raw g-scale preprocessing.
        # Task recipe / protocol extras below are applied exactly as for ViT1D rows.
        from imu_lm.models.baselines.inertia1_encoder import inertia1_downstream_config
        cfg = inertia1_downstream_config(cfg, enc.get("variant", "patchtst"), enc["run"],
                                         openwatch=(task_id == "microgesture_openwatch"),
                                         clip=(task_id == "language_humanml3d"),
                                         pretrained_ckpt=enc.get("ckpt", "best"),
                                         preset=enc.get("preset", "small"))
        ob = cfg["objective"][obj_type]
    else:
        dset(cfg, "vit1d.encoder.alt_attention", bool(enc["alt_attention"]))
        dset(cfg, "vit1d.shared_kernel", enc["shared_kernel"])
        cols = list(dget(cfg, "data.sensor_columns", ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]))
        if int(enc["channels"]) == 3:
            dset(cfg, "data.acc_only", True)
            dset(cfg, "data.sensor_columns", cols[:3])
        else:
            ddel(cfg, "data.acc_only")
            if len(cols) == 3:  # template was acc-only (not the case for our templates, but be safe)
                raise ValueError("template has 3 sensor columns but encoder needs 6")
    if protocol == "scratch":
        ob.pop("pretrained_run", None)
        ob.pop("pretrained_ckpt", None)
    else:
        ob["pretrained_run"] = enc["run"]
        ob["pretrained_ckpt"] = enc.get("ckpt", "best")
    for k, v in (tpl.get("extra") or {}).items():
        ob[k] = copy.deepcopy(v)

    # ---- controlled recipe ---------------------------------------------------
    for dotted, v in (task.get("recipe") or {}).items():
        if isinstance(v, dict) and isinstance(dget(cfg, dotted), dict):
            deep_update(dget(cfg, dotted), v)
        else:
            dset(cfg, dotted, v)
    for dotted, v in ((task.get("recipe_by_protocol") or {}).get(protocol) or {}).items():
        dset(cfg, dotted, v)
    # Physical (DC-preserving) scaling is the project default from 2026-08-20: it beat per-window
    # z-score at probe time by +0.08-0.10 HAR macro-F1 on every dataset pair measured. Set
    # BENCH_ZSCORE=1 to fall back to z-score for a comparison cell.
    PHYSICAL_DEFAULT = os.environ.get("BENCH_ZSCORE", "") != "1"
    if dget(cfg, "preprocess.normalize") is None:
        dset(cfg, "preprocess.normalize", {"enabled": True})
    if str(enc.get("normalize_method", "")) == "raw":
        # encoder was pretrained on raw (unnormalised, DC-preserving) input -> evaluate it the same way.
        # Copy the WHOLE preprocess block (impute + band filter + normalize off) from the pretrain
        # config when the card names one: pose/odometry templates carry no preprocess block at all, so
        # without this they would skip the 0.25-24 Hz filter the encoder was trained with.
        if enc.get("preprocess_from"):
            import yaml as _yaml
            pre = _yaml.safe_load(open(os.path.join(REPO, enc["preprocess_from"])))["preprocess"]
            cfg["preprocess"] = copy.deepcopy(pre)
        dset(cfg, "preprocess.normalize", {"enabled": False})
    elif PHYSICAL_DEFAULT and dget(cfg, "preprocess.normalize.enabled", True):
        deep_update(dget(cfg, "preprocess.normalize"),
                    {"method": "physical", "acc_scale": 9.81, "gyro_scale": 5.0,
                     "clip_acc": 8.0, "clip_gyro": 8.0, "n_acc": 3, "acc_units": "auto"})
    dset(cfg, "trainer.seed", int(seed))
    dset(cfg, "paths.runs_root", os.path.join(REPO, "runs"))

    # provenance block (ignored by the trainer, kept in the resolved config)
    cfg["bench"] = {"task": task_id, "encoder": encoder_id, "protocol": protocol, "seed": int(seed),
                    "template": tpl["config"], "registry_version": tasks.get("version", 1)}
    return run_name(task_id, encoder_id, protocol, seed), cfg


def write_config(task_id: str, encoder_id: str, protocol: str, seed: int = 42, out_dir: str = GEN_DIR,
                 tasks=None, encoders=None) -> Tuple[str, str]:
    name, cfg = build_config(task_id, encoder_id, protocol, seed, tasks, encoders)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".yaml")
    header = (f"# AUTO-GENERATED by bench/make_config.py -- do not edit.\n"
              f"# cell: task={task_id} encoder={encoder_id} protocol={protocol} seed={seed}\n"
              f"# template: {tasks['tasks'][task_id]['templates'][protocol]['config'] if tasks else ''}\n")
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return name, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--encoder")
    ap.add_argument("--protocol")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true", help="generate every grid cell")
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    tasks, encoders = load_tasks(), load_encoders()
    if args.all:
        for t, e, p in grid_cells(tasks):
            name, path = write_config(t, e, p, args.seed, tasks=tasks, encoders=encoders)
            print(path)
        return
    if not (args.task and args.encoder and args.protocol):
        ap.error("--task/--encoder/--protocol required (or --all)")
    if args.print:
        name, cfg = build_config(args.task, args.encoder, args.protocol, args.seed, tasks, encoders)
        print(f"# run: {name}")
        print(yaml.safe_dump(cfg, sort_keys=False))
    else:
        name, path = write_config(args.task, args.encoder, args.protocol, args.seed, tasks=tasks, encoders=encoders)
        print(path)


if __name__ == "__main__":
    main()
