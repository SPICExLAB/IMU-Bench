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
    if protocol != "scratch" and not enc.get("run") and not enc.get("weights"):
        raise ValueError(f"protocol={protocol} needs a pretrained encoder; {encoder_id} has none")
    # `run` names a checkpoint this project trained; `weights` a released file an
    # adapter loads itself (HarNet). Either counts as "pretrained"; neither being
    # set is what this guard is for.

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
    elif enc.get("external_encoder") == "normwear":
        # NormWear (Luo et al. 2024): the authors' released checkpoint, run through this harness by
        # imu_lm/models/baselines/normwear_encoder.py. All six channels and the harness preprocessing are kept.
        from imu_lm.models.baselines.normwear_encoder import normwear_downstream_config
        cfg = normwear_downstream_config(cfg, enc["weights"], chunk_samples=int(enc.get("chunk_samples", 256)))
        ob = cfg["objective"][obj_type]
    elif enc.get("external_encoder") == "harnet":
        # HarNet (Yuan et al. 2024): the authors' released UK Biobank weights, run
        # through this harness by imu_lm/models/baselines/harnet_encoder.py.
        from imu_lm.models.baselines.harnet_encoder import harnet_downstream_config
        cfg = harnet_downstream_config(cfg, enc.get("variant", "harnet5"), enc["weights"],
                                       openwatch=(task_id == "microgesture_openwatch"),
                                       clip=(task_id == "language_humanml3d"))
        ob = cfg["objective"][obj_type]
    else:
        dset(cfg, "vit1d.encoder.alt_attention", bool(enc["alt_attention"]))
        dset(cfg, "vit1d.shared_kernel", enc["shared_kernel"])
        # CLS ablation: an encoder pretrained with `use_cls: false` has no cls_token /
        # cls_norm weights, so the downstream encoder must be built the same way or the
        # strict state-dict load fails before training starts.
        if "use_cls" in enc:
            dset(cfg, "vit1d.encoder.use_cls", bool(enc["use_cls"]))
        # Architecture overrides for control encoders (e.g. a 0-layer "raw input + projection"
        # encoder): dotted keys applied verbatim so the downstream model matches the checkpoint.
        for dotted, v in (enc.get("vit1d_overrides") or {}).items():
            dset(cfg, dotted, v)
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
    elif enc.get("external_encoder") == "normwear":
        # as for HarNet below: a label for the branches' `pretrained_run` guard, and the release file as the checkpoint
        ob["pretrained_run"] = "normwear"
        ob["pretrained_ckpt"] = enc["weights"]
    elif enc.get("external_encoder") == "harnet":
        # HarNet's weights are the authors' released .mdl, not a run/ directory.
        # Every downstream branch resolves `pretrained_ckpt` as an absolute path
        # when it is not "best"/"latest", and HarnetEncoder.load_pretrained_state
        # understands the release's `module.feature_extractor.` key layout -- but
        # each branch still *guards* on a truthy `pretrained_run`, so it needs a
        # label. The variant name is that label; nothing else reads it.
        ob["pretrained_run"] = enc.get("variant", "harnet5")
        ob["pretrained_ckpt"] = enc["weights"]
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
    # ---- input normalisation --------------------------------------------------
    # A frozen or warm-started encoder must see downstream input preprocessed the way it was
    # pretrained. The encoder card's `normalize_method` (zscore | physical | raw) is authoritative
    # and is cross-checked against the pretrain config named by `preprocess_from`.
    #
    # History: from 2026-08-20 14:21 until this fix (2026-09-14) the bench applied physical
    # (DC-preserving) scaling to EVERY non-raw encoder, including encoders pretrained on per-window
    # z-score -- a train/eval preprocessing mismatch in 569 scored cells (all vd_* arms, mae_4src_z,
    # the 4-source SSL baselines). Those result JSONs were NOT rescored by this change.
    #
    # Overrides, for reproducing or contrasting cells:
    #   BENCH_ZSCORE=1    force z-score for any non-raw encoder (old comparison switch)
    #   BENCH_PHYSICAL=1  force physical for any non-raw encoder (reproduces the pre-fix protocol,
    #                     e.g. to resubmit a cell of a grid that was scored under it)
    # Encoders with no declared method (legacy rows, external adapters) and the scratch protocol
    # keep the previous default: physical unless BENCH_ZSCORE=1.
    force_z = os.environ.get("BENCH_ZSCORE", "") == "1"
    force_phys = os.environ.get("BENCH_PHYSICAL", "") == "1"
    if force_z and force_phys:
        raise ValueError("BENCH_ZSCORE=1 and BENCH_PHYSICAL=1 are mutually exclusive")
    PHYSICAL_DEFAULT = not force_z
    PHYSICAL_BLOCK = {"method": "physical", "acc_scale": 9.81, "gyro_scale": 5.0,
                      "clip_acc": 8.0, "clip_gyro": 8.0, "n_acc": 3, "acc_units": "auto"}
    declared = str(enc.get("normalize_method") or "")
    if declared not in ("", "zscore", "physical", "raw"):
        raise ValueError(f"encoder {encoder_id}: unknown normalize_method {declared!r}")
    if declared and protocol != "scratch" and enc.get("preprocess_from"):
        import yaml as _yaml
        _pre = (_yaml.safe_load(open(os.path.join(REPO, enc["preprocess_from"]))).get("preprocess") or {})
        _n = _pre.get("normalize") or {}
        _pre_method = str(_n.get("method", "zscore")) if _n.get("enabled", True) else "raw"
        if _pre_method != declared:
            raise ValueError(f"encoder {encoder_id}: registry normalize_method={declared!r} but its "
                             f"pretrain config {enc['preprocess_from']} uses {_pre_method!r}")
    if dget(cfg, "preprocess.normalize") is None:
        dset(cfg, "preprocess.normalize", {"enabled": True})
    if declared == "raw":
        # encoder was pretrained on raw (unnormalised, DC-preserving) input -> evaluate it the same way.
        # Copy the WHOLE preprocess block (impute + band filter + normalize off) from the pretrain
        # config when the card names one: pose/odometry templates carry no preprocess block at all, so
        # without this they would skip the 0.25-24 Hz filter the encoder was trained with.
        if enc.get("preprocess_from"):
            import yaml as _yaml
            pre = _yaml.safe_load(open(os.path.join(REPO, enc["preprocess_from"])))["preprocess"]
            cfg["preprocess"] = copy.deepcopy(pre)
        dset(cfg, "preprocess.normalize", {"enabled": False})
    elif dget(cfg, "preprocess.normalize.enabled", True):
        if protocol != "scratch" and declared in ("zscore", "physical") and not (force_z or force_phys):
            method = declared
        elif force_phys or (not force_z and PHYSICAL_DEFAULT):
            method = "physical"
        else:
            method = "zscore"
        norm = dget(cfg, "preprocess.normalize")
        if method == "physical":
            deep_update(norm, dict(PHYSICAL_BLOCK))
        else:
            for k in ("acc_scale", "gyro_scale", "clip_acc", "clip_gyro", "n_acc", "acc_units"):
                norm.pop(k, None)
            norm["method"] = "zscore"
            norm.setdefault("eps", 1e-6)
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
