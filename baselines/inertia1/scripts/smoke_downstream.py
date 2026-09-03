"""GPU smoke test for the Inertia-1 harness integration.

1. writes a RANDOM-INIT Inertia-1 checkpoint in the run-dir format
   (runs/_smoke_inertia1_<variant>/checkpoints/best.pt)
2. derives a tiny-budget copy of every configs/inertia1/*<variant>*.yaml that
   points at it, runs each through scripts/run_pretrain.py for a few steps
3. runs every scoring script on the resulting run dirs

Everything lands under runs/_smoke_* / <scratch>; nothing touches real runs.
Usage: python baselines/inertia1/scripts/smoke_downstream.py --variant patchtst [--only pd,arm]
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(REPO)
sys.path.insert(0, REPO)
from imu_lm.models.baselines.inertia1_encoder import Inertia1Encoder  # noqa: E402
from baselines.inertia1.scripts.pretrain_inertia1 import downstream_cfg  # noqa: E402

RUNS_ROOT = os.path.join(REPO, "runs")
SMOKE_CFG_DIR = os.path.join(REPO, "configs", "inertia1", "_smoke")


def sh(cmd: str, timeout: int = 3600) -> int:
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True, timeout=timeout)
    print(f"[exit {r.returncode} in {time.time()-t:.0f}s]", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="patchtst")
    ap.add_argument("--only", default=None, help="comma-separated substrings of config names to run")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    v_short = {"patchtst": "patchtst", "ar_transformer": "art"}[args.variant]
    smoke_run = f"_smoke_inertia1_{v_short}"

    # 1. random-init checkpoint
    cfg = downstream_cfg(args.variant, "small", smoke_run)
    enc = Inertia1Encoder(cfg)
    d = os.path.join(RUNS_ROOT, smoke_run, "checkpoints")
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(RUNS_ROOT, smoke_run, "artifacts"), exist_ok=True)
    torch.save({"model": enc.state_dict(), "cfg": cfg, "step": 0}, os.path.join(d, "best.pt"))
    torch.save({"model": enc.state_dict(), "cfg": cfg, "step": 0}, os.path.join(d, "latest.pt"))
    import json
    json.dump({"embedding_dim": enc.embed_dim, "backbone": "inertia1"}, open(os.path.join(RUNS_ROOT, smoke_run, "artifacts", "encoder_meta.json"), "w"))
    print("wrote smoke ckpt", d)

    # 2. tiny configs
    os.makedirs(SMOKE_CFG_DIR, exist_ok=True)
    cfgs = sorted(glob.glob(os.path.join(REPO, "configs", "inertia1", f"*inertia1_{v_short}*.yaml")))
    if args.only:
        keys = args.only.split(",")
        cfgs = [c for c in cfgs if any(k in os.path.basename(c) for k in keys)]
    results = {}
    smoke_names = []
    for c in cfgs:
        name = os.path.basename(c)[:-5]
        y = yaml.safe_load(open(c))
        blk = y["objective"][y["objective"]["type"]]
        blk["pretrained_run"] = smoke_run
        tr = y.setdefault("trainer", {})
        tr["max_steps"] = args.steps
        tr["val_every_steps"] = max(5, args.steps // 3)
        tr.setdefault("early_stopping", {})["enabled"] = True
        tr["early_stopping"]["patience"] = 999
        tr["val_max_batches"] = 5
        y.setdefault("logging", {})["ckpt_every_steps"] = max(5, args.steps // 3)
        y.setdefault("data", {})["num_workers"] = 4
        # pose probes: cut the file lists so loading is fast
        if "train_files" in blk and blk.get("loader", "").startswith("nymeria"):
            blk["train_fraction"] = 0.03
        sname = "_smoke_" + name
        yaml.safe_dump(y, open(os.path.join(SMOKE_CFG_DIR, sname + ".yaml"), "w"), sort_keys=False)
        rc = sh(f"python scripts/run_pretrain.py --config configs/base.yaml --model-config {SMOKE_CFG_DIR}/{sname}.yaml --run-name {sname} 2>&1 | tail -40")
        results[sname] = rc
        smoke_names.append(sname)

    # 3. scoring scripts
    L = f"Inertia-1 ({v_short}) SMOKE"
    TR = "/projects/b53017/vbc1591/processed_data_body_Ty_rwrist"
    VA = "/projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist"
    S = "_smoke_"
    scoring = {
        "arm_pose_err": f"python scripts/eval_arm_pose_position_error.py --runs '{L} probe={S}vit1d_arm_pose_probe_nymeria_inertia1_{v_short}' '{L} finetune={S}vit1d_arm_pose_finetune_nymeria_inertia1_{v_short}' --max-val-files 3 --out-json {SMOKE_CFG_DIR}/arm_pose_err.json 2>&1 | tail -15",
        "odom_wf_probe": f"python scripts/eval_odom_nymeria_worldframe.py --run {smoke_run} --runs-root runs --train-dir {TR} --val-dir {VA} --max-train-files 6 --max-val-files 3 --epochs 2 --out-dir {SMOKE_CFG_DIR}/odom_wf 2>&1 | tail -15",
        "odom_lv_probe": f"python scripts/eval_odom_nymeria_altattn.py --run {smoke_run} --runs-root runs --train-dir {TR} --val-dir {VA} --max-train-files 6 --max-val-files 3 --epochs 2 --out-dir {SMOKE_CFG_DIR}/odom_lv 2>&1 | tail -15",
        "odom_wf_finetune_score": f"python scripts/vis_odom_scratch_worldframe.py --run {S}vit1d_odom_worldframe_finetune_nymeria_inertia1_{v_short} --runs-root runs --val-dir {VA} --max-val-files 3 --max-windows 200 --out-dir {SMOKE_CFG_DIR}/odom_wf_ft --label smoke 2>&1 | tail -15",
        "openwatch": f"python scripts/eval_openwatch_windowed_protocol.py --runs '{L} probe win/noaug:{S}vit1d_gesture_probe_openwatch_6class_inertia1_{v_short}_windowed_stratified:openwatch_gesture_probe' '{L} finetune win/aug:{S}vit1d_gesture_finetune_openwatch_6class_inertia1_{v_short}_windowed_stratified_aug:openwatch_gesture_finetune' --splits val test --out-json {SMOKE_CFG_DIR}/openwatch.json 2>&1 | tail -15",
        "retrieval": f"python -u scripts/eval_retrieval_alignment.py --model ours --run {S}vit1d_clip_humanml3d_flu_probe_inertia1_{v_short}_unimtssim --ckpt best --label smoke --out {SMOKE_CFG_DIR}/retrieval.json --gallery-sizes 64,100 --n-subsets 5 --seed 42 --batch-size 128 --split val --limit-items 300 2>&1 | tail -15",
        "pads_subject": f"python scripts/eval_pads_subject_level.py --model-config {SMOKE_CFG_DIR}/{S}vit1d_pd_probe_pads_inertia1_{v_short}.yaml --run {S}vit1d_pd_probe_pads_inertia1_{v_short} --out {SMOKE_CFG_DIR}/pads_subject.json 2>&1 | tail -15",
    }
    for k, cmd in scoring.items():
        if args.only and not any(kk in cmd for kk in args.only.split(",")):
            continue
        results["score:" + k] = sh(cmd)

    print("\n=========== SMOKE SUMMARY")
    for k, rc in results.items():
        print(f"  {'OK ' if rc == 0 else 'FAIL'}  {k}")
    print("cleanup: rm -rf " + " ".join(os.path.join(RUNS_ROOT, n) for n in smoke_names) + f" {os.path.join(RUNS_ROOT, smoke_run)} {SMOKE_CFG_DIR}")


if __name__ == "__main__":
    main()
