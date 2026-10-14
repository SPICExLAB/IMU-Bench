#!/usr/bin/env python3
"""Promote the winning probe decoder, then re-run mae_4src_z on the three kinematic
tasks. Every other encoder keeps its small-decoder numbers until the deep decoder is shown to be
worth a wider re-run.

Selection rule: per task, the arm with the best value of the task's own primary metric on its
reported split. The same decoder is then applied to EVERY encoder, so the choice cannot favour one
encoder over another; it shifts the absolute numbers only.
"""
import glob, json, os, subprocess, sys, copy, yaml, collections
sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL"); os.chdir("/gpfs/home/vbc1591/IMU-SSL")
from bench.registry_io import load_tasks

TASKS = {"arm_pose_nymeria": "pose_decoder",
         "full_body_pose_nymeria": "pose_decoder",
         "odometry_nymeria_worldframe": "odom_decoder"}
tasks = load_tasks()["tasks"]

# ---- 1. pick the winner per task -------------------------------------------------
res = collections.defaultdict(dict)
for f in glob.glob("bench/generated/results_deco/*.json"):
    d = json.load(open(f)); res[d["task"]][d["arm"]] = d["metrics"]
winners = {}
for t, key in TASKS.items():
    prim = tasks[t]["metrics"]["primary"]
    higher = bool(tasks[t]["metrics"][prim].get("higher_is_better", True))
    arms = {a: m[prim] for a, m in res.get(t, {}).items() if prim in m}
    if len(arms) < 3:
        print(f"[promote] {t}: only {len(arms)}/3 arms scored — skipping"); continue
    best = max(arms, key=arms.get) if higher else min(arms, key=arms.get)
    winners[t] = best
    print(f"[promote] {t}: {arms} -> winner {best} ({prim}={arms[best]:.4f})")

if not winners:
    print("[promote] nothing to promote"); sys.exit(1)

# ---- 2. write the winning decoder into the task card as the shared probe recipe ---
POSE = {"d12a": dict(hidden_size=256, num_hidden_layers=4, num_temporal_layers=8,  num_attention_heads=8, dropout=0.1),
        "d12b": dict(hidden_size=256, num_hidden_layers=2, num_temporal_layers=10, num_attention_heads=8, dropout=0.1),
        "d12w": dict(hidden_size=384, num_hidden_layers=4, num_temporal_layers=8,  num_attention_heads=6, dropout=0.1)}
ODOM = {"d12a": dict(hidden_size=256, num_hidden_layers=12, num_attention_heads=8, dropout=0.1),
        "d12b": dict(hidden_size=192, num_hidden_layers=12, num_attention_heads=4, dropout=0.1),
        "d12w": dict(hidden_size=384, num_hidden_layers=12, num_attention_heads=6, dropout=0.1)}
reg = yaml.safe_load(open("bench/registry/tasks.yaml"))
for t, arm in winners.items():
    key = TASKS[t]; dec = (POSE if key == "pose_decoder" else ODOM)[arm]
    card = reg["tasks"][t]
    rbp = card.setdefault("recipe_by_protocol", {}).setdefault("probe", {})
    for k, v in dec.items(): rbp[f"vit1d.{key}.{k}"] = v
    card["_decoder_note"] = (f"probe decoder promoted to arm {arm} on 2026-08-27: chosen by the task's "
                             f"primary metric over three ~12-layer variants, then applied to every encoder")
yaml.safe_dump(reg, open("bench/registry/tasks.yaml", "w"), sort_keys=False)
print("[promote] task cards updated")

# ---- 3. archive the old probe results and resubmit every encoder ------------------
from bench.make_config import build_config
os.makedirs("bench/results/_superseded_small_decoder", exist_ok=True)
os.makedirs("bench/generated/slurm/redec", exist_ok=True)
jobs = []
for t in winners:
    encs = ["mae_4src_z"]
    for enc in encs:
        f = f"bench/results/{t}/{enc}__probe__s42.json"
        if os.path.exists(f):
            os.replace(f, os.path.join("bench/results/_superseded_small_decoder", f"{t}__{os.path.basename(f)}"))
    for enc in encs:
        name, cfg = build_config(t, enc, "probe")
        yaml.safe_dump(cfg, open(f"bench/generated/configs/{name}.yaml", "w"), sort_keys=False)
        # ONE job per cell, never a dual-partition twin: both twins `rm -rf runs/{name}` and then
        # train into the same directory, which silently corrupts the run (2026-09-04: arm pose came
        # back 8.14 instead of 8.05 from a byte-identical config because two jobs shared the dir).
        for part, acct, gres in (("gengpu","p32139","gpu:a100:1"),):
            ex = "#SBATCH --exclude=qgpu3006,qgpu3013,qgpu2014\n" if part == "gengpu" else ""
            sh = f"""#!/bin/bash
#SBATCH --account={acct}
#SBATCH --partition={part}
{ex}#SBATCH --gres={gres}
#SBATCH --time=14:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=rd{enc[:9]}{t[:5]}
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1
if [ -f bench/results/{t}/{enc}__probe__s42.json ]; then echo already-scored; exit 0; fi
rm -rf runs/{name}
python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/configs/{name}.yaml --run-name {name}
python -m bench.score --task {t} --encoder {enc} --protocol probe --seed 42
python -m bench.leaderboard >/dev/null 2>&1 || true
"""
            p = f"bench/generated/slurm/redec/{name}_{part}.sh"; open(p,"w").write(sh); os.chmod(p, 0o755)
            jobs.append(subprocess.check_output(["sbatch","--parsable",p], text=True).strip())
    print(f"[promote] {t}: resubmitted {len(encs)} encoders")
print(f"[promote] {len(jobs)} jobs submitted ({len(jobs)//2} cells)")
open("bench/generated/.redec_jobs","w").write(":".join(jobs))
