#!/usr/bin/env python3
"""Submit the three ICLR prep batches as SINGLE jobs per cell.

  A  recontaminated  re-run the 27 cells whose run directory had two concurrent jobs
                     writing into it (analysis/twin_collision_audit.txt)
  B  seeds           seeds 43 and 44 for mae_4src_z and simmtm_4src on the three
                     kinematic tasks, probe and finetune -> the run-to-run sigma the
                     paper needs before any "improvement" can be called real
  C  relcon          the 30 missing relcon_4src downstream cells; its encoder is
                     already trained to 100k steps but was never evaluated

One job per cell, never a dual-partition twin: twins both `rm -rf runs/{name}` and
then train into the same directory, which is what contaminated batch A in the first
place.  Batch A deletes the poisoned run directory instead of resuming it.

    python bench/generated/submit_iclr_batches.py A B C     # or any subset
    python bench/generated/submit_iclr_batches.py --dry A
"""
import copy, os, subprocess, sys, yaml

sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL")
os.chdir("/gpfs/home/vbc1591/IMU-SSL")
from bench.make_config import build_config
from bench.registry_io import run_name

AUDIT = "analysis/twin_collision_audit.txt"
HAR = ["har_wisdm", "har_utd_mhad", "har_pamap2", "har_recofit", "har_hhar",
       "har_mhealth", "har_samosa", "har_ut_watch", "har_prism"]
OTHER = ["arm_pose_nymeria", "full_body_pose_nymeria", "odometry_nymeria_worldframe",
         "microgesture_openwatch", "language_humanml3d", "pd_pads"]
KINEMATIC = ["arm_pose_nymeria", "full_body_pose_nymeria", "odometry_nymeria_worldframe"]
LONG = set(KINEMATIC) | {"language_humanml3d"}
# nodes that have produced ~200x slowdowns or exit-120 crashes
BAD_NODES = "qgpu3006,qgpu3013,qgpu2014,qgpu2006"

dry = "--dry" in sys.argv
which = [a for a in sys.argv[1:] if not a.startswith("-")] or ["A", "C"]   # B (multi-seed) is off by decision: this project reports one seed
os.makedirs("bench/generated/slurm/iclr", exist_ok=True)


def cells_A():
    """Contaminated cells, parsed back out of the audit file."""
    out = []
    if not os.path.exists(AUDIT):
        print(f"[A] {AUDIT} missing — run the audit first"); return out
    for line in open(AUDIT):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rn = line.split()[-1]                      # "  76 min   bench_<task>__<enc>__<proto>__s42"
        if not rn.startswith("bench_"):
            continue
        body = rn[len("bench_"):]
        try:
            task, enc, proto, seed = body.rsplit("__", 3)
        except ValueError:
            continue
        out.append((task, enc, proto, int(seed.lstrip("s")), True))
    return out


def cells_B():
    return [(t, e, p, s, False)
            for e in ("mae_4src_z", "simmtm_4src")
            for t in KINEMATIC
            for p in ("probe", "finetune")
            for s in (43, 44)]


def cells_C():
    return [(t, "relcon_4src", p, 42, False) for t in HAR + OTHER for p in ("probe", "finetune")]


def submit(task, enc, proto, seed, wipe, tag):
    res = f"bench/results/{task}/{enc}__{proto}__s{seed}.json"
    if os.path.exists(res) and not wipe:
        return None
    name = run_name(task, enc, proto, seed)
    try:
        _, cfg = build_config(task, enc, proto, seed)
    except Exception as e:
        print(f"[{tag}] skip {name}: {e}"); return None
    cfg = copy.deepcopy(cfg)
    if task in HAR:                       # Inertia-1 paper protocol, matches results_lrtune/
        ot = cfg["objective"]["type"]
        cfg["objective"][ot]["class_weighted_ce"] = True
        cfg["trainer"]["early_stopping"] = {"enabled": True, "monitor": "bal_acc", "patience": 20}
        cfg["trainer"]["optim"]["lr"] = 1e-3 if proto == "probe" else 1e-4
    yaml.safe_dump(cfg, open(f"bench/generated/configs/{name}.yaml", "w"), sort_keys=False)

    tl = "14:00:00" if task in LONG else "04:00:00"
    # full-body loads 867 Nymeria sessions up front and OOMs at 48G (2026-09-06)
    mem = {"language_humanml3d": "80G", "full_body_pose_nymeria": "96G"}.get(task, "48G")
    # a contaminated cell must start from nothing; resuming would inherit the poisoned state
    pre = (f"rm -f {res}\nrm -rf runs/{name}\n" if wipe
           else f"if [ -f {res} ]; then echo already-scored; exit 0; fi\n")
    resume = "" if wipe else " --resume latest"
    sh = f"""#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --exclude={BAD_NODES}
#SBATCH --gres=gpu:a100:1
#SBATCH --time={tl}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=8
#SBATCH --job-name={tag}{enc[:6]}{task.replace('har_','')[:6]}{proto[:2]}{seed}
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1
{pre}python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/configs/{name}.yaml --run-name {name}{resume}
python -m bench.score --task {task} --encoder {enc} --protocol {proto} --seed {seed}
python -m bench.leaderboard >/dev/null 2>&1 || true
"""
    path = f"bench/generated/slurm/iclr/{name}.sh"
    open(path, "w").write(sh); os.chmod(path, 0o755)
    if dry:
        print(f"[{tag}] would submit {name}"); return "DRY"
    return subprocess.check_output(["sbatch", "--parsable", path], text=True).strip()


all_jobs = {}
for batch, fn in (("A", cells_A), ("B", cells_B), ("C", cells_C)):
    if batch not in which:
        continue
    jobs = [j for c in fn() if (j := submit(*c, tag=batch))]
    all_jobs[batch] = jobs
    print(f"[{batch}] submitted {len(jobs)} cells")

if not dry and all_jobs:
    with open("bench/generated/.iclr_jobs", "w") as f:
        for b, js in all_jobs.items():
            f.write(f"{b}:{':'.join(js)}\n")
    print(f"total {sum(len(v) for v in all_jobs.values())} jobs -> bench/generated/.iclr_jobs")
