#!/usr/bin/env python3
"""Fill every missing downstream cell for the four 4-source MAE encoders.

Tasks: 9 HAR + arm pose, full-body pose, odometry, microgesture, language, PD — probe AND finetune.
HAR uses the Inertia-1-paper protocol (weighted CE, val bal-acc selection, probe lr 1e-3 / ft 1e-4)
so the cells line up with results_lrtune/; other tasks use their bench task-card recipes. Each encoder
carries normalize_method + preprocess_from, so a cell inherits the preprocessing its encoder was
pretrained with. Skips anything already scored; dual-submits to gengpu and b53017.
"""
import copy, os, subprocess, sys, yaml
sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL"); os.chdir("/gpfs/home/vbc1591/IMU-SSL")
from bench.make_config import build_config

ENCS  = sys.argv[1].split(",") if len(sys.argv) > 1 else ["mae_4src_dc","mae_4src_phys","mae_4src_physnp","mae_4src_z"]
HAR   = ["har_wisdm","har_utd_mhad","har_pamap2","har_recofit","har_hhar","har_mhealth","har_samosa","har_ut_watch","har_prism"]
OTHER = ["arm_pose_nymeria","full_body_pose_nymeria","odometry_nymeria_worldframe","microgesture_openwatch","language_humanml3d","pd_pads"]
LONG  = {"language_humanml3d","full_body_pose_nymeria","odometry_nymeria_worldframe","arm_pose_nymeria"}
os.makedirs("bench/generated/slurm/f4", exist_ok=True)
jobs = []
for enc in ENCS:
    if not os.path.exists(f"runs/{enc.replace('mae_','mae_')}/checkpoints/best.pt"):
        print(f"[skip] {enc}: no checkpoint yet"); continue
    for t in HAR + OTHER:
        for proto in ("probe", "finetune"):
            res = f"bench/results/{t}/{enc}__{proto}__s42.json"
            if os.path.exists(res): continue
            name, cfg = build_config(t, enc, proto); cfg = copy.deepcopy(cfg)
            if t in HAR:
                ot = cfg["objective"]["type"]
                cfg["objective"][ot]["class_weighted_ce"] = True
                cfg["trainer"]["early_stopping"] = {"enabled": True, "monitor": "bal_acc", "patience": 20}
                cfg["trainer"]["optim"]["lr"] = 1e-3 if proto == "probe" else 1e-4
            yaml.safe_dump(cfg, open(f"bench/generated/configs/{name}.yaml", "w"), sort_keys=False)
            tl  = "12:00:00" if t in LONG else "03:00:00"
            mem = "80G" if t == "language_humanml3d" else "48G"
            # SINGLE job per cell. Dual-partition twins both `rm -rf runs/{name}` and then train
            # into the same directory, silently corrupting the run (audit 2026-09-04: 27 bench
            # cells affected). Never re-introduce a second partition in this loop.
            for part, acct, gres in (("gengpu", "p32139", "gpu:a100:1"),):
                ex = "#SBATCH --exclude=qgpu3006,qgpu3013\n" if part == "gengpu" else ""
                sh = f"""#!/bin/bash
#SBATCH --account={acct}
#SBATCH --partition={part}
{ex}#SBATCH --gres={gres}
#SBATCH --time={tl}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=8
#SBATCH --job-name=f4{enc[-4:]}{t.replace('har_','')[:7]}{proto[:2]}
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1
if [ -f {res} ]; then echo already-scored; exit 0; fi
python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/configs/{name}.yaml --run-name {name} --resume latest
python -m bench.score --task {t} --encoder {enc} --protocol {proto} --seed 42
python -m bench.leaderboard >/dev/null 2>&1 || true
"""
                p = f"bench/generated/slurm/f4/{name}_{part}.sh"; open(p, "w").write(sh); os.chmod(p, 0o755)
                jobs.append(subprocess.check_output(["sbatch","--parsable",p], text=True).strip())
print(f"submitted {len(jobs)} jobs ({len(jobs)//2} cells x 2 partitions)")
open("bench/generated/.f4_jobs","w").write(":".join(jobs))
