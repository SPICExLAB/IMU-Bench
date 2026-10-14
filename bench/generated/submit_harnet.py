#!/usr/bin/env python3
"""HarNet-5s (Yuan et al. 2024) external baseline -- one job per bench cell.

First submission (2026-09-10) lost 20 of 22 cells in ~90 s each: every
downstream branch in imu_lm/models/ViT1D/run.py guards on a truthy
`pretrained_run`, and HarNet has none -- its weights are the authors' released
.mdl, not a run/ directory. bench/make_config.py now points `pretrained_run` at
the variant name and `pretrained_ckpt` at the .mdl path; the branches already
resolve a non-"best"/"latest" ckpt as an absolute path, and
HarnetEncoder.load_pretrained_state understands the release's
`module.feature_extractor.` key layout.

HarNet is a P2 reference row, not a P0 result, so it runs at Nice=500 and must
never sit in front of the 3-seed pretrains.

    python bench/generated/submit_harnet.py [--dry]
"""
import os, subprocess, sys, yaml

sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL")
os.chdir("/gpfs/home/vbc1591/IMU-SSL")
from bench.make_config import build_config

ENC = "harnet5"
HAR = ["har_wisdm", "har_utd_mhad", "har_pamap2", "har_recofit", "har_hhar",
       "har_mhealth", "har_samosa", "har_ut_watch", "har_prism"]
TASKS = HAR + ["microgesture_openwatch", "pd_pads", "language_humanml3d"]
LONG = {"language_humanml3d"}
BAD_NODES = "qgpu3006,qgpu3013,qgpu2014,qgpu2006"
NICE = 500

dry = "--dry" in sys.argv
os.makedirs("bench/generated/slurm/harnet", exist_ok=True)

jobs = []
for task in TASKS:
    for proto in ("probe", "finetune"):
        name = f"bench_{task}__{ENC}__{proto}__s42"
        res = f"bench/results/{task}/{ENC}__{proto}__s42.json"
        if os.path.exists(res):
            print(f"[have] {name}")
            continue
        try:
            _, cfg = build_config(task, ENC, proto)
        except Exception as exc:                            # noqa: BLE001
            print(f"[skip] {name}: {exc}"); continue
        yaml.safe_dump(cfg, open(f"bench/generated/configs/{name}.yaml", "w"), sort_keys=False)

        tl = "10:00:00" if task in LONG else "06:00:00"
        mem = "80G" if task in LONG else "48G"
        sh = f"""#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --exclude={BAD_NODES}
#SBATCH --gres=gpu:a100:1
#SBATCH --time={tl}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=8
#SBATCH --nice={NICE}
#SBATCH --job-name=H{task[:10]}{proto[:2]}
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1
if [ -f {res} ]; then echo already-scored; exit 0; fi
rm -rf runs/{name}
python scripts/run_pretrain.py --config configs/base.yaml \\
  --model-config bench/generated/configs/{name}.yaml --run-name {name}
python -m bench.score --task {task} --encoder {ENC} --protocol {proto} --seed 42
"""
        p = f"bench/generated/slurm/harnet/{name}.sh"
        open(p, "w").write(sh); os.chmod(p, 0o755)
        if dry:
            print(f"[dry] {name}"); continue
        jobs.append(subprocess.check_output(["sbatch", "--parsable", p], text=True).strip())

print(f"submitted {len(jobs)} cells" if not dry else "dry run complete")
