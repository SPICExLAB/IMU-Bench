#!/usr/bin/env python3
"""Pack 2x2-arm evaluation cells (4 per GPU) — all 10 HAR datasets + 3 core regression checks."""
import os, subprocess, sys, yaml, copy
sys.path.insert(0,"/gpfs/home/vbc1591/IMU-SSL"); os.chdir("/gpfs/home/vbc1591/IMU-SSL")
from bench.make_config import build_config
HAR=[f"har_{d}" for d in ["wisdm","utd_mhad","pamap2","recofit","hhar","mhealth","samosa","shoaib","ut_watch","prism"]]
CORE=["arm_pose_nymeria","full_body_pose_nymeria","microgesture_openwatch"]
NORM={"a":"zscore","b":"zscore","c":"physical","d":"physical"}
arm=sys.argv[1].lower(); dep=sys.argv[2] if len(sys.argv)>2 else None
eid=f"mae_nhanes2x2_{arm}"
cells=[t for t in HAR+CORE if not os.path.exists(f"bench/results/{t}/{eid}__probe__s42.json")]
print(f"arm {arm.upper()}: {len(cells)} cells to run")
for t in cells:                                    # materialise configs with the arm's own normalisation
    name,cfg=build_config(t,eid,"probe"); cfg=copy.deepcopy(cfg)
    if NORM[arm]=="physical":
        cfg.setdefault("preprocess",{}).setdefault("normalize",{}).update(
            {"enabled":True,"method":"physical","acc_scale":9.81,"gyro_scale":5.0,"clip_acc":8.0,"clip_gyro":8.0,"n_acc":3})
    yaml.safe_dump(cfg, open(f"bench/generated/configs/{name}.yaml","w"), sort_keys=False)
PACK=4
packs=[cells[i:i+PACK] for i in range(0,len(cells),PACK)]
os.makedirs("bench/generated/slurm/packs",exist_ok=True)
for i,pk in enumerate(packs):
    body=f"""#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --job-name=armpack{arm}{i}
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1
for i in $(seq 1 60); do [ -f runs/mae_nhanes2x2_arm{arm.upper()}/.done ] && break; sleep 60; done
"""
    for t in pk:
        run=f"bench_{t}__{eid}__probe__s42"
        body+=f"""
( if [ ! -f bench/results/{t}/{eid}__probe__s42.json ]; then
    python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/configs/{run}.yaml --run-name {run} &&
    python -m bench.score --task {t} --encoder {eid} --protocol probe --seed 42
  fi ) > logs/packcell_{run}.log 2>&1 &"""
    body+="\nwait\npython -m bench.leaderboard && python -m bench.leaderboard_html || true\necho ARMPACK_DONE\n"
    sh=f"bench/generated/slurm/packs/armpack{arm}{i}.sh"; open(sh,"w").write(body); os.chmod(sh,0o755)
    # SINGLE job per cell. Dual-partition twins both `rm -rf runs/{name}` and then train
    # into the same directory, silently corrupting the run (audit 2026-09-04: 27 bench
    # cells affected). Never re-introduce a second partition in this loop.
    for part in (["--account=p32139","--partition=gengpu"],):
        cmd=["sbatch","--parsable"]+part+([f"--dependency=afterany:{dep}"] if dep else [])+[sh]
        print(subprocess.check_output(cmd,text=True).strip(), os.path.basename(sh))
