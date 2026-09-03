#!/usr/bin/env python3
"""Wave-1 evaluation of the NHANES 2x2 arms: the cells that fit a 12 h budget.

Per arm (A..D): 6 HAR probes (the question) + arm-pose / full-body-pose / gesture probes
(the regression checks).  Odometry / language / PD / all finetunes are wave 2.
Each cell is a standard bench cell, so it goes through bench.make_config + bench.score
unchanged; the arm's encoder is registered on the fly in encoders.yaml.
"""
import os, subprocess, sys, yaml
sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL"); os.chdir("/gpfs/home/vbc1591/IMU-SSL")

ARMS = {"A": ("400 h NHANES + z-score", "zscore"), "B": ("20k h NHANES + z-score", "zscore"),
        "C": ("400 h NHANES + physical scaling", "physical"), "D": ("20k h NHANES + physical scaling", "physical")}
HAR = [f"har_{d}" for d in ["wisdm", "utd_mhad", "pamap2", "recofit", "hhar", "mhealth",
                          "samosa", "shoaib", "ut_watch", "prism"]]   # all 10 HAR eval sets
CORE = ["arm_pose_nymeria", "full_body_pose_nymeria", "microgesture_openwatch"]

def register():
    p = "bench/registry/encoders.yaml"; y = yaml.safe_load(open(p))
    for i, (arm, (desc, norm)) in enumerate(ARMS.items()):
        eid = f"mae_nhanes2x2_{arm.lower()}"
        if eid in y["encoders"]:
            continue
        y["encoders"][eid] = {
            "name": f"MAE 2x2 arm {arm} ({desc})", "short": f"2x2-{arm}", "group": "ablation", "order": 40 + i,
            "run": f"mae_nhanes2x2_arm{arm}", "ckpt": "best", "alt_attention": True, "channels": 6,
            "shared_kernel": "acc_gyro", "params_m": 21,
            "pretrain_data": f"LEAK-FREE 6-source pool (OCDetect, EatingSpeed, MotionX, SONIC, Capture24, NHANES) — no evaluation dataset appears; {desc}; balanced sampling, NHANES 25 % of mass; 50k steps",
            "pretrain_objective": "MAE (var-len, alt-attn) + modality loss weights acc 1.0 / gyro 2.0",
            "normalize_method": norm,
        }
    yaml.safe_dump(y, open(p, "w"), sort_keys=False)
    print("registered", ", ".join(f"mae_nhanes2x2_{a.lower()}" for a in ARMS))

def main():
    register()
    arm = sys.argv[1]
    eid = f"mae_nhanes2x2_{arm.lower()}"
    norm = ARMS[arm][1]
    for task in HAR + CORE:
        # the arm's own normalisation must be used downstream too
        from bench.make_config import build_config
        import copy
        name, cfg = build_config(task, eid, "probe")
        cfg = copy.deepcopy(cfg)
        if norm == "physical":
            cfg.setdefault("preprocess", {}).setdefault("normalize", {}).update({"enabled": True, "method": "physical", "acc_scale": 9.81,
                                                   "gyro_scale": 5.0, "clip_acc": 8.0, "clip_gyro": 8.0, "n_acc": 3})
        os.makedirs("bench/generated/configs", exist_ok=True)
        cp = f"bench/generated/configs/{name}.yaml"; yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
        for part in (["--account=p32139", "--partition=gengpu"], ["--account=b53017", "--partition=b53017"]):
            sh = f"bench/generated/slurm/{name}.sh"
            open(sh, "w").write(f"""#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name={name}
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1
if [ -f "bench/results/{task}/{eid}__probe__s42.json" ]; then echo "[w1] scored -> skip"; exit 0; fi
if squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -t RUNNING -o "%i" | grep -qv "^$SLURM_JOB_ID$"; then echo "[w1] twin RUNNING -> skip"; exit 0; fi
python scripts/run_pretrain.py --config configs/base.yaml --model-config {cp} --run-name {name}
python -m bench.score --task {task} --encoder {eid} --protocol probe --seed 42
python -m bench.leaderboard && python -m bench.leaderboard_html || true
""")
            os.chmod(sh, 0o755)
            print(subprocess.check_output(["sbatch", "--parsable"] + part + [sh], text=True).strip(), name)

if __name__ == "__main__":
    main()
