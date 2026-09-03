#!/usr/bin/env python3
"""Pack the mask-ratio ablation's evaluation cells (4 probe cells per GPU).

    python bench/generated/pack_maskratio_cells.py 40 <pretrain_job_id>[,<twin_id>]
    python bench/generated/pack_maskratio_cells.py r0            # the 0.75 anchor's 2 missing cells

Cells per arm: the 6 leaderboard HAR probes + arm-pose probe + microgesture probe.
The 0.75 arm is encoder `mae_recipe_r0` (same config, only mask_ratio differs), whose
HAR probes were already launched with the recipe ablation -- only its arm-pose and
gesture cells are added here.
"""
import os, subprocess, sys

sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL"); os.chdir("/gpfs/home/vbc1591/IMU-SSL")
from bench.make_config import write_config

HAR = [f"har_{d}" for d in ["wisdm", "utd_mhad", "pamap2", "recofit", "hhar", "mhealth"]]
CORE = ["arm_pose_nymeria", "microgesture_openwatch"]

arm = sys.argv[1].lower()
dep = sys.argv[2] if len(sys.argv) > 2 else None
if arm == "r0":
    eid, run_done, tasks = "mae_recipe_r0", "mae_recipe_R0", list(CORE)
else:
    eid, run_done, tasks = f"mae_mr{arm}", f"mae_mr{arm}", HAR + CORE

cells = [t for t in tasks if not os.path.exists(f"bench/results/{t}/{eid}__probe__s42.json")]
print(f"arm {arm}: {len(cells)} cells -> {cells}")
for t in cells:
    write_config(t, eid, "probe", 42)

PACK = 4
packs = [cells[i:i + PACK] for i in range(0, len(cells), PACK)]
os.makedirs("bench/generated/slurm/packs", exist_ok=True)
for i, pk in enumerate(packs):
    job = f"mrpk{arm}{i}"
    body = f"""#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --job-name={job}
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1
# twin guard: this pack is queued on two partitions; the second copy exits if the first is live.
if squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -t RUNNING -o "%i" | grep -qv "^$SLURM_JOB_ID$"; then
  echo "[mrpack] twin RUNNING elsewhere -> skip"; exit 0; fi
# the pretrain arm must have finished (its launcher touches .done)
for i in $(seq 1 90); do [ -f runs/{run_done}/.done ] && break; sleep 60; done
if [ ! -f runs/{run_done}/.done ]; then echo "[mrpack] runs/{run_done}/.done never appeared -> abort"; exit 1; fi
"""
    for t in pk:
        run = f"bench_{t}__{eid}__probe__s42"
        body += f"""
( if [ ! -f bench/results/{t}/{eid}__probe__s42.json ]; then
    python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/configs/{run}.yaml --run-name {run} &&
    python -m bench.score --task {t} --encoder {eid} --protocol probe --seed 42
  fi ) > logs/packcell_{run}.log 2>&1 &"""
    body += "\nwait\npython -m bench.leaderboard && python -m bench.leaderboard_html || true\necho MRPACK_DONE\n"
    sh = f"bench/generated/slurm/packs/{job}.sh"
    open(sh, "w").write(body); os.chmod(sh, 0o755)
    for acct, part in (("b53017", "b53017"), ("p32139", "gengpu")):
        cmd = ["sbatch", "--parsable", f"--account={acct}", f"--partition={part}"]
        if dep: cmd.append("--dependency=afterany:" + dep.replace(",", ":"))
        cmd.append(sh)
        print(subprocess.check_output(cmd, text=True).strip(), part, os.path.basename(sh), pk)
