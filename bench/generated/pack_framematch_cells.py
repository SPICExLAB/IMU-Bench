#!/usr/bin/env python3
"""Pack the frame-match ablation's evaluation cells (4 probe cells per GPU).

    python bench/generated/pack_framematch_cells.py --arm canon --deps 123,124
    python bench/generated/pack_framematch_cells.py --arm ref        # r0 + scratch, no deps

The ablation is a 3 x 2 design over ONE frozen-probe protocol:

    pretraining frame          x    downstream frame
    ----------------------          ----------------------
    canon  no rotation              native   (the task's own frame)
    oct    24 cube rotations        rot7     (one fixed random SO(3) rotation,
    so3    uniform SO(3)                      frame_perturb seed 7, all splits)

plus two reference rows in the rotated frame only (their native cells are already
scored in bench/results): `mae_recipe_r0` (the production 514 h octahedral encoder)
and `scratch` (no pretraining at all -- the floor).

Each pack holds one cell from each of the four tasks, so no pack ever loads two
Nymeria pose datasets into RAM at once.
"""
import argparse, os, subprocess, sys

sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL"); os.chdir("/gpfs/home/vbc1591/IMU-SSL")
from bench.make_config import write_config

TASKS     = ["arm_pose_nymeria", "microgesture_openwatch", "har_wisdm", "har_pamap2"]
ROT_TASKS = [t + "_rot7" for t in TASKS]


def cells_for(arm: str):
    """[(task, encoder, protocol)] for one dependency group."""
    if arm == "ref":
        return ([(t, "mae_recipe_r0", "probe") for t in ROT_TASKS] +
                [(t, "scratch", "scratch") for t in ROT_TASKS])
    eid = f"mae_frame_{arm}"
    return [(t, eid, "probe") for t in TASKS + ROT_TASKS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["canon", "oct", "so3", "ref"])
    ap.add_argument("--deps", default="", help="comma-separated pretrain job ids to wait on")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    wait_run = None if a.arm == "ref" else f"mae_frame_{a.arm}"
    todo = [c for c in cells_for(a.arm)
            if not os.path.exists(f"bench/results/{c[0]}/{c[1]}__{c[2]}__s42.json")]
    print(f"arm {a.arm}: {len(todo)} cells to run")
    for t, e, p in todo:
        write_config(t, e, p, 42)

    PACK = 4
    packs = [todo[i:i + PACK] for i in range(0, len(todo), PACK)]
    os.makedirs("bench/generated/slurm/packs", exist_ok=True)
    for i, pk in enumerate(packs):
        job = f"fmpk{a.arm}{i}"
        body = f"""#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --job-name={job}
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1
# twin guard: this pack is queued on two partitions; the second copy exits if the first is live.
if squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -t RUNNING -o "%i" | grep -qv "^$SLURM_JOB_ID$"; then
  echo "[fmpack] twin RUNNING elsewhere -> skip"; exit 0; fi
"""
        if wait_run:
            body += f"""# the pretraining arm must have finished (its launcher touches .done). 6 h backstop
# in case the dependency fired on a twin that was skipped rather than the one that ran.
for i in $(seq 1 360); do [ -f runs/{wait_run}/.done ] && break; sleep 60; done
if [ ! -f runs/{wait_run}/.done ]; then echo "[fmpack] runs/{wait_run}/.done never appeared -> abort"; exit 1; fi
"""
        for t, e, p in pk:
            run = f"bench_{t}__{e}__{p}__s42"
            body += f"""
( if [ ! -f bench/results/{t}/{e}__{p}__s42.json ]; then
    python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/configs/{run}.yaml --run-name {run} &&
    python -m bench.score --task {t} --encoder {e} --protocol {p} --seed 42
  fi ) > logs/packcell_{run}.log 2>&1 &"""
        body += "\nwait\npython -m bench.leaderboard && python -m bench.leaderboard_html || true\necho FMPACK_DONE\n"

        sh = f"bench/generated/slurm/packs/{job}.sh"
        open(sh, "w").write(body); os.chmod(sh, 0o755)
        if a.dry_run:
            print("  would submit", sh, pk); continue
        # SINGLE job per cell. Dual-partition twins both `rm -rf runs/{name}` and then train
        # into the same directory, silently corrupting the run (audit 2026-09-04: 27 bench
        # cells affected). Never re-introduce a second partition in this loop.
        for acct, part in (("p32139", "gengpu"),):
            cmd = ["sbatch", "--parsable", f"--account={acct}", f"--partition={part}"]
            if a.deps:
                cmd.append("--dependency=afterany:" + a.deps.replace(",", ":"))
            cmd.append(sh)
            print(subprocess.check_output(cmd, text=True).strip(), part, job, [f"{t}/{e}" for t, e, _ in pk])


if __name__ == "__main__":
    main()
