#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=z_followup
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
# wait for whichever physnp twin (gengpu / b53017 / gengpu-long) writes the .done marker
for i in $(seq 1 1400); do [ -f runs/mae_4src_z/.done ] && break; sleep 30; done
if [ ! -f runs/mae_4src_z/.done ]; then echo "physnp pretrain did not finish within the wait window"; exit 1; fi
echo "[followup] physnp pretrain complete -> submitting 9 HAR probe cells"
python - <<'PY'
import copy, os, subprocess, yaml, sys
sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL")
from bench.make_config import build_config
HAR = ["har_wisdm","har_utd_mhad","har_pamap2","har_recofit","har_hhar","har_mhealth","har_samosa","har_ut_watch","har_prism"]
os.makedirs("bench/generated/slurm/z9", exist_ok=True)
jobs = []
for t in HAR:
    name, cfg = build_config(t, "mae_4src_z", "probe"); cfg = copy.deepcopy(cfg)
    ot = cfg["objective"]["type"]
    cfg["objective"][ot]["class_weighted_ce"] = True
    cfg["trainer"]["early_stopping"] = {"enabled": True, "monitor": "bal_acc", "patience": 20}
    cfg["trainer"]["optim"]["lr"] = 1e-3
    yaml.safe_dump(cfg, open(f"bench/generated/configs/{name}.yaml", "w"), sort_keys=False)
    for part, acct, gres in (("gengpu", "p32139", "gpu:a100:1"), ("b53017", "b53017", "gpu:1")):
        sh = f"""#!/bin/bash
#SBATCH --account={acct}
#SBATCH --partition={part}
#SBATCH --gres={gres}
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=z{t.replace('har_','')[:9]}
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
if [ -f bench/results/{t}/mae_4src_z__probe__s42.json ]; then echo already-scored; exit 0; fi
python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/configs/{name}.yaml --run-name {name}
python -m bench.score --task {t} --encoder mae_4src_z --protocol probe --seed 42
"""
        p = f"bench/generated/slurm/z9/{name}_{part}.sh"; open(p, "w").write(sh); os.chmod(p, 0o755)
        jobs.append(subprocess.check_output(["sbatch", "--parsable", p], text=True).strip())
print(f"[followup] submitted {len(jobs)} jobs (9 cells x 2 partitions)")
PY
python analysis/_feat_diag.py 2>&1 | grep -E "eff-rank|ERR" || true
