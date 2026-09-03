#!/usr/bin/env python3
"""Finalize the full raw pool -> diversity weights -> 60% mask DC-preserving pretrain -> HAR probes.
Runs as a SLURM job chained afterok on the four raw full-size rebuilds."""
import os, sys, json, subprocess, yaml, copy
sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL"); os.chdir("/gpfs/home/vbc1591/IMU-SSL")
RAW = "/projects/b53017/vbc1591/mae_multidata_cache_raw"
# diversity-selected subsets per source: SUBSETS="nhanes=/path/idx.npy,capture24=/path/idx.npy,..."
_default = "nhanes=/gpfs/home/vbc1591/IMU-SSL/analysis/nhanes_diversity/out/indices_balanced_272h.npy"
SUBSETS = dict(kv.split("=", 1) for kv in os.environ.get("SUBSETS", _default).split(",") if kv)
for _s, _p in SUBSETS.items():
    assert os.path.exists(_p), f"missing diversity index for {_s}: {_p}"
    os.environ[f"SUBSET_{_s}"] = _p          # source_coverage.py restricts that source's pool to the subset
# 1) pool summary
import numpy as np, glob
tot = 0
print("=== full raw pool ===")
for f in sorted(glob.glob(f"{RAW}/*_train_x.npy")):
    s = os.path.basename(f).replace("_train_x.npy", "")
    if s == "realdataclean": continue
    n = np.load(f, mmap_mode="r").shape[0]
    t = f.replace("_x.npy", "_tmask.npy")
    # windows are 512 samples (10.24 s) with a 256-sample hop -> each full window adds 5.12 s of
    # NEW signal; a short clip's single window adds its whole valid length (tmask)
    h = (np.where(np.asarray(np.load(t)) >= 512, 256, np.asarray(np.load(t))).sum() / 50 / 3600) if os.path.exists(t) else n * 5.12 / 3600
    note = ""
    if s in SUBSETS:
        nsel = len(np.load(SUBSETS[s])); note = f"  <- diversity-selected {nsel} of {n} windows ({nsel*5.12/3600:.0f} h) used"; n, h = nsel, nsel * 5.12 / 3600
    tot += h; print(f"  {s:13s} {n:>9} windows {h:8.1f} h{note}")
print(f"  TOTAL {tot:.0f} h")
# 2) diversity weights on this exact pool
env = dict(os.environ, CACHE=RAW)
subprocess.run([sys.executable, "analysis/diversity_saturation/source_coverage.py"], check=True, env=env)
w = json.load(open("analysis/diversity_saturation/out/source_coverage.json"))["suggested_source_weights"]
w.pop("realdataclean", None)
print("weights:", {k: round(v, 2) for k, v in w.items()})
# 3) pretrain config: Rraw recipe + mask 0.60 + full pool + diversity weights + 100k steps
c = yaml.safe_load(open("configs/nhanes_scale/mae_recipe_Rraw.yaml"))
K = "mae_multidata_varlen_altattn_flu"; ob = c["objective"][K]
ob["cache_dir"] = RAW; ob["sampling_mode"] = "balanced"; ob["source_weights"] = dict(w); ob["epoch_windows"] = 200000
for n, s in ob["sources"].items():
    s["enabled"] = n in w
    for cap in ("max_hours_cached", "max_windows"): s.pop(cap, None)
    if n in SUBSETS: s["subset_index_file"] = SUBSETS[n]   # diversity-balanced subset (analysis/nhanes_diversity.py --source n)
c["objective"]["mask_ratio"] = 0.60
c["objective"]["cache_is_normalized"] = False; ob["cache_is_normalized"] = False
c["preprocess"]["normalize"] = {"enabled": False}                     # DC preserved
c["data"]["variable_length"]["include_short_clips"] = True
c["trainer"]["max_steps"] = 100000
c["trainer"]["early_stopping"] = {"enabled": True, "monitor": "val_loss", "patience": 999999, "min_delta": 0.0001}
c["_recipe_note"] = "FULL raw pool (short mocap clips recovered, diversity-balanced subsets: " + ",".join(f"{k}={len(np.load(v))*5.12/3600:.0f}h" for k, v in SUBSETS.items()) + "), mask 0.60, DC-preserving, diversity-weighted balanced sampling, octahedral rot, lr 1e-4, 100k steps"
yaml.safe_dump(c, open("configs/nhanes_scale/mae_full60_dc.yaml", "w"), sort_keys=False)
print("wrote configs/nhanes_scale/mae_full60_dc.yaml")
# 4) register encoder + launch pretrain + chain HAR probes
y = yaml.safe_load(open("bench/registry/encoders.yaml"))
y["encoders"]["mae_full60_dc"] = {"name": "MAE full-pool 60% mask, DC-preserving, diversity-weighted", "short": "MAE-full60-DC",
    "group": "ablation", "order": 60, "run": "mae_full60_dc", "ckpt": "best", "alt_attention": True, "channels": 6,
    "shared_kernel": "acc_gyro", "params_m": 21, "pretrain_data": f"leak-free raw pool ~{tot:.0f} h (7 sources, short mocap clips recovered), diversity-weighted",
    "pretrain_objective": "MAE mask 0.60, raw/DC-preserving, octahedral, lr 1e-4, 100k steps", "normalize_method": "raw"}
yaml.safe_dump(y, open("bench/registry/encoders.yaml", "w"), sort_keys=False)
sh = """#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --job-name=mae_full60_dc
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
if [ -f runs/mae_full60_dc/.done ]; then exit 0; fi
python -u scripts/run_pretrain.py --config configs/base.yaml --model-config configs/nhanes_scale/mae_full60_dc.yaml --run-name mae_full60_dc --resume latest && touch runs/mae_full60_dc/.done
"""
open("launch/launch_slurm_mae_full60_dc.sh", "w").write(sh); os.chmod("launch/launch_slurm_mae_full60_dc.sh", 0o755)
pj = subprocess.check_output(["sbatch", "--parsable", "launch/launch_slurm_mae_full60_dc.sh"], text=True).strip()
print("pretrain job", pj)
from bench.make_config import build_config
DS = [f"har_{d}" for d in ["wisdm", "utd_mhad", "pamap2", "recofit", "hhar", "mhealth", "samosa", "ut_watch", "prism"]]
cells = []
for t in DS:
    run = f"bench_{t}__mae_full60_dc__probe__s42"
    name, cfg = build_config(t, "mae_full60_dc", "probe"); cfg = copy.deepcopy(cfg)
    cfg["preprocess"]["normalize"] = {"enabled": False}; cfg.setdefault("trainer", {})["extra_best"] = {"macro_f1": "max"}
    yaml.safe_dump(cfg, open(f"bench/generated/configs/{run}.yaml", "w"), sort_keys=False); cells.append((run, t))
for i in range(0, len(cells), 3):
    pk = cells[i:i+3]; k = i // 3
    body = f"""#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --job-name=f60pk{k}
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
for i in $(seq 1 2000); do [ -f runs/mae_full60_dc/.done ] && break; sleep 60; done
"""
    for run, t in pk:
        body += f"""
( if [ ! -f bench/results/{t}/mae_full60_dc__probe__s42.json ]; then
    python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/configs/{run}.yaml --run-name {run} &&
    python -m bench.score --task {t} --encoder mae_full60_dc --protocol probe --seed 42
  fi ) > logs/packcell_{run}.log 2>&1 &"""
    body += "\nwait\npython -m bench.leaderboard && python -m bench.leaderboard_html || true\necho F60_DONE\n"
    p = f"bench/generated/slurm/packs/f60pk{k}.sh"; open(p, "w").write(body); os.chmod(p, 0o755)
    subprocess.check_output(["sbatch", "--parsable", f"--dependency=afterany:{pj}", p], text=True)
print("chained 9 HAR probe cells")
