#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --job-name=release_hours
#SBATCH --output=bench/generated/tmp/%x_%j.out
#SBATCH --error=bench/generated/tmp/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL; source TransfHAR_env/bin/activate
python - <<'PY'
import numpy as np, logging, yaml, pyarrow.dataset as ds, pyarrow.compute as pc
logging.basicConfig(level=logging.ERROR)
# --- EatingSpeed: wall-clock recording hours per subset from the right-wrist pickles (64 Hz) ---
from imu_lm.data import eatingspeed_wrist_loader as L
for sub in ("MO", "FD-II", "FD-I"):
    d = L.load_subset("/projects/b53017/vbc1591/EatingSpeed", sub, hand="R")
    T = sum(len(a) for a in d["X_R"]); print(f"EatingSpeed {sub}: {len(d['X_R'])} participants, {T/64/3600:.1f} h wall-clock (right wrist stream)", flush=True)
# --- HAR pool sets: hours from the master parquet, rows/50 Hz and timestamp span per session ---
cfg = yaml.safe_load(open("configs/base.yaml")); path = cfg["paths"]["dataset_path"]
d = ds.dataset(path)
for name in ["ut_watch", "wisdm", "recofit", "pamap2", "samosa", "shoaib", "utd_mhad"]:
    t = d.to_table(columns=["timestamp_ns", "session_id", "subject_id"], filter=pc.field("dataset") == name).to_pandas()
    span = t.groupby(["subject_id", "session_id"])["timestamp_ns"].agg(lambda s: s.max() - s.min()).sum() / 1e9 / 3600
    print(f"HAR {name:9s}: rows {len(t):>10,d} -> {len(t)/50/3600:6.1f} h at 50 Hz | timestamp span {span:6.1f} h | sessions {t.groupby(['subject_id','session_id']).ngroups}", flush=True)
PY
