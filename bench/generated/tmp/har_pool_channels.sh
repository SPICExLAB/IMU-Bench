#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --job-name=har_pool_channels
#SBATCH --output=bench/generated/tmp/%x_%j.out
#SBATCH --error=bench/generated/tmp/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL; source TransfHAR_env/bin/activate
python - <<'PY'
import pyarrow.dataset as ds, pyarrow.compute as pc, numpy as np, yaml
cfg = yaml.safe_load(open("configs/base.yaml"))
path = cfg["paths"]["dataset_path"]; cols = cfg["data"]["sensor_columns"]
print("parquet:", path, "| sensor_columns:", cols)
d = ds.dataset(path)
for name in ["pamap2", "recofit", "samosa", "shoaib", "ut_watch", "utd_mhad", "wisdm"]:
    t = d.to_table(columns=cols + ["subject_id"], filter=pc.field("dataset") == name)
    n = t.num_rows
    if n == 0:
        print(f"{name:9s}: no rows"); continue
    out = []
    for c in cols:
        a = t[c].to_numpy(zero_copy_only=False).astype(float)
        nn = np.isfinite(a); frac = nn.mean(); v = a[nn]
        out.append(f"{c}: nonnull {frac*100:5.1f}%  std {v.std():7.3f}  |p99| {np.percentile(np.abs(v),99):7.3f}")
    subj = len(set(t["subject_id"].to_pylist()))
    print(f"{name:9s}: rows {n:>10,d}  subjects {subj:3d}  ->  " + " | ".join(out), flush=True)
PY
