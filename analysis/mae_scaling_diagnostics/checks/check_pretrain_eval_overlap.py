"""How many downstream-eval subjects did MAE pretraining already see (unlabeled)?

Section 11's pretraining pool includes `realdata`, which contains the very
datasets used as HAR evals (pamap2/recofit/utd_mhad/wisdm). This measures the
resulting transductive overlap: the fraction of each HAR probe's held-out val
subjects that were in the MAE pretraining TRAIN split.

SessionKey is a dataclass carrying (dataset, subject_id, session_id), so the
subject is read straight off the split keys -- no parquet join needed.
"""
import sys
sys.path.insert(0, "analysis/mae_scaling_diagnostics")
from lib.common import load_merged_cfg
from imu_lm.data.loaders import build_session_index, make_splits

PQ = "/projects/b53017/vbc1591/IMULM_master_dataset_v2.parquet"

def subj_by_ds(keys):
    out = {}
    for k in keys:
        out.setdefault(k.dataset, set()).add(k.subject_id)
    return out

mae_cfg = load_merged_cfg("configs/vit1d_ssl_mae_varlen_altattn_multidata_flu_recipe100pct.yaml")
seven = mae_cfg["objective"]["mae_multidata_varlen_altattn_flu"]["sources"]["realdata"]["train_on_datasets"]
print("MAE realdata sources:", seven)
print("MAE   splits:", {k: mae_cfg["splits"].get(k) for k in ("seed", "val_ratio", "group_key")}, flush=True)

si = build_session_index(PQ, mae_cfg, dataset_filter=list(seven))
sp = make_splits(si, mae_cfg)
mae_train = subj_by_ds(sp["train_keys"])
mae_val   = subj_by_ds(sp["val_keys"])

print("\nMAE pretraining realdata split, subjects per dataset:")
for ds in seven:
    tr, va = len(mae_train.get(ds, set())), len(mae_val.get(ds, set()))
    print(f"  {ds:<10s} train={tr:>4d}  val={va:>3d}  ({100*tr/max(1,tr+va):.0f}% of subjects in pretrain-train)")

print("\n=== HAR probe val subjects that MAE pretraining ALREADY SAW (unlabeled) ===")
for ds in ["wisdm", "utd_mhad", "pamap2", "recofit"]:
    h = load_merged_cfg(f"configs/vit1d_har_probe_{ds}_mae_recipe100pct_multidata_flu.yaml")
    hsi = build_session_index(PQ, h, dataset_filter=[ds])
    hsp = make_splits(hsi, h)
    hval = subj_by_ds(hsp["val_keys"]).get(ds, set())
    seen = mae_train.get(ds, set())
    ov = hval & seen
    print(f"  {ds:<10s} probe splits seed={h['splits'].get('seed')} val_ratio={h['splits'].get('val_ratio')} | "
          f"val_subj={len(hval):>3d}  seen_in_pretrain={len(ov):>3d} "
          f"({100*len(ov)/max(1,len(hval)):>3.0f}%)  truly_held_out={len(hval)-len(ov)}", flush=True)
