#!/usr/bin/env python
"""Per-dataset window counts of the HAR pool ("realdata") in the reference pretraining corpus.

The cached arrays carry no provenance, so the split is reproduced from the same session index, split rule and
window specs the pretraining loader used (imu_lm.data.loaders: build_session_index -> make_splits ->
_build_window_specs), without reading any signal. The train total must equal the cached realdata train split
(93,510 windows), which is asserted. Writes analysis/bench_stats/har_pool_per_dataset.json.
"""
import json, os, sys
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from imu_lm.data.loaders import SessionKey, build_session_index, make_splits, _build_window_specs
from imu_lm.utils.helpers import cfg_get

REF = "configs/corpus_ablation/vit1d_ssl_mae_corpus_ref.yaml"     # same realdata block as mae_5src / every 5-source baseline
def merge(a, b):
    for k, v in b.items(): a[k] = merge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a
cfg = merge(yaml.safe_load(open("configs/base.yaml")), yaml.safe_load(open(REF)))
src = cfg["objective"]["mae_multidata_varlen_altattn_flu"]["sources"]["realdata"]
train_on = src.get("train_on_datasets") or cfg_get(cfg, ["data", "train_on_datasets"], None)
si = build_session_index(cfg_get(cfg, ["paths", "dataset_path"]), cfg, dataset_filter=list(train_on))
splits = make_splits(si, cfg)
max_gaps = int(cfg_get(cfg, ["windowing", "max_gaps_per_session"], 1_000_000_000))
gap, nrows = {}, {}
for _, r in si.iterrows():
    k = SessionKey(r["dataset"], str(r["subject_id"]), str(r["session_id"])); gap[k] = int(r.get("gap_count", 0)); nrows[k] = int(r["n_rows"])
out = {}
for split in ("train", "val"):
    keys = [k for k in splits[f"{split}_keys"] if gap.get(k, 0) <= max_gaps]
    kept, sess_idx, starts, counters = _build_window_specs(keys, nrows, cfg)
    per = {}
    for i in sess_idx.tolist():
        per[kept[i].dataset] = per.get(kept[i].dataset, 0) + 1
    out[split] = {"total": int(counters.get("windows", 0)), "per_dataset": dict(sorted(per.items()))}
    print(split, out[split])
assert out["train"]["total"] == 93510, out["train"]["total"]
json.dump({"config": REF, "hop_s": 5.12, **out}, open("analysis/bench_stats/har_pool_per_dataset.json", "w"), indent=1)
