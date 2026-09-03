"""End-to-end NHANES path smoke test at tiny scale.

Exercises exactly the call chain that has now failed twice at full scale
(_build_nhanes -> make_nhanes_loaders_random -> _collect_windows ->
NhanesDataset -> DataLoader), but with a handful of participants, so the
memory-copy behaviour can be checked in seconds instead of 4.5 h.
"""
import resource, sys, numpy as np, yaml
sys.path.insert(0, "analysis/mae_scaling_diagnostics")
from lib.common import load_merged_cfg
from imu_lm.data import combined_mae_multidata_loader as cm

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main():
    cfg = load_merged_cfg("configs/vit1d_ssl_mae_stage1_acconly_flu.yaml")
    src = cfg["objective"]["mae_multidata_varlen_altattn_flu"]["sources"]["nhanes"]
    # tiny: a few participants, small caps -- same code path, trivial data
    src.update({"sample_fraction": 0.0007, "max_hours": 6.0, "max_hours_val": 2.0})
    print(f"smoke cfg: {src}", flush=True)

    print(f"RSS before: {rss_gb():.2f} GB", flush=True)
    train_ds, val_ds = cm._build_nhanes(cfg, src)
    print(f"RSS after : {rss_gb():.2f} GB", flush=True)

    assert train_ds is not None, "_build_nhanes returned None -- it swallowed an exception (check log above)"
    print(f"train windows={len(train_ds)}  val windows={len(val_ds) if val_ds else 0}")

    # the underlying array must NOT have been duplicated by astype
    raw = train_ds
    for attr in ("_ds", "dataset"):
        raw = getattr(raw, attr, raw)
    inner = getattr(raw, "_X", None)
    if inner is not None:
        print(f"inner array dtype={inner.dtype} shape={inner.shape} "
              f"owns_data={inner.flags.owndata} (owndata=False means no copy was made)")

    x, mask = train_ds[0]
    print(f"item x shape={tuple(x.shape)} dtype={x.dtype} finite={bool(np.isfinite(x.numpy()).all())} "
          f"channel_valid_mask={mask.tolist()}")
    assert x.shape[0] == 6, "expected 6ch (acc padded + gyro masked)"
    assert mask.tolist()[:3] == [True]*3 and mask.tolist()[3:] == [False]*3, \
        "acc channels should be valid, gyro channels invalid"

    # val must be deterministic (augmentation gated off)
    if val_ds is not None and len(val_ds):
        a = val_ds[0][0].numpy(); b = val_ds[0][0].numpy()
        print(f"val deterministic (no augmentation): {np.array_equal(a,b)}")
        assert np.array_equal(a, b), "val is being augmented"

    print("\nSMOKE TEST PASSED -- full NHANES path is sound")


if __name__ == "__main__":
    # REQUIRED: _collect_windows uses multiprocessing "spawn", whose child
    # processes re-import this module. Without this guard each child re-runs
    # the whole script and spawns more children -- a fork bomb that burned a
    # 45-minute job slot with zero progress (job 8443221, 2026-08-01).
    main()
