#!/usr/bin/env python
"""12_recipe_val_rescore.py -- apples-to-apples val rescore of the
Section-11 recipe checkpoints on ONE identical, augmentation-free val set.
(Originally the 3-point 5/25/100% sweep; RUNS now also lists the 1%/10%
extension points, which are skipped with a MISSING message until trained.)

Why this exists (bug found 2026-07-28): the three recipe pretraining runs did
NOT see the same val set treatment, so their logged best_val_loss values
(0.7603 / 0.7579 / 0.7566) are not strictly comparable:

  - 5pct/25pct use scaling_mode=session_fraction, which forces
    collect_provenance=True, which makes make_combined_mae_multidata_loaders
    SKIP cache_dir entirely -> all 11 sources come from their raw builders.
    Those builders are INCONSISTENT about val augmentation: nymeria,
    openwatch and realdata gate on `split_name == "train"`, but
    totalcapture and capture24 (and the other cached-path sources) do not.
  - 100pct has no scaling_mode/overall_fraction -> collect_provenance=False
    -> all 11 sources load via CachedWindowDataset, whose __getitem__ applies
    apply_augment() UNCONDITIONALLY (combined_mae_multidata_loader.py:228),
    i.e. to val as well as train.

Confirmed from the run logs: `loaded from cache` count is 0 / 0 / 11 for
5pct / 25pct / 100pct. Val composition is identical (488,097 windows in all
three), only the augmentation treatment differs -- with random_rotation
(octahedral) enabled in all three configs.

This script removes that variable: it builds the val loader ONCE from the v3
cache with every augmentation block force-disabled, then scores all three
checkpoints against it with an identical per-batch mask sequence (torch seed
reset before each checkpoint), using the real
masked_1d_freq_channel_mask.forward_loss -- never a reimplementation (see
05_val_loss_breakdown.py's docstring for why that rule exists).

Does not modify any checkpoint, config, or existing result.
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from lib.common import OUT_DIR, REPO_ROOT, load_merged_cfg  # noqa: E402

# Same list 05_val_loss_breakdown.py / build_multidata_cache.py use.
_AUGMENTATION_KEYS = (
    "random_rotation", "gaussian_noise", "jitter", "bias_drift", "sensor_dropout",
    "time_crop", "frequency_mask", "gain_jitter", "axis_bias", "time_warp",
)

# The 100pct config is the one WITHOUT scaling_mode, so it loads from cache and
# leaves val untouched -- exactly the val loader we want, for all three.
BASE_CONFIG = "configs/vit1d_ssl_mae_varlen_altattn_multidata_flu_recipe100pct.yaml"
RUNS = {
    "1pct":   "runs/vit1d_ssl_mae_recipe_1pct",    # added 2026-08-08, skipped until trained
    "5pct":   "runs/vit1d_ssl_mae_recipe_5pct",
    "10pct":  "runs/vit1d_ssl_mae_recipe_10pct",   # added 2026-08-08, skipped until trained
    "25pct":  "runs/vit1d_ssl_mae_recipe_25pct",
    "100pct": "runs/vit1d_ssl_mae_recipe_100pct",
}
SEED = 1234


def _strip_aug(cfg):
    import copy
    cfg = copy.deepcopy(cfg)
    for key in _AUGMENTATION_KEYS:
        block = cfg.get(key)
        if isinstance(block, dict):
            block["enabled"] = False
    return cfg


def main():
    from imu_lm.data.combined_mae_multidata_loader import make_combined_mae_multidata_loaders
    from imu_lm.models.ViT1D.model import ViT1DDecoder, ViT1DEncoderAltAttn
    import imu_lm.objectives.masked_1d_freq_channel_mask as prod_mod

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[12] device={device}")

    cfg = _strip_aug(load_merged_cfg(BASE_CONFIG))
    for key in _AUGMENTATION_KEYS:
        blk = cfg.get(key)
        if isinstance(blk, dict) and blk.get("enabled"):
            raise RuntimeError(f"augmentation block '{key}' still enabled after strip")
    print("[12] all augmentation blocks force-disabled for this rescore")

    loaders = make_combined_mae_multidata_loaders(cfg)
    val_loader = loaders["val_loader"]
    print(f"[12] val windows = {len(val_loader.dataset)}")

    results = {}
    for name, run_dir in RUNS.items():
        ckpt_path = os.path.join(REPO_ROOT, run_dir, "checkpoints", "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[12] {name}: MISSING {ckpt_path}, skipping")
            continue

        encoder = ViT1DEncoderAltAttn(cfg).to(device)
        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim, out_channels=encoder.in_channels,
            target_T=0, cfg=cfg,
        ).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        encoder.load_state_dict(state["model"])
        decoder.load_state_dict(state["decoder"])
        encoder.eval(); decoder.eval()

        # Identical mask sequence for every checkpoint.
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        tot_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y, cvm = batch
                x = x.to(device, non_blocking=True)
                cvm = cvm.to(device, non_blocking=True)
                loss, _logs = prod_mod.forward_loss((x, y, cvm), encoder, decoder, cfg)
                tot_loss += float(loss.item())
                n_batches += 1

        mean_loss = tot_loss / max(1, n_batches)
        results[name] = {
            "clean_val_mse": mean_loss,
            "n_batches": n_batches,
            "ckpt_step": state.get("step"),
            "logged_best_val_loss": state.get("best_val_loss"),
        }
        print(f"[12] {name}: clean_val_mse={mean_loss:.6f} over {n_batches} batches "
              f"(ckpt step={state.get('step')}, originally-logged best_val_loss="
              f"{state.get('best_val_loss')})")
        del encoder, decoder
        torch.cuda.empty_cache()

    out_dir = os.path.join(OUT_DIR, "12_recipe_val_rescore")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "clean_val_rescore.json")
    with open(out_path, "w") as f:
        json.dump({"seed": SEED, "config": BASE_CONFIG, "results": results}, f, indent=2)
    print(f"\n[12] wrote {out_path}")

    if results:
        print("\n[12] === apples-to-apples (identical clean val, identical masks) ===")
        for name in ("5pct", "25pct", "100pct"):
            if name in results:
                r = results[name]
                print(f"  {name:>7s}: clean={r['clean_val_mse']:.6f}   "
                      f"(originally logged {r['logged_best_val_loss']:.6f})")


if __name__ == "__main__":
    main()
