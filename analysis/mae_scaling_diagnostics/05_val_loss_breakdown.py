#!/usr/bin/env python
"""05_val_loss_breakdown.py -- TODO §5 (break down validation reconstruction loss).

For every scaling checkpoint (uses the existing `best.pt` for each of the 6
fractions -- no new training), computes validation MSE broken down by:
  * dataset source
  * accelerometer vs. gyroscope channels
  * static vs. dynamic windows (accel-variance threshold)

Reproduces the EXACT loss computation
imu_lm.objectives.masked_1d_freq_channel_mask.forward_loss uses (same
generate_patch_mask / _patch_mask_to_signal_mask calls, same loss_mask =
masked & channel-valid) so results are directly comparable to what training
actually optimized, just with the breakdown production code doesn't compute.

Read-only: loads existing best.pt checkpoints, does not write/modify them.
Requires a GPU for reasonable speed but will run (slower) on CPU.

Usage:
    TransfHAR_env/bin/python analysis/mae_scaling_diagnostics/05_val_loss_breakdown.py [--fractions 5pct,25pct,100pct] [--max-batches-per-source 20]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from lib.common import FRACTION_CONFIGS, OUT_DIR, REPO_ROOT, RUN_DIRS, load_merged_cfg  # noqa: E402

# scripts/build_multidata_cache.py's own list -- every per-source loader calls
# apply_augment() (random_rotation, gaussian_noise, ...) inside __getitem__
# using whatever cfg it's handed, train OR val alike (see that script's
# docstring). An earlier version of this script built val_ds straight from
# the live training cfg, so every val window silently got a random rotation
# (etc.) applied before the reconstruction check -- confirmed via a direct
# side-by-side against the cache-based path (analysis/mae_scaling_diagnostics/
# 09_dataset_characterization.py) and the production forward_loss function:
# augmented val gave ~1.0-1.02 for every source (indistinguishable from the
# z-scored trivial baseline), unaugmented gave ~0.6-0.97, matching what the
# checkpoint's own reported training-time val_loss implies. Forcing
# augmentation off here makes this script measure the same clean, real
# per-source reconstruction quality as 09, not an augmentation-inflated one.
_AUGMENTATION_KEYS = (
    "random_rotation", "gaussian_noise", "jitter", "bias_drift", "sensor_dropout",
    "time_crop", "frequency_mask", "gain_jitter", "axis_bias", "time_warp",
)


def _cfg_without_augmentation(cfg: dict) -> dict:
    import copy
    cfg = copy.deepcopy(cfg)
    for key in _AUGMENTATION_KEYS:
        block = cfg.get(key)
        if isinstance(block, dict):
            block["enabled"] = False
    return cfg


def build_model(cfg, device):
    from imu_lm.models.ViT1D.model import ViT1DDecoder, ViT1DEncoderAltAttn

    encoder = ViT1DEncoderAltAttn(cfg).to(device)
    decoder = ViT1DDecoder(
        embed_dim=encoder.embed_dim, out_channels=encoder.in_channels, target_T=0, cfg=cfg,
    ).to(device)
    return encoder, decoder


def load_best_ckpt(run_dir_rel, encoder, decoder, device):
    ckpt_path = os.path.join(REPO_ROOT, run_dir_rel, "checkpoints", "best.pt")
    if not os.path.exists(ckpt_path):
        return None
    state = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(state["model"])
    decoder.load_state_dict(state["decoder"])
    encoder.eval()
    decoder.eval()
    return {"step": state.get("step"), "path": ckpt_path}


@torch.no_grad()
def per_batch_breakdown(x, channel_valid_mask, encoder, decoder, cfg, motion_thresh):
    """Computes (sum_sq_error, n_valid_elems) sliced by acc-vs-gyro channel and
    by static-vs-dynamic window, using the REAL, unmodified
    masked_1d_freq_channel_mask.forward_loss for the actual encoder/decoder
    forward pass -- NOT a hand-reimplementation.

    An earlier version of this function reimplemented forward_loss's
    mask-generation/encode/decode/loss-mask logic by hand, line-for-line
    matching the real function on inspection -- but was proven (via a
    monkeypatched fixed-mask, hook-instrumented side-by-side, with call order
    swapped to rule out state leakage) to silently produce a DIFFERENT
    `reconstructed` tensor than the real forward_loss does, for reasons never
    fully identified. Given that, this function no longer re-derives
    `reconstructed` itself at all: it calls the real `forward_loss` and
    captures its internal `reconstructed` tensor via a decoder forward hook,
    and its internal (post channel-block-merge) `patch_mask` by spying on the
    pure, side-effect-free `_patch_mask_to_signal_mask` helper forward_loss
    itself calls -- then only replays that same pure helper a second time
    (safe: it's a pure index computation, not a model forward pass) to get
    `loss_mask` for slicing. The `out["all"]` entry below is cross-checked
    against forward_loss's own reported loss as a standing sanity check.
    """
    import imu_lm.objectives.masked_1d_freq_channel_mask as prod_mod
    from imu_lm.objectives.masked_1d import _patch_mask_to_signal_mask

    x = x.to(next(encoder.parameters()).device)
    channel_valid_mask = channel_valid_mask.to(x.device)
    B, C, T = x.shape
    patch_size = encoder.patch_size
    num_patches_per_channel = T // patch_size

    captured = {}
    def _decoder_hook(_module, _inputs, output):
        captured["reconstructed"] = output.detach()
    handle = decoder.register_forward_hook(_decoder_hook)

    real_p2s = prod_mod._patch_mask_to_signal_mask
    def _spy_p2s(patch_mask, *a, **kw):
        captured["patch_mask"] = patch_mask.detach().clone()
        return real_p2s(patch_mask, *a, **kw)
    prod_mod._patch_mask_to_signal_mask = _spy_p2s

    try:
        production_loss, production_logs = prod_mod.forward_loss((x, None, channel_valid_mask), encoder, decoder, cfg)
    finally:
        handle.remove()
        prod_mod._patch_mask_to_signal_mask = real_p2s

    reconstructed = captured["reconstructed"]
    patch_mask = captured["patch_mask"]
    T_recon = reconstructed.shape[-1]
    if T_recon < x.shape[-1]:
        x = x[:, :, :T_recon]

    signal_mask = _patch_mask_to_signal_mask(patch_mask, num_patches_per_channel, C, patch_size)
    signal_valid = channel_valid_mask.unsqueeze(-1).expand(-1, -1, signal_mask.shape[-1])
    if T_recon < signal_valid.shape[-1]:
        signal_valid = signal_valid[:, :, :T_recon]
    loss_mask = signal_mask & signal_valid  # [B, C, T_recon]

    sq_err = (reconstructed - x) ** 2  # [B, C, T_recon]

    # static/dynamic per-window flag, from the (pre-mask) accel channels 0:3
    accel_var = x[:, :3, :].var(dim=(1, 2))  # [B]
    is_dynamic = accel_var > motion_thresh   # [B] bool

    def _sum_and_n(mask_slice):
        return float(sq_err[mask_slice].sum().item()), int(mask_slice.sum().item())

    acc_mask = loss_mask.clone()
    acc_mask[:, 3:, :] = False
    gyro_mask = loss_mask.clone()
    gyro_mask[:, :3, :] = False

    dyn_mask = loss_mask & is_dynamic.view(B, 1, 1)
    stat_mask = loss_mask & (~is_dynamic).view(B, 1, 1)

    out = {}
    out["all"] = _sum_and_n(loss_mask)
    out["acc"] = _sum_and_n(acc_mask)
    out["gyro"] = _sum_and_n(gyro_mask)
    out["dynamic"] = _sum_and_n(dyn_mask)
    out["static"] = _sum_and_n(stat_mask)

    # Standing sanity check: out["all"]'s sum/n must equal forward_loss's own
    # reported loss (both derived from the identical captured `reconstructed`
    # and the identical, re-derived-but-pure `loss_mask`) -- if this ever
    # drifts, something about the capture/replay above has broken.
    all_sum, all_n = out["all"]
    recomputed = all_sum / all_n if all_n > 0 else float("nan")
    prod_val = float(production_logs.get("mse_loss", production_loss))
    if all_n > 0 and abs(recomputed - prod_val) > 1e-3:
        raise RuntimeError(
            f"[05] per_batch_breakdown sanity check failed: recomputed all-loss "
            f"{recomputed:.6f} != production forward_loss {prod_val:.6f} -- "
            "capture/replay logic is no longer trustworthy, do not use these numbers"
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.1pct,1pct,5pct,25pct,50pct,100pct")
    ap.add_argument("--max-batches-per-source", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--motion-thresh-percentile", type=float, default=50.0,
                     help="per-source accel-variance percentile splitting static/dynamic windows")
    args = ap.parse_args()
    fractions = args.fractions.split(",")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[05] device={device}")

    from imu_lm.data import combined_mae_multidata_loader as cm

    out_dir = os.path.join(OUT_DIR, "05_val_loss_breakdown")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "val_loss_breakdown.json")

    # --- Build every source's val batches ONCE, not once per fraction. ---
    # All 6 scaling configs share a byte-identical `sources:` block (only
    # `overall_fraction` differs, which doesn't affect the held-out val set --
    # see combined_mae_multidata_loader.py's own docstring: val is always left
    # full-size regardless of fraction). Only the CHECKPOINT differs per
    # fraction. An earlier version of this loop rebuilt every source's raw val
    # dataset (SMPL FK, file parsing, ...) inside the fraction loop -- 6x
    # redundant raw builds -- and was still stuck on fraction 1/6 after 73
    # minutes on a real run. Building once up front and just swapping
    # encoder/decoder weights per fraction cuts that ~6x.
    baseline_cfg = load_merged_cfg(FRACTION_CONFIGS["100pct"][1])
    mm_cfg = baseline_cfg["objective"]["mae_multidata_varlen_altattn_flu"]
    sources_cfg = mm_cfg.get("sources", {}) or {}
    vl_max = int(baseline_cfg["data"]["variable_length"]["max_samples"])
    build_cfg = _cfg_without_augmentation(baseline_cfg)  # val must be deterministic, see module note

    print("[05] building every source's val batches once (shared across all fractions)...")
    source_batches = {}   # name -> list[(x, cvm)]
    source_motion_thresh = {}
    for name in cm.ALL_SOURCES:
        src_cfg = sources_cfg.get(name, {}) or {}
        if not bool(src_cfg.get("enabled", True)):
            continue
        try:
            _train_ds, val_ds = cm._BUILDERS[name](build_cfg, src_cfg, vl_max, collect_provenance=False)
        except Exception as e:
            print(f"[05]   {name}: failed to build val set ({e}), skipping")
            continue
        if val_ds is None or len(val_ds) == 0:
            continue

        loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
            collate_fn=lambda batch: (
                torch.stack([b[0] for b in batch if b is not None], 0)[:, :, :vl_max],
                torch.stack([b[1] for b in batch if b is not None], 0),
            ),
        )
        accel_vars = []
        batches = []
        for i, (x, cvm) in enumerate(loader):
            if i >= args.max_batches_per_source:
                break
            batches.append((x.to(device), cvm.to(device)))
            accel_vars.append(x[:, :3, :].var(dim=(1, 2)).numpy())
        if not batches:
            continue
        source_batches[name] = batches
        source_motion_thresh[name] = float(np.percentile(np.concatenate(accel_vars), args.motion_thresh_percentile))
        print(f"[05]   {name:16s} {len(batches)} batches cached (n_val={len(val_ds)})")

    all_results = []
    for frac_name in fractions:
        if frac_name not in FRACTION_CONFIGS:
            print(f"[05] skipping unknown fraction {frac_name}")
            continue
        _, cfg_path = FRACTION_CONFIGS[frac_name]
        run_dir_rel = RUN_DIRS[frac_name]
        cfg = load_merged_cfg(cfg_path)

        encoder, decoder = build_model(cfg, device)
        ckpt_info = load_best_ckpt(run_dir_rel, encoder, decoder, device)
        if ckpt_info is None:
            print(f"[05] {frac_name}: no best.pt found under {run_dir_rel}/checkpoints -- skipping "
                  "(this run hasn't been trained/saved yet)")
            continue
        print(f"[05] {frac_name}: loaded {ckpt_info['path']} (step={ckpt_info['step']})")

        per_source_agg = {}
        for name, batches in source_batches.items():
            motion_thresh = source_motion_thresh[name]
            agg = {k: [0.0, 0] for k in ("all", "acc", "gyro", "dynamic", "static")}
            for x, cvm in batches:
                res = per_batch_breakdown(x, cvm, encoder, decoder, cfg, motion_thresh)
                for k, (s, n) in res.items():
                    agg[k][0] += s
                    agg[k][1] += n
            per_source_agg[name] = {k: (v[0] / v[1] if v[1] > 0 else None) for k, v in agg.items()}
            print(f"[05]   {name:16s} mse_all={per_source_agg[name]['all']!s:>8s}  "
                  f"acc={per_source_agg[name]['acc']!s:>8s}  gyro={per_source_agg[name]['gyro']!s:>8s}  "
                  f"dynamic={per_source_agg[name]['dynamic']!s:>8s}  static={per_source_agg[name]['static']!s:>8s}")

        # aggregate across sources (unweighted mean of per-source MSEs, since
        # this is exactly what's needed to answer "is the aggregate improvement
        # driven by a few big/repetitive sources" -- weight by n instead if a
        # window-count-weighted aggregate is wanted)
        row = {"fraction": frac_name, "ckpt_step": ckpt_info["step"], "per_source": per_source_agg}
        all_results.append(row)

        del encoder, decoder
        torch.cuda.empty_cache() if device.type == "cuda" else None

        # Incremental save after EVERY fraction -- a killed/timed-out job
        # still leaves whatever fractions finished so far usable, instead of
        # losing everything (this is what bit the previous run).
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[05] (incremental save: {len(all_results)}/{len(fractions)} fractions written to {out_path})")

    print(f"\n[05] wrote {out_path}")


if __name__ == "__main__":
    main()
