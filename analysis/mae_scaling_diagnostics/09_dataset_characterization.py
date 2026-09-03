#!/usr/bin/env python
"""09_dataset_characterization.py -- per-source dataset comparison.

Not part of the numbered §1-8 TODO list -- a follow-up request to characterize
how the 11 pretraining sources actually differ from each other: signal
statistics (mean/var per channel), sample rate, coordinate-frame convention,
native activity/label taxonomy, and reconstruction accuracy under a trained
MAE checkpoint.

Deliberately reads from the existing on-disk CACHE
(/projects/b53017/vbc1591/mae_multidata_cache) instead of rebuilding sources
from raw -- fast (plain np.load + a few forward passes), safe to run on the
login node (no multiprocessing raw parsing, which OOM'd/hung interactively
for other scripts in this directory), and answers this specific question
without waiting on the multi-hour 01_composition_audit.py SLURM job.

Reconstruction MSE uses the 100% run's checkpoint (runs/vit1d_ssl_mae_varlen_altattn_multidata/checkpoints/best.pt)
as the representative, most-trained model; augmentation is forced off (same
convention as scripts/build_multidata_cache.py) so numbers are deterministic
and comparable across sources.

Usage:
    TransfHAR_env/bin/python analysis/mae_scaling_diagnostics/09_dataset_characterization.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from lib.common import FRACTION_CONFIGS, OUT_DIR, REPO_ROOT, RUN_DIRS, load_merged_cfg  # noqa: E402

CACHE_DIR = "/projects/b53017/vbc1591/mae_multidata_cache"
MAX_SAMPLE_WINDOWS = 3000  # cap for mean/var + reconstruction MSE, for speed on huge sources
                           # (run via SLURM with a real GPU -- see
                           # launch_slurm_dataset_characterization.sh -- the interactive
                           # login-node session OOM-kills this script even at 800)

# Hand-compiled from reading every per-source loader this session (native_hz /
# target_hz constants, FLU-vs-raw transform code, label/annotation schemes).
# All are canonicalized to CANONICAL_HZ=50 by combined_mae_multidata_loader
# regardless of native rate -- "native_hz" here is what the raw recording
# actually was before resampling.
METADATA = {
    "realdata": {
        "native_hz": "varies by sub-dataset (pamap2/recofit/samosa/shoaib/ut_watch/utd_mhad/wisdm)",
        "coordinate_frame": "raw device frame (no FLU transform -- imu_lm/data/loaders.py has none)",
        "channels": "acc+gyro (6ch)",
        "activity_labels": "discrete HAR activity classes (dataset_activity_id), per-subdataset taxonomy "
                            "(walking/running/cycling/etc., varies by which of the 7 sub-datasets)",
        "real_or_synthetic": "real",
    },
    "ocdetect": {
        "native_hz": 50,
        "coordinate_frame": "FLU (apply_flu_transform=True)",
        "channels": "acc+gyro (6ch)",
        "activity_labels": "binary: compulsive-behavior episode vs. background (OCD hand-washing/checking detection)",
        "real_or_synthetic": "real",
    },
    "nymeria": {
        "native_hz": "Aria glasses native rate, resampled to 50",
        "coordinate_frame": "FLU",
        "channels": "acc+gyro (6ch)",
        "activity_labels": "no discrete class label used here (loader strips to raw IMU + 24-joint SMPL pose "
                            "targets for MAE) -- source dataset has egocentric daily-activity atomic-action annotations",
        "real_or_synthetic": "real",
    },
    "totalcapture": {
        "native_hz": 60,
        "coordinate_frame": "FLU",
        "channels": "acc+gyro (6ch)",
        "activity_labels": "no discrete class label used here (pose-regression target only) -- source is "
                            "indoor/sports mocap (DIP-IMU / DIP_recalculate)",
        "real_or_synthetic": "real",
    },
    "imuposer": {
        "native_hz": "resampled to 50",
        "coordinate_frame": "FLU (despite add_gravity=False by default)",
        "channels": "acc+gyro (6ch)",
        "activity_labels": "no discrete class label used here (pose-regression target only)",
        "real_or_synthetic": "real",
    },
    "eatingspeed": {
        "native_hz": 64,
        "coordinate_frame": "raw sensor frame -- NO FLU transform (documented explicitly in "
                             "eatingspeed_wrist_loader.py: 'no coordinate transform is applied')",
        "channels": "acc+gyro (6ch)",
        "activity_labels": "eating-speed condition per subset: FD-I / FD-II / MO (fast/slow-eating-speed "
                            "protocol conditions, not a general activity taxonomy)",
        "real_or_synthetic": "real",
    },
    "motionx": {
        "native_hz": 30,
        "coordinate_frame": "FLU (synthetic, derived via SMPL FK from AMASS-style mocap)",
        "channels": "acc+gyro (6ch, synthetic gyro incl. optional injected white noise/bias drift)",
        "activity_labels": "large free-text motion-category/caption vocabulary (Motion-X++), not used by MAE "
                            "(labels stripped)",
        "real_or_synthetic": "synthetic (derived IMU from mocap)",
    },
    "humanml3d": {
        "native_hz": "20 (source mocap), resampled to 50 via SMPL FK",
        "coordinate_frame": "FLU (synthetic, same SMPL-FK-derivation approach as nymeria/motionx)",
        "channels": "acc+gyro (6ch)",
        "activity_labels": "free-text motion captions (AMASS/HumanML3D), not used by MAE (labels stripped)",
        "real_or_synthetic": "synthetic (derived IMU from mocap)",
    },
    "sonic": {
        "native_hz": "BVH source fps, resampled to 50",
        "coordinate_frame": "FLU",
        "channels": "acc+gyro (6ch)",
        "activity_labels": "none (raw BVH mocap, bones-seed) -- pure motion diversity source, no labels at all",
        "real_or_synthetic": "synthetic (derived IMU from BVH mocap)",
    },
    "capture24": {
        "native_hz": 100,
        "coordinate_frame": "raw device frame -- NO FLU transform (documented explicitly in capture24_loader.py: "
                             "'axis convention -- kept raw, no coordinate transform applied')",
        "channels": "accelerometer-ONLY (3ch real + 3ch zero-padded/masked-invalid, see AccOnlyPadTo6ChDataset)",
        "activity_labels": "free-living activity taxonomy (Capture-24 / Willetts et al.: sleep, sit-stand, "
                            "walking, bicycling, vehicle, mixed, etc.)",
        "real_or_synthetic": "real",
    },
    "openwatch": {
        "native_hz": "resampled to fixed window length",
        "coordinate_frame": "configurable, FLU in production config",
        "channels": "acc+gyro (6ch), optional PPG (unused here)",
        "activity_labels": "discrete gesture classes (6-class gesture taxonomy in the production probe configs)",
        "real_or_synthetic": "real",
    },
}


def cache_stats(name: str):
    x_path = os.path.join(CACHE_DIR, f"{name}_train_x.npy")
    mask_path = os.path.join(CACHE_DIR, f"{name}_train_mask.npy")
    if not (os.path.exists(x_path) and os.path.exists(mask_path)):
        return None
    X = np.load(x_path, mmap_mode="r")
    mask = np.load(mask_path)  # [N, 6] bool, per-source constant (all-True except capture24)
    n = len(X)
    n_sample = min(n, MAX_SAMPLE_WINDOWS)
    idx = np.sort(np.random.default_rng(0).choice(n, size=n_sample, replace=False)) if n > n_sample else np.arange(n)
    Xs = np.asarray(X[idx]).astype(np.float32)  # [n_sample, 6, T]
    ch_valid = mask[0]  # channel validity is constant across all windows for a source

    means, stds = [], []
    for c in range(6):
        if not ch_valid[c]:
            means.append(None)
            stds.append(None)
            continue
        vals = Xs[:, c, :].reshape(-1)
        means.append(float(vals.mean()))
        stds.append(float(vals.std()))

    val_x_path = os.path.join(CACHE_DIR, f"{name}_val_x.npy")
    n_val = len(np.load(val_x_path, mmap_mode="r")) if os.path.exists(val_x_path) else 0

    return {
        "n_train_cached": int(n), "n_val_cached": int(n_val),
        "channel_valid": ch_valid.tolist(),
        "mean_per_channel": means, "std_per_channel": stds,
        "T": int(X.shape[-1]),
    }


def build_model(cfg, device):
    from imu_lm.models.ViT1D.model import ViT1DDecoder, ViT1DEncoderAltAttn
    encoder = ViT1DEncoderAltAttn(cfg).to(device)
    decoder = ViT1DDecoder(embed_dim=encoder.embed_dim, out_channels=encoder.in_channels,
                            target_T=0, cfg=cfg).to(device)
    return encoder, decoder


@torch.no_grad()
def reconstruction_mse(name: str, encoder, decoder, cfg, device, vl_max: int):
    # Delegates to the ACTUAL production loss function
    # (imu_lm.objectives.masked_1d_freq_channel_mask.forward_loss) instead of
    # reimplementing the masking/loss math by hand -- an earlier hand-rolled
    # version of this function was verified to disagree with the production
    # loss by a wide margin (~1.01 vs ~0.81 mean across sources) on a direct
    # side-by-side check; not worth the risk of a second subtle bug, so this
    # now just calls the real thing and batches/aggregates its per-batch mean.
    from imu_lm.objectives.masked_1d_freq_channel_mask import forward_loss

    x_path = os.path.join(CACHE_DIR, f"{name}_val_x.npy")
    mask_path = os.path.join(CACHE_DIR, f"{name}_val_mask.npy")
    if not (os.path.exists(x_path) and os.path.exists(mask_path)):
        return None
    X = np.load(x_path, mmap_mode="r")
    mask = np.load(mask_path)
    n = len(X)
    if n == 0:
        return None
    n_sample = min(n, MAX_SAMPLE_WINDOWS)
    idx = np.sort(np.random.default_rng(1).choice(n, size=n_sample, replace=False)) if n > n_sample else np.arange(n)

    total_loss, total_batches = 0.0, 0
    batch_size = 128
    for start in range(0, n_sample, batch_size):
        bidx = idx[start:start + batch_size]
        xb = torch.from_numpy(np.asarray(X[bidx]).astype(np.float32))[:, :, :vl_max].to(device)
        cvm = torch.from_numpy(mask[bidx]).to(device)
        loss, _logs = forward_loss((xb, None, cvm), encoder, decoder, cfg)
        total_loss += float(loss) * len(bidx)
        total_batches += len(bidx)

    return total_loss / total_batches if total_batches > 0 else None


def main():
    import argparse
    global CACHE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=CACHE_DIR,
                     help="which cache to read window/mean/var stats from -- must match whatever "
                          "config generated it (e.g. mae_multidata_cache_v3 for the 100%% "
                          "ocdetect/capture24/eatingspeed rebuild); default is the original cache")
    ap.add_argument("--out-subdir", default="09_dataset_characterization",
                     help="subdirectory under out/ to write results into -- use a distinct name "
                          "per cache version so reruns don't silently overwrite each other")
    args = ap.parse_args()
    CACHE_DIR = args.cache_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[09] device={device}, cache_dir={CACHE_DIR}")

    cfg = load_merged_cfg(FRACTION_CONFIGS["100pct"][1])
    vl_max = int(cfg["data"]["variable_length"]["max_samples"])

    encoder = decoder = None
    ckpt_path = os.path.join(REPO_ROOT, RUN_DIRS["100pct"], "checkpoints", "best.pt")
    if os.path.exists(ckpt_path):
        encoder, decoder = build_model(cfg, device)
        state = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(state["model"])
        decoder.load_state_dict(state["decoder"])
        encoder.eval()
        decoder.eval()
        print(f"[09] loaded checkpoint {ckpt_path} (step={state.get('step')}) for reconstruction MSE")
    else:
        print(f"[09] WARNING: {ckpt_path} not found -- skipping reconstruction MSE")

    out_dir = os.path.join(OUT_DIR, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    def write_outputs(rows):
        with open(os.path.join(out_dir, "dataset_characterization.json"), "w") as f:
            json.dump(rows, f, indent=2)
        md_path = os.path.join(out_dir, "dataset_characterization.md")
        with open(md_path, "w") as f:
            f.write("| source | real/synth | native_hz | coord frame | channels | n_train | n_val | "
                    "acc mean (x,y,z) | acc std (x,y,z) | recon MSE |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for r in rows:
                am = r["mean_per_channel"][:3]
                asd = r["std_per_channel"][:3]
                am_s = ", ".join(f"{v:.3f}" if v is not None else "n/a" for v in am)
                asd_s = ", ".join(f"{v:.3f}" if v is not None else "n/a" for v in asd)
                mse_s = f"{r['reconstruction_mse_valset']:.4f}" if r["reconstruction_mse_valset"] is not None else "n/a"
                f.write(f"| {r['source']} | {r['real_or_synthetic']} | {r['native_hz']} | {r['coordinate_frame']} | "
                        f"{r['channels']} | {r['n_train_cached']} | {r['n_val_cached']} | {am_s} | {asd_s} | {mse_s} |\n")

    rows = []
    for name in METADATA:
        stats = cache_stats(name)
        if stats is None:
            print(f"[09] {name}: no cache found, skipping")
            continue
        recon_mse = None
        if encoder is not None:
            try:
                recon_mse = reconstruction_mse(name, encoder, decoder, cfg, device, vl_max)
            except Exception as e:
                print(f"[09] {name}: reconstruction MSE failed ({e})")
        row = {"source": name, **METADATA[name], **stats, "reconstruction_mse_valset": recon_mse}
        rows.append(row)
        acc_mean = [m for m in stats["mean_per_channel"][:3] if m is not None]
        print(f"[09] {name:14s} n_train={stats['n_train_cached']:7d} n_val={stats['n_val_cached']:6d} "
              f"acc_mean~{np.mean(acc_mean) if acc_mean else None!s:>8s} "
              f"recon_mse={recon_mse!s:>8s}")
        write_outputs(rows)  # incremental save -- survives an interrupted run

    print(f"\n[09] wrote {out_dir}/dataset_characterization.{{json,md}}")


if __name__ == "__main__":
    main()
