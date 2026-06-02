#!/usr/bin/env python3
"""Check whether RVQ codebook levels capture different frequency bands.

For each RVQ level k and each token index t:
  - find all windows that got assigned token t at level k
  - compute their mean spectral energy in 4 frequency bands
  - check if earlier levels correlate with low-frequency, later with high-frequency

Also checks dataset bias: do tokens cluster by dataset or by motion content?

Usage:
    MPLBACKEND=Agg python scripts/vis_rvq_frequency.py \
        --rvq-ckpt runs/vit1d_rvq_tokenizer_realdata_nopooling_8lvl/checkpoints/latest.pt \
        --parquet  /mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet \
        --out      data_vis/rvq_frequency.png
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.objectives.rvq_tokenizer import ResidualVectorQuantizer

WIN     = 128
FS      = 50.0
IMU_COLS = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]
DATASETS = ["pamap2","recofit","samosa","shoaib","ut_watch","utd_mhad","wisdm"]

# Frequency bands (Hz)
BANDS = [(0, 1), (1, 3), (3, 8), (8, 25)]
BAND_LABELS = ["0–1 Hz\n(gravity/slow)", "1–3 Hz\n(walk/run)", "3–8 Hz\n(fast motion)", "8–25 Hz\n(vibration/noise)"]


def load_rvq(ckpt_path):
    ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg     = ckpt["cfg"]
    obj_cfg = cfg["objective"][cfg["objective"]["type"]]
    encoder = ViT1DEncoder(cfg)
    encoder.load_state_dict(ckpt["model"], strict=True)
    encoder.eval()
    rvq = ResidualVectorQuantizer(
        num_levels=obj_cfg["num_levels"], num_codes=obj_cfg["num_codes"],
        code_dim=obj_cfg["code_dim"], ema_decay=obj_cfg["ema_decay"],
        commitment_weight=obj_cfg["commitment_weight"],
    )
    rvq.load_state_dict(ckpt["rvq"], strict=True)
    rvq.eval()
    return encoder, rvq


def spectral_band_energy(x_np: np.ndarray) -> np.ndarray:
    """x_np [B, C, T] -> [B, n_bands] mean energy across channels per band."""
    B, C, T = x_np.shape
    freqs = np.fft.rfftfreq(T, d=1.0/FS)
    fft   = np.abs(np.fft.rfft(x_np, axis=2)) ** 2   # [B, C, T//2+1]
    fft_mean = fft.mean(axis=1)                         # [B, T//2+1]  mean over channels
    band_e = np.zeros((B, len(BANDS)), dtype=np.float32)
    for i, (lo, hi) in enumerate(BANDS):
        mask = (freqs >= lo) & (freqs < hi)
        band_e[:, i] = fft_mean[:, mask].sum(axis=1) if mask.any() else 0.0
    # Normalize per window so we get fraction of energy per band
    total = band_e.sum(axis=1, keepdims=True) + 1e-8
    return band_e / total


@torch.no_grad()
def encode(x_np, encoder, rvq):
    x = torch.from_numpy(x_np).float()
    tokens, _ = encoder.forward_tokens(x)
    z = F.normalize(tokens, dim=-1)
    _, indices_list, _ = rvq(z)
    # indices_list[k]: [B, N_patches] — one index per (channel, patch) token
    # We take the most common token per window as the "window-level" code
    window_tokens = []
    for idx in indices_list:
        # Most frequent token per window
        wt = [np.bincount(idx[b].numpy(), minlength=rvq.quantizers[0].codebook.shape[0]).argmax()
              for b in range(x_np.shape[0])]
        window_tokens.append(np.array(wt))
    return window_tokens   # list of K arrays [B]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rvq-ckpt",     default="runs/vit1d_rvq_tokenizer_realdata_nopooling_8lvl/checkpoints/latest.pt")
    p.add_argument("--parquet",      default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--ocdetect-dir", default=None,
                   help="If set, load OCDetect parquets from this dir instead of master parquet")
    p.add_argument("--n-per-ds",     type=int, default=200, help="Windows per dataset/subject")
    p.add_argument("--out",          default="data_vis/rvq_frequency.png")
    return p.parse_args()


def _load_ocdetect_windows(data_dir: str, n_per_subj: int, rng):
    """Load windows from OCDetect_*.csv files. Returns (windows, group_labels)."""
    import glob
    files = sorted(glob.glob(os.path.join(data_dir, "OCDetect_*.csv")))
    if not files:
        sys.exit(f"No OCDetect_*.csv files found in {data_dir}")
    # CSV columns use spaces: "acc x", "acc y", etc.
    CSV_IMU = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]
    # Group by subject (first part of filename: OCDetect_XX_*)
    from collections import defaultdict
    subj_files = defaultdict(list)
    for f in files:
        subj_id = "_".join(os.path.basename(f).split("_")[:2])  # e.g. "OCDetect_03"
        subj_files[subj_id].append(f)
    all_wins, all_groups = [], []
    for subj, sfiles in sorted(subj_files.items()):
        frames = []
        rows_needed = n_per_subj * WIN * 3   # read just enough rows
        for fpath in sfiles[:3]:             # max 3 recordings per subject
            try:
                frames.append(pd.read_csv(fpath, usecols=CSV_IMU,
                                          nrows=rows_needed))
            except Exception:
                continue
            if sum(len(f) for f in frames) >= rows_needed:
                break
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        vals = df[CSV_IMU].values.T.astype(np.float32)   # [6, L]
        L    = vals.shape[1]
        if L < WIN:
            continue
        n = min(n_per_subj, L - WIN + 1)
        starts = rng.choice(L - WIN + 1, size=n, replace=False)
        for s in starts:
            all_wins.append(vals[:, s:s+WIN])
            all_groups.append(subj)
        print(f"  {subj}: {n} windows")
    return all_wins, all_groups


def main():
    args = parse_args()
    rng  = np.random.default_rng(42)

    print("Loading tokenizer ...")
    encoder, rvq = load_rvq(args.rvq_ckpt)
    num_levels = rvq.num_levels
    num_codes  = rvq.quantizers[0].codebook.shape[0]
    print(f"  {num_levels} levels x {num_codes} codes")

    # ── Sample windows ───────────────────────────────────────────────────────
    all_wins, all_ds = [], []

    if args.ocdetect_dir:
        print(f"Loading OCDetect windows from {args.ocdetect_dir} ...")
        all_wins, all_ds = _load_ocdetect_windows(args.ocdetect_dir, args.n_per_ds, rng)
        bias_label = "subject"
    else:
        needed = ["dataset","session_id","timestamp_ns"] + IMU_COLS
        bias_label = "dataset"
        for ds in DATASETS:
            df = pd.read_parquet(args.parquet, columns=needed,
                                 filters=[("dataset","==",ds)])
            if df.empty: continue
            df = df.sort_values(["session_id","timestamp_ns"]).reset_index(drop=True)
            vals = df[IMU_COLS].values.T.astype(np.float32)
            L = vals.shape[1]
            if L < WIN: continue
            n = min(args.n_per_ds, L - WIN + 1)
            starts = rng.choice(L - WIN + 1, size=n, replace=False)
            for s in starts:
                all_wins.append(vals[:, s:s+WIN])
                all_ds.append(ds)
            print(f"  {ds}: {n} windows")

    X  = np.stack(all_wins, axis=0)   # [N, C, T]
    DS = np.array(all_ds)
    print(f"Total: {len(X)} windows from {len(set(all_ds))} {bias_label}s")

    # ── Spectral energy per window ───────────────────────────────────────────
    print("Computing spectral energy ...")
    spec = spectral_band_energy(X)   # [N, 4]

    # ── Encode ───────────────────────────────────────────────────────────────
    print("Encoding ...")
    BS = 128
    level_tokens = [[] for _ in range(num_levels)]
    for start in range(0, len(X), BS):
        wt = encode(X[start:start+BS], encoder, rvq)
        for k in range(num_levels):
            level_tokens[k].append(wt[k])
    level_tokens = [np.concatenate(lt) for lt in level_tokens]   # [K, N]

    # ── Analysis 1: per-level spectral profile ────────────────────────────────
    # For each level k, compute mean spectral energy per token
    # Then average over all active tokens → mean spectral profile of that level
    print("\n── Spectral energy fraction per RVQ level ──")
    print(f"{'Level':>6}  {'0-1Hz':>8}  {'1-3Hz':>8}  {'3-8Hz':>8}  {'8-25Hz':>8}")
    print("-" * 50)
    level_profiles = np.zeros((num_levels, len(BANDS)))
    for k in range(num_levels):
        # Mean spectral energy of windows that hit each token, then average
        toks = level_tokens[k]
        used = np.unique(toks)
        profiles = []
        for t in used:
            mask = toks == t
            if mask.sum() > 0:
                profiles.append(spec[mask].mean(axis=0))
        if profiles:
            lp = np.stack(profiles).mean(axis=0)
        else:
            lp = spec.mean(axis=0)
        level_profiles[k] = lp
        print(f"  lvl {k+1:2d}  {lp[0]:8.3f}  {lp[1]:8.3f}  {lp[2]:8.3f}  {lp[3]:8.3f}")

    # ── Analysis 2: dataset bias ──────────────────────────────────────────────
    print(f"\n── {bias_label.capitalize()} bias in level-1 tokens ──")
    print(f"(high MI = tokens cluster by {bias_label} = BAD)")
    ds_list = sorted(set(all_ds))
    ds_to_i = {d: i for i, d in enumerate(ds_list)}
    ds_idx  = np.array([ds_to_i[d] for d in all_ds])

    # Normalized Mutual Information: NMI = MI / H(group_label)
    # NMI = 0 → tokens completely independent of group
    # NMI = 1 → tokens perfectly predict group (worst case)
    from sklearn.metrics import mutual_info_score
    from scipy.stats import entropy as scipy_entropy
    group_counts = np.bincount(ds_idx)
    group_probs  = group_counts / group_counts.sum()
    H_group      = float(scipy_entropy(group_probs))   # entropy of group label (nats)
    print(f"  H({bias_label}) = {H_group:.4f} nats  ({len(ds_list)} {bias_label}s)")
    mi_per_level = []
    nmi_per_level = []
    for k in range(num_levels):
        mi  = mutual_info_score(level_tokens[k], ds_idx)
        nmi = mi / H_group if H_group > 0 else 0.0
        mi_per_level.append(mi)
        nmi_per_level.append(nmi)
        print(f"  Level {k+1}: MI={mi:.4f} nats  NMI={nmi:.3f}  "
              f"({'HIGH BIAS' if nmi > 0.3 else 'ok' if nmi < 0.1 else 'moderate'})")
    print(f"  NMI=0 → tokens ignore {bias_label}  |  NMI=1 → tokens = {bias_label} identity")

    # ── Analysis 3: frequency content sorted by token index ──────────────────
    # For level 1: plot mean dominant frequency per token (sorted by token id)
    print("\n── Level 1: dominant frequency per token ──")
    toks_0   = level_tokens[0]
    used_tok = np.unique(toks_0)
    tok_dom_freq = {}
    for t in used_tok:
        mask = toks_0 == t
        sp   = spec[mask].mean(axis=0)
        tok_dom_freq[t] = sp   # spectral profile for this token

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"RVQ Codebook Frequency Analysis — {args.rvq_ckpt.split('/')[1]}",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # (0,0): spectral profile per level — line plot
    ax = fig.add_subplot(gs[0, 0])
    colors = plt.cm.viridis(np.linspace(0, 1, num_levels))
    for k in range(num_levels):
        ax.plot(BAND_LABELS, level_profiles[k], marker="o", color=colors[k],
                label=f"Level {k+1}", lw=1.5)
    ax.set_ylabel("Mean fraction of spectral energy", fontsize=9)
    ax.set_title("Spectral profile per RVQ level\n(same = no frequency separation)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, framealpha=0.7)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)

    # (0,1): heatmap: level × frequency band
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(level_profiles, aspect="auto", cmap="YlOrRd", origin="upper")
    ax2.set_xticks(range(len(BANDS)))
    ax2.set_xticklabels(BAND_LABELS, fontsize=7)
    ax2.set_yticks(range(num_levels))
    ax2.set_yticklabels([f"Level {k+1}" for k in range(num_levels)], fontsize=8)
    ax2.set_title("Spectral energy heatmap\n(level × frequency band)",
                  fontsize=9, fontweight="bold")
    for i in range(num_levels):
        for j in range(len(BANDS)):
            ax2.text(j, i, f"{level_profiles[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax2, fraction=0.04).ax.tick_params(labelsize=7)

    # (0,2): dataset mutual information per level
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(range(1, num_levels+1), nmi_per_level, color="#e74c3c", edgecolor="k", lw=0.5)
    ax3.axhline(0.1, color="green", lw=1, ls="--", alpha=0.7, label="ok (<0.1)")
    ax3.axhline(0.3, color="orange", lw=1, ls="--", alpha=0.7, label="high (>0.3)")
    ax3.set_ylim(0, 1.05)
    ax3.set_xlabel("RVQ Level", fontsize=9)
    ax3.set_ylabel(f"NMI(token, {bias_label})  [0=good, 1=bad]", fontsize=9)
    ax3.set_title(f"{bias_label.capitalize()} bias per level\n(NMI: 0=agnostic, 1=perfectly predicts {bias_label})",
                  fontsize=9, fontweight="bold")
    ax3.tick_params(labelsize=8)
    ax3.grid(axis="y", alpha=0.3)
    for i, v in enumerate(nmi_per_level):
        ax3.text(i+1, v+0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax3.legend(fontsize=7)

    # (1,0..2): scatter: dominant frequency vs token index for levels 1, 4, 8
    for col, k in enumerate([0, num_levels//2, num_levels-1]):
        ax4 = fig.add_subplot(gs[1, col])
        toks_k = level_tokens[k]
        used_k = np.unique(toks_k)
        # For each window, get its dominant band and the token it got
        dom_band = spec.argmax(axis=1)   # [N] 0-3
        jitter   = rng.uniform(-0.15, 0.15, size=len(used_k))
        for bi, band_name in enumerate(["0-1Hz","1-3Hz","3-8Hz","8-25Hz"]):
            mask_b  = dom_band == bi
            tok_b   = toks_k[mask_b]
            counts  = np.bincount(tok_b, minlength=num_codes)
            active  = np.where(counts > 0)[0]
            ax4.scatter(active, np.full(len(active), bi) + rng.uniform(-0.1,0.1,len(active)),
                        s=counts[active]*0.5+2, alpha=0.5, linewidths=0,
                        label=band_name)
        ax4.set_xlabel("Token index", fontsize=8)
        ax4.set_ylabel("Dominant freq band", fontsize=8)
        ax4.set_yticks([0,1,2,3])
        ax4.set_yticklabels(["0-1Hz","1-3Hz","3-8Hz","8-25Hz"], fontsize=7)
        ax4.set_title(f"Level {k+1}: token usage by dominant frequency\n(mixed = no separation)",
                      fontsize=8, fontweight="bold")
        ax4.grid(alpha=0.2)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.out}")

    # ── Key verdict ──────────────────────────────────────────────────────────
    max_band_diff = level_profiles.max(axis=0) - level_profiles.min(axis=0)
    print("\n── Verdict ──")
    print("Frequency separation across levels (max - min per band):")
    for i, bl in enumerate(BAND_LABELS):
        bar = "█" * int(max_band_diff[i] * 200)
        print(f"  {bl[:6]:>6}: {bar} {max_band_diff[i]:.4f}")
    print("  > 0.05 difference = some frequency separation exists")
    print("  < 0.01 difference = no frequency separation (levels are equivalent)")
    print(f"\n{bias_label.capitalize()} bias (level 1 NMI): {nmi_per_level[0]:.3f}")
    print(f"  NMI < 0.10 = tokens are {bias_label}-agnostic (good)")
    print(f"  NMI > 0.30 = tokens cluster by {bias_label} (bad)")


if __name__ == "__main__":
    main()
