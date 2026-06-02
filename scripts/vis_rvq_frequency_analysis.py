"""Verify that the RVQ codebook captures frequency-progressive content.

Two complementary analyses:

1. PSD of per-level contributions
   contribution_k = cumulative_k − cumulative_{k-1}
   If frequency-progressive design is working, each contribution should
   concentrate energy in an increasingly high frequency band.

2. Band-specific RMSE vs number of levels
   Bandpass-filter both the original and the cumulative reconstruction
   into the 4 target bands: [0-2, 2-5, 5-12, 12-25] Hz.
   Adding more levels should primarily reduce error in progressively
   higher frequency bands.

Usage
-----
    python scripts/vis_rvq_frequency_analysis.py \
        --run-dir runs/vit1d_rvq_tokenizer_ocdetect_anyrot \
        --n-windows 16 \
        --out data_vis/rvq_freq_analysis.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, welch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.objectives.rvq_tokenizer import ResidualVectorQuantizer, PatchDecoder, _temporal_preds_to_signal, _patches_to_signal
from imu_lm.utils.helpers import cfg_get
from imu_lm.data.augmentations.preprocess import preprocess_window

FS = 50.0
CH_NAMES = ["acc_F", "acc_L", "acc_U", "gyro_F", "gyro_L", "gyro_U"]

def _level_colors(n: int) -> list:
    import matplotlib.cm as cm
    return [cm.tab10(i / max(n - 1, 1)) for i in range(n)]

def _bands_from_cutoffs(cutoffs: list) -> tuple:
    """Convert RVQ cutoff list to (band_edges, band_labels) for analysis.
    cutoffs: e.g. [2,5,12,999] → bands (0,2),(2,5),(5,12),(12,25)
    """
    nyq = FS / 2.0
    real_cutoffs = [c for c in cutoffs if c < nyq]
    edges, labels = [], []
    lo = 0.0
    for hi in real_cutoffs:
        edges.append((lo, hi))
        labels.append(f"{lo:.0f}–{hi:.0f} Hz")
        lo = hi
    edges.append((lo, nyq))
    labels.append(f"{lo:.0f}–{nyq:.0f} Hz")
    return edges, labels


# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------

def bandpass(signal: np.ndarray, lo: float, hi: float, fs: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass. signal: [T] or [C, T]."""
    nyq = fs / 2.0
    lo_cut = max(lo, 0.1)   # avoid 0 Hz (DC removal)
    if hi >= nyq:
        # High-pass only
        b, a = butter(4, lo_cut / nyq, btype="high")
    else:
        b, a = butter(4, [lo_cut / nyq, hi / nyq], btype="band")
    axis = -1
    return filtfilt(b, a, signal, axis=axis)


def lowpass(signal: np.ndarray, cutoff: float, fs: float) -> np.ndarray:
    if cutoff >= fs / 2.0:
        return signal.copy()
    b, a = butter(4, cutoff / (fs / 2.0), btype="low")
    return filtfilt(b, a, signal, axis=-1)


def compute_psd(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean PSD across channels. signal: [C, T]. Returns (freqs, psd)."""
    psds = []
    for c in range(signal.shape[0]):
        f, p = welch(signal[c], fs=fs, nperseg=min(64, signal.shape[-1]))
        psds.append(p)
    return f, np.mean(psds, axis=0)


def band_rmse(pred: np.ndarray, target: np.ndarray, lo: float, hi: float, fs: float) -> float:
    """RMSE between pred and target restricted to [lo, hi] Hz band. Shapes: [C, T]."""
    pred_b   = bandpass(pred,   lo, hi, fs)
    target_b = bandpass(target, lo, hi, fs)
    return float(np.sqrt(np.mean((pred_b - target_b) ** 2)))


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(run_dir: str, device: torch.device):
    ckpt_path = os.path.join(run_dir, "checkpoints", "latest.pt")
    if not os.path.exists(ckpt_path):
        ckpts = sorted(f for f in os.listdir(os.path.join(run_dir, "checkpoints"))
                       if f.endswith(".pt"))
        ckpt_path = os.path.join(run_dir, "checkpoints", ckpts[-1])
    print(f"Loading: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    cfg = state["cfg"]

    encoder = ViT1DEncoder(cfg).to(device)
    encoder.load_state_dict(state["model"], strict=True)
    encoder.eval()

    rvq_cfg        = cfg_get(cfg, ["objective", "rvq_tokenizer_ocdetect_anyrot"], {}) or {}
    num_levels     = int(rvq_cfg.get("num_levels",    4))
    num_codes      = int(rvq_cfg.get("num_codes",     512))
    code_dim       = int(rvq_cfg.get("code_dim",      256))
    ema_decay      = float(rvq_cfg.get("ema_decay",   0.99))
    cmt_w          = float(rvq_cfg.get("commitment_weight", 0.25))
    dec_hidden     = int(rvq_cfg.get("decoder_hidden_dim", 512))
    pool_channels  = bool(rvq_cfg.get("pool_channels", True))
    num_channels   = int(rvq_cfg.get("num_channels",   6))
    cutoff_hz      = list(rvq_cfg.get("cutoff_hz", [2.0, 5.0, 12.0, 999.0]))

    dec_out_dim = (num_channels * encoder.patch_size) if pool_channels else encoder.patch_size

    rvq = ResidualVectorQuantizer(num_levels, num_codes, code_dim, ema_decay, cmt_w).to(device)
    rvq.load_state_dict(state["rvq"], strict=True)
    rvq.eval()

    decoders = nn.ModuleList([
        PatchDecoder(code_dim, dec_out_dim, dec_hidden) for _ in range(num_levels)
    ]).to(device)
    decoders.load_state_dict(state["decoders"], strict=True)
    decoders.eval()

    proj = None
    if encoder.embed_dim != code_dim and "rvq_proj" in state:
        proj = nn.Linear(encoder.embed_dim, code_dim).to(device)
        proj.load_state_dict(state["rvq_proj"], strict=True)
        proj.eval()

    step = state.get("step", "?")
    print(f"Model step: {step}   levels: {num_levels}   pool_channels: {pool_channels}   cutoffs: {cutoff_hz}")
    return encoder, rvq, decoders, proj, cfg, pool_channels, num_channels, cutoff_hz


# ---------------------------------------------------------------------------
# Get per-level cumulative reconstructions
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_cumulative_recons(
    x: torch.Tensor,
    encoder, rvq, decoders, proj, pool_channels: bool,
) -> list[np.ndarray]:
    """Return K arrays [B, C, T], one per cumulative level."""
    B, C, T = x.shape
    tokens, Pt = encoder.forward_tokens(x)
    D = tokens.shape[-1]

    if pool_channels:
        tokens_q = tokens.view(B, C, Pt, D).mean(dim=1)
    else:
        tokens_q = tokens

    z = proj(tokens_q) if proj is not None else tokens_q
    z = F.normalize(z, dim=-1)

    residual     = z
    cumulative_q = torch.zeros_like(z)
    recons = []

    for vq, dec in zip(rvq.quantizers, decoders):
        q, _ = vq(residual)
        cumulative_q = cumulative_q + q
        residual     = residual - q.detach()

        pred = dec(cumulative_q)
        if pool_channels:
            sig = _temporal_preds_to_signal(pred, B, C, Pt, encoder.patch_size)
        else:
            sig = _patches_to_signal(pred, B, C, Pt, encoder.patch_size)
        recons.append(sig[:, :, :T].cpu().numpy())   # [B, C, T]

    return recons


# ---------------------------------------------------------------------------
# Load data windows
# ---------------------------------------------------------------------------

def get_windows(cfg, n_windows: int, device: torch.device) -> torch.Tensor:
    import glob
    from imu_lm.data.OCDetect_dataloader import (
        DATA_DIR_DEFAULT, _load_recording,
    )
    from imu_lm.data.windowing import compute_T_and_hop

    rvq_cfg  = cfg_get(cfg, ["objective", "rvq_tokenizer_ocdetect_anyrot"], {}) or {}
    data_dir = rvq_cfg.get("data_dir", DATA_DIR_DEFAULT)
    apply_flu = bool(rvq_cfg.get("apply_flu_transform", False))
    win_T, _ = compute_T_and_hop(cfg)

    files = sorted(glob.glob(os.path.join(data_dir, "OCDetect_*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "OCDetect_*.csv")))

    raw_windows = []
    for path in files:
        if len(raw_windows) >= n_windows:
            break
        try:
            imu, _, ignore = _load_recording(path, apply_flu_transform=apply_flu)
        except Exception as e:
            continue
        usable = np.where(ignore == 0)[0]
        i = 0
        while i + win_T <= len(imu) and len(raw_windows) < n_windows:
            seg = usable[(usable >= i) & (usable < i + win_T)]
            if len(seg) == win_T:
                raw_windows.append(imu[i:i + win_T].astype(np.float32))
            i += win_T

    windows_norm = []
    for w in raw_windows[:n_windows]:
        wn = preprocess_window(w, cfg)
        if wn is not None:
            windows_norm.append(wn)

    return torch.from_numpy(np.stack(windows_norm)).to(device)


# ---------------------------------------------------------------------------
# Main analysis + plotting
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/vit1d_rvq_tokenizer_ocdetect_anyrot")
    ap.add_argument("--n-windows", type=int, default=16)
    ap.add_argument("--out", default="data_vis/rvq_freq_analysis.png")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = torch.device(args.device)

    encoder, rvq, decoders, proj, cfg, pool_channels, num_ch, cutoff_hz = load_model(args.run_dir, device)

    BAND_EDGES, BAND_LABELS = _bands_from_cutoffs(cutoff_hz)

    print(f"Loading {args.n_windows} windows …")
    x = get_windows(cfg, args.n_windows, device)
    print(f"Data shape: {tuple(x.shape)}")

    print("Running RVQ …")
    recons = get_cumulative_recons(x, encoder, rvq, decoders, proj, pool_channels)
    # recons: list of K arrays [B, C, T]

    x_np = x.cpu().numpy()   # [B, C, T]
    K = len(recons)
    B, C, T = x_np.shape
    LEVEL_COLORS = _level_colors(K)

    # Band labels: one per level (pad/trim to K)
    level_band_labels = (BAND_LABELS + [BAND_LABELS[-1]] * K)[:K]

    # ── Analysis 1: PSD of per-level contributions ────────────────────────────
    contributions = [recons[0]] + [recons[k] - recons[k - 1] for k in range(1, K)]

    psd_original = []
    psd_contribs = [[] for _ in range(K)]
    for b in range(B):
        f, p = compute_psd(x_np[b], FS)
        psd_original.append(p)
        for k in range(K):
            _, pc = compute_psd(contributions[k][b], FS)
            psd_contribs[k].append(pc)

    psd_orig_mean    = np.mean(psd_original, axis=0)
    psd_contrib_mean = [np.mean(psd_contribs[k], axis=0) for k in range(K)]

    # ── Analysis 2: band-specific RMSE vs number of levels ───────────────────
    n_bands = len(BAND_EDGES)
    band_rmse_table = np.zeros((K, n_bands))
    for k in range(K):
        for bi, (lo, hi) in enumerate(BAND_EDGES):
            rmses = [band_rmse(recons[k][b], x_np[b], lo, hi, FS) for b in range(B)]
            band_rmse_table[k, bi] = np.mean(rmses)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(max(16, K * 2), 9))
    fig.suptitle(
        f"RVQ Frequency Analysis  —  {os.path.basename(args.run_dir)}\n"
        f"({B} windows, {K} levels)",
        fontsize=12, fontweight="bold"
    )
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    # ── Panel A: PSD of each level's contribution ─────────────────────────────
    ax_psd = fig.add_subplot(gs[0, :])
    ax_psd.semilogy(f, psd_orig_mean, color="0.4", lw=1.5, ls="--", label="Original signal", zorder=10)
    for k in range(K):
        ax_psd.semilogy(f, psd_contrib_mean[k], color=LEVEL_COLORS[k], lw=1.5,
                        label=f"L{k+1} contribution  (target: {level_band_labels[k]})")

    for k, (lo, hi) in enumerate(BAND_EDGES):
        ax_psd.axvspan(lo, min(hi, FS / 2), alpha=0.07, color=LEVEL_COLORS[min(k, K - 1)])
        ax_psd.axvline(lo, color=LEVEL_COLORS[min(k, K - 1)], lw=0.8, ls=":")

    ax_psd.set_xlabel("Frequency (Hz)", fontsize=10)
    ax_psd.set_ylabel("Power spectral density (log)", fontsize=10)
    ax_psd.set_title("A  —  PSD of each level's contribution\n"
                      "Level k adds signal in its target band if energy peaks there",
                      fontsize=10)
    ax_psd.legend(fontsize=7, ncol=max(2, K // 2))
    ax_psd.set_xlim(0, FS / 2)
    ax_psd.grid(True, lw=0.3, alpha=0.4)

    # ── Panel B: Band RMSE vs levels (lines per band) ─────────────────────────
    ax_lines = fig.add_subplot(gs[1, 0])
    x_ticks = np.arange(1, K + 1)
    band_colors = _level_colors(n_bands)
    for bi, (lo, hi) in enumerate(BAND_EDGES):
        ax_lines.plot(x_ticks, band_rmse_table[:, bi],
                      color=band_colors[bi], marker="o", lw=1.5, ms=4,
                      label=f"{BAND_LABELS[bi]}")
    ax_lines.set_xlabel("Cumulative RVQ levels used", fontsize=10)
    ax_lines.set_ylabel("Band RMSE", fontsize=10)
    ax_lines.set_title("B  —  Reconstruction error per frequency band\n"
                        "Higher bands should improve most when their target level is added",
                        fontsize=10)
    ax_lines.set_xticks(x_ticks)
    ax_lines.set_xticklabels([f"L1-{k}" for k in x_ticks], fontsize=7)
    ax_lines.legend(fontsize=7)
    ax_lines.grid(True, lw=0.3, alpha=0.4)

    # ── Panel C: Band RMSE heatmap ─────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[1, 1])
    rmse_norm = band_rmse_table / (band_rmse_table[0:1, :] + 1e-8)
    im = ax_heat.imshow(rmse_norm.T, aspect="auto", cmap="RdYlGn_r",
                        vmin=0.3, vmax=1.05, origin="upper")
    ax_heat.set_xticks(range(K))
    ax_heat.set_xticklabels([f"L1-{k+1}" for k in range(K)], fontsize=7)
    ax_heat.set_yticks(range(n_bands))
    ax_heat.set_yticklabels(BAND_LABELS, fontsize=8)
    ax_heat.set_xlabel("Cumulative levels", fontsize=10)
    ax_heat.set_title("C  —  Normalised RMSE heatmap\n"
                       "Green = big improvement vs L1-only; red = little improvement",
                       fontsize=10)
    plt.colorbar(im, ax=ax_heat, shrink=0.8, label="RMSE / RMSE(L1-only)")

    for k in range(K):
        for bi in range(n_bands):
            ax_heat.text(k, bi, f"{rmse_norm[k, bi]:.2f}",
                         ha="center", va="center", fontsize=7,
                         color="white" if rmse_norm[k, bi] < 0.6 else "black")

    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {args.out}")

    # ── Console summary ───────────────────────────────────────────────────────
    col_w = 10
    print(f"\n{'':20s}", end="")
    for k in range(K):
        print(f"{'L1-'+str(k+1):>{col_w}}", end="")
    print()
    for bi, label in enumerate(BAND_LABELS):
        print(f"{label:20s}", end="")
        for k in range(K):
            print(f"{band_rmse_table[k, bi]:>{col_w}.4f}", end="")
        print()

    print(f"\nFreq centre of mass per contribution (Hz) — should increase L1 → L{K}:")
    for k in range(K):
        psd = psd_contrib_mean[k]
        com = float(np.sum(f * psd) / (np.sum(psd) + 1e-12))
        print(f"  L{k+1}: {com:.2f} Hz   (target band: {level_band_labels[k]})")


if __name__ == "__main__":
    main()
