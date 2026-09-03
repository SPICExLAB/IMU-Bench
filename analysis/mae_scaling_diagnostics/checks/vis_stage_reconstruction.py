#!/usr/bin/env python
"""Visualize MAE reconstructions for the Stage-1 / Stage-2 checkpoints.

Plots the model's reconstruction against the ground-truth window, shading the
regions the model could NOT see (masked patches) -- those shaded spans are the
only part that is actually a prediction; unshaded regions were visible input.

The reconstruction is captured from the REAL
masked_1d_freq_channel_mask.forward_loss via a decoder forward hook, never
recomputed by hand. That rule exists because an earlier hand-rolled
reimplementation in this repo (05_val_loss_breakdown.py) was proven to produce
a measurably DIFFERENT reconstruction than the production path despite looking
line-for-line equivalent -- so anything plotted here is what the model really
outputs, not a lookalike.

Usage:
    python analysis/mae_scaling_diagnostics/checks/vis_stage_reconstruction.py \
        --stage stage1 --n 4 --out-dir data_vis/stage_recon
"""
from __future__ import annotations

import argparse, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.common import REPO_ROOT, load_merged_cfg  # noqa: E402

# Validated categorical slots 1-2 (see dataviz references/palette.md);
# checked with validate_palette.js: CVD dE 24.7, normal dE 33.6, all PASS.
C_TRUE, C_PRED = "#2a78d6", "#eb6834"
INK, INK_MUTED, SURFACE, MASK_BAND = "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e3"

STAGES = {
    "stage1": ("configs/vit1d_ssl_mae_stage1_acconly_flu.yaml",
               "runs/vit1d_ssl_mae_stage1_acconly", "Stage 1 - acc-only (187,223h)"),
    "stage2": ("configs/vit1d_ssl_mae_stage2_accgyro_flu.yaml",
               "runs/vit1d_ssl_mae_stage2_accgyro", "Stage 2 - acc+gyro (warm-started)"),
}
CH = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]


def capture(x, cvm, encoder, decoder, cfg):
    """Run the production forward_loss; return (reconstruction, signal_mask, loss)."""
    import imu_lm.objectives.masked_1d_freq_channel_mask as prod
    from imu_lm.objectives.masked_1d import _patch_mask_to_signal_mask

    grab = {}
    h = decoder.register_forward_hook(lambda m, i, o: grab.__setitem__("rec", o.detach()))
    real = prod._patch_mask_to_signal_mask
    prod._patch_mask_to_signal_mask = lambda pm, *a, **k: (grab.__setitem__("pm", pm.detach().clone()), real(pm, *a, **k))[1]
    try:
        loss, _ = prod.forward_loss((x, None, cvm), encoder, decoder, cfg)
    finally:
        h.remove(); prod._patch_mask_to_signal_mask = real

    rec = grab["rec"]
    C, T = x.shape[1], rec.shape[-1]
    sig = _patch_mask_to_signal_mask(grab["pm"], T // encoder.patch_size, C, encoder.patch_size)
    return rec, sig, float(loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES), required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--n", type=int, default=4, help="example windows to plot")
    ap.add_argument("--out-dir", default="data_vis/stage_recon")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from imu_lm.data.combined_mae_multidata_loader import make_combined_mae_multidata_loaders
    from imu_lm.models.ViT1D.model import ViT1DDecoder, ViT1DEncoderAltAttn

    cfg_path, run_dir, title = STAGES[a.stage]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_merged_cfg(cfg_path)
    # plot clean signal, not an augmented view
    for k in ("random_rotation", "gaussian_noise", "gain_jitter", "axis_bias", "time_warp"):
        if isinstance(cfg.get(k), dict): cfg[k]["enabled"] = False
    cfg["data"]["num_workers"] = 0

    enc = ViT1DEncoderAltAttn(cfg).to(dev)
    dec = ViT1DDecoder(embed_dim=enc.embed_dim, out_channels=enc.in_channels,
                       target_T=0, cfg=cfg).to(dev)
    st = torch.load(os.path.join(REPO_ROOT, run_dir, "checkpoints", f"{a.ckpt}.pt"),
                    map_location=dev, weights_only=False)
    enc.load_state_dict(st["model"]); dec.load_state_dict(st["decoder"])
    enc.eval(); dec.eval()
    step = st.get("step")
    print(f"[vis] {a.stage}: {a.ckpt}.pt step={step:,} best_val_loss={st.get('best_val_loss'):.6f}", flush=True)

    vds = make_combined_mae_multidata_loaders(cfg)["val_loader"].dataset
    g = np.random.default_rng(a.seed)
    idx = sorted(g.choice(len(vds), a.n, replace=False).tolist())
    xs_, ms_ = zip(*[vds[i] for i in idx])
    x = torch.stack([t if torch.is_tensor(t) else torch.as_tensor(t) for t in xs_]).to(dev)
    cvm = torch.stack([t if torch.is_tensor(t) else torch.as_tensor(t) for t in ms_]).to(dev)
    print(f"[vis] sampled val windows {idx} of {len(vds):,}", flush=True)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    with torch.no_grad():
        rec, sig, loss = capture(x, cvm, enc, dec, cfg)
    print(f"[vis] batch loss={loss:.4f}  reconstruction {tuple(rec.shape)}", flush=True)
    T = rec.shape[-1]
    _m = sig[:, :, :T].bool() & cvm.unsqueeze(-1).expand(-1, -1, T)
    _t, _p = x[:, :, :T][_m], rec[_m]
    print(f"[vis] MASKED-region stats: truth std={_t.std():.3f}  recon std={_p.std():.3f}  "
          f"ratio={_p.std()/_t.std():.3f}  corr={torch.corrcoef(torch.stack([_t,_p]))[0,1]:.3f}",
          flush=True)
    print(f"[vis] predict-zero baseline MSE on these masked samples = {(_t**2).mean():.4f}", flush=True)

    xs, rs, ms = x[:, :, :T].cpu().numpy(), rec.cpu().numpy(), sig[:, :, :T].cpu().numpy()
    valid = cvm[0].cpu().numpy()
    rows = [c for c in range(xs.shape[1]) if valid[c]]        # skip zero-padded channels

    os.makedirs(a.out_dir, exist_ok=True)
    fig, axes = plt.subplots(len(rows), a.n, figsize=(4.1 * a.n, 1.7 * len(rows)),
                             sharex=True, squeeze=False, facecolor=SURFACE)
    t = np.arange(T) / 50.0
    for j in range(a.n):
        for i, c in enumerate(rows):
            ax = axes[i][j]; ax.set_facecolor(SURFACE)
            # shade masked spans = what the model had to infer
            m = ms[j, c].astype(bool); d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
            for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                ax.axvspan(t[s], t[min(e, T - 1)], color=MASK_BAND, lw=0, zorder=0)
            ax.plot(t, xs[j, c], color=C_TRUE, lw=2.0, zorder=2,
                    label="ground truth" if (i == 0 and j == 0) else None)
            ax.plot(t, rs[j, c], color=C_PRED, lw=2.0, zorder=3, alpha=0.95,
                    label="reconstruction" if (i == 0 and j == 0) else None)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color("#d6d5d1"); ax.spines[sp].set_linewidth(0.8)
            ax.tick_params(colors=INK_MUTED, labelsize=7, length=3)
            ax.grid(True, color="#eceae6", lw=0.7, zorder=1)
            if j == 0: ax.set_ylabel(CH[c], color=INK, fontsize=9)
            if i == 0: ax.set_title(f"window {j+1}", color=INK_MUTED, fontsize=8.5)
            if i == len(rows) - 1: ax.set_xlabel("seconds", color=INK_MUTED, fontsize=8)

    mr = float(ms[:, rows, :].mean())   # valid channels only (see fix note)
    fig.suptitle(f"{title}   -   step {step:,}   -   batch MSE {loss:.4f}",
                 color=INK, fontsize=12.5, y=1.0)
    fig.text(0.5, 0.935, f"shaded = masked (model saw nothing there); "
                         f"{100*mr:.0f}% of valid samples masked",
             ha="center", color=INK_MUTED, fontsize=9)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.012), labelcolor=INK)
    fig.tight_layout(rect=[0, 0.03, 1, 0.905])
    out = os.path.join(a.out_dir, f"{a.stage}_recon_{a.ckpt}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE); plt.close(fig)
    print(f"[vis] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
