#!/usr/bin/env python3
"""t-SNE of the PAPER's reference encoder (5-source MAE) -- v5.

v4 used the eleven-source development encoder, whose pretraining pool contains
the evaluation participants of every dataset in the plot, so the figure was
descriptive only and needed a disclaimer in two places. The reference encoder
was trained on five sources (HAR-RealData = pamap2/recofit/samosa/shoaib/
ut_watch/utd_mhad/wisdm, OCDetect, EatingSpeed, Motion-X++, SONIC), which
splits the same fourteen plotted sources into a 2x2 the figure can be read as
an actual control:

                 in the pretraining pool        held out
  real IMU       pamap2 recofit samosa          hhar mhealth prism
                 shoaib ut_watch utd_mhad       nymeria (real wrist)
                 wisdm
  mocap-synth    motionx sonic                  humanml3d

v4's islands were SONIC and HumanML3D, i.e. both mocap-synthesised sources.
Under the reference encoder SONIC is IN the pool and HumanML3D is not, so if
both stay isolated while hhar/mhealth/prism/nymeria stay inside the shared
region, the separation is a property of synthesised signals and not of
pretraining exposure. That is the claim this version is built to test.

Other changes over v4, all for print legibility:
  - activity panel colours the seven semantic FAMILIES, not the ~45 individual
    activities; v4's hue=family/shade=member legend could not be resolved at
    column width.
  - no titles inside the figures (the LaTeX caption carries them).
  - exposure is encoded by marker (filled circle = in pool, open triangle =
    held out) on top of the dataset hue, and a third panel shows the 2x2
    directly.

Windows, preprocessing, label normalisation and per-source caps are v4's,
imported rather than copied, so the only deliberate difference from v4 is the
encoder.

Usage
-----
    python scripts/plot_ssl_embeddings_v5.py
    python scripts/plot_ssl_embeddings_v5.py --ckpt <path> --out <dir>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot_ssl_embeddings_v4 import (  # noqa: E402
    DS_COLOR_MAP,
    FAMILY_COLORS,
    FAMILY_ORDER,
    SEED,
    extract_windows,
)

REF_CKPT = ("runs/vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_"
            "motionx_sonic/checkpoints/best.pt")

# --------------------------------------------------------------- the 2x2 ----
# `enabled` sources of configs/vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_
# eatingspeed_motionx_sonic_flu.yaml. OCDetect and EatingSpeed are in the pool
# but carry no activity labels and are not plotted.
IN_POOL = {"pamap2", "recofit", "samosa", "shoaib", "ut_watch", "utd_mhad",
           "wisdm", "motionx", "sonic"}
# nymeria is `processed_data_body_Ty_rwrist` -- recorded wrist IMU, not synthesis
SYNTHETIC = {"motionx", "sonic", "humanml3d"}

CELL_ORDER = ["real, in pool", "real, held out",
              "mocap-synth, in pool", "mocap-synth, held out"]
CELL_COLOR = {"real, in pool": "#2a78d6", "real, held out": "#eb6834",
              "mocap-synth, in pool": "#1baf7a", "mocap-synth, held out": "#eda100"}

FAMILY_MAIN = {"locomotion": "#2a78d6", "cycling": "#eb6834",
               "household": "#1baf7a", "posture": "#eda100",
               "hand/desk": "#e87ba4", "sport": "#008300", "stairs": "#4a3aa7"}

MARK_IN, MARK_OUT = "o", "^"


def cell_of(ds: str) -> str:
    return (f"{'mocap-synth' if ds in SYNTHETIC else 'real'}, "
            f"{'in pool' if ds in IN_POOL else 'held out'}")


def embed(X, ckpt_path):
    import torch
    from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc = ViT1DEncoderAltAttn(ckpt["cfg"])
    missing, _ = enc.load_state_dict(ckpt["model"], strict=False)
    assert not missing, missing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc.eval().to(device)
    print(f"[embed] {ckpt_path} on {device}", flush=True)
    Z = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            Z.append(enc(torch.from_numpy(X[i:i + 256]).to(device)).cpu().numpy())
    return np.concatenate(Z, 0)


# ------------------------------------------------------------- statistics ----
def analysis(Z, ds_lab, act_lab, out, ckpt_path):
    """v4's metrics plus the ones the 2x2 makes available."""
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    Zn = normalize(Z)
    N = len(Zn)
    S = Zn @ Zn.T
    np.fill_diagonal(S, -np.inf)
    K = 10
    nn_idx = np.argpartition(-S, K, axis=1)[:, :K]

    same_ds_row = (ds_lab[nn_idx] == ds_lab[:, None]).mean(axis=1)
    same_ds = float(same_ds_row.mean())
    lab_m = act_lab != "(no label)"
    same_act_row = (act_lab[nn_idx] == act_lab[:, None]).mean(axis=1)
    same_act = float(same_act_row[lab_m].mean())

    _, ds_cnt = np.unique(ds_lab, return_counts=True)
    ds_chance = float(((ds_cnt / N) ** 2).sum())
    _, ac_cnt = np.unique(act_lab[lab_m], return_counts=True)
    act_chance = float(((ac_cnt / lab_m.sum()) ** 2).sum())

    sil_ds = float(silhouette_score(Zn, ds_lab, metric="cosine"))
    keep = np.isin(act_lab, [a for a, c in zip(*np.unique(act_lab, return_counts=True))
                             if c >= 20])
    sil_act = float(silhouette_score(Zn[keep], act_lab[keep], metric="cosine"))

    # per-dataset isolation: how much of a window's neighbourhood is its own set
    per_ds = []
    for ds in sorted(set(ds_lab)):
        m = ds_lab == ds
        per_ds.append((ds, int(m.sum()), float(same_ds_row[m].mean()),
                       float(m.mean()), cell_of(ds)))
    per_ds.sort(key=lambda r: -r[2])

    # the control: isolation by exposure and by signal origin
    cells = {}
    for c in CELL_ORDER:
        m = np.array([cell_of(d) == c for d in ds_lab])
        if m.any():
            cells[c] = (int(m.sum()), float(same_ds_row[m].mean()))

    # does activity structure survive on held-out datasets?
    held = np.array([d not in IN_POOL for d in ds_lab]) & lab_m
    inp = np.array([d in IN_POOL for d in ds_lab]) & lab_m
    act_held = float(same_act_row[held].mean()) if held.any() else float("nan")
    act_in = float(same_act_row[inp].mean()) if inp.any() else float("nan")

    # cross-dataset activity retrieval (neighbours from OTHER datasets only)
    S_cross = S.copy()
    for ds in np.unique(ds_lab):
        m = ds_lab == ds
        S_cross[np.ix_(m, m)] = -np.inf
    nn_x = np.argpartition(-S_cross, K, axis=1)[:, :K]
    same_act_cross = (act_lab[nn_x] == act_lab[:, None]).mean(axis=1)

    lines, shared = [], []
    for a in np.unique(act_lab[lab_m]):
        m = act_lab == a
        n_ds = len(np.unique(ds_lab[m]))
        if n_ds >= 2 and m.sum() >= 20:
            shared.append(a)
            lines.append((a, int(m.sum()), n_ds, float(same_act_cross[m].mean()),
                          float(m.mean())))
    lines.sort(key=lambda r: -r[3])

    # leave-one-dataset-out kNN on the shared activities
    lodo = []
    sh_mask = np.isin(act_lab, shared)
    for ds in np.unique(ds_lab):
        te = sh_mask & (ds_lab == ds)
        tr = sh_mask & (ds_lab != ds)
        if te.sum() < 10 or tr.sum() < 50:
            continue
        te = te & np.isin(act_lab, list(set(act_lab[tr])))
        if te.sum() < 10:
            continue
        sims = Zn[te] @ Zn[tr].T
        top = np.argpartition(-sims, K, axis=1)[:, :K]
        lab_tr = act_lab[tr]
        preds = []
        for row in top:
            v, c = np.unique(lab_tr[row], return_counts=True)
            preds.append(v[np.argmax(c)])
        acc = float((np.array(preds) == act_lab[te]).mean())
        maj = float(np.max(np.unique(act_lab[te], return_counts=True)[1]) / te.sum())
        lodo.append((ds, int(te.sum()), acc, maj, ds in IN_POOL))

    p = f"{out}/analysis_report_v5.md"
    with open(p, "w") as f:
        f.write("# SSL embedding space, reference 5-source encoder\n\n")
        f.write(f"Encoder: `{ckpt_path}`\n\n")
        f.write(f"N={N} windows, k={K} cosine neighbours, {Z.shape[1]}-d frozen "
                f"embeddings. Windows, preprocessing and caps are v4's.\n\n")
        f.write("## 1. kNN neighbourhood purity\n\n")
        f.write(f"- same-DATASET fraction: **{same_ds:.3f}** (chance {ds_chance:.3f}, "
                f"{same_ds/ds_chance:.1f}x)\n")
        f.write(f"- same-ACTIVITY fraction: **{same_act:.3f}** (chance {act_chance:.3f}, "
                f"{same_act/act_chance:.1f}x)\n\n")
        f.write("## 2. Silhouette (cosine)\n\n")
        f.write(f"- by dataset: **{sil_ds:.3f}**\n- by activity (n>=20): **{sil_act:.3f}**\n\n")
        f.write("## 3. Isolation by exposure and signal origin (the control)\n\n")
        f.write("Mean same-dataset fraction among the 10 nearest neighbours, by cell.\n"
                "If isolation tracked pretraining exposure, the two 'held out' cells\n"
                "would be high and the two 'in pool' cells low.\n\n")
        f.write("| cell | n | same-dataset@10 |\n|---|---|---|\n")
        for c in CELL_ORDER:
            if c in cells:
                n, v = cells[c]
                f.write(f"| {c} | {n} | {v:.3f} |\n")
        f.write(f"\nActivity purity among labelled windows: in-pool datasets "
                f"{act_in:.3f}, held-out datasets {act_held:.3f}.\n\n")
        f.write("## 4. Per-dataset isolation\n\n")
        f.write("| dataset | n | same-dataset@10 | chance | cell |\n|---|---|---|---|---|\n")
        for ds, n, v, ch, c in per_ds:
            f.write(f"| {ds} | {n} | {v:.3f} | {ch:.3f} | {c} |\n")
        f.write("\n## 5. Cross-dataset activity retrieval\n\n")
        f.write("Same-activity fraction among 10 NN restricted to OTHER datasets\n"
                "(activities present in >=2 datasets, n>=20):\n\n")
        f.write("| activity | n | #datasets | cross-ds same-act@10 | chance |\n|---|---|---|---|---|\n")
        for a, n, nd, sc, ch in lines:
            f.write(f"| {a} | {n} | {nd} | {sc:.3f} | {ch:.3f} |\n")
        f.write("\n## 6. Leave-one-dataset-out kNN activity classification\n\n")
        f.write("| held-out dataset | in pretraining pool | n test | kNN acc | majority baseline |\n")
        f.write("|---|---|---|---|---|\n")
        for ds, n, acc, maj, isin in lodo:
            f.write(f"| {ds} | {'yes' if isin else 'no'} | {n} | {acc:.3f} | {maj:.3f} |\n")
    print("wrote", p, flush=True)
    return dict(same_ds=same_ds, ds_chance=ds_chance, same_act=same_act,
                act_chance=act_chance, sil_ds=sil_ds, sil_act=sil_act, cells=cells,
                act_in=act_in, act_held=act_held)


# ---------------------------------------------------------------- figures ----
def _frame(ax):
    ax.set_facecolor("#ffffff")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#d7d6cc"); sp.set_linewidth(0.8)


def panel_dataset(ax, xy, ds_lab, fontsize=7.0, s=7):
    """Hue = dataset, marker = pretraining exposure."""
    for ds in sorted(set(ds_lab)):
        m = ds_lab == ds
        isin = ds in IN_POOL
        ax.scatter(xy[m, 0], xy[m, 1], s=s if isin else s * 1.5,
                   c=DS_COLOR_MAP[ds], marker=MARK_IN if isin else MARK_OUT,
                   alpha=0.8, linewidths=0,
                   label=f"{ds} ({int(m.sum())})" if isin
                         else f"{ds} ({int(m.sum())}) — held out")
    _frame(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=fontsize,
              framealpha=0.95, markerscale=1.8, handletextpad=0.3, borderpad=0.4)


def panel_cells(ax, xy, ds_lab, fontsize=8.5, s=7):
    """The control panel: exposure x signal origin."""
    cl = np.array([cell_of(d) for d in ds_lab])
    for c in CELL_ORDER:
        m = cl == c
        if not m.any():
            continue
        isin = "in pool" in c
        ax.scatter(xy[m, 0], xy[m, 1], s=s if isin else s * 1.5, c=CELL_COLOR[c],
                   marker=MARK_IN if isin else MARK_OUT, alpha=0.75, linewidths=0,
                   label=f"{c} ({int(m.sum())})")
    _frame(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=fontsize,
              framealpha=0.95, markerscale=1.8, handletextpad=0.3, borderpad=0.4)


def panel_family(ax, xy, act_lab, fontsize=8.5, s=7):
    """Seven semantic families instead of ~45 individual activities."""
    fam = np.array([FAMILY_COLORS[a][1] if a in FAMILY_COLORS else "other"
                    for a in act_lab])
    m_lab = fam != "other"
    for f_ in FAMILY_ORDER:
        m = fam == f_
        if not m.any():
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=s, c=FAMILY_MAIN[f_], alpha=0.8,
                   linewidths=0, label=f"{f_} ({int(m.sum())})")
    _frame(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=fontsize,
              framealpha=0.95, markerscale=1.8, handletextpad=0.3, borderpad=0.4)
    return m_lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=REF_CKPT)
    ap.add_argument("--out", default="analysis/embedding_viz")
    ap.add_argument("--cache", default=None,
                    help="reuse an existing _v5.npz instead of re-embedding")
    ap.add_argument("--no-analysis", action="store_true",
                    help="skip the O(N^2) statistics (redraw only)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    os.makedirs(args.out, exist_ok=True)
    npz = args.cache or f"{args.out}/ssl_embeddings_v5.npz"

    if args.cache and os.path.exists(npz):
        d = np.load(npz, allow_pickle=True)
        Z, ts, ds_lab, act_lab = d["Z"], d["tsne"], d["dataset"], d["activity"]
        print(f"[cache] {npz}: {Z.shape}", flush=True)
    else:
        X, ds_lab, act_lab = extract_windows()
        print(f"total windows: {len(X)}", flush=True)
        Z = embed(X, args.ckpt)
        print("t-SNE ...", flush=True)
        ts = TSNE(n_components=2, perplexity=30, metric="cosine", init="pca",
                  random_state=SEED).fit_transform(Z)
        np.savez_compressed(npz, Z=Z, tsne=ts, dataset=ds_lab, activity=act_lab)
        print("wrote", npz, flush=True)

    stats = (None if args.no_analysis
             else analysis(Z, ds_lab, act_lab, args.out, args.ckpt))

    fam = np.array([FAMILY_COLORS[a][1] if a in FAMILY_COLORS else "other"
                    for a in act_lab])
    labeled = fam != "other"

    for fname, fn, arg, fs in [
            ("ssl_tsne_by_dataset_v5.png", panel_dataset, (ts, ds_lab), 6.4),
            ("ssl_tsne_by_exposure_v5.png", panel_cells, (ts, ds_lab), 7.6),
            ("ssl_tsne_by_activity_v5.png", panel_family,
             (ts[labeled], act_lab[labeled]), 7.6)]:
        fig, ax = plt.subplots(figsize=(5.5, 3.1), dpi=400)
        fn(ax, *arg, fontsize=fs, s=2.4)
        fig.tight_layout()
        fig.savefig(f"{args.out}/{fname}", facecolor="#ffffff", bbox_inches="tight")
        plt.close(fig)
        print("wrote", f"{args.out}/{fname}", flush=True)

    # one overview sheet for the repo (titles allowed here, unlike the paper figs)
    fig, axes = plt.subplots(1, 3, figsize=(26, 6.4), dpi=130)
    panel_cells(axes[0], ts, ds_lab, fontsize=9, s=7)
    axes[0].set_title("signal origin x pretraining exposure", fontsize=12)
    panel_dataset(axes[1], ts, ds_lab, fontsize=7, s=7)
    axes[1].set_title("dataset", fontsize=12)
    panel_family(axes[2], ts[labeled], act_lab[labeled], fontsize=9, s=7)
    axes[2].set_title("activity family (labelled windows)", fontsize=12)
    fig.suptitle(f"Reference 5-source MAE encoder — t-SNE — "
                 f"{len(set(ds_lab))} sources, {len(ds_lab)} windows "
                 f"({int(labeled.sum())} activity-labelled) — "
                 + (f"same-dataset@10 {stats['same_ds']:.3f} "
                    f"(chance {stats['ds_chance']:.3f})" if stats else ""),
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{args.out}/ssl_embedding_projections_v5.png",
                facecolor="#ffffff", bbox_inches="tight")
    print("wrote", f"{args.out}/ssl_embedding_projections_v5.png", flush=True)


if __name__ == "__main__":
    main()
