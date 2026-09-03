#!/usr/bin/env python3
"""Visualize odometry trajectory reconstruction: predicted vs. ground truth.

Trains a frozen-encoder odometry probe (mirrors eval_odom_nymeria_altattn.py's
data/model/reconstruction pipeline exactly, reusing its helper functions) for
one pretrained run, then plots a handful of held-out validation-window
trajectories (top-down XZ floor-plane view, predicted vs ground truth), plus
summary histograms of per-window ATE and drift across the full val set.

Usage
-----
    python scripts/vis_odom_trajectories.py \\
        --run vit1d_ssl_mae_varlen_altattn_multidata \\
        --train-dir /projects/b53017/vbc1591/processed_data_body_Ty_rwrist \\
        --val-dir   /projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist \\
        --out-dir vis --label "Ours (MAE)"
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the exact data/model/reconstruction code from the benchmark script so
# the visualization matches what's reported in EXPERIMENTS.md. Only the
# training loop is duplicated locally (to keep the trained head around for
# plotting) -- eval_odom_nymeria_altattn.py itself is left untouched since
# other benchmark jobs are reading it concurrently.
import eval_odom_nymeria_altattn as odom


def reconstruct_traj(v_flu_pred, gyro_raw, R0, ps, target_hz=50):
    """Same math as odom.reconstruct_and_score, but returns the full [Ns+1, 3] trajectory."""
    Pt, Q = v_flu_pred.shape[0], ps // 4
    Ns, dt = Pt * 4, 1.0 / target_hz
    R = R0.copy().astype(np.float64)
    Rs = [R.copy()]
    for t in range(gyro_raw.shape[0]):
        R = R @ odom.exp_so3(gyro_raw[t].astype(np.float64) * dt)
        Rs.append(R.copy())
    pos = np.zeros(3)
    traj = [pos.copy()]
    for i, vf in enumerate(v_flu_pred.reshape(Ns, 3).astype(np.float64)):
        pos = pos + (Rs[i * Q + Q // 2] @ vf) * (Q * dt)
        traj.append(pos.copy())
    return np.asarray(traj)


def train_probe_keep_head(enc, tr, va, device, ps, epochs, patience, lr, batch_size):
    """Copy of odom.train_probe that also returns the trained head."""
    Pt = tr["X"].shape[-1] // ps
    head = odom.RegisterOdomHead(enc.embed_dim, Pt).to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler()
    dl = DataLoader(TensorDataset(torch.from_numpy(tr["X"]), torch.from_numpy(tr["Y"])),
                    batch_size=batch_size, shuffle=True, drop_last=True)
    best, best_state, no_imp = float("inf"), None, 0
    for ep in range(epochs):
        head.train(); enc.eval()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with autocast():
                with torch.no_grad():
                    feat = odom.features(enc, xb)
                loss = F.mse_loss(head(feat), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        m = odom.eval_metrics(enc, head, va, device, ps, max_windows=500)
        if m["vel_mse"] < best - 1e-6:
            best, best_state, no_imp = m["vel_mse"], {k: v.cpu().clone() for k, v in head.state_dict().items()}, 0
        else:
            no_imp += 1
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"    epoch {ep+1:3d} val_vel_mse={m['vel_mse']:.5f} ATE={m['ate']:.4f} best={best:.5f}")
        if patience > 0 and no_imp >= patience:
            print(f"    early stop @ epoch {ep+1}")
            break
    if best_state:
        head.load_state_dict(best_state)
    final_metrics = odom.eval_metrics(enc, head, va, device, ps)
    return head, final_metrics


@torch.no_grad()
def collect_all_window_scores(enc, head, va, device, ps, batch_size=256, max_windows=3000, seed=0):
    """Per-window (not averaged) ATE/drift/path-length, for the histogram plots.

    ``reconstruct_and_score`` has an inner pure-Python per-timestep loop, so
    running it over every validation window (can be 10,000s) is prohibitively
    slow -- cap to a random sample of ``max_windows``, same spirit as the
    benchmark script's own ``max_windows=2000`` default for its final eval.
    """
    enc.eval(); head.eval()
    n_total = len(va["X"])
    if n_total > max_windows:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(n_total, size=max_windows, replace=False))
    else:
        idx = np.arange(n_total)

    X = torch.from_numpy(va["X"][idx])
    preds = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        preds.append(head(odom.features(enc, xb)).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    ates, ferrs, plens = [], [], []
    for i in range(len(preds)):
        ate, ferr, plen = odom.reconstruct_and_score(
            preds[i], va["GY"][idx[i]], va["R0"][idx[i]], va["POS"][idx[i]], ps
        )
        ates.append(ate); ferrs.append(ferr); plens.append(plen)
    return preds, np.asarray(ates), np.asarray(ferrs), np.asarray(plens), idx


def plot_trajectories(preds, va, ps, num_examples, out_path, label, seed=0):
    """Top-down (X-Z floor plane) predicted vs. ground-truth trajectories for a
    handful of validation windows, chosen from the more-interesting (longer
    path-length) half so the plots aren't dominated by near-static windows."""
    rng = np.random.RandomState(seed)
    n = len(preds)
    plens = np.asarray([
        np.sum(np.linalg.norm(np.diff(va["POS"][i], axis=0), axis=1)) for i in range(n)
    ])
    moving_idx = np.where(plens > np.median(plens))[0]
    pick_pool = moving_idx if len(moving_idx) >= num_examples else np.arange(n)
    idxs = rng.choice(pick_pool, size=min(num_examples, len(pick_pool)), replace=False)

    ncols = 3
    nrows = int(np.ceil(len(idxs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for ax_i, idx in enumerate(idxs):
        ax = axes[ax_i // ncols][ax_i % ncols]
        traj = reconstruct_traj(preds[idx], va["GY"][idx], va["R0"][idx], ps)
        gt = va["POS"][idx]
        ax.plot(gt[:, 0], gt[:, 2], "-", color="tab:blue", linewidth=2, label="ground truth")
        ax.plot(traj[:, 0], traj[:, 2], "--", color="tab:orange", linewidth=2, label="predicted")
        ax.scatter([gt[0, 0]], [gt[0, 2]], color="green", marker="o", s=60, zorder=5, label="start")
        ax.scatter([gt[-1, 0]], [gt[-1, 2]], color="blue", marker="x", s=60, zorder=5)
        ax.scatter([traj[-1, 0]], [traj[-1, 2]], color="orange", marker="x", s=60, zorder=5)
        ate = float(np.sqrt(np.mean(np.sum((traj - gt) ** 2, axis=1))))
        ax.set_title(f"window {idx}  ATE={ate:.3f} m", fontsize=10)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=8, loc="best")

    for ax_i in range(len(idxs), nrows * ncols):
        axes[ax_i // ncols][ax_i % ncols].axis("off")

    fig.suptitle(f"{label}: odometry trajectory reconstruction (top-down X-Z), "
                 f"{num_examples} random above-median-motion val windows", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_summary_hist(ates, drifts, out_path, label):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(ates, bins=30, color="tab:blue", edgecolor="white")
    axes[0].axvline(float(np.mean(ates)), color="red", linestyle="--",
                     label=f"mean={np.mean(ates):.3f} m")
    axes[0].set_xlabel("ATE (m)"); axes[0].set_ylabel("# windows")
    axes[0].set_title("Per-window absolute trajectory error"); axes[0].legend()

    finite_drift = drifts[np.isfinite(drifts)]
    axes[1].hist(finite_drift, bins=30, color="tab:orange", edgecolor="white")
    axes[1].axvline(float(np.mean(finite_drift)), color="red", linestyle="--",
                     label=f"mean={np.mean(finite_drift):.1f}%")
    axes[1].set_xlabel("drift (final_err / path_len, %)"); axes[1].set_ylabel("# windows")
    axes[1].set_title("Per-window drift"); axes[1].legend()

    fig.suptitle(f"{label}: odometry error distribution over the full val set (n={len(ates)})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--train-dir", default="/projects/b53017/vbc1591/processed_data_body_Ty_rwrist")
    p.add_argument("--val-dir", default="/projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist")
    p.add_argument("--max-train-files", type=int, default=867)
    p.add_argument("--max-val-files", type=int, default=101)
    p.add_argument("--window-seconds", type=float, default=2.56)
    p.add_argument("--add-gravity", action="store_true", default=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-examples", type=int, default=6)
    p.add_argument("--max-hist-windows", type=int, default=3000,
                    help="Cap on validation windows scored for the histogram/trajectory plots "
                         "(reconstruct_and_score has an unvectorized per-timestep Python loop, "
                         "so scoring all tens-of-thousands of val windows is prohibitively slow)")
    p.add_argument("--out-dir", default="vis")
    p.add_argument("--label", default=None, help="Title/filename label (defaults to --run)")
    p.add_argument("--head-checkpoint", default=None,
                    help="Path to an odom_head.pt saved by eval_odom_nymeria_altattn.py's "
                         "--out-dir (e.g. data_vis/nymeria_odom_altattn_probe_multidata/odom_head.pt). "
                         "If given, skips training entirely (pure inference) -- val windows still "
                         "need to be rebuilt from the pkl sessions, but the ~50-epoch probe training "
                         "is skipped, so this is much faster.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    label = args.label or args.run
    slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    enc, cfg = odom.load_encoder(os.path.join(args.runs_root, args.run), args.step, device)
    cfg = {**cfg, "windowing": {**cfg.get("windowing", {}), "window_seconds": args.window_seconds}}
    T = int(round(args.window_seconds * odom.FS))
    ps = int((cfg.get("vit1d", {}) or {}).get("patch_size", 16))
    hop = T // 2

    va_files = sorted(glob.glob(os.path.join(args.val_dir, "*.pkl")))[:args.max_val_files]

    def load(files):
        out = []
        for f in files:
            r = odom.load_session_local_vel_flu(f, target_hz=int(odom.FS), add_gravity=args.add_gravity)
            if r is not None:
                out.append(r)
        return out

    va_sess = load(va_files)
    keys = ["X", "Y", "GY", "R0", "POS"]
    va = dict(zip(keys, odom.build_windows(va_sess, cfg, T, ps, hop, int(odom.FS))))

    if args.head_checkpoint:
        # Pure inference: load the already-trained head, skip the ~50-epoch
        # training loop entirely (and skip loading the training sessions,
        # which for the full 867-file set is the single largest time cost).
        print(f"sessions val={len(va_sess)}  windows val={len(va['X'])}")
        ckpt = torch.load(args.head_checkpoint, map_location=device, weights_only=False)
        assert ckpt["embed_dim"] == enc.embed_dim and ckpt["patch_size"] == ps, (
            f"head checkpoint architecture mismatch: saved embed_dim={ckpt['embed_dim']} "
            f"patch_size={ckpt['patch_size']} vs current encoder embed_dim={enc.embed_dim} patch_size={ps}"
        )
        head = odom.RegisterOdomHead(ckpt["embed_dim"], ckpt["num_patches"]).to(device)
        head.load_state_dict(ckpt["head_state_dict"])
        head.eval()
        m = odom.eval_metrics(enc, head, va, device, ps)
        print(f"[loaded head from {args.head_checkpoint}, previously: {ckpt.get('metrics')}]")
    else:
        tr_files = sorted(glob.glob(os.path.join(args.train_dir, "*.pkl")))[:args.max_train_files]
        tr_sess = load(tr_files)
        tr = dict(zip(keys, odom.build_windows(tr_sess, cfg, T, ps, hop, int(odom.FS))))
        print(f"sessions train={len(tr_sess)} val={len(va_sess)}")
        print(f"windows train={len(tr['X'])} val={len(va['X'])}")

        head, m = train_probe_keep_head(
            enc, tr, va, device, ps, args.epochs, args.patience, args.lr, args.batch_size
        )
        # Persist so a future rerun (e.g. re-plotting with different --num-examples)
        # can pass --head-checkpoint and skip training entirely.
        head_path = os.path.join(args.out_dir, f"odom_head_{slug}.pt")
        torch.save({
            "head_state_dict": head.state_dict(),
            "embed_dim": enc.embed_dim,
            "num_patches": tr["X"].shape[-1] // ps,
            "patch_size": ps,
            "run": args.run,
            "metrics": m,
        }, head_path)
        print(f"saved trained probe head to {head_path}")
    print(f"{args.run}: vel_mse={m['vel_mse']:.5f} ATE={m['ate']:.4f}m drift={m['drift']:.2f}%")

    preds, ates, ferrs, plens, idx = collect_all_window_scores(
        enc, head, va, device, ps, max_windows=args.max_hist_windows
    )
    moving = plens > 0.1
    drifts = np.where(moving, ferrs / np.maximum(plens, 1e-9) * 100.0, np.nan)

    # va restricted to the same sampled windows so plot_trajectories' positional
    # indexing into va lines up with preds/ates (which are already subsampled).
    va_sub = {k: v[idx] for k, v in va.items()}
    plot_trajectories(preds, va_sub, ps, args.num_examples,
                       os.path.join(args.out_dir, f"odom_traj_{slug}.png"), label)
    plot_summary_hist(ates, drifts, os.path.join(args.out_dir, f"odom_hist_{slug}.png"), label)

    np.savez(
        os.path.join(args.out_dir, f"odom_raw_{slug}.npz"),
        preds=preds, ates=ates, ferrs=ferrs, plens=plens,
        vel_mse=m["vel_mse"], ate_mean=m["ate"], drift_mean=m["drift"],
    )
    print(f"saved {os.path.join(args.out_dir, f'odom_raw_{slug}.npz')}")


if __name__ == "__main__":
    main()
