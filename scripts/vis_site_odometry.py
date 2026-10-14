#!/usr/bin/env python
"""Qualitative odometry figure for the website: ground-truth pelvis trajectory against the trajectory
reconstructed from InrtlWrist (fine-tuned) and from the same backbone trained from scratch, on 2.56 s
Nymeria validation windows under the benchmark's wrist-frame protocol (bench/score/odometry_local.py).

    python scripts/vis_site_odometry.py --out website/static/images/odometry_windows.png [--sessions 3] [--windows 6]
"""
import argparse, glob, os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import eval_odom_nymeria_altattn as lf                       # noqa: E402
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DEncoderAltAttn, ViT1DOdomDecoder   # noqa: E402
from imu_lm.utils.helpers import cfg_get                      # noqa: E402


def load_scratch_model(run_dir, step, device):
    """Encoder + ViT1DOdomDecoder rebuilt from the checkpoint's own config (best.pt)."""
    obj = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"), map_location="cpu", weights_only=False)
    cfg = obj["cfg"]
    alt = bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False))
    enc = (ViT1DEncoderAltAttn(cfg) if alt else ViT1DEncoder(cfg)).to(device)
    enc.load_state_dict(obj["model"], strict=True); enc.eval()
    T = int(round(float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56)) * lf.FS))
    ps = int((cfg.get("vit1d", {}) or {}).get("patch_size", 16))
    dec = ViT1DOdomDecoder(embed_dim=enc.embed_dim, num_patches_per_channel=T // ps, in_channels=enc.in_channels, cfg=cfg, out_dim=12).to(device)
    dec.load_state_dict(obj["odom_decoder"], strict=True); dec.eval()
    print(f"[odom] loaded {os.path.basename(run_dir)} step={obj.get('step')} T={T} ps={ps}")
    return enc, dec, cfg, T, ps
from bench.score.odometry_local import _build_windows_by_session, _exp_so3_batch   # noqa: E402

TEAL, AMBER, GREY, INK, MUTED = "#1d6f8f", "#c2662a", "#8a96a6", "#16202e", "#6b7788"


def trajectories(v_flu, gyro, R0, ps, hz=50):
    """Predicted velocities in the wrist frame -> world trajectory, exactly as the scorer integrates them."""
    N, Pt = v_flu.shape[:2]; Q, dt = ps // 4, 1.0 / hz; Ns = Pt * 4
    mids = {i * Q + Q // 2 for i in range(Ns)}
    R = R0.astype(np.float64).copy(); R_at = {0: R.copy()} if 0 in mids else {}
    for t in range(gyro.shape[1]):
        R = R @ _exp_so3_batch(gyro[:, t].astype(np.float64) * dt)
        if t + 1 in mids: R_at[t + 1] = R.copy()
    v = v_flu.reshape(N, Ns, 3).astype(np.float64)
    steps = np.stack([np.einsum("nij,nj->ni", R_at[i * Q + Q // 2], v[:, i]) for i in range(Ns)], axis=1) * (Q * dt)
    return np.concatenate([np.zeros((N, 1, 3)), np.cumsum(steps, axis=1)], axis=1)


def predict(run, X, device):
    enc, dec, cfg, T, ps = load_scratch_model(run, None, device)
    obj = torch.load(os.path.join(run, "checkpoints", "best.pt"), map_location="cpu", weights_only=False)
    enc.load_state_dict(obj["model"], strict=True); dec.load_state_dict(obj["odom_decoder"], strict=True)
    enc.eval(); dec.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            tok, _ = enc.forward_tokens(torch.from_numpy(X[i:i + 64]).float().to(device))
            p = dec(tok); out.append(p.reshape(p.shape[0], p.shape[1], 4, 3).cpu().numpy())
    return np.concatenate(out), cfg, T, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-dir", default="/projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist")
    ap.add_argument("--run-ours", default="runs/bench_odometry_nymeria_local__mae_5src__finetune__s42")
    ap.add_argument("--run-scratch", default="runs/bench_odometry_nymeria_local__scratch__scratch__s42")
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-scratch", action="store_true", help="plot ground truth and InrtlWrist only")
    ap.add_argument("--best-pct", type=float, default=25.0, help="with --select best: keep this percentile of relative error")
    ap.add_argument("--select", choices=["random", "best"], default="random",
                    help="best: among windows with above-median motion, sample from the lowest-error quarter (ATE / path length)")
    a = ap.parse_args()
    device = torch.device("cpu")
    files = sorted(glob.glob(os.path.join(a.val_dir, "*.pkl")))
    pick = files[:: max(1, len(files) // a.sessions)][: a.sessions]
    sess = [r for f in pick if (r := lf.load_session_local_vel_flu(f, target_hz=int(lf.FS), add_gravity=True)) is not None]
    print(f"[odom] {len(sess)} sessions:", [os.path.basename(f)[:28] for f in pick])
    enc, dec, cfg, T, ps = load_scratch_model(a.run_ours, None, device)
    (X, Y, GY, R0, POS), _ = _build_windows_by_session(lf, sess, cfg, T, ps, T // 2)
    plen = np.sum(np.linalg.norm(np.diff(POS, axis=1), axis=2), axis=1)
    cand = np.where(plen > np.percentile(plen, 75))[0]
    rng = np.random.RandomState(a.seed)
    if a.select == "best":
        P_all, *_ = predict(a.run_ours, X[cand], device)
        tr_all = trajectories(P_all, GY[cand], R0[cand], ps)
        ate_all = np.sqrt(np.mean(np.sum((tr_all - POS[cand].astype(np.float64)) ** 2, axis=2), axis=1))
        rel = ate_all / plen[cand]
        keep = cand[rel <= np.percentile(rel, a.best_pct)]
        long_ = keep[plen[keep] >= np.percentile(plen[keep], 50)]        # the longer half of the low-error windows
        idx = np.sort(rng.choice(long_, size=a.windows, replace=False))
        print(f"[odom] {len(X)} windows; {len(cand)} with above-median motion; {len(keep)} in the lowest-error quarter; picked {idx.tolist()}")
    else:
        idx = np.sort(rng.choice(cand, size=a.windows, replace=False))
        print(f"[odom] {len(X)} windows, {len(cand)} with path length above the 75th percentile; picked {idx.tolist()}")
    P_ours, *_ = predict(a.run_ours, X[idx], device)
    P_scr, *_ = predict(a.run_scratch, X[idx], device)
    tr_ours = trajectories(P_ours, GY[idx], R0[idx], ps); tr_scr = trajectories(P_scr, GY[idx], R0[idx], ps); gt = POS[idx].astype(np.float64)
    ate = lambda tr: np.sqrt(np.mean(np.sum((tr - gt) ** 2, axis=2), axis=1))
    a_ours, a_scr = ate(tr_ours), ate(tr_scr)
    print("[odom] ATE ours:", np.round(a_ours, 3), "scratch:", np.round(a_scr, 3))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ncol = min(4, len(idx)) if len(idx) <= 4 else 3; nrow = int(np.ceil(len(idx) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.6 * nrow + 0.4), facecolor="white", squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= len(idx): ax.axis("off"); continue
        g, o, s = gt[k], tr_ours[k], tr_scr[k]   # world frame is Y-up: plot the ground plane X-Z
        ax.plot(g[:, 0], g[:, 2], color=TEAL, lw=2.4, label="Ground truth")
        ax.plot(o[:, 0], o[:, 2], color=AMBER, lw=2.0, ls="--", label="InrtlWrist, fine-tuned")
        if not a.no_scratch: ax.plot(s[:, 0], s[:, 2], color=GREY, lw=1.6, ls=":", label="Same backbone, from scratch")
        ax.scatter([0], [0], s=40, color=INK, zorder=4)
        for tr, c in ((g, TEAL), (o, AMBER)) + (() if a.no_scratch else ((s, GREY),)): ax.scatter([tr[-1, 0]], [tr[-1, 2]], s=22, color=c, zorder=4)
        ax.set_aspect("equal", adjustable="datalim"); ax.set_facecolor("white")
        ax.tick_params(labelsize=8, colors=MUTED); [sp.set_color("#dde3ec") for sp in ax.spines.values()]
        ax.grid(True, color="#eef1f6", lw=0.8); ax.set_xlabel("x (m)", fontsize=8, color=MUTED); ax.set_ylabel("z (m)", fontsize=8, color=MUTED)
        ax.set_title(f"ATE {a_ours[k]:.2f} m" if a.no_scratch else f"ATE {a_ours[k]:.2f} m  vs  scratch {a_scr[k]:.2f} m", fontsize=9.5, color=INK)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=10, labelcolor=INK, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(a.out, dpi=200, facecolor="white"); print("[odom] wrote", a.out)


if __name__ == "__main__":
    main()
