#!/usr/bin/env python
"""One schematic figure per benchmark task for the documentation (docs/figures/task_<name>.png):
the wrist-IMU input window on the left, the encoder/decoder in the middle, the prediction target on the right.
Input traces are real Nymeria right-wrist IMU windows (one validation session); the skeleton is a ground-truth
Nymeria frame (docs/figures/pose_snapshot.npz); the labels, captions and classes are illustrative.

    python scripts/vis_task_figures.py --out docs/figures [--val-dir ...]
"""
import argparse, glob, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

TEAL, AMBER, VIOLET, GREEN, INK, MUTED, LINE, SURF2 = "#1d6f8f", "#c2662a", "#7c5fa8", "#2f7d6b", "#16202e", "#6b7788", "#dde3ec", "#f1f4f8"
CH = ["acc x", "acc y", "acc z", "gyr x", "gyr y", "gyr z"]
CHC = [TEAL, TEAL, TEAL, AMBER, AMBER, AMBER]
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
ARM = [14, 17, 19, 21, 23]  # R collar, R shoulder, R elbow, R wrist, R hand


def load_windows(val_dir, run):
    """Real 2.56 s wrist windows [N, 6, 128] at 50 Hz and their pelvis trajectories, from one validation session."""
    import torch
    import eval_odom_nymeria_altattn as lf
    f = sorted(glob.glob(os.path.join(val_dir, "*.pkl")))[3]
    sess = lf.load_session_local_vel_flu(f, target_hz=int(lf.FS), add_gravity=True)
    run = run if os.path.isabs(run) else os.path.join(REPO, run)
    cfg = torch.load(os.path.join(run, "checkpoints", "best.pt"), map_location="cpu", weights_only=False)["cfg"]
    X, Y, GY, R0, POS = lf.build_windows([sess], cfg, 128, 16, 64, int(lf.FS))
    return X, POS


def box(ax, x, y, w, h, text, fc="white", ec=LINE, fs=9.5, color=INK, bold=False, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025", fc=fc, ec=ec, lw=lw, transform=ax.transAxes, clip_on=False))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, fontweight="bold" if bold else "normal", transform=ax.transAxes, wrap=True)


def arrow(ax, x0, x1, y, text=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=16, color=MUTED, lw=1.4, clip_on=False))
    if text:
        ax.text((x0 + x1) / 2, y + 0.07, text, ha="center", va="bottom", fontsize=8.5, color=MUTED, transform=ax.transAxes)


def traces(ax, x, seconds, title, hz=50, label_hz=None):
    """x: [6, T]. Stacked lanes, one per channel."""
    T = x.shape[1]; t = np.arange(T) / hz
    for c in range(6):
        v = x[c]; v = (v - v.mean()) / (v.std() + 1e-6)
        ax.plot(t, v * 0.32 + (5 - c), color=CHC[c], lw=1.0)
        ax.text(-0.02 * seconds, 5 - c, CH[c], ha="right", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(-0.3 * seconds, seconds); ax.set_ylim(-0.8, 5.8); ax.set_yticks([])
    ax.set_xticks([0, seconds]); ax.set_xticklabels(["0", f"{seconds:g} s"], fontsize=8, color=MUTED)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0); ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.text(0, -0.65, f"one wrist, 6 channels, {label_hz or hz} Hz", fontsize=8, color=MUTED)


def skeleton(ax, pts, joints=None, color=TEAL, lw=2.2):
    for j, par in enumerate(SMPL_PARENTS):
        if par < 0: continue
        c = color if (joints is None or (j in joints and par in joints)) else LINE
        ax.plot([pts[j, 0], pts[par, 0]], [pts[j, 1], pts[par, 1]], color=c, lw=lw if c == color else 1.4, solid_capstyle="round", zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=10, color=[color if (joints is None or j in joints) else LINE for j in range(len(pts))], zorder=3)
    ax.set_aspect("equal"); ax.axis("off")


def figure(name, left_fn, mid_text, right_fn, out, right_width=0.34):
    fig = plt.figure(figsize=(12.5, 3.9), facecolor="white")
    axL = fig.add_axes([0.05, 0.14, 0.33, 0.72]); left_fn(axL)
    axM = fig.add_axes([0.385, 0.14, 0.235, 0.72]); axM.axis("off")
    arrow(axM, 0.0, 0.14, 0.5); box(axM, 0.15, 0.26, 0.70, 0.48, mid_text, fc=SURF2, ec=LINE, fs=8.6); arrow(axM, 0.86, 1.0, 0.5)
    axR = fig.add_axes([0.64, 0.1, right_width, 0.8]); right_fn(axR)
    fig.savefig(out, dpi=170, facecolor="white"); plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(REPO, "docs", "figures"))
    ap.add_argument("--val-dir", default="/projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist")
    ap.add_argument("--odom-run", default="runs/bench_odometry_nymeria_local__mae_5src__finetune__s42", help="any odometry run; only its config is read, to build windows")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    X, POS = load_windows(a.val_dir, a.odom_run)
    plen = np.sum(np.linalg.norm(np.diff(POS, axis=1), axis=2), axis=1)
    i = int(np.argsort(plen)[-20])                      # a window with clear motion
    i -= i % 2                                         # hop = T/2: windows i, i+2, i+4, i+6 are contiguous
    w256 = X[i]; w1024 = np.concatenate([X[i], X[i + 2], X[i + 4], X[i + 6]], axis=1)
    snap = np.load(os.path.join(REPO, "docs", "figures", "pose_snapshot.npz")); g = snap["g"][0]

    # 1. pose
    def right_pose(ax):
        ax.axis("off")
        a1 = ax.inset_axes([0.0, 0.0, 0.48, 0.86]); skeleton(a1, g[:, [0, 1]], color=TEAL)
        a2 = ax.inset_axes([0.52, 0.0, 0.48, 0.86]); skeleton(a2, g[:, [0, 1]], joints=set(ARM) | {12, 9, 6, 3, 0}, color=AMBER)
        ax.text(0.24, 0.98, "full-body pose\n23 SMPL joints \u00b7 MPJPE (mm)", transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK)
        ax.text(0.76, 0.98, "arm pose\n5 right-arm joints \u00b7 position error (cm)", transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK)
    figure("pose", lambda ax: traces(ax, w1024, 10.24, "Input: 10.24 s window (Nymeria, real right-wrist IMU)"),
           "encoder\n+ light attention decoder\n\u2192 joint rotations\nper 0.32 s", right_pose, os.path.join(a.out, "task_pose.png"))

    # 2. odometry
    def right_odom(ax):
        p = POS[i]; ax.plot(p[:, 0], p[:, 2], color=TEAL, lw=2.4); ax.scatter([p[0, 0]], [p[0, 2]], s=40, color=INK, zorder=4); ax.scatter([p[-1, 0]], [p[-1, 2]], s=26, color=TEAL, zorder=4)
        ax.set_aspect("equal", adjustable="datalim"); ax.set_xlabel("x (m)", fontsize=8, color=MUTED); ax.set_ylabel("z (m)", fontsize=8, color=MUTED); ax.tick_params(labelsize=7, colors=MUTED)
        for s in ax.spines.values(): s.set_color(LINE)
        ax.set_title("pelvis trajectory over the window (top-down)\nabsolute trajectory error, ATE (m)", fontsize=8.5, color=INK)
    figure("odometry", lambda ax: traces(ax, w256, 2.56, "Input: 2.56 s window (Nymeria, real right-wrist IMU)"),
           "encoder\n+ velocity decoder\n\u2192 wrist-frame velocity,\nrotated by the gyroscope\nand integrated", right_odom, os.path.join(a.out, "task_odometry.png"))

    # 3. microgesture
    def chips(ax, labels, hot, title, cols=1, fs=9):
        ax.axis("off"); ax.set_title(title, fontsize=8.5, color=INK)
        n = len(labels); rows = int(np.ceil(n / cols))
        for k, lab in enumerate(labels):
            r, c = k % rows, k // rows; y = 0.92 - (r + 0.5) * (0.84 / rows); x = 0.02 + c * (0.96 / cols)
            box(ax, x, y - 0.3 * (0.84 / rows), 0.96 / cols - 0.03, 0.6 * (0.84 / rows), lab, fc=(AMBER if lab == hot else "white"), ec=(AMBER if lab == hot else LINE), color=("white" if lab == hot else INK), fs=fs, bold=(lab == hot))
    figure("microgesture", lambda ax: traces(ax, w256[:, :100], 1.0, "Input: 1 s window (OpenWatch, 100 Hz)", hz=100),
           "encoder\n+ linear head\n\u2192 six windows per clip,\nvoted into a label", lambda ax: chips(ax, ["double clench", "double pinch", "pinch down", "pinch up", "slide", "background"], "pinch up", "clip label · macro-F1 over 6 classes"),
           os.path.join(a.out, "task_microgesture.png"))

    # 4. retrieval
    def right_text(ax):
        ax.axis("off"); ax.set_title("retrieve the caption of the motion (and the reverse)\nRecall@1 in galleries of 100", fontsize=8.5, color=INK)
        caps = ["a person walks forward and turns around", "a man kicks with his left leg", "someone sits down on a chair", "the person waves with the right hand"]
        for k, c in enumerate(caps):
            hot = k == 1; y = 0.86 - k * 0.22
            box(ax, 0.02, y - 0.08, 0.96, 0.16, "“%s”" % c, fc=(VIOLET if hot else "white"), ec=(VIOLET if hot else LINE), color=("white" if hot else INK), fs=8.5, bold=hot)
    figure("retrieval", lambda ax: traces(ax, w1024, 10.24, "Input: 10.24 s window (HumanML3D, simulated right wrist)"),
           "encoder\n+ projection into\nthe CLIP text space", right_text, os.path.join(a.out, "task_retrieval.png"))

    # 5. parkinson's
    def right_pd(ax):
        ax.axis("off"); ax.set_title("one prediction per participant\n(mean posterior over all windows) · participant macro-F1", fontsize=8.5, color=INK)
        cls = [("healthy control", 0.18), ("Parkinson's disease", 0.64), ("differential diagnosis", 0.18)]
        for k, (lab, p) in enumerate(cls):
            y = 0.72 - k * 0.26
            ax.text(0.02, y, lab, fontsize=9, color=INK, va="center", transform=ax.transAxes)
            ax.add_patch(FancyBboxPatch((0.50, y - 0.07), 0.46 * p, 0.14, boxstyle="round,pad=0.005", fc=(VIOLET if p > 0.5 else LINE), ec="none", transform=ax.transAxes))
            ax.text(0.50 + 0.46 * p + 0.02, y, f"{p:.2f}", fontsize=8, color=MUTED, va="center", transform=ax.transAxes)
    figure("parkinsons", lambda ax: traces(ax, w256, 2.56, "Input: 2.56 s windows (PADS, 11 assessment tasks x 2 wrists)"),
           "encoder\n+ linear head per window\n\u2192 posteriors averaged\nover all windows", right_pd, os.path.join(a.out, "task_parkinsons.png"))

    # 6. activity
    def right_har(ax):
        ax.axis("off"); ax.set_title("activity label per window · macro-F1 on four datasets", fontsize=8.5, color=INK)
        sets = [("HHAR (6)", "bike · sit · stand · walk · stairs up · stairs down"), ("MHEALTH (12)", "standing · walking · jogging · cycling · climbing stairs · ..."),
                ("PRISM (5)", "five daily-living procedures"), ("TransfHAR (60)", "writing · typing · tying a shoelace · stirring coffee · ...")]
        for k, (nm, ex) in enumerate(sets):
            y = 0.85 - k * 0.24
            ax.text(0.02, y, nm, fontsize=9, color=INK, va="center", fontweight="bold", transform=ax.transAxes)
            ax.text(0.02, y - 0.1, ex, fontsize=7.8, color=MUTED, va="center", transform=ax.transAxes)
    figure("activity", lambda ax: traces(ax, w256, 2.56, "Input: 2.56 s window (HHAR / MHEALTH / PRISM / TransfHAR)"),
           "encoder\n+ linear head", right_har, os.path.join(a.out, "task_activity.png"))


if __name__ == "__main__":
    main()
