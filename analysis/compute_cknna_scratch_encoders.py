#!/usr/bin/env python3
"""
CKNNA representation similarity across downstream encoders trained FROM SCRATCH.

Motivation: EXPERIMENTS.md's "Scratch" column is a *performance* baseline
(supervised, no SSL pretraining) trained independently for each downstream
task (HAR, arm pose, full-body pose, gesture, odometry, CLIP). This script
asks a different question: despite having no shared pretraining and no
shared training data, do these task-specific encoders converge to similar
*representation geometries* -- the "platonic representation hypothesis"
(Huh et al. 2024) question, applied across our own downstream tasks instead
of across different published models.

Method:
  1. Draw N "probe" windows from Nymeria wrist-IMU val sessions (the pkl
     files under NYMERIA_VAL_PKL_ROOT) -- one shared set of (session, center
     timestamp) anchors, reused identically for every task.
  2. For each task's encoder, crop its OWN native window length (2.56s or
     10.24s, per its training config) centered at that SAME anchor, run that
     encoder's OWN preprocessing pipeline (from its checkpoint's stored
     cfg), and take the pooled [D] embedding. Same real-world moment in,
     each encoder's own native resolution/normalization -- not a single
     forced window length.
  3. Compute pairwise CKNNA (and linear CKA for reference) between every
     pair of encoders' [N, D] embedding sets.

Known caveats (by design, not oversights -- see EXPERIMENTS.md write-up):
  - Only arm-pose/full-body-pose/odometry are trained on Nymeria wrist IMU,
    so only those three see in-distribution probe data. HAR (WISDM),
    gesture (OpenWatch), and CLIP (HumanML3D) see fully out-of-domain
    signal at eval time -- expected, analogous to probing a model with data
    it never trained on to test representation geometry, not task accuracy.
  - Gesture's encoder was trained on watch-frame axes (acc_F/L/U,
    gyro_F/L/U) but receives Nymeria's body-frame x/y/z channels in the same
    channel *positions* -- a coordinate-convention mismatch on top of the
    domain shift above.

Usage:
    python analysis/compute_cknna_scratch_encoders.py \
        --n-probe 512 --topk 10 --out-dir analysis/cknna_scratch
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DEncoderAltAttn
from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.utils.helpers import cfg_get
from analysis.cknna_metric import cknna, cka

FS = 50.0  # Nymeria wrist IMU sampling rate (imu50hz pkl files)

# task label -> (run dir under runs/, checkpoint filename)
TASKS = {
    "HAR (WISDM)":               ("vit1d_har_supervised_wisdm_altattn_scratch", "best.pt"),
    "Arm pose (Nymeria)":        ("vit1d_arm_pose_supervised_nymeria_altattn_scratch", "best.pt"),
    "Full-body pose (Nymeria)":  ("vit1d_full_body_pose_supervised_nymeria_altattn_scratch", "best_mpjpe.pt"),
    "Gesture (OpenWatch)":       ("vit1d_gesture_supervised_openwatch_6class_altattn_scratch", "best.pt"),
    "Odometry (Nymeria, world)": ("vit1d_odom_worldframe_supervised_nymeria_altattn_scratch", "best.pt"),
    "CLIP (HumanML3D)":          ("vit1d_clip_humanml3d_scratch", "best.pt"),
}

NYMERIA_VAL_PKL_ROOT = "/projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist"

# Sequential blue ramp, from the repo's dataviz palette (references/palette.md).
_SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


# ── Encoder loading ─────────────────────────────────────────────────────────

def load_encoder(runs_root: str, run_name: str, ckpt_file: str, device: torch.device):
    ckpt_path = os.path.join(runs_root, run_name, "checkpoints", ckpt_file)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]

    alt_attention = bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False))
    enc_cls = ViT1DEncoderAltAttn if alt_attention else ViT1DEncoder
    enc = enc_cls(cfg).to(device)
    enc.load_state_dict(ckpt["model"], strict=True)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    window_seconds = float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56))
    sensor_columns = cfg_get(cfg, ["data", "sensor_columns"], None)
    step = ckpt.get("step", "?")
    print(f"  loaded {run_name} (step={step}, alt_attention={alt_attention}, "
          f"window_seconds={window_seconds}, sensor_columns={sensor_columns})")
    return enc, cfg, window_seconds


# ── Probe set: shared Nymeria wrist-IMU anchors ─────────────────────────────

def load_probe_sessions(pkl_root: str, n_sessions: int, seed: int) -> list[np.ndarray]:
    files = sorted(glob.glob(os.path.join(pkl_root, "*.pkl")))
    if not files:
        raise FileNotFoundError(f"No pkl files found under {pkl_root}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(files, size=min(n_sessions, len(files)), replace=False)

    sessions = []
    for fp in chosen:
        with open(fp, "rb") as f:
            d = pickle.load(f)
        accel = d["wrist_imu_data"]["accel"].numpy()  # [T, 3]
        gyro = d["wrist_imu_data"]["gyro"].numpy()    # [T, 3]
        sig = np.concatenate([accel, gyro], axis=1).astype(np.float32)  # [T, 6]
        sessions.append(sig)
    print(f"  loaded {len(sessions)} Nymeria val sessions "
          f"({sum(len(s) for s in sessions) / FS / 60:.1f} min total)")
    return sessions


def sample_anchors(sessions: list[np.ndarray], n_probe: int, max_half_window: int,
                    seed: int) -> list[tuple[int, int]]:
    """Pick (session_idx, center_sample) pairs with room for the LARGEST task window
    on both sides, so the exact same anchors are reusable across every task."""
    rng = np.random.default_rng(seed)
    valid = [si for si, sig in enumerate(sessions) if len(sig) > 2 * max_half_window + 1]
    if not valid:
        raise ValueError("No probe session long enough for the largest task window")

    anchors = []
    for i in range(n_probe):
        si = valid[i % len(valid)]
        T = len(sessions[si])
        c = int(rng.integers(max_half_window, T - max_half_window))
        anchors.append((si, c))
    return anchors


# ── Feature extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(enc, cfg, window_seconds: float, sessions: list[np.ndarray],
                        anchors: list[tuple[int, int]], device: torch.device,
                        batch_size: int = 64) -> torch.Tensor:
    num_samples = int(round(window_seconds * FS))
    half = num_samples // 2

    windows = []
    for si, c in anchors:
        raw = sessions[si][c - half: c - half + num_samples]  # [T, 6]
        proc = preprocess_window(raw.copy(), cfg)              # [6, T] float32
        if proc is None:
            raise RuntimeError(
                "preprocess_window dropped a probe window -- Nymeria wrist IMU "
                "should never have enough missing data to trigger this; "
                "investigate before trusting the CKNNA result."
            )
        windows.append(proc)
    X = np.stack(windows).astype(np.float32)  # [N, 6, T]

    feats = []
    for i in range(0, len(X), batch_size):
        x = torch.from_numpy(X[i:i + batch_size]).to(device)
        f = enc(x)  # [B, D] pooled (CLS or mean, per encoder's own `pooling`)
        feats.append(f.cpu())
    feats = torch.cat(feats, dim=0)
    return Fn.normalize(feats, dim=-1)  # unit-norm rows -> linear kernel = cosine sim


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_heatmap(matrix: np.ndarray, labels: list[str], title: str, out_path: str):
    n = len(labels)
    cmap = LinearSegmentedColormap.from_list("seq_blue", _SEQ_BLUE)

    fig, ax = plt.subplots(figsize=(1.15 * n + 2, 1.05 * n + 1.5))
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            color = "white" if v > 0.6 else "#0d1117"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("similarity", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoders", default=None,
                   help="override the encoder set: 'Label=run_name:ckpt_file' entries separated by ';'. "
                        "Used for the task-similarity table, which compares the MAE encoder after "
                        "fine-tuning on each benchmark task instead of the from-scratch encoders.")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--nymeria-val-root", default=NYMERIA_VAL_PKL_ROOT)
    p.add_argument("--n-sessions", type=int, default=16,
                    help="number of Nymeria val sessions to draw probe anchors from")
    p.add_argument("--n-probe", type=int, default=512, help="number of probe windows")
    p.add_argument("--topk", type=int, default=10, help="CKNNA mutual-kNN neighborhood size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="analysis/cknna_scratch")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    global TASKS
    args = parse_args()
    if args.encoders:          # task-similarity run: MAE after fine-tuning on each source task
        TASKS = {}
        for part in args.encoders.split(";"):
            lab, spec = part.split("=", 1)
            run, _, ck = spec.partition(":")
            TASKS[lab.strip()] = (run.strip(), (ck or "best.pt").strip())
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading encoders...")
    encoders, cfgs, window_secs = {}, {}, {}
    for name, (run_name, ckpt_file) in TASKS.items():
        enc, cfg, ws = load_encoder(args.runs_root, run_name, ckpt_file, device)
        encoders[name], cfgs[name], window_secs[name] = enc, cfg, ws

    print("\nLoading Nymeria val probe sessions...")
    sessions = load_probe_sessions(args.nymeria_val_root, args.n_sessions, args.seed)

    max_half = int(round(max(window_secs.values()) * FS)) // 2
    anchors = sample_anchors(sessions, args.n_probe, max_half, args.seed)
    print(f"  sampled {len(anchors)} shared probe anchors "
          f"(max half-window={max_half} samples)")

    print("\nExtracting embeddings per task...")
    embeddings = {}
    for name in TASKS:
        feats = extract_embeddings(encoders[name], cfgs[name], window_secs[name],
                                    sessions, anchors, device)
        embeddings[name] = feats
        print(f"  {name}: {tuple(feats.shape)}")

    names = list(TASKS.keys())
    n = len(names)
    cknna_mat = np.eye(n, dtype=np.float32)
    cka_mat = np.eye(n, dtype=np.float32)

    print(f"\nComputing pairwise CKNNA (topk={args.topk}) and linear CKA...")
    for i in range(n):
        for j in range(i + 1, n):
            fa, fb = embeddings[names[i]], embeddings[names[j]]
            s_cknna = cknna(fa, fb, topk=args.topk, unbiased=True)
            s_cka = cka(fa, fb, unbiased=False)
            cknna_mat[i, j] = cknna_mat[j, i] = s_cknna
            cka_mat[i, j] = cka_mat[j, i] = s_cka
            print(f"  {names[i]:<28} vs {names[j]:<28} "
                  f"CKNNA={s_cknna:.4f}  CKA={s_cka:.4f}")

    np.savez(os.path.join(args.out_dir, "cknna_results.npz"),
              cknna=cknna_mat, cka=cka_mat, task_names=np.array(names))

    import csv
    csv_path = os.path.join(args.out_dir, "cknna_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_a", "task_b", "cknna", "cka"])
        for i in range(n):
            for j in range(n):
                if i < j:
                    w.writerow([names[i], names[j], f"{cknna_mat[i, j]:.4f}", f"{cka_mat[i, j]:.4f}"])
    print(f"\nSaved {csv_path}")

    plot_heatmap(cknna_mat, names,
                 f"CKNNA representation similarity\n(Scratch encoders, topk={args.topk}, "
                 f"N={args.n_probe} Nymeria probe windows)",
                 os.path.join(args.out_dir, "cknna_heatmap.png"))
    plot_heatmap(cka_mat, names,
                 "Linear CKA (reference)\n(Scratch encoders, same probe set)",
                 os.path.join(args.out_dir, "cka_heatmap.png"))


if __name__ == "__main__":
    main()
