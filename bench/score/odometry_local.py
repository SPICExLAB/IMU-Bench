"""Local-frame odometry scorer: vel MSE, ATE (m) and drift (%) over ALL val windows of a
trainer-produced run (probe / finetune / scratch all save `odom_decoder` in the checkpoint).

Target (imu_lm/data/nymeria_real_wrist_odom_loader_FLU.py): pelvis velocity expressed in the
wrist FLU frame, 4 sub-patches per 0.32 s patch. Reconstruction follows
scripts/eval_odom_nymeria_altattn.py::reconstruct_and_score: the wrist orientation is tracked
from the ground-truth orientation at the window start by integrating the raw gyro, each
predicted FLU velocity is rotated to the world frame at its sub-patch midpoint, and the
positions are a cumulative sum. The rotation integration here is vectorised over windows;
`_check_against_reference` verifies it against the per-window reference implementation.
Replaces the world-frame task (bench/score/odometry.py) from 2026-09-18.
"""
from __future__ import annotations

import glob
import os
import sys
from typing import Any, Dict, Tuple

import numpy as np

from bench.registry_io import REPO
from bench.score.common import load_ckpt_meta, nymeria_participant as participant_of, save_units, units_safely

sys.path.insert(0, os.path.join(REPO, "scripts"))


def _exp_so3_batch(w: np.ndarray) -> np.ndarray:
    """Rodrigues for a batch of rotation vectors [N, 3] -> [N, 3, 3] (float64)."""
    theta = np.linalg.norm(w, axis=1)
    small = theta < 1e-9
    k = np.where(small[:, None], 0.0, w / np.where(small, 1.0, theta)[:, None])
    K = np.zeros((len(w), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(theta)[:, None, None], (1.0 - np.cos(theta))[:, None, None]
    R = np.eye(3)[None] + s * K + c * (K @ K)
    R[small] = np.eye(3)
    return R


def reconstruct_batch(v_flu: np.ndarray, gyro: np.ndarray, R0: np.ndarray, pos_gt: np.ndarray, ps: int,
                      target_hz: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised eval_odom_nymeria_altattn.reconstruct_and_score.

    v_flu [N, Pt, 4, 3], gyro [N, T, 3] raw FLU rad/s, R0 [N, 3, 3] world<-FLU at window start,
    pos_gt [N, Pt*4+1, 3] world pelvis positions relative to the window start.
    Returns per-window (ate, final_error, path_length).
    """
    N, Pt = v_flu.shape[:2]
    Q, dt = ps // 4, 1.0 / target_hz
    Ns = Pt * 4
    mids = {i * Q + Q // 2 for i in range(Ns)}
    R = R0.astype(np.float64).copy()
    R_at = {0: R.copy()} if 0 in mids else {}
    for t in range(gyro.shape[1]):
        R = R @ _exp_so3_batch(gyro[:, t].astype(np.float64) * dt)
        if t + 1 in mids:
            R_at[t + 1] = R.copy()
    v = v_flu.reshape(N, Ns, 3).astype(np.float64)
    steps = np.stack([np.einsum("nij,nj->ni", R_at[i * Q + Q // 2], v[:, i]) for i in range(Ns)], axis=1) * (Q * dt)
    traj = np.concatenate([np.zeros((N, 1, 3)), np.cumsum(steps, axis=1)], axis=1)
    pos_gt = pos_gt.astype(np.float64)
    ate = np.sqrt(np.mean(np.sum((traj - pos_gt) ** 2, axis=2), axis=1))
    ferr = np.linalg.norm(traj[:, -1] - pos_gt[:, -1], axis=1)
    plen = np.sum(np.linalg.norm(np.diff(pos_gt, axis=1), axis=2), axis=1)
    return ate, ferr, plen


def _check_against_reference(preds, va, idx, ps, n=16):
    import eval_odom_nymeria_altattn as ref
    k = min(n, len(preds))
    ate_b, ferr_b, plen_b = reconstruct_batch(preds[:k], va["GY"][idx[:k]], va["R0"][idx[:k]], va["POS"][idx[:k]], ps)
    for i in range(k):
        a, f, p = ref.reconstruct_and_score(preds[i], va["GY"][idx[i]], va["R0"][idx[i]], va["POS"][idx[i]], ps)
        if not (np.isclose(a, ate_b[i], rtol=1e-6, atol=1e-9) and np.isclose(f, ferr_b[i], rtol=1e-6, atol=1e-9)
                and np.isclose(p, plen_b[i], rtol=1e-6, atol=1e-9)):
            raise AssertionError(f"vectorised reconstruction differs from reference on window {i}: "
                                 f"{(a, f, p)} vs {(ate_b[i], ferr_b[i], plen_b[i])}")


def _build_windows_by_session(lf, sessions, cfg, T, ps, hop):
    """lf.build_windows run one session at a time and concatenated: identical arrays (the
    function keeps no cross-session state) plus the session index of every window."""
    parts, sess_of = [], []
    for si, sess in enumerate(sessions):
        try:
            arrs = lf.build_windows([sess], cfg, T, ps, hop, int(lf.FS))
        except ValueError as e:     # np.stack of an empty list: the session yields no window
            if "at least one array" not in str(e):
                raise
            continue
        parts.append(arrs)
        sess_of.append(np.full(len(arrs[0]), si, dtype=np.int64))
    arrays = tuple(np.concatenate([p[k] for p in parts], axis=0) for k in range(5))
    return arrays, np.concatenate(sess_of)


def score(run_dir: str, ckpt_path: str, task: Dict[str, Any], max_windows: int = 0) -> Tuple[Dict[str, float], Dict[str, Any]]:
    import torch
    import eval_odom_nymeria_altattn as lf              # noqa: E402  (scripts/)
    from vis_odom_scratch import load_scratch_model     # noqa: E402
    from imu_lm.utils.helpers import cfg_get

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, decoder, cfg, T, ps = load_scratch_model(run_dir, None, device)   # builds modules from ckpt cfg
    meta = load_ckpt_meta(ckpt_path)                                       # then load the SELECTED ckpt
    enc.load_state_dict(meta["_obj"]["model"], strict=True)
    decoder.load_state_dict(meta["_obj"]["odom_decoder"], strict=True)
    enc.eval(); decoder.eval()
    ck_step = int(meta["step"])
    hop = T // 2

    val_dir = task["dataset"]["val"]
    va_files = sorted(glob.glob(os.path.join(val_dir, "*.pkl")))
    va_loaded = [(f, r) for f in va_files
                 if (r := lf.load_session_local_vel_flu(f, target_hz=int(lf.FS), add_gravity=True)) is not None]
    va_sess = [r for _, r in va_loaded]
    va_sess_files = [os.path.basename(f) for f, _ in va_loaded]
    # one session at a time -> the same windows as lf.build_windows(va_sess, ...) + a session index
    win_arrays, win_sess = _build_windows_by_session(lf, va_sess, cfg, T, ps, hop)
    va = dict(zip(["X", "Y", "GY", "R0", "POS"], win_arrays))
    n_total = len(va["X"])
    idx = np.arange(n_total)
    if max_windows and n_total > max_windows:
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(n_total, size=max_windows, replace=False))

    # build_windows already cuts X to the acc-only prefix when cfg.data.acc_only is set
    # (GY keeps the raw gyro: it only drives the classical orientation integration).
    X = va["X"][idx]
    n_in = int(getattr(enc, "num_input_channels", enc.in_channels))
    if bool(cfg_get(cfg, ["data", "acc_only"], False)) and X.shape[1] != n_in:
        X = X[:, :n_in]

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).float().to(device)
            tokens, _ = enc.forward_tokens(xb)
            pred = decoder(tokens)
            B, Pt, _ = pred.shape
            preds.append(pred.reshape(B, Pt, 4, 3).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    _check_against_reference(preds, va, idx, ps)
    ates, ferrs, plens = [], [], []
    for i in range(0, len(preds), 4096):
        j = idx[i:i + 4096]
        a, f, p = reconstruct_batch(preds[i:i + 4096], va["GY"][j], va["R0"][j], va["POS"][j], ps)
        ates.append(a); ferrs.append(f); plens.append(p)
    ates, ferrs, plens = map(np.concatenate, (ates, ferrs, plens))
    moving = plens > 0.1
    drift = float(np.mean(ferrs[moving] / plens[moving]) * 100.0) if moving.any() else float("nan")
    metrics = {
        "vel_mse": float(np.mean((preds - va["Y"][idx]) ** 2)),
        "ate_m": float(np.mean(ates)),
        "drift_pct": drift,
    }
    info = {"checkpoint_step": ck_step, "n_val_windows": int(len(idx)), "n_val_windows_total": int(n_total),
            "n_val_sessions": len(va_sess), "T": int(T), "frame": "wrist FLU (local)"}
    units_safely(_save_odom_units, run_dir, ckpt_path, task, metrics, ck_step, idx, n_total, win_sess,
                 va_sess_files, ates, ferrs, plens, np.mean((preds - va["Y"][idx]) ** 2, axis=(1, 2, 3)), T, hop)
    return metrics, info


def _save_odom_units(run_dir, ckpt_path, task, metrics, ck_step, idx, n_total, win_sess, va_sess_files,
                     ates, ferrs, plens, vel_mse_w, T, hop):
    """Per-window sidecar: ATE (the primary), final error, path length, vel MSE + session / participant."""
    sess_files = np.asarray(va_sess_files)[win_sess[idx]]
    within = idx - np.searchsorted(win_sess, win_sess[idx])      # window number inside its session
    save_units(run_dir, ckpt_path, task, kind="mean",
               arrays={"unit_key": np.char.add(np.char.add(sess_files, "#"), within.astype(str)),
                       "value": ates, "final_err_m": ferrs, "path_len_m": plens, "vel_mse": vel_mse_w},
               groups={"participant": [participant_of(f) for f in sess_files], "session": sess_files},
               metric_reported=metrics.get(task["metrics"]["primary"]), checkpoint_step=ck_step,
               notes=f"unit = {T}-sample val window (hop {hop}); value = per-window ATE (m); "
                     f"session = val pkl file; participant parsed from the file name"
                     + (f"; random subset of {len(idx)}/{n_total} windows (max_windows)" if len(idx) != n_total else ""),
               extra_meta={"n_val_windows_total": int(n_total)})
