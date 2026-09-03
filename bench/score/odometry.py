"""World-frame odometry scorer: vel MSE, ATE (m) and drift (%) over ALL val
windows of a trainer-produced run (probe / finetune / scratch all save
`odom_decoder` in the checkpoint). Reuses the window builder and the
cumulative-sum reconstruction from scripts/eval_odom_nymeria_worldframe.py and
the checkpoint loader from scripts/vis_odom_scratch.py.
"""
from __future__ import annotations

import glob
import os
import sys
from typing import Any, Dict, Tuple

import numpy as np

from bench.registry_io import REPO
from bench.score.common import load_ckpt_meta

sys.path.insert(0, os.path.join(REPO, "scripts"))


def score(run_dir: str, ckpt_path: str, task: Dict[str, Any], max_windows: int = 0) -> Tuple[Dict[str, float], Dict[str, Any]]:
    import torch
    import eval_odom_nymeria_worldframe as wf          # noqa: E402  (scripts/)
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
    va_sess = [r for f in va_files
               if (r := wf.load_session_local_vel_flu(f, target_hz=int(wf.FS), add_gravity=True)) is not None]
    va = dict(zip(["X", "Y", "POS"], wf.build_windows(va_sess, cfg, T, ps, hop, int(wf.FS))))
    n_total = len(va["X"])
    idx = np.arange(n_total)
    if max_windows and n_total > max_windows:
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(n_total, size=max_windows, replace=False))

    acc_only = bool(cfg_get(cfg, ["data", "acc_only"], False))
    X = va["X"][idx]
    if acc_only:
        X = X[:, :int(getattr(enc, "num_input_channels", enc.in_channels))]

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).float().to(device)
            tokens, _ = enc.forward_tokens(xb)
            pred = decoder(tokens)
            B, Pt, _ = pred.shape
            preds.append(pred.reshape(B, Pt, 4, 3).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    ates, ferrs, plens = [], [], []
    for i in range(len(preds)):
        ate, ferr, plen = wf.reconstruct_and_score(preds[i], va["POS"][idx[i]], ps, target_hz=int(wf.FS))
        ates.append(ate); ferrs.append(ferr); plens.append(plen)
    ates, ferrs, plens = map(np.asarray, (ates, ferrs, plens))
    moving = plens > 0.1
    drift = float(np.mean(ferrs[moving] / plens[moving]) * 100.0) if moving.any() else float("nan")
    metrics = {
        "vel_mse": float(np.mean((preds - va["Y"][idx]) ** 2)),
        "ate_m": float(np.mean(ates)),
        "drift_pct": drift,
    }
    info = {"checkpoint_step": ck_step, "n_val_windows": int(len(idx)), "n_val_windows_total": int(n_total),
            "n_val_sessions": len(va_sess), "T": int(T)}
    return metrics, info
