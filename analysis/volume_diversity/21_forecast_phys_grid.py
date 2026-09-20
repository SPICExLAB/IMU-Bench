#!/usr/bin/env python
"""Pre-registered forecast for the T-phys random-type x volume grid.

The grid (not yet run): K = 1/10/50/100 % of the 1,000 T-phys clusters, drawn
nested and source-stratified at random; N = 0.1/1/10/100 % of the 4-source pool,
drawn uniformly inside the chosen clusters. 12 feasible cells, one of which
(K100 N100 = the whole pool) is already measured as vd_ek1000_f100.

Forecast = per-task ridge regression fitted on every completed 4-source arm at
the same 50k-step budget and probe protocol:
    y ~ 1 + log10(N) + log10(phys effective types) + hinge(N below 2,000 windows)
Uncertainty = leave-one-out RMSE. A constant model (the mean of all arms) is fitted
alongside; if the full model does not beat it out of sample, the honest forecast
is "flat" and the constant model's error is the interval.

Inputs are the realised N / effective types / source mix of each planned cell from
the seed-42 simulation on the CURRENT T-phys labels (before the EatingSpeed gyro
unit fix). Writes out/phys_random_grid_forecast.json
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

OUT = "analysis/volume_diversity/out"
TASKS = [("har_wisdm", "macro_f1", "WISDM", "macro-F1", 1), ("har_utd_mhad", "macro_f1", "UTD-MHAD", "macro-F1", 1),
         ("har_pamap2", "macro_f1", "PAMAP2", "macro-F1", 1), ("har_recofit", "macro_f1", "RecoFit", "macro-F1", 1),
         ("microgesture_openwatch", "clip_macro_f1", "Microgesture", "clip F1", 1),
         ("arm_pose_nymeria", "pos_err_cm", "Arm pose", "cm", -1), ("full_body_pose_nymeria", "mpjpe_mm", "Full-body pose", "mm", -1),
         ("odometry_nymeria_worldframe", "ate_m", "Odometry", "ATE m", -1)]
SCRATCH = {"har_wisdm": 0.6056, "har_utd_mhad": 0.7010, "har_pamap2": 0.2190, "har_recofit": 0.6630,
           "microgesture_openwatch": 0.4876, "arm_pose_nymeria": 8.1529, "full_body_pose_nymeria": 92.428,
           "odometry_nymeria_worldframe": 0.1400}
NOISE = {"arm_pose_nymeria": 0.108}
CELLS = [  # K %, N %, realised N, realised phys effective types, window source mix %
    (1, 0.1, 1738, 8.6, [53, 26, 3, 18]), (1, 1, 15050, 8.6, [54, 26, 3, 17]),
    (10, 0.1, 1738, 81.9, [71, 17, 4, 8]), (10, 1, 17381, 83.5, [71, 17, 4, 8]), (10, 10, 172876, 83.2, [71, 17, 4, 8]),
    (50, 0.1, 1738, 356.5, [69, 19, 2, 9]), (50, 1, 17381, 415.4, [70, 18, 3, 9]), (50, 10, 173809, 420.2, [71, 18, 3, 9]),
    (100, 0.1, 1738, 624.0, [71, 18, 2, 9]), (100, 1, 17381, 808.0, [72, 17, 3, 8]), (100, 10, 173809, 830.8, [72, 17, 3, 8]),
    (100, 100, 1738094, 832.8, [72, 17, 3, 8]),
]
MEASURED = {(100, 100): "vd_ek1000_f100"}


def result(task, metric, arm):
    p = f"bench/results/{task}/{arm}__probe__s42.json"
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    for k in ("metrics", "test", "result"):
        if isinstance(j.get(k), dict) and metric in j[k]:
            return float(j[k][metric])
    return float(j[metric]) if metric in j else None


def val_loss(arm):
    p, v = f"runs/{arm}/logs/metrics.txt", None
    if os.path.exists(p):
        for line in open(p):
            m = re.search(r"val[_ ]loss[=: ]+([0-9.]+)", line)
            if m:
                v = float(m.group(1))
    return v


def design(n, eff):
    ln = np.log10(n)
    return np.column_stack([np.ones_like(ln), ln, np.log10(eff), np.maximum(0.0, np.log10(2000) - ln)])


def ridge(X, y, lam=1e-2):
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X[:, 1:] - mu) / sd])
    P = lam * np.eye(Z.shape[1]); P[0, 0] = 0
    w = np.linalg.solve(Z.T @ Z + P, Z.T @ y)
    return lambda Xn: np.column_stack([np.ones(len(Xn)), (Xn[:, 1:] - mu) / sd]) @ w


def loo(X, y, const=False):
    err = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        pred = y[m].mean() if const else ridge(X[m], y[m])(X[i:i + 1])[0]
        err.append(y[i] - pred)
    return float(np.sqrt(np.mean(np.square(err))))


def main():
    arms = {}
    for f in ["arms.json", "axis_arms.json", "subject_arms.json", "enc_type_arms.json", "enc_type_volume_grid.json"]:
        d = json.load(open(os.path.join(OUT, f)))
        for k, v in d.get("arms", d).items():
            if isinstance(v, dict) and "phys" in v:
                arms.setdefault(k, {}).update(v)
    feats = {k: (float(v.get("n_windows") or v.get("n")), float(v["phys"]["effective_types_shannon"]))
             for k, v in arms.items() if (v.get("n_windows") or v.get("n"))}
    cn = np.array([c[2] for c in CELLS], float); ce = np.array([c[3] for c in CELLS], float)
    Xc = design(cn, ce)

    out = {"cells": [{"k_pct": c[0], "n_pct": c[1], "n": c[2], "phys_eff": c[3], "mix": c[4],
                      "measured_arm": MEASURED.get((c[0], c[1]))} for c in CELLS], "tasks": []}
    targets = [("val_loss", None, "MAE val loss", "loss", -1)] + TASKS
    for task, metric, name, unit, sign in targets:
        ys, ns, es, names = [], [], [], []
        for arm, (n, e) in feats.items():
            y = val_loss(arm) if task == "val_loss" else result(task, metric, arm)
            if y is not None and e > 0:
                ys.append(y); ns.append(n); es.append(e); names.append(arm)
        y = np.array(ys); X = design(np.array(ns), np.array(es))
        rm_full, rm_const = loo(X, y), loo(X, y, const=True)
        pred = ridge(X, y)(Xc)
        # nearest measured arm per cell in standardised (log N, log eff) space
        Z = np.column_stack([np.log10(ns), np.log10(es)]); zs = Z.std(0)
        zc = np.column_stack([np.log10(cn), np.log10(ce)])
        near = []
        for i in range(len(CELLS)):
            d = np.sqrt((((Z - zc[i]) / zs) ** 2).sum(1)); j = int(np.argmin(d))
            near.append({"arm": names[j], "value": float(y[j]), "dist": float(d[j]),
                         "n": float(ns[j]), "phys_eff": float(es[j])})
        measured = [result(task, metric, c["measured_arm"]) if (c["measured_arm"] and task != "val_loss")
                    else (val_loss(c["measured_arm"]) if c["measured_arm"] else None) for c in out["cells"]]
        # judge the model where the new grid lives: arms with >= 1,738 windows only
        inr = np.array(ns) >= 1738
        rm_full_in = loo(X[inr], y[inr]); rm_const_in = loo(X[inr], y[inr], const=True)
        informative = rm_full_in < 0.9 * rm_const_in
        spread = float(pred.max() - pred.min())
        interval = rm_full_in if informative else rm_const_in
        # extrapolation: cell diversity below what in-range arms of this task ever reached
        eff_min_in = float(np.array(es)[inr].min())
        extrap = [bool(c[3] < 0.8 * eff_min_in) for c in CELLS]
        if not informative:
            pred = np.full(len(CELLS), float(y[inr].mean()))
        i_k1, i_k100, i_n01, i_n100 = 1, 9, 8, 11
        types_eff = float(pred[i_k100] - pred[i_k1]) * (1 if sign > 0 else -1)
        vol_eff = float(pred[i_n100] - pred[i_n01]) * (1 if sign > 0 else -1)
        out["tasks"].append({"task": task, "name": name, "unit": unit, "higher_better": sign > 0,
                             "n_arms": len(y), "loo_rmse_model": rm_full, "loo_rmse_const": rm_const,
                             "model_informative": bool(informative), "const_value": float(y[inr].mean()),
                             "n_arms_in_range": int(inr.sum()), "loo_rmse_model_in_range": rm_full_in,
                             "loo_rmse_const_in_range": rm_const_in, "eff_min_in_range": eff_min_in,
                             "extrapolated": extrap, "types_effect_at_1pct": types_eff, "volume_effect_at_k100": vol_eff,
                             "types_effect_resolvable": bool(abs(types_eff) > 2 * interval),
                             "volume_effect_resolvable": bool(abs(vol_eff) > 2 * interval),
                             "scratch": SCRATCH.get(task), "noise_band": NOISE.get(task),
                             "pred": [float(p) for p in pred], "pred_spread": spread,
                             "interval": interval,
                             "measured": measured, "nearest": near,
                             "observed_range": [float(y.min()), float(y.max())]})
        print(f"{name:15s} in-range arms={int(inr.sum()):2d}  LOO model {rm_full_in:.4f} const {rm_const_in:.4f} "
              f"{'MODEL' if informative else 'FLAT '}  interval {interval:.4f}  types effect@1% {types_eff:+.4f} "
              f"({'resolvable' if abs(types_eff) > 2*interval else 'within 2x error'})  volume effect@K100 {vol_eff:+.4f} "
              f"({'resolvable' if abs(vol_eff) > 2*interval else 'within 2x error'})  extrapolated cells {sum(extrap)}  "
              f"measured K100N100 {measured[11]:.4f} vs forecast {pred[11]:.4f}")
    json.dump(out, open(os.path.join(OUT, "phys_random_grid_forecast.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "phys_random_grid_forecast.json"))


if __name__ == "__main__":
    main()
