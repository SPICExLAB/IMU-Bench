"""Bootstrap confidence intervals for IMU-Bench cells, computed from the per-unit
sidecars that every scorer writes to

    bench/results/_units/<task>/<encoder>__<protocol>__s<seed>.npz

(see bench/score/common.py::save_units). Every cell is ONE training seed (a
standing project decision), so these CIs quantify evaluation-sample
uncertainty only: how much the reported number would move if the same
checkpoint were scored on a fresh draw of evaluation units from the same
population. They say nothing about seed-to-seed training variance.

Resampling plan
---------------
Each sidecar lists its group levels coarse -> fine (HAR: subject, session;
pose / odometry: participant, session; microgesture: subject; PD: subject;
language: caption). A resample draws groups with replacement and gives every
unit an integer weight (= how many times its group was drawn); the metric is
recomputed on the weighted units, which is exactly the metric of the
duplicated sample. The level is chosen automatically:

  1. the COARSEST level with >= MIN_GROUPS (=10) groups -- below that a cluster
     bootstrap is degenerate (HHAR / MHEALTH have 2 val subjects, PRISM 6,
     microgesture test 5); else the next finer level; finally the units;
  2. classification metrics (macro-F1 / bal-acc) with label-pure groups (every
     group >= 95 % one class, e.g. HAR sessions that are single-activity
     segments, PD subjects, single clips) are resampled WITHIN class strata, so
     every class stays in every resample. Unstratified, a resample that drops
     a class's only sessions still scores that class (predicted elsewhere) at
     F1 = 0 and biases macro-F1 downward (MHEALTH: 0.69 -> lower bound 0.50);
  3. a stratified level where most classes have a single group (User-watch:
     60 classes, 70 val recordings) carries no between-group information, so
     the plan falls through to the next finer level.
TASK_LEVELS overrides the candidate levels per task (User-watch is a time
split within recordings -- participants are in train AND val -- so subjects
are not its sampling unit). The CI at every level is also reported
(`levels` in summary.json), so the sensitivity to the choice is visible.

95 % percentile interval, 1,000 resamples, numpy default_rng(seed=0) -- the
reported `lo` / `hi`. Macro-F1 over heterogeneous sessions is a skewed statistic
(HHAR: bootstrap mean 0.03-0.04 below the point estimate), so every cell also
carries `boot_mean` / `bias` and a BCa interval (`bca_lo` / `bca_hi`;
acceleration from a jackknife over the resampled groups, skipped above
MAX_JACKKNIFE groups).

Usage
-----
    python -m bench.bootstrap                      # all sidecars -> bench/results/_bootstrap/summary.json
    python -m bench.bootstrap --check              # recomputed-from-units vs result JSON; exit 1 on mismatch
    python -m bench.bootstrap --task har_mhealth   # restrict
    python -m bench.bootstrap --pair har_hhar mae_5src__probe__s42 simmtm_5src__probe__s42
    python -m bench.bootstrap --per-group microgesture_openwatch_releasesplit mae_5src__probe__s42
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(REPO, "bench", "results")
UNITS_ROOT = os.path.join(RESULTS_ROOT, "_units")
BOOT_ROOT = os.path.join(RESULTS_ROOT, "_bootstrap")

N_BOOT = 1000
SEED = 0
ALPHA = 0.05
MIN_GROUPS = 10
MAX_JACKKNIFE = 5000        # BCa acceleration needs one metric evaluation per group
PURITY = 0.95               # a group is label-pure if >= 95 % of its units share one class
STRATIFIABLE = ("macro_f1", "bal_acc", "acc")
# Candidate group levels per task, overriding the sidecar's own list (coarse -> fine).
TASK_LEVELS = {
    # time split within each recording: every participant is in train AND val, so the
    # evaluation population is recordings / time, not new subjects
    "har_userwatch": ["session"],
}

# The ten IMU-Bench paper tasks and the main-table encoders (used for the default
# paired comparisons; the CI itself is computed for every sidecar found).
PAPER_TASKS = [
    "arm_pose_nymeria", "full_body_pose_nymeria", "odometry_nymeria_local", "microgesture_openwatch",
    "language_humanml3d", "pd_pads", "har_hhar", "har_mhealth", "har_prism", "har_userwatch",
]
PAPER_ENCODERS = ["mae_5src", "limubert_5src", "simmtm_5src", "focal_5src", "relcon_5src",
                  "mae_5src_noaltattn", "mae_5src_acconly"]
REFERENCE_ENCODER = "mae_5src"

# |metric_from_units - result JSON| allowed by --check. Classification / retrieval / ATE are
# recomputed from the very same predictions, so they must agree to float round-off; pose
# numbers are float32 means over millions of joint terms (arm) or a re-evaluation of a
# trainer-logged value (full body, batch-averaged, AMP) and get the tolerance from the brief.
TOLERANCE = {"pos_err_cm": 0.01, "mpjpe_mm": 0.05}
DEFAULT_TOL = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Metric kinds. Each takes the sidecar arrays and a per-unit weight vector
# (None = every unit once) and returns the metric in the sidecar's
# metric_scale (e.g. retrieval R@1 in %, pose in cm / mm).
# ─────────────────────────────────────────────────────────────────────────────

def _codes(a: Dict[str, np.ndarray]) -> Tuple[np.ndarray, int]:
    """Joint (y_true, y_pred) -> flat confusion-cell code over the union label set (cached)."""
    if "_cm_code" not in a:
        labels, inv = np.unique(np.concatenate([a["y_true"], a["y_pred"]]), return_inverse=True)
        n = len(a["y_true"])
        K = len(labels)
        a["_cm_code"] = (inv[:n] * K + inv[n:]).astype(np.int64)
        a["_cm_K"] = np.asarray(K)
    return a["_cm_code"], int(a["_cm_K"])


def _confusion(a, w) -> np.ndarray:
    code, K = _codes(a)
    cm = np.bincount(code, weights=w, minlength=K * K) if w is not None else \
        np.bincount(code, minlength=K * K).astype(np.float64)
    return cm.reshape(K, K)


def metric_macro_f1(a, w=None) -> float:
    """sklearn f1_score(average='macro', labels=None, zero_division=0): macro over the union of
    labels present (weight > 0) in y_true or y_pred; F1 = 2TP / (2TP + FP + FN)."""
    cm = _confusion(a, w)
    tp = np.diag(cm)
    n_true, n_pred = cm.sum(axis=1), cm.sum(axis=0)
    present = (n_true > 0) | (n_pred > 0)
    denom = n_true + n_pred
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1[present].mean()) if present.any() else 0.0


def metric_bal_acc(a, w=None) -> float:
    """sklearn balanced_accuracy_score: mean recall over classes present in y_true."""
    cm = _confusion(a, w)
    n_true = cm.sum(axis=1)
    keep = n_true > 0
    return float((np.diag(cm)[keep] / n_true[keep]).mean()) if keep.any() else 0.0


def metric_acc(a, w=None) -> float:
    c = (a["y_true"] == a["y_pred"]).astype(np.float64)
    return float(np.average(c, weights=w)) if w is not None else float(c.mean())


def metric_mean(a, w=None) -> float:
    v = a["value"].astype(np.float64)
    return float(np.average(v, weights=w)) if w is not None else float(v.mean())


def metric_r1_two_dirs(a, w=None) -> float:
    """Mean of IMU->text and text->IMU R@1 over the pooled G=100 gallery queries (fraction)."""
    hit = a["hit"].astype(np.float64)
    d = a["direction"]
    w = np.ones_like(hit) if w is None else w
    out = []
    for k in (0, 1):
        m = d == k
        out.append(np.sum(w[m] * hit[m]) / np.sum(w[m]))
    return float(0.5 * (out[0] + out[1]))


METRIC_KINDS: Dict[str, Callable] = {
    "macro_f1": metric_macro_f1,
    "bal_acc": metric_bal_acc,
    "acc": metric_acc,
    "mean": metric_mean,
    "r1_two_dirs": metric_r1_two_dirs,
}


# ─────────────────────────────────────────────────────────────────────────────
# Sidecar IO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Sidecar:
    path: str
    arrays: Dict[str, np.ndarray]
    meta: Dict[str, Any]

    @property
    def n_units(self) -> int:
        return int(len(self.arrays[self.meta.get("length_key", "unit_key")]))

    @property
    def group_levels(self) -> List[str]:
        lv = TASK_LEVELS.get(self.meta.get("task"), self.meta.get("group_levels") or [])
        return [g for g in lv if f"group__{g}" in self.arrays]

    def groups(self, level: str) -> np.ndarray:
        if level == "unit":
            return np.arange(self.n_units)
        return self.arrays[f"group__{level}"]

    def metric(self, w: Optional[np.ndarray] = None) -> float:
        fn = METRIC_KINDS[self.meta["metric_kind"]]
        return float(fn(self.arrays, w) * float(self.meta.get("metric_scale", 1.0)))

    @property
    def task(self) -> str:
        """The directory decides the task, not meta_json.

        A sidecar can be MOVED between task folders (the deep-decoder probes became the *_deepdec
        appendix tasks on 2026-09-19) without its meta being rewritten. Trusting meta then makes two
        sidecars claim the same cell and the later one silently wins, which is how the deep-decoder
        numbers once ended up as the confidence intervals of the light-decoder cells.
        """
        return os.path.basename(os.path.dirname(self.path)) if self.path else self.meta["task"]

    @property
    def cell(self) -> str:
        m = self.meta
        return f"{self.task}/{m['encoder']}__{m['protocol']}__s{m['seed']}"


def load_sidecar(path: str) -> Sidecar:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta_json"}
        meta = json.loads(str(z["meta_json"]))
    return Sidecar(path=path, arrays=arrays, meta=meta)


def units_root() -> str:
    return os.environ.get("BENCH_UNITS_DIR") or UNITS_ROOT


def sidecar_path(task: str, cell_id: str) -> str:
    return os.path.join(units_root(), task, cell_id + ".npz")


def iter_sidecars(tasks: Optional[Sequence[str]] = None, root: Optional[str] = None):
    for p in sorted(glob.glob(os.path.join(root or units_root(), "*", "*.npz"))):
        t = os.path.basename(os.path.dirname(p))
        if tasks and t not in tasks:
            continue
        try:
            yield load_sidecar(p)
        except Exception as e:     # keep the summary robust to one bad file
            print(f"[bootstrap] skipping {p}: {e!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────────────

def _group_index(g: np.ndarray) -> Tuple[np.ndarray, int]:
    uniq, inv = np.unique(g, return_inverse=True)
    return inv, len(uniq)


def _group_strata(sc: Sidecar, gidx: np.ndarray, G: int) -> Optional[np.ndarray]:
    """Per-group class stratum (majority y_true) if the metric is a classification metric and
    every group is label-pure, else None."""
    if sc.meta.get("metric_kind") not in STRATIFIABLE or "y_true" not in sc.arrays:
        return None
    _, yinv = np.unique(sc.arrays["y_true"], return_inverse=True)
    M = np.zeros((G, int(yinv.max()) + 1))
    np.add.at(M, (gidx, yinv), 1.0)
    purity = M.max(axis=1) / M.sum(axis=1)
    return M.argmax(axis=1) if bool(np.all(purity >= PURITY)) else None


@dataclass
class Plan:
    level: str
    gidx: np.ndarray            # per-unit group index
    n_groups: int
    strata: Optional[np.ndarray] = None   # per-group stratum, or None (unstratified)

    def describe(self) -> Dict[str, Any]:
        d = {"resample_level": self.level, "n_groups": int(self.n_groups), "stratified": self.strata is not None}
        if self.strata is not None:
            sizes = np.bincount(self.strata)
            sizes = sizes[sizes > 0]
            d["n_strata"] = int(len(sizes))
            d["n_singleton_strata"] = int(np.sum(sizes == 1))
        return d


def plan_resampling(sc: Sidecar, level: Optional[str] = None, min_groups: int = MIN_GROUPS) -> Plan:
    """See the module docstring. `level` forces a level (no fall-through checks)."""
    cands = [level] if level else sc.group_levels + ["unit"]
    for lvl in cands:
        gidx, G = _group_index(sc.groups(lvl))
        strata = _group_strata(sc, gidx, G)
        if level is None and lvl != "unit":
            if G < min_groups:
                continue
            if strata is not None:
                sizes = np.bincount(strata)
                sizes = sizes[sizes > 0]
                if np.mean(sizes >= 2) < 0.5:     # most classes have one group: nothing to resample
                    continue
        return Plan(lvl, gidx, G, strata)
    raise RuntimeError("unreachable")


def choose_level(sc: Sidecar, min_groups: int = MIN_GROUPS) -> str:
    return plan_resampling(sc, min_groups=min_groups).level


def iter_counts(n_groups: int, n_boot: int = N_BOOT, seed: int = SEED, strata: Optional[np.ndarray] = None):
    """Yield n_boot integer draw-count vectors [n_groups] (groups drawn with replacement, within
    strata if given). Identical stream for identical (n_groups, strata, seed) -> paired resampling."""
    rng = np.random.default_rng(seed)
    members = [np.arange(n_groups)] if strata is None else [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for _ in range(n_boot):
        c = np.zeros(n_groups, dtype=np.int64)
        for m in members:
            c += np.bincount(m[rng.integers(0, len(m), size=len(m))], minlength=n_groups)
        yield c


def draw_counts(n_groups: int, n_boot: int = N_BOOT, seed: int = SEED,
                strata: Optional[np.ndarray] = None) -> np.ndarray:
    """[n_boot, n_groups] matrix form of iter_counts (small problems / inspection only)."""
    return np.stack(list(iter_counts(n_groups, n_boot, seed, strata)))


def _ci(stats: np.ndarray, alpha: float = ALPHA) -> Tuple[float, float]:
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _bca(point: float, stats: np.ndarray, jack: Optional[np.ndarray], alpha: float = ALPHA):
    """BCa interval from bootstrap replicates + jackknife (leave-one-group-out) values."""
    from scipy.stats import norm
    if jack is None:
        return None, None
    p_less = np.mean(stats < point) + 0.5 * np.mean(stats == point)
    if p_less <= 0.0 or p_less >= 1.0:
        return None, None
    z0 = norm.ppf(p_less)
    d = jack.mean() - jack
    den = 6.0 * np.sum(d ** 2) ** 1.5
    acc = float(np.sum(d ** 3) / den) if den > 0 else 0.0
    z = norm.ppf([alpha / 2, 1 - alpha / 2])
    q = norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
    lo, hi = np.percentile(stats, 100 * q)
    return float(lo), float(hi)


def _jackknife(metric_fn, pl: Plan) -> Optional[np.ndarray]:
    if pl.n_groups > MAX_JACKKNIFE:
        return None
    return np.array([metric_fn((pl.gidx != g).astype(np.float64)) for g in range(pl.n_groups)])


def bootstrap_cell(sc: Sidecar, level: Optional[str] = None, n_boot: int = N_BOOT, seed: int = SEED,
                   alpha: float = ALPHA) -> Dict[str, Any]:
    pl = plan_resampling(sc, level)
    stats = np.array([sc.metric(c[pl.gidx].astype(np.float64))
                      for c in iter_counts(pl.n_groups, n_boot, seed, pl.strata)])
    lo, hi = _ci(stats, alpha)
    point = sc.metric()
    bca_lo, bca_hi = _bca(point, stats, _jackknife(sc.metric, pl), alpha)
    return {"point": point, "lo": lo, "hi": hi, "se": float(np.std(stats, ddof=1)),
            "boot_mean": float(stats.mean()), "bias": float(stats.mean() - point),
            "bca_lo": bca_lo, "bca_hi": bca_hi, "n_units": sc.n_units, **pl.describe()}


def same_units(a: Sidecar, b: Sidecar) -> Tuple[bool, str]:
    if a.task != b.task:
        return False, "different tasks"
    if a.n_units != b.n_units:
        return False, f"n_units {a.n_units} vs {b.n_units}"
    if "unit_key" in a.arrays and "unit_key" in b.arrays and not np.array_equal(a.arrays["unit_key"], b.arrays["unit_key"]):
        return False, "unit_key differs"
    for k in ("y_true", "direction"):
        if k in a.arrays and k in b.arrays and not np.array_equal(a.arrays[k], b.arrays[k]):
            return False, f"{k} differs"
    for lvl in set(a.group_levels) & set(b.group_levels):
        if not np.array_equal(a.groups(lvl), b.groups(lvl)):
            return False, f"group__{lvl} differs"
    return True, "ok"


def bootstrap_paired(a: Sidecar, b: Sidecar, level: Optional[str] = None, n_boot: int = N_BOOT,
                     seed: int = SEED, alpha: float = ALPHA) -> Dict[str, Any]:
    """CI of metric(a) - metric(b), resampling the SAME groups for both cells.
    Requires identical evaluation units (same task, same unit order/keys, same groups)."""
    ok, why = same_units(a, b)
    if not ok:
        raise ValueError(f"paired bootstrap needs identical units: {a.cell} vs {b.cell}: {why}")
    pl = plan_resampling(a, level)      # units (and so the plan) are identical for a and b
    diffs = np.empty(n_boot)
    for i, c in enumerate(iter_counts(pl.n_groups, n_boot, seed, pl.strata)):
        w = c[pl.gidx].astype(np.float64)
        diffs[i] = a.metric(w) - b.metric(w)
    lo, hi = _ci(diffs, alpha)
    point = a.metric() - b.metric()
    bca_lo, bca_hi = _bca(point, diffs, _jackknife(lambda w: a.metric(w) - b.metric(w), pl), alpha)
    return {"a": a.cell, "b": b.cell, "diff": point, "lo": lo, "hi": hi, "bca_lo": bca_lo, "bca_hi": bca_hi,
            "bias": float(diffs.mean() - point),
            "se": float(np.std(diffs, ddof=1)), "frac_a_gt_b": float(np.mean(diffs > 0)),
            "frac_a_lt_b": float(np.mean(diffs < 0)), "excludes_zero": bool(lo > 0 or hi < 0),
            "n_units": a.n_units, **pl.describe(), "higher_is_better": a.meta.get("higher_is_better")}


def per_group(sc: Sidecar, level: Optional[str] = None) -> List[Dict[str, Any]]:
    """Per-group metric (e.g. microgesture per-subject accuracy / macro-F1, PD n/a)."""
    lvls = sc.group_levels
    level = level or (lvls[0] if lvls else "unit")
    g = sc.groups(level)
    rows = []
    for gv in np.unique(g):
        w = (g == gv).astype(np.float64)
        row = {"group": str(gv), "n_units": int(w.sum()), sc.meta["primary_metric"]: sc.metric(w)}
        if "y_true" in sc.arrays:
            row["acc"] = metric_acc(sc.arrays, w)
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Cross-check against the result JSONs
# ─────────────────────────────────────────────────────────────────────────────

def result_json_path(sc: Sidecar) -> str:
    m = sc.meta
    return os.path.join(RESULTS_ROOT, m["task"], f"{m['encoder']}__{m['protocol']}__s{m['seed']}.json")


def reported_style_value(sc: Sidecar) -> float:
    """The primary metric aggregated the way the result JSON aggregated it. Usually identical
    to sc.metric(); for the trainer-logged full-body MPJPE the JSON value is the unweighted
    mean over val batches of per-batch means (imu_lm/runtime_consistency/trainer.py::_eval),
    so it is recomputed that way from the per-window values + their batch index."""
    if sc.meta.get("reported_aggregation") == "batch_mean" and "x__batch" in sc.arrays:
        v, bi = sc.arrays["value"].astype(np.float64), sc.arrays["x__batch"]
        per_batch = [v[bi == b].mean() for b in np.unique(bi)]
        return float(np.mean(per_batch) * float(sc.meta.get("metric_scale", 1.0)))
    return sc.metric()


def check_sidecar(sc: Sidecar, result_json: Optional[str] = None) -> Dict[str, Any]:
    """Recompute the primary metric from the units and compare it with the result JSON."""
    m = sc.meta
    pm = m["primary_metric"]
    tol = TOLERANCE.get(pm, DEFAULT_TOL)
    out: Dict[str, Any] = {"cell": sc.cell, "primary_metric": pm, "from_units": sc.metric(), "tol": tol}
    for name, v in (m.get("check_metrics") or {}).items():   # scorer-side values stored at save time
        out[name] = v
    p = result_json or result_json_path(sc)
    if not os.path.exists(p):
        out.update(status="no_result_json", json=None)
        return out
    js = json.load(open(p))
    stored = js.get("metrics", {}).get(pm)
    out["json"] = stored
    out["json_path"] = os.path.relpath(p, REPO)
    same_ckpt = (js.get("run") == m.get("run")) and (js.get("checkpoint") == m.get("checkpoint"))
    out["same_run_ckpt"] = bool(same_ckpt)
    if stored is None:
        out["status"] = "no_metric_in_json"
        return out
    compare = reported_style_value(sc)   # full body: batch-averaged like the trainer log
    out["compared_value"] = compare
    out["abs_diff"] = abs(float(compare) - float(stored))
    if not same_ckpt:
        out["status"] = "stale_sidecar"       # JSON was re-scored from another run / checkpoint
    else:
        out["status"] = "ok" if out["abs_diff"] <= tol else "MISMATCH"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def _default_pairs(scs: Dict[str, Sidecar]) -> List[Tuple[str, str]]:
    """Every encoder vs the reference encoder (mae_5src) within the same task and protocol,
    plus scratch vs mae_5src finetune."""
    pairs = []
    for key, sc in scs.items():
        m = sc.meta
        enc, proto = m["encoder"], m["protocol"]
        if enc == REFERENCE_ENCODER:
            continue
        ref_proto = "finetune" if proto == "scratch" else proto
        ref = f"{sc.task}/{REFERENCE_ENCODER}__{ref_proto}__s{m['seed']}"
        if ref in scs:
            pairs.append((ref, key))
    return pairs


def build_summary(tasks: Optional[Sequence[str]] = None, n_boot: int = N_BOOT, seed: int = SEED,
                  pairs: bool = True) -> Dict[str, Any]:
    scs = {sc.cell: sc for sc in iter_sidecars(tasks)}
    cells: Dict[str, Any] = {}
    for key in sorted(scs):
        sc = scs[key]
        try:
            main = bootstrap_cell(sc, n_boot=n_boot, seed=seed)
            levels = {}
            for lvl in sc.group_levels + ["unit"]:
                r = bootstrap_cell(sc, level=lvl, n_boot=n_boot, seed=seed)
                levels[lvl] = {k: r[k] for k in ("lo", "hi", "bca_lo", "bca_hi", "se", "bias", "n_groups", "stratified")}
            chk = check_sidecar(sc)
            cells[key] = {**main, "primary_metric": sc.meta["primary_metric"],
                          "higher_is_better": sc.meta.get("higher_is_better"),
                          "reported": chk.get("json"), "check": chk.get("status"),
                          "run": sc.meta.get("run"), "checkpoint": sc.meta.get("checkpoint"),
                          "group_levels": sc.group_levels, "levels": levels,
                          "notes": sc.meta.get("notes", "")}
        except Exception as e:
            cells[key] = {"error": repr(e)}
    paired: Dict[str, Any] = {}
    if pairs:
        for a, b in _default_pairs(scs):
            try:
                paired[f"{a} - {b}"] = bootstrap_paired(scs[a], scs[b], n_boot=n_boot, seed=seed)
            except Exception as e:
                paired[f"{a} - {b}"] = {"error": repr(e)}
    return {
        "meta": {"created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                 "n_boot": n_boot, "seed": seed, "alpha": ALPHA,
                 "ci": "percentile (lo/hi); BCa alongside (bca_lo/bca_hi, jackknife over resampled groups)",
                 "min_groups": MIN_GROUPS,
                 "resample_rule": "coarsest group level with >= min_groups groups (TASK_LEVELS overrides the "
                                  "candidates); classification metrics with label-pure groups resampled within "
                                  "class strata, skipping a level where most strata hold a single group; else the "
                                  "next finer level, finally units. Every level is also reported under `levels`",
                 "task_levels_override": TASK_LEVELS,
                 "paired_rule": f"metric(a) - metric(b), a = {REFERENCE_ENCODER} (same task/protocol; "
                                "finetune for scratch), same resampled groups for both"},
        "cells": cells,
        "paired": paired,
    }


def _fmt(v, nd=4):
    return "None" if v is None else f"{v:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", action="append", default=None, help="restrict to these tasks (repeatable)")
    ap.add_argument("--check", action="store_true", help="recomputed-vs-JSON check only; exit 1 on MISMATCH")
    ap.add_argument("--pair", nargs=3, metavar=("TASK", "CELL_A", "CELL_B"), help="paired CI of A - B")
    ap.add_argument("--per-group", nargs=2, metavar=("TASK", "CELL"), help="per-group metric table")
    ap.add_argument("--level", default=None, help="force a resample / grouping level")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=os.path.join(BOOT_ROOT, "summary.json"))
    ap.add_argument("--no-pairs", action="store_true")
    args = ap.parse_args(argv)

    if args.pair:
        t, ca, cb = args.pair
        r = bootstrap_paired(load_sidecar(sidecar_path(t, ca)), load_sidecar(sidecar_path(t, cb)),
                             level=args.level, n_boot=args.n_boot, seed=args.seed)
        print(json.dumps(r, indent=2))
        return 0
    if args.per_group:
        t, c = args.per_group
        sc = load_sidecar(sidecar_path(t, c))
        for row in per_group(sc, args.level):
            print(json.dumps(row))
        return 0
    if args.check:
        bad = 0
        for sc in iter_sidecars(args.task):
            r = check_sidecar(sc)
            bad += r["status"] == "MISMATCH"
            cv = r.get("compared_value")
            agg = "" if cv is None or cv == r["from_units"] else f" (as aggregated in the JSON: {_fmt(cv, 6)})"
            print(f"{r['status']:15s} {r['cell']:60s} {r['primary_metric']}: units={_fmt(r['from_units'], 6)}{agg} "
                  f"json={_fmt(r.get('json'), 6)} |d|={_fmt(r.get('abs_diff'), 6 if r['tol'] >= 0.01 else 12)} tol={r['tol']}")
        return 1 if bad else 0

    s = build_summary(args.task, n_boot=args.n_boot, seed=args.seed, pairs=not args.no_pairs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = f"{args.out}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=1)
    os.replace(tmp, args.out)
    for k, c in s["cells"].items():
        if "error" in c:
            print(f"{k:60s} ERROR {c['error']}")
            continue
        print(f"{k:60s} {c['primary_metric']}={_fmt(c['point'])} [{_fmt(c['lo'])}, {_fmt(c['hi'])}] "
              f"BCa [{_fmt(c.get('bca_lo'))}, {_fmt(c.get('bca_hi'))}] bias={_fmt(c.get('bias'))} "
              f"n={c['n_units']} groups={c['n_groups']} ({c['resample_level']}"
              f"{', class-stratified' if c.get('stratified') else ''}) check={c['check']}")
    print(f"[bootstrap] wrote {os.path.relpath(args.out, REPO)} ({len(s['cells'])} cells, {len(s['paired'])} pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
