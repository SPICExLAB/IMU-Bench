#!/usr/bin/env python3
"""SmartFallMM right-wrist Meta sensor: pairing completeness, real sampling rates, acc/gyro time overlap, units.

    python analysis/smartfallmm/01_inspect_meta_wrist.py

Reads every accelerometer/gyroscope file of young/ and old/ meta_wrist, parses robustly (malformed cells are
coerced and dropped) and writes analysis/smartfallmm/meta_wrist_files.csv with one row per (group, trial).
"""
import os, re, sys
import numpy as np, pandas as pd

R = "/projects/b53017/vbc1591/SmartFallMM-Dataset"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta_wrist_files.csv")
PAT = re.compile(r"S(\d+)A(\d+)T(\d+)\.csv$")


def load(path):
    """Two layouts occur: 6 columns (epoch ms, local ISO time, elapsed s, x, y, z) and 4 columns (local time, x, y, z;
    the older group's gyroscope). Time is taken from the LOCAL time string in both, so the two layouts share a clock."""
    try:
        df = pd.read_csv(path, header=None, dtype=str, on_bad_lines="skip")
    except Exception:
        return None
    if df.shape[1] >= 6:
        tcol, xyz = 1, [3, 4, 5]
    elif df.shape[1] == 4:
        tcol, xyz = 0, [1, 2, 3]
    else:
        return None
    t = pd.to_datetime(df[tcol], errors="coerce")
    x = df[xyz].apply(pd.to_numeric, errors="coerce")
    ok = t.notna() & x.notna().all(axis=1)
    if ok.sum() < 5:
        return None
    t = t[ok].astype("int64").values / 1e9; x = x[ok].values.astype(np.float64)
    o = np.argsort(t, kind="stable")
    return t[o], x[o]


rows = []
for grp in ("young", "old"):
    A = {f for f in os.listdir(f"{R}/{grp}/accelerometer/meta_wrist") if PAT.search(f)}
    G = {f for f in os.listdir(f"{R}/{grp}/gyroscope/meta_wrist") if PAT.search(f)}
    for f in sorted(A | G):
        s, a, tr = map(int, PAT.search(f).groups())
        row = dict(group=grp, file=f, subject=s, activity=a, trial=tr, has_acc=f in A, has_gyro=f in G)
        ac = load(f"{R}/{grp}/accelerometer/meta_wrist/{f}") if f in A else None
        gy = load(f"{R}/{grp}/gyroscope/meta_wrist/{f}") if f in G else None
        for name, d in (("acc", ac), ("gyro", gy)):
            if d is None:
                continue
            t, x = d
            dur = t[-1] - t[0]; dt = np.diff(t)
            row.update({f"{name}_n": len(t), f"{name}_dur": dur, f"{name}_rate": (len(t) - 1) / dur if dur > 0 else np.nan,
                        f"{name}_dt_med_ms": 1000 * np.median(dt), f"{name}_gap_max_s": dt.max(),
                        f"{name}_t0": t[0], f"{name}_t1": t[-1]})
        if ac is not None:
            row["acc_norm_med"] = float(np.median(np.linalg.norm(ac[1], axis=1))); row["acc_abs_max"] = float(np.abs(ac[1]).max())
            row["acc_clip_frac"] = float((np.abs(ac[1]) >= 1.99).any(axis=1).mean())
        if gy is not None:
            row["gyro_abs_p99"] = float(np.percentile(np.abs(gy[1]), 99)); row["gyro_abs_max"] = float(np.abs(gy[1]).max())
            row["gyro_clip_frac"] = float((np.abs(gy[1]) >= 124.5).any(axis=1).mean())
        if ac is not None and gy is not None:
            row["overlap_s"] = max(0.0, min(ac[0][-1], gy[0][-1]) - max(ac[0][0], gy[0][0]))
            row["start_offset_s"] = gy[0][0] - ac[0][0]
        rows.append(row)
df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
print(f"wrote {OUT} ({len(df)} trials)")
pd.set_option("display.width", 220)
for grp, g in df.groupby("group"):
    both = g[g.has_acc & g.has_gyro & g.acc_n.notna() & g.gyro_n.notna()]
    print(f"\n=== {grp}: files acc={int(g.has_acc.sum())} gyro={int(g.has_gyro.sum())} paired+parsable={len(both)} | subjects={both.subject.nunique()} activities={sorted(both.activity.unique())}")
    q = lambda c: f"med {both[c].median():.1f} (p5 {both[c].quantile(.05):.1f}, p95 {both[c].quantile(.95):.1f})"
    print("  acc  rate Hz:", q("acc_rate"), "| dt med ms:", q("acc_dt_med_ms"), "| dur s:", q("acc_dur"))
    print("  gyro rate Hz:", q("gyro_rate"), "| dt med ms:", q("gyro_dt_med_ms"), "| dur s:", q("gyro_dur"))
    print("  overlap s:", q("overlap_s"), f"| zero overlap: {(both.overlap_s <= 0).mean() * 100:.1f}% | overlap<2.56s: {(both.overlap_s < 2.56).mean() * 100:.1f}% | |start offset| med {both.start_offset_s.abs().median():.1f}s")
    print("  units: |acc| median", round(both.acc_norm_med.median(), 3), "| acc abs max p99", round(both.acc_abs_max.quantile(.99), 2),
          "| gyro p99 median", round(both.gyro_abs_p99.median(), 1), "| gyro abs max p99", round(both.gyro_abs_max.quantile(.99), 1))
    print("  acc-rate clusters:", both.acc_rate.round(0).value_counts().head(5).to_dict(), "| gyro-rate clusters:", both.gyro_rate.round(0).value_counts().head(5).to_dict())

FALL = {10, 11, 12, 13, 14}
both = df[df.has_acc & df.has_gyro & df.acc_n.notna() & df.gyro_n.notna()].copy(); both["kind"] = np.where(both.activity.isin(FALL), "fall", "ADL")
print("\n=== clipping (share of samples with any axis at the sensor limit: acc >= 1.99 g, gyro >= 124.5 deg/s) ===")
print(both.groupby(["group", "kind"])[["acc_clip_frac", "gyro_clip_frac"]].agg(["median", "mean", lambda v: (v > 0).mean()]).round(3).to_string())
print("\nby activity (young):"); print(both[both.group == "young"].groupby("activity")[["acc_clip_frac", "gyro_clip_frac", "overlap_s"]].median().round(3).T.to_string())
ok = both[(both.overlap_s >= 2.56)]
print("\n=== usable (paired, overlap >= 2.56 s) ===")
for grp, g in ok.groupby("group"):
    per = g.groupby("subject").size()
    print(f"  {grp}: {len(g)} trials, {g.subject.nunique()} subjects (trials/subject min {per.min()} med {int(per.median())} max {per.max()}), fall trials {int((g.kind == 'fall').sum())}, ADL {int((g.kind == 'ADL').sum())}, total overlap {g.overlap_s.sum() / 3600:.2f} h")
    print("     50/50 Hz:", int(((g.acc_rate.round() == 50) & (g.gyro_rate.round() == 50)).sum()), "| other rates:", int(((g.acc_rate.round() != 50) | (g.gyro_rate.round() != 50)).sum()),
          "| subjects:", sorted(g.subject.unique()))
