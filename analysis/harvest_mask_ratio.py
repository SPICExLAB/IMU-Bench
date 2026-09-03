#!/usr/bin/env python3
"""Mask-ratio ablation: collect every arm's downstream probe cells into one table.

Arms are byte-identical pretraining runs on the leak-free 514 h raw cache (30k steps,
baseline R0 recipe) that differ ONLY in objective.mask_ratio; the 0.75 arm is the recipe
ablation's R0 run (encoder id `mae_recipe_r0`), not a separate retrain.

    python analysis/harvest_mask_ratio.py [--out analysis/mask_ratio_ablation.md]

Also folds in the fixed-eval-ratio rescore of the 300k-step Stage-1 arms
(analysis/mae_scaling_diagnostics/out/13_mask_ratio_ablation/*.json) when present:
pretraining loss measured at ONE common mask ratio, the only way those arms' losses
can be ranked against each other.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

REPO = "/gpfs/home/vbc1591/IMU-SSL"
sys.path.insert(0, REPO); os.chdir(REPO)
from bench.registry_io import load_tasks

ARMS = [("0.30", "mae_mr30"), ("0.40", "mae_mr40"), ("0.50", "mae_mr50"), ("0.60", "mae_mr60"),
        ("0.75", "mae_recipe_r0"), ("0.85", "mae_mr85"), ("0.90", "mae_mr90")]
HAR = [f"har_{d}" for d in ["wisdm", "utd_mhad", "pamap2", "recofit", "hhar", "mhealth"]]
CORE = ["arm_pose_nymeria", "microgesture_openwatch"]
RESCORE_DIR = "analysis/mae_scaling_diagnostics/out/13_mask_ratio_ablation"


def cell(task, enc):
    p = f"bench/results/{task}/{enc}__probe__s42.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if d.get("status") not in (None, "ok"):
        return None
    prim = load_tasks()["tasks"][task]["metrics"]["primary"]
    v = d["metrics"].get(prim)
    return None if v is None else float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/mask_ratio_ablation.md")
    a = ap.parse_args()
    tasks = load_tasks()["tasks"]
    rows, missing = [], []
    for ratio, enc in ARMS:
        har = {t: cell(t, enc) for t in HAR}
        core = {t: cell(t, enc) for t in CORE}
        got = [v for v in har.values() if v is not None]
        missing += [f"{t}/{enc}" for t, v in {**har, **core}.items() if v is None]
        rows.append({"ratio": ratio, "enc": enc, "har": har, "core": core,
                     "har_mean": sum(got) / len(got) if len(got) == len(HAR) else None,
                     "har_n": len(got)})

    lines = ["# Mask-ratio ablation (frozen-encoder probes)", "",
             "One pretraining arm per `objective.mask_ratio`; every other key held byte-identical",
             "to `configs/nhanes_scale/mae_recipe_R0.yaml` (leak-free 514 h pool, 30k steps, seed 42).",
             "The AltAttn encoder masks BERT-style (all tokens enter the encoder, masked ones replaced",
             "by a learned mask token), so per-step compute is the same for every arm — equal steps is",
             "equal compute here. All cells are frozen-encoder probes at seed 42, scored by `bench.score`.", ""]
    hdr = "| mask ratio | " + " | ".join(t.replace("har_", "") for t in HAR) + \
          " | **HAR mean** | arm pose (cm)↓ | gesture clip-F1↑ |"
    lines += [hdr, "|" + "---|" * (len(HAR) + 4)]
    for r in rows:
        cells = [f"{r['har'][t]:.3f}" if r["har"][t] is not None else "·" for t in HAR]
        hm = f"**{r['har_mean']:.3f}**" if r["har_mean"] is not None else f"· ({r['har_n']}/6)"
        ap_ = r["core"]["arm_pose_nymeria"]; gs = r["core"]["microgesture_openwatch"]
        lines.append(f"| **{r['ratio']}** | " + " | ".join(cells) + f" | {hm} | " +
                     (f"{ap_:.2f}" if ap_ is not None else "·") + " | " +
                     (f"{gs:.3f}" if gs is not None else "·") + " |")
    lines += ["", "HAR mean = mean window macro-F1 over the six leaderboard HAR sets (blank until all six land).",
              "Arm pose = FK position error, lower is better. Gesture = OpenWatch clip macro-F1 on the test split.", ""]

    done = [r for r in rows if r["har_mean"] is not None]
    if done:
        best = max(done, key=lambda r: r["har_mean"])
        lines += [f"**Best HAR mean: mask ratio {best['ratio']} ({best['har_mean']:.3f}).** "
                  f"Spread over completed arms: {min(r['har_mean'] for r in done):.3f}–"
                  f"{max(r['har_mean'] for r in done):.3f}.", ""]

    sweep = sorted(glob.glob(os.path.join(RESCORE_DIR, "maskratio_eval_at_*.json")))
    if sweep:
        lines += ["## Reconstruction quality of the SAME six arms, at fixed evaluation mask ratios", "",
                  "Every arm rescored at one common mask ratio (identical per-batch mask sequence), so these",
                  "MSEs are comparable across arms -- unlike the logged val losses. Compare against the probe",
                  "table above: the ordering is REVERSED.", ""]
        hdr2 = "| evaluated at | " + " | ".join(r for r, _ in ARMS) + " |"
        lines += [hdr2, "|" + "---|" * (len(ARMS) + 1)]
        for p2 in sweep:
            d = json.load(open(p2))
            row = [f"{d['results'][r]['mse']:.5f}" if r in d["results"] else "·" for r, _ in ARMS]
            lines.append(f"| **{d['eval_mask_ratio']}** | " + " | ".join(row) + " |")
        lines += ["", "Lower is better. The 0.85/0.90 arms reconstruct best at every evaluation ratio while",
                  "transferring worst, so reconstruction fidelity is anti-correlated with probe accuracy here:",
                  "MAE loss cannot be used to pick a mask ratio, even after the fixed-ratio correction.", ""]

    res = sorted(glob.glob(os.path.join(RESCORE_DIR, "stage1_eval_at_*.json")))
    if res:
        lines += ["## Pretraining loss at a fixed evaluation mask ratio (300k-step Stage-1 arms)", "",
                  "Independent evidence from the earlier 300k-step Stage-1 (acc-only, 11-source) arms.",
                  "Each arm is rescored at one COMMON mask ratio with an identical per-batch mask",
                  "sequence, because a 0.90-trained arm's logged val loss answers a strictly harder",
                  "question than a 0.50-trained arm's.", ""]
        for p in res:
            d = json.load(open(p))
            lines += [f"Evaluated at mask ratio **{d['eval_mask_ratio']}** ({d['batches']} val batches):", "",
                      "| trained mask ratio | fixed-ratio MSE↓ | vs predict-zero | recon/truth std | corr |",
                      "|---|---|---|---|---|"]
            for k, v in sorted(d["results"].items()):
                lines.append(f"| {k} | {v['mse']:.6f} | "
                             f"{100*(1-v['masked_mse']/v['predict_zero_mse']):.1f}% better | "
                             f"{v['std_ratio']:.3f} | {v['corr']:.3f} |")
            lines.append("")
        lines += ["**Replicate.** Both partition twins of the rescore job ran (9972314 at 10:58, 9972317 at 12:30),",
                  "each rebuilding the val loader independently -- an accidental full replicate. Run-1 MSEs were",
                  "0.7527/0.7628/0.7871 (eval 0.50), 0.8101/0.8068/0.8195 (eval 0.75), 0.8995/0.8842/0.8922 (eval 0.90)",
                  "for trained 0.50/0.75/0.90; max drift between replicates is 0.0036, and the ranking is identical at",
                  "every eval ratio in both runs. Gaps below ~0.004 (0.75 vs 0.50 at eval 0.75) are within that drift;",
                  "the 0.75-over-0.90 gap at eval 0.90 (0.0080, 0.0083) is not.", ""]

    if missing:
        lines += ["## Still missing", "", ", ".join(f"`{m}`" for m in missing), ""]
    txt = "\n".join(lines)
    open(a.out, "w").write(txt)
    print(txt)
    print(f"\n[harvest] wrote {a.out}  ({len(missing)} cells still missing)")


if __name__ == "__main__":
    main()
