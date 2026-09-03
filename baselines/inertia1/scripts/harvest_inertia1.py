"""Harvest every Inertia-1 result into one markdown report
(analysis/inertia1/inertia1_results.md), each task next to the reference rows
of Experiments_table.md that were produced with the identical protocol.

    python baselines/inertia1/scripts/harvest_inertia1.py

Missing pieces are printed as "pending" (jobs still running) -- re-run any time.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(REPO)
RUNS = os.path.join(REPO, "runs")
OUT = os.path.join(REPO, "analysis", "inertia1", "inertia1_results.md")
VARIANTS = [("patchtst", "Inertia-1 PatchTST-small (NHANES repro)"), ("art", "Inertia-1 ART-small (NHANES repro)")]


# ------------------------------------------------------------------ helpers
def _val_rows(run: str) -> List[Dict[str, float]]:
    p = os.path.join(RUNS, run, "logs", "metrics.txt")
    if not os.path.exists(p):
        return []
    rows = []
    for line in open(p):
        if "split=val" not in line:
            continue
        d = {}
        for tok in line.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
        rows.append(d)
    return rows


def best_val_row(run: str, monitor: str = "loss", mode: str = "min") -> Optional[Dict[str, float]]:
    rows = [r for r in _val_rows(run) if monitor in r]
    if not rows:
        return None
    return (min if mode == "min" else max)(rows, key=lambda r: r[monitor])


def run_status(run: str) -> str:
    d = os.path.join(RUNS, run)
    if not os.path.isdir(d):
        return "pending"
    if os.path.exists(os.path.join(d, "checkpoints", "best.pt")):
        # completion marker lives in the SLURM stdout (logs/<run>_<jobid>.out)
        for lg in sorted(glob.glob(os.path.join(REPO, "logs", run + "_*.out")), reverse=True):
            try:
                if "run complete" in open(lg, errors="ignore").read():
                    return "done"
            except OSError:
                pass
        return "running"
    return "started"


def fmt(x, nd=2, pct=False):
    if x is None:
        return "—"
    return f"{100*x:.{nd}f}%" if pct else f"{x:.{nd}f}"


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ sections
def sec_pretrain(md):
    md.append("## 0. Pretraining (our reproduction — no official Inertia-1 weights exist)\n")
    md.append("| Variant | run | steps | best val loss | status |\n|---|---|---|---|---|")
    for v, label in VARIANTS:
        run = f"inertia1_{v}_small_nhanes"
        p = os.path.join(RUNS, run, "logs", "metrics.txt")
        best, step = None, None
        if os.path.exists(p):
            for line in open(p):
                d = json.loads(line)
                if "val_loss" in d:
                    best, step = d["best_val"], d["step"]
                elif "step" in d:
                    step = d["step"]
        st = "done" if os.path.exists(os.path.join(RUNS, run, "checkpoints", "best.pt")) and step and step >= 170000 else ("running" if step else "pending")
        md.append(f"| {label} | `{run}` | {step or '—'} | {fmt(best, 5)} | {st} |")
    md.append("")


def sec_arm_pose(md):
    md.append("## 1. Arm pose (Nymeria real-wrist, 5 joints) — cf. Experiments_table.md §1\n")
    md.append("| Model | val_loss | pos_err (cm) | ang_err (deg) | status | run |\n|---|---|---|---|---|---|")
    ref = load_json("results_arm_pose_position_error.json") or {}
    for name in ["Ours (MAE)", "Ours (MAE) finetune", "FOCAL", "SimMTM", "Scratch"]:
        r = ref.get(name)
        if r:
            b = best_val_row(r["run"])
            md.append(f"| {name} | {fmt(b['loss'],5) if b else '—'} | {fmt(r['pos_err_cm'])} | {fmt(r['ang_err_deg'])} | ref | `{r['run']}` |")
    b = best_val_row("vit1d_arm_pose_probe_nymeria_mae_acconly_multidata")
    md.append(f"| Ours (MAE, acc-only) | {fmt(b['loss'],5) if b else '—'} | — | — | ref | `vit1d_arm_pose_probe_nymeria_mae_acconly_multidata` |")
    for v, label in VARIANTS:
        res = load_json(f"analysis/inertia1/arm_pose_position_error_{v}.json") or {}
        for proto in ["probe", "finetune"]:
            run = f"vit1d_arm_pose_{proto}_nymeria_inertia1_{v}"
            b = best_val_row(run)
            r = next((x for k, x in res.items() if x.get("run") == run), None)
            md.append(f"| **{label} {proto}** | {fmt(b['loss'],5) if b else '—'} | {fmt(r['pos_err_cm']) if r else '—'} | {fmt(r['ang_err_deg']) if r else '—'} | {run_status(run)} | `{run}` |")
    md.append("")


def sec_full_body(md):
    md.append("## 2. Full-body pose (Nymeria real-wrist, 23 joints) — cf. §2 (mpjpe = best val mpjpe_mm)\n")
    md.append("| Model | mpjpe (mm) | @step | status | run |\n|---|---|---|---|---|")
    for name, run in [("Ours (MAE)", "vit1d_full_body_pose_probe_nymeria_mae_multidata"),
                      ("Ours (MAE) finetune", "vit1d_full_body_pose_finetune_nymeria_mae_multidata"),
                      ("Ours (MAE, acc-only)", "vit1d_full_body_pose_probe_nymeria_mae_acconly_multidata"),
                      ("RelCon", "vit1d_full_body_pose_probe_nymeria_relcon_multidata"),
                      ("SimMTM", "vit1d_full_body_pose_probe_nymeria_simmtm_multidata"),
                      ("Scratch", "vit1d_full_body_pose_supervised_nymeria_altattn_scratch")]:
        b = best_val_row(run, "mpjpe_mm")
        md.append(f"| {name} | {fmt(b['mpjpe_mm']) if b else '—'} | {int(b['step']) if b else '—'} | ref | `{run}` |")
    for v, label in VARIANTS:
        for proto in ["probe", "finetune"]:
            run = f"vit1d_full_body_pose_{proto}_nymeria_inertia1_{v}"
            b = best_val_row(run, "mpjpe_mm")
            md.append(f"| **{label} {proto}** | {fmt(b['mpjpe_mm']) if b else '—'} | {int(b['step']) if b else '—'} | {run_status(run)} | `{run}` |")
    md.append("")


def _odom_csv(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    return rows[-1] if rows else None


def sec_odom(md):
    md.append("## 3. Odometry (Nymeria pelvis) — cf. §4 (local-velocity) and §4b (world-frame)\n")
    md.append("| Model | protocol | vel_mse | ATE (m) | drift % | status | source |\n|---|---|---|---|---|---|---|")
    refs = [("Ours (MAE)", "world-frame probe", "data_vis/nymeria_odom_worldframe_probe_multidata/odom_probe.csv"),
            ("Ours (MAE)", "local-vel probe", "data_vis/nymeria_odom_mae_probe_multidata/odom_probe.csv"),
            ("RelCon", "local-vel probe", "data_vis/nymeria_odom_relcon_probe_multidata/odom_probe.csv")]
    for name, proto, p in refs:
        r = _odom_csv(p)
        if r:
            md.append(f"| {name} | {proto} | {fmt(float(r['vel_mse']),5)} | {fmt(float(r['ate']),4)} | {fmt(float(r['drift']))} | ref | `{p}` |")
    md.append("| Ours (MAE) finetune | world-frame finetune | 0.01656 | 0.1293 | 34.08 | ref (§4b) | `vit1d_odom_worldframe_finetune_nymeria_mae_multidata` |")
    md.append("| TLIO / Air-IO | world-frame | 0.02419 / 0.02461 | 0.1510 / 0.1444 | 38.79 / 38.56 | ref (§4b) | `baselines/odometry/` |")
    for v, label in VARIANTS:
        for proto, p in [("world-frame probe", f"data_vis/nymeria_odom_worldframe_probe_inertia1_{v}/odom_probe.csv"),
                         ("local-vel probe", f"data_vis/nymeria_odom_inertia1_{v}_probe/odom_probe.csv"),
                         ("world-frame finetune", f"data_vis/nymeria_odom_worldframe_finetune_inertia1_{v}/odom_probe.csv")]:
            r = _odom_csv(p)
            if r is None:
                # finetune scorer may write a differently named csv
                alt = glob.glob(os.path.dirname(p) + "/*.csv")
                r = _odom_csv(alt[0]) if alt else None
            if r:
                md.append(f"| **{label}** | {proto} | {fmt(float(r['vel_mse']),5)} | {fmt(float(r['ate']),4)} | {fmt(float(r['drift']))} | done | `{p}` |")
            else:
                md.append(f"| **{label}** | {proto} | — | — | — | pending | `{p}` |")
    md.append("")


def sec_gesture(md):
    md.append("## 4. Microgesture (OpenWatch, paper protocol: 1 s windows, k=3 clip voting) — cf. analysis/openwatch_fair_comparison.md\n")
    md.append("| Model | condition | split | Window F1 | Window Acc | Clip F1 (k=3) | Clip Acc (k=3) |\n|---|---|---|---|---|---|---|")
    ref = load_json("results_openwatch_paperprotocol.json") or []
    for r in ref:
        cond = "aug" if r["name"].endswith("win/aug") else "noaug"
        nm = "Ours (MAE) probe" if "probe" in r["name"] else "Ours (MAE) finetune"
        md.append(f"| {nm} | {cond} | {r['split']} | {fmt(r['window']['macro_f1'],3)} | {fmt(r['window']['acc'],3)} | {fmt(r['clip']['macro_f1'],3)} | {fmt(r['clip']['acc'],3)} |")
    for v, label in VARIANTS:
        res = load_json(f"results_openwatch_inertia1_{v}.json")
        if not res:
            md.append(f"| **{label}** | — | — | pending | | | |")
            continue
        for r in res:
            cond = "aug" if r["name"].endswith("win/aug") else "noaug"
            proto = "probe" if "probe" in r["name"] else "finetune"
            md.append(f"| **{label} {proto}** | {cond} | {r['split']} | {fmt(r['window']['macro_f1'],3)} | {fmt(r['window']['acc'],3)} | {fmt(r['clip']['macro_f1'],3)} | {fmt(r['clip']['acc'],3)} |")
    md.append("")


def _ret_row(d, gallery: str):
    p = d["protocols"]["first"]
    if gallery == "full":
        g = p["full"]
    else:
        fx = p["fixed"]
        g = fx[str(gallery)] if isinstance(fx, dict) else next(x for x in fx if int(x["gallery"]) == int(gallery))
    a, b = g["imu2text"], g["text2imu"]
    return (f"{a['R@1']:.1f} / {a['R@5']:.1f} / {a['R@10']:.1f} / {a['MRR']:.1f}",
            f"{b['R@1']:.1f} / {b['R@5']:.1f} / {b['R@10']:.1f} / {b['MRR']:.1f}")


def sec_clip(md):
    md.append("## 5. Language alignment (HumanML3D CLIP retrieval, val, wrist-only) — cf. §13d\n")
    for gallery in ["100", "full"]:
        md.append(f"**Gallery = {gallery}** (R@1 / R@5 / R@10 / MRR, %)\n")
        md.append("| Model | IMU→Text | Text→IMU |\n|---|---|---|")
        for name, p in [("Ours (MAE) finetune, i2t", "analysis/retrieval_eval/humanml3d__ours_mae_finetune_i2t.json"),
                        ("Ours (MAE) frozen probe", "analysis/retrieval_eval/humanml3d__ours_mae_probe.json"),
                        ("Ours scratch, i2t", "analysis/retrieval_eval/humanml3d__ours_scratch_i2t.json"),
                        ("UniMTS official zero-shot (wrist)", "analysis/retrieval_eval/humanml3d__unimts_official_wrist.json")]:
            d = load_json(p)
            if d:
                try:
                    a, b = _ret_row(d, gallery)
                    md.append(f"| {name} | {a} | {b} |")
                except Exception as e:  # noqa
                    md.append(f"| {name} | ? ({e}) | |")
        for v, label in VARIANTS:
            for proto, p in [("frozen probe", f"analysis/retrieval_eval/inertia1/humanml3d__inertia1_{v}_probe.json"),
                             ("finetune, i2t", f"analysis/retrieval_eval/inertia1/humanml3d__inertia1_{v}_finetune_i2t.json")]:
                d = load_json(p)
                if d:
                    a, b = _ret_row(d, gallery)
                    md.append(f"| **{label} {proto}** | {a} | {b} |")
                else:
                    md.append(f"| **{label} {proto}** | pending | |")
        md.append("")


def sec_pads(md):
    md.append("## 6. Parkinson's disease (PADS) — cf. §14 (window-level @ best-val-loss ckpt; subject-level via eval_pads_subject_level.py)\n")
    md.append("| Model | task | acc | bal_acc | macro_f1 | bal_acc @best-bal_acc ckpt | subject bal_acc (mean-prob) | status | run |\n|---|---|---|---|---|---|---|---|---|")
    rows = [("Ours (MAE)", "pd", "vit1d_pd_probe_pads_mae"), ("Ours (MAE) finetune", "pd", "vit1d_pd_finetune_pads_mae"),
            ("SimMTM finetune", "pd", "vit1d_pd_finetune_pads_simmtm"), ("RelCon", "pd", "vit1d_pd_probe_pads_relcon"),
            ("Scratch", "pd", "vit1d_pd_supervised_pads_scratch"),
            ("Ours (MAE)", "pdhc", "vit1d_pdhc_probe_pads_mae"), ("Ours (MAE)", "pddd", "vit1d_pddd_probe_pads_mae")]
    for v, label in VARIANTS:
        rows += [(f"**{label} probe**", "pd", f"vit1d_pd_probe_pads_inertia1_{v}"), (f"**{label} finetune**", "pd", f"vit1d_pd_finetune_pads_inertia1_{v}"),
                 (f"**{label} probe**", "pdhc", f"vit1d_pdhc_probe_pads_inertia1_{v}"), (f"**{label} probe**", "pddd", f"vit1d_pddd_probe_pads_inertia1_{v}")]
    for name, task, run in rows:
        b = best_val_row(run, "loss")
        bb = best_val_row(run, "bal_acc", "max")
        subj = load_json(f"analysis/pads_subject_level/{run}.json")
        sb = subj["subject_mean_prob"]["bal_acc"] if subj else None
        st = "ref" if "inertia1" not in run else run_status(run)
        md.append(f"| {name} | {task} | {fmt(b['acc'],4,True) if b else '—'} | {fmt(b['bal_acc'],4,True) if b else '—'} | {fmt(b['macro_f1'],3) if b else '—'} | {fmt(bb['bal_acc'],4,True) if bb else '—'} | {fmt(sb,4,True)} | {st} | `{run}` |")
    md.append("")


def main():
    md = ["# Inertia-1 external baseline — results on our 6 tasks\n",
          "Inertia-1 (Xu et al. 2026, arXiv:2607.06617; github.com/yang-ai-lab/Inertia-1) ships **no pretrained weights** "
          "(GitHub issue #1 is an unanswered weights request), so both rows are OUR reproductions of their two robust objectives "
          "(paper §4.1/§4.3: PatchTST + AR-Transformer, small preset ≈8.5M params, 30 s @ 20 Hz triaxial accelerometer in g, "
          "no normalisation) pretrained with their recipe on NHANES (their pretraining corpus; raw archives on disk here). "
          "Downstream: identical data/split/window/decoder/optimiser to the sibling rows; only the encoder differs "
          "(acc-only input, raw g-scale, anti-aliased 50→20 Hz inside the adapter, tokens interpolated to the T/16 grid for "
          "pose/odometry decoders). See baselines/inertia1/ and imu_lm/models/baselines/inertia1_encoder.py.\n"]
    for f in (sec_pretrain, sec_arm_pose, sec_full_body, sec_odom, sec_gesture, sec_clip, sec_pads):
        try:
            f(md)
        except Exception as e:  # keep harvesting the rest
            md.append(f"\n_(section {f.__name__} failed: {e!r})_\n")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
