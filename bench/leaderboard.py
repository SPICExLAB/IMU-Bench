#!/usr/bin/env python3
"""Build LEADERBOARD.md (and bench/leaderboard.json) from bench/results/**.json.

    python -m bench.leaderboard            # writes LEADERBOARD.md + bench/leaderboard.json
    python -m bench.leaderboard --stdout   # print only

Rules
-----
* Only `status == "ok"` results are ranked; "partial"/"invalid" are listed
  greyed-out in the detail tables.
* A cell with several seeds is shown as mean ± sd (n).
* Ranking within a task uses the task's primary metric and direction.
* The overall column is the mean rank across the tasks an encoder covers
  under that protocol (n_tasks shown); an encoder missing a task gets no
  overall score for that protocol until it is filled.
* External specialist baselines are shown in the per-task tables in italics
  and never enter the encoder ranking.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from bench.schema import Result
import subprocess
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from bench.registry_io import REPO, grid_cells, load_encoders, load_tasks, run_name, task_ids
from bench.schema import RESULTS_ROOT, Result, iter_results

MD_PATH = os.path.join(REPO, "LEADERBOARD.md")
JSON_PATH = os.path.join(REPO, "bench", "leaderboard.json")


# ---------------------------------------------------------------------------
def squeue_state() -> Dict[str, str]:
    """job name -> state for the current user (empty if squeue unavailable)."""
    try:
        out = subprocess.check_output(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j|%T"],
                                      text=True, stderr=subprocess.DEVNULL, timeout=20)
    except Exception:
        return {}
    d: Dict[str, str] = {}
    for line in out.splitlines():
        if "|" in line:
            n, st = line.rsplit("|", 1)
            d[n.strip()] = st.strip()
    return d


def fmt_metric(task: Dict[str, Any], key: str, v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    f = task["metrics"].get(key, {}).get("fmt", "{:.3f}")
    try:
        return f.format(v)
    except Exception:
        return f"{v:.3f}"


def hib(task: Dict[str, Any], key: str) -> bool:
    return bool(task["metrics"].get(key, {}).get("higher_is_better", True))


class Cell:
    def __init__(self):
        self.results: List[Result] = []

    @property
    def ok(self) -> List[Result]:
        return [r for r in self.results if r.status == "ok"]

    def primary(self) -> Tuple[Optional[float], Optional[float], int]:
        vals = [r.metrics.get(r.primary_metric) for r in self.ok]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None, 0
        m = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else None
        return m, sd, len(vals)

    def metric(self, key: str) -> Optional[float]:
        vals = [r.metrics.get(key) for r in self.ok if r.metrics.get(key) is not None]
        return statistics.fmean(vals) if vals else None


def collect() -> Dict[Tuple[str, str, str], Cell]:
    cells: Dict[Tuple[str, str, str], Cell] = defaultdict(Cell)
    for r in iter_results():
        cells[(r.task, r.encoder, r.protocol)].results.append(r)
    return cells


def inject_aggregates(tasks, cells) -> Tuple[List[str], List[str]]:
    """Synthesise virtual aggregate tasks (tasks.yaml `aggregates:`) from their
    member tasks: one cell per (encoder, protocol) = mean of the members' primary
    metric, present only when EVERY member is scored. Members marked
    `aggregate_into` are hidden from the main tables. Returns (main_tids, real_tids)."""
    real_tids = task_ids(tasks)
    aggs = tasks.get("aggregates") or {}
    for aid, a in aggs.items():
        mk = a["metric"]["key"]
        tasks["tasks"][aid] = {
            "name": a["name"], "subtitle": a["subtitle"], "order": a.get("order", 999), "virtual": True,
            "family": "aggregate", "members": list(a["members"]),
            "eval_split": "val", "scorer": "bench.leaderboard.inject_aggregates",
            "selection": {"split": "val", "monitor": "per member task", "mode": "-"},
            "metrics": {"primary": mk, mk: {"higher_is_better": bool(a["metric"].get("higher_is_better", True)),
                                            "fmt": a["metric"].get("fmt", "{:.3f}"), "label": a["metric"]["label"],
                                            "desc": f"mean of the members' primary metric over {len(a['members'])} datasets: "
                                                    + ", ".join(a["members"])}},
            "input": {}, "output": {}, "dataset": {"name": "aggregate"},
        }
        pairs = {(e, p) for (t, e, p) in cells if t in a["members"]}
        for (e, p) in sorted(pairs):
            vals, legacy = [], False
            for m in a["members"]:
                c = cells.get((m, e, p))
                v = c.primary()[0] if c else None
                if v is None:
                    continue
                vals.append(v)
                legacy = legacy or any(x.provenance.get("legacy_run") for x in c.ok)
            if len(vals) < int(a.get("min_members", len(a["members"]))):
                continue
            r = Result(task=aid, encoder=e, protocol=p, seed=42,
                       metrics={mk: statistics.fmean(vals), "n_datasets": float(len(vals))},
                       primary_metric=mk, run=None, provenance={"source": "aggregate", "legacy_run": legacy},
                       status="ok", notes=f"mean over {len(vals)}/{len(a['members'])} member datasets")
            cells[(aid, e, p)].results.append(r)
    hidden = {t for t, k in tasks["tasks"].items()
              if k.get("aggregate_into") or k.get("hidden")}
    main = [t for t in task_ids(tasks) if t not in hidden]
    return main, real_tids


# ---------------------------------------------------------------------------
def rank_table(tasks, encoders, cells, protocol: str, tids: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    """Markdown rows for one protocol's main table + json payload."""
    enc_ids = [e for e, spec in sorted(encoders.items(), key=lambda kv: kv[1].get("order", 999))
               if not spec.get("external") and not spec.get("hide_in_summary")]   # ablation arms live in the coverage grid, not the headline tables
    # per-task ranks
    ranks: Dict[str, Dict[str, int]] = {}
    for t in tids:
        task = tasks["tasks"][t]
        pk = task["metrics"]["primary"]
        vals = []
        for e in enc_ids:
            if e == "scratch" or encoders[e].get("group") == "reference":   # reference rows, never ranked
                continue
            proto = protocol
            m, _, n = cells[(t, e, proto)].primary() if (t, e, proto) in cells else (None, None, 0)
            if m is not None:
                vals.append((e, m))
        vals.sort(key=lambda ev: ev[1], reverse=hib(task, pk))
        ranks[t] = {e: i + 1 for i, (e, _) in enumerate(vals)}
    header = "| Encoder | " + " | ".join(
        f"{tasks['tasks'][t]['name']}<br><sub>{tasks['tasks'][t]['metrics'][tasks['tasks'][t]['metrics']['primary']]['label']} {'↑' if hib(tasks['tasks'][t], tasks['tasks'][t]['metrics']['primary']) else '↓'}</sub>"
        for t in tids) + " | Mean rank | Tasks |"
    sep = "|---|" + "---:|" * len(tids) + "---:|---:|"
    rows = [header, sep]
    payload = {"protocol": protocol, "rows": []}
    body = []
    for e in enc_ids:
        # run-less encoders (scratch, arch-only controls) only ever have `scratch` cells
        proto = ("scratch" if (e == "scratch" or not (encoders[e].get("run") or encoders[e].get("weights")))
                 else protocol)   # external-weight baselines (harnet) have no run dir but do have probe/finetune cells
        row_cells, rk, covered = [], [], 0
        for t in tids:
            task = tasks["tasks"][t]
            c = cells.get((t, e, proto))
            m, sd, n = c.primary() if c else (None, None, 0)
            if m is None:
                row_cells.append("·")
                continue
            covered += 1
            s = fmt_metric(task, task["metrics"]["primary"], m)
            if sd is not None:
                s += f" ± {fmt_metric(task, task['metrics']['primary'], sd)}"
            if n > 1:
                s += f" <sub>(n={n})</sub>"
            r_ = ranks[t].get(e)
            if r_ == 1:
                s = f"**{s}**"
            if c and any(x.provenance.get("legacy_run") for x in c.ok):
                s += "<sup>†</sup>"
            row_cells.append(s)
            if r_ is not None:
                rk.append(r_)
        mean_rank = statistics.fmean(rk) if rk else None
        is_ref = e == "scratch" or encoders[e].get("group") == "reference"
        label = ("_" + encoders[e]["name"] + " — reference_") if is_ref else encoders[e]["name"]
        mr_s = "ref" if is_ref else (f"{mean_rank:.2f}" if mean_rank is not None else "—")
        body.append((1e9 + 1 if is_ref else (mean_rank if mean_rank is not None else 1e9), -1 if is_ref else covered, e,
                     f"| {label} | " + " | ".join(row_cells) + f" | {mr_s} | {covered}/{len(tids)} |"))
        payload["rows"].append({"encoder": e, "mean_rank": mean_rank, "covered": covered,
                                "cells": {t: cells[(t, e, proto)].primary()[0] if (t, e, proto) in cells else None for t in tids}})
    # sort: full coverage first, then by mean rank
    body.sort(key=lambda x: (-x[1], x[0]))
    rows += [b[3] for b in body]
    return rows, payload


def detail_table(tasks, encoders, cells, t: str, jobs: Dict[str, str]) -> List[str]:
    task = tasks["tasks"][t]
    mkeys = [k for k in task["metrics"] if k != "primary"]
    pk = task["metrics"]["primary"]
    lines = []
    lines.append(f"| Encoder | Protocol | " + " | ".join(
        ("**" if k == pk else "") + task["metrics"][k].get("label", k) + ("**" if k == pk else "") for k in mkeys)
        + " | seeds | ckpt (step) | run | src |")
    lines.append("|---|---|" + "---:|" * len(mkeys) + "---:|---|---|---|")
    enc_order = sorted(encoders.items(), key=lambda kv: kv[1].get("order", 999))
    for e, spec in enc_order:
        for proto in (["probe", "finetune", "scratch", "legacy_valloss", "external"]):
            c = cells.get((t, e, proto))
            if not c or not c.results:
                continue
            oks = c.ok
            use = oks if oks else c.results
            vals = " | ".join(fmt_metric(task, k, statistics.fmean([r.metrics[k] for r in use if r.metrics.get(k) is not None]))
                              if any(r.metrics.get(k) is not None for r in use) else "—" for k in mkeys)
            r0 = use[0]
            ck = f"{r0.checkpoint or ''}" + (f" ({r0.checkpoint_step})" if r0.checkpoint_step else "")
            src = "bench" if r0.provenance.get("source", "").startswith("scorer") and not r0.provenance.get("legacy_run") else (
                "legacy†" if r0.provenance.get("legacy_run") else r0.provenance.get("source", "")[:40])
            name = spec["name"]
            if spec.get("external"):
                name = f"*{name}*"
            status = "" if oks else f" ⚠ {use[0].status}"
            seeds = ",".join(str(r.seed) for r in use) if not spec.get("external") else "—"
            runcell = f"`{r0.run}`" if r0.run else "—"
            lines.append(f"| {name}{status} | {proto} | {vals} | {seeds} | {ck} | {runcell} | {src} |")
    return lines


def coverage_table(tasks, encoders, cells, jobs: Dict[str, str], tids: List[str]) -> List[str]:
    lines = ["| Encoder | Protocol | " + " | ".join(tasks["tasks"][t]["name"] for t in tids) + " |",
             "|---|---|" + "---|" * len(tids)]
    g = tasks["grid"]
    for e in g["encoders"] + [g["scratch_encoder"]]:
        protos = g["protocols"] if e != g["scratch_encoder"] else ["scratch"]
        for p in protos:
            row = []
            for t in tids:
                c = cells.get((t, e, p))
                if c and c.ok:
                    row.append("✅" + (f"×{len(c.ok)}" if len(c.ok) > 1 else ""))
                    continue
                rn = run_name(t, e, p, 42)
                st = jobs.get(rn) or jobs.get(f"bscore_{t}__{e}__{p}")
                if st:
                    row.append({"RUNNING": "🔄", "PENDING": "⏳"}.get(st, st))
                elif os.path.isdir(os.path.join(REPO, "runs", rn)):
                    row.append("🟡 trained, unscored")
                else:
                    row.append("⬜")
            lines.append(f"| {encoders[e]['short']} | {p} | " + " | ".join(row) + " |")
    return lines


def io_table(tasks, tids) -> List[str]:
    lines = ["| Task | Dataset / split | Input | Output | Selection | Reported split | Primary metric |",
             "|---|---|---|---|---|---|---|"]
    for t in tids:
        k = tasks["tasks"][t]
        i = k["input"]
        sel = k["selection"]
        lines.append(
            f"| **{k['name']}**<br><sub>{k['subtitle']}</sub> | {k['dataset']['name']}<br><sub>{k['dataset'].get('split_rule','')}</sub> | "
            f"`[{i['channels']}, {i['T']}]` @ {i['hz']} Hz, {i['window_seconds']} s, {i['frame']} | `{k['output']['shape']}`<br><sub>{k['output']['desc']}</sub> | "
            f"{sel.get('monitor')} ({sel.get('mode','min')}) on {sel.get('split')} | {k['eval_split']} | "
            f"{k['metrics'][k['metrics']['primary']]['label']} {'↑' if hib(k, k['metrics']['primary']) else '↓'} |")
    return lines


def build_markdown() -> Tuple[str, Dict[str, Any]]:
    tasks, encoders = load_tasks(), load_encoders()
    cells = collect()
    n_ok = sum(len(c.ok) for c in cells.values())
    tids, real_tids = inject_aggregates(tasks, cells)     # main-table columns / real tasks
    jobs = squeue_state()
    from bench.schema import now_iso, git_sha
    out: List[str] = []
    out.append("# IMU-Bench Leaderboard")
    out.append("")
    out.append(f"_Generated {now_iso()} from `bench/results/` ({n_ok} scored results, git {git_sha()}). "
               f"Regenerate with `python -m bench.leaderboard`. Every number links to a result JSON produced by a "
               f"`bench/score/*` scorer — nothing is hand-typed._")
    out.append("")
    out.append("**Controlled protocol (v1):** one encoder architecture (ViT1D, 21M), one recipe per task shared by every "
               "encoder (`bench/registry/tasks.yaml`), seed 42 (+43/44 where shown), checkpoint selected on val by the "
               "task's rule, scored by one scorer per task. `†` = harvested from a pre-harness run whose config is "
               "recipe-equivalent (same template/LR/selection); all other cells were trained by the harness. "
               "Bold = best per column. Lower-is-better metrics are marked ↓.")
    out.append("")
    payload = {"generated": now_iso(), "tables": {}}
    for proto in ("probe", "finetune"):
        out.append(f"## {'Frozen probe' if proto == 'probe' else 'Full finetune'} — primary metric per task")
        out.append("")
        rows, pl = rank_table(tasks, encoders, cells, proto, tids)
        out += rows
        out.append("")
        payload["tables"][proto] = pl
    dm = deco_md(tasks, encoders, cells)
    if dm:
        out.append("## Probe-decoder capacity — `mae_4src_z`")
        out.append("")
        out.append("Same frozen encoder, same data and recipe; only the task head changes, from the board's "
                   "1-layer default to ~12-layer decoders. **Not comparable to the frozen-probe table above** — "
                   "an ablation, excluded from the ranking until one decoder is promoted into the task cards for "
                   "every encoder. Bold = best across the `mae_4src_z` rows only.")
        out.append("")
        out += dm
        out.append("")
        out.append("> **Contamination caveat.** Every cell in the 3×3 grid above ran with a concurrent duplicate job writing into the same run directory for part of training (a dual-partition submission bug, fixed 2026-09-04). Single-job re-runs of the promoted arms reproduce full-body (91.64 → 91.45 mm) and odometry (0.1224 → 0.1228 ATE) but NOT arm pose: the promoted d12w decoder scores 8.14–8.16 cm when run alone, i.e. no better than the 1-layer default. **Treat the arm-pose row as noise, not a win.** The `mae_4src_z` numbers in the frozen-probe table are from the clean single-job re-runs.")
        out.append("")
    out.append("## Task cards — unified input / output")
    out.append("")
    out += io_table(tasks, real_tids)
    out.append("")
    out.append("## Coverage of the controlled grid")
    out.append("")
    out.append("✅ scored · 🔄 running · ⏳ queued · 🟡 trained but not yet scored · ⬜ not launched")
    out.append("")
    out += coverage_table(tasks, encoders, cells, jobs, real_tids)
    out.append("")
    for t in [x for x in tids if tasks["tasks"][x].get("virtual")] + real_tids:
        k = tasks["tasks"][t]
        out.append(f"## {k['name']} — `{t}`")
        out.append("")
        out.append(f"_{k['subtitle']}._ Scorer: `{k['scorer']}`. Selection: {k['selection']}. Reported split: **{k['eval_split']}**.")
        pm = k["metrics"][k["metrics"]["primary"]]
        if pm.get("desc"):
            out.append(f"Primary metric **{pm['label']}**: {pm['desc']}.")
        out.append("")
        out += detail_table(tasks, encoders, cells, t, jobs)
        out.append("")
    out.append("## Caveats that travel with the numbers")
    out.append("")
    out.append("* Pose / odometry / PD have **no held-out test split** in v1: the checkpoint is selected on val and val is "
               "reported (identical for every row, but optimistic in absolute terms). Microgesture and language report a "
               "genuine test split. A v2 split policy (carving test from the fixed val session lists) is the first "
               "protocol change to make before publishing absolute numbers.")
    out.append("* Full-body pose MPJPE is the trainer's own validation MPJPE at the selected checkpoint (min over the val "
               "curve under the bench selection rule) — a noise-mined statistic; multi-seed cells are what to trust.")
    out.append("* PADS: primary is **subject macro-F1** over 94 val subjects (14 HC / 58 PD / 22 DD); ±0.05 is noise. "
               "It replaced subject balanced accuracy on 2026-08-18 because bal-acc rewards degenerate classifiers on this "
               "split (a probe that never predicts PD scored 0.54 while classifying 31 % of subjects correctly); the "
               "`#classes predicted` column flags such cells. Every frozen probe on PADS is at/near chance — only the finetune "
               "column carries signal. Checkpoints are still selected on val window bal-acc (v1); selecting on macro-F1 is the "
               "next protocol change and needs re-runs.")
    out.append("* Language: fixed 8k-step budget, final checkpoint (`latest.pt`); §13e showed val-loss selection under-selects. "
               "Seed sd ≈ 0.7 MRR — single-seed differences < 1.5 pp are not real.")
    out.append("* External rows come from their own scripts (see `source` in the JSON) and are context, not competitors, "
               "under the encoder ranking.")
    out.append("* **HAR pretrain/eval overlap:** every ViT1D row pretrained on the `realdata` MAE source, which contains "
               "pamap2, recofit, utd_mhad and wisdm — 4 of the 6 HAR evaluation datasets (HHAR and MHEALTH are clean). "
               "External rows (Inertia-1, DeepConvLSTM) never saw them. Treat the HAR column as favourable to the ViT1D "
               "rows until the leakage-free re-pretrain (source rebuilt as `realdataclean` = samosa/shoaib/ut_watch, 2026-08-18) lands.")
    out.append("* HAR enters the main tables as ONE aggregate column (mean window macro-F1 over WISDM, UTD-MHAD, PAMAP2, "
               "RecoFit, HHAR, MHEALTH; present only when all six are scored). Selection is min val loss and val is reported "
               "(no test split); scores are re-evaluated from the checkpoint with pooled predictions, not read from training logs.")
    return "\n".join(out) + "\n", payload



DECO_TASKS = ["arm_pose_nymeria", "full_body_pose_nymeria", "odometry_nymeria_local"]
DECO_ARMS = [("d12a", "12L balanced"), ("d12b", "12L temporal-heavy"), ("d12w", "12L wide")]
DECO_REFS = [("scratch", "scratch"), ("mobileposer", "external"), ("tlio", "external"), ("airio", "external")]


def _small_decoder_baseline(tasks, cells, tids, pks):
    """The pre-promotion 1-layer-decoder probe number. Once a deep decoder is promoted the live
    cell holds the DEEP result, so the baseline is read from the archive promote wrote."""
    out = {}
    for t in tids:
        arch = os.path.join(REPO, "bench", "results", "_superseded_small_decoder",
                            f"{t}__mae_4src_z__probe__s42.json")
        if os.path.exists(arch):
            try:
                out[t] = json.load(open(arch)).get("metrics", {}).get(pks[t]); continue
            except Exception:
                pass
        c = cells.get((t, "mae_4src_z", "probe"))
        out[t] = c.primary()[0] if c else None
    return out


def deco_md(tasks, encoders, cells):
    """Probe-decoder capacity ablation on mae_4src_z — its own table, never merged into the
    ranking above: these cells use a deeper task head than every other number in this file."""
    import json as _json
    deco = {}
    for f in glob.glob(os.path.join(REPO, "bench", "generated", "results_deco", "*.json")):
        try:
            d = _json.load(open(f))
        except Exception:
            continue
        deco.setdefault(d["task"], {})[d["arm"]] = d.get("metrics", {})
    if not deco:
        return []
    tids = [t for t in DECO_TASKS if t in tasks["tasks"]]
    pks = {t: tasks["tasks"][t]["metrics"]["primary"] for t in tids}
    hdr = ["Probe decoder"] + [f"{tasks['tasks'][t]['name']} ({tasks['tasks'][t]['metrics'][pks[t]]['label']} {'↑' if hib(tasks['tasks'][t], pks[t]) else '↓'})" for t in tids]
    base = _small_decoder_baseline(tasks, cells, tids, pks)
    best = {}
    for t in tids:
        vals = [v for v in [base[t]] + [deco.get(t, {}).get(a, {}).get(pks[t]) for a, _ in DECO_ARMS] if v is not None]
        best[t] = (max if hib(tasks["tasks"][t], pks[t]) else min)(vals) if vals else None

    def cel(t, v, mark=True):
        if v is None:
            return "—"
        s_ = fmt_metric(tasks["tasks"][t], pks[t], v)
        return f"**{s_}**" if mark and best[t] is not None and abs(v - best[t]) < 1e-9 else s_

    rows = [["small decoder (board default, 1 layer)"] + [cel(t, base[t]) for t in tids]]
    for a, label in DECO_ARMS:
        rows.append([label] + [cel(t, deco.get(t, {}).get(a, {}).get(pks[t])) for t in tids])
    for eid, proto in DECO_REFS:
        if eid not in encoders:
            continue
        vals = [(cells[(t, eid, proto)].primary()[0] if (t, eid, proto) in cells else None) for t in tids]
        if any(v is not None for v in vals):
            rows.append([f"_{encoders[eid]['name']}_ (reference)"] + [cel(t, v, mark=False) for t, v in zip(tids, vals)])
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    md, payload = build_markdown()
    if args.stdout:
        print(md)
        return
    with open(MD_PATH, "w") as f:
        f.write(md)
    with open(JSON_PATH, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {MD_PATH} and {JSON_PATH}")


if __name__ == "__main__":
    main()
