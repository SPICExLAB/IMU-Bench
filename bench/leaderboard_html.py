#!/usr/bin/env python3
"""Render bench/results/ as a self-contained HTML page (bench/leaderboard.html).

    python -m bench.leaderboard_html [--out path]

Design: utilitarian benchmark board. Numbers in tabular mono, best cell per
column tinted, coverage grid as chips, per-task detail in collapsible blocks.
Both light and dark themes via CSS tokens.
"""
from __future__ import annotations

import argparse
import html
import os
import statistics
from typing import Any, Dict, List

from bench.leaderboard import Cell, collect, fmt_metric, hib, inject_aggregates, squeue_state
from bench.registry_io import REPO, load_encoders, load_tasks, run_name, task_ids
from bench.schema import git_sha, now_iso

OUT = os.path.join(REPO, "bench", "leaderboard.html")

CSS = """
:root{--bg:#F6F7F5;--panel:#FFFFFF;--ink:#171F1B;--muted:#5C6963;--rule:#D9DFDB;--accent:#1F6F8B;--accent-ink:#FFFFFF;
--best:#DCEFE6;--best-ink:#0F3F30;--legacy:#F1EBDD;--warn:#B8741A;--run:#2E7D5B;--pend:#8A6D1F;--miss:#E7EAE7;--chip:#EEF1EE;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0F1512;--panel:#161D19;--ink:#E5ECE7;--muted:#95A29B;--rule:#2A3430;--accent:#63B4CF;--accent-ink:#0B1A20;
--best:#1B3A31;--best-ink:#BFE8D6;--legacy:#2E2A1E;--warn:#E0A24C;--run:#5FC79A;--pend:#D7B45B;--miss:#1D2521;--chip:#202A25;}}
:root[data-theme="dark"]{--bg:#0F1512;--panel:#161D19;--ink:#E5ECE7;--muted:#95A29B;--rule:#2A3430;--accent:#63B4CF;--accent-ink:#0B1A20;
--best:#1B3A31;--best-ink:#BFE8D6;--legacy:#2E2A1E;--warn:#E0A24C;--run:#5FC79A;--pend:#D7B45B;--miss:#1D2521;--chip:#202A25;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Source Sans 3","Segoe UI",system-ui,-apple-system,Helvetica,Arial,sans-serif}
main{max-width:1240px;margin:0 auto;padding:32px 24px 80px}
h1{font-family:Charter,"Iowan Old Style","Palatino Linotype",Georgia,serif;font-weight:700;font-size:40px;letter-spacing:-.01em;margin:0 0 4px;text-wrap:balance}
h2{font-family:Charter,"Iowan Old Style",Georgia,serif;font-size:22px;margin:40px 0 10px;text-wrap:balance}
h3{font-size:16px;margin:24px 0 8px}
p{max-width:72ch}
.lede{color:var(--muted);margin:0 0 18px;max-width:80ch}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600}
.status{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 6px}
.stat{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:10px 14px;min-width:120px}
.stat b{display:block;font-size:22px;font-variant-numeric:tabular-nums;font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
.stat span{color:var(--muted);font-size:12px}
.wrap{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px 12px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap;vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12px;letter-spacing:.02em;position:sticky;top:0;background:var(--panel)}
th small{display:block;font-weight:400;text-transform:none;letter-spacing:0}
td:first-child,th:first-child{text-align:left}
td.num{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
td.best{background:var(--best);color:var(--best-ink);font-weight:700}
td.miss{color:var(--muted)}
tr.ext td{font-style:italic;color:var(--muted)}
tr.ref td{border-top:2px solid var(--rule)}
sup.leg{color:var(--warn);font-weight:700}
.rank{display:inline-block;min-width:22px;text-align:center;border-radius:4px;background:var(--chip);font-size:12px;padding:0 4px;margin-left:6px}
.chip{display:inline-block;border-radius:4px;padding:1px 7px;font-size:12px;font-family:ui-monospace,Menlo,Consolas,monospace}
.c-ok{background:var(--best);color:var(--best-ink)}.c-run{color:var(--run);border:1px solid var(--run)}.c-pend{color:var(--pend);border:1px solid var(--pend)}
.c-tr{color:var(--warn);border:1px solid var(--warn)}.c-miss{background:var(--miss);color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:14px 16px}
.card h3{margin:0 0 2px;font-family:Charter,Georgia,serif;font-size:18px}
.card .sub{color:var(--muted);font-size:13px;margin-bottom:8px}
.card dl{display:grid;grid-template-columns:96px 1fr;gap:3px 10px;margin:0;font-size:13px}
.card dt{color:var(--muted)}.card dd{margin:0}
code{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.92em;background:var(--chip);padding:0 4px;border-radius:3px}
details{margin:10px 0}summary{cursor:pointer;font-weight:600}
.legend{color:var(--muted);font-size:13px;margin:6px 0 0}
a{color:var(--accent)}
footer{color:var(--muted);font-size:12px;margin-top:40px}
"""


def esc(s: Any) -> str:
    return html.escape(str(s))


def main_table(tasks, encoders, cells, protocol, tids) -> str:
    enc_ids = [e for e, s in sorted(encoders.items(), key=lambda kv: kv[1].get("order", 999)) if not s.get("external") and not s.get("hide_in_summary")]   # ablation arms live in the coverage grid, not the headline tables
    ranks: Dict[str, Dict[str, int]] = {}
    for t in tids:
        task = tasks["tasks"][t]; pk = task["metrics"]["primary"]
        vals = []
        for e in enc_ids:
            if e == "scratch":          # reference row, never ranked
                continue
            c = cells.get((t, e, protocol))
            m = c.primary()[0] if c else None
            if m is not None:
                vals.append((e, m))
        vals.sort(key=lambda ev: ev[1], reverse=hib(task, pk))
        ranks[t] = {e: i + 1 for i, (e, _) in enumerate(vals)}
    head = "<tr><th>Encoder</th>" + "".join(
        f"<th>{esc(tasks['tasks'][t]['name'])}<small>{esc(tasks['tasks'][t]['metrics'][tasks['tasks'][t]['metrics']['primary']]['label'])} "
        f"{'↑' if hib(tasks['tasks'][t], tasks['tasks'][t]['metrics']['primary']) else '↓'}</small></th>" for t in tids) + "<th>Mean rank</th><th>Tasks</th></tr>"
    rows = []
    for e in enc_ids:
        proto = "scratch" if (e == "scratch" or encoders[e].get("run") is None) else protocol
        tds, rk, cov = [], [], 0
        for t in tids:
            task = tasks["tasks"][t]; pk = task["metrics"]["primary"]
            c = cells.get((t, e, proto))
            m, sd, n = c.primary() if c else (None, None, 0)
            if m is None:
                tds.append('<td class="num miss">·</td>'); continue
            cov += 1
            s = fmt_metric(task, pk, m) + (f" ± {fmt_metric(task, pk, sd)}" if sd is not None else "") + (f" <small>(n={n})</small>" if n > 1 else "")
            if any(x.provenance.get("legacy_run") for x in c.ok):
                s += '<sup class="leg" title="harvested from a recipe-equivalent pre-harness run">†</sup>'
            r_ = ranks[t].get(e)
            cls = "num best" if r_ == 1 else "num"
            rank_html = f'<span class="rank">#{r_}</span>' if r_ else ""
            tds.append(f'<td class="{cls}">{s}{rank_html}</td>')
            if r_ is not None:
                rk.append(r_)
        mr = statistics.fmean(rk) if rk else None
        is_ref = e == "scratch" or encoders[e].get("group") == "reference"
        name = (encoders[e]["name"] + " <small style='color:var(--muted)'>reference</small>") if is_ref else esc(encoders[e]["name"])
        rows.append((1e9 + 1 if is_ref else (mr if mr is not None else 1e9), 1 if is_ref else -cov,
                     f"<tr{' class=ref' if is_ref else ''}><td>{name}</td>{''.join(tds)}"
                     f"<td class='num'>{'ref' if is_ref else ('—' if mr is None else f'{mr:.2f}')}</td><td class='num'>{cov}/{len(tids)}</td></tr>"))
    rows.sort(key=lambda r: (r[1], r[0]))
    return f'<div class="wrap"><table>{head}{"".join(r[2] for r in rows)}</table></div>'


def coverage(tasks, encoders, cells, jobs, tids) -> str:
    g = tasks["grid"]
    head = "<tr><th>Encoder</th><th>Protocol</th>" + "".join(f"<th>{esc(tasks['tasks'][t]['name'])}</th>" for t in tids) + "</tr>"
    rows = []
    for e in g["encoders"] + [g["scratch_encoder"]]:
        for p in (g["protocols"] if e != g["scratch_encoder"] else ["scratch"]):
            tds = []
            for t in tids:
                c = cells.get((t, e, p))
                if c and c.ok:
                    tds.append(f'<td><span class="chip c-ok">scored{"×"+str(len(c.ok)) if len(c.ok)>1 else ""}</span></td>'); continue
                rn = run_name(t, e, p, 42)
                st = jobs.get(rn) or jobs.get(f"bscore_{t}__{e}__{p}")
                if st == "RUNNING":
                    tds.append('<td><span class="chip c-run">running</span></td>')
                elif st:
                    tds.append(f'<td><span class="chip c-pend">{esc(st.lower())}</span></td>')
                elif os.path.isdir(os.path.join(REPO, "runs", rn)):
                    tds.append('<td><span class="chip c-tr">trained · unscored</span></td>')
                else:
                    tds.append('<td><span class="chip c-miss">not launched</span></td>')
            rows.append(f"<tr><td>{esc(encoders[e]['short'])}</td><td>{p}</td>{''.join(tds)}</tr>")
    return f'<div class="wrap"><table>{head}{"".join(rows)}</table></div>'


def task_cards(tasks, tids) -> str:
    out = []
    for t in tids:
        k = tasks["tasks"][t]; i = k["input"]; sel = k["selection"]; pm = k["metrics"][k["metrics"]["primary"]]
        out.append(f"""<div class="card"><h3>{esc(k['name'])}</h3><div class="sub">{esc(k['subtitle'])}</div>
<dl><dt>Data / split</dt><dd>{esc(k['dataset']['name'])}<br><span style="color:var(--muted)">{esc(k['dataset'].get('split_rule',''))}</span></dd>
<dt>Input</dt><dd><code>[{i['channels']}, {i['T']}]</code> {i['window_seconds']} s @ {i['hz']} Hz, {esc(i['frame'])}, gravity {esc(i.get('gravity','included'))}</dd>
<dt>Output</dt><dd><code>{esc(k['output']['shape'])}</code> — {esc(k['output']['desc'])}</dd>
<dt>Head</dt><dd>{esc(k['output'].get('head',''))}</dd>
<dt>Selection</dt><dd>{esc(sel.get('monitor'))} ({esc(sel.get('mode','min'))}) on {esc(sel.get('split'))}</dd>
<dt>Reported</dt><dd><b>{esc(k['eval_split'])}</b> split</dd>
<dt>Primary</dt><dd><b>{esc(pm['label'])}</b> {'↑' if hib(k, k['metrics']['primary']) else '↓'}<br><span style="color:var(--muted)">{esc(pm.get('desc',''))}</span></dd>
<dt>Scorer</dt><dd><code>{esc(k['scorer'])}</code></dd></dl></div>""")
    return f'<div class="cards">{"".join(out)}</div>'


def detail(tasks, encoders, cells, t) -> str:
    task = tasks["tasks"][t]; mkeys = [k for k in task["metrics"] if k != "primary"]; pk = task["metrics"]["primary"]
    head = "<tr><th>Encoder</th><th>Protocol</th>" + "".join(f"<th>{'<b>' if k==pk else ''}{esc(task['metrics'][k].get('label',k))}{'</b>' if k==pk else ''}</th>" for k in mkeys) + "<th>seeds</th><th>ckpt (step)</th><th>run</th><th>src</th></tr>"
    rows = []
    for e, spec in sorted(encoders.items(), key=lambda kv: kv[1].get("order", 999)):
        for proto in ("probe", "finetune", "scratch", "legacy_valloss", "external"):
            c = cells.get((t, e, proto))
            if not c or not c.results:
                continue
            use = c.ok or c.results
            tds = []
            for k in mkeys:
                vals = [r.metrics[k] for r in use if r.metrics.get(k) is not None]
                tds.append(f'<td class="num">{fmt_metric(task, k, statistics.fmean(vals)) if vals else "—"}</td>')
            r0 = use[0]
            src = "bench" if r0.provenance.get("source", "").startswith("scorer") and not r0.provenance.get("legacy_run") else ("legacy†" if r0.provenance.get("legacy_run") else esc(r0.provenance.get("source", "")[:48]))
            st = "" if c.ok else f' <span class="chip c-tr">{esc(use[0].status)}</span>'
            cls = ' class="ext"' if spec.get("external") else ""
            rows.append(f"<tr{cls}><td>{esc(spec['name'])}{st}</td><td>{proto}</td>{''.join(tds)}<td class='num'>{','.join(str(r.seed) for r in use) if not spec.get('external') else '—'}</td>"
                        f"<td>{esc(r0.checkpoint or '')}{f' ({r0.checkpoint_step})' if r0.checkpoint_step else ''}</td><td><code>{esc(r0.run or '—')}</code></td><td>{src}</td></tr>")
    return f'<div class="wrap"><table>{head}{"".join(rows)}</table></div>'


def render(fragment: bool = False) -> str:
    tasks, encoders = load_tasks(), load_encoders()
    cells = collect(); jobs = squeue_state()
    tids, real_tids = inject_aggregates(tasks, cells)
    n_ok = sum(len(c.ok) for c in cells.values() if any(not r.encoder or True for r in c.ok))
    n_bench = sum(1 for c in cells.values() for r in c.ok if r.protocol != "external" and not r.provenance.get("legacy_run"))
    n_leg = sum(1 for c in cells.values() for r in c.ok if r.provenance.get("legacy_run"))
    n_run = sum(1 for k, v in jobs.items() if (k.startswith("bench_") or k.startswith("bscore_")) and v == "RUNNING")
    n_pend = sum(1 for k, v in jobs.items() if (k.startswith("bench_") or k.startswith("bscore_")) and v == "PENDING")
    grid_n = len(tasks["grid"]["encoders"]) * len(tasks["grid"]["protocols"]) * len(real_tids) + len(real_tids)
    filled = sum(1 for (t, e, p), c in cells.items() if p != "external" and c.ok)
    head = ("<title>IMU-Bench Board</title><style>" + CSS + "</style><main>") if fragment else (
        f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>IMU-Bench Board</title><style>{CSS}</style></head><body><main>")
    parts = [head,
             "<div class='eyebrow'>IMU-SSL · controlled multi-task benchmark</div><h1>IMU-Bench Board</h1>",
             f"<p class='lede'>Wrist-IMU encoders adapted to six tasks under one recipe per task, one scorer per task, one result schema. "
             f"Every cell is a JSON in <code>bench/results/</code>; nothing is typed by hand. Snapshot {esc(now_iso())} · git {esc(git_sha())}.</p>",
             "<div class='status'>",
             f"<div class='stat'><b>{filled}/{grid_n}</b><span>grid cells scored</span></div>",
             f"<div class='stat'><b>{n_bench}</b><span>harness-trained results</span></div>",
             f"<div class='stat'><b>{n_leg}</b><span>legacy† harvested</span></div>",
             f"<div class='stat'><b>{n_run}</b><span>jobs running</span></div>",
             f"<div class='stat'><b>{n_pend}</b><span>jobs queued</span></div></div>",
             "<h2>Frozen probe</h2><p class='legend'>Encoder frozen, task head trained. Best per column tinted; #n = rank within task; † = recipe-equivalent legacy run.</p>",
             main_table(tasks, encoders, cells, "probe", tids),
             "<h2>Full finetune</h2><p class='legend'>Encoder warm-started and trainable. Scratch row = same architecture from random init.</p>",
             main_table(tasks, encoders, cells, "finetune", tids),
             "<h2>Coverage of the controlled grid</h2>", coverage(tasks, encoders, cells, jobs, real_tids),
             "<h2>Task cards — unified input / output</h2>", task_cards(tasks, real_tids)]
    for t in [x for x in tids if tasks["tasks"][x].get("virtual")] + real_tids:
        k = tasks["tasks"][t]
        parts.append(f"<details><summary>{esc(k['name'])} — all metrics, all rows (<code>{t}</code>)</summary>{detail(tasks, encoders, cells, t)}</details>")
    parts.append("<h2>Caveats that travel with the numbers</h2><ul>"
                 "<li>Pose, odometry and PD select and report on <b>val</b> (no test split exists for the fixed Nymeria/PADS partitions in v1); microgesture and language report a genuine test split.</li>"
                 "<li>Full-body MPJPE is best-of-curve on val — noise-mined by ±1–2 mm; multi-seed cells are what to trust.</li>"
                 "<li>Language: fixed 8k-step budget, final checkpoint; seed sd ≈ 0.7 MRR.</li>"
                 "<li>PADS subject-level balanced accuracy is over 94 val subjects (14 HC / 58 PD / 22 DD).</li>"
                 "<li>External rows come from their own scripts and are context, not competitors under the encoder ranking.</li></ul>")
    parts.append("<footer>Regenerate: <code>python -m bench.leaderboard && python -m bench.leaderboard_html</code>. Registry: <code>bench/registry/*.yaml</code>.</footer></main>" + ("" if fragment else "</body></html>"))
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=OUT); ap.add_argument("--fragment", action="store_true"); a = ap.parse_args()
    s = render(fragment=a.fragment)
    with open(a.out, "w") as f:
        f.write(s)
    print(f"wrote {a.out} ({len(s)//1024} KB)")


if __name__ == "__main__":
    main()
