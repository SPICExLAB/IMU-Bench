#!/usr/bin/env python3
"""IMU-Bench board, organised like the ICLR paper (bench/leaderboard_paper.html).

    python -m bench.leaderboard_paper [--out path]

Rows = encoders trained on the 5-source reference corpus (MAE + four SSL baselines + two ablations),
the from-scratch reference, and task-specific methods (own recipes, never ranked). Columns = the nine
IMU-Bench tasks, none of which is in the reference corpus (WISDM, UTD-MHAD, PAMAP2, RecoFit, SAMoSA and
UT-Watch were dropped from the benchmark on 2026-09-17). Every number comes from
bench/results/**.json; a missing cell shows its job state from squeue. The scaling figure is
<paper>/Pictures/volume_scaling_5src.png, copied next to the page.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import html
import json
import os
import re
import shutil
import statistics
import subprocess
from typing import Dict, Optional, Tuple

from bench.leaderboard import collect
from bench.registry_io import REPO, load_tasks

OUT = os.path.join(REPO, "bench", "leaderboard_paper.html")
FIG_SRC = "/gpfs/home/vbc1591/paper_iclr/Pictures/volume_scaling_5src.png"
STATS = os.path.join(REPO, "analysis", "bench_stats")

COLS = [  # task id, short label, metric label
    ("arm_pose_nymeria", "Arm pose", "pos. err. cm"),
    ("full_body_pose_nymeria", "Full-body pose", "MPJPE mm"),
    ("odometry_nymeria_local", "Odometry (wrist frame)", "ATE m"),
    ("microgesture_openwatch", "Microgesture", "clip F1"),
    ("language_humanml3d", "IMU–text", "R@1, G=100"),
    ("pd_pads", "Parkinson's", "subject F1"),
    ("har_hhar", "HHAR", "macro-F1"),
    ("har_mhealth", "MHEALTH", "macro-F1"),
    ("har_prism", "PRISM", "macro-F1"),
    ("har_userwatch", "TransfHAR", "macro-F1"),
]
# A task whose primary metric alone would misrepresent it gets a second column in the wide tables.
# Parkinson's subject-level aggregation is near-degenerate (most models label every participant with
# the majority class), so the window-level macro-F1 is reported beside the subject one.
# No extra columns. The PD subject metric was one until 2026-09-22; it is quantised (72 distinct
# values over 116 scored cells against 115 for the window metric) and carrying it in the main table
# invited readers to rank on it. It is still scored, and the per-task Parkinson's table and the
# appendix discussion report it; the main table does not.
EXTRA_COLS = {}
HAR3 = ["har_hhar", "har_mhealth", "har_prism", "har_userwatch"]
CONTROLLED = [  # encoder id, label, group, pretrain run
    ("mae_5src", "InrtlWrist (Ours)", "ssl", "vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic"),
    ("limubert_5src", "LIMU-BERT", "ssl", "vit1d_ssl_limubert_5src"),
    ("simmtm_5src", "SimMTM", "ssl", "vit1d_ssl_simmtm_5src"),
    ("focal_5src", "FOCAL", "ssl", "vit1d_ssl_focal_5src"),
    ("relcon_5src", "RelCon", "ssl", "vit1d_ssl_relcon_5src"),
    ("sslwear_5src", "SSL-Wearables", "ssl", "vit1d_ssl_sslwear_5src"),
    ("dino_5src", "DINO", "ssl", "vit1d_ssl_dino_5src"),
    ("jepa_5src", "I-JEPA", "ssl", "vit1d_ssl_jepa_5src"),
    ("mae_5src_noaltattn", "InrtlWrist, plain attention", "ablation", "vit1d_ssl_mae_varlen_noaltattn_5src"),
    ("mae_5src_acconly", "InrtlWrist, accelerometer only", "ablation", "vit1d_ssl_mae_varlen_acconly_5src"),
]
# A controlled row is a claim about one objective under a fixed protocol, so it enters the paper tables
# only once every task has been scored under both protocols; until then it would print a row that is
# mostly dashes and invite comparisons across different subsets of the benchmark (user decision
# 2026-09-23, on DINO at 9/20 cells and I-JEPA at 4/20). The gate clears itself: the row reappears on
# the next regeneration after its last cell lands, with no edit here.
from bench.schema import RESULTS_ROOT as _RESULTS_ROOT


def _complete(enc: str) -> bool:
    for task, _, _ in COLS:
        for proto in ("probe", "finetune"):
            f = os.path.join(_RESULTS_ROOT, task, f"{enc}__{proto}__s42.json")
            try:
                if json.load(open(f)).get("status") != "ok":
                    return False
            except (OSError, ValueError):
                return False
    return True


_CONTROLLED_ALL = CONTROLLED
CONTROLLED = [r for r in CONTROLLED if _complete(r[0])]

EXTERNAL = [("mobileposer", "MobilePoser"), ("tlio", "TLIO"), ("airio", "Air-IO"), ("mixtoken", "Mix-Token"),
            ("normwear", "NormWear"), ("unimts", "UniMTS, official checkpoint"), ("unimts_scratch", "UniMTS, from scratch"),
            ("deepconvlstm", "DeepConvLSTM")]
CAP_STEPS = 160000


def esc(s) -> str:
    return html.escape(str(s))


def squeue() -> Dict[str, Tuple[str, str]]:
    try:
        out = subprocess.check_output(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j|%T|%r"],
                                      text=True, timeout=60)
    except Exception:
        return {}
    d = {}
    for line in out.splitlines():
        j, st, r = (line.split("|") + ["", "", ""])[:3]
        if st == "RUNNING" or j not in d:
            d[j] = (st, r)
    return d


def val(cells, task, enc, proto) -> Optional[float]:
    c = cells.get((task, enc, proto))
    return c.primary()[0] if c else None


def fmt(tasks, task, v) -> str:
    m = tasks[task]["metrics"]
    return m.get(m["primary"], {}).get("fmt", "{:.3f}").format(v)


def hib(tasks, task) -> bool:
    m = tasks[task]["metrics"]
    return bool(m.get(m["primary"], {}).get("higher_is_better", True))


def state_chip(q, task, enc, proto) -> str:
    st = q.get(f"bench_{task}__{enc}__{proto}__s42")
    if not st:
        return '<td class="na" title="not part of this comparison">·</td>'
    s, r = st
    if s == "RUNNING":
        return '<td><span class="chip run">running</span></td>'
    if r.startswith("Dependency"):
        return '<td><span class="chip wait" title="starts when its pretraining run finishes">after pretrain</span></td>'
    return '<td><span class="chip queue">queued</span></td>'


def board(tasks, cells, q, proto) -> str:
    rows_ctrl = [(e, lab, grp) for e, lab, grp, _ in CONTROLLED]
    # best per column across controlled rows + scratch
    best = {}
    for t, _, _ in COLS:
        vals = [(val(cells, t, e, proto), e) for e, _, _ in rows_ctrl] + [(val(cells, t, "scratch", "scratch"), "scratch")]
        vals = [(v, e) for v, e in vals if v is not None]
        if vals:
            best[t] = (max if hib(tasks, t) else min)(vals)[1]
    head = "".join(f'<th scope="col">{esc(lab)}<small>{esc(unit)} {"↑" if hib(tasks, t) else "↓"}</small></th>'
                   for t, lab, unit in COLS)
    head += '<th scope="col">HAR mean<small>4 sets ↑</small></th>'
    out = [f'<table class="lb"><thead><tr><th scope="col" class="enc">Encoder</th>{head}</tr></thead><tbody>']

    def har_mean(enc, p):
        vs = [val(cells, t, enc, p) for t in HAR3]
        return statistics.fmean(vs) if all(v is not None for v in vs) else None

    group_label = {"ssl": "Self-supervised, same backbone, same corpus", "ablation": "Ablations of MAE"}
    prev = None
    for e, lab, grp in rows_ctrl:
        if grp != prev:
            out.append(f'<tr class="grp"><th colspan="{len(COLS) + 2}" scope="rowgroup">{group_label[grp]}</th></tr>')
            prev = grp
        tds = []
        for t, _, _ in COLS:
            v = val(cells, t, e, proto)
            if v is None:
                tds.append(state_chip(q, t, e, proto))
            else:
                cls = "num best" if best.get(t) == e else "num"
                tds.append(f'<td class="{cls}">{fmt(tasks, t, v)}</td>')
        hm = har_mean(e, proto)
        tds.append(f'<td class="num agg">{hm:.3f}</td>' if hm is not None else '<td class="na">·</td>')
        tag = ' <span class="tag">reference model</span>' if e == "mae_5src" else ""
        out.append(f'<tr><th scope="row" class="enc">{esc(lab)}{tag}</th>{"".join(tds)}</tr>')
    out.append(f'<tr class="grp"><th colspan="{len(COLS) + 2}" scope="rowgroup">No pretraining</th></tr>')
    tds = []
    for t, _, _ in COLS:
        v = val(cells, t, "scratch", "scratch")
        cls = "num best" if (v is not None and best.get(t) == "scratch") else "num"
        tds.append(f'<td class="{cls}">{fmt(tasks, t, v)}</td>' if v is not None else state_chip(q, t, "scratch", "scratch"))
    hm = har_mean("scratch", "scratch")
    tds.append(f'<td class="num agg">{hm:.3f}</td>' if hm is not None else '<td class="na">·</td>')
    out.append(f'<tr><th scope="row" class="enc">Trained from scratch</th>{"".join(tds)}</tr>')
    out.append(f'<tr class="grp"><th colspan="{len(COLS) + 2}" scope="rowgroup">Task-specific methods, own recipes, not ranked</th></tr>')
    for e, lab in EXTERNAL:
        vs = [val(cells, t, e, "external") for t, _, _ in COLS]
        if not any(v is not None for v in vs):
            continue
        tds = "".join(f'<td class="num ext">{fmt(tasks, t, v)}</td>' if v is not None else '<td class="na">·</td>'
                      for (t, _, _), v in zip(COLS, vs))
        out.append(f'<tr class="ext"><th scope="row" class="enc">{esc(lab)}</th>{tds}<td class="na">·</td></tr>')
    out.append("</tbody></table>")
    return "".join(out)


def pretrain_panel(q) -> str:
    items = []
    for e, lab, grp, run in CONTROLLED:
        if e == "mae_5src":
            continue
        m = os.path.join(REPO, "runs", run, "logs", "metrics.txt")
        step, vloss = 0, None
        if os.path.exists(m):
            for line in open(m):
                s = re.search(r"step=(\d+)", line)
                if s:
                    step = max(step, int(s.group(1)))
                if "split=val" in line:
                    v = re.search(r" loss=([0-9.]+)", line)
                    vloss = float(v.group(1)) if v else vloss
        st = q.get(run)
        if st and st[0] == "RUNNING":
            chip, word = "run", "training"
        elif st:
            chip, word = "queue", ("waiting for its motif-distance stage" if st[1].startswith("Dependency")
                                   else "waiting for a GPU")
        elif step:
            chip, word = "ok", "finished"
        else:
            chip, word = "queue", "not started"
        pct = min(100, step / CAP_STEPS * 100)
        vl = f"val. loss {vloss:.4f}" if vloss is not None else "no validation yet"
        items.append(f'<li><div class="pt-head"><b>{esc(lab)}</b><span class="chip {chip}">{word}</span></div>'
                     f'<div class="bar" role="img" aria-label="{step:,} of {CAP_STEPS:,} steps"><span style="width:{pct:.1f}%"></span></div>'
                     f'<div class="pt-meta"><span>{step:,} / {CAP_STEPS:,} steps (cap)</span><span>{vl}</span></div></li>')
    return '<ul class="pretrain">' + "".join(items) + "</ul>"


def task_table(tasks) -> str:
    har = json.load(open(os.path.join(STATS, "har_parquet_stats.json")))
    oth = json.load(open(os.path.join(STATS, "other_stats.json")))
    ny = oth["nymeria"]
    ny_h = ny["train"]["hours"] + ny["val"]["hours"]
    hm_h = sum(v["hours"] for v in oth["humanml3d"].values()) / 2
    pads = oth["pads"]["pads"] if "pads" in oth["pads"] else oth["pads"]
    ow = oth["openwatch"]

    def hs(k):
        for src in (har, oth.get("hhar", {}), oth.get("mhealth", {})):
            if k in src:
                return src[k]

    in_corpus = {"wisdm", "utd_mhad", "pamap2", "recofit", "samosa", "ut_watch"}
    rows = [("Arm pose", "Nymeria", ny["participants_total"], ny_h, "5-joint rotations", "session", False),
            ("Full-body pose", "Nymeria", None, None, "23-joint rotations", "session", False),
            ("Inertial odometry", "Nymeria", None, None, "pelvis velocity (wrist frame)", "session", False),
            ("Microgesture", "OpenWatch", ow["canonical_subjects"], ow["hours"], "6 classes / clip", "subject", False),
            ("IMU–text retrieval", "HumanML3D (simulated)", None, hm_h, "text embedding", "official", False),
            ("Parkinson's disease", "PADS", pads["subjects"], pads["hours"], "3 classes / subject", "subject", False)]
    names = {"wisdm": "WISDM", "utd_mhad": "UTD-MHAD", "pamap2": "PAMAP2", "recofit": "RecoFit", "samosa": "SAMoSA",
             "ut_watch": "UT-Watch", "hhar": "HHAR", "mhealth": "MHEALTH", "prism": "PRISM"}
    ncls = {"wisdm": 18, "utd_mhad": 21, "pamap2": 18, "recofit": 13, "samosa": 27, "ut_watch": 24, "hhar": 6,
            "mhealth": 12, "prism": 96}
    for k in ["hhar", "mhealth", "prism"]:  # the six sets bundled into common pretraining pools were dropped 2026-09-17
        s = hs(k)
        rows.append(("Activity recognition", names[k], s["subjects"], s["hours"], f"{ncls[k]} classes / window", "subject",
                     k in in_corpus))
    import pandas as _pd   # user-study watch (added 2026-09-18), split in time within each recording
    uw = _pd.read_parquet("/projects/b53017/vbc1591/userstudy_watch_dataset.parquet",
                          columns=["subject_id", "session_id", "timestamp_ns", "dataset_activity_id"])
    uw_h = float(uw.groupby("session_id").timestamp_ns.agg(lambda t: (t.max() - t.min()) / 3.6e12).sum())
    rows.append(("Activity recognition", "TransfHAR", int(uw.subject_id.nunique()), uw_h,
                 f"{uw.dataset_activity_id.nunique()} classes / window", "time", False))
    body = []
    for task, ds, subj, hrs, out, split, leak in rows:
        s_subj = "″" if (ds == "Nymeria" and subj is None) else ("–" if subj is None else f"{subj:,}")
        s_h = "″" if (ds == "Nymeria" and hrs is None) else (f"{hrs:,.1f}" if hrs >= 1 else f"{hrs:.2f}")
        note = '<span class="chip leak" title="inside the reference pretraining corpus">in corpus</span>' if leak else ""
        body.append(f'<tr><th scope="row">{esc(task)}</th><td>{esc(ds)} {note}</td><td class="num">{s_subj}</td>'
                    f'<td class="num">{s_h}</td><td>{esc(out)}</td><td>{split}</td></tr>')
    return ('<table class="plain"><thead><tr><th scope="col">Task</th><th scope="col">Dataset</th><th scope="col">Subjects</th>'
            '<th scope="col">Hours</th><th scope="col">Output</th><th scope="col">Split</th></tr></thead><tbody>'
            + "".join(body) + "</tbody></table>")


def corpus_table() -> str:
    import numpy as np
    v1 = "/projects/b53017/vbc1591/mae_multidata_cache"
    man = json.load(open("/projects/b53017/vbc1591/diversity_saturation/pool_manifest.json"))
    cum = [0] + man["cumulative_sizes"]
    v3 = {s: cum[i + 1] - cum[i] for i, s in enumerate(man["sources"])}
    src = [("realdata", "Seven activity-recognition sets", "real"), ("ocdetect", "OCDetect", "real"),
           ("eatingspeed", "EatingSpeed", "real"), ("motionx", "Motion-X", "synthetic"), ("sonic", "SONIC", "synthetic")]
    rows, tot, tot3 = [], 0, 0
    for k, name, kind in src:
        n = int(np.load(os.path.join(v1, f"{k}_train_x.npy"), mmap_mode="r").shape[0])
        tot += n; tot3 += v3[k]
        rows.append(f'<tr><th scope="row">{esc(name)}</th><td>{kind}</td><td class="num">{n:,}</td>'
                    f'<td class="num">{n * 5.12 / 3600:,.1f}</td><td class="num">{v3[k]:,}</td></tr>')
    rows.append(f'<tr class="total"><th scope="row">Total</th><td></td><td class="num">{tot:,}</td>'
                f'<td class="num">{tot * 5.12 / 3600:,.1f}</td><td class="num">{tot3:,}</td></tr>')
    return ('<table class="plain"><thead><tr><th scope="col">Source</th><th scope="col">Type</th><th scope="col">Windows</th>'
            '<th scope="col">Hours</th><th scope="col">Scaling pool</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


CSS = """
:root{--ground:#F3F6F5;--panel:#FFFFFF;--ink:#14201D;--ink2:#3E4B47;--muted:#63716C;--rule:#D5DEDA;--rule2:#E6ECE9;
--accent:#1C6B84;--accent-soft:#E3F0F4;--best:#DCEFE6;--best-ink:#0E3E2F;--warn:#9A5E0C;--warn-soft:#F7EBD9;
--run:#1E7A55;--chip:#EDF2F0;--bar:#1C6B84;--bar-bg:#E3EAE7;
--display:"Newsreader",Charter,"Iowan Old Style",Georgia,serif;--sans:"Source Sans 3","Segoe UI",system-ui,-apple-system,Helvetica,Arial,sans-serif;
--mono:"JetBrains Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--ground:#0E1413;--panel:#151C1B;--ink:#E3EBE8;--ink2:#BAC7C2;--muted:#8F9E99;
--rule:#27322F;--rule2:#1E2826;--accent:#63B7CF;--accent-soft:#16303A;--best:#173A2F;--best-ink:#BDE8D5;--warn:#E3A857;--warn-soft:#33281A;
--run:#6CCB9C;--chip:#1D2725;--bar:#63B7CF;--bar-bg:#24302D}}
:root[data-theme="dark"]{--ground:#0E1413;--panel:#151C1B;--ink:#E3EBE8;--ink2:#BAC7C2;--muted:#8F9E99;--rule:#27322F;--rule2:#1E2826;
--accent:#63B7CF;--accent-soft:#16303A;--best:#173A2F;--best-ink:#BDE8D5;--warn:#E3A857;--warn-soft:#33281A;--run:#6CCB9C;--chip:#1D2725;
--bar:#63B7CF;--bar-bg:#24302D}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font:16px/1.55 var(--sans)}
.page{max-width:1180px;margin:0 auto;padding-inline:24px;padding-block:36px 72px}
header.top{display:grid;gap:10px;padding-bottom:22px;border-bottom:1px solid var(--rule)}
.eyebrow{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:600}
h1{font-family:var(--display);font-weight:600;font-size:clamp(34px,5.5vw,50px);line-height:1.05;margin:0;letter-spacing:-.01em}
h2{font-family:var(--display);font-weight:600;font-size:26px;margin:0;text-wrap:balance}
.lede{margin:0;max-width:70ch;color:var(--ink2);font-size:17px}
.stats{display:flex;flex-wrap:wrap;gap:8px 22px;margin-top:6px;font-size:14px;color:var(--muted)}
.stats b{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--ink);font-weight:600;margin-right:4px}
section{display:grid;gap:12px;margin-top:44px}
section>p,.note{margin:0;max-width:72ch;color:var(--ink2)}
.note{font-size:14px;color:var(--muted)}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--rule)}
.tabs button{font:600 14px var(--sans);color:var(--muted);background:none;border:0;border-bottom:2px solid transparent;padding:8px 12px;cursor:pointer;margin-bottom:-1px}
.tabs button[aria-selected="true"]{color:var(--accent);border-bottom-color:var(--accent)}
.tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.scroll{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px 12px;border-bottom:1px solid var(--rule2);text-align:left;white-space:nowrap;vertical-align:middle}
thead th{font-size:12.5px;font-weight:600;color:var(--muted);background:var(--panel);border-bottom:1px solid var(--rule)}
thead th small{display:block;font-weight:400;font-size:11.5px;letter-spacing:0}
table.lb thead th:not(.enc){text-align:right}
th.enc{position:sticky;left:0;background:var(--panel);z-index:1;font-weight:600;color:var(--ink)}
tbody th[scope=row]{font-weight:600}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;font-size:13.5px}
td.best{background:var(--best);color:var(--best-ink);font-weight:600}
td.agg{color:var(--ink2)}
td.na{color:var(--muted);text-align:right}
table.lb td:not(.num){text-align:right}
tr.grp th{background:var(--ground);color:var(--muted);font-size:12px;letter-spacing:.06em;text-transform:uppercase;font-weight:600;position:static}
tr.ext th,tr.ext td{color:var(--muted);font-style:italic}
tr.total th,tr.total td{border-top:1px solid var(--rule);font-weight:600}
.tag{font:600 11px var(--sans);color:var(--accent);background:var(--accent-soft);border-radius:3px;padding:1px 6px;margin-left:6px;vertical-align:1px}
.chip{display:inline-block;font:600 11.5px var(--sans);border-radius:3px;padding:1px 7px;white-space:nowrap}
.chip.run{color:var(--run);border:1px solid var(--run)}
.chip.queue{color:var(--muted);border:1px solid var(--rule)}
.chip.wait{color:var(--warn);border:1px dashed var(--warn)}
.chip.ok{background:var(--best);color:var(--best-ink)}
.chip.leak{background:var(--warn-soft);color:var(--warn);margin-left:4px}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:13px;color:var(--muted);margin:0}
.legend span{display:inline-flex;gap:6px;align-items:center}
.sw{display:inline-block;width:14px;height:14px;border-radius:3px;background:var(--best)}
ul.pretrain{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
ul.pretrain li{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:12px 14px;display:grid;gap:8px}
.pt-head{display:flex;justify-content:space-between;align-items:center;gap:8px}
.bar{height:6px;border-radius:3px;background:var(--bar-bg);overflow:hidden}
.bar span{display:block;height:100%;background:var(--bar)}
.pt-meta{display:flex;justify-content:space-between;gap:8px;font:12.5px var(--mono);color:var(--muted);flex-wrap:wrap}
.stack{display:grid;gap:32px}
.stack>div{display:grid;gap:10px;min-width:0}
.narrow{max-width:720px}
h3{font-family:var(--sans);font-size:15px;margin:0;font-weight:700}
ol.findings{margin:0;padding-left:1.2em;display:grid;gap:10px;max-width:78ch;color:var(--ink2)}
ol.findings b{color:var(--ink)}
figure{margin:0;background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:14px}
figure img{display:block;width:100%;height:auto}
figcaption{font-size:13px;color:var(--muted);margin-top:8px;max-width:90ch}
.dec td:first-child{font-family:var(--mono);font-size:13px}
code{font-family:var(--mono);font-size:.88em;background:var(--chip);padding:1px 5px;border-radius:3px}
footer{margin-top:48px;padding-top:16px;border-top:1px solid var(--rule);font-size:13px;color:var(--muted)}
@media (max-width:520px){.page{padding-inline:16px}}
"""


def render() -> str:
    tasks = load_tasks()["tasks"]
    cells = collect()
    q = squeue()
    n_ctrl = n_done = 0
    for t, _, _ in COLS:
        for e, _, _, _ in CONTROLLED:
            for p in ("probe", "finetune"):
                if val(cells, t, e, p) is not None or f"bench_{t}__{e}__{p}__s42" in q:
                    n_ctrl += 1
                    n_done += val(cells, t, e, p) is not None
    n_run = sum(1 for j, (s, _) in q.items() if s == "RUNNING")
    n_q = sum(1 for j, (s, _) in q.items() if s != "RUNNING")
    now = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    probe, ft = board(tasks, cells, q, "probe"), board(tasks, cells, q, "finetune")
    body = f"""<title>IMU-Bench</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Source+Sans+3:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="page">
<header class="top">
  <div class="eyebrow">Benchmark · wrist-worn IMU representations</div>
  <h1>IMU-Bench</h1>
  <p class="lede">Ten tasks from eight datasets in six families — arm and full-body pose, inertial odometry, microgesture, IMU–text retrieval, Parkinson's disease and activity recognition — scored under frozen probing, fine-tuning and training from scratch with one backbone, one decoder per task and fixed splits.</p>
  <div class="stats"><span><b>{n_done}/{n_ctrl}</b>controlled cells scored</span><span><b>{n_run}</b>jobs running</span><span><b>{n_q}</b>queued</span><span>updated {now}</span></div>
</header>

<section aria-labelledby="lb-h">
  <h2 id="lb-h">Leaderboard</h2>
  <p>Every encoder below was pretrained on the same five-source reference corpus (197,094 windows, about 280 h), which contains none of the evaluation sets. Activity recognition is scored on HHAR, MHEALTH, PRISM and our user-study smartwatch set (60 fine-grained activities, split in time within each recording).</p>
  <div class="tabs" role="tablist" aria-label="Protocol">
    <button id="tab-probe" role="tab" aria-selected="true" aria-controls="pane-probe">Frozen probe</button>
    <button id="tab-ft" role="tab" aria-selected="false" aria-controls="pane-ft">Fine-tune</button>
  </div>
  <div id="pane-probe" role="tabpanel" aria-labelledby="tab-probe"><div class="scroll">{probe}</div></div>
  <div id="pane-ft" role="tabpanel" aria-labelledby="tab-ft" hidden><div class="scroll">{ft}</div></div>
  <p class="legend"><span><i class="sw"></i>best controlled result per task (scratch included)</span><span><span class="chip run">running</span><span class="chip queue">queued</span><span class="chip wait">after pretrain</span> job state of a missing cell</span><span>· not part of this comparison</span></p>
  <p class="note">Frozen probes on the pose and odometry tasks use deep temporal decoders (4 cross-channel + 8 temporal layers for pose, 12 layers for odometry); MAE's earlier shallow-decoder probes on those three tasks were withdrawn and are being re-run. Task-specific methods use their own recipes and inputs and are shown for orientation only.</p>
</section>

<section aria-labelledby="pt-h">
  <h2 id="pt-h">Baselines in training</h2>
  <p>Each baseline uses the configuration of its published row with the same five sources and the same cached windows as MAE, early stopping with patience 10, capped at 160,000 steps. MAE itself stopped at 154,000.</p>
  {pretrain_panel(q)}
</section>

<section aria-labelledby="tk-h">
  <div class="stack">
    <div><h2 id="tk-h">Tasks</h2><div class="scroll">{task_table(tasks)}</div>
      <p class="note">Subjects and hours are measured from the data, train and evaluation splits together. Nymeria participants recur across sessions (81 of 82 validation participants also appear in training). OpenWatch subjects are canonical identities after duplicates are merged. The user-study set is split in time (first 80 % of each recording for training), so its participants recur. HumanML3D hours exclude mirrored copies.</p></div>
    <div class="narrow"><h2>Reference corpus</h2><div class="scroll">{corpus_table()}</div>
      <p class="note">10.24 s windows at 50 % overlap; hours count the 5.12 s stride. The scaling analysis draws from a ten-times-larger pool of the same sources (last column).</p></div>
  </div>
</section>

<section aria-labelledby="au-h">
  <h2 id="au-h">What the audit found</h2>
  <ol class="findings">
    <li><b>Duplicated participants in OpenWatch.</b> Two groups of subject IDs ({{39, 41, 42, 46, 47, 48, 49}} and {{40, 43}}) are the same people, so random subject splits put one person in train and test. IMU-Bench splits over the 43 canonical identities; under that split a re-implemented Mix-Token reaches 0.500 clip F1 on real recordings.</li>
    <li><b>Transductive overlap in activity-recognition pretraining.</b> A common pretraining pool bundles WISDM, UTD-MHAD, PAMAP2, RecoFit, SAMoSA, Shoaib and UT-Watch; nearly every probe validation subject was seen, unlabelled, during pretraining. IMU-Bench therefore leaves WISDM, UTD-MHAD, PAMAP2, RecoFit, SAMoSA and UT-Watch out of its evaluation set.</li>
    <li><b>Inconsistent sensor units across sources.</b> EatingSpeed gyroscope data are stored in degrees per second while the other sources use radians per second. The encoders on this page are unaffected: their input is z-scored per window and channel, which removes any constant scale. Physical-units pipelines must convert.</li>
  </ol>
</section>

<section aria-labelledby="dc-h">
  <h2 id="dc-h">Why the decoder is fixed</h2>
  <p>Same frozen MAE encoder, same data and recipe, full-body pose on Nymeria; only the probe decoder changes.</p>
  <div class="scroll" style="max-width:560px"><table class="plain dec"><thead><tr><th scope="col">Probe decoder (cross-channel × temporal layers)</th><th scope="col" style="text-align:right">MPJPE mm ↓</th></tr></thead><tbody>
    <tr><td>0 × 0, linear readout</td><td class="num">105.42</td></tr>
    <tr><td>1 × 0</td><td class="num">98.17</td></tr>
    <tr><td>1 × 4</td><td class="num">94.63</td></tr>
    <tr><td>1 × 6</td><td class="num">94.21</td></tr>
    <tr><td>1 × 16</td><td class="num">94.13</td></tr>
  </tbody></table></div>
  <p class="note">An 11.3 mm spread from the readout alone, larger than the gap between most pretraining objectives. Measured on the earlier 11-source MAE; mean of the last five evaluations.</p>
</section>

<section aria-labelledby="sc-h">
  <h2 id="sc-h">How much pretraining data a wrist IMU needs</h2>
  <p>With all 1,000 motion modes available, 0.1 % of a 1.96-million-window pool (1,959 windows) matches the full pool on reconstruction loss and on the kinematic probes; 0.5 % of the pool already covers 99.7 % of its motion modes.</p>
  <figure><img src="volume_scaling_5src.png" alt="Eight small charts: MAE validation loss, arm pose, full-body pose, odometry, HHAR, MHEALTH and PRISM against the share of the pretraining pool from 0.1% to 100%, mostly flat; the last chart shows motion-mode coverage rising from 18% at 0.01% of the pool to 100% by 5%.">
  <figcaption>Each panel is one metric on its own axis; x is the share of the five-source pool on a log scale, every model trained for 50,000 steps. The dashed line marks the reference corpus size. HHAR and MHEALTH cells are still running.</figcaption></figure>
</section>

<footer>Generated by <code>python -m bench.leaderboard_paper</code> from <code>bench/results/</code> and the task registry. Single seed (42) throughout.</footer>
</div>
<script>
(function(){{
  var tabs=[["tab-probe","pane-probe"],["tab-ft","pane-ft"]];
  function show(id){{tabs.forEach(function(t){{var on=t[0]===id;document.getElementById(t[0]).setAttribute("aria-selected",on);document.getElementById(t[1]).hidden=!on;}});
    try{{localStorage.setItem("imubench-tab",id);}}catch(e){{}}}}
  tabs.forEach(function(t){{document.getElementById(t[0]).addEventListener("click",function(){{show(t[0]);}});}});
  try{{var s=localStorage.getItem("imubench-tab");if(s&&document.getElementById(s))show(s);}}catch(e){{}}
}})();
</script>
"""
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    open(a.out, "w").write(render())
    if os.path.exists(FIG_SRC):
        shutil.copy(FIG_SRC, os.path.join(os.path.dirname(a.out), "volume_scaling_5src.png"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
