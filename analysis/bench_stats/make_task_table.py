#!/usr/bin/env python
"""IMU-Bench task table (paper Table 1) from the bench registry + measured dataset stats.

    python analysis/bench_stats/make_task_table.py [out.tex]

Inputs: bench/registry/tasks.yaml (tasks, metrics, splits, class counts),
        analysis/bench_stats/har_parquet_stats.json (har_parquet_stats.py),
        analysis/bench_stats/other_stats.json (other_stats.py).
Subjects / hours are measured from the data, never copied from prose. Datasets that overlap the
reference 5-source pretraining corpus (its `realdata` source) are marked with a dagger.
"""
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = sys.argv[1] if len(sys.argv) > 1 else "/gpfs/home/vbc1591/paper_iclr/Sections/tab_imubench_tasks.tex"

TASKS = yaml.safe_load(open(os.path.join(REPO, "bench/registry/tasks.yaml")))["tasks"]
HAR = json.load(open(os.path.join(HERE, "har_parquet_stats.json")))
OTH = json.load(open(os.path.join(HERE, "other_stats.json")))
# datasets inside the reference corpus' `realdata` source (configs/vit1d_ssl_mae_varlen_altattn_realdata_*)
IN_5SRC = {"wisdm", "utd_mhad", "pamap2", "recofit", "samosa", "shoaib", "ut_watch"}


def har_stats(key):
    for src in (HAR, OTH.get("hhar", {}), OTH.get("mhealth", {})):
        if key in src:
            return src[key]
    raise KeyError(key)


def fmt_h(h):
    if h is None:
        return "--"
    if float(h).is_integer() and h >= 100:          # published round totals (300 h, 2,600 h) without a decimal
        return f"{int(h):,}"
    return f"{h:,.1f}" if h >= 1 else f"{h:.2f}"


def arrow(task):
    m = TASKS[task]["metrics"]
    return r"$\uparrow$" if m[m["primary"]].get("higher_is_better", True) else r"$\downarrow$"


def n_classes(task):
    """How many classes the benchmark actually trains a head for.

    Prefer the head size recorded by the scorer: a dataset's raw label column can be far larger than
    the task (PRISM ships 96 fine-grained procedure steps but the task is its 5 global ADL classes),
    so counting distinct labels in the parquet over-reports. Fall back to the subtitle.
    """
    import glob as _glob
    import json as _json
    import re
    for f in sorted(_glob.glob(os.path.join(REPO, "bench/results", task, "*__s42.json"))):
        try:
            k = ((_json.load(open(f)).get("provenance") or {}).get("info") or {}).get("n_classes_head")
        except Exception:
            continue
        if k:
            return int(k)
    s = TASKS[task].get("subtitle", "") + " " + str((TASKS[task].get("output") or {}).get("shape", ""))
    m = re.search(r"\((\d+) (?:raw labels|classes)", s) or re.search(r"K=(\d+)", s)
    return int(m.group(1)) if m else None


# Hours in Table 1 are RELEASED dataset sizes (user request 2026-09-22), not the windows that enter training. For the
# evaluation sets held in full the measured values below already are the release sizes. Where the benchmark or the
# reference corpus uses a part of a release, the release size is given here with its provenance:
RELEASE_HOURS = {
    # task datasets
    "nymeria":     (300.0, "published: Ma et al. 2024, 300 h / 264 participants; we hold 968 sessions = 253.1 h"),
    # pretraining sources (the reference corpus takes 197,094 windows = 280 h of these, Table 16)
    "ocdetect":    (2600.0, "published: Burchard et al. 2026 (Sci. Data), 2,600 h of all-day wrist recordings, 22 participants"),
    "eatingspeed": (251.7 + 261.7 + 12.5, "wall-clock recording hours measured from the pickles: FD-I 251.7 (README 251.70), FD-II 261.7 (README 261.68), MO 12.5"),
    "motionx":     (40.4, "the released Motion-X++ subsets we hold (7 subsets, 25,861 clips), measured 2026-08-20"),
    "sonic":       (144.2, "BONES-SEED 288.3 raw h with exactly 50 % mirrored duplicates removed"),
    # HAR pool sets: rows / 50 Hz in the master parquet (release held in full; bench/generated/tmp/release_hours_7065675.out)
    "ut_watch": (45.3, "parquet"), "wisdm": (44.7, "parquet"), "recofit": (43.0, "parquet"),
    "pamap2": (7.1, "parquet"), "samosa": (6.7, "parquet"), "shoaib": (6.5, "parquet"),
}

rows = []
ny = OTH["nymeria"]
ny_h = RELEASE_HOURS["nymeria"][0]                 # released size; the 253.1 h we hold are in Appendix B
ny_subj = ny["participants_total"]
rows.append(("Pose", "Arm pose", "Nymeria", ny_subj, ny_h, "5-joint rotations",
             "pos.\\ err.\\ (cm) " + arrow("arm_pose_nymeria"), "session"))
rows.append(("Pose", "Full-body pose", "Nymeria", None, None, "23-joint rotations",
             "MPJPE (mm) " + arrow("full_body_pose_nymeria"), "session"))
rows.append(("Pose", "Inertial odometry", "Nymeria", None, None, "wrist-frame velocity",
             "ATE (m) " + arrow("odometry_nymeria_local"), "session"))
ow = OTH["openwatch"]
rows.append(("Gesture", "Microgesture", "OpenWatch", ow["canonical_subjects"], ow["hours"], "6 classes / clip",
             "clip F1 " + arrow("microgesture_openwatch"), "subject"))
hm = OTH["humanml3d"]
# every official list pairs each clip with its mirrored copy "M<id>" of identical length
# (checked: 11,692/730/2,192 pairs), so unique motion = half of the listed hours
hm_h = sum(v["hours"] for v in hm.values()) / 2
rows.append(("Language", "IMU--text retrieval", "HumanML3D (sim.)", None, hm_h, "text embedding",
             "R@1, $G{=}100$ " + arrow("language_humanml3d"), "official"))
pads = OTH["pads"]
pads = pads[next(iter(pads))] if "subjects" not in pads else pads
rows.append(("Clinical", "Parkinson's disease", "PADS", pads["subjects"], pads["hours"], "3 classes / subject",
             "subject F1 " + arrow("pd_pads"), "subject"))
# WISDM, UTD-MHAD, PAMAP2, RecoFit, SAMoSA and UT-Watch were removed from IMU-Bench on 2026-09-17: they are bundled
# into common pretraining pools (incl. the reference corpus), so their evaluation subjects are seen during pretraining.
HAR_ORDER = [("har_hhar", "hhar", "HHAR"), ("har_mhealth", "mhealth", "MHEALTH"), ("har_prism", "prism", "PRISM")]
# user-study watch (added 2026-09-18): stats measured from its parquet; split in time within each recording
import pandas as _pd
_uw = _pd.read_parquet("/projects/b53017/vbc1591/userstudy_watch_dataset.parquet",
                       columns=["subject_id", "session_id", "timestamp_ns", "dataset_activity_id"])
_uw_h = float(_uw.groupby("session_id").timestamp_ns.agg(lambda t: (t.max() - t.min()) / 3.6e12).sum())
USERWATCH = {"subjects": int(_uw.subject_id.nunique()), "hours": _uw_h, "classes": int(_uw.dataset_activity_id.nunique())}
for task, key, name in HAR_ORDER:
    st = har_stats(key)
    k = n_classes(task) or st.get("classes")
    rows.append(("Activity", "Activity recognition", name + (r"$^\dagger$" if key in IN_5SRC else ""),
                 st["subjects"], st["hours"], f"{k} classes / window", "macro-F1 " + arrow(task), "subject"))
rows.append(("Activity", "Activity recognition", "TransfHAR", USERWATCH["subjects"], USERWATCH["hours"],
             f"{USERWATCH['classes']} classes / window", "macro-F1 " + arrow("har_userwatch"), "time"))

# ---- the reference pretraining corpus, as the last block (numbers parsed from the generated corpus table) ----
import re as _re
_corpus = open(os.path.join(os.path.dirname(OUT), "tab_pretrain_corpus.tex")).read()
PRE_NAME = {"HAR sets": "HAR pool (6 sets)", "OCDetect": "OCDetect", "EatingSpeed": "EatingSpeed",
            "Motion-X": "Motion-X (sim.)", "BONES-SEED": "BONES-SEED (sim.)"}
pre_rows, pre_total = [], None
for m in _re.finditer(r"^([A-Za-z\\-][^&\n]*?)(?:\$\^\\dagger\$)?\s*&\s*(real|syn\.)?\s*&\s*([\d,]+)\s*&\s*([\d.]+)\s*&", _corpus, flags=_re.M):
    name, _, wins, hrs = m.group(1).strip(), m.group(2), m.group(3), float(m.group(4))
    if name == "Total":
        pre_total = (wins, hrs)
    elif name in PRE_NAME:
        pre_rows.append((PRE_NAME[name], wins, hrs))
assert len(pre_rows) == 5 and pre_total, (pre_rows, pre_total)
# The HAR pool is one cached array without provenance; analysis/bench_stats/har_pool_per_dataset.py reproduces its
# per-dataset window counts from the same session index and window specs (train total asserted = 93,510). Split
# the pool row into one row per set, largest first; UTD-MHAD contributes no window (clips shorter than 10.24 s).
_hp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "har_pool_per_dataset.json")
HAR_NAME = {"pamap2": "PAMAP2", "recofit": "RecoFit", "samosa": "SAMoSA", "shoaib": "Shoaib", "ut_watch": "UT-Watch",
            "utd_mhad": "UTD-MHAD", "wisdm": "WISDM"}
if os.path.exists(_hp_path):
    _hp = json.load(open(_hp_path))
    _per = {k: v for k, v in _hp["train"]["per_dataset"].items() if v > 0}
    assert sum(_per.values()) == int(pre_rows[0][1].replace(",", "")), (sum(_per.values()), pre_rows[0])
    _split = [(HAR_NAME.get(k, k) + " (HAR)", f"{v:,}", RELEASE_HOURS[k][0]) for k, v in sorted(_per.items(), key=lambda kv: -RELEASE_HOURS[kv[0]][0])]
    pre_rows = _split + [(n, w, RELEASE_HOURS[k][0]) for (n, w, _), k in zip(pre_rows[1:], ("ocdetect", "eatingspeed", "motionx", "sonic"))]
    pre_total = (pre_total[0], sum(h for _, _, h in pre_rows))

lines = [r"% Generated by IMU-SSL/analysis/bench_stats/make_task_table.py -- edit the script, not this file.",
         r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\caption{IMU-Bench tasks and the pretraining sources. Hours are the sizes of the released datasets: measured from the data for the sets we hold in full, "
         r"and the published totals for Nymeria (300\,h, of which we hold 253) and OCDetect (2,600\,h); mirrored copies (HumanML3D, BONES-SEED) are excluded. "
         r"Every task is scored under frozen probing, fine-tuning and training from scratch; the output, metric, split rule and subject count of each task are in Appendix~\ref{app:tasks}. "
         r"TransfHAR is the smartwatch set of \citet{bradshaw2026transfhar}. "
         r"The last block lists the sources of the reference corpus on which every self-supervised encoder, the MAE included, is pretrained; "
         r"the corpus draws 197,094 windows (280\,h) from them (Table~\ref{tab:pretraining_data_composition}) and shares no dataset with the tasks above.}",
         r"\label{tab:imubench_tasks}",
         r"\setlength{\tabcolsep}{3.5pt}",
         r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
         r"\begin{tabular}{llr}", r"\toprule",
         r"Task & Dataset & Hours \\", r"\midrule"]
prev = None
for fam, task, ds, subj, hrs, out, met, split in rows:
    if prev is not None and fam != prev:
        lines.append(r"\midrule")
    s_h = "--" if hrs is None else fmt_h(hrs)
    if ds == "Nymeria" and subj is None:            # the three Nymeria tasks share one recording set
        s_h = r"\multicolumn{1}{c}{''}"
    lines.append(f"{task if (fam != prev or fam != 'Activity') else ''} & {ds} & {s_h} \\\\")
    prev = fam
lines.append(r"\midrule")
for i, (name, wins, hrs) in enumerate(pre_rows):
    lines.append(f"{'MAE pretraining' if i == 0 else ''} & {name} & {fmt_h(hrs)} \\\\")
# no total row: the hour column is each dataset's RELEASED size, and summing sizes of sources we only partly use is meaningless
lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]
open(OUT, "w").write("\n".join(lines))
print(f"wrote {OUT} ({len(rows)} tasks)")
for r in rows:
    print("  ", r)

sys.path.insert(0, os.path.join(REPO, "paper_results"))
from paper_io import write_csv  # noqa: E402
write_csv("tab1_tasks", "tasks.csv", ["family", "task", "dataset", "subjects", "hours", "output", "metric", "split"],
          [[fam, task, ds.replace("$^\\dagger$", ""), subj, hrs, out, met.replace(" $\\uparrow$", " (higher better)")
            .replace(" $\\downarrow$", " (lower better)").replace("$G{=}100$", "G=100").replace("\\ ", " ").replace("\\", ""), split]
           for fam, task, ds, subj, hrs, out, met, split in rows])
