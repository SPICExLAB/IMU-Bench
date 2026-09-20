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


rows = []
ny = OTH["nymeria"]
ny_h = ny["train"]["hours"] + ny["val"]["hours"]
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
rows.append(("Activity", "Activity recognition", "User study (ours)", USERWATCH["subjects"], USERWATCH["hours"],
             f"{USERWATCH['classes']} classes / window", "macro-F1 " + arrow("har_userwatch"), "time"))

lines = [r"% Generated by IMU-SSL/analysis/bench_stats/make_task_table.py -- edit the script, not this file.",
         r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\caption{IMU-Bench tasks. Subjects and hours are measured from the released data (train and evaluation "
         r"splits together). Every task is scored under frozen probing, fine-tuning and training from scratch. "
         r"Splits: \emph{subject} = subject-disjoint; \emph{session} = session-disjoint (Nymeria participants recur "
         r"across sessions); \emph{time} = first 80\,\% of each recording for training, last 20\,\% for evaluation (participants recur); \emph{official} = the dataset's own split. OpenWatch subjects are canonical identities after "
         r"removing duplicated participants; HumanML3D hours exclude mirrored copies. "
         r"None of these datasets is part of the reference pretraining corpus.}",
         r"\label{tab:imubench_tasks}",
         r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{llrrlll}", r"\toprule",
         r"Task & Dataset & Subj. & Hours & Output & Metric & Split \\", r"\midrule"]
prev = None
for fam, task, ds, subj, hrs, out, met, split in rows:
    if prev is not None and fam != prev:
        lines.append(r"\midrule")
    s_subj = "--" if subj is None else f"{subj:,}"
    s_h = "--" if hrs is None else fmt_h(hrs)
    if ds == "Nymeria" and subj is None:
        s_subj, s_h = r"\multicolumn{1}{c}{''}", r"\multicolumn{1}{c}{''}"
    lines.append(f"{task if (fam != prev or fam != 'Activity') else ''} & {ds} & {s_subj} & {s_h} & {out} & {met} & {split} \\\\")
    prev = fam
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
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
