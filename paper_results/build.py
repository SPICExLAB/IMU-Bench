#!/usr/bin/env python3
"""Regenerate every paper table/figure and the tidy CSVs behind them, one folder per paper element.

    python paper_results/build.py            # from the IMU-SSL repo root, TransfHAR_env active

Each generator writes the LaTeX/figure into the paper (paper_iclr/Sections, paper_iclr/Pictures) AND its
numbers into paper_results/<element>/ via paper_results/paper_io.py, so the paper and the CSVs never
disagree. Pending cells are empty in the CSVs and "--" in the paper. See paper_results/README.md.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PAPER = "/gpfs/home/vbc1591/paper_iclr"
STEPS = [  # (paper element folder, command, cwd)
    ("tab1_tasks", ["analysis/bench_stats/make_task_table.py"], REPO),
    ("appendix_pretrain_corpus", ["analysis/bench_stats/make_pretrain_table.py"], REPO),
    ("tab2_main_results", ["analysis/paper_tables/make_main_table.py"], REPO),
    ("tab2_main_results/per_task", ["analysis/paper_tables/make_task_tables.py"], REPO),
    ("appendix_selection", ["analysis/paper_tables/make_selection_table.py"], REPO),
    ("appendix_pretrain_details + appendix_downstream_protocols", ["analysis/paper_tables/make_impl_tables.py"], REPO),
    ("appendix_decoder_controls + appendix_corpus_budget", ["analysis/paper_tables/make_ablation_tables.py"], REPO),
    ("appendix_audit_openwatch", ["analysis/paper_tables/make_audit_table.py"], REPO),
    ("fig2_transfer", ["plot_transferability_5src.py"], os.path.join(REPO, "analysis")),
    ("fig3_scaling_grid", ["analysis/diversity_saturation/33_plot_grid_nonleaky.py"], REPO),
    ("appendix_volume_scaling", ["analysis/diversity_saturation/31_plot_volume_5src.py"], REPO),
]
COPIES = [  # rendered figures copied next to their data (reference only; the paper reads Pictures/)
    ("fig1_overview", os.path.join(PAPER, "Sections/fig_overview.tex")),
    ("fig2_transfer", os.path.join(PAPER, "Pictures/transfer5_frozen.png")),
    ("fig2_transfer", os.path.join(PAPER, "Pictures/transfer5_trainable.png")),
    ("fig3_scaling_grid", os.path.join(PAPER, "Pictures/grid_5src_nonleaky.png")),
    ("fig3_scaling_grid", os.path.join(PAPER, "Pictures/grid_5src_nonleaky.pdf")),
    ("appendix_volume_scaling", os.path.join(PAPER, "Pictures/volume_scaling_5src.png")),
    ("appendix_volume_scaling", os.path.join(PAPER, "Pictures/volume_scaling_5src.pdf")),
]


def main():
    ok = True
    for element, cmd, cwd in STEPS:
        r = subprocess.run([sys.executable] + cmd, cwd=cwd, capture_output=True, text=True)
        status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
        ok &= r.returncode == 0
        print(f"[{status:>10}] {element:58s} <- {' '.join(cmd)}")
        if r.returncode:
            print("            " + (r.stderr.strip().splitlines() or ["?"])[-1])
    for folder, src in COPIES:
        if os.path.exists(src):
            os.makedirs(os.path.join(HERE, folder), exist_ok=True)
            shutil.copy2(src, os.path.join(HERE, folder, "current_" + os.path.basename(src)))
    boot = os.path.join(REPO, "bench/results/_bootstrap/summary.json")
    if os.path.exists(boot):
        sys.path.insert(0, HERE)
        from paper_io import write_csv
        d = json.load(open(boot))
        items = d.items() if isinstance(d, dict) else enumerate(d)
        rows = [[k] + [v.get(c) for c in ("point", "lo", "hi", "n_units", "n_groups", "resample_level")]
                for k, v in items if isinstance(v, dict)]
        write_csv("appendix_confidence_intervals", "ci_summary.csv",
                  ["cell", "point", "lo", "hi", "n_units", "n_groups", "resample_level"], rows)
        print(f"[{'ok':>10}] appendix_confidence_intervals{'':30s} <- bench/results/_bootstrap/summary.json")
    else:
        print(f"[{'pending':>10}] appendix_confidence_intervals (bench/results/_bootstrap/summary.json not written yet)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
