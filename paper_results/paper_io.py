"""Shared writer for paper_results/: every paper table/figure generator calls `write_csv` so the
numbers behind each figure/table land in one folder per paper element (see paper_results/README.md)."""
import csv
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def write_csv(folder: str, name: str, header, rows) -> str:
    d = os.path.join(ROOT, folder)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
    return path
