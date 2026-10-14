#!/usr/bin/env python3
"""Sync the `primary_metric` field of existing result JSONs with the task registry.

`Cell.primary()` reads `primary_metric` out of each result JSON, not out of the task
card, because that field records which metric the cell was scored on. So changing
`metrics.primary` in bench/registry/tasks.yaml has NO effect on already-scored cells
until they are re-stamped -- the tables keep reporting the old metric while the card
claims the new one, which is worse than either alone.

This re-stamps it. It is metadata, not an observation: the new metric must already be
present in the file (it always is -- every scorer writes the task's full metric set),
so nothing is recomputed and no number changes. A file whose new primary is missing is
left alone and reported, never silently emptied.

    python -m bench.restamp_primary --task pd_pads [--dry-run] [--results-root DIR]
"""
from __future__ import annotations

import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.registry_io import REPO, load_tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--results-root", default=None,
                    help="default: bench/results (and bench/results_nosel if present)")
    a = ap.parse_args()

    want = load_tasks()["tasks"][a.task]["metrics"]["primary"]
    roots = [a.results_root] if a.results_root else [
        os.path.join(REPO, "bench", "results"), os.path.join(REPO, "bench", "results_nosel")]
    changed = already = skipped = 0
    for root in roots:
        d = os.path.join(root, a.task)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            p = os.path.join(d, fn)
            js = json.load(open(p))
            cur = js.get("primary_metric")
            if cur == want:
                already += 1
                continue
            if (js.get("metrics") or {}).get(want) is None:
                print(f"  SKIP (no {want} in file): {os.path.relpath(p, REPO)}")
                skipped += 1
                continue
            print(f"  {os.path.relpath(p, REPO)}: {cur} -> {want}"
                  f"   ({js['metrics'].get(cur)} -> {js['metrics'][want]})")
            if not a.dry_run:
                js["primary_metric"] = want
                with open(p, "w") as f:
                    json.dump(js, f, indent=1)
            changed += 1
    verb = "would restamp" if a.dry_run else "restamped"
    print(f"\n{a.task}: {verb} {changed}, already correct {already}, skipped {skipped}")
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
