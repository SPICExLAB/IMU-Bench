#!/usr/bin/env python3
"""Aggregate scripts/eval_retrieval_alignment.py JSONs into markdown tables.

    python scripts/summarize_retrieval_alignment.py analysis/retrieval_eval [--caption-mode first|all]

Prints one table per dataset (HumanML3D, Nymeria) x gallery (full, 100, 64):
rows = models (ordered by a manifest if present, else by filename), columns =
IMU->Text R@1 / R@5 / R@10 / MRR (MedR)  |  Text->IMU R@1 / R@5 / R@10 / MRR (MedR).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List

# Display order + labels. Key = JSON basename without extension. Anything not
# listed is appended (label = meta.label or basename).
ROW_ORDER: Dict[str, str] = {
    # HumanML3D (UniMTS_data val, 1,197 clips, wrist-only unless noted)
    "humanml3d__ours_mae_finetune_i2t":        "Ours pretrained (MAE init, finetune, i2t-only loss) — headline",
    "humanml3d__ours_mae_finetune_sym":        "Ours pretrained (MAE init, finetune, symmetric loss)",
    "humanml3d__ours_mae_probe":               "Ours pretrained (MAE init, frozen-encoder probe)",
    "humanml3d__ours_scratch_sym":             "Ours scratch (random init, symmetric loss)",
    "humanml3d__ours_scratch_i2t":             "Ours scratch (random init, i2t-only loss)",
    "humanml3d__unimts_scratch_jointmask_wrist": "UniMTS from scratch (native joint-mask recipe, 100 ep) — wrist-only eval",
    "humanml3d__unimts_scratch_fullgrid_wrist":  "UniMTS from scratch (full-grid trained) — wrist-only eval",
    "humanml3d__unimts_scratch_fullgrid_full":   "UniMTS from scratch (full-grid trained) — 22-joint eval (input advantage, reference)",
    "humanml3d__unimts_official_wrist":          "UniMTS official checkpoint, zero-shot — wrist-only (⚠ pretrained on HumanML3D itself)",
    # Nymeria (synthetic wrist IMU + atomic_action captions, held-out sessions)
    "nymeria__ours_mae_finetune":              "Ours pretrained (MAE init, finetune)",
    "nymeria__ours_mae_probe":                 "Ours pretrained (MAE init, frozen-encoder probe)",
    "nymeria__ours_scratch":                   "Ours scratch (random init)",
    "nymeria__unimts_scratch_synthetic":       "UniMTS from scratch",
    "nymeria__unimts_official":                "UniMTS official checkpoint, zero-shot",
}

DATASET_TITLES = {
    "humanml3d": "HumanML3D (UniMTS_data val split: 1,197 held-out clips, aug_texts captions)",
    "nymeria": "Nymeria (synthetic wrist IMU + atomic_action captions, held-out val sessions)",
}


def _fmt(d: Dict[str, Any]) -> str:
    return (f"{d['R@1']:.2f} / {d['R@5']:.2f} / {d['R@10']:.2f} / {d['MRR']:.2f} "
            f"(MedR {d['MedR']:.0f})")


def _load(dirpath: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for fp in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        with open(fp) as fh:
            out[os.path.splitext(os.path.basename(fp))[0]] = json.load(fh)
    return out


def _ordered_keys(keys: List[str]) -> List[str]:
    known = [k for k in ROW_ORDER if k in keys]
    rest = sorted(k for k in keys if k not in ROW_ORDER)
    return known + rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--caption-mode", default="first", choices=["all", "first"],
                    help="'first' (default): one caption per clip (its first line) -> symmetric galleries, "
                         "same #queries both directions, matches UniMTS's ValDataset and the legacy in-batch "
                         "number's one-caption-per-clip convention. 'all': every caption is used (COCO-style; "
                         "IMU->Text counts a hit if ANY of the clip's captions ranks top-k, Text->IMU has one "
                         "query per unique caption) -- HumanML3D aug_texts carries ~5-6 captions/clip so this "
                         "inflates IMU->Text relative to Text->IMU by construction.")
    ap.add_argument("--galleries", default="full,100,64")
    args = ap.parse_args()

    res = _load(args.dir)
    if not res:
        raise SystemExit(f"no JSON results in {args.dir}")
    galleries = [g for g in args.galleries.split(",") if g]

    for ds in ("humanml3d", "nymeria"):
        keys = _ordered_keys([k for k, r in res.items() if r.get("dataset") == ds])
        if not keys:
            continue
        r0 = res[keys[0]]
        p0 = r0["protocols"][args.caption_mode]
        print(f"\n### {DATASET_TITLES.get(ds, ds)}")
        print(f"caption_mode={args.caption_mode}; items={r0['n_items']}, caption pairs={r0['n_caption_pairs']}, "
              f"unique captions={p0['n_unique_captions']}. All numbers in % except MedR (median rank).\n")
        for g in galleries:
            if g == "full":
                title = (f"**Full gallery** — IMU→Text: {p0['full']['n_imu']} queries vs {p0['full']['n_text']} captions; "
                         f"Text→IMU: {p0['full']['n_text']} queries vs {p0['full']['n_imu']} IMU windows")
            else:
                fx0 = p0["fixed"].get(g, {})
                title = (f"**Fixed gallery G={g}** — {fx0.get('n_subsets', '?')} random subsets of {g} clips "
                         f"(text gallery = their unique captions, mean {fx0.get('mean_text_gallery', float('nan')):.0f}); "
                         f"ranks pooled over subsets" + ("; R@1 here is the deterministic analogue of the legacy "
                                                         "batch-64 acc" if g == "64" else ""))
            print(title + "\n")
            print("| Model | IMU→Text R@1 / R@5 / R@10 / MRR (MedR) | Text→IMU R@1 / R@5 / R@10 / MRR (MedR) | Source |")
            print("|---|---|---|---|")
            for k in keys:
                r = res[k]
                proto = r["protocols"][args.caption_mode]
                blk = proto["full"] if g == "full" else proto["fixed"].get(g)
                if blk is None or "skipped" in blk:
                    continue
                label = ROW_ORDER.get(k) or r["meta"].get("label") or k
                m = r["meta"]
                src = m.get("run") or m.get("checkpoint") or ""
                if m.get("ckpt_step", -1) not in (None, -1):
                    src += f" (best.pt, step {m['ckpt_step']})"
                print(f"| {label} | {_fmt(blk['imu2text'])} | {_fmt(blk['text2imu'])} | `{src}` |")
            print()


if __name__ == "__main__":
    main()
