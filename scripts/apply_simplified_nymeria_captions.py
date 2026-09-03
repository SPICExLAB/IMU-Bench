"""Apply an LLM-generated caption-simplification map to Nymeria's
atomic_action.csv files, writing a new ``atomic_action_augmented.csv`` per
sequence that has BOTH the original detailed caption and the simplified
HumanML3D-style caption as two separate description columns -- i.e.
augmenting the text data, not replacing it.

Consumes the ``{original_caption: simplified_caption}`` map produced by
``scripts/generate_simplified_nymeria_captions.py``. Writing this as a
sibling CSV (rather than mutating the original or writing a side-table
the loader has to special-case) means the existing, unmodified
``_parse_caption_csv``/``NYMERIA_CAPTION_SOURCES`` machinery already knows
how to read it -- see the ``atomic_action_augmented`` registry entry in
``imu_lm.data.nymeria_caption_dataset_FLU``, which just points at this
filename with two columns instead of one.

Usage
-----
    python scripts/apply_simplified_nymeria_captions.py \\
        --captions-root /projects/b53017/vbc1591/nymeria_captions \\
        --simplified-map /projects/b53017/vbc1591/nymeria_captions/atomic_action_simplified_map.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ORIGINAL_COLUMN = "Describe my atomic actions"
SIMPLIFIED_COLUMN = "Describe my atomic actions (simplified)"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captions-root", required=True)
    p.add_argument("--simplified-map", required=True)
    p.add_argument("--source-filename", default="atomic_action.csv")
    p.add_argument("--out-filename", default="atomic_action_augmented.csv")
    args = p.parse_args()

    with open(args.simplified_map) as f:
        simplified_map: dict[str, str] = json.load(f)
    logger.info("Loaded %d simplified-caption mappings", len(simplified_map))

    csv_files = sorted(glob.glob(os.path.join(args.captions_root, "*", args.source_filename)))
    logger.info("Found %d %s files to augment", len(csv_files), args.source_filename)

    n_rows, n_missing = 0, 0
    for src_path in csv_files:
        seq_dir = os.path.dirname(src_path)
        out_path = os.path.join(seq_dir, args.out_filename)

        with open(src_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

        if ORIGINAL_COLUMN not in fieldnames:
            logger.warning("skip %s: missing column %r", src_path, ORIGINAL_COLUMN)
            continue

        out_fieldnames = fieldnames + [SIMPLIFIED_COLUMN]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_fieldnames)
            writer.writeheader()
            for row in rows:
                orig = (row.get(ORIGINAL_COLUMN) or "").strip()
                simplified = simplified_map.get(orig, "") if orig else ""
                if orig and not simplified:
                    n_missing += 1
                row[SIMPLIFIED_COLUMN] = simplified
                writer.writerow(row)
                n_rows += 1

    logger.info("done. wrote %d augmented CSVs, %d rows total, %d rows missing a simplified caption",
                len(csv_files), n_rows, n_missing)


if __name__ == "__main__":
    main()
