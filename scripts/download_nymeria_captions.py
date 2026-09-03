"""Download Nymeria narration caption CSVs.

Reads a Nymeria download manifest (e.g. ``nymeria_download_urls_caption.json``,
caption-only, 236/1100 sequences; or ``Nymeria_download_urls_all.json``, every
file type across all 1100 sequences) — ``sequences.<id>.<file_key>`` holds a
signed CDN `download_url`, `filename`, `sha1sum`, `file_size_bytes` for each
sequence that has that file; sequences without it are skipped. Downloads each
CSV to ``<out_root>/<sequence_id>/<out_filename>``, verifying the sha1 hash
after download. Idempotent: skips files that already exist and verify
correctly; only re-downloads missing/corrupt ones.

``--file-key`` selects which narration type (manifest dict key), default
``narration_motion_narration_csv`` (4 aspect-columns, ~5s segments, the
original caption source this benchmark's Nymeria CLIP task uses). Also
present in the full manifest: ``narration_atomic_action_csv`` (1 column,
~5s segments, longer free-text per row, covers 845/1100 sequences — more
than 3x the coverage of motion_narration) and
``narration_activity_summarization_csv`` (1 column, ~30s segments, coarser
activity-level summaries, 864/1100 sequences). Note: none of the three are
"motion-only" in the sense of omitting scene/object references — all three
are naturalistic first-person narrations that mention rooms/objects/context
just like motion_narration.csv does; verified directly by reading sample
files, not assumed.

Usage
-----
    python scripts/download_nymeria_captions.py \\
        --manifest /home/vbc1591/IMU-SSL/nymeria_download_urls_caption.json \\
        --out-root /projects/b53017/vbc1591/nymeria_captions \\
        [--file-key narration_atomic_action_csv] \\
        [--out-filename atomic_action.csv]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(url: str, out_path: str, expected_sha1: str, retries: int = 3) -> bool:
    for attempt in range(1, retries + 1):
        try:
            tmp_path = out_path + ".part"
            urllib.request.urlretrieve(url, tmp_path)
            actual = _sha1(tmp_path)
            if actual != expected_sha1:
                os.remove(tmp_path)
                logger.warning("sha1 mismatch (attempt %d/%d): %s (expected %s, got %s)",
                               attempt, retries, out_path, expected_sha1, actual)
                continue
            os.replace(tmp_path, out_path)
            return True
        except Exception as exc:
            logger.warning("download failed (attempt %d/%d) for %s: %s", attempt, retries, out_path, exc)
            time.sleep(1.0)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--file-key", default="narration_motion_narration_csv")
    ap.add_argument("--out-filename", default="motion_narration.csv")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    sequences = manifest["sequences"]
    to_download = {
        sid: info[args.file_key]
        for sid, info in sequences.items()
        if info and args.file_key in info
    }
    logger.info("manifest: %d total sequences, %d with %s", len(sequences), len(to_download), args.file_key)

    os.makedirs(args.out_root, exist_ok=True)

    n_ok, n_skipped, n_failed = 0, 0, 0
    for i, (sid, info) in enumerate(sorted(to_download.items()), 1):
        seq_dir = os.path.join(args.out_root, sid)
        os.makedirs(seq_dir, exist_ok=True)
        out_path = os.path.join(seq_dir, args.out_filename)

        if os.path.exists(out_path) and _sha1(out_path) == info["sha1sum"]:
            n_skipped += 1
            continue

        ok = _download_one(info["download_url"], out_path, info["sha1sum"])
        if ok:
            n_ok += 1
        else:
            n_failed += 1
            logger.error("FAILED after retries: %s", sid)

        if i % 50 == 0 or i == len(to_download):
            logger.info("progress: %d/%d (downloaded=%d skipped=%d failed=%d)",
                        i, len(to_download), n_ok, n_skipped, n_failed)

    logger.info("done. downloaded=%d skipped(already-had)=%d failed=%d / total=%d",
                n_ok, n_skipped, n_failed, len(to_download))
    if n_failed:
        raise SystemExit(f"{n_failed} caption downloads failed after retries")


if __name__ == "__main__":
    main()
