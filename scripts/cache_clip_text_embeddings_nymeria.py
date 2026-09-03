"""Pre-compute and cache frozen CLIP text embeddings for Nymeria captions.

Sibling to ``scripts/cache_clip_text_embeddings.py`` (HumanML3D), adapted for
Nymeria's narration CSV formats: reads every
``<captions_root>/*/<filename>`` for the selected ``--caption-source``
(``motion_narration`` default, or ``atomic_action``/``activity_summarization``
-- see ``imu_lm.data.nymeria_caption_dataset_FLU.NYMERIA_CAPTION_SOURCES``,
the single source of truth for filename/column-schema per narration type),
collects every non-empty caption across that source's description column(s),
encodes them in batches through the frozen CLIP text encoder, and writes a
single ``{caption_str: np.ndarray[D_clip]}`` dict to a ``.pt`` file -- same
cache format/consumer (``imu_lm.objectives.contrastive_clip_1d``) as the
HumanML3D cache. Different caption sources get separate cache files (see
``--out`` default), since they're different, non-overlapping caption pools.

Usage
-----
    python scripts/cache_clip_text_embeddings_nymeria.py \\
        --captions-root  /projects/b53017/vbc1591/nymeria_captions \\
        [--caption-source atomic_action] \\
        [--out           /projects/b53017/vbc1591/nymeria_captions/clip_text_emb_cache_atomic_action.pt] \\
        [--clip-model    openai/clip-vit-base-patch32] \\
        [--batch-size    512] \\
        [--device        cuda]
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imu_lm.data.nymeria_caption_dataset_FLU import NYMERIA_CAPTION_SOURCES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _collect_captions(captions_root: str, caption_source: str) -> list[str]:
    source = NYMERIA_CAPTION_SOURCES[caption_source]
    filename, columns = source["filename"], source["columns"]

    captions: set[str] = set()
    csv_files = sorted(glob.glob(os.path.join(captions_root, "*", filename)))
    if not csv_files:
        raise RuntimeError(f"No */{filename} files found under {captions_root}")

    for path in csv_files:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in columns:
                    val = row.get(col)
                    if val and val.strip():
                        captions.add(val.strip())

    captions.add("")
    logger.info("Collected %d unique captions from %d sequence CSVs (source=%s)",
                len(captions), len(csv_files), caption_source)
    return sorted(captions)


@torch.no_grad()
def _encode_captions(
    captions: list[str],
    clip_model: str,
    batch_size: int,
    device: torch.device,
    local_files_only: bool,
) -> dict[str, np.ndarray]:
    from transformers import CLIPTokenizer, CLIPTextModel

    logger.info("Loading CLIP model: %s", clip_model)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model, local_files_only=local_files_only)
    model = CLIPTextModel.from_pretrained(clip_model, local_files_only=local_files_only)
    model.eval().to(device)

    cache: dict[str, np.ndarray] = {}
    n = len(captions)

    for start in range(0, n, batch_size):
        batch_caps = captions[start : start + batch_size]
        inputs = tokenizer(batch_caps, padding=True, truncation=True, max_length=77, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        emb = F.normalize(outputs.pooler_output, dim=-1)
        emb_cpu = emb.cpu().float().numpy()
        for cap, e in zip(batch_caps, emb_cpu):
            cache[cap] = e
        if (start // batch_size + 1) % 10 == 0 or start + batch_size >= n:
            logger.info("  Encoded %d / %d captions (batch %d)", min(start + batch_size, n), n, batch_size)

    del model
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute frozen CLIP text embeddings for Nymeria captions.")
    parser.add_argument("--captions-root", required=True)
    parser.add_argument("--caption-source", default="motion_narration", choices=list(NYMERIA_CAPTION_SOURCES))
    parser.add_argument("--out", default=None,
                         help="Defaults to <captions-root>/clip_text_emb_cache[_<source>].pt")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    default_name = ("clip_text_emb_cache.pt" if args.caption_source == "motion_narration"
                     else f"clip_text_emb_cache_{args.caption_source}.pt")
    out_path = args.out or os.path.join(args.captions_root, default_name)
    device = torch.device(args.device)

    captions = _collect_captions(args.captions_root, args.caption_source)
    logger.info("Encoding on device: %s  (batch_size=%d)", device, args.batch_size)
    cache = _encode_captions(
        captions, clip_model=args.clip_model, batch_size=args.batch_size,
        device=device, local_files_only=args.local_files_only,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(cache, out_path)

    embed_dim = next(iter(cache.values())).shape[0]
    logger.info("Saved %d embeddings (dim=%d) -> %s", len(cache), embed_dim, out_path)


if __name__ == "__main__":
    main()
