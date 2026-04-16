"""Pre-compute and cache frozen CLIP text embeddings for HumanML3D.

Reads every caption from ``<humanml3d_root>/texts/*.txt``, encodes them in
batches through the frozen CLIP text encoder, and writes a single
``{caption_str: np.ndarray[D_clip]}`` dict to a ``.pt`` file.

At training time the dataset loads this cache and returns ready-made
embeddings instead of raw strings, so the CLIP text model never needs to be
loaded at all — saving GPU memory and eliminating the per-batch tokenisation
overhead.

Usage
-----
    python scripts/cache_clip_text_embeddings.py \\
        --humanml3d-root /home/yiquan/HumanML3D \\
        --out            /home/yiquan/HumanML3D/clip_text_emb_cache.pt \\
        [--clip-model    openai/clip-vit-base-patch32] \\
        [--batch-size    512] \\
        [--device        cuda]

Output
------
    A ``torch.save``-compatible dict::

        {
            "<caption string>": np.ndarray  shape=[D_clip], dtype=float32,
            ...
        }

    Embeddings are L2-normalised (unit vectors), matching the output of
    ``CLIPTextEncoder.forward``.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

def _collect_captions(texts_dir: str) -> list[str]:
    """Return sorted list of unique caption strings from all *.txt files."""
    captions: set[str] = set()
    txt_files = sorted(glob.glob(os.path.join(texts_dir, "*.txt")))
    if not txt_files:
        raise RuntimeError(f"No *.txt files found under {texts_dir}")

    for path in txt_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("#")
                if parts and parts[0]:
                    captions.add(parts[0])

    # Always include the empty-string fallback used by _load_clip_imu
    captions.add("")

    logger.info("Collected %d unique captions from %d files", len(captions), len(txt_files))
    return sorted(captions)


@torch.no_grad()
def _encode_captions(
    captions: list[str],
    clip_model: str,
    batch_size: int,
    device: torch.device,
    local_files_only: bool,
) -> dict[str, np.ndarray]:
    """Encode all captions through CLIP and return a {caption: np.ndarray} dict."""
    from transformers import CLIPTokenizer, CLIPTextModel

    logger.info("Loading CLIP model: %s", clip_model)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model, local_files_only=local_files_only)
    model = CLIPTextModel.from_pretrained(clip_model, local_files_only=local_files_only)
    model.eval().to(device)

    cache: dict[str, np.ndarray] = {}
    n = len(captions)

    for start in range(0, n, batch_size):
        batch_caps = captions[start : start + batch_size]

        inputs = tokenizer(
            batch_caps,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        # CLIP uses the [EOS] token (pooler_output) as the text representation
        emb = outputs.pooler_output          # [B, D_clip]
        emb = F.normalize(emb, dim=-1)       # L2-normalise (unit sphere)
        emb_cpu = emb.cpu().float().numpy()  # [B, D_clip]

        for cap, e in zip(batch_caps, emb_cpu):
            cache[cap] = e

        if (start // batch_size + 1) % 10 == 0 or start + batch_size >= n:
            logger.info(
                "  Encoded %d / %d captions (batch %d)",
                min(start + batch_size, n), n, batch_size,
            )

    del model
    return cache


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute frozen CLIP text embeddings for HumanML3D."
    )
    parser.add_argument(
        "--humanml3d-root", required=True,
        help="Root of the HumanML3D dataset (contains joints/ and texts/).",
    )
    parser.add_argument(
        "--out", default=None,
        help=(
            "Output .pt path.  Defaults to "
            "<humanml3d-root>/clip_text_emb_cache.pt"
        ),
    )
    parser.add_argument(
        "--clip-model", default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model identifier (default: openai/clip-vit-base-patch32).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Tokenisation / encoding batch size (default: 512).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="Do not fetch CLIP weights from HuggingFace hub.",
    )
    args = parser.parse_args()

    texts_dir = os.path.join(args.humanml3d_root, "texts")
    if not os.path.isdir(texts_dir):
        raise FileNotFoundError(f"texts/ directory not found under {args.humanml3d_root}")

    out_path = args.out or os.path.join(args.humanml3d_root, "clip_text_emb_cache.pt")
    device   = torch.device(args.device)

    # Step 1: collect unique captions
    captions = _collect_captions(texts_dir)

    # Step 2: encode
    logger.info("Encoding on device: %s  (batch_size=%d)", device, args.batch_size)
    cache = _encode_captions(
        captions,
        clip_model       = args.clip_model,
        batch_size       = args.batch_size,
        device           = device,
        local_files_only = args.local_files_only,
    )

    # Step 3: save
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(cache, out_path)

    embed_dim = next(iter(cache.values())).shape[0]
    logger.info(
        "Saved %d embeddings (dim=%d) → %s",
        len(cache), embed_dim, out_path,
    )


if __name__ == "__main__":
    main()
