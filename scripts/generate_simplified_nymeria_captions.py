"""Use a local LLM to rewrite Nymeria's long, scene-referencing atomic_action
captions into short, HumanML3D-style motion-only descriptions, and cache the
result as ``{original_caption: simplified_caption}``.

Nymeria's atomic_action.csv captions are detailed first-person paragraphs
that reference rooms/objects heavily (e.g. "C opens the laundry closet using
both of her hands... leans towards the dryer and touches the lid"). HumanML3D
captions are short, third-person, motion-only sentences ("a person walks
forward then sits down"). This script asks an instruction-tuned LLM to
rewrite the former into the latter style, discarding scene/object references
and keeping only the physical motion. The result is used to AUGMENT (not
replace) the original captions -- see
``imu_lm.data.nymeria_caption_dataset_FLU.NYMERIA_CAPTION_SOURCES`` and
``scripts/apply_simplified_nymeria_captions.py``, which writes a new
``atomic_action_augmented.csv`` per sequence with both the original and
simplified text as two separate description columns.

Only unique captions are processed (deduplicated across all sequences) to
avoid redundant generation -- Nymeria's atomic_action captions repeat across
segments/sequences even though they're less repetitive than motion_narration.

Usage
-----
    python scripts/generate_simplified_nymeria_captions.py \\
        --captions-root /projects/b53017/vbc1591/nymeria_captions \\
        --model         /scratch/cxk2993/huggingface/hub  (resolved via HF_HOME) \\
        --model-id      Qwen/Qwen2.5-0.5B-Instruct \\
        --out           /projects/b53017/vbc1591/nymeria_captions/atomic_action_simplified_map.json \\
        [--limit N]     debug: only process the first N unique captions
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imu_lm.data.nymeria_caption_dataset_FLU import NYMERIA_CAPTION_SOURCES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You rewrite detailed first-person activity descriptions into ONE short, "
    "simple sentence describing ONLY the person's physical body motion, in "
    "the exact style of HumanML3D motion-capture captions. Rules:\n"
    "- Use third person ('a person ...'), present tense.\n"
    "- Describe ONLY body motion/pose (walks, turns, bends, reaches, sits, "
    "stands, raises an arm, etc.) -- do NOT mention rooms, furniture, "
    "appliances, or the environment.\n"
    "- NEVER name a specific object, even a small everyday one. Do not say "
    "'a banana', 'a tissue', 'the ball', 'a bag', 'a drawer', 'a spatula', "
    "etc. -- not even generic-sounding ones. If the action involves "
    "manipulating something, either say 'something'/'it', or better, drop "
    "the object entirely and keep only the body motion (e.g. prefer "
    "'a person reaches down' over 'a person picks up something').\n"
    "- Do NOT mention other people (no 'peer', 'colleague', 'friend') -- "
    "describe only this one person's own motion, even if the input "
    "describes them interacting with someone else (e.g. 'chatting while "
    "gesturing' -> 'gestures with both hands', drop the chatting/peer part "
    "entirely since it's not a body motion).\n"
    "- One sentence, at most ~15 words. No explanations, no quotes, just the sentence.\n\n"
    "Example 1:\n"
    "Input: C is walking in the hallway while holding some clothes hangers "
    "with both of his hands then enters the living room and turns to the left.\n"
    "Output: a person walks forward while holding something with both hands, then turns left.\n\n"
    "Example 2:\n"
    "Input: C is sitting on a chair and slightly leaning her body forward "
    "while playing an arcade game, moving both hands rapidly on the buttons.\n"
    "Output: a person sits and leans forward while moving both hands rapidly.\n\n"
    "Example 3:\n"
    "Input: C bends down in front of the sink then puts down his left hand "
    "and opens the right most cabinet drawer using her right hand.\n"
    "Output: a person bends down then reaches down and opens something with their right hand.\n\n"
    "Example 4:\n"
    "Input: C is standing in front of the kitchen counter while holding the "
    "banana with her left hand and peeling it off with her right hand.\n"
    "Output: a person stands while holding something with one hand and moving the other hand over it.\n\n"
    "Example 5:\n"
    "Input: After kicking the ball with his right foot, he steps backward "
    "and stands in the backyard while waiting for his peer to kick the ball "
    "towards him.\n"
    "Output: a person kicks with their right foot, then steps backward and stands still.\n\n"
    "Example 6:\n"
    "Input: After closing the kitchen cabinet with his right hand while "
    "standing, C turns left, takes a few steps forward, stands, and leans "
    "forward to tear off a piece of tissue from the roll using both hands.\n"
    "Output: a person turns left, steps forward, stands, then leans forward and tears something with both hands.\n"
)


def _collect_unique_captions(captions_root: str, caption_source: str) -> list[str]:
    source = NYMERIA_CAPTION_SOURCES[caption_source]
    filename, columns = source["filename"], source["columns"]
    csv_files = sorted(glob.glob(os.path.join(captions_root, "*", filename)))
    if not csv_files:
        raise RuntimeError(f"No */{filename} found under {captions_root}")

    captions: set[str] = set()
    for path in csv_files:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in columns:
                    val = row.get(col)
                    if val and val.strip():
                        captions.add(val.strip())

    logger.info("Collected %d unique captions from %d sequence CSVs (source=%s)",
                len(captions), len(csv_files), caption_source)
    return sorted(captions)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captions-root", required=True)
    p.add_argument("--caption-source", default="atomic_action", choices=list(NYMERIA_CAPTION_SOURCES))
    p.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--out", default=None,
                    help="Defaults to <captions-root>/<caption-source>_simplified_map.json")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="debug: only process first N captions")
    p.add_argument("--checkpoint-every", type=int, default=2000,
                    help="write partial results to --out every N captions (resumable)")
    p.add_argument("--local-files-only", action="store_true",
                    help="don't touch the network/cache-write path -- required when the HF cache "
                         "dir is read-only to this user (e.g. a shared cache owned by someone else)")
    p.add_argument("--no-auth", action="store_true", default=True,
                    help="force anonymous access, ignoring any stored HF token (default: on -- "
                         "this environment's stored token is invalid/stale and causes spurious "
                         "401 'Repository Not Found' errors on public repos otherwise)")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path = args.out or os.path.join(args.captions_root, f"{args.caption_source}_simplified_map.json")

    captions = _collect_unique_captions(args.captions_root, args.caption_source)
    if args.limit:
        captions = captions[: args.limit]

    # Resume support: skip captions already mapped in an existing partial output.
    existing: dict[str, str] = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        logger.info("Resuming: %d captions already mapped in %s", len(existing), out_path)
    todo = [c for c in captions if c not in existing]
    logger.info("%d captions to process (%d already done)", len(todo), len(existing))

    if not todo:
        logger.info("Nothing to do.")
        return

    hf_token = False if args.no_auth else None
    logger.info("Loading model: %s (local_files_only=%s, no_auth=%s)",
                args.model_id, args.local_files_only, args.no_auth)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, local_files_only=args.local_files_only, token=hf_token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=args.device,
        local_files_only=args.local_files_only, token=hf_token,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def build_prompt(caption: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Input: {caption}\nOutput:"},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    mapping = dict(existing)
    n = len(todo)
    for start in range(0, n, args.batch_size):
        batch_caps = todo[start : start + args.batch_size]
        prompts = [build_prompt(c) for c in batch_caps]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(args.device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_only = out_ids[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for cap, simplified in zip(batch_caps, decoded):
            simplified = simplified.strip().split("\n")[0].strip()
            if simplified.lower().startswith("output:"):
                simplified = simplified[len("output:"):].strip()
            mapping[cap] = simplified

        done = start + len(batch_caps)
        if done % args.checkpoint_every < args.batch_size or done == n:
            with open(out_path, "w") as f:
                json.dump(mapping, f, indent=1)
            logger.info("progress: %d/%d  (checkpoint saved -> %s)", done, n, out_path)

    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=1)
    logger.info("done. %d total captions mapped -> %s", len(mapping), out_path)


if __name__ == "__main__":
    main()
