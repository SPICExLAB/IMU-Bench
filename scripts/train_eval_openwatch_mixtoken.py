#!/usr/bin/env python3
"""Mix-Token (Bonazzi et al., arXiv:2605.04791) trained + evaluated under the
§5f fair-comparison protocol (scripts/openwatch_paper_protocol_common.py):
identical contamination-safe posture-stratified split, identical 6
restricted-sub-interval 1s windows per clip, identical window/clip metrics
as our ViT1D MAE and NormWear rows.

Training recipe = the paper's §3.3.3, verbatim where stated:
  class-weighted CE (inverse frequency) | AdamW lr 1e-3, wd 1e-3,
  exponential LR decay | dropout on heads | grad-clip value 1.0 | AMP |
  <=100 epochs, batch 16, seed 42 | early stop on val macro-F1, patience 5,
  best-epoch checkpoint.
Preprocessing = paper §3.2: channel-wise z-score with train-split stats.
Data condition: --condition noaug | aug (train on augmented release, hold
out real recordings). Model input: RAW device axes (the paper's models never
saw an FLU permutation) — the filterbank/feature pipeline is axis-agnostic
anyway, but we keep the paper's convention.

UNSPECIFIED-by-paper choices are marked in baselines/mixtoken/*.py.

Usage
-----
    python scripts/train_eval_openwatch_mixtoken.py --condition aug \
        --out-json results_openwatch_mixtoken_aug.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from baselines.mixtoken.features import d_feat, n_tokens, window_tokens  # noqa: E402
from baselines.mixtoken.model import FixedFilterbank, MixToken  # noqa: E402
from scripts.openwatch_paper_protocol_common import (  # noqa: E402
    NUM_CLASSES, SEED, build_protocol_split, build_window_arrays,
    inverse_frequency_weights, window_and_clip_metrics,
)

EPOCHS = 100
BATCH = 16
PATIENCE = 5
LR = 1e-3
WD = 1e-3
LR_GAMMA = 0.95  # UNSPECIFIED: "exponential decay" — per-epoch gamma


def featurize(X: np.ndarray, fb: FixedFilterbank, cache_path: str | None):
    """X [N, W, C] raw -> (X_fb [N, C*B, W], tokens [N, T, D])."""
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["xfb"], d["tok"]
    xfb = np.stack([fb(x).T for x in X]).astype(np.float32)          # [N, C*B, W]
    tok = np.stack([window_tokens(x) for x in X]).astype(np.float32)  # [N, T, D]
    if cache_path:
        np.savez(cache_path, xfb=xfb, tok=tok)
    return xfb, tok


def zscore_fit(a: np.ndarray, axes):
    mu = a.mean(axis=axes, keepdims=True)
    sd = a.std(axis=axes, keepdims=True) + 1e-6
    return mu, sd


@torch.no_grad()
def predict(model, xfb, tok, device, bs=256):
    model.eval()
    out = []
    for i in range(0, len(xfb), bs):
        a = torch.from_numpy(xfb[i:i + bs]).to(device)
        b = torch.from_numpy(tok[i:i + bs]).to(device)
        out.append(model(a, b).argmax(-1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["noaug", "aug"], required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--cache-dir", default="cache_openwatch_mixtoken")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.cache_dir, exist_ok=True)
    print("device:", device, "condition:", args.condition)

    train, val, test = build_protocol_split(args.condition)
    print(f"records train={len(train)} val={len(val)} test={len(test)}")

    fb = FixedFilterbank()
    splits = {}
    for name, recs in [("train", train), ("val", val), ("test", test)]:
        X, y, cid = build_window_arrays(recs, flu=False)  # raw device axes
        cache = os.path.join(args.cache_dir, f"{args.condition}_{name}.npz")
        t0 = time.time()
        xfb, tok = featurize(X, fb, cache)
        print(f"{name}: windows={len(y)} xfb={xfb.shape} tok={tok.shape} ({time.time()-t0:.0f}s)")
        splits[name] = dict(xfb=xfb, tok=tok, y=y, cid=cid, clip_labels={i: r.label for i, r in enumerate(recs)})

    # channel-wise z-score with TRAIN stats (paper §3.2), applied to both inputs
    mu_fb, sd_fb = zscore_fit(splits["train"]["xfb"], axes=(0, 2))
    mu_tk, sd_tk = zscore_fit(splits["train"]["tok"], axes=(0,))
    for s in splits.values():
        s["xfb"] = ((s["xfb"] - mu_fb) / sd_fb).astype(np.float32)
        s["tok"] = ((s["tok"] - mu_tk) / sd_tk).astype(np.float32)

    model = MixToken(n_channels=6, n_bands=fb.n_bands, n_tokens=n_tokens(6), d_feat=d_feat(6),
                     num_classes=NUM_CLASSES).to(device)
    print(f"Mix-Token params: {model.n_params()/1e3:.1f}K")

    cw = torch.tensor(inverse_frequency_weights(splits["train"]["y"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=LR_GAMMA)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    tr = splits["train"]
    n = len(tr["y"])
    best_f1, best_state, best_epoch, bad = -1.0, None, -1, 0
    for epoch in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            a = torch.from_numpy(tr["xfb"][idx]).to(device)
            b = torch.from_numpy(tr["tok"][idx]).to(device)
            yb = torch.from_numpy(tr["y"][idx]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(a, b)
                loss = F.cross_entropy(logits.float(), yb, weight=cw)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(idx)
        sched.step()

        va = splits["val"]
        vp = predict(model, va["xfb"], va["tok"], device)
        vm = window_and_clip_metrics(va["y"], vp, va["cid"], va["clip_labels"])
        f1 = vm["window"]["macro_f1"]
        pi = torch.softmax(model.fusion_w.detach(), 0).cpu().numpy()
        print(f"epoch {epoch:3d} loss={tot/n:.4f} val_win_f1={f1:.4f} val_clip_acc={vm['clip']['acc']:.4f} "
              f"pi={np.round(pi, 3)} lr={sched.get_last_lr()[0]:.2e}")
        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[early stop] epoch={epoch} best_epoch={best_epoch} best_val_win_f1={best_f1:.4f}")
                break

    model.load_state_dict(best_state)
    result = {"model": "Mix-Token (reimpl.)", "condition": args.condition, "params": model.n_params(),
              "best_epoch": best_epoch, "seed": args.seed,
              "fusion_pi": torch.softmax(model.fusion_w.detach(), 0).cpu().numpy().tolist()}
    for split in ["val", "test"]:
        s = splits[split]
        p = predict(model, s["xfb"], s["tok"], device)
        result[split] = window_and_clip_metrics(s["y"], p, s["cid"], s["clip_labels"])
        print(f"[{split}] window acc={result[split]['window']['acc']:.4f} f1={result[split]['window']['macro_f1']:.4f} | "
              f"clip acc={result[split]['clip']['acc']:.4f} f1={result[split]['clip']['macro_f1']:.4f}")
    with open(args.out_json, "w") as fh:
        json.dump(result, fh, indent=2)
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
