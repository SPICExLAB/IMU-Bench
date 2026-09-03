#!/usr/bin/env python
"""Uniform weight averaging ("model soup" / SWA-style) of step checkpoints of ONE run.

    python scripts/soup_checkpoints.py --run rtune_soup --min-step 4000 --max-step 8000 [--out soup_4000_8000]

Averages every float tensor of `model`, `clip_head`, `text_enc` (and any other
extra module state dict present) over runs/<run>/checkpoints/step_*.pt whose
step is in [min_step, max_step], and writes runs/<run>/checkpoints/<out>.pt
with the same layout as best.pt (cfg/step copied from the last member) so
scripts/eval_retrieval_alignment.py --ckpt <out> can load it. Requires the run
to have been trained with logging.save_step_checkpoints: true.
"""
import argparse, glob, os, re, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--min-step", type=int, required=True)
    ap.add_argument("--max-step", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = args.run if os.path.isdir(args.run) else os.path.join(ROOT, "runs", args.run)
    ck_dir = os.path.join(run_dir, "checkpoints")
    files = []
    for f in sorted(glob.glob(os.path.join(ck_dir, "step_*.pt"))):
        m = re.search(r"step_(\d+)\.pt$", f)
        if m and args.min_step <= int(m.group(1)) <= args.max_step:
            files.append((int(m.group(1)), f))
    if not files:
        raise SystemExit(f"no step checkpoints in [{args.min_step},{args.max_step}] under {ck_dir}")
    out_name = args.out or f"soup_{args.min_step}_{args.max_step}"
    keys = ("model", "clip_head", "text_enc", "text_mlp", "head", "extra_modules")
    acc, n, last = {}, 0, None
    for step, f in files:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        last = ck
        for k in keys:
            if k not in ck or ck[k] is None:
                continue
            sd = ck[k]
            if k == "extra_modules":          # dict of state dicts
                for mk, msd in sd.items():
                    for pk, v in msd.items():
                        if torch.is_tensor(v) and v.is_floating_point():
                            acc.setdefault((k, mk, pk), torch.zeros_like(v, dtype=torch.float64)).add_(v.double())
            else:
                for pk, v in sd.items():
                    if torch.is_tensor(v) and v.is_floating_point():
                        acc.setdefault((k, pk), torch.zeros_like(v, dtype=torch.float64)).add_(v.double())
        n += 1
        print(f"  + step {step}")
    out = {k: v for k, v in last.items() if k not in ("optimizer", "scheduler")}
    for key, s in acc.items():
        mean = (s / n)
        if len(key) == 2:
            k, pk = key; out[k][pk] = mean.to(last[k][pk].dtype)
        else:
            k, mk, pk = key; out[k][mk][pk] = mean.to(last[k][mk][pk].dtype)
    out["soup"] = {"members": [s for s, _ in files], "n": n}
    path = os.path.join(ck_dir, f"{out_name}.pt")
    torch.save(out, path)
    print(f"wrote {path}  (uniform average of {n} checkpoints, steps {files[0][0]}..{files[-1][0]})")

if __name__ == "__main__":
    main()
