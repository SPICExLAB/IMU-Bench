#!/usr/bin/env python
"""Summarise the HumanML3D CLIP-alignment tuning sweep.

Reads analysis/retrieval_tune/{val,test}/<arm>.json (written by
launch/launch_slurm_retrieval_tune.sh via scripts/eval_retrieval_alignment.py)
plus each run's best val loss / step, and prints one markdown table sorted by
val G=100 IMU->Text MRR (the selection metric). Main protocol = one caption
per item (caption_mode=first), matching Experiments_table.md §13d.

    python scripts/summarize_retrieval_tune.py [analysis/retrieval_tune] [--sort mrr_i2t|mrr_t2i|r1_i2t]
"""
import glob, json, os, sys, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = {  # Section-13d reference rows (val split, first-caption, G=100 / full)
    "UniMTS 22-joint (ref)": {"g100": ((40.27, 79.67, 89.43, 57.2), (2.07, 7.97, 15.07, 7.29)),
                             "full": ((9.36, 34.92, 49.96, 22.45), (0.56, 1.22, 1.78, 1.35))},
    "Ours parent (13d row)": {"g100": ((39.42, 73.88, 84.25, 54.85), (37.54, 72.02, 82.4, 53.04)),
                             "full": ((11.11, 35.34, 47.79, 22.71), (10.44, 29.67, 44.56, 20.48))},
}

def _m(d):
    return (d["R@1"], d["R@5"], d["R@10"], d["MRR"])

def load(path):
    j = json.load(open(path))
    p = j["protocols"]["first"]
    g = p["fixed"]["100"]
    return {"g100": (_m(g["imu2text"]), _m(g["text2imu"])),
            "full": (_m(p["full"]["imu2text"]), _m(p["full"]["text2imu"])),
            "medr_full": (p["full"]["imu2text"].get("MedR"), p["full"]["text2imu"].get("MedR")),
            "step": j["meta"].get("ckpt_step"), "n": j["n_items"]}

def best_val(run):
    f = os.path.join(ROOT, "runs", run, "logs", "metrics.txt")
    if not os.path.exists(f):
        return None, None, None
    best = (None, None, None)
    for line in open(f):
        if "split=val" not in line:
            continue
        kv = dict(x.split("=", 1) for x in line.split() if "=" in x)
        v = float(kv["loss"])
        if best[0] is None or v < best[0]:
            best = (v, int(kv["step"]), float(kv.get("acc_imu2text", "nan")) * 100)
    return best

def fmt4(t):
    return " / ".join(f"{x:.1f}" for x in t)

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    base = args[0] if args else os.path.join(ROOT, "analysis", "retrieval_tune")
    sort_key = "mrr_i2t"
    for a in sys.argv[1:]:
        if a.startswith("--sort"):
            sort_key = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
    rows = []
    for f in sorted(glob.glob(os.path.join(base, "val", "*.json"))):
        arm = os.path.splitext(os.path.basename(f))[0]
        v = load(f)
        tf = os.path.join(base, "test", arm + ".json")
        t = load(tf) if os.path.exists(tf) else None
        bl, bs, bacc = best_val("rtune_" + arm)
        rows.append((arm, v, t, bl, bs, bacc))
    idx = {"mrr_i2t": lambda r: r[1]["g100"][0][3], "mrr_t2i": lambda r: r[1]["g100"][1][3],
           "r1_i2t": lambda r: r[1]["g100"][0][0]}[sort_key]
    rows.sort(key=idx, reverse=True)

    print("### Tuning sweep — VAL split (selection set; 1,197 clips), first-caption protocol, %")
    print("| arm | best val loss (step) | G=100 IMU→Text R@1/5/10/MRR | G=100 Text→IMU R@1/5/10/MRR | full IMU→Text R@1/5/10/MRR | full Text→IMU R@1/5/10/MRR |")
    print("|---|---|---|---|---|---|")
    for name, ref in REF.items():
        print(f"| *{name}* | — | {fmt4(ref['g100'][0])} | {fmt4(ref['g100'][1])} | {fmt4(ref['full'][0])} | {fmt4(ref['full'][1])} |")
    for arm, v, t, bl, bs, bacc in rows:
        bls = f"{bl:.4f} ({bs})" if bl is not None else "—"
        print(f"| {arm} | {bls} | {fmt4(v['g100'][0])} | {fmt4(v['g100'][1])} | {fmt4(v['full'][0])} | {fmt4(v['full'][1])} |")
    print()
    print("### Same arms — TEST split (held-out confirmation; 4,384 clips), first-caption protocol, %")
    print("| arm | G=100 IMU→Text R@1/5/10/MRR | G=100 Text→IMU R@1/5/10/MRR | full IMU→Text R@1/5/10/MRR (MedR) | full Text→IMU R@1/5/10/MRR (MedR) |")
    print("|---|---|---|---|---|")
    for arm, v, t, bl, bs, bacc in rows:
        if t is None:
            print(f"| {arm} | (no test json) | | | |"); continue
        print(f"| {arm} | {fmt4(t['g100'][0])} | {fmt4(t['g100'][1])} | {fmt4(t['full'][0])} ({t['medr_full'][0]}) | {fmt4(t['full'][1])} ({t['medr_full'][1]}) |")

if __name__ == "__main__":
    main()
