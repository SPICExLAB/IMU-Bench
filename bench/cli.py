#!/usr/bin/env python3
"""`imu-bench`: one command for running, checking and submitting an encoder.

    imu-bench tasks                                   # the ten task cards: dataset, split, metric
    imu-bench eval  --encoder my_card.yaml            # register a card, then train + score all 20 cells locally
    imu-bench eval  --encoder my_enc --tasks pd_pads --protocols probe        # a subset
    imu-bench eval  --encoder my_enc --slurm          # submit train -> score chains instead (bench.launch)
    imu-bench check   my_enc                          # is the submission complete and protocol-faithful? which track?
    imu-bench package my_enc --out submission/        # results + card + configs, laid out for the Hugging Face PR
    imu-bench leaderboard                             # bench/leaderboard_tracks.json + the two track tables
    imu-bench fetch-weights [harnet10 imagebind_imu ...]  # released weights of the open-track models, where the cards expect them
    imu-bench audit <dir> --subject-regex 'id_(\d+)/' [--acc-cols AccX AccY AccZ]   # duplicate / mirrored participants

An encoder card is a YAML file `{<id>: {...}}` (see examples/plugin/inrtlwrist_plugin.yaml and
imu_lm/models/baselines/plugin_encoder.py). `eval --encoder <file>` copies it to bench/registry/encoders.d/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List

import yaml

from bench.registry_io import REG_DIR, REPO, load_encoders, load_tasks, run_name

PROTOCOLS = ["probe", "finetune"]
CARD_FIELDS = ["name", "params_m", "pretrain_data", "normalize_method"]


def bench_tasks() -> List[str]:
    from bench.leaderboard_tracks import load_spec
    spec = load_spec()
    return [c[0] for c in spec["columns"] if c[0] != "har_mean"] + list(spec["har_mean"])


def register_card(path: str) -> str:
    with open(path) as f:
        cards = yaml.safe_load(f) or {}
    if len(cards) != 1:
        raise SystemExit(f"{path}: an encoder card file holds exactly one encoder, found {len(cards)}")
    enc_id = next(iter(cards))
    dst_dir = os.path.join(REG_DIR, "encoders.d")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"{enc_id}.yaml")
    if os.path.abspath(path) != os.path.abspath(dst):
        shutil.copy(path, dst)
    print(f"[imu-bench] registered {enc_id} -> {os.path.relpath(dst, REPO)}")
    return enc_id


def resolve_tasks(arg: List[str]) -> List[str]:
    if not arg or arg == ["all"]:
        return bench_tasks()
    known = load_tasks()["tasks"]
    bad = [t for t in arg if t not in known]
    if bad:
        raise SystemExit(f"unknown task(s) {bad}; see `imu-bench tasks`")
    return arg


# ---------------------------------------------------------------------------
def cmd_tasks(a) -> None:
    tasks = load_tasks()["tasks"]
    print(f"{'task':28s} {'dataset':34s} {'primary metric':26s} reported split")
    for t in bench_tasks():
        c = tasks[t]
        m = c["metrics"][c["metrics"]["primary"]]
        ds = c.get("dataset", {})
        ds = ds.get("name", "") if isinstance(ds, dict) else str(ds)
        arrow = "higher" if m.get("higher_is_better", True) else "lower"
        print(f"{t:28s} {ds[:34]:34s} {c['metrics']['primary'] + ' (' + arrow + ')':26s} {c.get('eval_split', '')}")


def cmd_eval(a) -> None:
    from bench.make_config import write_config
    enc = register_card(a.encoder) if os.path.isfile(a.encoder) else a.encoder
    encoders = load_encoders()
    if enc not in encoders:
        raise SystemExit(f"encoder {enc!r} is not registered; pass a card file or add it to bench/registry/")
    tasks = load_tasks()
    cells = [(t, p) for t in resolve_tasks(a.tasks) for p in a.protocols]
    if a.slurm:
        from bench import launch
        for t, p in cells:
            name, path = launch.train_script(t, enc, p, a.seed, tasks, encoders)
            jid = launch._sbatch(path, a.dry_run)
            launch._log_submission(jid, "train+score", t, enc, p, a.seed, name, path)
        return
    py = sys.executable
    for t, p in cells:
        res = os.path.join(REPO, "bench", "results", t, f"{enc}__{p}__s{a.seed}.json")
        if os.path.exists(res) and not a.force:
            print(f"[imu-bench] {t}/{p}: result exists, skipping (--force to rerun)")
            continue
        name, cfg_path = write_config(t, enc, p, a.seed, tasks=tasks, encoders=encoders)
        train = [py, os.path.join(REPO, "scripts", "run_pretrain.py"), "--config", os.path.join(REPO, "configs", "base.yaml"),
                 "--model-config", cfg_path, "--run-name", name]
        score = [py, "-m", "bench.score", "--task", t, "--encoder", enc, "--protocol", p, "--seed", str(a.seed)]
        for cmd in (train, score):
            print("[imu-bench] $", " ".join(os.path.relpath(c, REPO) if c.startswith(REPO) else c for c in cmd))
            if not a.dry_run:
                subprocess.run(cmd, cwd=REPO, check=True)


def _effective(c: Dict[str, Any]) -> Dict[str, Any]:
    """Make two flattened configs comparable on what the trainer actually does: a missing normalize block
    is the trainer's default per-window z-score, and under z-score a constant unit gain on the input
    (data.acc_scale / gyro_scale) cancels."""
    c = dict(c)
    if not any(k.startswith("preprocess.normalize") for k in c):
        c.update({"preprocess.normalize.enabled": True, "preprocess.normalize.method": "zscore",
                  "preprocess.normalize.eps": 1e-6})
    if c.get("preprocess.normalize.enabled") and c.get("preprocess.normalize.method", "zscore") == "zscore":
        c.pop("data.acc_scale", None)
        c.pop("data.gyro_scale", None)
    return c


def check(enc: str, verbose: bool = True) -> Dict[str, Any]:
    """Completeness, protocol fidelity and card fields of one encoder's submission."""
    from bench.make_config import build_config
    encoders, tasks = load_encoders(), load_tasks()
    if enc not in encoders:
        raise SystemExit(f"encoder {enc!r} is not registered")
    card = encoders[enc]
    problems, cells = [], []

    def flat(d, p=""):
        out = {}
        if isinstance(d, dict):
            for k, v in d.items():
                out.update(flat(v, f"{p}.{k}" if p else k))
        else:
            out[p] = d
        return out

    for t in bench_tasks():
        for p in PROTOCOLS:
            f = os.path.join(REPO, "bench", "results", t, f"{enc}__{p}__s42.json")
            if not os.path.exists(f):
                problems.append(f"missing result {t}/{p}")
                continue
            d = json.load(open(f))
            if d.get("status") != "ok":
                problems.append(f"{t}/{p}: status {d.get('status')!r}")
            if d.get("eval_split") != tasks["tasks"][t].get("eval_split", d.get("eval_split")):
                problems.append(f"{t}/{p}: evaluated on {d.get('eval_split')!r}, the card reports "
                                f"{tasks['tasks'][t].get('eval_split')!r}")
            gen = os.path.join(REPO, "bench", "generated", "configs", run_name(t, enc, p, 42) + ".yaml")
            if os.path.exists(gen):
                _, now = build_config(t, enc, p, 42, tasks, encoders)
                old, new = _effective(flat(yaml.safe_load(open(gen)))), _effective(flat(now))
                diff = sorted(k for k in set(old) | set(new)
                              if old.get(k) != new.get(k) and not k.startswith(("run", "wandb", "paths", "bench")))
                if diff:
                    problems.append(f"{t}/{p}: trained config differs from the current task card in {diff[:4]}")
            cells.append(f)
    for k in CARD_FIELDS:
        if not card.get(k):
            problems.append(f"card field `{k}` is empty")
    tracks = ["open"]
    if card.get("track") == "controlled" or card.get("group") in ("ours", "ssl_baseline"):
        tracks.insert(0, "controlled")
    if not (card.get("weights_url") or card.get("run") or card.get("weights")):
        problems.append("no public weights: set `weights_url` (open track requires public weights)")
    rep = dict(encoder=enc, cells=len(cells), of=2 * len(bench_tasks()), problems=problems, eligible_tracks=tracks,
               ok=not problems)
    if verbose:
        print(f"[imu-bench] {enc}: {rep['cells']}/{rep['of']} cells, eligible for {', '.join(tracks)} track")
        for x in problems:
            print("  -", x)
        print("[imu-bench] OK, ready to package" if rep["ok"] else f"[imu-bench] {len(problems)} problem(s)")
    return rep


def cmd_check(a) -> None:
    rep = check(a.encoder)
    sys.exit(0 if rep["ok"] else 1)


def cmd_package(a) -> None:
    rep = check(a.encoder)
    if not rep["ok"] and not a.allow_incomplete:
        raise SystemExit("[imu-bench] fix the problems above first (or --allow-incomplete for a draft)")
    out = os.path.abspath(a.out)
    enc, card = a.encoder, load_encoders()[a.encoder]
    for t in bench_tasks():
        for p in PROTOCOLS:
            for sub, name in (("results", f"{enc}__{p}__s42.json"),):
                src = os.path.join(REPO, "bench", sub, t, name)
                if os.path.exists(src):
                    os.makedirs(os.path.join(out, sub, t), exist_ok=True)
                    shutil.copy(src, os.path.join(out, sub, t, name))
            gen = os.path.join(REPO, "bench", "generated", "configs", run_name(t, enc, p, 42) + ".yaml")
            if os.path.exists(gen):
                os.makedirs(os.path.join(out, "configs"), exist_ok=True)
                shutil.copy(gen, os.path.join(out, "configs"))
    os.makedirs(os.path.join(out, "registry"), exist_ok=True)
    with open(os.path.join(out, "registry", f"{enc}.yaml"), "w") as f:
        yaml.safe_dump({enc: card}, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(out, "check.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(f"[imu-bench] wrote {out}\n  submit it with:\n"
          f"  python -c \"from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='YiquanLi/IMU-Bench-results', "
          f"repo_type='dataset', folder_path='{out}', create_pr=True, commit_message='Add {enc}')\"")


def cmd_leaderboard(a) -> None:
    from bench import leaderboard_tracks
    argv = ["--quiet"] if a.quiet else []
    if a.site:
        argv += ["--site", a.site]
    if a.readme:
        argv += ["--readme", a.readme]
    sys.argv = ["leaderboard_tracks"] + argv
    leaderboard_tracks.main()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "audit":           # its own parser (bench/audit.py)
        from bench.audit import main as audit_main
        return audit_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "fetch-weights":   # bench/fetch_weights.py
        from bench.fetch_weights import main as fetch_main
        return fetch_main(sys.argv[2:])
    ap = argparse.ArgumentParser(prog="imu-bench", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks", help="list the ten task cards").set_defaults(fn=cmd_tasks)
    e = sub.add_parser("eval", help="train and score an encoder on the task cards")
    e.add_argument("--encoder", required=True, help="registered encoder id, or a card .yaml to register")
    e.add_argument("--tasks", nargs="*", default=["all"])
    e.add_argument("--protocols", nargs="*", default=PROTOCOLS, choices=PROTOCOLS)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--slurm", action="store_true", help="submit SLURM train -> score chains instead of running here")
    e.add_argument("--force", action="store_true", help="rerun cells that already have a result")
    e.add_argument("--dry-run", action="store_true")
    e.set_defaults(fn=cmd_eval)
    c = sub.add_parser("check", help="check a submission for completeness and protocol fidelity")
    c.add_argument("encoder")
    c.set_defaults(fn=cmd_check)
    p = sub.add_parser("package", help="lay out results, card and configs for a leaderboard pull request")
    p.add_argument("encoder")
    p.add_argument("--out", default="submission")
    p.add_argument("--allow-incomplete", action="store_true")
    p.set_defaults(fn=cmd_package)
    lb = sub.add_parser("leaderboard", help="regenerate the two-track leaderboard")
    lb.add_argument("--site", default=None)
    lb.add_argument("--readme", default=None)
    lb.add_argument("--quiet", action="store_true")
    lb.set_defaults(fn=cmd_leaderboard)
    sub.add_parser("audit", help="find duplicated / mirrored participants in a dataset release (see bench/audit.py)")
    sub.add_parser("fetch-weights", help="download the released weights of the open-track models (bench/fetch_weights.py)")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
