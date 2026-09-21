#!/usr/bin/env python3
"""Extra full-body pose metrics for finished IMU-Bench cells, under the MobilePoser evaluation protocol.

    python -m bench.rescore_full_body_extra                       # every full_body_pose_nymeria cell of the paper
    python -m bench.rescore_full_body_extra --cells mae_5src__probe --max-batches 2     # smoke test

Why. The benchmark's primary metric is the root-relative MPJPE (mm) over the 23 predicted joints. The task-specific
rows (MobilePoser, IMUPoser) were scored by MobilePoser's FullMotionEvaluator (baselines/full_body_pose/SimplePoser/
evaluate_nymeria.py), whose convention differs: the joints in `joint_set.ignored` = [0, 7, 8, 10, 11, 20, 22, 23]
(root, ankles, feet, left wrist, hands) are set to identity in BOTH prediction and truth, positions come from forward
kinematics with the mean shape, and errors are averaged over all 24 joints (the root contributes zero). This script
re-evaluates our checkpoints with exactly that convention so that joint error, global angle error and mesh error are
comparable across every row of the table:

    mp_joint_err_cm         FullMotionEvaluator errs[0]: joint position error, 24 joints
    mp_global_ang_err_deg   errs[3]: global joint angle error, 24 joints
    mp_local_ang_err_deg    errs[2]: local joint angle error, 24 joints
    mp_mesh_err_cm          errs[1]: SMPL vertex error, 6,890 vertices
    local_ang_err_deg /     the same two angle errors in the BENCHMARK's convention: over the 23 predicted joints,
    global_ang_err_deg      nothing ignored (no counterpart for the external systems, which do not predict 7 joints)

It also recomputes the primary MPJPE as a self-check. Metrics are ADDED to the existing result JSON; the primary
metric and the checkpoint are untouched. Poses are evaluated at the decoder's rate (32 patches x 4 sub-patches per
10.24 s window, 12.5 Hz); the external systems run at 30 fps on the same sessions and window length.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from bench.leaderboard_paper import CONTROLLED                       # noqa: E402
from bench.schema import RESULTS_ROOT                                # noqa: E402

TASK = "full_body_pose_nymeria"
IGNORED = [0, 7, 8, 10, 11, 20, 22, 23]      # SimplePoser/config.py joint_set.ignored (Nymeria adaptation)
CELLS = [f"{e}__{p}" for e, _, _, _ in CONTROLLED for p in ("probe", "finetune")] + \
        ["control_random__probe", "control_raw__probe", "scratch__scratch"]


def evaluate(run_dir: str, ckpt_path: str, max_batches: int = 0):
    import numpy as np
    import torch
    from torch.amp import autocast
    import core.articulate as art
    from core.paths import Paths
    from imu_lm.data.nymeria_real_wrist_full_body_loader_FLU import make_nymeria_real_wrist_full_body_loaders_flu
    from imu_lm.data.windowing import compute_T_and_hop
    from imu_lm.models.ViT1D.model import ViT1DPoseDecoder
    from imu_lm.models.baselines.inertia1_encoder import build_encoder
    from imu_lm.objectives.pose_1d import _assemble_full_R, _rot6d_to_matrix
    from imu_lm.utils.helpers import cfg_get
    from bench.score.common import acc_only_slice, load_ckpt_meta

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_ckpt_meta(ckpt_path)
    ckpt, cfg = meta["_obj"], meta["cfg"]
    obj_type = cfg_get(cfg, ["objective", "type"], "pose_probe")
    ob = cfg_get(cfg, ["objective", obj_type], {}) or {}
    J = int(ob.get("num_joints", 5))
    K = int((cfg_get(cfg, ["vit1d", "pose_decoder"], {}) or {}).get("sub_patches", 1))
    encoder = build_encoder(cfg).to(device); encoder.load_state_dict(ckpt["model"], strict=True); encoder.eval()
    T, _ = compute_T_and_hop(cfg)
    decoder = ViT1DPoseDecoder(embed_dim=encoder.embed_dim, num_patches_per_channel=T // int(cfg_get(cfg, ["vit1d", "patch_size"], 16)),
                               in_channels=encoder.in_channels, num_joints=J, cfg=cfg, sub_patches=K).to(device)
    decoder.load_state_dict(ckpt["pose_decoder"], strict=True); decoder.eval()
    root = ob.get("val_pkl_root") or ob.get("train_pkl_root") or ob.get("pkl_root")
    files_cfg = ob.get("val_files")
    if files_cfg is None or files_cfg == "all":
        val_files = sorted(glob.glob(os.path.join(root, "*.pkl")))
    elif isinstance(files_cfg, list):
        val_files = [f if os.path.isabs(f) else os.path.join(root, f) for f in files_cfg]
    else:
        val_files = [files_cfg if os.path.isabs(files_cfg) else os.path.join(root, files_cfg)]
    loaders = make_nymeria_real_wrist_full_body_loaders_flu(cfg=cfg, pkl_root="/", train_files=[], val_files=val_files,
                                                            add_gravity=bool(ob.get("add_gravity", True)))
    val_loader = loaders["val_loader"]
    bm = art.model.ParametricModel(Paths.SMPL_FILE, device=device)
    amp = bool(cfg_get(cfg, ["trainer", "amp"], False)) and device.type == "cuda"
    eye = torch.eye(3, device=device)
    deg = art.math.radian_to_degree
    S = {k: 0.0 for k in ("mpjpe", "lae23", "gae23", "je", "ve", "lae", "gae")}
    n_pose = n_win = nb = 0
    CH = 8192                                     # poses per FK/mesh chunk (6,890 vertices each)
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            x, y_full, gt_pos = batch[0], batch[1], batch[2]
            x = acc_only_slice(x.to(device).float(), cfg); y_full = y_full.to(device).float(); gt_pos = gt_pos.to(device)
            with autocast("cuda", enabled=amp):
                tokens, _ = encoder.forward_tokens(x)
                pred = decoder(tokens)
            pred = pred.float()
            B, Pt, _ = pred.shape
            JL = y_full.shape[-1] // (K * 6)
            Rp = _assemble_full_R(pred, y_full, B, Pt, K, J, JL).reshape(-1, JL, 3, 3)          # GT root spliced in
            Rt = _rot6d_to_matrix(y_full.reshape(B, Pt, K, JL, 6)).reshape(-1, JL, 3, 3)
            gtp = gt_pos.reshape(-1, JL, 3).float()
            for s in range(0, Rp.shape[0], CH):
                p, t, g = Rp[s:s + CH], Rt[s:s + CH], gtp[s:s + CH]
                n = p.shape[0]
                # --- benchmark convention: 23 predicted joints, nothing ignored -------------------------------
                gp, jp = bm.forward_kinematics(p.reshape(n, -1))[:2]
                gt_, _ = bm.forward_kinematics(t.reshape(n, -1))[:2]
                jp = jp.to(device); rel = jp - jp[:, 0:1]
                S["mpjpe"] += float((rel[:, 1:JL] - g[:, 1:JL]).norm(dim=-1).sum()) / (JL - 1) * 1000.0
                S["lae23"] += float(deg(art.math.angle_between(p[:, 1:].reshape(-1, 3, 3), t[:, 1:].reshape(-1, 3, 3))).sum()) / (JL - 1)
                S["gae23"] += float(deg(art.math.angle_between(gp.to(device)[:, 1:].reshape(-1, 3, 3), gt_.to(device)[:, 1:].reshape(-1, 3, 3))).sum()) / (JL - 1)
                # --- MobilePoser protocol: ignored joints -> identity in both, mean shape, 24-joint means ------
                pm, tm = p.clone(), t.clone()
                pm[:, IGNORED] = eye; tm[:, IGNORED] = eye
                Gp, Jp, Vp = bm.forward_kinematics(pm, calc_mesh=True)
                Gt, Jt, Vt = bm.forward_kinematics(tm, calc_mesh=True)
                S["je"] += float((Jp - Jt).norm(dim=2).mean(dim=1).sum()) * 100.0
                S["ve"] += float((Vp - Vt).norm(dim=2).mean(dim=1).sum()) * 100.0
                S["lae"] += float(deg(art.math.angle_between(pm.reshape(-1, 3, 3), tm.reshape(-1, 3, 3))).reshape(n, -1).mean(dim=1).sum())
                S["gae"] += float(deg(art.math.angle_between(Gp.reshape(-1, 3, 3), Gt.reshape(-1, 3, 3))).reshape(n, -1).mean(dim=1).sum())
                n_pose += n
            n_win += B; nb += 1
            if max_batches and nb >= max_batches:
                break
    m = {k: v / n_pose for k, v in S.items()}
    return ({"mpjpe_mm_recomputed": m["mpjpe"], "local_ang_err_deg": m["lae23"], "global_ang_err_deg": m["gae23"],
             "mp_joint_err_cm": m["je"], "mp_mesh_err_cm": m["ve"], "mp_local_ang_err_deg": m["lae"],
             "mp_global_ang_err_deg": m["gae"]},
            {"n_windows": n_win, "n_poses": n_pose, "n_val_windows_total": len(val_loader.dataset), "step": int(meta["step"]),
             "max_batches": max_batches, "amp": amp})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="*", default=CELLS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-batches", type=int, default=0, help="smoke test: prefix of the val batches; result JSON is NOT updated")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for cell in a.cells:
        path = os.path.join(RESULTS_ROOT, TASK, f"{cell}__s{a.seed}.json")
        if not os.path.exists(path):
            print(f"[skip] {cell}: no result"); continue
        r = json.load(open(path))
        if r.get("status") != "ok":
            print(f"[skip] {cell}: status {r.get('status')}"); continue
        if "mp_joint_err_cm" in r["metrics"] and not a.force and not a.max_batches:
            print(f"[have] {cell}"); continue
        run_dir = os.path.join(REPO, "runs", r["run"]); ck = os.path.join(run_dir, "checkpoints", r["checkpoint"])
        t0 = time.time()
        m, info = evaluate(run_dir, ck, a.max_batches)
        print(f"[{cell}] {info['n_windows']} windows / {info['n_poses']} poses in {time.time() - t0:.0f}s | reported mpjpe "
              f"{r['metrics']['mpjpe_mm']:.2f} recomputed {m['mpjpe_mm_recomputed']:.2f} | " +
              " ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "mpjpe_mm_recomputed"), flush=True)
        if a.max_batches:
            continue
        r["metrics"].update({k: float(v) for k, v in m.items()})
        r.setdefault("provenance", {}).setdefault("info", {})["extra_metrics"] = dict(
            info, scorer="bench.rescore_full_body_extra", ignored_joints=IGNORED, date=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        json.dump(r, open(path, "w"), indent=2)


if __name__ == "__main__":
    main()
