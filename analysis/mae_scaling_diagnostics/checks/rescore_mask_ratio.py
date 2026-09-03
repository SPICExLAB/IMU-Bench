#!/usr/bin/env python
"""Fair comparison across the mask-ratio ablation arms.

WHY THIS IS REQUIRED, not optional: the MAE loss is computed ONLY over masked
patches. An arm trained (and validated) at mask_ratio=0.90 is scored on a
strictly harder inference problem than one at 0.50, so their logged val losses
answer different questions and cannot be ranked against each other. A lower
number at 0.50 would mean "easier task", not "better representation".

This rescoes every arm's checkpoint at ONE fixed mask ratio on ONE val set,
with an identical per-batch mask sequence (torch seed reset before each
checkpoint), using the real masked_1d_freq_channel_mask.forward_loss captured
through the production path -- never a reimplementation.

It also reports, per arm, the same regression-to-the-mean diagnostics used for
the reconstruction figures (recon/truth std ratio and correlation over masked
samples) plus the predict-zero baseline, since MSE alone hides whether a model
is reconstructing or just predicting the conditional mean.

Usage:
    python analysis/mae_scaling_diagnostics/checks/rescore_mask_ratio.py \
        --stage stage1 --eval-mask-ratio 0.75 --batches 60
"""
from __future__ import annotations

import argparse, json, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.common import OUT_DIR, REPO_ROOT, load_merged_cfg  # noqa: E402

ARMS = {
    "stage1": [("0.50", "configs/vit1d_ssl_mae_stage1_acconly_mr50_flu.yaml", "runs/vit1d_ssl_mae_stage1_acconly_mr50"),
               ("0.75", "configs/vit1d_ssl_mae_stage1_acconly_flu.yaml",      "runs/vit1d_ssl_mae_stage1_acconly"),
               ("0.90", "configs/vit1d_ssl_mae_stage1_acconly_mr90_flu.yaml", "runs/vit1d_ssl_mae_stage1_acconly_mr90")],
    # 2026-08-20 mask-ratio sweep: 6 arms on the leak-free 514 h raw cache, 30k steps,
    # identical apart from objective.mask_ratio (0.75 = the recipe ablation's R0 run).
    "maskratio": [("0.30", "configs/maskratio/mae_mr30.yaml",        "runs/mae_mr30"),
                  ("0.40", "configs/maskratio/mae_mr40.yaml",        "runs/mae_mr40"),
                  ("0.50", "configs/maskratio/mae_mr50.yaml",        "runs/mae_mr50"),
                  ("0.60", "configs/maskratio/mae_mr60.yaml",        "runs/mae_mr60"),
                  ("0.75", "configs/nhanes_scale/mae_recipe_R0.yaml", "runs/mae_recipe_R0"),
                  ("0.85", "configs/maskratio/mae_mr85.yaml",        "runs/mae_mr85"),
                  ("0.90", "configs/maskratio/mae_mr90.yaml",        "runs/mae_mr90")],
    "stage2": [("0.50", "configs/vit1d_ssl_mae_stage2_accgyro_mr50_flu.yaml", "runs/vit1d_ssl_mae_stage2_accgyro_mr50"),
               ("0.75", "configs/vit1d_ssl_mae_stage2_accgyro_flu.yaml",      "runs/vit1d_ssl_mae_stage2_accgyro"),
               ("0.90", "configs/vit1d_ssl_mae_stage2_accgyro_mr90_flu.yaml", "runs/vit1d_ssl_mae_stage2_accgyro_mr90")],
}
AUG = ("random_rotation", "gaussian_noise", "jitter", "bias_drift", "sensor_dropout",
       "time_crop", "frequency_mask", "gain_jitter", "axis_bias", "time_warp")
SEED = 1234


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(ARMS), required=True)
    ap.add_argument("--eval-mask-ratio", type=float, default=0.75)
    ap.add_argument("--batches", type=int, default=60, help="val batches per arm (0 = full val)")
    ap.add_argument("--ckpt", default="best")
    a = ap.parse_args()

    from imu_lm.data.combined_mae_multidata_loader import make_combined_mae_multidata_loaders
    from imu_lm.models.ViT1D.model import ViT1DDecoder, ViT1DEncoderAltAttn
    import imu_lm.objectives.masked_1d_freq_channel_mask as prod
    from imu_lm.objectives.masked_1d import _patch_mask_to_signal_mask

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arms = ARMS[a.stage]

    # ONE val loader, built from the FIRST arm's config, reused for every arm:
    # all arms in a stage share a cache/source list, so this is the same data.
    cfg = load_merged_cfg(arms[0][1])
    for k in AUG:
        if isinstance(cfg.get(k), dict): cfg[k]["enabled"] = False
    cfg["objective"]["mask_ratio"] = a.eval_mask_ratio     # THE fixed evaluation ratio
    cfg["data"]["num_workers"] = 8
    val = make_combined_mae_multidata_loaders(cfg)["val_loader"]
    print(f"[rescore] stage={a.stage} eval_mask_ratio={a.eval_mask_ratio} "
          f"val_windows={len(val.dataset):,} batches={a.batches or 'all'}", flush=True)

    results = {}
    for ratio, cfg_path, run_dir in arms:
        ck = os.path.join(REPO_ROOT, run_dir, "checkpoints", f"{a.ckpt}.pt")
        if not os.path.exists(ck) or os.path.getsize(ck) == 0:
            print(f"[rescore] arm {ratio}: checkpoint missing/empty ({ck}) -- skipping", flush=True)
            continue
        enc = ViT1DEncoderAltAttn(cfg).to(dev)
        dec = ViT1DDecoder(embed_dim=enc.embed_dim, out_channels=enc.in_channels,
                           target_T=0, cfg=cfg).to(dev)
        st = torch.load(ck, map_location=dev, weights_only=False)
        enc.load_state_dict(st["model"]); dec.load_state_dict(st["decoder"])
        enc.eval(); dec.eval()

        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)   # identical masks per arm
        tot, n = 0.0, 0
        se = sz = 0.0; nm = 0
        s_t = s_t2 = s_p = s_p2 = s_tp = 0.0
        with torch.no_grad():
            for i, (x, y, cvm) in enumerate(val):
                if a.batches and i >= a.batches: break
                x, cvm = x.to(dev), cvm.to(dev)
                grab = {}
                h = dec.register_forward_hook(lambda m, i_, o: grab.__setitem__("rec", o.detach()))
                real = prod._patch_mask_to_signal_mask
                prod._patch_mask_to_signal_mask = lambda pm, *ar, **kw: (
                    grab.__setitem__("pm", pm.detach().clone()), real(pm, *ar, **kw))[1]
                try:
                    loss, _ = prod.forward_loss((x, y, cvm), enc, dec, cfg)
                finally:
                    h.remove(); prod._patch_mask_to_signal_mask = real
                tot += float(loss); n += 1
                rec = grab["rec"]; T = rec.shape[-1]
                sig = _patch_mask_to_signal_mask(grab["pm"], T // enc.patch_size, x.shape[1], enc.patch_size)
                m = sig[:, :, :T].bool() & cvm.unsqueeze(-1).expand(-1, -1, T)
                t, p = x[:, :, :T][m].double(), rec[m].double()
                se += float(((t - p) ** 2).sum()); sz += float((t ** 2).sum()); nm += t.numel()
                s_t += float(t.sum()); s_t2 += float((t*t).sum())
                s_p += float(p.sum()); s_p2 += float((p*p).sum()); s_tp += float((t*p).sum())
        mt, mp = s_t/nm, s_p/nm
        vt, vp = s_t2/nm - mt*mt, s_p2/nm - mp*mp
        cov = s_tp/nm - mt*mp
        corr = cov/((vt*vp) ** 0.5 + 1e-12)
        results[ratio] = {"mse": tot/max(1, n), "masked_mse": se/nm, "predict_zero_mse": sz/nm,
                          "std_ratio": (vp/vt) ** 0.5, "corr": corr,
                          "ckpt_step": st.get("step"), "logged_best_val": st.get("best_val_loss")}
        r = results[ratio]
        print(f"[rescore] arm mask_ratio={ratio}: fixed-ratio MSE={r['mse']:.6f}  "
              f"vs predict-zero {r['predict_zero_mse']:.4f} "
              f"({100*(1-r['masked_mse']/r['predict_zero_mse']):.1f}% better)  "
              f"std_ratio={r['std_ratio']:.3f} corr={r['corr']:.3f}  "
              f"(trained-ratio logged val was {r['logged_best_val']:.6f})", flush=True)
        del enc, dec; torch.cuda.empty_cache()

    out_dir = os.path.join(OUT_DIR, "13_mask_ratio_ablation"); os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{a.stage}_eval_at_{a.eval_mask_ratio}.json")
    json.dump({"stage": a.stage, "eval_mask_ratio": a.eval_mask_ratio,
               "batches": a.batches, "seed": SEED, "results": results}, open(p, "w"), indent=2)
    print(f"\n[rescore] wrote {p}")
    if results:
        best = min(results, key=lambda k: results[k]["mse"])
        print(f"[rescore] BEST at a common eval ratio of {a.eval_mask_ratio}: "
              f"trained-mask_ratio={best} (MSE {results[best]['mse']:.6f})")


if __name__ == "__main__":
    main()
