#!/usr/bin/env python
"""Generate the HumanML3D CLIP-alignment tuning sweep (2026-08-15).

Parent recipe = configs/vit1d_clip_humanml3d_flu_finetune_mae_unimtssim_texttrainable_lr5e-5_i2t.yaml
(the Section-13 "Ours pretrained (MAE init, finetune)" row: 39.4/73.9/84.3/57.2
i2t at G=100 vs UniMTS 22-joint 40.3/79.7/89.4/57.2). That run overfits hard
(train in-batch acc ~88% vs val 45%, train loss 0.42 vs val 2.04, best val at
step 6000/8000), so the sweep is one-factor-at-a-time around the parent along
LR / text-tower LR / regularisation / batch size / schedule / temperature axes.

Every arm = parent YAML + a small dict of overrides, written to
configs/retrieval_tune/<arm>.yaml. Run name = rtune_<arm>.
"""
import copy, os, sys, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.join(ROOT, "configs", "vit1d_clip_humanml3d_flu_finetune_mae_unimtssim_texttrainable_lr5e-5_i2t.yaml")
OUT_DIR = os.path.join(ROOT, "configs", "retrieval_tune")

def _set(d, path, val):
    cur = d
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = val

# name -> list of (path, value) overrides.  O = objective.imu_text_clip_flu
O = ["objective", "imu_text_clip_flu"]
T = ["trainer"]
ARMS = {
    # ── controls ────────────────────────────────────────────────────────────
    "base_s42":        [],
    "base_s43":        [(T + ["seed"], 43)],
    # ── A. learning rate ────────────────────────────────────────────────────
    "lr2e-5":          [(T + ["optim", "lr"], 2.0e-5)],
    "lr1e-4":          [(T + ["optim", "lr"], 1.0e-4)],
    # ── B. text tower ───────────────────────────────────────────────────────
    "textmult0.1":     [(O + ["text_lr_mult"], 0.1)],
    "textmult0.3":     [(O + ["text_lr_mult"], 0.3)],
    "textlast4":       [(O + ["text_train_last_n_layers"], 4)],
    "textfrozen":      [(O + ["freeze_text_encoder"], True)],
    "headmult10":      [(O + ["head_lr_mult"], 10.0)],
    # ── C. regularisation ───────────────────────────────────────────────────
    "wd0.05":          [(T + ["optim", "weight_decay"], 0.05)],
    "wd0.1_nodecay1d": [(T + ["optim", "weight_decay"], 0.1), (O + ["no_decay_norm_bias"], True)],
    "droppath0.1":     [(["vit1d", "encoder", "drop_path_rate"], 0.1)],
    "droppath0.2":     [(["vit1d", "encoder", "drop_path_rate"], 0.2)],
    "dropout0.1":      [(["vit1d", "encoder", "hidden_dropout_prob"], 0.1),
                        (["vit1d", "encoder", "attention_probs_dropout_prob"], 0.1)],
    "llrd0.75":        [(O + ["layer_decay"], 0.75)],
    "llrd0.85":        [(O + ["layer_decay"], 0.85)],
    "aug_noise0.2":    [(["gaussian_noise"], {"enabled": True, "sigma": 0.2})],
    "aug_time":        [(["time_crop"], {"enabled": True, "min_crop_ratio": 0.8}),
                        (["time_warp"], {"enabled": True, "max_warp": 0.1, "smooth_sigma": 20})],
    # ── D. batch size (more in-batch negatives) ─────────────────────────────
    "bs128":           [(["data", "batch_size"], 128)],
    "bs256":           [(["data", "batch_size"], 256)],
    "bs128_lr1e-4":    [(["data", "batch_size"], 128), (T + ["optim", "lr"], 1.0e-4)],
    # ── E. schedule ─────────────────────────────────────────────────────────
    "steps16k":        [(T + ["max_steps"], 16000)],
    "steps4k":         [(T + ["max_steps"], 4000)],
    # ── F. temperature ──────────────────────────────────────────────────────
    "tau0.1_fixed":    [(O + ["init_temperature"], 0.1), (O + ["freeze_logit_scale"], True)],
    "tau0.03":         [(O + ["init_temperature"], 0.03)],
    # ── ROUND 2 (2026-08-15 evening; round 1 = no single knob beats base 55.0±2 MRR) ──
    "base_s44":        [(T + ["seed"], 44)],
    "base_s45":        [(T + ["seed"], 45)],
    "lr1e-4_wd0.1nd":  [(T + ["optim", "lr"], 1.0e-4), (T + ["optim", "weight_decay"], 0.1), (O + ["no_decay_norm_bias"], True)],
    "lr1e-4_tau0.1fixed": [(T + ["optim", "lr"], 1.0e-4), (O + ["init_temperature"], 0.1), (O + ["freeze_logit_scale"], True)],
    "bs32":            [(["data", "batch_size"], 32)],
    "bs32_steps16k":   [(["data", "batch_size"], 32), (T + ["max_steps"], 16000)],
    # stronger text tower (CLIP ViT-L/14 text transformer, 768-d) -- a model change, not a
    # hyper-parameter; UniMTS uses ViT-B/32 text, so flag it if it ends up in the table
    "textL14":         [(O + ["clip_model"], "openai/clip-vit-large-patch14"), (O + ["clip_embed_dim"], 768)],
    # parent recipe with every 500-step checkpoint kept -> post-hoc uniform soups +
    # per-step val retrieval (launch_slurm_retrieval_tune_soup.sh)
    "soupsrc_s42":     [(["logging", "save_step_checkpoints"], True)],
    "soupsrc_s43":     [(["logging", "save_step_checkpoints"], True), (T + ["seed"], 43)],
    # ── ROUND 3: around the only round-2 positive (lr1e-4 + tau 0.1 fixed = 56.5 val MRR) ──
    "lr1e-4_tau0.1fixed_s43": [(T + ["optim", "lr"], 1.0e-4), (O + ["init_temperature"], 0.1), (O + ["freeze_logit_scale"], True), (T + ["seed"], 43)],
    "lr1e-4_tau0.1fixed_s44": [(T + ["optim", "lr"], 1.0e-4), (O + ["init_temperature"], 0.1), (O + ["freeze_logit_scale"], True), (T + ["seed"], 44)],
    "lr1e-4_tau0.1fixed_soupsrc": [(T + ["optim", "lr"], 1.0e-4), (O + ["init_temperature"], 0.1), (O + ["freeze_logit_scale"], True), (["logging", "save_step_checkpoints"], True)],
    "lr2e-4_tau0.1fixed": [(T + ["optim", "lr"], 2.0e-4), (O + ["init_temperature"], 0.1), (O + ["freeze_logit_scale"], True)],
    "lr1e-4_tau0.15fixed": [(T + ["optim", "lr"], 1.0e-4), (O + ["init_temperature"], 0.15), (O + ["freeze_logit_scale"], True)],
    "lr1e-4_tau0.07fixed": [(T + ["optim", "lr"], 1.0e-4), (O + ["init_temperature"], 0.07), (O + ["freeze_logit_scale"], True)],
    "lr1e-4_tau0.1fixed_steps16k_soupsrc": [(T + ["optim", "lr"], 1.0e-4), (O + ["init_temperature"], 0.1), (O + ["freeze_logit_scale"], True), (T + ["max_steps"], 16000), (["logging", "save_step_checkpoints"], True), (["logging", "ckpt_every_steps"], 1000)],
    "lr1e-4_tau0.1fixed_steps16k_soupsrc_s43": [(T + ["optim", "lr"], 1.0e-4), (O + ["init_temperature"], 0.1), (O + ["freeze_logit_scale"], True), (T + ["max_steps"], 16000), (["logging", "save_step_checkpoints"], True), (["logging", "ckpt_every_steps"], 1000), (T + ["seed"], 43)],
    "lr1e-4_tau0.1fixed_steps16k_soupsrc_s44": [(T + ["optim", "lr"], 1.0e-4), (O + ["init_temperature"], 0.1), (O + ["freeze_logit_scale"], True), (T + ["max_steps"], 16000), (["logging", "save_step_checkpoints"], True), (["logging", "ckpt_every_steps"], 1000), (T + ["seed"], 44)],
}

def main(names=None):
    with open(PARENT) as fh:
        parent = yaml.safe_load(fh)
    os.makedirs(OUT_DIR, exist_ok=True)
    names = names or list(ARMS)
    for name in names:
        cfg = copy.deepcopy(parent)
        for path, val in ARMS[name]:
            _set(cfg, path, val)
        out = os.path.join(OUT_DIR, f"{name}.yaml")
        with open(out, "w") as fh:
            fh.write(f"# AUTO-GENERATED by scripts/gen_retrieval_tune_configs.py -- arm '{name}'\n"
                     f"# parent: {os.path.relpath(PARENT, ROOT)}\n"
                     f"# overrides: {ARMS[name]}\n")
            yaml.safe_dump(cfg, fh, sort_keys=False)
        print(out)

if __name__ == "__main__":
    main(sys.argv[1:] or None)
