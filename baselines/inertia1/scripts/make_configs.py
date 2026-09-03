"""Generate the downstream configs for the Inertia-1 external baseline.

Each config is derived from the corresponding *existing* MAE (acc-only where
available) config so that data, split, window, decoder/head, optimiser and
early-stopping recipe are byte-identical to the rows already in
Experiments_table.md; ONLY the encoder side changes:

    vit1d.encoder / shared_kernel  ->  removed
    inertia1: {variant, preset: small, input_hz, acc_scale_to_g, ...}
    objective.<type>.pretrained_run -> inertia1_<variant>_small_nhanes
    data.acc_only: true + acc sensor_columns   (Inertia-1 is accelerometer-only,
                                                like the mae_acconly ablation rows)
    preprocess.filter/normalize.enabled: false (raw g-scale input, as in
                                                Inertia-1's own pipeline)

Writes configs/inertia1/<run-name>.yaml and prints the run names.
"""
from __future__ import annotations

import copy
import os
import sys

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CFG = os.path.join(REPO, "configs")
OUT = os.path.join(CFG, "inertia1")
G = 9.80665

VARIANTS = {"patchtst": "inertia1_patchtst_small_nhanes", "ar_transformer": "inertia1_art_small_nhanes"}
SHORT = {"patchtst": "patchtst", "ar_transformer": "art"}

# (run-name template, template config, extra tweaks)
TASKS = [
    # 1. arm pose (Nymeria, 5 joints)
    ("vit1d_arm_pose_probe_nymeria_inertia1_{v}", "vit1d_arm_pose_probe_nymeria_mae_acconly_multidata_flu.yaml", {}),
    ("vit1d_arm_pose_finetune_nymeria_inertia1_{v}", "vit1d_arm_pose_finetune_nymeria_mae_multidata_flu.yaml", {}),
    # 2. full-body pose (Nymeria, 23 joints)
    ("vit1d_full_body_pose_probe_nymeria_inertia1_{v}", "vit1d_full_body_pose_probe_nymeria_mae_acconly_multidata_flu.yaml", {"fixed_budget": 100000}),
    ("vit1d_full_body_pose_finetune_nymeria_inertia1_{v}", "vit1d_full_body_pose_finetune_nymeria_mae_multidata_flu.yaml", {}),
    # 3. odometry world-frame finetune (probe is script-based on the pretrain run)
    ("vit1d_odom_worldframe_finetune_nymeria_inertia1_{v}", "vit1d_odom_worldframe_finetune_nymeria_mae_multidata_flu.yaml", {}),
    # 4. microgesture (OpenWatch, paper protocol windowed + posture-stratified)
    ("vit1d_gesture_probe_openwatch_6class_inertia1_{v}_windowed_stratified",
     "vit1d_gesture_probe_openwatch_6class_mae_acconly_multidata_flu_windowed_stratified.yaml", {"openwatch": True}),
    ("vit1d_gesture_finetune_openwatch_6class_inertia1_{v}_windowed_stratified",
     "vit1d_gesture_finetune_openwatch_6class_mae_multidata_flu_windowed_stratified.yaml", {"openwatch": True}),
    ("vit1d_gesture_probe_openwatch_6class_inertia1_{v}_windowed_stratified_aug",
     "vit1d_gesture_probe_openwatch_6class_mae_acconly_multidata_flu_windowed_stratified_aug.yaml", {"openwatch": True}),
    ("vit1d_gesture_finetune_openwatch_6class_inertia1_{v}_windowed_stratified_aug",
     "vit1d_gesture_finetune_openwatch_6class_mae_multidata_flu_windowed_stratified_aug.yaml", {"openwatch": True}),
    # 5. language alignment (HumanML3D CLIP, unimts_sim source)
    ("vit1d_clip_humanml3d_flu_probe_inertia1_{v}_unimtssim", "vit1d_clip_humanml3d_flu_probe_mae_acconly_unimtssim.yaml", {"clip": True}),
    ("vit1d_clip_humanml3d_flu_finetune_inertia1_{v}_unimtssim_texttrainable_lr5e-5_i2t",
     "vit1d_clip_humanml3d_flu_finetune_mae_unimtssim_texttrainable_lr5e-5_i2t.yaml", {"clip": True}),
    # 6. Parkinson's disease (PADS): 3-class probe/finetune + binary probes
    ("vit1d_pd_probe_pads_inertia1_{v}", "vit1d_pd_probe_pads_mae_multidata_flu.yaml", {}),
    ("vit1d_pd_finetune_pads_inertia1_{v}", "vit1d_pd_finetune_pads_mae_multidata_flu.yaml", {}),
    ("vit1d_pdhc_probe_pads_inertia1_{v}", "vit1d_pdhc_probe_pads_mae_multidata_flu.yaml", {}),
    ("vit1d_pddd_probe_pads_inertia1_{v}", "vit1d_pddd_probe_pads_mae_multidata_flu.yaml", {}),
]


def _find_objective_block(cfg):
    obj = cfg["objective"]
    t = obj["type"]
    return obj.setdefault(t, {})


def convert(cfg: dict, variant: str, tweaks: dict) -> dict:
    sys.path.insert(0, REPO)
    from imu_lm.models.baselines.inertia1_encoder import inertia1_downstream_config
    cfg = inertia1_downstream_config(cfg, variant, VARIANTS[variant],
                                     openwatch=bool(tweaks.get("openwatch")), clip=bool(tweaks.get("clip")))
    if tweaks.get("fixed_budget"):
        # Template has early_stopping.enabled=false + max_steps=10M (runs to the 1-day
        # walltime; sibling rows were cancelled after plateau / TIMEOUT). Use an explicit
        # fixed budget instead -- and keep early stopping ENABLED with a huge patience,
        # since enabled=false stops best.pt from ever being written (memory note).
        tr = cfg.setdefault("trainer", {})
        tr["max_steps"] = int(tweaks["fixed_budget"])
        tr["early_stopping"] = {"enabled": True, "monitor": "val_loss", "patience": 999999}
    return cfg


HEADER = """# AUTO-GENERATED by baselines/inertia1/scripts/make_configs.py -- do not edit by hand.
# Inertia-1 external baseline (Xu et al. 2026, arXiv:2607.06617), variant={variant}
# ({run}), derived from {template} with ONLY the encoder side changed:
#   * inertia1 block replaces vit1d.encoder (adapter: imu_lm/models/baselines/inertia1_encoder.py)
#   * accelerometer-only input (Inertia-1 is acc-only) -> data.acc_only like the mae_acconly rows
#   * raw g-scale input, no per-window z-score / band-pass (Inertia-1 uses none)
# Everything else (data, split, window, decoder/head, optimiser, early stopping) is identical
# to the sibling rows in Experiments_table.md.
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    names = []
    for variant, run in VARIANTS.items():
        for name_t, template, tweaks in TASKS:
            name = name_t.format(v=SHORT[variant])
            src = os.path.join(CFG, template)
            with open(src) as f:
                cfg = yaml.safe_load(f)
            new = convert(cfg, variant, tweaks)
            path = os.path.join(OUT, name + ".yaml")
            with open(path, "w") as f:
                f.write(HEADER.format(variant=variant, run=run, template=template))
                yaml.safe_dump(new, f, sort_keys=False, default_flow_style=None, width=120)
            names.append(name)
    print("\n".join(names))


if __name__ == "__main__":
    main()
