#!/bin/bash
# Row registry for the complete retrieval eval (sourced by
# launch_slurm_eval_retrieval_alignment*.sh). Each row = "<name>|<kind>|<args...>"
#   kind=ours   : args = <run_name> <label>
#   kind=unimts : args = <checkpoint> <dataset> <label> [extra evaluator flags]
UNIMTS_CK=baselines/language_alignment/UniMTS/checkpoint

declare -A ROWS
# ── HumanML3D, UniMTS_data setting (Section 13b / 13c rows) ─────────────────
ROWS[humanml3d__ours_mae_finetune_i2t]="ours|vit1d_clip_humanml3d_finetune_mae_unimtssim_texttrainable_lr5e-5_i2t|Ours MAE finetune, i2t-only loss (job 8173825)"
ROWS[humanml3d__ours_mae_finetune_sym]="ours|vit1d_clip_humanml3d_finetune_mae_unimtssim_texttrainable_lr5e-5|Ours MAE finetune, symmetric loss (job 8147232)"
ROWS[humanml3d__ours_mae_probe]="ours|vit1d_clip_humanml3d_probe_mae_unimtssim|Ours MAE frozen probe (13c)"
ROWS[humanml3d__ours_scratch_sym]="ours|vit1d_clip_humanml3d_fromscratch_unimtssim_texttrainable_lr5e-5|Ours scratch, symmetric loss (job 8155972)"
ROWS[humanml3d__ours_scratch_i2t]="ours|vit1d_clip_humanml3d_fromscratch_unimtssim_texttrainable_lr5e-5_i2t|Ours scratch, i2t-only loss (job 8173826)"
ROWS[humanml3d__unimts_scratch_jointmask_wrist]="unimts|$UNIMTS_CK/unimts_userdata_scratch_jointmask_100ep/last.pth|humanml3d|UniMTS scratch, joint-mask recipe 100ep last.pth (job 8192640), wrist-only|--wrist-only 1"
ROWS[humanml3d__unimts_scratch_fullgrid_wrist]="unimts|$UNIMTS_CK/unimts_userdata_scratch_valsplit/best.pth|humanml3d|UniMTS scratch, full-grid trained (job 8163821), wrist-only eval|--wrist-only 1"
ROWS[humanml3d__unimts_scratch_fullgrid_full]="unimts|$UNIMTS_CK/unimts_userdata_scratch_valsplit/best.pth|humanml3d|UniMTS scratch, full-grid trained (job 8163821), 22-joint eval|--wrist-only 0"
ROWS[humanml3d__unimts_official_wrist]="unimts|$UNIMTS_CK/UniMTS.pth|humanml3d|UniMTS official zero-shot, wrist-only (contaminated: pretrained on HumanML3D)|--wrist-only 1"
# ── Nymeria, synthetic wrist IMU + atomic_action captions (Section 13a) ─────
ROWS[nymeria__ours_mae_finetune]="ours|vit1d_clip_nymeria_atomicaction_synthetic_finetune_mae_multidata|Ours MAE finetune (job 7692004)"
ROWS[nymeria__ours_mae_probe]="ours|vit1d_clip_nymeria_atomicaction_synthetic_mae_probe|Ours MAE frozen probe (job 7692003)"
ROWS[nymeria__ours_scratch]="ours|vit1d_clip_nymeria_atomicaction_synthetic_fromscratch_multidata|Ours scratch (job 8282334)"
ROWS[nymeria__unimts_scratch_synthetic]="unimts|$UNIMTS_CK/nymeria_scratch_synthetic/best_val_loss.pth|nymeria|UniMTS scratch, synthetic IMU (job 8158173)|--imu-source synthetic --caption-source atomic_action"
ROWS[nymeria__unimts_official]="unimts|$UNIMTS_CK/UniMTS.pth|nymeria|UniMTS official zero-shot (clean: never saw Nymeria)|--imu-source synthetic --caption-source atomic_action"

HUMANML3D_ROWS="humanml3d__ours_mae_finetune_i2t humanml3d__ours_mae_finetune_sym humanml3d__ours_mae_probe humanml3d__ours_scratch_sym humanml3d__ours_scratch_i2t humanml3d__unimts_scratch_jointmask_wrist humanml3d__unimts_scratch_fullgrid_wrist humanml3d__unimts_scratch_fullgrid_full humanml3d__unimts_official_wrist"
NYMERIA_ROWS="nymeria__ours_mae_finetune nymeria__ours_mae_probe nymeria__ours_scratch nymeria__unimts_scratch_synthetic nymeria__unimts_official"

# run_row <name> <out_dir> <common evaluator flags>  -> returns evaluator exit code
run_row() {
    local name="$1" out="$2"; shift 2
    local spec="${ROWS[$name]:-}"
    if [[ -z "$spec" ]]; then echo "unknown row: $name"; return 2; fi
    IFS='|' read -r kind a b c d <<< "$spec"
    echo "=================== [$name] $spec"
    if [[ "$kind" == "ours" ]]; then
        python -u scripts/eval_retrieval_alignment.py --model ours --run "$a" --ckpt best \
            --label "$b" --out "$out/$name.json" "$@"
    else
        # shellcheck disable=SC2086
        python -u scripts/eval_retrieval_alignment.py --model unimts --checkpoint "$a" --dataset "$b" \
            --label "$c" --out "$out/$name.json" "$@" $d
    fi
}
