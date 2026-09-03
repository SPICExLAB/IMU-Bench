#!/bin/bash
# Submit every Inertia-1 downstream job (6 tasks x {probe, finetune} x 2 variants)
# plus the chained scoring jobs, gated on the pretraining job of each variant.
#
#   bash baselines/inertia1/submit_inertia1_downstream.sh <PT_JOB_PATCHTST> <PT_JOB_ART> [variants...]
#
# Long jobs (pose / odom / CLIP)  -> buy-in node   (--account=b53017 --partition=b53017)
# Short jobs (PADS / gesture)     -> gengpu, 4 h   (--account=p32139 --partition=gengpu)
# Scoring jobs                    -> buy-in node
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
PT_PATCHTST="${1:?pretrain job id (patchtst)}"
PT_ART="${2:?pretrain job id (ar_transformer)}"
shift 2
VARIANTS="${*:-patchtst art}"
JOB=baselines/inertia1/launch_slurm_job_inertia1.sh
LOG=baselines/inertia1/.downstream_jobids
mkdir -p logs analysis/inertia1 analysis/retrieval_eval/inertia1 analysis/pads_subject_level
BUYIN="--account=b53017 --partition=b53017"
GENGPU="--account=p32139 --partition=gengpu"
TRAIN_DIR=/projects/b53017/vbc1591/processed_data_body_Ty_rwrist
VAL_DIR=/projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist

sub() {  # sub <partition-flags> <time> <job-name> <dependency or ""> ENV=... -> prints job id
    local part="$1" time="$2" name="$3" dep="$4"; shift 4
    local depflag=""
    [ -n "$dep" ] && depflag="--dependency=afterok:$dep"
    local id
    id=$(env "$@" sbatch --parsable $part --time="$time" --job-name="$name" $depflag "$JOB")
    echo "$id $name dep=[$dep]" | tee -a "$LOG" >&2
    echo "$id"
}
train() {  # train <part> <time> <run-name> <dep>
    sub "$1" "$2" "$3" "$4" MODEL_CONFIG="configs/inertia1/$3.yaml" RUN_NAME="$3"
}
score() {  # score <part> <time> <job-name> <dep> <cmd>
    sub "$1" "$2" "$3" "$4" SCORE_CMD="$5"
}

for V in $VARIANTS; do
    case "$V" in
        patchtst) PT=$PT_PATCHTST; PR=inertia1_patchtst_small_nhanes; LABEL="Inertia-1 (PatchTST)" ;;
        art)      PT=$PT_ART;      PR=inertia1_art_small_nhanes;      LABEL="Inertia-1 (ART)" ;;
        *) echo "unknown variant $V"; exit 1 ;;
    esac
    echo "=== variant $V (pretrain job $PT, run $PR)"

    # ---- 1. arm pose ---------------------------------------------------------------
    A1=$(train "$BUYIN" 12:00:00 vit1d_arm_pose_probe_nymeria_inertia1_$V "$PT")
    A2=$(train "$BUYIN" 12:00:00 vit1d_arm_pose_finetune_nymeria_inertia1_$V "$PT")
    score "$BUYIN" 02:00:00 score_arm_pose_inertia1_$V "$A1:$A2" \
        "python scripts/eval_arm_pose_position_error.py --runs '$LABEL probe=vit1d_arm_pose_probe_nymeria_inertia1_$V' '$LABEL finetune=vit1d_arm_pose_finetune_nymeria_inertia1_$V' --out-json analysis/inertia1/arm_pose_position_error_$V.json" >/dev/null

    # ---- 2. full-body pose (mpjpe is logged by the run itself) -------------------------
    train "$BUYIN" 12:00:00 vit1d_full_body_pose_probe_nymeria_inertia1_$V "$PT" >/dev/null
    train "$BUYIN" 12:00:00 vit1d_full_body_pose_finetune_nymeria_inertia1_$V "$PT" >/dev/null

    # ---- 3. odometry ----------------------------------------------------------------
    score "$BUYIN" 04:00:00 score_odom_worldframe_probe_inertia1_$V "$PT" \
        "python scripts/eval_odom_nymeria_worldframe.py --run $PR --runs-root runs --train-dir $TRAIN_DIR --val-dir $VAL_DIR --out-dir data_vis/nymeria_odom_worldframe_probe_inertia1_$V" >/dev/null
    score "$BUYIN" 04:00:00 score_odom_localvel_probe_inertia1_$V "$PT" \
        "python scripts/eval_odom_nymeria_altattn.py --run $PR --runs-root runs --train-dir $TRAIN_DIR --val-dir $VAL_DIR --out-dir data_vis/nymeria_odom_inertia1_${V}_probe" >/dev/null
    O1=$(train "$BUYIN" 12:00:00 vit1d_odom_worldframe_finetune_nymeria_inertia1_$V "$PT")
    score "$BUYIN" 02:00:00 score_odom_worldframe_finetune_inertia1_$V "$O1" \
        "python scripts/vis_odom_scratch_worldframe.py --run vit1d_odom_worldframe_finetune_nymeria_inertia1_$V --runs-root runs --val-dir $VAL_DIR --out-dir data_vis/nymeria_odom_worldframe_finetune_inertia1_$V --label '$LABEL finetune'" >/dev/null

    # ---- 4. microgesture (OpenWatch paper protocol) ----------------------------------
    G1=$(train "$GENGPU" 04:00:00 vit1d_gesture_probe_openwatch_6class_inertia1_${V}_windowed_stratified "$PT")
    G2=$(train "$GENGPU" 04:00:00 vit1d_gesture_finetune_openwatch_6class_inertia1_${V}_windowed_stratified "$PT")
    G3=$(train "$GENGPU" 04:00:00 vit1d_gesture_probe_openwatch_6class_inertia1_${V}_windowed_stratified_aug "$PT")
    G4=$(train "$GENGPU" 04:00:00 vit1d_gesture_finetune_openwatch_6class_inertia1_${V}_windowed_stratified_aug "$PT")
    score "$BUYIN" 02:00:00 score_openwatch_inertia1_$V "$G1:$G2:$G3:$G4" \
        "python scripts/eval_openwatch_windowed_protocol.py --runs '$LABEL probe win/noaug:vit1d_gesture_probe_openwatch_6class_inertia1_${V}_windowed_stratified:openwatch_gesture_probe' '$LABEL probe win/aug:vit1d_gesture_probe_openwatch_6class_inertia1_${V}_windowed_stratified_aug:openwatch_gesture_probe' '$LABEL finetune win/noaug:vit1d_gesture_finetune_openwatch_6class_inertia1_${V}_windowed_stratified:openwatch_gesture_finetune' '$LABEL finetune win/aug:vit1d_gesture_finetune_openwatch_6class_inertia1_${V}_windowed_stratified_aug:openwatch_gesture_finetune' --splits val test --out-json results_openwatch_inertia1_$V.json" >/dev/null

    # ---- 5. language alignment (HumanML3D CLIP retrieval) ---------------------------
    C1=$(train "$BUYIN" 08:00:00 vit1d_clip_humanml3d_flu_probe_inertia1_${V}_unimtssim "$PT")
    C2=$(train "$BUYIN" 08:00:00 vit1d_clip_humanml3d_flu_finetune_inertia1_${V}_unimtssim_texttrainable_lr5e-5_i2t "$PT")
    RET="--gallery-sizes 64,100 --n-subsets 200 --seed 42 --batch-size 128 --split val"
    score "$BUYIN" 03:00:00 score_retrieval_inertia1_$V "$C1:$C2" \
        "python -u scripts/eval_retrieval_alignment.py --model ours --run vit1d_clip_humanml3d_flu_probe_inertia1_${V}_unimtssim --ckpt best --label '$LABEL frozen probe' --out analysis/retrieval_eval/inertia1/humanml3d__inertia1_${V}_probe.json $RET && python -u scripts/eval_retrieval_alignment.py --model ours --run vit1d_clip_humanml3d_flu_finetune_inertia1_${V}_unimtssim_texttrainable_lr5e-5_i2t --ckpt best --label '$LABEL finetune, i2t-only loss' --out analysis/retrieval_eval/inertia1/humanml3d__inertia1_${V}_finetune_i2t.json $RET" >/dev/null

    # ---- 6. Parkinson's disease (PADS) --------------------------------------------------
    P1=$(train "$GENGPU" 04:00:00 vit1d_pd_probe_pads_inertia1_$V "$PT")
    P2=$(train "$GENGPU" 04:00:00 vit1d_pd_finetune_pads_inertia1_$V "$PT")
    P3=$(train "$GENGPU" 04:00:00 vit1d_pdhc_probe_pads_inertia1_$V "$PT")
    P4=$(train "$GENGPU" 04:00:00 vit1d_pddd_probe_pads_inertia1_$V "$PT")
    PADS_CMD=""
    for r in vit1d_pd_probe_pads_inertia1_$V vit1d_pd_finetune_pads_inertia1_$V vit1d_pdhc_probe_pads_inertia1_$V vit1d_pddd_probe_pads_inertia1_$V; do
        PADS_CMD="$PADS_CMD python scripts/eval_pads_subject_level.py --model-config configs/inertia1/$r.yaml --run $r --out analysis/pads_subject_level/$r.json;"
    done
    score "$BUYIN" 02:00:00 score_pads_subject_level_inertia1_$V "$P1:$P2:$P3:$P4" "$PADS_CMD" >/dev/null
done
echo "all submitted; ids in $LOG"
