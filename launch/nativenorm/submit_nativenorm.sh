#!/bin/bash
# Submit the native-preprocessing family: SimMTM / FOCAL / RelCon re-pretrained under their own
# input handling, then the 10 IMU-Bench probe cells for each.
#
# Idempotent: any job whose name is already queued or running is skipped, so a partial
# submission can be re-run safely. Pass --dry-run to print what would be submitted.
#
#   bash launch/nativenorm/submit_nativenorm.sh [--dry-run]
#
# Dependency graph:
#   relcon_motifdist_5src_native ──> vit1d_ssl_relcon_5src_native ──> 10 relcon_5src_native cells
#   vit1d_ssl_simmtm_5src_native ─────────────────────────────────> 10 simmtm_5src_native cells
#   vit1d_ssl_focal_5src_native  ─────────────────────────────────> 10 focal_5src_native cells
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"

TASKS=(arm_pose_nymeria full_body_pose_nymeria odometry_nymeria_local microgesture_openwatch
       language_humanml3d pd_pads har_hhar har_mhealth har_prism har_userwatch)

queued() { squeue -u "$USER" -h -o "%j" | grep -qx "$1"; }

submit_pretrain() {   # name script [dependency]
    local name="$1" script="$2" dep="${3:-}"
    if queued "$name"; then echo "[skip] $name already queued"; echo ""; return; fi
    if [[ -f "runs/$name/checkpoints/best.pt" ]]; then echo "[skip] $name already has best.pt"; echo ""; return; fi
    local args=(); [[ -n "$dep" ]] && args+=(--dependency="afterok:$dep")
    if [[ -n "$DRY" ]]; then echo "[dry-run] sbatch ${args[*]} $script" >&2; echo "DRYJID"; return; fi
    local jid; jid=$(sbatch --parsable "${args[@]}" "$script")
    echo "[submit] $name -> $jid" >&2
    echo "$jid"
}

echo "=== stage 1: RelCon motif-distance network on un-normalised input ==="
MD=$(submit_pretrain relcon_motifdist_5src_native launch/nativenorm/relcon_motifdist_5src_native.sh)

echo "=== stage 2: the three pretrains ==="
S=$(submit_pretrain vit1d_ssl_simmtm_5src_native launch/nativenorm/vit1d_ssl_simmtm_5src_native.sh)
F=$(submit_pretrain vit1d_ssl_focal_5src_native  launch/nativenorm/vit1d_ssl_focal_5src_native.sh)
R=$(submit_pretrain vit1d_ssl_relcon_5src_native launch/nativenorm/vit1d_ssl_relcon_5src_native.sh "$MD")

echo "=== stage 3: 10 probe cells per arm ==="
submit_cells() {   # encoder pretrain_jid
    local enc="$1" dep="$2"
    for t in "${TASKS[@]}"; do
        local cell="bench_${t}__${enc}__probe__s42"
        if queued "$cell"; then echo "[skip] $cell already queued"; continue; fi
        if [[ -f "bench/results/${t}/${enc}__probe__s42.json" ]]; then echo "[skip] $cell already scored"; continue; fi
        local args=(--cell "$t" "$enc" probe)   # gengpu, as every other probe cell in the benchmark
        [[ -n "$dep" && "$dep" != "DRYJID" ]] && args+=(--dependency "afterok:$dep")
        [[ -n "$DRY" ]] && args+=(--dry-run)
        python -m bench.launch "${args[@]}"
    done
}
submit_cells simmtm_5src_native "$S"
submit_cells focal_5src_native  "$F"
submit_cells relcon_5src_native "$R"

echo "=== done ==="
