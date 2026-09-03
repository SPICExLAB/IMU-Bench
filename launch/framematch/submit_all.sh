#!/bin/bash
# FRAME-MATCH ABLATION -- submit the whole chain (2026-08-20).
#
#   Q: does the coordinate frame a model is PRETRAINED in have to match the frame
#      its downstream data arrives in?
#
#   3 pretraining arms x 2 downstream frames, frozen probe, 4 tasks:
#       canon (no rotation) / oct (24 cube rotations) / so3 (uniform SO(3))
#         x  native frame / rot7 (one fixed random SO(3) rotation, all splits)
#     + reference rows in the rotated frame: mae_recipe_r0 (production 514 h
#       encoder) and scratch (no pretraining).
#
#   cache build (CPU, ~30 min) -> 3 pretrains (30k steps each) -> 8 probe packs
#   of 4 cells. Everything is chained, idempotent and twin-guarded; re-running
#   this script only submits what is still missing.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

CACHE_JID=$(sbatch --parsable launch/framematch/00_build_cache.sh)
echo "cache build: $CACHE_JID"

for ARM in canon oct so3; do
  JIDS=()
  for SPEC in "b53017:b53017" "p32139:gengpu"; do
    ACCT="${SPEC%%:*}"; PART="${SPEC##*:}"
    JIDS+=("$(sbatch --parsable --account="$ACCT" --partition="$PART" \
        --job-name="mae_frame_${ARM}" --dependency=afterok:"$CACHE_JID" \
        --export=ALL,ARM="$ARM" launch/framematch/10_pretrain_arm.sh)")
  done
  echo "pretrain ${ARM}: ${JIDS[*]}"
  python bench/generated/pack_framematch_cells.py --arm "$ARM" --deps "$(IFS=,; echo "${JIDS[*]}")"
done

# r0 + scratch reference rows need no pretraining -- submit immediately.
python bench/generated/pack_framematch_cells.py --arm ref

echo
echo "harvest when the packs finish:  python analysis/harvest_frame_match.py"
