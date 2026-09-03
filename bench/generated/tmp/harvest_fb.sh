#!/bin/bash
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
while read -r e p r; do
  python -m bench.score --task full_body_pose_nymeria --encoder $e --protocol $p --run $r 2>&1 | grep "bench.score\]" | tail -1
done <<'LIST'
mae_11src finetune vit1d_full_body_pose_finetune_nymeria_mae_multidata
mae_5src probe vit1d_full_body_pose_probe_nymeria_mae_realdata_ocdetect_eatingspeed_motionx_sonic
mae_5src finetune vit1d_full_body_pose_finetune_nymeria_mae_realdata_ocdetect_eatingspeed_motionx_sonic
mae_noaltattn probe vit1d_full_body_pose_probe_nymeria_mae_noaltattn_multidata
mae_acconly probe vit1d_full_body_pose_probe_nymeria_mae_acconly_multidata
focal probe vit1d_full_body_pose_probe_nymeria_focal_multidata
simmtm probe vit1d_full_body_pose_probe_nymeria_simmtm_multidata
relcon probe vit1d_full_body_pose_probe_nymeria_relcon_multidata
scratch scratch vit1d_full_body_pose_supervised_nymeria_altattn_scratch
LIST
echo HARVEST_FB_DONE
