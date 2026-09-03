#!/bin/bash
source TransfHAR_env/bin/activate
for ds in wisdm utd_mhad pamap2 recofit hhar mhealth; do
  for spec in "mae_11src probe vit1d_har_probe_${ds}_mae_multidata" \
              "mae_11src finetune vit1d_har_finetune_${ds}_mae_multidata" \
              "mae_5src probe vit1d_har_probe_${ds}_mae_realdata_ocdetect_eatingspeed_motionx_sonic" \
              "mae_5src finetune vit1d_har_finetune_${ds}_mae_realdata_ocdetect_eatingspeed_motionx_sonic" \
              "mae_noaltattn probe vit1d_har_probe_${ds}_mae_noaltattn_multidata" \
              "mae_acconly probe vit1d_har_probe_${ds}_mae_acconly_multidata" \
              "focal probe vit1d_har_probe_${ds}_focal_multidata" \
              "simmtm probe vit1d_har_probe_${ds}_simmtm_multidata" \
              "relcon probe vit1d_har_probe_${ds}_relcon_multidata" \
              "scratch scratch vit1d_har_supervised_${ds}_altattn_scratch" \
              "deepconvlstm external deepconvlstm_har_${ds}"; do
    set -- $spec
    if [ "$1" = "deepconvlstm" ]; then
      python -m bench.score --task har_$ds --encoder $1 --protocol $2 --run $3 --seed 0 --note "DeepConvLSTM supervised reference (Experiments_table §3), same windows/split" 2>&1 | grep -E "bench.score\]|Error|Traceback" | tail -1
    else
      python -m bench.score --task har_$ds --encoder $1 --protocol $2 --run $3 --legacy 2>&1 | grep -E "bench.score\]|Error|Traceback" | tail -1
    fi
  done
done
echo ALL_DONE
