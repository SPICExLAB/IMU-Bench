#!/bin/bash
# Submit all 10 missing HHAR/MHEALTH probe cells (5 methods x 2 datasets).
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
mkdir -p logs

sbatch launch/har_hhar_mhealth/launch_har_probe_hhar_focal.sh
sbatch launch/har_hhar_mhealth/launch_har_probe_hhar_simmtm.sh
sbatch launch/har_hhar_mhealth/launch_har_probe_hhar_relcon.sh
sbatch launch/har_hhar_mhealth/launch_har_probe_hhar_mae_noaltattn.sh
sbatch launch/har_hhar_mhealth/launch_har_probe_hhar_mae_acconly.sh
sbatch launch/har_hhar_mhealth/launch_har_probe_mhealth_focal.sh
sbatch launch/har_hhar_mhealth/launch_har_probe_mhealth_simmtm.sh
sbatch launch/har_hhar_mhealth/launch_har_probe_mhealth_relcon.sh
sbatch launch/har_hhar_mhealth/launch_har_probe_mhealth_mae_noaltattn.sh
sbatch launch/har_hhar_mhealth/launch_har_probe_mhealth_mae_acconly.sh
