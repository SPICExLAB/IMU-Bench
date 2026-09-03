#!/bin/bash
# Harvest the decoder-capacity sweep. Reports BOTH the noise-mined `best`
# (Table-2 protocol) and the converged final-5-eval mean -- judge on final5,
# since EXPERIMENTS.md section 2 showed `best` falls with the NUMBER of evals,
# not model quality.
cd /gpfs/home/vbc1591/IMU-SSL
printf '%-8s %-10s %10s %10s %9s\n' DEC "2d x 1d" BEST FINAL5 EVALS
row() {
  d=runs/$1; m=$d/logs/metrics.txt
  if [[ ! -f "$m" ]]; then printf '%-8s %-10s %10s %10s %9s\n' "$2" "$3" - - "not started"; return; fi
  grep "split=val" "$m" | awk '{for(i=1;i<=NF;i++){split($i,a,"=");v[a[1]]=a[2]}; print v["mpjpe_mm"]}' > /tmp/d.txt
  n=$(wc -l < /tmp/d.txt)
  b=$(sort -g /tmp/d.txt | head -1)
  f=$(tail -5 /tmp/d.txt | awk '{s+=$1}END{if(NR)printf "%.2f", s/NR}')
  printf '%-8s %-10s %10.2f %10s %9s\n' "$2" "$3" "$b" "$f" "$n"
}
row vit1d_full_body_pose_probe_nymeria_mae_multidata_dec0x0 "0x0" "0 x 0 (lin)"
row vit1d_full_body_pose_probe_nymeria_mae_multidata_lr3e-4_cos "1x0*" "1 x 0"
for r in 2:0 4:0 8:0 1:1 1:2 1:4 1:6 1:8 1:12 1:16 4:2 2:4; do
  a=${r%%:*}; b=${r##*:}
  row vit1d_full_body_pose_probe_nymeria_mae_multidata_dec${a}x${b} "${a}x${b}" "$a x $b"
done
echo
echo "* control = existing lr3e-4_cos run (same config, decoder 1x0)"
echo "incumbent (untuned, Table 2): 97.63 mm"
