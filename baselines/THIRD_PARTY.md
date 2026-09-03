# Third-party baseline repositories

These are cloned locally but NOT vendored into IMU-Bench (kept out of the index).
Clone the upstream and apply the matching patch from `baselines/patches/` where one exists.

| path | upstream | local patch |
|---|---|---|
| `baselines/HAR/DeepConvLSTM` | https://github.com/STRCWearlab/DeepConvLSTM.git | — |
| `baselines/SSL/NormWear` | https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear.git | `patches/SSL_NormWear.patch` |
| `baselines/SSL/relcon` | https://github.com/maxxu05/relcon.git | — |
| `baselines/full_body_pose/SimplePoser` | https://github.com/SPICExLAB/SimplePoser.git | `patches/full_body_pose_SimplePoser.patch` |
| `baselines/inertia1/repo` | https://github.com/yang-ai-lab/Inertia-1.git | — |
| `baselines/language_alignment/UniMTS` | https://github.com/xiyuanzh/UniMTS.git | `patches/language_alignment_UniMTS.patch` |
| `baselines/odometry/Air-IO` | https://github.com/Air-IO/Air-IO.git | — |
| `baselines/odometry/TLIO` | https://github.com/CathIAS/TLIO.git | — |
