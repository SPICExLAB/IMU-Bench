| source | real/synth | native_hz | coord frame | channels | n_train | n_val | acc mean (x,y,z) | acc std (x,y,z) | recon MSE |
|---|---|---|---|---|---|---|---|---|---|
| realdata | real | varies by sub-dataset (pamap2/recofit/samosa/shoaib/ut_watch/utd_mhad/wisdm) | raw device frame (no FLU transform -- imu_lm/data/loaders.py has none) | acc+gyro (6ch) | 93510 | 4189 | -0.000, 0.000, -0.000 | 1.000, 1.000, 1.000 | 0.7238 |
| ocdetect | real | 50 | FLU (apply_flu_transform=True) | acc+gyro (6ch) | 8703 | 1087 | 0.000, 0.000, 0.000 | 1.000, 1.000, 1.000 | 0.7449 |
| nymeria | real | Aria glasses native rate, resampled to 50 | FLU | acc+gyro (6ch) | 157885 | 18269 | 0.000, -0.000, 0.000 | 0.999, 0.999, 0.999 | 0.7268 |
| totalcapture | real | 60 | FLU | acc+gyro (6ch) | 461 | 47 | -0.000, -0.000, 0.000 | 1.000, 1.000, 1.000 | 0.7838 |
| imuposer | real | resampled to 50 | FLU (despite add_gravity=False by default) | acc+gyro (6ch) | 137 | 33 | -0.000, -0.000, -0.000 | 1.000, 1.000, 1.000 | 0.9276 |
| eatingspeed | real | 64 | raw sensor frame -- NO FLU transform (documented explicitly in eatingspeed_wrist_loader.py: 'no coordinate transform is applied') | acc+gyro (6ch) | 57536 | 10114 | 0.000, 0.000, -0.000 | 0.999, 0.999, 0.999 | 0.7460 |
| motionx | synthetic (derived IMU from mocap) | 30 | FLU (synthetic, derived via SMPL FK from AMASS-style mocap) | acc+gyro (6ch, synthetic gyro incl. optional injected white noise/bias drift) | 4683 | 269 | -0.000, -0.000, 0.000 | 1.000, 1.000, 1.000 | 0.9387 |
| humanml3d | synthetic (derived IMU from mocap) | 20 (source mocap), resampled to 50 via SMPL FK | FLU (synthetic, same SMPL-FK-derivation approach as nymeria/motionx) | acc+gyro (6ch) | 2825 | 353 | 0.000, 0.000, 0.000 | 0.809, 0.809, 0.809 | 0.5864 |
| sonic | synthetic (derived IMU from BVH mocap) | BVH source fps, resampled to 50 | FLU | acc+gyro (6ch) | 32662 | 11804 | -0.000, 0.000, -0.000 | 1.000, 1.000, 1.000 | 0.7939 |
| capture24 | real | 100 | raw device frame -- NO FLU transform (documented explicitly in capture24_loader.py: 'axis convention -- kept raw, no coordinate transform applied') | accelerometer-ONLY (3ch real + 3ch zero-padded/masked-invalid, see AccOnlyPadTo6ChDataset) | 124881 | 25336 | -0.000, -0.000, 0.000 | 0.999, 0.999, 1.000 | 0.7751 |
| openwatch | real | resampled to fixed window length | configurable, FLU in production config | acc+gyro (6ch), optional PPG (unused here) | 1842 | 314 | -0.000, 0.000, 0.000 | 1.000, 1.000, 1.000 | 0.6650 |
