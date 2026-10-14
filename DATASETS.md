# Getting the data

IMU-Bench does not redistribute any dataset. Every evaluation set has to be downloaded from its source under
that source's licence. This page gives the access steps (checked on 2026-10-14), the licence, the files the
harness reads, and the directory layout it expects.

**Gates at a glance.** No dataset needs a signed data-use agreement or a credentialed account. Three need a free
registration, and the rest download directly.

| Task(s) | Dataset | Licence | Access |
|---|---|---|---|
| Arm pose, full-body pose, odometry | Nymeria | CC BY-NC 4.0 | email sign-up; download links expire after 14 days |
| Full-body pose (scorer) | SMPL body model | SMPL-Model licence (non-commercial) | registration |
| Microgesture | OpenWatch | CC BY 4.0 | Hugging Face login + automatic gate |
| IMU–text retrieval | HumanML3D + AMASS | MIT (texts) / AMASS research licence | AMASS registration |
| Parkinson's | PADS (PhysioNet) | CC BY-NC-SA 4.0 | open access, no account |
| Activity | HHAR, MHEALTH (UCI) | CC BY 4.0 | direct download |
| Activity | PrISM | see below | partly public |
| Activity | TransfHAR (ours) | — | release in preparation |

Some sets are only partly public, and the TransfHAR release is still in preparation. See [Known gaps](#known-gaps)
before planning a full run.

## Directory layout

The task cards name our cluster's paths. Put the data under one directory with the layout below and set
`IMUBENCH_DATA_ROOT` to it. Every dataset path in the registry and in the generated configs is then rewritten
to that directory.

```
$IMUBENCH_DATA_ROOT/
  processed_data_body_Ty_rwrist/          Nymeria training recordings, one *_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl each (867)
  processed_data_test_body_Ty_rwrist/     Nymeria evaluation recordings (101)
  openwatch_raw/open_watch_without_augmentations/
  HumanML3D/                              HumanML3D texts + UniMTS_data/data/{sim,aug_texts}
  pads_dataset.parquet                    scripts/prepare_pads_dataset.py
  hhar_dataset.parquet                    scripts/prepare_hhar_dataset.py
  mhealth_dataset.parquet                 scripts/prepare_mhealth_dataset.py
  userstudy_watch_dataset.parquet         scripts/prepare_userstudy_watch.py (TransfHAR)
  IMULM_master_dataset_v2.parquet         PrISM, in the shared activity schema
```

```bash
export IMUBENCH_DATA_ROOT=/data/imubench
imu-bench eval --encoder my_encoder.yaml
```

Two further paths are repo-relative and stay inside the clone: `assets/smpl/basicmodel_m.pkl` (the SMPL model)
and the HF cache for the CLIP text tower used by the retrieval task.

## Evaluation datasets

### Nymeria (arm pose, full-body pose, odometry)
- **Source:** https://www.projectaria.com/datasets/nymeria/, code at https://github.com/facebookresearch/nymeria_dataset.
  The original release is on the `nymeria_dataset_legacy` branch; `main` now holds NymeriaPlus.
- **Licence:** CC BY-NC 4.0. Derived data may be shared for non-commercial use, with attribution.
- **Access:**
  1. Enter your email under "Access the Dataset" on the project page. This accepts the licence and opens the download page.
  2. Download the URL file there, or build a filtered one in the explorer (https://explorer.projectaria.com/nymeria).
  3. Run `download.py -i <urls.json> -o <out>` from the Nymeria repository.
  The links expire after 14 days; fetch a new URL file when they do. The files are personal signed URLs, so never commit them.
- **What the benchmark needs:** two data groups per sequence. Skip the head, observer and video groups; the full
  dataset is about 70 TB.
  - `body_motion`: `body/xdata.npz`, the XSens ground truth, about 1.1 TB.
  - `recording_rwrist`: `recording_rwrist/data/motion.vrs` and `mps/slam/closed_loop_trajectory.csv`, about 0.5 TB.
- **Cite:** Ma et al., *Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild*, ECCV 2024.

### SMPL (full-body pose scorer)
- **Source:** https://smpl.is.tue.mpg.de/. Register, download "SMPL for Python", and place the male model at
  `assets/smpl/basicmodel_m.pkl` (see `assets/smpl/README.md`).
- **Licence:** SMPL-Model licence, non-commercial scientific research. The model files may not be redistributed.

### OpenWatch (microgesture)
- **Source:** https://huggingface.co/datasets/pietrobonazzi/openwatch
- **Licence:** CC BY 4.0.
- **Access:** log in to Hugging Face, accept the gate (approved automatically), then
  `huggingface-cli download --repo-type dataset pietrobonazzi/openwatch`.
- **What the benchmark needs:** only `open_watch_without_augmentations/`. In the augmented release, participants 51–85
  are mirrored copies of real participants. The CSVs hold raw sensor counts; the task card converts them to m/s² and
  rad/s.
- **Cite:** Bonazzi, Ahmed et al., *OpenWatch: A Multimodal Benchmark for Hand Gesture Recognition on Smartwatches*,
  arXiv:2605.04791.

### HumanML3D (IMU–text retrieval)
- **Source:** https://github.com/EricGuo5513/HumanML3D (code and texts, MIT). The motion comes from AMASS
  (https://amass.is.tue.mpg.de/), which needs a registration and allows no redistribution of the data or of
  anything derived from it.
- **Access:**
  1. Register at AMASS, and download the SMPL+H G archives together with SMPL+H and DMPL.
  2. Run the HumanML3D scripts to rebuild the motions, texts and official `train/val/test.txt` splits.
- **Simulated wrist IMU:** the task reads simulated right-wrist IMU and caption paraphrases in the UniMTS format,
  `UniMTS_data/data/sim/<id>.npy` and `UniMTS_data/data/aug_texts/<id>.txt`
  (UniMTS, https://github.com/xiyuanzh/UniMTS). These two folders are **not** in the public `UniMTS_data.zip`
  on Hugging Face. See [Known gaps](#known-gaps).
- **Cite:** Guo et al., CVPR 2022; Mahmood et al., *AMASS*, ICCV 2019; Zhang et al., *UniMTS*, NeurIPS 2024.

### PADS (Parkinson's disease)
- **Source:** https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/
- **Licence:** CC BY-NC-SA 4.0. **Open access:** the PhysioNet page reads "Anyone can access the files". No account,
  training course or data-use agreement is needed.
- **Access:** download the ZIP, or
  `wget -r -N -c -np https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/`, then run
  `python scripts/prepare_pads_dataset.py`.
- **Cite:** Varghese et al., *Machine learning in the Parkinson's disease smartwatch (PADS) dataset*,
  npj Parkinson's Disease 10, 9 (2024).

### HHAR and MHEALTH (activity)
- **Sources:** https://archive.ics.uci.edu/dataset/344 (HHAR) and https://archive.ics.uci.edu/dataset/319 (MHEALTH),
  both CC BY 4.0, direct download.
- **Build:** `python scripts/prepare_hhar_dataset.py` (reads `Watch_accelerometer.csv` / `Watch_gyroscope.csv`) and
  `python scripts/prepare_mhealth_dataset.py`.
- **Cite:** Stisen et al., SenSys 2015; Banos et al., IWAAL 2014.

### PrISM (activity)
- **Source:** the PrISM procedure-tracking recordings of Arakawa, Goel et al. (CMU), https://github.com/cmusmashlab/prism.
  The repository is MIT-licensed; whether that licence covers the data is not stated.
- **What the task uses:** 31 participants and 16.3 h over eight procedures. The five classes are our grouping of the
  procedure steps into activities of daily living (food, household, personal care, appliance, desk/device).
- **Availability:** the public download (Dropbox, linked from the repository) holds the `cooking` and `latte_making`
  sessions, about 5.2 of the 16.3 h. See [Known gaps](#known-gaps).
- **Cite:** Arakawa et al., *PrISM-Tracker*, Proc. ACM IMWUT 6(4), 2022.

### TransfHAR (activity)
Sixty activities recorded with a smartwatch for this paper. Its release is in preparation; until then the task
cannot be run outside our group.

## Pretraining corpus (controlled track only)

Only controlled-track submissions retrain on the IMU-Bench corpus; open-track submissions bring their own weights.

| Source | Where | Licence |
|---|---|---|
| WISDM 2019 | https://archive.ics.uci.edu/dataset/507 | CC BY 4.0 |
| RecoFit | https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors (`git lfs pull`) | CDLA-Permissive-2.0 |
| PAMAP2 | https://archive.ics.uci.edu/dataset/231 | CC BY 4.0 |
| SAMoSA | https://github.com/cmusmashlab/SAMoSA | CMU non-commercial research licence (no redistribution) |
| Shoaib (complex activities) | https://www.utwente.nl/en/eemcs/ps/research/dataset/ | no licence stated; cite Shoaib et al., Sensors 2016 |
| UT-Watch | https://doi.org/10.18738/T8/NNDFQD | CC0 1.0 |
| OCDetect | https://zenodo.org/records/13924901 | CC BY 4.0 |
| EatingSpeed | https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/CN8VBB | CC BY-NC-ND 4.0 (no sharing of adapted data) |
| Motion-X++ | https://huggingface.co/datasets/YuhongZhang/Motion-Xplusplus | Motion-X licence (CC BY-NC-SA 4.0 + per-subset licences) |
| BONES-SEED | https://huggingface.co/datasets/bones-studio/seed (`scripts/download/download_bones_seed.py`) | BONES-SEED licence (gated; no public re-upload; attribution required) |

Caches are built with `scripts/build_multidata_cache.py --config configs/base.yaml --model-config <pretrain config> --out-dir <cache dir>`.
Because of the EatingSpeed, SAMoSA, AMASS and BONES-SEED terms, the built cache cannot be shared as a file.

## Known gaps

We are working on each of these. Until they are closed, a third party can run 7 of the 10 tasks end to end:
arm pose, full-body pose, odometry, microgesture, Parkinson's, HHAR and MHEALTH.

- **IMU–text retrieval.** The simulated wrist IMU and caption paraphrases (`UniMTS_data/data/{sim,aug_texts}`) are
  not in the public UniMTS download, and they derive from AMASS, so we cannot redistribute our copy.
  You can regenerate them with the UniMTS pipeline (`pos2bvh.py`, then IMUSim `bvh2ts.py`, then `text_aug.py`).
  The paraphrases are written by a language model, so a regenerated set will not match ours exactly.
- **PrISM.** Only two of the eight procedures are in the public download.
- **TransfHAR.** Release in preparation.
- **Nymeria preprocessing.** The converter from the raw Nymeria groups to the per-recording `.pkl` files is not in
  this repository yet.
