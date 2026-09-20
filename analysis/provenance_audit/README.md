# Pretraining-corpus provenance audit (2026-09-18)

Question: does the 5-source reference corpus (mae_5src and all 5src baselines) contain motion that also appears in
an IMU-Bench evaluation set?

- `hml_split_sources.json` — HumanML3D test clips per source (AMASS subsets + HumanAct12) from `index.csv`.
- `dupcheck_dense.py` / `dupcheck_dense_result.json` — sequence-level search of every HumanML3D test clip (incl.
  mirrors; 447,888 two-second snippets, 1-frame hop) against all 7 Motion-X++ subsets used in pretraining
  (107,124 snippets resampled 30->20 fps), after re-centring and heading alignment. Closest clip 3.6 cm, 1st
  percentile 4.4 cm, median 9.6 cm; no snippet under 2 cm.
- `inject_result_dense.json` — sensitivity check: 448 real test clips planted as copies (random offsets, 20->30->20
  fps round trip) are found at median 0.1 mm, p95 2.7 mm, 99.8 % under 2 cm.
- SONIC/BONES-SEED: different skeleton, no data-level check; every take dates from 2021-2024 (metadata parquet),
  after AMASS (2019) and HumanAct12 (2020).
- Other findings: PRISM (CMU) and SAMoSA (CMU, in the pretraining pool) may share participants — not checkable;
  437 of 2,192 HumanML3D test originals share frames of the same AMASS file with a train/val clip (HumanML3D's own
  split; affects all encoders equally); utd_mhad contributes 0 pretraining windows (clips shorter than 10.24 s).
