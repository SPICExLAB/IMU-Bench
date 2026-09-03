# HAR protocol + LR sweep (2026-08-23)

216 cells: {MAE-11src, MAE-5src, Inertia-1 PatchTST-S, Inertia-1 ART-S} x 9 HAR datasets x {probe, finetune} x 3 LRs,
ALL under the Inertia-1 paper protocol: inverse-frequency class-weighted CE + checkpoint selection by validation
balanced accuracy. MAE encoders: per-window z-score input (as pretrained); Inertia-1: raw input (as pretrained).
Probe grid {3e-4, 1e-3, 3e-3}; finetune grid {3e-5, 1e-4, 3e-4}. Scored: pooled val-pass on the bal-acc-selected
checkpoint (bench.score.har). Full per-LR values: results_lrtune/SUMMARY.json; per-cell JSONs alongside.

Headline (macro-F1, mean over 9 sets, best LR per cell):
  probe:    11src 0.564 | 5src 0.560 | I1-PT-S 0.603 | I1-ART-S 0.623
  finetune: 11src 0.634 | 5src 0.637 | I1-PT-S 0.651 | I1-ART-S 0.648
Wins by dataset (finetune): MAE takes UTD (5src 0.827), RecoFit (0.791), PAMAP2, SAMoSA; Inertia-1 takes MHEALTH
(ART 0.934), HHAR (PT 0.793), WISDM, UT-Watch, PrISM. Probe: Inertia-1 8/9 (MAE only PAMAP2).
Protocol effects: Inertia-1 gains from ITS OWN protocol too (PT-S finetune 0.684 -> 0.709 on the 7 shared sets);
probes prefer the highest LR tested for Inertia-1 (3e-3 on most sets) - a 1e-2 point may still help; finetune LR is
dataset-specific (3e-4 on gesture-like sets, 3e-5 on small/imbalanced ones); PAMAP2 finetune < probe for every encoder.


## Inertia-1-on-5src result (2026-08-24, 108/108 cells): means over 9 sets at best-LR — PT-5src probe 0.590 / finetune 0.641; ART-5src probe 0.565 / 0.614. References: mae_5src 0.560 / 0.637; I1-PT-NHANES 0.603 / 0.651; I1-ART-NHANES 0.623 / 0.648. Reading: (a) on the SAME ~250 h acc-only pool, PatchTST's frozen features still beat the 6-ch MAE probe (+0.030) and sit only −0.013 under its NHANES self → for PatchTST the probe advantage is mostly ARCHITECTURE+RECIPE (acc-only, 30 s g windows, patch recon), not NHANES scale; (b) MHEALTH probe 0.871 on 250 h (MAE 0.710, NHANES 0.900) — the MHEALTH gap is recipe, not data; (c) ART degrades much more without NHANES (probe −0.058) → the AR objective is the data-hungry one; (d) MAE keeps PAMAP2 (0.561 vs 0.435) and UTD/RecoFit finetunes. Corpus caveat: 250 h acc-only vs 280 h 6-ch, motionx absent, sonic 18 h — the 30 s recipe's own data losses are part of the comparison.
