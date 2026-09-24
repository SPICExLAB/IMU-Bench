# PD cells scored WITHOUT class-weighted CE (superseded 2026-09-24)

A copy of every pd_pads cell the paper uses, as scored before
`objective.utd_mhad_{probe,finetune}.class_weighted_ce` and
`objective.supervised.class_weighted_ce` were turned on in the task card.

Why they were replaced: PADS trains at 17 % HC / 58 % PD / 24 % DD and the subject
metric averages ~180 windows per subject, so unweighted CE left a prior bias that the
averaging could not cancel. 73 of 118 cells never predicted all three classes at subject
level and 15 predicted PD for all 94 subjects (subject macro-F1 exactly 0.2544). The
window classifier was not collapsed in those cells, so the fault was prior
miscalibration, not the features.

These files are NOT read by the leaderboard (bench/schema.py skips directories starting
with "_"). To roll back, copy pd_pads/*.json back into bench/results/pd_pads/ and
_units/*.npz back into bench/results/_units/pd_pads/.
