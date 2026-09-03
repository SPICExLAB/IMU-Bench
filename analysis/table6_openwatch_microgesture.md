# Table 6 — OpenWatch microgesture recognition (paper-ready)

Source of numbers: `EXPERIMENTS.md` §5a (OpenWatch **augmented** release,
`open_watch_augmentations`). Last updated 2026-08-15.

> **⚠️ Read §5f before publishing this table.** The OpenWatch release itself
> was found contaminated (byte-identical duplicate subjects among the "50
> real participants"; aug ids 51-85 are wrist-mirrored copies of specific
> real subjects). Every number below uses the pre-discovery split and can
> carry train→val leakage; rows are comparable to each other, not to a
> leakage-free protocol. Contamination-safe, paper-protocol (windowed +
> stratified) reruns of Ours(MAE) are in flight: jobs 9388527-9388847 (ViT1D + Mix-Token + NormWear, one shared protocol) →
> `analysis/table6_openwatch_new.md` (paper-format, all 10 rows) + `analysis/openwatch_fair_comparison.md` — those are the numbers to quote
> against the paper's Table 2 once landed.

## What changed vs. the previous draft

The old table carried `NormWear-Base ... 78.8%†` as a row in the `Acc.`
column. That row was removed. It is not the same measurement as the rest of
the column:

| | Our rows | The cited 78.8% |
|---|---|---|
| Unit | whole-recording acc (1 CSV → 1 resampled 512-sample window → 1 prediction) | **clip-level (k=3)** from 1s/100-sample windows at stride 20 |
| Split | our seeded unstratified subject split | the paper's posture-stratified split |
| Modality | IMU-only (6ch acc+gyro, no PPG) | not confirmed IMU-only |

The paper's *own* window-level number for that same NormWear row is 68.2%, so
the aggregation step alone moves it ~10pp with no model change. Subtracting
73.02% from 78.8% produced a "5.8pp gap" that does not correspond to any real
quantity. It is replaced by a **first-party NormWear-Base row** run on this
same augmented release, our split, our metric code (job 9382934).

Note: §5c's existing first-party NormWear number (52.04%) is on the *no-aug*
release and belongs to §5's table — it could not simply be moved here, which
is why a new run was needed.

## Table 6 (markdown)

| Model | Acc. ↑ | Bal. Acc. ↑ | Macro-F1 ↑ |
|---|---|---|---|
| FOCAL | 55.56% | 40.65% | 0.412 |
| SimMTM | 62.43% | 53.59% | 0.540 |
| RelCon | 67.20% | 58.27% | 0.604 |
| NormWear-Base (first-party)† | 53.97% | 59.62% | 0.517 |
| Scratch | 58.73% | 53.83% | 0.517 |
| Ours (MAE) | 66.67% | 56.91% | 0.590 |
| Ours (MAE), finetuned | **73.02%** | **63.30%** | **0.672** |
| Ours (MAE), no alt-attn | — | — | — |
| Ours (MAE), acc-only | — | — | — |

† Our own run of the official NormWear-Base checkpoint (Luo et al. 2026,
136.1M params, Apache-2.0), frozen backbone + linear head, under this table's
split/protocol — not the number reported by Bonazzi et al.

The two `—` rows are **cancelled** SLURM jobs (7620352 / 7620353), not
pending; they need resubmitting or removing from the table.

## Cited numbers (separate block — do NOT put in the table above)

Reported by Bonazzi et al. (arXiv:2605.04791) Table 2 under *their* protocol
and *their* split:

| Method (cited, not run here) | Window Acc. | Clip Acc. (k=3) |
|---|---|---|
| Mix-Token (their proposed method) | 90.8% | 95.5% |
| Apple-CNN | not reported | 87.9% |
| NormWear-Base (headline; augmented, possibly +PPG) | 68.2% | 78.8% |
| NormWear-Base* (no-aug, IMU-only) | 37.0% | 69.0% |

Mix-Token and Apple-CNN have **no** asterisked no-aug/IMU-only row in Table 2
(only NormWear does) and no public checkpoint was reproduced here, so they
cannot be given a first-party row at all. A gap to Mix-Token's 95.5% differs
on two axes simultaneously (data condition *and* aggregation level) and is
therefore uninterpretable as a model-quality difference.

## LaTeX

```latex
\begin{table}[t]
\centering
\caption{Microgesture recognition on OpenWatch (augmented release, IMU-only).
All rows are whole-recording accuracy on the same seeded subject split with
the same metric code, so the column is internally comparable.}
\label{tab:openwatch}
\begin{tabular}{lccc}
\toprule
Model & Acc. $\uparrow$ & Bal. Acc. $\uparrow$ & Macro-F1 $\uparrow$ \\
\midrule
FOCAL                        & 55.56\% & 40.65\% & 0.412 \\
SimMTM                       & 62.43\% & 53.59\% & 0.540 \\
RelCon                       & 67.20\% & 58.27\% & 0.604 \\
NormWear-Base$^{\dagger}$    & 53.97\% & 59.62\% & 0.517 \\
Scratch                      & 58.73\% & 53.83\% & 0.517 \\
\midrule
Ours (MAE)                   & 66.67\% & 56.91\% & 0.590 \\
Ours (MAE), finetuned        & \textbf{73.02\%} & \textbf{63.30\%} & \textbf{0.672} \\
Ours (MAE), no alt-attn      & --      & --      & --    \\
Ours (MAE), acc-only         & --      & --      & --    \\
\bottomrule
\end{tabular}

\vspace{2pt}
\footnotesize $^{\dagger}$Our own run of the official NormWear-Base
checkpoint (136M params) under this table's protocol, not the value reported
by Bonazzi et al.
\end{table}

\begin{table}[t]
\centering
\caption{Results reported by Bonazzi et al.~\cite{openwatch} under
\emph{their} protocol (1\,s windows, stride~20, $k{=}3$ consecutive-run clip
voting, posture-stratified split). Context only: these are not comparable to
Table~\ref{tab:openwatch}, which reports whole-recording accuracy on a
different split.}
\label{tab:openwatch-cited}
\begin{tabular}{lcc}
\toprule
Method & Window Acc. & Clip Acc. ($k{=}3$) \\
\midrule
Mix-Token (their method)              & 90.8\% & 95.5\% \\
Apple-CNN                             & n/r    & 87.9\% \\
NormWear-Base                         & 68.2\% & 78.8\% \\
NormWear-Base$^{*}$ (no-aug, IMU-only)& 37.0\% & 69.0\% \\
\bottomrule
\end{tabular}
\end{table}
```

## Two caveats worth stating in the paper

1. **These are validation-split numbers**, and `best.pt` is selected by
   val_loss on that same split (this benchmark's standing convention). There
   is a held-out test split defined but not reported here.
2. **The augmented-release ranking has never been checked under a
   posture-stratified split.** §5e redid only the *no-aug* table, and there
   the split change moved every frozen-probe row by 6–18pp and flipped which
   architecture pattern wins. The same confound (2 of the dataset's only 3
   walking-posture subjects land in the 5-subject val set) applies to these
   augmented numbers too.
