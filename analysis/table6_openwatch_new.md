# Table 6 (new configuration): Microgesture recognition on OpenWatch

OpenWatch paper protocol (arXiv:2605.04791): 1s/100-sample windows, stride 20, 6 windows per clip (centered 2.0s sub-interval), clip label = k=3 consecutive-run else majority. One contamination-safe posture-stratified split for every row (33/5/5 canonical identities; test = 87 clips / 522 windows). IMU-only. Inverse-frequency class-weighted CE; model selection on val window macro-F1. Bold = column best. `–` = run not finished (see job table in EXPERIMENTS.md §5f).

## Trained with OpenWatch augmentation (paper headline condition)

**clip-level (k=3), test split, trained with OpenWatch augmentation**

| Model | Acc. ↑ | Bal. Acc. ↑ | Macro-F1 ↑ |
|---|---|---|---|
| FOCAL | 40.23% | 48.94% | 0.407 |
| SimMTM | 51.72% | 62.87% | 0.515 |
| RelCon | 44.83% | 53.90% | 0.443 |
| Scratch | 51.72% | 33.86% | 0.349 |
| Ours (MAE) | 59.77% | **72.50%** | 0.615 |
| Ours (MAE), finetuned | **66.67%** | 56.65% | 0.580 |
| Ours (MAE), no alt-attn | 57.47% | 68.06% | 0.582 |
| Ours (MAE), acc-only | 57.47% | 65.32% | 0.564 |
| Mix-Token (reimpl.) | 64.37% | 69.62% | **0.641** |
| NormWear-Base (first-party) | 54.02% | 57.48% | 0.494 |

**window-level, test split, trained with OpenWatch augmentation**

| Model | Acc. ↑ | Bal. Acc. ↑ | Macro-F1 ↑ |
|---|---|---|---|
| FOCAL | 36.21% | 39.01% | 0.335 |
| SimMTM | 43.30% | 49.75% | 0.415 |
| RelCon | 38.51% | 45.91% | 0.377 |
| Scratch | 49.81% | 35.24% | 0.354 |
| Ours (MAE) | 49.43% | 55.57% | 0.482 |
| Ours (MAE), finetuned | **63.41%** | 52.69% | **0.544** |
| Ours (MAE), no alt-attn | 50.19% | 57.23% | 0.494 |
| Ours (MAE), acc-only | 48.66% | 54.28% | 0.471 |
| Mix-Token (reimpl.) | 57.66% | **58.31%** | 0.542 |
| NormWear-Base (first-party) | 49.04% | 53.05% | 0.454 |

## Trained on real recordings only

**clip-level (k=3), test split, trained on real recordings only**

| Model | Acc. ↑ | Bal. Acc. ↑ | Macro-F1 ↑ |
|---|---|---|---|
| FOCAL | 41.38% | 45.40% | 0.376 |
| SimMTM | 55.17% | 61.55% | 0.532 |
| RelCon | 42.53% | 50.94% | 0.419 |
| Scratch | 60.92% | 48.64% | 0.488 |
| Ours (MAE) | 59.77% | 71.32% | 0.606 |
| Ours (MAE), finetuned | **77.01%** | 67.78% | **0.704** |
| Ours (MAE), no alt-attn | 65.52% | **73.24%** | 0.647 |
| Ours (MAE), acc-only | 58.62% | 67.29% | 0.586 |
| Mix-Token (reimpl.) | 62.07% | 52.63% | 0.500 |
| NormWear-Base (first-party) | 51.72% | 47.63% | 0.453 |

**window-level, test split, trained on real recordings only**

| Model | Acc. ↑ | Bal. Acc. ↑ | Macro-F1 ↑ |
|---|---|---|---|
| FOCAL | 37.93% | 39.09% | 0.337 |
| SimMTM | 47.13% | 51.04% | 0.447 |
| RelCon | 42.15% | 48.46% | 0.407 |
| Scratch | 51.92% | 40.37% | 0.403 |
| Ours (MAE) | 50.77% | 56.44% | 0.490 |
| Ours (MAE), finetuned | **67.24%** | 55.23% | **0.580** |
| Ours (MAE), no alt-attn | 55.17% | **58.70%** | 0.524 |
| Ours (MAE), acc-only | 49.62% | 54.56% | 0.478 |
| Mix-Token (reimpl.) | 59.20% | 50.03% | 0.490 |
| NormWear-Base (first-party) | 50.38% | 45.24% | 0.425 |

## LaTeX

```latex
\begin{table}[t]
\centering
\caption{Microgesture recognition on OpenWatch under the OpenWatch protocol (clip-level, $k{=}3$; trained with OpenWatch augmentation; IMU-only; test split). All rows share one contamination-safe posture-stratified subject split, the same 1\,s/stride-20 windows (6 per clip), class-weighted CE and val macro-F1 model selection. Mix-Token is our reimplementation (no official code); NormWear-Base is the official checkpoint with the paper's head recipe. Higher is better.}
\label{tab:openwatch-aug-clip}
\begin{tabular}{lccc}
\toprule
Model & Acc.\ $\uparrow$ & Bal.\ Acc.\ $\uparrow$ & Macro-F1 $\uparrow$ \\
\midrule
FOCAL & 40.23\% & 48.94\% & 0.407 \\
SimMTM & 51.72\% & 62.87\% & 0.515 \\
RelCon & 44.83\% & 53.90\% & 0.443 \\
Scratch & 51.72\% & 33.86\% & 0.349 \\
\midrule
Ours (MAE) & 59.77\% & 72.50\% & 0.615 \\
Ours (MAE), finetuned & 66.67\% & 56.65\% & 0.580 \\
Ours (MAE), no alt-attn & 57.47\% & 68.06\% & 0.582 \\
Ours (MAE), acc-only & 57.47\% & 65.32\% & 0.564 \\
\midrule
Mix-Token (reimpl.) & 64.37\% & 69.62\% & 0.641 \\
NormWear-Base (first-party) & 54.02\% & 57.48\% & 0.494 \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\begin{table}[t]
\centering
\caption{Microgesture recognition on OpenWatch under the OpenWatch protocol (window-level; trained with OpenWatch augmentation; IMU-only; test split). All rows share one contamination-safe posture-stratified subject split, the same 1\,s/stride-20 windows (6 per clip), class-weighted CE and val macro-F1 model selection. Mix-Token is our reimplementation (no official code); NormWear-Base is the official checkpoint with the paper's head recipe. Higher is better.}
\label{tab:openwatch-aug-window}
\begin{tabular}{lccc}
\toprule
Model & Acc.\ $\uparrow$ & Bal.\ Acc.\ $\uparrow$ & Macro-F1 $\uparrow$ \\
\midrule
FOCAL & 36.21\% & 39.01\% & 0.335 \\
SimMTM & 43.30\% & 49.75\% & 0.415 \\
RelCon & 38.51\% & 45.91\% & 0.377 \\
Scratch & 49.81\% & 35.24\% & 0.354 \\
\midrule
Ours (MAE) & 49.43\% & 55.57\% & 0.482 \\
Ours (MAE), finetuned & 63.41\% & 52.69\% & 0.544 \\
Ours (MAE), no alt-attn & 50.19\% & 57.23\% & 0.494 \\
Ours (MAE), acc-only & 48.66\% & 54.28\% & 0.471 \\
\midrule
Mix-Token (reimpl.) & 57.66\% & 58.31\% & 0.542 \\
NormWear-Base (first-party) & 49.04\% & 53.05\% & 0.454 \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\begin{table}[t]
\centering
\caption{Microgesture recognition on OpenWatch under the OpenWatch protocol (clip-level, $k{=}3$; trained on real recordings only; IMU-only; test split). All rows share one contamination-safe posture-stratified subject split, the same 1\,s/stride-20 windows (6 per clip), class-weighted CE and val macro-F1 model selection. Mix-Token is our reimplementation (no official code); NormWear-Base is the official checkpoint with the paper's head recipe. Higher is better.}
\label{tab:openwatch-noaug-clip}
\begin{tabular}{lccc}
\toprule
Model & Acc.\ $\uparrow$ & Bal.\ Acc.\ $\uparrow$ & Macro-F1 $\uparrow$ \\
\midrule
FOCAL & 41.38\% & 45.40\% & 0.376 \\
SimMTM & 55.17\% & 61.55\% & 0.532 \\
RelCon & 42.53\% & 50.94\% & 0.419 \\
Scratch & 60.92\% & 48.64\% & 0.488 \\
\midrule
Ours (MAE) & 59.77\% & 71.32\% & 0.606 \\
Ours (MAE), finetuned & 77.01\% & 67.78\% & 0.704 \\
Ours (MAE), no alt-attn & 65.52\% & 73.24\% & 0.647 \\
Ours (MAE), acc-only & 58.62\% & 67.29\% & 0.586 \\
\midrule
Mix-Token (reimpl.) & 62.07\% & 52.63\% & 0.500 \\
NormWear-Base (first-party) & 51.72\% & 47.63\% & 0.453 \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\begin{table}[t]
\centering
\caption{Microgesture recognition on OpenWatch under the OpenWatch protocol (window-level; trained on real recordings only; IMU-only; test split). All rows share one contamination-safe posture-stratified subject split, the same 1\,s/stride-20 windows (6 per clip), class-weighted CE and val macro-F1 model selection. Mix-Token is our reimplementation (no official code); NormWear-Base is the official checkpoint with the paper's head recipe. Higher is better.}
\label{tab:openwatch-noaug-window}
\begin{tabular}{lccc}
\toprule
Model & Acc.\ $\uparrow$ & Bal.\ Acc.\ $\uparrow$ & Macro-F1 $\uparrow$ \\
\midrule
FOCAL & 37.93\% & 39.09\% & 0.337 \\
SimMTM & 47.13\% & 51.04\% & 0.447 \\
RelCon & 42.15\% & 48.46\% & 0.407 \\
Scratch & 51.92\% & 40.37\% & 0.403 \\
\midrule
Ours (MAE) & 50.77\% & 56.44\% & 0.490 \\
Ours (MAE), finetuned & 67.24\% & 55.23\% & 0.580 \\
Ours (MAE), no alt-attn & 55.17\% & 58.70\% & 0.524 \\
Ours (MAE), acc-only & 49.62\% & 54.56\% & 0.478 \\
\midrule
Mix-Token (reimpl.) & 59.20\% & 50.03\% & 0.490 \\
NormWear-Base (first-party) & 50.38\% & 45.24\% & 0.425 \\
\bottomrule
\end{tabular}
\end{table}
```

## Cited (Bonazzi et al. Table 2 — their split, their run; context only)

| Model | Window Acc | Window F1 | Clip Acc (k=3) | Clip F1 |
|---|---|---|---|---|
| Mix-Token (aug) | 90.8% | 0.768 | 95.5% | 0.903 |
| Apple-CNN (aug) | 77.4% | 0.576 | 87.9% | 0.756 |
| NormWear-Base (aug; possibly +PPG) | 68.2% | 0.430 | 78.8% | 0.565 |
| NormWear-Base* (no-aug, IMU-only) | 37.0% | 0.340 | 69.0% | 0.620 |
