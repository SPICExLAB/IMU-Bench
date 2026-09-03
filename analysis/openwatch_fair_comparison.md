# OpenWatch fair comparison (§5f protocol)

Identical split (contamination-safe, posture-stratified, canonical identities: 33/5/5; val=102 clips/612 windows, test=87 clips/522 windows), identical 6 windows/clip (1s, stride 20, centered 2.0s sub-interval), identical k=3 clip voting and metric code. Head recipe now IDENTICAL across all rows: inverse-frequency class-weighted CE, model selection on val window macro-F1 (paper §3.3.3). ViT1D rows tagged [unweighted CE, val_loss sel.] are the repo-default recipe kept as an ablation. **Test is the paper-comparable split.**

### Trained on augmented release, evaluated on real recordings (vs. paper headline rows)

| Model | Params | Split | Window F1 | Window Acc | Clip F1 (k=3) | Clip Acc (k=3) |
|---|---|---|---|---|---|---|
| Ours (MAE) probe | ~21M | val | 0.553 | 0.580 | 0.638 | 0.627 |
| Ours (MAE) probe | ~21M | test | 0.482 | 0.494 | 0.615 | 0.598 |
| Ours (MAE) finetune | ~21M | val | 0.704 | 0.778 | 0.783 | 0.833 |
| Ours (MAE) finetune | ~21M | test | 0.544 | 0.634 | 0.580 | 0.667 |
| Ours (MAE) probe [unweighted CE, val_loss sel.] | ~21M | val | 0.580 | 0.698 | 0.738 | 0.804 |
| Ours (MAE) probe [unweighted CE, val_loss sel.] | ~21M | test | 0.466 | 0.611 | 0.530 | 0.667 |
| Ours (MAE) finetune [unweighted CE, val_loss sel.] | ~21M | val | 0.667 | 0.758 | 0.794 | 0.863 |
| Ours (MAE) finetune [unweighted CE, val_loss sel.] | ~21M | test | 0.530 | 0.626 | 0.586 | 0.678 |
| Inertia-1 PatchTST-small (NHANES repro) probe | ~8.5M | val | 0.465 | 0.475 | 0.541 | 0.539 |
| Inertia-1 PatchTST-small (NHANES repro) probe | ~8.5M | test | 0.470 | 0.479 | 0.532 | 0.529 |
| Inertia-1 PatchTST-small (NHANES repro) finetune | ~8.5M | val | 0.535 | 0.631 | 0.662 | 0.735 |
| Inertia-1 PatchTST-small (NHANES repro) finetune | ~8.5M | test | 0.501 | 0.567 | 0.585 | 0.632 |
| Inertia-1 ART-small (NHANES repro) probe | ~8.5M | val | 0.500 | 0.520 | 0.571 | 0.569 |
| Inertia-1 ART-small (NHANES repro) probe | ~8.5M | test | 0.457 | 0.489 | 0.532 | 0.552 |
| Inertia-1 ART-small (NHANES repro) finetune | ~8.5M | val | 0.548 | 0.657 | 0.617 | 0.735 |
| Inertia-1 ART-small (NHANES repro) finetune | ~8.5M | test | 0.485 | 0.596 | 0.575 | 0.667 |
| Mix-Token (reimpl.) | 219K | val | 0.575 | 0.618 | 0.608 | 0.657 |
| Mix-Token (reimpl.) | 219K | test | 0.542 | 0.577 | 0.641 | 0.644 |
| NormWear-Base (first-party) | 136M | val | 0.529 | 0.547 | 0.579 | 0.588 |
| NormWear-Base (first-party) | 136M | test | 0.454 | 0.490 | 0.494 | 0.540 |

*Cited from Bonazzi et al. Table 2 — their split, their run; context only, NOT the same split:*

| Model (cited) | Params | Window F1 | Window Acc | Clip F1 | Clip Acc |
|---|---|---|---|---|---|
| Mix-Token (cited, Bonazzi et al.) | 223K | 0.768 | 0.908 | 0.903 | 0.955 |
| Apple-CNN (cited) | 51K | 0.576 | 0.774 | 0.756 | 0.879 |
| NormWear-Base (cited; possibly +PPG) | 136M | 0.430 | 0.682 | 0.565 | 0.788 |

### Trained on real recordings only (vs. paper NormWear-Base* row)

| Model | Params | Split | Window F1 | Window Acc | Clip F1 (k=3) | Clip Acc (k=3) |
|---|---|---|---|---|---|---|
| Ours (MAE) probe | ~21M | val | 0.561 | 0.577 | 0.674 | 0.676 |
| Ours (MAE) probe | ~21M | test | 0.490 | 0.508 | 0.606 | 0.598 |
| Ours (MAE) finetune | ~21M | val | 0.664 | 0.747 | 0.784 | 0.824 |
| Ours (MAE) finetune | ~21M | test | 0.580 | 0.672 | 0.704 | 0.770 |
| Ours (MAE) probe [unweighted CE, val_loss sel.] | ~21M | val | 0.601 | 0.703 | 0.720 | 0.794 |
| Ours (MAE) probe [unweighted CE, val_loss sel.] | ~21M | test | 0.481 | 0.609 | 0.536 | 0.667 |
| Ours (MAE) finetune [unweighted CE, val_loss sel.] | ~21M | val | 0.596 | 0.719 | 0.678 | 0.775 |
| Ours (MAE) finetune [unweighted CE, val_loss sel.] | ~21M | test | 0.539 | 0.648 | 0.572 | 0.690 |
| Inertia-1 PatchTST-small (NHANES repro) probe | ~8.5M | val | 0.486 | 0.515 | 0.542 | 0.569 |
| Inertia-1 PatchTST-small (NHANES repro) probe | ~8.5M | test | 0.485 | 0.496 | 0.517 | 0.529 |
| Inertia-1 PatchTST-small (NHANES repro) finetune | ~8.5M | val | 0.531 | 0.654 | 0.644 | 0.735 |
| Inertia-1 PatchTST-small (NHANES repro) finetune | ~8.5M | test | 0.448 | 0.544 | 0.476 | 0.575 |
| Inertia-1 ART-small (NHANES repro) probe | ~8.5M | val | 0.514 | 0.542 | 0.594 | 0.598 |
| Inertia-1 ART-small (NHANES repro) probe | ~8.5M | test | 0.476 | 0.508 | 0.505 | 0.529 |
| Inertia-1 ART-small (NHANES repro) finetune | ~8.5M | val | 0.544 | 0.647 | 0.661 | 0.735 |
| Inertia-1 ART-small (NHANES repro) finetune | ~8.5M | test | 0.459 | 0.546 | 0.549 | 0.632 |
| Mix-Token (reimpl.) | 219K | val | 0.589 | 0.672 | 0.668 | 0.735 |
| Mix-Token (reimpl.) | 219K | test | 0.490 | 0.592 | 0.500 | 0.621 |
| NormWear-Base (first-party) | 136M | val | 0.592 | 0.621 | 0.598 | 0.637 |
| NormWear-Base (first-party) | 136M | test | 0.425 | 0.504 | 0.453 | 0.517 |

*Cited from Bonazzi et al. Table 2 — their split, their run; context only, NOT the same split:*

| Model (cited) | Params | Window F1 | Window Acc | Clip F1 | Clip Acc |
|---|---|---|---|---|---|
| NormWear-Base* (cited, no-aug IMU-only) | 136M | 0.340 | 0.370 | 0.620 | 0.690 |

## LaTeX

```latex
\begin{table}[t]
\centering
\caption{OpenWatch microgesture recognition under the OpenWatch protocol (1\,s windows, stride 20, 6 windows/clip, $k{=}3$ clip voting), IMU-only, test split. All rows: identical contamination-safe posture-stratified subject split, identical windows, identical metric code. Mix-Token is our reimplementation (no official code); NormWear-Base is the official checkpoint with the paper's head recipe.}
\label{tab:openwatch-fair}
\begin{tabular}{llcccc}
\toprule
Model & Params & Win.\ F1 & Win.\ Acc & Clip F1 & Clip Acc \\
\midrule
\multicolumn{6}{l}{\emph{Trained with OpenWatch augmentation, evaluated on real recordings}} \\
Ours (MAE) probe & ~21M & 0.482 & 0.494 & 0.615 & 0.598 \\
Ours (MAE) finetune & ~21M & 0.544 & 0.634 & 0.580 & 0.667 \\
Ours (MAE) probe [unweighted CE, val_loss sel.] & ~21M & 0.466 & 0.611 & 0.530 & 0.667 \\
Ours (MAE) finetune [unweighted CE, val_loss sel.] & ~21M & 0.530 & 0.626 & 0.586 & 0.678 \\
Inertia-1 PatchTST-small (NHANES repro) probe & ~8.5M & 0.470 & 0.479 & 0.532 & 0.529 \\
Inertia-1 PatchTST-small (NHANES repro) finetune & ~8.5M & 0.501 & 0.567 & 0.585 & 0.632 \\
Inertia-1 ART-small (NHANES repro) probe & ~8.5M & 0.457 & 0.489 & 0.532 & 0.552 \\
Inertia-1 ART-small (NHANES repro) finetune & ~8.5M & 0.485 & 0.596 & 0.575 & 0.667 \\
Mix-Token (reimpl.) & 219K & 0.542 & 0.577 & 0.641 & 0.644 \\
NormWear-Base (first-party) & 136M & 0.454 & 0.490 & 0.494 & 0.540 \\
\midrule
\multicolumn{6}{l}{\emph{Trained on real recordings only}} \\
Ours (MAE) probe & ~21M & 0.490 & 0.508 & 0.606 & 0.598 \\
Ours (MAE) finetune & ~21M & 0.580 & 0.672 & 0.704 & 0.770 \\
Ours (MAE) probe [unweighted CE, val_loss sel.] & ~21M & 0.481 & 0.609 & 0.536 & 0.667 \\
Ours (MAE) finetune [unweighted CE, val_loss sel.] & ~21M & 0.539 & 0.648 & 0.572 & 0.690 \\
Inertia-1 PatchTST-small (NHANES repro) probe & ~8.5M & 0.485 & 0.496 & 0.517 & 0.529 \\
Inertia-1 PatchTST-small (NHANES repro) finetune & ~8.5M & 0.448 & 0.544 & 0.476 & 0.575 \\
Inertia-1 ART-small (NHANES repro) probe & ~8.5M & 0.476 & 0.508 & 0.505 & 0.529 \\
Inertia-1 ART-small (NHANES repro) finetune & ~8.5M & 0.459 & 0.546 & 0.549 & 0.632 \\
Mix-Token (reimpl.) & 219K & 0.490 & 0.592 & 0.500 & 0.621 \\
NormWear-Base (first-party) & 136M & 0.425 & 0.504 & 0.453 & 0.517 \\
\bottomrule
\end{tabular}
\end{table}
```
