# Figure 4 (Sections/6_Scaling.tex) -- three variants of the eight-probe average

Drawn 2026-09-24 after the Parkinson's cells were re-scored with class-weighted CE,
to decide how the PD probe should enter the average.

- `A_subject.png`  PD = subject_macro_f1, eight probes. What the repo currently
  generates: the task's primary metric, so it matches Table 2.
- `B_window.png`   PD = window_macro_f1, eight probes. Cleaner, but the figure and
  the main table would then report this one task in different units.
- `C_noPD.png`     PD dropped, seven probes.

Why the question arises: with the head fixed, the single-motion-mode cell
(k1_f0p01) goes from 28.2 to 56.5 subject macro-F1 -- the highest value in the
whole column, above the full pool's 45.4. That is almost certainly the aggregation
noise of 94 participants, not an effect of the corpus: on this metric the
never-trained control beats five of the nine pretrained encoders.

Reproduce: patch line 60 of 33_plot_grid_nonleaky.py for B; skip the Parkinson's
panel in 34_plot_scaling_avg.py (and relax its `len(tasks) == 8` assert) for C.
Since 2026-09-24 (evening) the repo generates B (user choice): 33_plot_grid_nonleaky.py reads window_macro_f1.
