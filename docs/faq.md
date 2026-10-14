# FAQ

**My model only uses the accelerometer.**
Set `channels: 3` in the encoder card; the gyroscope channels are dropped before your model. The row is marked
accelerometer-only on the leaderboard.

**My model expects 30 Hz input and 10 s windows.**
Set `model_hz: 30`; the harness resamples (anti-aliased when downsampling). Window lengths are fixed by the task
cards (1 s to 10.24 s); a model that needs longer context sees the task window only.

**My model normalises its own input.**
Set `normalize_method: raw` and do it inside `forward`. The benchmark fixes the raw input, not the normalisation;
the released encoders use per-window z-scoring, physical-unit and raw inputs are supported as well.

**Can I tune the probe or the fine-tuning recipe?**
No. The decoder, learning rate, budget and selection rule belong to the task card; `imu-bench check` compares every
trained config with the card, and a cell that changed them is listed outside the ranking. Fine-tuning is the
protocol for showing what the encoder reaches with adaptation.

**Can I submit embeddings instead of an encoder?**
Not at the moment; see [Submission modes](guides/submission-modes.md#embeddings-instead-of-an-encoder).

**I only ran some tasks.**
`imu-bench package --allow-incomplete` lays the results out and the pull request is accepted, but the row is not
ranked until all twenty cells (ten tasks × two protocols) exist.

**How long does a full run take?**
On one A100, a few hours for the twenty cells of a 21 M-parameter encoder; the kinematic fine-tunes dominate.

**A reference row does not reproduce the released number.**
Check the data preparation first (`Datasets`); the reference rows `control_random`, `control_raw`, `scratch` and
`mae_5src` exist so that a setup can be verified before a new model is trusted.

**Which seed?**
42 for every cell on the leaderboard. Differences inside the measurement noise are not separated; the paper reports
the noise band of the kinematic tasks and bold marks ties at the printed precision.

**Why is the leaderboard regenerated instead of edited?**
So that every number traces to a result file with its checkpoint step and config. `imu-bench leaderboard` rebuilds
the tables from `bench/results/`.
