# Tracks and leaderboard submission

## Tracks

Both tracks use the same task cards, protocols and scorers; they differ in what the encoder may be.

=== "Controlled track"

    *With the corpus and the model held fixed, which pretraining objective gives the most useful representation?*

    - Pretrain only on the released corpus: 197,094 unlabeled wrist windows built with
      `scripts/build_multidata_cache.py` from the sources listed in [Datasets](datasets.md).
    - Use the shared backbone (12-layer, width-384 ViT over 0.32 s patches, 21 M parameters) and the shared
      pretraining trainer (`scripts/run_pretrain.py`).
    - At most 160,000 pretraining updates, checkpoint selected on the corpus validation split.
    - Only the objective (loss, views, masking, heads discarded after pretraining) is free.

=== "Open track"

    *What is the best wrist-IMU encoder that anyone can train or download, at any scale?*

    - Any architecture, size, pretraining corpus and objective.
    - Weights must be public, or released with the submission.
    - The pretraining corpus must not contain any recording from the evaluation splits of the ten tasks; declare it
      in the card (`pretrain_data`) so that it can be audited.
    - Input normalisation and other preprocessing inside the encoder are the submitter's choice (`normalize_method`).

## Submitting

Submissions are pull requests to the public results repository, the way SWE-bench collects experiments.

```bash
imu-bench check my_encoder
imu-bench package my_encoder --out submission/
```

`package` lays out `results/<task>/…json`, `registry/my_encoder.yaml`, the generated `configs/` and `check.json`.

```python
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id="YiquanLi/IMU-Bench-results", repo_type="dataset",
                      folder_path="submission", create_pr=True, commit_message="Add my_encoder")
```

What happens next:

1. We run `imu-bench check` on the pull request; a missing cell, a changed recipe or an incomplete card is reported
   on the PR.
2. We check the declared pretraining corpus against the evaluation sets ([Data audit and leakage](audit.md)). A row
   whose corpus contains evaluation recordings is listed as *overlap* and left out of the ranking of the affected
   tasks.
3. For open-track submissions with public weights we re-run a sample of cells; a mismatch beyond the printed
   precision is reported on the PR.
4. After merging, the leaderboard on the website and in the Space is regenerated from the merged files.

!!! tip "Partial submissions"
    `imu-bench package --allow-incomplete` is accepted; the row appears in the per-task tables and is ranked once
    all twenty cells exist.
