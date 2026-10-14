# Roadmap

Planned, in rough order:

- **Lite subset.** One task per family with a one-GPU-hour budget, for development runs.
- **Docker image.** An official image with pinned dependencies, and the requirement that leaderboard runs use it.
- **Spot re-runs.** Every open-track submission with public weights re-run on a sample of cells, with the outcome
  reported on the pull request.
- **Embeddings mode.** A frozen-probe-only path for the global tasks, listed outside the ranking, for models whose
  weights cannot be released.
- **Data.** The Nymeria preprocessing converter, the TransfHAR release and the regenerated HumanML3D wrist simulation
  (see the gaps in [Datasets](guides/datasets.md)).
- **Multi-seed rows** for the leaderboard's top entries.
