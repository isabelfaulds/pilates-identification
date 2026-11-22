# Same-recording LSTM pilot results

Small, versioned result snapshots from `data/processed/experiments/lstm_pilot_v2/`.
Raw videos, frame-level data, trained weights, and full generated outputs remain
local and ignored by Git.

- `metrics.json`: clean/noisy classification metrics, class counts, confusion
  matrices, selected epoch and limitations.
- `run_config.json`: class order, temporal settings, seed, source-data checksum,
  provisional trimming and split constraints.

These results use provisional video-title labels and held-out chunks from the
same recordings. They are not unseen-person or unseen-video accuracy estimates.
See [experiment details](../../docs/pose-classification.md).
