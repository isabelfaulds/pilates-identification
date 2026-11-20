# Data layout

```text
data/
  raw/                         Original input videos, including subfolders
  processed/
    landmarks/<video_id>/      Current extraction outputs
      frames.csv               Frame metadata and image/world landmarks
      sequence.npz             Ordered arrays, timestamps, confidence masks
      summary.json             Provenance and quality statistics
      preview.jpg              Visual review sheet
    features/                  Future engineered features and final labeled CSV
    windows/                   Future temporal model inputs
```

Raw videos and generated data are ignored by Git. The small sample preview in
`docs/examples/` is included for reviewing this pipeline.

Run `python tools/pose/extract.py` using the project's Python environment to
process all supported videos recursively in `data/raw/`. The Lagree downloader
also saves here. CSV `training_data` flags select downloads; they do not restrict
which local videos get processed.

Keep per-video landmarks as the reproducible intermediate stage. A final
training CSV can be built later after interval annotation, feature engineering,
and video/person-level dataset splitting.
