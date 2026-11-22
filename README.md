# Pilates Identification

Exercise video selection and download tooling for a Pilates identification project.
The catalog currently contains 1,160 Lagree Micro exercise videos, with named
movement groups and a `training_data` selection column.

## Setup

Requires Node.js and Google Chrome.

```sh
npm install --prefix tools/lagree
cp .env.example .env
```

Fill in your Lagree credentials in `.env`. Credentials, downloaded videos, and
browser sessions are excluded from Git.

## Choose and download training videos

Edit `output/lagree/micro-exercises.csv`. Set `training_data` to `1` to select a
video, or `0` to leave it out. The `group` column can contain multiple groups
separated by semicolons.

```sh
node tools/lagree/download.cjs --list
node tools/lagree/download.cjs
```

Videos save to `data/raw/` with lowercase names and underscores. Existing
files are skipped; use `--force` to replace them. Download only videos you have
permission to save.

## Refresh the catalog

```sh
node tools/lagree/catalog.cjs
```

Refreshes the catalog while preserving training selections by exercise ID.
See `tools/lagree/README.md` for details.

## Extract pose landmarks

The Python pipeline preserves per-frame timestamps and all 33 MediaPipe pose
landmarks, including missing detections and confidence masks. It exports CSV,
ordered NumPy arrays, quality statistics, and a skeleton preview for each video.
See [setup, data schema, and modeling notes](docs/pose-pipeline.md).

```sh
.venv/bin/python tools/pose/extract.py
.venv/bin/python tools/pose/combine.py
```

Processes every supported video under `data/raw/`, including subfolders, and
saves per-video results to `data/processed/landmarks/<video_id>/`. The extractor
also works with uncataloged videos. See [data folder layout](data/README.md).
The second command combines completed sequences into `data/processed/landmarks.csv`
with a `video_name` column, preserving each video's frame order and timestamps.

## Joint-angle features

```sh
.venv/bin/python tools/pose/features.py
```

Reads the combined landmark CSV and writes `data/processed/features/frames.csv`.
Adds left/right elbow, knee, and hip angles, timestamp-based lags, trailing
averages, lagged averages, and angle changes per second, each with a validity flag. Original columns, frame order, and missing detections are retained.
See [angle definitions and missing-data handling](docs/pose-pipeline.md#joint-angle-features).

## Classify two-second snippets

```sh
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python tools/pose/train_lstm.py --output data/processed/experiments/my_run
```

A six-class LSTM pilot uses 60-frame windows with training-only scaling and
protected same-recording test chunks. See [training, paired noisy tests, saved
model prediction and limitations](docs/pose-classification.md). The completed
revised run is local at `data/processed/experiments/lstm_pilot_v2/`.
