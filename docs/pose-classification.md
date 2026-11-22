# Two-second LSTM classification pilot

This experiment classifies a two-second landmark snippet into one of six exercise
classes. It is a **same-recording pilot**, not a test of unseen people, cameras,
or independent recordings.

## Run

```sh
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python tools/pose/train_lstm.py --output data/processed/experiments/my_run
```

Use a fresh output directory to preserve earlier experiments. Input defaults to
`data/processed/landmarks.csv`. No new video downloads or MediaPipe extraction
are needed. The first run is retained in `lstm_pilot`; the revised split with a
minimum training-coverage check is saved in `lstm_pilot_v2`.

## Labels

`config/pose_classes.json` defines class indices and exercise-title mappings:

0. Bear (includes Reverse bear)
1. Elevator lunge
2. Plank to pike (includes version 2 and Reverse plank to pike)
3. Spider kick
4. Standing inner thigh
5. Standing outer thigh

Labels are integers and the loss is `sparse_categorical_crossentropy`; one-hot
encoding is not needed. Video IDs and titles are used only for grouping and
labeling, never as predictive features.

## Snippets and splitting

- Sample 60 ordered frames at 30 FPS for each two-second snippet. The pilot
  requires enough source frames to sample without repeating frames.
- Candidate start times are 0.5 seconds apart. Skip snippets with fewer than 50%
  valid raw angle values; do not fill these with fabricated poses.
- Provisionally trim the first 10 and last 5 seconds of each recording. These
  intervals are based on source titles and edge trimming, not verified frame
  annotations. Setup, transitions and camera cuts still need review.
- For each source video, reserve two random test chunks and one validation chunk
  using seed 42. These are chosen before fitting the model.
- Exclude training windows intersecting any held-out chunk or its 1.5-second
  buffer. Held-out chunks also have a separating guard interval. Assert that no
  source frames are shared across splits.
- Require at least `max(2, min(10, candidate_count // 3))` training windows per
  video; redraw chunk positions if coverage is insufficient. This rule uses
  available sample counts, not model accuracy.
- Training windows can overlap each other. They are not independent recordings.

The initial split left only two Spider kick training windows. The revised split
uses the coverage rule above and keeps nine; no model architecture was selected
using test performance. Both run folders are retained for transparency.

## Features and model

Recalculate angles and temporal features inside each snippet, resetting history
at its start. This prevents pre-snippet context leaking into an example. The first
frames therefore have missing lag/mean values until sufficient history exists.

Each frame contains 54 numeric values (angles, lagged angles, averages, lagged
averages, and signed angle rates) plus 54 validity indicators. Training-only means
and standard deviations scale numeric values. Missing values become the training
mean (zero after scaling), and validity indicators distinguish them from observed
zeros. Entirely missing training features use a mean of zero and scale of one.

The input is `(batch, 60, 108)`. A 32-unit LSTM feeds a six-class softmax output.
Adam minimizes sparse categorical cross-entropy. Training uses inverse-frequency
class weights, batch size 16, up to 100 epochs, and validation-loss early stopping
with patience 12 and restoration of the best weights. The validation set contains
only nine snippets, so model selection is itself uncertain.

No training augmentation is used in this initial experiment.

## Clean and noisy tests

Both tests use the exact same 18 held-out source chunks. The noisy condition adds
independent Gaussian noise with standard deviation 0.005 to finite world XYZ
coordinates, then recalculates angles and temporal features. MediaPipe reports
world coordinates in estimated meters, so this is a nominal 5 mm landmark-noise
experiment, not a calibrated real-world camera-noise model.

Confidence scores are retained. This tests sensitivity to coordinate jitter; it
does not simulate new subjects, camera angles, detection loss, or camera cuts.
Noise is not created from training chunks and presented as independent test data.

## Outputs

Each run saves:

- `model.keras`: trained model and class-probability output.
- `preprocessing.json`: feature order, training-only scaling, class order, seed,
  temporal settings and experiment configuration.
- `split_manifest.json`: chunk times, source frame indices and split membership.
- `metrics.json`: accuracy, macro F1, per-class metrics and confusion matrices for
  clean/noisy conditions, plus a training-majority-class baseline.
- `clean_test_predictions.csv`, `noisy_test_predictions.csv`: paired per-chunk
  labels, predictions and probabilities.
- `history.json`: training/validation learning curves.

The revised run has 455 overlapping training snippets, nine validation snippets,
and 18 test snippets. Its clean and noisy conditions both classified 17/18 chunks
correctly (94.4% accuracy; macro F1 0.926). This is a small pilot result under one
split and one noise level, not an estimate of performance on new recordings. The
same accuracy does not imply identical probability outputs or universal robustness.

## Predict a snippet using the saved model

```sh
.venv/bin/python tools/pose/predict_lstm.py \
  --input data/processed/landmarks.csv \
  --experiment data/processed/experiments/lstm_pilot_v2 \
  --video-id 4hDFKBRn --start 26.5
```

The input is an extracted landmark CSV, not an RGB video. New videos first go
through pose extraction and CSV combination. The command does not require a
known exercise label for the new video; it returns the predicted class and all
six probabilities. This initial model has no trained unknown class or calibrated
confidence threshold.
