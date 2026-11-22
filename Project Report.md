# Project Report

### Problem Statement

Action recognition from a short video snippet.

The goal is to identify a specific Lagree exercise from approximately two to three
seconds of new video. MediaPipe extracts body landmarks from each frame, and a
classifier uses the resulting movement sequence to predict an exercise label.
We are focusing on individual exercises first; broader categories can be
considered later.

### Common Techniques

LSTM and GRU models process ordered sequences and can learn how features change
over time. For example, a plank-to-pike sequence may contain changes in hip angle
that help distinguish it from a held plank position.

The initial pilot uses two-second examples: 60 frames at 30 frames per second.
A three-second example would contain approximately 90 frames. The model input has the shape:

```text
number of examples × frames per example × features per frame
```

During training, the model learns a mapping from labeled movement sequences to
exercise categories. During inference, it applies that learned mapping to a new
snippet. Test data evaluates this mapping; it is not used to train the model.

A simpler comparison is an SVM or MLP using summaries of each time window, such
as mean angles, angle ranges, and variability. These summaries contain less
information about movement order than an ordered sequence.

### Processing Pipeline

```text
Raw video → MediaPipe landmarks → angle and temporal features
          → labeled time windows → classifier → exercise prediction
```

- Input videos: `data/raw/`
- Combined landmarks: `data/processed/landmarks.csv`
- Landmarks with engineered features: `data/processed/features/frames.csv`

The current dataset contains nine video sequences and 17,785 frame rows. These
rows are correlated observations, not 17,785 independent training examples.
Landmark extraction, feature processing, and a preliminary six-class LSTM
experiment are implemented. This experiment uses held-out sections of the same
recordings; independent-recording evaluation remains future work.

### Current Features

Each frame retains its video identity, timestamp, landmarks, and confidence
information. Video names and IDs are metadata, not classifier inputs.

Joint angles are calculated separately for the left and right sides:

| Angle | Landmarks; angle measured at the middle joint |
| --- | --- |
| Elbow | Shoulder–elbow–wrist |
| Knee | Hip–knee–ankle |
| Hip | Shoulder–hip–knee |

Angles use MediaPipe's estimated 3D world coordinates and are measured in degrees.
An unreliable angle remains missing, with a validity flag, rather than becoming
zero.

Temporal features currently include:

- Angle lags of 0.1, 0.5, and 1 second.
- Trailing angle averages over 0.5 and 1 second.
- Each trailing average lagged by 0.5 seconds.
- Signed angle change per second, calculated from consecutive valid frames and actual timestamps.

These features use current or past observations only. History resets between
videos or annotated segments, across long frame gaps, and when an angle becomes
invalid. Some features require up to 1.5 seconds of valid history to become
available. Training and inference must use the same history and warm-up rules.

### Possible Additional Features

Joint velocity vectors could additionally describe landmark movement in 3D.
Angle changes per second are already included in the current features. A dense vector
field across the full image is not required for the initial skeleton-based model.

Averages may reduce landmark jitter, but they can also smooth out useful motion.
We can compare angles alone, angles with lags, and angles with lags and averages
to test whether these features improve validation performance.

### Training and Evaluation Plan

1. Mark usable exercise intervals, excluding setup, explanations, and transitions.
   Video titles alone do not provide reliable frame-level labels.
2. Split by recording or instructor before creating time windows. Keep related
   recordings together where possible to reduce leakage.
3. Create fixed-duration windows within labeled intervals (two seconds in the initial pilot). Training
   windows may overlap, but they must not overlap validation or test data.
4. Fit preprocessing on training data only. Use validation data for model and
   feature choices, then evaluate the selected approach on held-out test data.
5. Compare classification accuracy, per-class F1-scores, confusion matrices, and
   end-to-end inference latency.

Several classes currently have only one source recording. A held-out section of
the same recording can support a preliminary test, with a gap and feature-history
reset at the boundary, but it does not establish performance on a new person or
camera angle. More independent recordings are needed for that evaluation.

### Initial Two-Second Classification Experiment

The pilot uses 60 ordered frames at 30 FPS, with 54 numeric features and 54
validity indicators per frame. A 32-unit LSTM predicts one of six classes with
softmax. Integer labels use sparse categorical cross-entropy. Reverse Bear and
reverse/version-2 Plank to pike are grouped with their base exercises in the
editable class map.

The revised split contains 455 overlapping training windows, nine validation
windows, and 18 test windows. Test and validation chunks are separated from
training windows by 1.5-second buffers, and feature history resets for each chunk.
Scaling is fitted on training data only. A minimum training-coverage rule was
added because the first split left only two Spider kick training windows; the
revised split retains nine.

Clean test chunks and paired copies with nominal 5 mm coordinate noise both
produced 17 correct predictions out of 18 (94.4% accuracy; macro F1 0.926). Noise
is added before recalculating derived features. These are same-recording results
with provisional video-title labels and edge trimming, not evidence of
recognition on unseen instructors or cameras. The small test and validation
sets and overlapping training windows limit the conclusions.

See [experiment details](docs/pose-classification.md) for commands, artifacts,
class mappings and limitations.

The small [pilot results snapshot](reports/lstm_pilot/README.md) is included in the repository; bulk data and trained model files remain local.
