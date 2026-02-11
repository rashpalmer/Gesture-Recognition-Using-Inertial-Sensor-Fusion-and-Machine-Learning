# Gesture Recognition Using Inertial Sensor Fusion and Machine Learning

**BEng EEE Final Year Project — Newcastle University, 2026**
**Author:** Rashaan Palmer

A MATLAB pipeline that classifies six hand gestures from smartphone IMU data (accelerometer, gyroscope, magnetometer) using a multi-stage sensor fusion architecture and machine learning. The system fuses raw inertial measurements through an Extended Kalman Filter for attitude estimation and a linear Kalman Filter for motion tracking, segments gesture boundaries from energy signals, extracts physically meaningful features, and classifies gestures at 95.6% accuracy using SVM with leave-one-out cross-validation.

The project targets industrial environments — specifically warehouses, where worker injury rates are five times the industry average — and uses the smartphone as a proof-of-concept platform before migration to dedicated wrist-worn hardware.

---

## Architecture & Pipeline Flow

The pipeline processes raw sensor data through six sequential stages. Each stage consumes a well-defined struct and produces the next, following the chain:

```
raw file → data → imu → est → seg → feat → cls
```

### Stage-by-Stage Breakdown

| Stage | Function | Input | Output | What It Does |
|-------|----------|-------|--------|--------------|
| 1. Import | `read_phone_data` | `.mat` or `.csv` file | `data` struct | Parses MATLAB Mobile exports into standardised `[t, acc, gyr, mag]` arrays |
| 2. Preprocess | `preprocess_imu` | `data`, `params` | `imu` struct | Low-pass filtering (2nd-order Butterworth at 25 Hz), gyro bias estimation from static segments, magnetometer hard-iron calibration, stationary detection, resampling to 100 Hz |
| 3. Attitude Estimation | `ekf_attitude_quat` | `imu`, `params` | `est` struct | 7-state Extended Kalman Filter (quaternion + gyro bias) with accelerometer gravity correction and optional magnetometer heading updates |
| 4. Motion Tracking | `kf_linear_motion` | `imu`, `est`, `params` | `motion` struct | 6-state linear Kalman Filter (velocity + position) using gravity-compensated world-frame acceleration with Zero-Velocity Updates (ZUPT) |
| 5. Segmentation | `segment_gesture` | `imu`, `params` | `seg` struct | Energy-based detection using normalised gyroscope magnitude and accelerometer deviation with hysteresis thresholds |
| 6a. Feature Extraction | `extract_features` | `imu`, `est`, `seg`, `params` | `feat` struct | 30+ time-domain and frequency-domain features including RMS, peak values, integrals, axis dominance ratios, zero-crossing counts, and pitch deltas |
| 6b. Classification | `classify_gesture_rules` | `feat`, `params` | `cls` struct | Rule-based decision tree using physical gesture characteristics |
| 6b. Classification (ML) | `ml_predict_baseline` | `feat`, `params` | `cls` struct | Trained ML model (kNN, SVM, Decision Tree, or Random Forest) loaded from `models/` |

### Entry Points

There are three main entry points depending on the task:

- **`main_gesture_demo`** — Run the full pipeline on a single recording. This is the primary demonstration script.
- **`run_training_robust`** — Process all labeled gesture folders, extract features, train and compare multiple classifiers (kNN, SVM, Decision Tree, Random Forest), and save the best model.
- **`run_batch_eval`** — Batch-evaluate the pipeline across a directory of recordings and generate a summary report.

Each entry point is a thin orchestrator that calls the same underlying functions in order, so changing any single module (e.g. improving segmentation) automatically takes effect everywhere.

### Data Flow Diagram

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
│ .mat / .csv  │────▶│ read_phone   │────▶│ data                 │
│ (raw file)   │     │ _data        │     │  .t .acc .gyr .mag   │
└──────────────┘     └──────────────┘     └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │ preprocess_imu       │
                                          │  LPF, bias, calib    │
                                          └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                          ┌───────────────│ imu                  │───────────────┐
                          │               │  .t .acc .gyr .mag   │               │
                          │               │  .flags.stationary   │               │
                          │               └──────────────────────┘               │
                          ▼                                                      ▼
               ┌──────────────────┐                                   ┌──────────────────┐
               │ ekf_attitude     │                                   │ segment_gesture   │
               │ _quat            │                                   │  energy-based     │
               └────────┬─────────┘                                   └────────┬─────────┘
                        │                                                      │
                        ▼                                                      ▼
               ┌──────────────────┐                                   ┌──────────────────┐
               │ est              │                                   │ seg              │
               │  .q .euler .b_g  │                                   │  .winIdx .score  │
               └────────┬─────────┘                                   └────────┬─────────┘
                        │                                                      │
                        ▼                                                      │
               ┌──────────────────┐                                            │
               │ kf_linear_motion │                                            │
               │  ZUPT-corrected  │                                            │
               └────────┬─────────┘                                            │
                        │                                                      │
                        └──────────────────┬───────────────────────────────────┘
                                           ▼
                                ┌──────────────────────┐
                                │ extract_features      │
                                │  30+ features         │
                                └──────────┬───────────┘
                                           │
                                           ▼
                                ┌──────────────────────┐
                                │ classify (rules / ML) │
                                │  → label + confidence │
                                └──────────────────────┘
```

---

## Repository Structure

```
gesture-recognition-sensor-fusion/
│
├── README.md                           ← You are here
│
├── src/
│   ├── main/                           ← Entry-point scripts (thin orchestrators)
│   │   ├── main_gesture_demo.m         ← Single-file demo: full pipeline end-to-end
│   │   ├── run_training.m              ← Train ML classifier from labeled data
│   │   ├── run_training_robust.m       ← Multi-classifier comparison pipeline
│   │   ├── run_batch_eval.m            ← Batch evaluation across a directory
│   │   ├── run_ml_pipeline.m           ← Standalone ML training & evaluation
│   │   └── MLTRAININGEVALUATIONPIPELINE.m  ← Legacy monolithic training script
│   │
│   ├── io/                             ← Data import/export
│   │   ├── read_phone_data.m           ← Parse .mat/.csv from MATLAB Mobile
│   │   ├── convert_mobile_data.m       ← Convert MATLAB Mobile timetable format
│   │   └── export_helpers.m            ← Export results, features, logs, reports
│   │
│   ├── preprocess/                     ← Signal conditioning
│   │   ├── preprocess_imu.m            ← LPF, bias removal, calibration, resampling
│   │   ├── calibrate_mag_simple.m      ← Magnetometer hard/soft-iron calibration
│   │   └── resample_signals.m          ← Resample to uniform rate with gap handling
│   │
│   ├── fusion/                         ← Sensor fusion algorithms
│   │   ├── ekf_attitude_quat.m         ← 7-state EKF: quaternion attitude + gyro bias
│   │   ├── kf_linear_motion.m          ← 6-state linear KF: velocity + position (ZUPT)
│   │   └── complementary_filter.m      ← Simple complementary filter (baseline)
│   │
│   ├── gestures/                       ← Gesture detection and classification
│   │   ├── segment_gesture.m           ← Energy-based gesture boundary detection
│   │   ├── extract_features.m          ← 30+ time/frequency domain features
│   │   └── classify_gesture_rules.m    ← Rule-based classifier (transparent baseline)
│   │
│   ├── ml/                             ← Machine learning
│   │   ├── ml_train_baseline.m         ← Train kNN/SVM/Tree/RF with cross-validation
│   │   └── ml_predict_baseline.m       ← Predict using trained model with fallback
│   │
│   ├── viz/                            ← Plotting (isolated from algorithms)
│   │   ├── plot_diagnostics.m          ← 5-figure pipeline diagnostic suite
│   │   ├── plot_gesture_segment.m      ← Single gesture segment visualisation
│   │   └── generate_poster_figures.m   ← Publication-quality poster figures
│   │
│   ├── utils/                          ← Shared utilities
│   │   ├── config_params.m             ← ALL tuning parameters (single source of truth)
│   │   ├── quat_utils.m               ← Hamilton quaternion library (struct of handles)
│   │   ├── time_utils.m               ← Time operations, resampling, static detection
│   │   └── assert_utils.m             ← Input validation and assertion helpers
│   │
│   ├── tests/                          ← Unit test suite (MATLAB Unit Testing Framework)
│   │   ├── test_quat_math.m           ← 16 tests: quaternion operations and identities
│   │   ├── test_ekf_static.m          ← 8 tests: EKF convergence on synthetic static data
│   │   ├── test_segmentation.m        ← 8 tests: energy segmentation edge cases
│   │   ├── test_import.m              ← 8 tests: file loading and struct validation
│   │   ├── test_feature_extraction.m  ← Feature extraction correctness
│   │   └── test_quat_utils.m          ← Additional quaternion utility tests
│   │
│   └── experimental/                   ← Prototypes and standalone tests
│       ├── realtime_gesture_processor.m  ← Streaming gesture recognition (handle class)
│       ├── generate_synthetic_gesture.m  ← Synthetic IMU data generator for testing
│       ├── test_directML_bypass.m        ← Direct ML model test without pipeline
│       └── test_on_hled_out_data.m       ← Held-out evaluation script
│
├── data/                               ← Sensor recordings (never edited by code)
│   ├── CircleData/                     ← circle gesture recordings (.mat)
│   ├── Flip-UpData/                    ← flip_up gesture recordings
│   ├── Flip-DownData/                  ← flip_down gesture recordings
│   ├── ShakeData/                      ← shake gesture recordings
│   ├── TwistData/                      ← twist gesture recordings
│   └── PushForwardData/               ← push_forward gesture recordings
│
├── models/                             ← Trained ML models (generated by training)
│   └── gesture_model.mat              ← Saved classifier + normalisation parameters
│
└── outputs/                            ← Generated outputs (gitignored)
    ├── figures/                        ← Diagnostic plots
    ├── poster_figures/                 ← Publication-quality figures
    └── logs/                           ← Run logs (run_YYYYMMDD_HHMM.mat)
```

---

## Key Components

### Sensor Fusion Core

**`ekf_attitude_quat.m`** — The primary sensor fusion algorithm. Implements a 7-state Extended Kalman Filter with state vector `[q₀ q₁ q₂ q₃ bx by bz]ᵀ` (unit quaternion + gyroscope bias). The prediction step propagates orientation using bias-corrected gyroscope readings via quaternion kinematics. The update step corrects roll and pitch using accelerometer gravity measurements, and optionally corrects yaw using magnetometer heading. Includes accelerometer magnitude gating (rejects measurements outside 8–12 m/s² window to avoid corrupting estimates during dynamic motion), magnetometer anomaly rejection, and quaternion renormalisation at every step.

**`kf_linear_motion.m`** — A 6-state linear Kalman Filter estimating velocity and position in the world frame. Uses the attitude quaternion from the EKF to rotate body-frame acceleration into the world frame and subtract gravity. Applies Zero-Velocity Updates (ZUPT) during detected stationary periods to prevent unbounded integration drift. The ZUPT detector uses concurrent gyroscope and accelerometer stillness thresholds.

**`complementary_filter.m`** — A simpler baseline fusion algorithm using a weighted blend of gyroscope integration (fast, drifts) and accelerometer-derived tilt (slow, noisy). Configurable via `params.comp.alpha` (default 0.98 = 98% gyro trust). Exists for comparison against the EKF approach.

### Signal Processing

**`preprocess_imu.m`** — Applies a 2nd-order Butterworth low-pass filter at 25 Hz, estimates and removes gyroscope bias from initial static samples, calibrates the magnetometer using `calibrate_mag_simple`, detects stationary segments (used downstream by ZUPT and bias estimation), and optionally resamples all signals to a uniform 100 Hz.

**`segment_gesture.m`** — Detects gesture boundaries using a normalised energy signal computed from gyroscope magnitude and accelerometer deviation. Uses dual thresholds with hysteresis: `energy_high` (1.5) triggers gesture start, `energy_low` (0.5) triggers end. Enforces minimum/maximum duration constraints (0.2–3.0 s) and minimum inter-gesture gap (0.3 s). Returns up to `max_gestures` (5) ranked by confidence score.

### Feature Engineering

**`extract_features.m`** — Computes 30+ features from the segmented gesture window, including:

- **Time-domain:** RMS per gyro/accel axis, peak values, duration, zero-crossing count, total angular rotation, signed axis integrals
- **Orientation-derived:** Pitch delta (start-to-end), roll delta, yaw delta (from EKF output)
- **Axis dominance:** `rot_trans_ratio` (rotational vs translational energy — the single most important feature for classification), `gyr_x_dominance`, `gyr_y_dominance`, `roll_pitch_ratio`
- **Frequency-domain:** FFT-based spectral energy in configurable bands, dominant frequency per axis

### Classification

**`classify_gesture_rules.m`** — A transparent, hand-tuned decision tree using physically meaningful thresholds. Checks gesture characteristics in frequency-optimised order with early termination for high-confidence matches. Useful as a baseline and for understanding what distinguishes each gesture class.

**`ml_train_baseline.m`** / **`ml_predict_baseline.m`** — Trains and evaluates kNN, SVM, Decision Tree, and Random Forest classifiers with optional feature standardisation, stratified cross-validation, and per-class precision/recall/F1 metrics. The best model (currently SVM) achieves 95.6% accuracy on leave-one-out cross-validation across 160 samples.

### Quaternion Library

**`quat_utils.m`** — A struct-of-function-handles library implementing Hamilton convention quaternion math. All quaternion functions return 4×1 column vectors. Key functions: `normalize`, `multiply`, `conjugate`, `rotate`, `rotateBatch`, `fromEuler`, `toEuler` (returns 3 separate scalars: `[roll, pitch, yaw]`), `fromAxisAngle`, `fromOmega`, `slerp`, `fromRotMat`, `toRotMat`.

**Important:** `quat_utils` uses a handle-dispatch pattern, not string dispatch:
```matlab
qu = quat_utils();          % Get function handle struct
qn = qu.normalize(q);       % Call via handle
[r, p, y] = qu.toEuler(q);  % toEuler returns 3 separate outputs
```

---

## Setup & Installation

### Prerequisites

- **MATLAB R2022b or later** (tested on R2024a)
  - Signal Processing Toolbox (for `butter`, `filtfilt`)
  - Statistics and Machine Learning Toolbox (for `fitcknn`, `fitcsvm`, `fitctree`, `fitcensemble`, `crossval`)
- **iPhone with MATLAB Mobile** (for data collection) — or use the synthetic data generator for testing

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rashpalmer/gesture-recognition-sensor-fusion.git
   cd gesture-recognition-sensor-fusion
   ```

2. Open MATLAB and navigate to the repository root:
   ```matlab
   cd('path/to/gesture-recognition-sensor-fusion')
   ```

3. Add all source files to the MATLAB path:
   ```matlab
   addpath(genpath('src'));
   ```
   The main scripts do this automatically, but you need it if calling functions directly.

4. Verify the installation by running the test suite:
   ```matlab
   results = [runtests('test_quat_math'), runtests('test_ekf_static'), ...
              runtests('test_segmentation'), runtests('test_import')];
   fprintf('=== %d Passed, %d Failed ===\n', sum([results.Passed]), sum([results.Failed]));
   ```
   Expected: **40 Passed, 0 Failed**.

### Data Collection

Record gesture data using MATLAB Mobile on an iPhone:

1. Open MATLAB Mobile → Sensors → enable Accelerometer, Gyroscope, Magnetometer
2. Set sample rate to 100 Hz
3. Hold the phone naturally, perform the gesture, stop recording
4. Transfer the `.mat` file to the appropriate folder under `data/`

File naming convention: `<gesture_label>_<number>.mat` (e.g. `twist_003.mat`). The training pipeline infers the label from either the filename or the parent folder name.

---

## How to Run

### Single Gesture Demo (full pipeline)

```matlab
% Run on a specific recording
main_gesture_demo('data/TwistData/twist_001.mat')

% Run with file selection dialog
main_gesture_demo
```

This prints a 9-stage progress log and displays diagnostic plots showing every stage of the pipeline.

### Train ML Classifiers

```matlab
% Full multi-classifier comparison (recommended)
run_training_robust

% Basic training from labeled directory
run_training('data/labeled/')
```

`run_training_robust` processes all six gesture folders, extracts features, trains kNN/SVM/Decision Tree/Random Forest, prints a comparison table, and saves the best model to `models/gesture_model.mat`.

### Batch Evaluation

```matlab
% Evaluate on all files in a directory
run_batch_eval('data/TwistData/')

% Evaluate everything
run_batch_eval('data/')
```

### Run Test Suite

```matlab
% Full suite (40 tests)
results = [runtests('test_quat_math'), runtests('test_ekf_static'), ...
           runtests('test_segmentation'), runtests('test_import')];
fprintf('=== %d Passed, %d Failed ===\n', sum([results.Passed]), sum([results.Failed]));

% Individual test files
runtests('test_quat_math')       % 16 quaternion operation tests
runtests('test_ekf_static')      % 8 EKF convergence tests
runtests('test_segmentation')    % 8 segmentation edge-case tests
runtests('test_import')          % 8 data import tests
```

### Generate Poster Figures

```matlab
% Requires run_training_robust to have been executed first
% (needs allFeatures, allLabels, featureNames in workspace)
run_training_robust
generate_poster_figures
```

Saves 15 publication-quality PNGs to `outputs/poster_figures/`.

### Test with Synthetic Data (no hardware needed)

```matlab
addpath(genpath('src'));
params = config_params();

% Generate a synthetic twist gesture
data = generate_synthetic_gesture('twist', 'Duration', 1.5, 'NoiseLevel', 0.1);

% Run through the full pipeline
imu = preprocess_imu(data, params);
est = ekf_attitude_quat(imu, params);
motion = kf_linear_motion(imu, est, params);
seg = segment_gesture(imu, params);
feat = extract_features(imu, est, seg, params);
cls = classify_gesture_rules(feat, params);

fprintf('Detected: %s (%.0f%% confidence)\n', cls.label, cls.score * 100);
```

---

## Configuration

All tuning parameters live in a single file: **`src/utils/config_params.m`**. No magic numbers exist in algorithm functions — everything flows from `params`.

### Safe to Change

| Parameter Group | Key Fields | Effect |
|----------------|------------|--------|
| `params.sampling.targetFs` | Target sample rate | Change if your sensor runs at a different rate |
| `params.preprocess.lpf_cutoff` | Low-pass filter cutoff (Hz) | Increase for faster gestures, decrease for smoother signals |
| `params.ekf.R_acc` | Accelerometer measurement noise | Higher = trust gyro more (less accel correction) |
| `params.ekf.R_mag` | Magnetometer measurement noise | Higher = trust gyro more for heading |
| `params.segmentation.energy_high` | Gesture detection threshold | Lower = more sensitive (may detect non-gestures) |
| `params.segmentation.energy_low` | Gesture end threshold | Lower = longer detected windows |
| `params.segmentation.min_duration` | Minimum gesture length (s) | Filters out spurious detections |
| `params.ml.method` | ML algorithm: `'knn'`, `'svm'`, `'tree'`, `'ensemble'` | Switch classifier type |
| `params.ml.k` | kNN neighbours | Only affects kNN classifier |
| `params.viz.show_plots` | Enable/disable plotting | Set `false` for batch processing speed |

### Method Switching

The pipeline supports drop-in method switching via flags:

```matlab
params = config_params();

% Switch fusion algorithm
params.fusion.method = 'ekf';             % Default: full EKF
params.fusion.method = 'complementary';   % Simpler baseline

% Switch classifier
params.classifier.method = 'rules';       % Transparent rule-based
params.classifier.method = 'ml';          % Trained ML model

% Switch segmentation (future)
params.segmentation.method = 'energy';    % Default energy-based
```

### Do Not Change (Unless You Understand the Consequences)

| Parameter | Reason |
|-----------|--------|
| `params.frames.gravity_world` | Must match your coordinate convention; changing breaks EKF corrections |
| `params.ekf.Q_quat`, `params.ekf.Q_gyro_bias` | Process noise — changes EKF convergence rate and stability |
| `params.segmentation.gyro_norm_factor`, `params.segmentation.acc_norm_factor` | Normalisation constants in the energy formula; changing shifts all thresholds |
| `params.gestures.labels` | Must match folder names used in training data |

---

## Common Workflows

### "I want to add a new gesture"

1. Record 20+ samples using MATLAB Mobile
2. Create `data/NewGestureData/` and place `.mat` files there
3. Add `'new_gesture'` to `params.gestures.labels` in `config_params.m`
4. Add a new folder entry in `run_training_robust.m`'s `folderMap`
5. Optionally add rules in `classify_gesture_rules.m`
6. Re-run `run_training_robust`

### "I want better segmentation"

Edit only `src/gestures/segment_gesture.m`. The struct contract (`seg.winIdx`, `seg.windows`, `seg.score`) is what downstream code depends on — keep that interface and change the internals freely. Run `runtests('test_segmentation')` after editing.

### "I want to try a different fusion approach"

1. Create `src/fusion/my_new_filter.m` returning an `est` struct with at minimum `.q` (Nx4), `.euler` (Nx3), and `.b_g` (Nx3)
2. Add `params.fusion.method = 'my_new'` support in `main_gesture_demo.m`
3. Verify with `runtests('test_ekf_static')` adapted for your filter

### "I want to evaluate on new data"

```matlab
run_batch_eval('path/to/new/data/')
```

### "I want to compare classifiers"

`run_training_robust` already trains and compares kNN, SVM, Decision Tree, and Random Forest. Check the console output for the comparison table and per-class metrics.

---

## Development Notes

### Struct I/O Contracts

Every stage passes data through standardised structs. This is the architectural decision that makes modules independently replaceable:

| Struct | Key Fields | Producer |
|--------|-----------|----------|
| `data` | `.t` (Nx1), `.acc` (Nx3), `.gyr` (Nx3), `.mag` (Nx3), `.meta` | `read_phone_data` |
| `imu` | `.t`, `.dt` ((N-1)x1), `.Fs`, `.acc`, `.gyr`, `.mag`, `.flags.stationary` (Nx1 logical), `.calib` | `preprocess_imu` |
| `est` | `.q` (Nx4), `.b_g` (Nx3), `.euler` (Nx3), `.Ptrace` | `ekf_attitude_quat` |
| `motion` | `.v` (Nx3), `.p` (Nx3), `.a_world` (Nx3), `.zupt_flag` (Nx1 logical) | `kf_linear_motion` |
| `seg` | `.winIdx` [iStart iEnd], `.windows` (cell), `.score` | `segment_gesture` |
| `feat` | `.x` (1xM), `.names` (1xM cellstr), `.values` (struct), `.debug` | `extract_features` |
| `cls` | `.label` (string), `.score` (0-1), `.method` ("rules"\|"ml"), `.reason` | Classifiers |

**Critical detail:** `imu.dt` is an (N-1)×1 vector from `diff(imu.t)`, not a scalar. The EKF extends it internally via `dt_vec = [imu.dt; imu.dt(end)]`.

### Quaternion Convention

Hamilton convention: `q = [w, x, y, z]` with `w` as the scalar part. All `quat_utils` functions return 4×1 column vectors. `toEuler` is the exception — it returns three separate scalar outputs `[roll, pitch, yaw]`, not a vector.

### EKF Observability

The EKF corrects roll and pitch via gravity (always observable) but yaw only via the magnetometer (weakly observable in magnetically disturbed environments). This is not a bug — it's a fundamental property of the measurement model. In the test suite, yaw-axis bias estimation has a 0.02 rad/s tolerance vs 0.005 rad/s for roll/pitch.

### Data Collection Strategy

Recordings were collected with intentionally varied speeds and free phone orientation (not a fixed wrist mount) to create training data that is robust to real-world wrist-worn device behaviour. This is a deliberate engineering decision, not a limitation.

### Known Limitations

- **Position estimation drifts** even with ZUPT, because double integration of noisy accelerometer data without external position references (e.g. GPS) is fundamentally ill-conditioned. Position output is useful for qualitative gesture shape but not absolute localisation.
- **Magnetometer is unreliable** in environments with ferromagnetic interference (near machinery, steel shelving). The EKF includes magnitude-based rejection, but heading accuracy degrades.
- **Small dataset** (160 samples across 6 classes). Leave-one-out cross-validation is used instead of a held-out test set to maximise both training and evaluation data.
- **Remaining classification errors** (7/160) are concentrated in flip gesture confusions (flip_up vs flip_down). Other gesture classes are essentially solved.
- **`MLTRAININGEVALUATIONPIPELINE.m`** is a legacy monolithic script from early development. Use `run_training_robust` instead.

### Plotting Isolation

Algorithm functions (`ekf_attitude_quat`, `segment_gesture`, etc.) never create figures. All visualisation is isolated in `src/viz/`. This prevents accidentally breaking algorithms when changing figure layout.

---

## Testing

The test suite validates core mathematical operations, filter convergence, segmentation behaviour, and data import robustness.

### Quick Command (all 40 tests)

```matlab
results = [runtests('test_quat_math'), runtests('test_ekf_static'), ...
           runtests('test_segmentation'), runtests('test_import')];
fprintf('\n=== TOTALS: %d Passed, %d Failed ===\n', ...
        sum([results.Passed]), sum([results.Failed]));
```

### What Each Test File Covers

| File | Tests | What It Validates |
|------|-------|-------------------|
| `test_quat_math` | 16 | Identity, normalisation, multiplication, rotation, Euler round-trips, SLERP, axis-angle, batch rotation, conjugate, numerical stability |
| `test_ekf_static` | 8 | Quaternion norm preservation, gravity alignment from bad initial guess, gyro bias convergence (separate XY and Z tolerances), bounded drift, covariance boundedness, accelerometer noise rejection, innovation whiteness, tilted static recovery |
| `test_segmentation` | 8 | Single gesture detection, windowing accuracy, quiet signal rejection, multi-gesture separation, sub-threshold rejection, minimum duration enforcement, accelerometer disturbance handling, consistent re-runs |
| `test_import` | 8 | Standard MAT loading, time monotonicity, missing magnetometer handling, custom timestamps, sampling rate detection, alternative field naming, CSV format support, struct validation |

### Calculator Sanity Checks (Casio fx-991EX)

For quick manual verification without MATLAB:

- **Quaternion unit norm:** `w² + x² + y² + z²` should equal 1.0
- **RMS check:** `√(Σxᵢ² / N)` — verify against MATLAB's `rms()` output
- **Segmentation threshold:** Compare `|ω| / 3.0` against 1.5 to predict whether a gyro reading triggers detection
- **Energy formula:** `energy = gyr_mag/3.0 + 0.3 × (acc_dev/5.0)` — must exceed 1.5 to detect

---

## Future Improvements

These are designed-for extension points, not wishful thinking — the architecture already supports them:

- **Dedicated wrist-worn hardware** — Replace `read_phone_data` with a new I/O module for the target IMU (e.g. Bosch BNO055). The rest of the pipeline is hardware-agnostic.
- **Expanded gesture vocabulary** — Add new gesture folders, retrain. The feature extraction and ML pipeline scale naturally.
- **Real-time deployment** — `realtime_gesture_processor.m` already implements a streaming circular-buffer architecture. Convert to C/C++ via MATLAB Coder for embedded targets.
- **Adaptive thresholds** — Replace fixed segmentation thresholds with user-calibrated values (e.g. during a setup phase).
- **Deep learning** — Replace `extract_features` + `ml_predict_baseline` with a CNN/LSTM operating on raw windowed sensor data. The segmentation output provides training windows.
- **Multi-user normalisation** — Add a calibration step that normalises for different arm lengths and gesture speeds across users.

---

## References

Key technical sources underpinning the implementation:

1. Sasiadek, J.Z. & Hartana, P. (2000). Sensor fusion for navigation of an autonomous unmanned aerial vehicle. *IEEE International Conference on Robotics and Automation*.
2. Wang, M. et al. (2023). Multi-sensor integrated navigation positioning system. *Information Fusion*.
3. Zhuang, Y. et al. (2017). Noise-aware localisation algorithms for wireless sensor networks. *Computer Communications*.
4. Madgwick, S.O.H. (2010). An efficient orientation filter for inertial and inertial/magnetic sensor arrays. *Internal Report, University of Bristol*.

Full reference list: 29+ sources covering sensor fusion theory, Kalman filtering, industrial safety, and gesture recognition. See the project report for the complete bibliography.

---

## License

This project was developed as a Final Year Project for the BEng Electrical and Electronic Engineering programme at Newcastle University. Contact the author for usage permissions.
