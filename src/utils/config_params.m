function params = config_params()
%CONFIG_PARAMS Central configuration for gesture recognition pipeline

    % Global
    params.EPS = 1e-10;
    params.verbose = true;

    % Constants
    params.constants.g = 9.81;
    params.constants.deg2rad = pi/180;
    params.constants.rad2deg = 180/pi;

    % Sampling
    params.sampling.expected_Fs = 100;
    params.sampling.targetFs = 100;
    params.sampling.resample_target = 100;
    params.sampling.do_resample = true;

    % Preprocessing
    params.preprocess.acc_scale = 1;
    params.preprocess.gyr_scale = 1;
    params.preprocess.mag_scale = 1;
    params.preprocess.lpf_cutoff = 25;
    params.preprocess.lpf_order = 2;
    params.preprocess.use_lpf = true;
    params.preprocess.static_threshold = 0.5;
    params.preprocess.static_window = 50;
    params.preprocess.estimate_gyro_bias = true;
    params.preprocess.mag_calibration = true;
    params.preprocess.mag_outlier_threshold = 3;

    % Coordinate frames
    params.frames.gravity_world = [0; 0; -9.81];
    params.frames.mag_ref_world = [1; 0; 0];

    % EKF
    params.ekf.Q_quat = 1e-6;
    params.ekf.Q_gyro_bias = 1e-8;
    params.ekf.R_acc = 0.5;
    params.ekf.R_mag = 2.0;
    params.ekf.P0_quat = 0.1;
    params.ekf.P0_bias = 0.01;
    params.ekf.init_gyro_bias = [0; 0; 0];
    params.ekf.acc_magnitude_window = [8, 12];
    params.ekf.use_mag_update = true;
    params.ekf.mag_rejection_threshold = 5;
    params.ekf.enforce_symmetry = true;
    params.ekf.store_full_covariance = false;

    % Linear KF
    params.kf.Q_velocity = 0.1;
    params.kf.Q_position = 0.01;
    params.kf.R_zupt = 0.01;
    params.kf.P0_velocity = 1;
    params.kf.P0_position = 0.1;
    params.kf.zupt_gyro_threshold = 0.2;
    params.kf.zupt_acc_threshold = 1.0;
    params.kf.zupt_window = 10;

    % Complementary filter
    params.comp.alpha = 0.98;
    params.comp.beta = 0.1;

    % Fusion
    params.fusion.method = 'ekf';

    % Segmentation
    params.segmentation.method = 'energy';
    params.segmentation.energy_low = 0.5;
    params.segmentation.energy_high = 1.5;
    params.segmentation.gyro_norm_factor = 3.0;
    params.segmentation.acc_norm_factor = 5.0;
    params.segmentation.acc_weight = 0.3;
    params.segmentation.min_duration = 0.2;
    params.segmentation.max_duration = 3.0;
    params.segmentation.pre_buffer = 0.1;
    params.segmentation.post_buffer = 0.1;
    params.segmentation.max_gestures = 5;
    params.segmentation.min_gap = 0.3;

    % Features
    params.features.compute_fft = true;
    params.features.fft_nfft = 256;
    params.features.freq_bands = [0.5, 5; 5, 15; 15, 25];
    params.features.xcorr_max_lag_ratio = 0.25;
    params.features.list = {'duration', 'rms_gyr_x', 'rms_gyr_y', 'rms_gyr_z', ...
        'total_rotation', 'dominant_axis', 'energy_ratio_x', 'energy_ratio_y', ...
        'energy_ratio_z', 'phase_lag_xy', 'acc_range'};

    % Gesture labels
    params.gestures.labels = {'flip_up', 'flip_down', 'shake', 'twist', ...
        'push_forward', 'circle', 'unknown'};

    % Rule thresholds
    params.gestures.rules.twist_min_gyr_z_rms = 1.5;
    params.gestures.rules.twist_max_xy_ratio = 0.5;
    params.gestures.rules.flip_min_gyr_rms = 2.0;
    params.gestures.rules.flip_min_delta_pitch = 45;
    params.gestures.rules.shake_min_zc = 4;
    params.gestures.rules.shake_min_gyr_rms = 1.5;
    params.gestures.rules.push_min_acc_range = 5.0;
    params.gestures.rules.push_max_gyr_rms = 1.0;
    params.gestures.rules.circle_min_duration = 0.8;
    params.gestures.rules.circle_min_lag = 5;
    params.gestures.rules.circle_min_rotation = 180;
    params.gestures.rules.min_confidence = 0.3;
    params.gestures.rules.early_termination = true;
    params.gestures.rules.early_termination_threshold = 0.9;

    % ML settings
    params.ml.method = 'knn';
    params.ml.k = 5;
    params.ml.kernel = 'rbf';
    params.ml.standardize = true;
    params.ml.cross_validate = true;
    params.ml.cv_folds = 5;
    params.ml.minConfidence = 0.3;
    params.ml.model_path = 'models/gesture_model.mat';
    params.ml.feature_selection = false;
    params.ml.max_features = 20;

    % Classifier
    params.classifier.method = 'rules';

    % Visualization
    params.viz.enabled = true;
    params.viz.show_plots = true;
    params.viz.save_plots = false;

    % Paths
    params.paths.data_raw = 'data/';
    params.paths.models = 'models/';
    params.paths.outputs = 'outputs/';

    % Logging
    params.logging.enabled = true;
    params.logging.level = 'info';

end