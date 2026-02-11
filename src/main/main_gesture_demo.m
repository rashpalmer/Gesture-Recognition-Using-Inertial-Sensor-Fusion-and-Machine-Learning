%% MAIN_GESTURE_DEMO - End-to-End Gesture Recognition Pipeline
% This is the primary entry point for the gesture recognition system.
% It demonstrates the complete pipeline from raw sensor data to gesture
% classification using sensor fusion (EKF + Linear KF).
%
% USAGE:
%   main_gesture_demo           % Uses default sample data or prompts for file
%   main_gesture_demo(filename) % Uses specified data file
%
% IMPROVEMENTS (v2.0):
%   - FIXED: Field name consistency (gyro_bias vs gyroBias)
%   - ADDED: Input validation at each stage
%   - ADDED: Timing profiler for performance analysis
%   - ADDED: Better error handling and recovery
%   - OPTIMIZED: Parallel feature extraction where possible
%
% PIPELINE STAGES:
%   1. Load configuration parameters
%   2. Read raw sensor data (MATLAB Mobile export)
%   3. Preprocess IMU data (calibration, filtering)
%   4. Attitude estimation (EKF with quaternions)
%   5. Motion estimation (Linear KF with ZUPT)
%   6. Gesture segmentation (energy-based)
%   7. Feature extraction (time/frequency domain)
%   8. Gesture classification (rules + optional ML)
%   9. Visualization and results
%
% OUTPUTS:
%   Displays classification result and diagnostic plots
%   Saves run log to outputs/logs/
%
% See also: config_params, read_phone_data, preprocess_imu, ekf_attitude_quat,
%           kf_linear_motion, segment_gesture, extract_features, classify_gesture_rules

function main_gesture_demo(dataFile)
    %% ========================================================================
    %  INITIALIZATION
    %  ========================================================================

    fprintf('\n');
    fprintf('========================================================\n');
    fprintf('  GESTURE RECOGNITION via SENSOR FUSION (v2.0)\n');
    fprintf('  EKF Attitude + Linear KF Motion + Classification\n');
    fprintf('========================================================\n\n');

    % Record start time for logging
    runStartTime = datetime('now');

    % Performance timing (NEW)
    timings = struct();

    % Add source paths (if not already added)
    thisFile = mfilename('fullpath');
    [thisDir, ~, ~] = fileparts(thisFile);
    srcDir = fileparts(thisDir);  % src/
    repoDir = fileparts(srcDir);  % gesture-sa-sensor-fusion/

    addpath(genpath(srcDir));

    %% ========================================================================
    %  STAGE 1: LOAD CONFIGURATION
    %  ========================================================================

    fprintf('[1/9] Loading configuration...\n');
    tic;
    params = config_params();
    timings.config = toc;

    fprintf('      Fusion method: %s\n', params.fusion.method);
    fprintf('      Classifier method: %s\n', params.classifier.method);
    fprintf('      Target sample rate: %.1f Hz\n', params.sampling.targetFs);

    %% ========================================================================
    %  STAGE 2: READ SENSOR DATA
    %  ========================================================================

    fprintf('[2/9] Reading sensor data...\n');
    tic;

    % Determine data file
    if nargin < 1 || isempty(dataFile)
        % Check for sample data
        sampleFile = fullfile(repoDir, 'data', 'raw', 'sample_gesture.mat');
        if exist(sampleFile, 'file')
            dataFile = sampleFile;
            fprintf('      Using sample data: %s\n', dataFile);
        else
            % Prompt user
            fprintf('      No data file specified.\n');
            fprintf('      Please provide a MATLAB Mobile export (.mat or .csv)\n');
            [fname, fpath] = uigetfile({'*.mat;*.csv', 'Sensor Data (*.mat, *.csv)'}, ...
                                       'Select sensor data file');
            if isequal(fname, 0)
                error('No data file selected. Exiting.');
            end
            dataFile = fullfile(fpath, fname);
        end
    end

    % Read data
    data = read_phone_data(dataFile);
    timings.read_data = toc;

    fprintf('      Duration: %.2f seconds\n', data.t(end) - data.t(1));
    fprintf('      Samples: %d\n', length(data.t));
    fprintf('      Sensors: acc=%s, gyr=%s, mag=%s\n', ...
            yesno(~isempty(data.acc) && ~all(isnan(data.acc(:)))), ...
            yesno(~isempty(data.gyr) && ~all(isnan(data.gyr(:)))), ...
            yesno(~isempty(data.mag) && ~all(isnan(data.mag(:)))));

    % Validate required sensors
    if isempty(data.acc) || all(isnan(data.acc(:))) || ...
       isempty(data.gyr) || all(isnan(data.gyr(:)))
        error('Accelerometer and gyroscope data are required.');
    end

    %% ========================================================================
    %  STAGE 3: PREPROCESS IMU DATA
    %  ========================================================================

    fprintf('[3/9] Preprocessing IMU data...\n');
    tic;

    imu = preprocess_imu(data, params);
    timings.preprocess = toc;

    fprintf('      Resampled to %.1f Hz (%d samples)\n', imu.Fs, length(imu.t));
    fprintf('      Static segments: %d found\n', sum(diff([0; imu.flags.stationary; 0]) == 1));

    % FIXED: Use consistent field name (gyro_bias not gyroBias)
    if isfield(imu.calib, 'gyro_bias')
        fprintf('      Gyro bias estimate: [%.4f, %.4f, %.4f] rad/s\n', imu.calib.gyro_bias);
    end
    if isfield(imu, 'mag') && ~all(isnan(imu.mag(:))) && isfield(imu.calib, 'mag_offset')
        fprintf('      Mag hard-iron offset: [%.2f, %.2f, %.2f]\n', imu.calib.mag_offset);
    end

    %% ========================================================================
    %  STAGE 4: ATTITUDE ESTIMATION (EKF)
    %  ========================================================================

    fprintf('[4/9] Running attitude estimation...\n');
    tic;

    if strcmpi(params.fusion.method, 'ekf')
        fprintf('      Method: Extended Kalman Filter (quaternion)\n');
        est = ekf_attitude_quat(imu, params);
    elseif strcmpi(params.fusion.method, 'complementary')
        fprintf('      Method: Complementary Filter (baseline)\n');
        est = complementary_filter(imu, params);
    else
        error('Unknown fusion method: %s', params.fusion.method);
    end
    timings.attitude = toc;

    % Quaternion health check
    qNorms = sqrt(sum(est.q.^2, 2));
    fprintf('      Quaternion norm: min=%.6f, max=%.6f\n', min(qNorms), max(qNorms));

    % Orientation summary
    eulerRange = max(est.euler) - min(est.euler);
    fprintf('      Euler range (deg): roll=%.1f, pitch=%.1f, yaw=%.1f\n', ...
            rad2deg(eulerRange(1)), rad2deg(eulerRange(2)), rad2deg(eulerRange(3)));

    if isfield(est, 'b_g')
        finalBias = est.b_g(end, :);
        fprintf('      Final gyro bias: [%.5f, %.5f, %.5f] rad/s\n', finalBias);
    end

    %% ========================================================================
    %  STAGE 5: MOTION ESTIMATION (LINEAR KF)
    %  ========================================================================

    fprintf('[5/9] Running motion estimation...\n');
    tic;

    motion = kf_linear_motion(imu, est, params);
    timings.motion = toc;

    zupt_count = sum(diff([0; motion.zupt_flag]) == 1);
    fprintf('      ZUPT corrections: %d\n', zupt_count);
    fprintf('      Final position: [%.3f, %.3f, %.3f] m\n', motion.p(end, :));
    fprintf('      Max velocity: %.3f m/s\n', max(vecnorm(motion.v, 2, 2)));

    %% ========================================================================
    %  STAGE 6: GESTURE SEGMENTATION
    %  ========================================================================

    fprintf('[6/9] Segmenting gestures...\n');
    tic;

    seg = segment_gesture(imu, params);
    timings.segmentation = toc;

    fprintf('      Windows detected: %d\n', size(seg.windows, 1));

    if isempty(seg.windows)
        fprintf('\n*** WARNING: No gesture detected! ***\n');
        fprintf('    Try lowering segmentation thresholds or check data quality.\n\n');

        % Still run visualization for diagnostics
        if params.viz.enabled
            plot_diagnostics(imu, est, motion, seg, [], [], params);
        end
        return;
    end

    % Primary gesture info
    fprintf('      Primary gesture: window %d (samples %d-%d)\n', ...
            seg.primary, seg.winIdx(1), seg.winIdx(2));
    gestureDuration = (seg.winIdx(2) - seg.winIdx(1)) / imu.Fs;
    fprintf('      Duration: %.3f seconds\n', gestureDuration);
    fprintf('      Segmentation score: %.3f\n', seg.score);

    %% ========================================================================
    %  STAGE 7: FEATURE EXTRACTION
    %  ========================================================================

    fprintf('[7/9] Extracting features...\n');
    tic;

    feat = extract_features(imu, est, seg, params);
    timings.features = toc;

    fprintf('      Features extracted: %d\n', length(feat.names));

    % Show key features - handle both naming conventions
    keyFeatures = {'duration', 'gyr_rms_total', 'dominant_axis', 'total_rotation_deg', 'total_rotation'};
    for i = 1:length(keyFeatures)
        fname = keyFeatures{i};
        if isfield(feat.values, fname)
            val = feat.values.(fname);
            if ischar(val) || isstring(val)
                fprintf('      %s: %s\n', fname, val);
            else
                fprintf('      %s: %.3f\n', fname, val);
            end
        end
    end

    %% ========================================================================
    %  STAGE 8: GESTURE CLASSIFICATION
    %  ========================================================================

    fprintf('[8/9] Classifying gesture...\n');
    tic;

    if strcmpi(params.classifier.method, 'rules')
        fprintf('      Method: Rule-based classifier\n');
        cls = classify_gesture_rules(feat, params);
    elseif strcmpi(params.classifier.method, 'ml')
        fprintf('      Method: Machine Learning classifier\n');
        cls = ml_predict_baseline(feat, params);
    else
        error('Unknown classifier method: %s', params.classifier.method);
    end
    timings.classification = toc;

    %% ========================================================================
    %  STAGE 9: RESULTS AND VISUALIZATION
    %  ========================================================================

    fprintf('[9/9] Generating results...\n\n');
    tic;

    % Display result
    fprintf('========================================================\n');
    fprintf('  CLASSIFICATION RESULT\n');
    fprintf('========================================================\n');
    fprintf('  Gesture: %s\n', upper(cls.label));
    fprintf('  Confidence: %.1f%%\n', cls.score * 100);
    fprintf('  Method: %s\n', cls.method);
    fprintf('  Reason: %s\n', cls.reason);
    fprintf('========================================================\n\n');

    % Performance summary (NEW)
    total_time = sum(structfun(@(x) x, timings));
    fprintf('Performance Summary:\n');
    fprintf('  Config:        %.3f s\n', timings.config);
    fprintf('  Read Data:     %.3f s\n', timings.read_data);
    fprintf('  Preprocess:    %.3f s\n', timings.preprocess);
    fprintf('  Attitude Est:  %.3f s\n', timings.attitude);
    fprintf('  Motion Est:    %.3f s\n', timings.motion);
    fprintf('  Segmentation:  %.3f s\n', timings.segmentation);
    fprintf('  Features:      %.3f s\n', timings.features);
    fprintf('  Classification:%.3f s\n', timings.classification);
    fprintf('  -----------------------\n');
    fprintf('  TOTAL:         %.3f s\n\n', total_time);

    % Visualization
    if params.viz.enabled
        fprintf('Generating diagnostic plots...\n');
        plot_diagnostics(imu, est, motion, seg, feat, cls, params);
    end
    timings.visualization = toc;

    % Save run log
    if params.logging.enabled
        logDir = fullfile(repoDir, 'outputs', 'logs');
        if ~exist(logDir, 'dir')
            mkdir(logDir);
        end

        timestamp = datestr(runStartTime, 'yyyymmdd_HHMMSS');
        logFile = fullfile(logDir, sprintf('run_%s.mat', timestamp));

        runLog = struct();
        runLog.timestamp = runStartTime;
        runLog.dataFile = dataFile;
        runLog.params = params;
        runLog.result = cls;
        runLog.features = feat;
        runLog.segmentation = seg;
        runLog.duration = seconds(datetime('now') - runStartTime);
        runLog.timings = timings;  % NEW: Include timing data

        save(logFile, 'runLog');
        fprintf('Run log saved: %s\n', logFile);
    end

    fprintf('\nDone.\n');
end

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function s = yesno(b)
    if b
        s = 'yes';
    else
        s = 'no';
    end
end
