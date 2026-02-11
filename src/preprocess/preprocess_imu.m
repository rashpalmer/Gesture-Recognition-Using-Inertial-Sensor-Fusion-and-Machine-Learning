function imu = preprocess_imu(data, params)
%PREPROCESS_IMU Preprocess raw IMU data for sensor fusion
%   imu = preprocess_imu(data, params) takes raw sensor data and produces
%   cleaned, calibrated IMU streams ready for fusion algorithms.
%
%   INPUTS:
%       data    - Raw data struct from read_phone_data()
%       params  - Configuration from config_params() (optional)
%
%   OUTPUT:
%       imu - Preprocessed IMU struct:
%           .t          - Nx1 time vector (seconds)
%           .dt         - (N-1)x1 time differences
%           .Fs         - Sample rate (Hz)
%           .acc        - Nx3 calibrated accelerometer (m/s²)
%           .gyr        - Nx3 calibrated gyroscope (rad/s)
%           .mag        - Nx3 calibrated magnetometer (µT)
%           .acc_raw    - Nx3 original accelerometer
%           .gyr_raw    - Nx3 original gyroscope
%           .mag_raw    - Nx3 original magnetometer
%           .flags.stationary - Nx1 logical (true = likely stationary)
%           .calib      - Calibration parameters used
%               .gyro_bias  - 3x1 estimated gyro bias
%               .acc_bias   - 3x1 estimated acc bias (optional)
%               .mag_offset - 3x1 hard-iron offset
%               .mag_scale  - 3x1 soft-iron scale
%
%   IMPROVEMENTS (v2.0):
%       - FIXED: rms() calls now include dimension argument
%       - ADDED: Input validation for data struct
%       - ADDED: Better handling of missing sensors
%       - OPTIMIZED: Vectorized operations where possible
%
%   PREPROCESSING STEPS:
%       1. Unit conversion (if needed)
%       2. Resampling to uniform rate
%       3. Low-pass filtering (optional)
%       4. Static segment detection
%       5. Gyroscope bias estimation
%       6. Magnetometer calibration (hard-iron)
%
%   EXAMPLE:
%       data = read_phone_data('sensor_log.mat');
%       params = config_params();
%       imu = preprocess_imu(data, params);
%
%   Author: Sensor Fusion Demo (Improved)
%   Date: 2026

    %% Default parameters
    if nargin < 2
        params = config_params();
    end

    fprintf('Preprocessing IMU data...\n');

    %% Input Validation
    if ~isstruct(data)
        error('preprocess_imu:InvalidInput', 'data must be a struct');
    end

    requiredFields = {'acc', 'gyr', 't'};
    for i = 1:length(requiredFields)
        if ~isfield(data, requiredFields{i})
            error('preprocess_imu:MissingField', 'data.%s is required', requiredFields{i});
        end
    end

    if isempty(data.acc) || isempty(data.gyr)
        error('preprocess_imu:EmptyData', 'Accelerometer and gyroscope data cannot be empty');
    end

    %% Initialize output structure
    imu = struct();
    imu.acc_raw = data.acc;
    imu.gyr_raw = data.gyr;

    % Handle optional magnetometer
    if isfield(data, 'mag') && ~isempty(data.mag)
        imu.mag_raw = data.mag;
    else
        imu.mag_raw = nan(size(data.acc));
    end

    %% Step 1: Unit Conversion
    acc = data.acc * params.preprocess.acc_scale;
    gyr = data.gyr * params.preprocess.gyr_scale;

    if isfield(data, 'mag') && ~isempty(data.mag)
        mag = data.mag * params.preprocess.mag_scale;
    else
        mag = nan(size(acc));
    end

    t = data.t(:);

    % Handle mismatched sensor lengths (common with MATLAB Mobile)
    n_acc = size(acc, 1);
    n_gyr = size(gyr, 1);
    n_mag = size(mag, 1);
    n_t = length(t);
    n_min = min([n_acc, n_gyr, n_mag, n_t]);
    
    if n_acc ~= n_t || n_gyr ~= n_t
        % Truncate all to shortest length
        t = t(1:n_min);
        acc = acc(1:n_min, :);
        gyr = gyr(1:n_min, :);
        mag = mag(1:n_min, :);
    end

    %% Step 2: Resampling to Uniform Rate
    if params.sampling.do_resample
        Fs_target = params.sampling.resample_target;

        % Check current sample rate
        dt_raw = diff(t);
        Fs_current = 1 / median(dt_raw);

        fprintf('  Original sample rate: %.1f Hz (std: %.3f s)\n', ...
            Fs_current, std(dt_raw));

        if abs(Fs_current - Fs_target) > 5 || std(dt_raw) > 0.005
            fprintf('  Resampling to %.1f Hz...\n', Fs_target);

            tu = time_utils();
            [t_new, acc] = tu.resampleUniform(t, acc, Fs_target);
            [~, gyr] = tu.resampleUniform(t, gyr, Fs_target);

            if ~all(isnan(mag(:)))
                [~, mag] = tu.resampleUniform(t, mag, Fs_target);
            else
                mag = nan(length(t_new), 3);
            end

            t = t_new;
        end
    end

    n = length(t);
    imu.t = t;
    imu.dt = diff(t);
    imu.Fs = 1 / mean(imu.dt);

    fprintf('  Working sample rate: %.1f Hz\n', imu.Fs);

    %% Step 3: Low-Pass Filtering
    if params.preprocess.use_lpf
        fc = params.preprocess.lpf_cutoff;
        order = params.preprocess.lpf_order;

        % Design Butterworth filter
        [b, a] = butter(order, fc / (imu.Fs/2), 'low');

        % Apply zero-phase filtering
        acc = filtfilt(b, a, acc);
        gyr = filtfilt(b, a, gyr);

        if ~all(isnan(mag(:)))
            mag = filtfilt(b, a, mag);
        end

        fprintf('  Applied %d-order LPF at %.1f Hz\n', order, fc);
    end

    %% Step 4: Static Segment Detection
    tu = time_utils();
    [imu.flags.stationary, static_windows] = tu.findStaticSegments(gyr, params);

    n_static = sum(imu.flags.stationary);
    fprintf('  Detected %d static samples (%.1f%%)\n', ...
        n_static, 100*n_static/n);

    %% Step 5: Gyroscope Bias Estimation
    if params.preprocess.estimate_gyro_bias && n_static > 10
        % Estimate bias from static segments
        gyro_static = gyr(imu.flags.stationary, :);
        gyro_bias = mean(gyro_static, 1)';

        % Subtract bias
        gyr = gyr - gyro_bias';

        imu.calib.gyro_bias = gyro_bias;
        fprintf('  Gyro bias: [%.4f, %.4f, %.4f] rad/s\n', gyro_bias);
    else
        imu.calib.gyro_bias = [0; 0; 0];
    end

    %% Step 6: Accelerometer Check (gravity reference)
    % During static periods, |acc| should be ~9.81 m/s²
    if n_static > 10
        acc_static = acc(imu.flags.stationary, :);
        acc_mag_mean = mean(sqrt(sum(acc_static.^2, 2)));

        fprintf('  Static acc magnitude: %.3f m/s² (expected: 9.81)\n', acc_mag_mean);

        % Optionally scale to match gravity
        if abs(acc_mag_mean - params.constants.g) > 0.5
            scale_factor = params.constants.g / acc_mag_mean;
            acc = acc * scale_factor;
            fprintf('  Applied acc scale correction: %.4f\n', scale_factor);
        end
    end

    imu.calib.acc_bias = [0; 0; 0];  % Could estimate if needed

    %% Step 7: Magnetometer Calibration
    if params.preprocess.mag_calibration && ~all(isnan(mag(:)))
        [mag_cal, mag_offset, mag_scale] = calibrate_mag_simple(mag, params);
        mag = mag_cal;
        imu.calib.mag_offset = mag_offset;
        imu.calib.mag_scale = mag_scale;
        fprintf('  Mag offset: [%.2f, %.2f, %.2f] µT\n', mag_offset);
    else
        imu.calib.mag_offset = [0; 0; 0];
        imu.calib.mag_scale = [1; 1; 1];
    end

    %% Step 8: Outlier Rejection for Magnetometer
    if params.preprocess.mag_calibration && ~all(isnan(mag(:)))
        mag_mag = sqrt(sum(mag.^2, 2));
        mag_median = median(mag_mag);
        mag_mad = median(abs(mag_mag - mag_median));

        threshold = params.preprocess.mag_outlier_threshold;
        outliers = abs(mag_mag - mag_median) > threshold * 1.4826 * mag_mad;

        if sum(outliers) > 0
            fprintf('  Rejected %d magnetometer outliers (%.1f%%)\n', ...
                sum(outliers), 100*sum(outliers)/n);
            % Mark outliers but don't remove (let EKF handle with high R)
            imu.flags.mag_outlier = outliers;
        else
            imu.flags.mag_outlier = false(n, 1);
        end
    else
        imu.flags.mag_outlier = false(n, 1);
    end

    %% Store processed data
    imu.acc = acc;
    imu.gyr = gyr;
    imu.mag = mag;

    %% Summary statistics
    fprintf('Preprocessing complete:\n');
    fprintf('  Duration: %.2f seconds\n', t(end) - t(1));
    fprintf('  Samples:  %d\n', n);

    % FIXED: rms() with dimension argument for compatibility
    fprintf('  Acc RMS:  [%.2f, %.2f, %.2f] m/s²\n', ...
        rms(acc(:,1), 1), rms(acc(:,2), 1), rms(acc(:,3), 1));
    fprintf('  Gyr RMS:  [%.3f, %.3f, %.3f] rad/s\n', ...
        rms(gyr(:,1), 1), rms(gyr(:,2), 1), rms(gyr(:,3), 1));

end

function [mag_cal, offset, scale] = calibrate_mag_simple(mag, params)
%CALIBRATE_MAG_SIMPLE Simple hard-iron magnetometer calibration
%   Estimates center offset by finding sphere center
%
%   This is a simplified approach suitable for quick calibration.
%   For better results, use ellipsoid fitting (soft-iron correction).

    % Remove NaN rows
    valid_idx = ~any(isnan(mag), 2);
    mag_valid = mag(valid_idx, :);

    if size(mag_valid, 1) < 100
        warning('Not enough magnetometer data for calibration');
        mag_cal = mag;
        offset = [0; 0; 0];
        scale = [1; 1; 1];
        return;
    end

    % Simple approach: center = mean of extremes
    offset = mean([min(mag_valid); max(mag_valid)])';

    % Alternatively, use least-squares sphere fitting
    % [offset, radius] = fitSphere(mag_valid);

    % Apply offset
    mag_cal = mag - offset';

    % Scale factors (simplified - assume sphere not ellipsoid)
    scale = [1; 1; 1];

    % Normalize to unit sphere for consistent measurement model
    % mag_cal = mag_cal / median(sqrt(sum(mag_cal.^2, 2)));

end
