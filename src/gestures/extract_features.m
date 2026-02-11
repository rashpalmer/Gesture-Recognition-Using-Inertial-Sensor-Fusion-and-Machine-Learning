function feat = extract_features(imu, est, seg, params)
%EXTRACT_FEATURES Extract features from a segmented gesture
%   feat = extract_features(imu, est, seg, params) computes a feature vector
%   from the primary gesture segment.
%
%   INPUTS:
%       imu     - Preprocessed IMU data
%       est     - Attitude estimation from ekf_attitude_quat() (can be empty)
%       seg     - Segmentation results from segment_gesture()
%       params  - Configuration from config_params() (optional)
%
%   OUTPUT:
%       feat - Feature struct:
%           .x          - 1xM feature vector (numeric)
%           .names      - 1xM cell array of feature names
%           .values     - Struct with named feature values
%           .debug      - Additional debug information
%
%   VERSION 2.2 CHANGES:
%       - ADDED: Twist-specific features for flip_down vs twist discrimination
%           * gyr_x_dominance: X rotation / total rotation (high for twist)
%           * gyr_y_dominance: Y rotation / total rotation (high for flips)
%           * roll_pitch_ratio: |X| / |Y| integral ratio (twist >> 1, flip << 1)
%           * gyr_xy_ratio: signed X/Y ratio
%           * axis_dominance_diff: |X_dom - Y_dom| (separation metric)
%       - ADDED: gyr_z_integral, gyr_x_mean_signed for completeness
%       - RETAINED: All pitch features from v2.1 for flip_up/flip_down
%       - REMOVED: Redundant backward-compat aliases to reduce feature bloat
%
%   Author: Sensor Fusion Project
%   Date: 2026

    %% Input validation
    if nargin < 4 || isempty(params)
        params = config_params();
    end
    if nargin < 2 || isempty(est)
        est = [];
    end

    validateattributes(imu, {'struct'}, {'nonempty'}, mfilename, 'imu', 1);
    validateattributes(seg, {'struct'}, {'nonempty'}, mfilename, 'seg', 3);

    assert(isfield(imu, 'acc'), 'IMU struct must contain acc field');
    assert(isfield(imu, 'gyr'), 'IMU struct must contain gyr field');
    assert(isfield(imu, 't'), 'IMU struct must contain t field');

    EPS = params.EPS;

    %% Get gesture window
    if seg.n_gestures > 0
        idx_start = seg.winIdx(1);
        idx_end = seg.winIdx(2);
    else
        idx_start = 1;
        idx_end = length(imu.t);
    end

    idx_start = max(1, min(idx_start, length(imu.t)));
    idx_end = max(idx_start, min(idx_end, length(imu.t)));

    t_win = imu.t(idx_start:idx_end);
    acc_win = imu.acc(idx_start:idx_end, :);
    gyr_win = imu.gyr(idx_start:idx_end, :);

    n_samples = length(t_win);
    Fs = imu.Fs;
    dt = 1 / Fs;

    %% Initialize outputs
    feat.values = struct();
    feat.names = {};
    feat.x = [];

    %% ==================== BASIC FEATURES ====================
    
    %% Duration
    feat.values.duration = t_win(end) - t_win(1);
    add_feature('duration', feat.values.duration);

    %% RMS Acceleration (per axis)
    feat.values.rms_acc_x = rms(acc_win(:,1), 1);
    feat.values.rms_acc_y = rms(acc_win(:,2), 1);
    feat.values.rms_acc_z = rms(acc_win(:,3), 1);
    feat.values.rms_acc_total = rms(sqrt(sum(acc_win.^2, 2)), 1);

    add_feature('rms_acc_x', feat.values.rms_acc_x);
    add_feature('rms_acc_y', feat.values.rms_acc_y);
    add_feature('rms_acc_z', feat.values.rms_acc_z);
    add_feature('rms_acc_total', feat.values.rms_acc_total);

    %% RMS Gyroscope (per axis)
    feat.values.rms_gyr_x = rms(gyr_win(:,1), 1);
    feat.values.rms_gyr_y = rms(gyr_win(:,2), 1);
    feat.values.rms_gyr_z = rms(gyr_win(:,3), 1);
    feat.values.rms_gyr_total = rms(sqrt(sum(gyr_win.^2, 2)), 1);

    add_feature('rms_gyr_x', feat.values.rms_gyr_x);
    add_feature('rms_gyr_y', feat.values.rms_gyr_y);
    add_feature('rms_gyr_z', feat.values.rms_gyr_z);
    add_feature('rms_gyr_total', feat.values.rms_gyr_total);

    %% Peak Gyroscope (per axis, signed)
    [~, peak_idx_x] = max(abs(gyr_win(:,1)));
    [~, peak_idx_y] = max(abs(gyr_win(:,2)));
    [~, peak_idx_z] = max(abs(gyr_win(:,3)));

    feat.values.peak_gyr_x = gyr_win(peak_idx_x, 1);
    feat.values.peak_gyr_y = gyr_win(peak_idx_y, 2);
    feat.values.peak_gyr_z = gyr_win(peak_idx_z, 3);

    add_feature('peak_gyr_x', feat.values.peak_gyr_x);
    add_feature('peak_gyr_y', feat.values.peak_gyr_y);
    add_feature('peak_gyr_z', feat.values.peak_gyr_z);

    %% Peak Gyroscope magnitudes (absolute)
    feat.values.peak_gyr_x_abs = abs(feat.values.peak_gyr_x);
    feat.values.peak_gyr_y_abs = abs(feat.values.peak_gyr_y);
    feat.values.peak_gyr_z_abs = abs(feat.values.peak_gyr_z);
    feat.values.gyr_peak_abs = max([feat.values.peak_gyr_x_abs, ...
                                    feat.values.peak_gyr_y_abs, ...
                                    feat.values.peak_gyr_z_abs]);

    add_feature('peak_gyr_x_abs', feat.values.peak_gyr_x_abs);
    add_feature('peak_gyr_y_abs', feat.values.peak_gyr_y_abs);
    add_feature('peak_gyr_z_abs', feat.values.peak_gyr_z_abs);
    add_feature('gyr_peak_abs', feat.values.gyr_peak_abs);

    %% Dominant rotation axis
    peak_abs = [feat.values.peak_gyr_x_abs, feat.values.peak_gyr_y_abs, feat.values.peak_gyr_z_abs];
    [~, dominant_axis] = max(peak_abs);
    feat.values.dominant_axis = dominant_axis;
    add_feature('dominant_axis', dominant_axis);

    %% Total rotation angle
    gyr_mag = sqrt(sum(gyr_win.^2, 2));
    feat.values.total_rotation = sum(gyr_mag) * dt;
    feat.values.total_rotation_deg = feat.values.total_rotation * params.constants.rad2deg;

    add_feature('total_rotation', feat.values.total_rotation);
    add_feature('total_rotation_deg', feat.values.total_rotation_deg);

    %% Zero-crossing counts
    zc_x = count_zero_crossings(gyr_win(:,1));
    zc_y = count_zero_crossings(gyr_win(:,2));
    zc_z = count_zero_crossings(gyr_win(:,3));

    feat.values.zero_cross_gyr_x = zc_x;
    feat.values.zero_cross_gyr_y = zc_y;
    feat.values.zero_cross_gyr_z = zc_z;
    feat.values.zero_cross_gyr_total = zc_x + zc_y + zc_z;

    add_feature('zero_cross_gyr_x', zc_x);
    add_feature('zero_cross_gyr_y', zc_y);
    add_feature('zero_cross_gyr_z', zc_z);
    add_feature('zero_cross_gyr_total', zc_x + zc_y + zc_z);

    %% Energy distribution ratios
    energy_x = sum(gyr_win(:,1).^2);
    energy_y = sum(gyr_win(:,2).^2);
    energy_z = sum(gyr_win(:,3).^2);
    energy_total = energy_x + energy_y + energy_z + EPS;

    feat.values.energy_ratio_x = energy_x / energy_total;
    feat.values.energy_ratio_y = energy_y / energy_total;
    feat.values.energy_ratio_z = energy_z / energy_total;

    add_feature('energy_ratio_x', feat.values.energy_ratio_x);
    add_feature('energy_ratio_y', feat.values.energy_ratio_y);
    add_feature('energy_ratio_z', feat.values.energy_ratio_z);

    %% Acceleration range
    acc_mag = sqrt(sum(acc_win.^2, 2));
    feat.values.acc_range = max(acc_mag) - min(acc_mag);
    feat.values.acc_range_x = max(acc_win(:,1)) - min(acc_win(:,1));
    feat.values.acc_range_y = max(acc_win(:,2)) - min(acc_win(:,2));
    feat.values.acc_range_z = max(acc_win(:,3)) - min(acc_win(:,3));

    add_feature('acc_range', feat.values.acc_range);
    add_feature('acc_range_x', feat.values.acc_range_x);
    add_feature('acc_range_y', feat.values.acc_range_y);
    add_feature('acc_range_z', feat.values.acc_range_z);

    %% Mean and variance
    feat.values.mean_gyr_x = mean(gyr_win(:,1));
    feat.values.mean_gyr_y = mean(gyr_win(:,2));
    feat.values.mean_gyr_z = mean(gyr_win(:,3));
    feat.values.var_gyr_total = var(gyr_mag);

    add_feature('mean_gyr_x', feat.values.mean_gyr_x);
    add_feature('mean_gyr_y', feat.values.mean_gyr_y);
    add_feature('mean_gyr_z', feat.values.mean_gyr_z);
    add_feature('var_gyr_total', feat.values.var_gyr_total);

    %% Skewness and Kurtosis
    if n_samples > 3
        feat.values.skew_gyr_x = skewness(gyr_win(:,1));
        feat.values.skew_gyr_y = skewness(gyr_win(:,2));
        feat.values.skew_gyr_z = skewness(gyr_win(:,3));
        feat.values.kurt_gyr_total = kurtosis(gyr_mag);

        add_feature('skew_gyr_x', feat.values.skew_gyr_x);
        add_feature('skew_gyr_y', feat.values.skew_gyr_y);
        add_feature('skew_gyr_z', feat.values.skew_gyr_z);
        add_feature('kurt_gyr_total', feat.values.kurt_gyr_total);
    end

    %% Jerk features
    if n_samples > 2
        jerk = diff(acc_win) * Fs;
        feat.values.jerk_rms_total = rms(sqrt(sum(jerk.^2, 2)), 1);
        feat.values.jerk_max = max(sqrt(sum(jerk.^2, 2)));

        add_feature('jerk_rms_total', feat.values.jerk_rms_total);
        add_feature('jerk_max', feat.values.jerk_max);
    end

    %% Phase features (circular motion)
    if n_samples > 10
        gx_std = std(gyr_win(:,1));
        gy_std = std(gyr_win(:,2));

        if gx_std > EPS && gy_std > EPS
            gx_norm = (gyr_win(:,1) - mean(gyr_win(:,1))) / gx_std;
            gy_norm = (gyr_win(:,2) - mean(gyr_win(:,2))) / gy_std;

            max_lag_ratio = 0.25;
            if isfield(params, 'features') && isfield(params.features, 'xcorr_max_lag_ratio')
                max_lag_ratio = params.features.xcorr_max_lag_ratio;
            end
            max_lag = round(n_samples * max_lag_ratio);

            [xcorr_val, lags] = xcorr(gx_norm, gy_norm, max_lag);
            [~, max_idx] = max(abs(xcorr_val));
            phase_lag = lags(max_idx);

            feat.values.phase_lag_xy = phase_lag / Fs;
            feat.values.xcorr_lag_xy = phase_lag;
        else
            feat.values.phase_lag_xy = 0;
            feat.values.xcorr_lag_xy = 0;
        end

        add_feature('phase_lag_xy', feat.values.phase_lag_xy);
        add_feature('xcorr_lag_xy', feat.values.xcorr_lag_xy);
    else
        feat.values.phase_lag_xy = 0;
        feat.values.xcorr_lag_xy = 0;
        add_feature('phase_lag_xy', 0);
        add_feature('xcorr_lag_xy', 0);
    end

    %% Frequency features
    if params.features.compute_fft && n_samples >= 16
        try
            nfft = min(params.features.fft_nfft, 2^nextpow2(n_samples));
            Y = fft(gyr_mag - mean(gyr_mag), nfft);
            P = abs(Y(1:nfft/2+1)).^2;
            f = Fs * (0:(nfft/2)) / nfft;

            [~, dom_idx] = max(P(2:end));
            feat.values.dominant_freq = f(dom_idx + 1);
            add_feature('dominant_freq', feat.values.dominant_freq);

            bands = params.features.freq_bands;
            total_power = sum(P) + EPS;

            for b = 1:size(bands, 1)
                f_low = bands(b, 1);
                f_high = bands(b, 2);
                band_idx = f >= f_low & f <= f_high;
                band_energy = sum(P(band_idx)) / total_power;

                feat_name = sprintf('freq_band_%d_%d', round(f_low), round(f_high));
                feat.values.(matlab.lang.makeValidName(feat_name)) = band_energy;
                add_feature(feat_name, band_energy);
            end
        catch
            % FFT failed, skip
        end
    end

    %% ==================== GYRO INTEGRAL FEATURES (v2.1+) ====================
    % Critical for flip/twist discrimination - works even without EKF
    
    % Y-axis integral (PITCH) - positive = flip_up, negative = flip_down
    gyr_y_integral = sum(gyr_win(:, 2)) * dt;
    feat.values.gyr_y_integral = gyr_y_integral;
    feat.values.gyr_y_integral_deg = gyr_y_integral * params.constants.rad2deg;
    add_feature('gyr_y_integral', gyr_y_integral);
    add_feature('gyr_y_integral_deg', feat.values.gyr_y_integral_deg);
    
    % X-axis integral (ROLL) - important for twist
    gyr_x_integral = sum(gyr_win(:, 1)) * dt;
    feat.values.gyr_x_integral = gyr_x_integral;
    feat.values.gyr_x_integral_deg = gyr_x_integral * params.constants.rad2deg;
    add_feature('gyr_x_integral', gyr_x_integral);
    add_feature('gyr_x_integral_deg', feat.values.gyr_x_integral_deg);
    
    % Z-axis integral (YAW) - for completeness
    gyr_z_integral = sum(gyr_win(:, 3)) * dt;
    feat.values.gyr_z_integral = gyr_z_integral;
    feat.values.gyr_z_integral_deg = gyr_z_integral * params.constants.rad2deg;
    add_feature('gyr_z_integral', gyr_z_integral);
    add_feature('gyr_z_integral_deg', feat.values.gyr_z_integral_deg);
    
    % Signed mean gyro (quick direction indicators)
    feat.values.gyr_y_mean_signed = mean(gyr_win(:, 2));
    feat.values.gyr_x_mean_signed = mean(gyr_win(:, 1));
    add_feature('gyr_y_mean_signed', feat.values.gyr_y_mean_signed);
    add_feature('gyr_x_mean_signed', feat.values.gyr_x_mean_signed);

%% ==================== TEMPORAL SHAPE FEATURES (v2.3 NEW) ====================

% Peak timing: WHERE in the gesture the max rotation occurs
% flip_up tends to peak early, flip_down may peak later (or vice versa)
[~, peak_time_idx] = max(abs(gyr_win(:,2)));  % Y-axis peak
feat.values.peak_time_ratio_y = peak_time_idx / n_samples;  % 0=start, 1=end
add_feature('peak_time_ratio_y', feat.values.peak_time_ratio_y);

[~, peak_time_idx_x] = max(abs(gyr_win(:,1)));
feat.values.peak_time_ratio_x = peak_time_idx_x / n_samples;
add_feature('peak_time_ratio_x', feat.values.peak_time_ratio_x);

% First-half vs second-half energy split
half = floor(n_samples / 2);
energy_first_y = sum(gyr_win(1:half, 2).^2);
energy_second_y = sum(gyr_win(half+1:end, 2).^2);
feat.values.energy_asymmetry_y = (energy_first_y - energy_second_y) / ...
    (energy_first_y + energy_second_y + EPS);
add_feature('energy_asymmetry_y', feat.values.energy_asymmetry_y);

% Positive vs negative area under gyr_y curve
pos_area_y = sum(max(gyr_win(:,2), 0)) * dt;
neg_area_y = sum(min(gyr_win(:,2), 0)) * dt;
feat.values.gyr_y_pos_area = pos_area_y;
feat.values.gyr_y_neg_area = neg_area_y;
feat.values.gyr_y_area_ratio = pos_area_y / (abs(neg_area_y) + EPS);
add_feature('gyr_y_pos_area', pos_area_y);
add_feature('gyr_y_neg_area', neg_area_y);
add_feature('gyr_y_area_ratio', feat.values.gyr_y_area_ratio);

%% ==================== ACCELERATION DIRECTION FEATURES (v2.3) ====================
acc_var = var(acc_win, 0, 1);  % 1x3
acc_var_total = sum(acc_var) + EPS;
feat.values.acc_var_ratio_x = acc_var(1) / acc_var_total;
feat.values.acc_var_ratio_y = acc_var(2) / acc_var_total;
feat.values.acc_var_ratio_z = acc_var(3) / acc_var_total;
add_feature('acc_var_ratio_x', feat.values.acc_var_ratio_x);
add_feature('acc_var_ratio_y', feat.values.acc_var_ratio_y);
add_feature('acc_var_ratio_z', feat.values.acc_var_ratio_z);

feat.values.rot_trans_ratio = feat.values.rms_gyr_total / (feat.values.rms_acc_total + EPS);
add_feature('rot_trans_ratio', feat.values.rot_trans_ratio);

%% Dominant Y-rotation sign
feat.values.gyr_y_sign = sign(feat.values.peak_gyr_y);
add_feature('gyr_y_sign', feat.values.gyr_y_sign);
    
    %% ==================== TWIST-SPECIFIC FEATURES (v2.2 NEW) ====================
    % These discriminate twist from flip_down (the main confusion pair)
    
    % Total integral magnitude for normalization
    total_integral_mag = abs(gyr_x_integral) + abs(gyr_y_integral) + abs(gyr_z_integral) + EPS;
    
    % X-axis dominance: HIGH for twist (rotation around forearm axis)
    % twist: X >> Y → dominance ~ 0.6-0.9
    % flip:  Y >> X → dominance ~ 0.1-0.3
    feat.values.gyr_x_dominance = abs(gyr_x_integral) / total_integral_mag;
    add_feature('gyr_x_dominance', feat.values.gyr_x_dominance);
    
    % Y-axis dominance: HIGH for flips
    feat.values.gyr_y_dominance = abs(gyr_y_integral) / total_integral_mag;
    add_feature('gyr_y_dominance', feat.values.gyr_y_dominance);
    
    % Z-axis dominance: for yaw-heavy gestures (circle, some shakes)
    feat.values.gyr_z_dominance = abs(gyr_z_integral) / total_integral_mag;
    add_feature('gyr_z_dominance', feat.values.gyr_z_dominance);
    
    % Roll-to-Pitch ratio: |X| / |Y|
    % twist: HIGH (X >> Y) → ratio > 1, often 2-5+
    % flip:  LOW  (Y >> X) → ratio < 1, often 0.1-0.5
    roll_pitch_ratio = abs(gyr_x_integral) / (abs(gyr_y_integral) + EPS);
    feat.values.roll_pitch_ratio = min(roll_pitch_ratio, 10);  % Cap outliers
    add_feature('roll_pitch_ratio', feat.values.roll_pitch_ratio);
    
    % Signed X/Y ratio (preserves twist direction)
    gyr_xy_ratio = gyr_x_integral / (abs(gyr_y_integral) + EPS);
    feat.values.gyr_xy_ratio = max(-10, min(10, gyr_xy_ratio));  % Clamp [-10, 10]
    add_feature('gyr_xy_ratio', feat.values.gyr_xy_ratio);
    
    % Axis dominance difference: |X_dom - Y_dom|
    % High for gestures with clear single-axis rotation (twist, pure flips)
    % Low for multi-axis gestures (circle, diagonal movements)
    feat.values.axis_dominance_diff = abs(feat.values.gyr_x_dominance - feat.values.gyr_y_dominance);
    add_feature('axis_dominance_diff', feat.values.axis_dominance_diff);
    
    % X vs Y winner (categorical): 1 = X dominant (twist), -1 = Y dominant (flip)
    feat.values.xy_dominant_axis = sign(abs(gyr_x_integral) - abs(gyr_y_integral));
    add_feature('xy_dominant_axis', feat.values.xy_dominant_axis);

    %% ==================== EULER ANGLE FEATURES (IF EKF AVAILABLE) ====================
    if ~isempty(est) && isfield(est, 'euler')
        if size(est.euler, 1) >= idx_end
            euler_win = est.euler(idx_start:idx_end, :);

            % Total Euler angle change
            euler_change = euler_win(end,:) - euler_win(1,:);
            euler_change = wrapToPi(euler_change);

            feat.values.delta_roll = euler_change(1);
            feat.values.delta_pitch = euler_change(2);
            feat.values.delta_yaw = euler_change(3);
            feat.values.delta_roll_deg = euler_change(1) * params.constants.rad2deg;
            feat.values.delta_pitch_deg = euler_change(2) * params.constants.rad2deg;
            feat.values.delta_yaw_deg = euler_change(3) * params.constants.rad2deg;

            add_feature('delta_roll', euler_change(1));
            add_feature('delta_pitch', euler_change(2));
            add_feature('delta_yaw', euler_change(3));
            add_feature('delta_roll_deg', feat.values.delta_roll_deg);
            add_feature('delta_pitch_deg', feat.values.delta_pitch_deg);
            add_feature('delta_yaw_deg', feat.values.delta_yaw_deg);
            
            % Pitch-specific features for flip_up vs flip_down
            pitch_win = euler_win(:, 2);
            
            feat.values.euler_pitch_range = max(pitch_win) - min(pitch_win);
            feat.values.euler_pitch_range_deg = feat.values.euler_pitch_range * params.constants.rad2deg;
            add_feature('euler_pitch_range', feat.values.euler_pitch_range);
            add_feature('euler_pitch_range_deg', feat.values.euler_pitch_range_deg);
            
            feat.values.euler_pitch_direction = sign(euler_change(2));
            add_feature('euler_pitch_direction', feat.values.euler_pitch_direction);
            
            pitch_relative = pitch_win - pitch_win(1);
            feat.values.euler_pitch_max = max(pitch_relative);
            feat.values.euler_pitch_min = min(pitch_relative);
            feat.values.euler_pitch_max_deg = feat.values.euler_pitch_max * params.constants.rad2deg;
            feat.values.euler_pitch_min_deg = feat.values.euler_pitch_min * params.constants.rad2deg;
            add_feature('euler_pitch_max', feat.values.euler_pitch_max);
            add_feature('euler_pitch_min', feat.values.euler_pitch_min);
            add_feature('euler_pitch_max_deg', feat.values.euler_pitch_max_deg);
            add_feature('euler_pitch_min_deg', feat.values.euler_pitch_min_deg);
            
            feat.values.euler_pitch_asymmetry = feat.values.euler_pitch_max + feat.values.euler_pitch_min;
            add_feature('euler_pitch_asymmetry', feat.values.euler_pitch_asymmetry);
            
            % Roll range (for twist detection via EKF)
            roll_win = euler_win(:, 1);
            feat.values.euler_roll_range = max(roll_win) - min(roll_win);
            feat.values.euler_roll_range_deg = feat.values.euler_roll_range * params.constants.rad2deg;
            add_feature('euler_roll_range', feat.values.euler_roll_range);
            add_feature('euler_roll_range_deg', feat.values.euler_roll_range_deg);
        end
    end

    %% Debug info
    feat.debug.n_samples = n_samples;
    feat.debug.idx_start = idx_start;
    feat.debug.idx_end = idx_end;
    feat.debug.t_start = t_win(1);
    feat.debug.t_end = t_win(end);
    feat.debug.Fs = Fs;
    feat.debug.version = '2.2';

    %% Nested function to add features consistently
    function add_feature(name, value)
        if ~isfinite(value)
            value = 0;
        end
        feat.names{end+1} = name;
        feat.x(end+1) = value;
    end

end

%% ==================== HELPER FUNCTIONS ====================

function count = count_zero_crossings(signal)
    signal = signal - mean(signal);
    signs = sign(signal);
    signs(signs == 0) = 1;
    count = sum(abs(diff(signs)) == 2);
end

function angle = wrapToPi(angle)
    angle = mod(angle + pi, 2*pi) - pi;
end

function s = skewness(x)
    x = x(:);
    mu = mean(x);
    sigma = std(x);
    if sigma < 1e-10
        s = 0;
    else
        s = mean(((x - mu) / sigma).^3);
    end
end

function k = kurtosis(x)
    x = x(:);
    mu = mean(x);
    sigma = std(x);
    if sigma < 1e-10
        k = 0;
    else
        k = mean(((x - mu) / sigma).^4);
    end
end