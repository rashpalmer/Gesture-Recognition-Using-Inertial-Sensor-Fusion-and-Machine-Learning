function seg = segment_gesture(imu, params)
%SEGMENT_GESTURE Detect and segment gestures from IMU data
%   seg = segment_gesture(imu, params) identifies gesture boundaries using
%   energy-based thresholding with hysteresis.
%
%   INPUTS:
%       imu     - Preprocessed IMU data from preprocess_imu()
%       params  - Configuration from config_params() (optional)
%
%   OUTPUT:
%       seg - Segmentation results struct:
%           .windows    - Mx2 matrix of [start_idx, end_idx] for each gesture
%           .n_gestures - Number of gestures detected
%           .energy     - Nx1 motion energy signal
%           .state      - Nx1 state machine output (0=quiet, 1=active)
%           .primary    - Index of most prominent gesture
%           .winIdx     - [start, end] of primary gesture
%           .score      - Confidence score for primary gesture
%
%   IMPROVEMENTS (v2.0):
%       - FIXED: Normalized energy calculation (gyro and acc balanced)
%       - ADDED: Input validation
%       - ADDED: Configurable normalization factors
%       - OPTIMIZED: Vectorized operations where possible
%       - ADDED: Debug information for troubleshooting
%
%   Author: Sensor Fusion Demo (Improved)
%   Date: 2026

    %% Input validation and default parameters
    if nargin < 2 || isempty(params)
        params = config_params();
    end

    validateattributes(imu, {'struct'}, {'nonempty'}, mfilename, 'imu', 1);
    assert(isfield(imu, 'gyr'), 'IMU struct must contain gyr field');
    assert(isfield(imu, 'acc'), 'IMU struct must contain acc field');
    assert(isfield(imu, 't'), 'IMU struct must contain t field');

    if params.verbose
        fprintf('Segmenting gestures...\n');
    end

    %% Initialize
    n = length(imu.t);
    dt = mean(imu.dt);
    Fs = imu.Fs;
    EPS = params.EPS;

    % Get thresholds from params
    energy_low = params.segmentation.energy_low;
    energy_high = params.segmentation.energy_high;
    min_duration = params.segmentation.min_duration;
    max_duration = params.segmentation.max_duration;
    pre_buffer = params.segmentation.pre_buffer;
    post_buffer = params.segmentation.post_buffer;
    max_gestures = params.segmentation.max_gestures;
    min_gap = params.segmentation.min_gap;

    % Get normalization factors (FIX: balanced energy calculation)
    gyro_norm_factor = params.segmentation.gyro_norm_factor;
    acc_norm_factor = params.segmentation.acc_norm_factor;
    acc_weight = params.segmentation.acc_weight;

    % Convert durations to samples
    min_samples = round(min_duration * Fs);
    max_samples = round(max_duration * Fs);
    pre_samples = round(pre_buffer * Fs);
    post_samples = round(post_buffer * Fs);
    gap_samples = round(min_gap * Fs);

    %% Compute motion energy - FIXED: Normalized components
    % Primary: gyroscope magnitude (rotation is key for gestures)
    gyr_mag = sqrt(sum(imu.gyr.^2, 2));
    gyr_mag_norm = gyr_mag / gyro_norm_factor;  % Normalize to typical range

    % Secondary: accelerometer magnitude deviation from gravity
    acc_mag = sqrt(sum(imu.acc.^2, 2));
    acc_dev = abs(acc_mag - params.constants.g);
    acc_dev_norm = acc_dev / acc_norm_factor;  % Normalize to typical range

    % Combined energy (FIXED: balanced weighting after normalization)
    energy = gyr_mag_norm + acc_weight * acc_dev_norm;

    % Smooth energy signal
    window_size = max(3, round(0.05 * Fs));  % 50ms window
    energy_smooth = movmean(energy, window_size);

    seg.energy = energy_smooth;
    seg.energy_raw = energy;

    %% Hysteresis thresholding (state machine)
    state = zeros(n, 1);  % 0 = quiet, 1 = active
    current_state = 0;

    for k = 1:n
        if current_state == 0  % Quiet
            if energy_smooth(k) > energy_high
                current_state = 1;
            end
        else  % Active
            if energy_smooth(k) < energy_low
                current_state = 0;
            end
        end
        state(k) = current_state;
    end

    seg.state = state;

    %% Find contiguous active regions
    d = diff([0; state; 0]);
    starts = find(d == 1);
    ends = find(d == -1) - 1;

    n_raw = length(starts);
    if params.verbose
        fprintf('  Found %d raw active regions\n', n_raw);
    end

    %% Filter by duration
    windows = [];

    for i = 1:n_raw
        duration_samples = ends(i) - starts(i) + 1;

        if duration_samples >= min_samples && duration_samples <= max_samples
            % Add pre/post buffer
            win_start = max(1, starts(i) - pre_samples);
            win_end = min(n, ends(i) + post_samples);

            windows = [windows; win_start, win_end];
        end
    end

    %% Merge close windows
    if size(windows, 1) > 1
        merged = windows(1, :);

        for i = 2:size(windows, 1)
            if windows(i, 1) - merged(end, 2) < gap_samples
                % Merge with previous
                merged(end, 2) = windows(i, 2);
            else
                % Start new window
                merged = [merged; windows(i, :)];
            end
        end

        windows = merged;
    end

    %% Limit number of gestures (keep highest energy ones)
    if size(windows, 1) > max_gestures
        energies = zeros(size(windows, 1), 1);
        for i = 1:size(windows, 1)
            energies(i) = sum(energy_smooth(windows(i,1):windows(i,2)));
        end

        [~, idx] = sort(energies, 'descend');
        windows = windows(idx(1:max_gestures), :);

        % Re-sort by time
        [~, idx] = sort(windows(:, 1));
        windows = windows(idx, :);
    end

    %% Store results
    seg.windows = windows;
    seg.n_gestures = size(windows, 1);

    if params.verbose
        fprintf('  Detected %d valid gestures\n', seg.n_gestures);
    end

    %% Select primary gesture (highest energy)
    if seg.n_gestures > 0
        energies = zeros(seg.n_gestures, 1);
        for i = 1:seg.n_gestures
            energies(i) = sum(energy_smooth(windows(i,1):windows(i,2)));
        end

        [max_energy, primary] = max(energies);
        seg.primary = primary;
        seg.winIdx = windows(primary, :);

        % Compute confidence score based on energy prominence
        if seg.n_gestures > 1
            sorted_energies = sort(energies, 'descend');
            seg.score = max_energy / (max_energy + sorted_energies(2) + EPS);
        else
            seg.score = 1.0;
        end

        if params.verbose
            fprintf('  Primary gesture: window %d (%.2f - %.2f s), score: %.2f\n', ...
                primary, imu.t(seg.winIdx(1)), imu.t(seg.winIdx(2)), seg.score);
        end
    else
        seg.primary = 0;
        seg.winIdx = [1, n];
        seg.score = 0;
        if params.verbose
            fprintf('  No gestures detected, using entire signal\n');
        end
    end

    % Store time reference and debug info
    seg.t = imu.t;
    seg.Fs = Fs;
    seg.debug.n_raw_regions = n_raw;
    seg.debug.energy_threshold_low = energy_low;
    seg.debug.energy_threshold_high = energy_high;
    seg.debug.min_samples = min_samples;
    seg.debug.max_samples = max_samples;

end
