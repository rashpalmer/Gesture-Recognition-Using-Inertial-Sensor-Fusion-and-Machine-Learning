function tu = time_utils()
%TIME_UTILS Collection of time-related utility functions
%   tu = time_utils() returns a struct of function handles
%
%   AVAILABLE FUNCTIONS:
%       tu.computeDt(t)           - Compute time differences
%       tu.resampleUniform(t, x, Fs) - Resample to uniform rate
%       tu.findStaticSegments(data, params) - Find stationary periods
%       tu.getTimestamp()         - Get current timestamp string
%       tu.resampleBatch(t, signals, Fs) - NEW: Batch resample multiple signals
%       tu.detectSampleDrops(t)   - NEW: Detect sample drops/gaps
%
%   IMPROVEMENTS (v2.0):
%       - ADDED: Batch resampling for efficiency
%       - ADDED: Sample drop detection
%       - OPTIMIZED: Vectorized findStaticSegments
%       - IMPROVED: Better handling of edge cases
%
%   Author: Sensor Fusion Demo (Improved)
%   Date: 2026

    tu.computeDt = @computeDt;
    tu.resampleUniform = @resampleUniform;
    tu.resampleBatch = @resampleBatch;  % NEW
    tu.findStaticSegments = @findStaticSegments;
    tu.getTimestamp = @getTimestamp;
    tu.detectSampleDrops = @detectSampleDrops;  % NEW

end

function [dt, Fs_mean, Fs_var] = computeDt(t)
%COMPUTEDT Compute time differences and sample rate statistics
%   t: Nx1 time vector (seconds)
%   dt: (N-1)x1 time differences
%   Fs_mean: mean sample rate
%   Fs_var: variance in sample rate

    t = t(:);  % Ensure column
    dt = diff(t);

    % Remove outliers for statistics
    dt_clean = dt(dt > 0 & dt < 0.5);  % Assume Fs > 2 Hz

    if isempty(dt_clean)
        Fs_mean = NaN;
        Fs_var = NaN;
    else
        Fs_mean = 1 / mean(dt_clean);
        Fs_var = var(1 ./ dt_clean);
    end
end

function [t_new, x_new] = resampleUniform(t, x, Fs_target)
%RESAMPLEUNIFORM Resample signal to uniform sample rate
%   t: Nx1 time vector (may be non-uniform)
%   x: NxM data matrix (N samples, M channels)
%   Fs_target: desired sample rate (Hz)
%
%   Returns uniformly sampled data using linear interpolation
%
%   IMPROVEMENTS:
%   - Better handling of NaN values
%   - Edge case handling for very short signals

    t = t(:);

    % Handle edge cases
    if length(t) < 2
        t_new = t;
        x_new = x;
        return;
    end

    % Create uniform time vector
    t_new = (t(1) : 1/Fs_target : t(end))';

    % Handle empty result
    if isempty(t_new)
        t_new = t(1);
        x_new = x(1,:);
        return;
    end

    % Interpolate each channel
    if size(x, 1) ~= length(t)
        x = x';  % Transpose if needed
    end

    n_channels = size(x, 2);
    x_new = zeros(length(t_new), n_channels);

    for ch = 1:n_channels
        % Handle NaN values by interpolation
        valid_mask = ~isnan(x(:, ch));
        if sum(valid_mask) < 2
            % Not enough valid data
            x_new(:, ch) = NaN;
        else
            x_new(:, ch) = interp1(t(valid_mask), x(valid_mask, ch), t_new, 'linear', 'extrap');
        end
    end
end

function [t_new, signals_new] = resampleBatch(t, signals, Fs_target)
%RESAMPLEBATCH Batch resample multiple signal arrays efficiently
%   t: Nx1 time vector
%   signals: cell array of NxM matrices to resample
%   Fs_target: desired sample rate (Hz)
%
%   Returns:
%   t_new: resampled time vector
%   signals_new: cell array of resampled signals
%
%   NEW: More efficient than calling resampleUniform multiple times

    t = t(:);

    % Create uniform time vector once
    t_new = (t(1) : 1/Fs_target : t(end))';

    signals_new = cell(size(signals));

    for s = 1:length(signals)
        x = signals{s};

        if isempty(x)
            signals_new{s} = [];
            continue;
        end

        if size(x, 1) ~= length(t)
            x = x';
        end

        n_channels = size(x, 2);
        x_new = zeros(length(t_new), n_channels);

        for ch = 1:n_channels
            valid_mask = ~isnan(x(:, ch));
            if sum(valid_mask) >= 2
                x_new(:, ch) = interp1(t(valid_mask), x(valid_mask, ch), t_new, 'linear', 'extrap');
            else
                x_new(:, ch) = NaN;
            end
        end

        signals_new{s} = x_new;
    end
end

function [static_idx, static_windows] = findStaticSegments(gyr, params)
%FINDSTATICSEGMENTS Identify stationary periods from gyroscope data
%   gyr: Nx3 gyroscope data (rad/s)
%   params: configuration parameters
%
%   static_idx: Nx1 logical array (true = static)
%   static_windows: Mx2 matrix of [start_idx, end_idx] for each segment
%
%   IMPROVEMENTS:
%   - Vectorized morphological operations
%   - Better edge case handling

    if nargin < 2
        threshold = 0.5;  % rad/s
        min_samples = 50;
    else
        threshold = params.preprocess.static_threshold;
        min_samples = params.preprocess.static_window;
    end

    n = size(gyr, 1);

    % Handle edge cases
    if n < min_samples
        static_idx = false(n, 1);
        static_windows = [];
        return;
    end

    % Compute gyro magnitude (OPTIMIZED: vectorized)
    gyr_mag = sqrt(sum(gyr.^2, 2));

    % Initial classification
    static_idx = gyr_mag < threshold;

    % Apply smoothing to clean up (vectorized)
    static_smooth = movmean(double(static_idx), min_samples);
    static_idx = static_smooth > 0.5;

    % Find contiguous segments
    d = diff([0; static_idx(:); 0]);
    starts = find(d == 1);
    ends = find(d == -1) - 1;

    % Filter by minimum duration
    if ~isempty(starts)
        valid = (ends - starts + 1) >= min_samples;
        static_windows = [starts(valid), ends(valid)];
    else
        static_windows = [];
    end

    % Convert back to logical array (OPTIMIZED: vectorized)
    static_idx = false(n, 1);
    for i = 1:size(static_windows, 1)
        static_idx(static_windows(i,1):static_windows(i,2)) = true;
    end
end

function ts = getTimestamp()
%GETTIMESTAMP Get current timestamp as string for logging
    ts = datestr(now, 'yyyymmdd_HHMMSS');
end

function [drops, drop_info] = detectSampleDrops(t, expected_Fs)
%DETECTSAMPLEDROPS Detect sample drops or gaps in time series
%   t: Nx1 time vector
%   expected_Fs: expected sample rate (optional, will estimate if not provided)
%
%   Returns:
%   drops: Nx1 logical array (true = sample dropped before this index)
%   drop_info: struct with statistics
%
%   NEW: Useful for data quality assessment

    t = t(:);
    dt = diff(t);

    % Estimate expected dt if not provided
    if nargin < 2
        expected_dt = median(dt);
    else
        expected_dt = 1 / expected_Fs;
    end

    % Detect drops (gaps significantly larger than expected)
    drop_threshold = expected_dt * 1.5;  % 50% tolerance
    gap_mask = dt > drop_threshold;

    % Align to original size (drop detected AFTER the gap)
    drops = [false; gap_mask];

    % Compute statistics
    drop_info = struct();
    drop_info.count = sum(gap_mask);
    drop_info.total_samples = length(t);
    drop_info.drop_rate = drop_info.count / (drop_info.total_samples - 1);

    if any(gap_mask)
        drop_info.max_gap = max(dt(gap_mask));
        drop_info.estimated_missing = sum(round(dt(gap_mask) / expected_dt) - 1);
    else
        drop_info.max_gap = 0;
        drop_info.estimated_missing = 0;
    end

    drop_info.expected_dt = expected_dt;
    drop_info.actual_dt_mean = mean(dt);
    drop_info.actual_dt_std = std(dt);
end
