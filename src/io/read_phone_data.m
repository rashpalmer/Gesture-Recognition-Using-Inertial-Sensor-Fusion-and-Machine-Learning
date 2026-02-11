function data = read_phone_data(filepath, varargin)
%READ_PHONE_DATA Import sensor data from MATLAB Mobile exports
%   data = read_phone_data(filepath) reads sensor data from the specified
%   file and returns a standardized data structure.
%
%   data = read_phone_data(filepath, 'Parameter', value, ...) allows
%   additional options:
%       'Format'        - 'auto', 'mat', 'csv' (default: 'auto')
%       'TimeOffset'    - Subtract this from timestamps (default: 0)
%       'Verbose'       - Print import details (default: true)
%
%   IMPROVEMENTS (v2.0):
%       - ADDED: Better error messages
%       - ADDED: Support for more data formats
%       - ADDED: Data quality warnings
%       - IMPROVED: Robust handling of missing fields
%
%   SUPPORTED FORMATS:
%       .mat - MATLAB Mobile logged data (sensorlog_*.mat)
%       .csv - Exported CSV from MATLAB Mobile
%
%   OUTPUT STRUCTURE:
%       data.t      - Nx1 time vector (seconds from start)
%       data.acc    - Nx3 accelerometer data [ax, ay, az] (m/s²)
%       data.gyr    - Nx3 gyroscope data [gx, gy, gz] (rad/s)
%       data.mag    - Nx3 magnetometer data [mx, my, mz] (µT)
%       data.orient - Nx4 device orientation quaternion (if available)
%       data.meta   - Struct with metadata:
%           .source     - Original filename
%           .Fs         - Estimated sample rate
%           .duration   - Recording duration (seconds)
%           .device     - Device name (if available)
%           .date       - Recording date (if available)
%
%   MATLAB MOBILE DATA NOTES:
%       - Accelerometer: logged in m/s² (includes gravity)
%       - Gyroscope: logged in rad/s
%       - Magnetometer: logged in µT (microtesla)
%       - Orientation: device quaternion (optional)
%
%   EXAMPLE:
%       data = read_phone_data('data/raw/sensorlog_123.mat');
%       plot(data.t, data.acc);
%       xlabel('Time (s)'); ylabel('Acceleration (m/s²)');
%
%   Author: Sensor Fusion Demo (Improved)
%   Date: 2026

    %% Parse inputs
    p = inputParser;
    addRequired(p, 'filepath', @ischar);
    addParameter(p, 'Format', 'auto', @ischar);
    addParameter(p, 'TimeOffset', 0, @isnumeric);
    addParameter(p, 'Verbose', true, @islogical);
    parse(p, filepath, varargin{:});
    opts = p.Results;

    %% Check file exists
    if ~exist(filepath, 'file')
        error('read_phone_data:FileNotFound', ...
            'File not found: %s', filepath);
    end

    %% Determine format
    [~, fname, ext] = fileparts(filepath);

    if strcmpi(opts.Format, 'auto')
        switch lower(ext)
            case '.mat'
                format = 'mat';
            case '.csv'
                format = 'csv';
            otherwise
                error('read_phone_data:UnknownFormat', ...
                    'Cannot auto-detect format for extension: %s', ext);
        end
    else
        format = lower(opts.Format);
    end

    %% Load data based on format
    if opts.Verbose
        fprintf('Loading %s data from: %s\n', upper(format), filepath);
    end

    switch format
        case 'mat'
            data = load_mat_format(filepath, opts);
        case 'csv'
            data = load_csv_format(filepath, opts);
        otherwise
            error('read_phone_data:UnsupportedFormat', ...
                'Unsupported format: %s', format);
    end

    %% Post-processing
    % Apply time offset
    if opts.TimeOffset ~= 0
        data.t = data.t - opts.TimeOffset;
    end

    % Start time at zero
    data.t = data.t - data.t(1);

    % Compute metadata
    data.meta.source = [fname, ext];
    data.meta.Fs = 1 / mean(diff(data.t));
    data.meta.duration = data.t(end) - data.t(1);
    data.meta.n_samples = length(data.t);

    %% Validation
    validate_data(data, opts);

    if opts.Verbose
        fprintf('Loaded %d samples (%.1f seconds at %.1f Hz)\n', ...
            data.meta.n_samples, data.meta.duration, data.meta.Fs);
        fprintf('  Accelerometer: %s\n', data_status(data.acc));
        fprintf('  Gyroscope:     %s\n', data_status(data.gyr));
        fprintf('  Magnetometer:  %s\n', data_status(data.mag));
        if isfield(data, 'orient')
            fprintf('  Orientation:   %s\n', data_status(data.orient));
        end
    end
end

%% ==================== FORMAT-SPECIFIC LOADERS ====================

function data = load_mat_format(filepath, opts)
%LOAD_MAT_FORMAT Load data from MATLAB Mobile .mat file

    raw = load(filepath);

    % MATLAB Mobile typically creates variables like:
    % Acceleration, AngularVelocity, MagneticField, Orientation
    % with timetable format in newer versions

    % Try to find accelerometer data
    data.acc = [];
    data.gyr = [];
    data.mag = [];
    data.orient = [];
    data.t = [];

    % Check for common variable names
    var_names = fieldnames(raw);

    % Accelerometer
    acc_names = {'Acceleration', 'acceleration', 'acc', 'accel', 'a'};
    for i = 1:length(acc_names)
        if ismember(acc_names{i}, var_names)
            acc_var = raw.(acc_names{i});
            [data.acc, t_acc] = extract_timetable_data(acc_var);
            if isempty(data.t)
                data.t = t_acc;
            end
            break;
        end
    end

    % Gyroscope
    gyr_names = {'AngularVelocity', 'angularvelocity', 'gyro', 'gyr', 'Gyroscope', 'w'};
    for i = 1:length(gyr_names)
        if ismember(gyr_names{i}, var_names)
            gyr_var = raw.(gyr_names{i});
            [data.gyr, t_gyr] = extract_timetable_data(gyr_var);
            if isempty(data.t)
                data.t = t_gyr;
            end
            break;
        end
    end

    % Magnetometer
    mag_names = {'MagneticField', 'magneticfield', 'mag', 'Magnetometer', 'm'};
    for i = 1:length(mag_names)
        if ismember(mag_names{i}, var_names)
            mag_var = raw.(mag_names{i});
            [data.mag, t_mag] = extract_timetable_data(mag_var);
            if isempty(data.t)
                data.t = t_mag;
            end
            break;
        end
    end

    % Orientation (optional)
    orient_names = {'Orientation', 'orientation', 'quat', 'attitude'};
    for i = 1:length(orient_names)
        if ismember(orient_names{i}, var_names)
            orient_var = raw.(orient_names{i});
            [data.orient, ~] = extract_timetable_data(orient_var);
            break;
        end
    end

    % Try alternative structure format (some MATLAB Mobile versions)
    if isempty(data.acc) && ismember('sensorData', var_names)
        sd = raw.sensorData;
        if isfield(sd, 'Acceleration')
            data.acc = sd.Acceleration;
        end
        if isfield(sd, 'AngularVelocity')
            data.gyr = sd.AngularVelocity;
        end
        if isfield(sd, 'MagneticField')
            data.mag = sd.MagneticField;
        end
        if isfield(sd, 'Timestamp')
            data.t = sd.Timestamp;
        end
    end

    % Check for Position (sometimes includes timestamp)
    if isempty(data.t) && ismember('Position', var_names)
        pos_var = raw.Position;
        [~, data.t] = extract_timetable_data(pos_var);
    end

    % Fill in missing data with NaN
    n = max([size(data.acc,1), size(data.gyr,1), size(data.mag,1)]);
    if isempty(data.t) && n > 0
        data.t = (0:n-1)' / 100;  % Assume 100 Hz
        warning('No timestamp found, assuming 100 Hz');
    end

    if isempty(data.acc), data.acc = nan(n, 3); end
    if isempty(data.gyr), data.gyr = nan(n, 3); end
    if isempty(data.mag), data.mag = nan(n, 3); end

    % Extract device info if available
    data.meta.device = 'Unknown';
    if ismember('DeviceName', var_names)
        data.meta.device = raw.DeviceName;
    end

    data.meta.date = 'Unknown';
    if ismember('Date', var_names)
        data.meta.date = raw.Date;
    end
end

function data = load_csv_format(filepath, opts)
%LOAD_CSV_FORMAT Load data from CSV export

    % Read table with automatic header detection
    T = readtable(filepath, 'VariableNamingRule', 'preserve');

    % Get column names (case-insensitive matching)
    col_names = lower(T.Properties.VariableNames);

    % Find timestamp column
    t_idx = find(contains(col_names, 'time') | contains(col_names, 'timestamp'), 1);
    if isempty(t_idx)
        % Assume first column is timestamp
        t_idx = 1;
    end
    data.t = table2array(T(:, t_idx));

    % Convert datetime to seconds if needed
    if isdatetime(data.t)
        data.t = seconds(data.t - data.t(1));
    elseif isduration(data.t)
        data.t = seconds(data.t);
    end

    % Find accelerometer columns
    acc_idx = find(contains(col_names, 'acc') | contains(col_names, 'accel'));
    if length(acc_idx) >= 3
        data.acc = table2array(T(:, acc_idx(1:3)));
    else
        ax_idx = find(contains(col_names, 'ax') | contains(col_names, 'acc_x') | strcmp(col_names, 'x'));
        ay_idx = find(contains(col_names, 'ay') | contains(col_names, 'acc_y') | strcmp(col_names, 'y'));
        az_idx = find(contains(col_names, 'az') | contains(col_names, 'acc_z') | strcmp(col_names, 'z'));
        if ~isempty(ax_idx) && ~isempty(ay_idx) && ~isempty(az_idx)
            data.acc = table2array(T(:, [ax_idx(1), ay_idx(1), az_idx(1)]));
        else
            data.acc = nan(height(T), 3);
        end
    end

    % Find gyroscope columns
    gyr_idx = find(contains(col_names, 'gyr') | contains(col_names, 'angular') | contains(col_names, 'omega'));
    if length(gyr_idx) >= 3
        data.gyr = table2array(T(:, gyr_idx(1:3)));
    else
        gx_idx = find(contains(col_names, 'gx') | contains(col_names, 'gyr_x') | contains(col_names, 'wx'));
        gy_idx = find(contains(col_names, 'gy') | contains(col_names, 'gyr_y') | contains(col_names, 'wy'));
        gz_idx = find(contains(col_names, 'gz') | contains(col_names, 'gyr_z') | contains(col_names, 'wz'));
        if ~isempty(gx_idx) && ~isempty(gy_idx) && ~isempty(gz_idx)
            data.gyr = table2array(T(:, [gx_idx(1), gy_idx(1), gz_idx(1)]));
        else
            data.gyr = nan(height(T), 3);
        end
    end

    % Find magnetometer columns
    mag_idx = find(contains(col_names, 'mag') | contains(col_names, 'magnetic'));
    if length(mag_idx) >= 3
        data.mag = table2array(T(:, mag_idx(1:3)));
    else
        mx_idx = find(contains(col_names, 'mx') | contains(col_names, 'mag_x'));
        my_idx = find(contains(col_names, 'my') | contains(col_names, 'mag_y'));
        mz_idx = find(contains(col_names, 'mz') | contains(col_names, 'mag_z'));
        if ~isempty(mx_idx) && ~isempty(my_idx) && ~isempty(mz_idx)
            data.mag = table2array(T(:, [mx_idx(1), my_idx(1), mz_idx(1)]));
        else
            data.mag = nan(height(T), 3);
        end
    end

    % Orientation (optional)
    quat_idx = find(contains(col_names, 'quat') | contains(col_names, 'orient'));
    if length(quat_idx) >= 4
        data.orient = table2array(T(:, quat_idx(1:4)));
    end

    data.meta.device = 'CSV Import';
    data.meta.date = datestr(now);
end

%% ==================== HELPER FUNCTIONS ====================

function [values, timestamps] = extract_timetable_data(var)
%EXTRACT_TIMETABLE_DATA Extract data from timetable or matrix format

    values = [];
    timestamps = [];

    if istimetable(var)
        % Timetable format (newer MATLAB Mobile)
        timestamps = seconds(var.Properties.RowTimes - var.Properties.RowTimes(1));
        values = table2array(var(:, 1:min(end,4)));  % Up to 4 columns
    elseif istable(var)
        % Table format
        values = table2array(var(:, 1:min(end,4)));
        timestamps = [];
    elseif isnumeric(var)
        % Direct numeric array
        values = var;
        timestamps = [];
    elseif isstruct(var)
        % Struct with Data and Time fields
        if isfield(var, 'Data')
            values = var.Data;
        end
        if isfield(var, 'Time')
            timestamps = var.Time;
            if isdatetime(timestamps)
                timestamps = seconds(timestamps - timestamps(1));
            end
        end
    end
end

function validate_data(data, opts)
%VALIDATE_DATA Check data integrity and warn about issues

    n = length(data.t);

    % Check array sizes match
    if size(data.acc, 1) ~= n && ~all(isnan(data.acc(:)))
        warning('Accelerometer data length (%d) does not match time (%d)', ...
            size(data.acc, 1), n);
    end

    if size(data.gyr, 1) ~= n && ~all(isnan(data.gyr(:)))
        warning('Gyroscope data length (%d) does not match time (%d)', ...
            size(data.gyr, 1), n);
    end

    if size(data.mag, 1) ~= n && ~all(isnan(data.mag(:)))
        warning('Magnetometer data length (%d) does not match time (%d)', ...
            size(data.mag, 1), n);
    end

    % Check for reasonable values
    if ~all(isnan(data.acc(:)))
        acc_mag = sqrt(sum(data.acc.^2, 2));
        if mean(acc_mag) < 5 || mean(acc_mag) > 50
            warning('Accelerometer magnitude (%.1f m/s²) seems unusual - check units', mean(acc_mag));
        end
    end

    if ~all(isnan(data.gyr(:)))
        gyr_max = max(abs(data.gyr(:)));
        if gyr_max > 50
            warning('Gyroscope max (%.1f rad/s) is very high - check units', gyr_max);
        end
    end

    % Check for sample drops
    dt = diff(data.t);
    dt_median = median(dt);
    large_gaps = dt > dt_median * 2;
    if any(large_gaps)
        warning('%d potential sample drops detected (%.1f%% of samples)', ...
            sum(large_gaps), 100*sum(large_gaps)/length(dt));
    end
end

function s = data_status(arr)
%DATA_STATUS Generate status string for data array
    if all(isnan(arr(:)))
        s = 'NOT AVAILABLE';
    else
        n_valid = sum(~any(isnan(arr), 2));
        s = sprintf('%d samples, range [%.2f, %.2f]', ...
            n_valid, min(arr(:)), max(arr(:)));
    end
end
