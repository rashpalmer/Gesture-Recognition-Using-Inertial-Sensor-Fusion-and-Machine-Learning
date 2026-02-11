function [t_out, varargout] = resample_signals(t_in, targetFs, varargin)
%RESAMPLE_SIGNALS Resample multiple signals to uniform sampling rate
%
%   Resamples time series data to a fixed sampling rate using various
%   interpolation methods. Handles edge cases like irregular sampling,
%   gaps, and variable-length inputs.
%
%   Syntax:
%       [t_out, sig1_out, sig2_out, ...] = resample_signals(t_in, targetFs, sig1, sig2, ...)
%       [t_out, ...] = resample_signals(t_in, targetFs, sig1, ..., 'Method', 'linear')
%       [t_out, ...] = resample_signals(t_in, targetFs, sig1, ..., 'Name', Value)
%
%   Inputs:
%       t_in     - Original time vector (Nx1, seconds)
%       targetFs - Target sampling frequency (Hz)
%       sig1, sig2, ... - Signals to resample (Nx1 or NxM arrays)
%
%   Name-Value Parameters:
%       'Method'      - Interpolation method: 'linear' (default), 'spline', 
%                       'pchip', 'nearest', 'makima'
%       'StartTime'   - Output start time (default: t_in(1))
%       'EndTime'     - Output end time (default: t_in(end))
%       'GapThreshold'- Max gap before inserting NaN (default: 5/targetFs)
%       'FillGaps'    - How to handle gaps: 'nan', 'hold', 'interp' (default: 'nan')
%       'Verbose'     - Print diagnostic info (default: false)
%
%   Outputs:
%       t_out         - Uniformly sampled time vector
%       sig1_out, ... - Resampled signals (same order as inputs)
%
%   Example:
%       % Resample accelerometer and gyroscope to 100 Hz
%       [t, acc, gyr] = resample_signals(raw_t, 100, raw_acc, raw_gyr);
%
%       % Use spline interpolation with custom time range
%       [t, acc] = resample_signals(raw_t, 100, raw_acc, ...
%                                   'Method', 'spline', ...
%                                   'StartTime', 0, 'EndTime', 10);
%
%   Notes:
%       - Input signals can be vectors (Nx1) or matrices (NxM)
%       - Output preserves the column structure of inputs
%       - Large gaps in data are marked with NaN by default
%       - For IMU data, 'linear' interpolation is usually sufficient
%
%   See also: interp1, resample, time_utils
%
%   Author: Claude (Anthropic)
%   Date: January 2025

    %% Parse inputs
    [signals, options] = parseInputs(t_in, targetFs, varargin{:});
    
    %% Validate time vector
    t_in = t_in(:);  % Ensure column
    N_in = length(t_in);
    
    if N_in < 2
        error('resample_signals:tooShort', 'Need at least 2 samples to resample');
    end
    
    % Check monotonicity
    dt_in = diff(t_in);
    if any(dt_in <= 0)
        error('resample_signals:notMonotonic', 'Time vector must be strictly increasing');
    end
    
    %% Analyze input sampling
    dtMean = mean(dt_in);
    dtStd = std(dt_in);
    FsIn = 1 / dtMean;
    
    if options.Verbose
        fprintf('  [resample_signals] Input: %d samples, ~%.1f Hz (CV=%.1f%%)\n', ...
            N_in, FsIn, 100*dtStd/dtMean);
    end
    
    %% Create output time vector
    tStart = options.StartTime;
    tEnd = options.EndTime;
    dt_out = 1 / targetFs;
    
    t_out = (tStart : dt_out : tEnd)';
    N_out = length(t_out);
    
    if options.Verbose
        fprintf('  [resample_signals] Output: %d samples at %.1f Hz\n', N_out, targetFs);
        fprintf('  [resample_signals] Time range: [%.3f, %.3f] s\n', tStart, tEnd);
    end
    
    %% Detect gaps in input data
    gapIndices = find(dt_in > options.GapThreshold);
    if ~isempty(gapIndices) && options.Verbose
        fprintf('  [resample_signals] Detected %d gaps (>%.3f s)\n', ...
            length(gapIndices), options.GapThreshold);
    end
    
    %% Resample each signal
    numSignals = length(signals);
    varargout = cell(1, numSignals);
    
    for i = 1:numSignals
        sig_in = signals{i};
        
        % Validate signal dimensions
        if size(sig_in, 1) ~= N_in
            error('resample_signals:sizeMismatch', ...
                'Signal %d has %d rows, expected %d (same as time vector)', ...
                i, size(sig_in, 1), N_in);
        end
        
        numCols = size(sig_in, 2);
        sig_out = zeros(N_out, numCols);
        
        % Interpolate each column
        for c = 1:numCols
            sig_out(:, c) = interp1(t_in, sig_in(:, c), t_out, options.Method, 'extrap');
        end
        
        % Handle gaps
        if ~isempty(gapIndices) && ~strcmp(options.FillGaps, 'interp')
            sig_out = handleGaps(t_in, t_out, sig_out, gapIndices, dt_in, options);
        end
        
        % Handle extrapolation (times outside input range)
        outsideMask = (t_out < t_in(1)) | (t_out > t_in(end));
        if any(outsideMask)
            if options.Verbose
                fprintf('  [resample_signals] Warning: %d samples extrapolated\n', sum(outsideMask));
            end
            % Set extrapolated values to NaN or edge values based on method
            if strcmp(options.FillGaps, 'nan')
                sig_out(outsideMask, :) = NaN;
            elseif strcmp(options.FillGaps, 'hold')
                % Hold first/last values
                beforeMask = t_out < t_in(1);
                afterMask = t_out > t_in(end);
                sig_out(beforeMask, :) = repmat(sig_in(1, :), sum(beforeMask), 1);
                sig_out(afterMask, :) = repmat(sig_in(end, :), sum(afterMask), 1);
            end
        end
        
        varargout{i} = sig_out;
    end
    
    if options.Verbose
        fprintf('  [resample_signals] Resampling complete\n');
    end
end

%% ======================== HELPER FUNCTIONS ========================

function [signals, options] = parseInputs(t_in, targetFs, varargin)
%PARSEINPUTS Parse and validate inputs

    % Default options
    options = struct();
    options.Method = 'linear';
    options.StartTime = t_in(1);
    options.EndTime = t_in(end);
    options.GapThreshold = 5 / targetFs;  % 5x expected sample period
    options.FillGaps = 'nan';
    options.Verbose = false;
    
    % Separate signals from name-value pairs
    signals = {};
    nvStart = length(varargin) + 1;  % Index where name-value pairs start
    
    for i = 1:length(varargin)
        if ischar(varargin{i}) || isstring(varargin{i})
            % Found start of name-value pairs
            nvStart = i;
            break;
        else
            % This is a signal
            signals{end+1} = varargin{i}; %#ok<AGROW>
        end
    end
    
    % Parse name-value pairs
    i = nvStart;
    while i <= length(varargin)
        name = varargin{i};
        if i + 1 > length(varargin)
            error('resample_signals:missingValue', 'Missing value for parameter %s', name);
        end
        value = varargin{i+1};
        
        switch lower(name)
            case 'method'
                validMethods = {'linear', 'spline', 'pchip', 'nearest', 'makima'};
                if ~ismember(lower(value), validMethods)
                    error('resample_signals:invalidMethod', ...
                        'Method must be one of: %s', strjoin(validMethods, ', '));
                end
                options.Method = lower(value);
            case 'starttime'
                options.StartTime = value;
            case 'endtime'
                options.EndTime = value;
            case 'gapthreshold'
                options.GapThreshold = value;
            case 'fillgaps'
                validFills = {'nan', 'hold', 'interp'};
                if ~ismember(lower(value), validFills)
                    error('resample_signals:invalidFillGaps', ...
                        'FillGaps must be one of: %s', strjoin(validFills, ', '));
                end
                options.FillGaps = lower(value);
            case 'verbose'
                options.Verbose = logical(value);
            otherwise
                error('resample_signals:unknownParam', 'Unknown parameter: %s', name);
        end
        
        i = i + 2;
    end
    
    if isempty(signals)
        error('resample_signals:noSignals', 'At least one signal must be provided');
    end
end

function sig_out = handleGaps(t_in, t_out, sig_out, gapIndices, dt_in, options)
%HANDLEGAPS Mark or fill gaps in resampled signal

    for g = 1:length(gapIndices)
        idx = gapIndices(g);
        gapStart = t_in(idx);
        gapEnd = t_in(idx + 1);
        
        % Find output samples within this gap
        inGap = (t_out > gapStart) & (t_out < gapEnd);
        
        if any(inGap)
            switch options.FillGaps
                case 'nan'
                    sig_out(inGap, :) = NaN;
                case 'hold'
                    % Hold last valid value
                    sig_out(inGap, :) = repmat(sig_out(find(t_out <= gapStart, 1, 'last'), :), ...
                        sum(inGap), 1);
                % 'interp' case: already interpolated, do nothing
            end
        end
    end
end

%% ======================== ADDITIONAL UTILITY FUNCTIONS ========================

function [t_out, data_out] = resampleStruct(data, targetFs, fields, varargin)
%RESAMPLESTRUCT Resample multiple fields of a data struct
%
%   [t_out, data_out] = resampleStruct(data, targetFs, {'acc', 'gyr', 'mag'})
%
%   This is a convenience wrapper for resampling common IMU data structures.

    if nargin < 3 || isempty(fields)
        fields = {'acc', 'gyr', 'mag'};
    end
    
    % Build input cell array
    signals = {};
    validFields = {};
    for i = 1:length(fields)
        if isfield(data, fields{i}) && ~isempty(data.(fields{i}))
            signals{end+1} = data.(fields{i}); %#ok<AGROW>
            validFields{end+1} = fields{i}; %#ok<AGROW>
        end
    end
    
    % Resample
    outputs = cell(1, length(signals));
    [t_out, outputs{:}] = resample_signals(data.t, targetFs, signals{:}, varargin{:});
    
    % Build output struct
    data_out = struct();
    data_out.t = t_out;
    data_out.dt = 1 / targetFs;
    data_out.Fs = targetFs;
    
    for i = 1:length(validFields)
        data_out.(validFields{i}) = outputs{i};
    end
    
    % Copy metadata if present
    if isfield(data, 'meta')
        data_out.meta = data.meta;
        data_out.meta.resampled = true;
        data_out.meta.originalFs = 1 / mean(diff(data.t));
    end
end

function quality = assessSamplingQuality(t)
%ASSESSSAMPLINGQUALITY Compute metrics about sampling regularity
%
%   quality = assessSamplingQuality(t)
%
%   Returns struct with:
%       .Fs          - Estimated sampling rate
%       .dtMean      - Mean sample period
%       .dtStd       - Sample period standard deviation
%       .dtCV        - Coefficient of variation (%)
%       .numGaps     - Number of significant gaps
%       .maxGap      - Maximum gap duration
%       .minGap      - Minimum sample period
%       .isRegular   - True if CV < 5%

    dt = diff(t(:));
    
    quality = struct();
    quality.dtMean = mean(dt);
    quality.dtStd = std(dt);
    quality.dtCV = 100 * quality.dtStd / quality.dtMean;
    quality.Fs = 1 / quality.dtMean;
    quality.maxGap = max(dt);
    quality.minGap = min(dt);
    quality.numGaps = sum(dt > 3 * quality.dtMean);
    quality.isRegular = quality.dtCV < 5;
    quality.N = length(t);
    quality.duration = t(end) - t(1);
end

function printSamplingReport(t)
%PRINTSAMPLINGREPORT Print detailed sampling analysis

    q = assessSamplingQuality(t);
    
    fprintf('\n=== Sampling Quality Report ===\n');
    fprintf('  Samples:     %d\n', q.N);
    fprintf('  Duration:    %.2f s\n', q.duration);
    fprintf('  Est. Fs:     %.2f Hz\n', q.Fs);
    fprintf('  Mean dt:     %.4f s\n', q.dtMean);
    fprintf('  Std dt:      %.4f s\n', q.dtStd);
    fprintf('  CV:          %.2f%%\n', q.dtCV);
    fprintf('  Max gap:     %.4f s (%.1fx mean)\n', q.maxGap, q.maxGap/q.dtMean);
    fprintf('  Min period:  %.4f s (%.1fx mean)\n', q.minGap, q.minGap/q.dtMean);
    fprintf('  Num gaps:    %d (>3x mean)\n', q.numGaps);
    fprintf('  Regular:     %s\n', mat2str(q.isRegular));
    fprintf('================================\n\n');
end
