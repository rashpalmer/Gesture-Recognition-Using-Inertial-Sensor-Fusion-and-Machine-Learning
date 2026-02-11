function assert_utils()
%ASSERT_UTILS Assertion and validation utilities for gesture recognition pipeline
%
%   This file contains helper functions for validating data structures,
%   checking numerical properties, and providing informative error messages.
%
%   Functions:
%       assertStruct(s, requiredFields, name)     - Validate struct has required fields
%       assertSize(x, expectedSize, name)         - Validate array dimensions
%       assertNumeric(x, name)                    - Validate numeric type
%       assertFinite(x, name)                     - Check for NaN/Inf
%       assertRange(x, minVal, maxVal, name)      - Validate value range
%       assertQuaternion(q, name)                 - Validate quaternion format
%       assertTimeSeries(t, name)                 - Validate time vector
%       assertSampleRate(Fs, name)                - Validate sampling rate
%       assertDataStruct(data)                    - Validate raw data struct
%       assertImuStruct(imu)                      - Validate preprocessed IMU struct
%       assertEstStruct(est)                      - Validate fusion estimate struct
%       warnIf(condition, msgId, msg)             - Conditional warning
%       softAssert(condition, msg)                - Non-fatal assertion with warning
%
%   Usage:
%       % Import specific functions
%       au = assert_utils();
%       au.assertStruct(data, {'t', 'acc', 'gyr'}, 'data');
%       au.assertQuaternion(q, 'attitude');
%
%   Author: Claude (Anthropic)
%   Date: January 2025

    % Return struct of function handles
    assert_utils = struct();
    assert_utils.assertStruct = @assertStruct;
    assert_utils.assertSize = @assertSize;
    assert_utils.assertNumeric = @assertNumeric;
    assert_utils.assertFinite = @assertFinite;
    assert_utils.assertRange = @assertRange;
    assert_utils.assertQuaternion = @assertQuaternion;
    assert_utils.assertTimeSeries = @assertTimeSeries;
    assert_utils.assertSampleRate = @assertSampleRate;
    assert_utils.assertDataStruct = @assertDataStruct;
    assert_utils.assertImuStruct = @assertImuStruct;
    assert_utils.assertEstStruct = @assertEstStruct;
    assert_utils.warnIf = @warnIf;
    assert_utils.softAssert = @softAssert;
    assert_utils.validateParams = @validateParams;
    assert_utils.checkMemory = @checkMemory;
    
    assignin('caller', 'ans', assert_utils);
end

%% ======================== STRUCT VALIDATION ========================

function assertStruct(s, requiredFields, name)
%ASSERTSTRUCT Validate that a struct contains required fields
%
%   Inputs:
%       s              - Struct to validate
%       requiredFields - Cell array of required field names
%       name           - Name of struct for error messages

    if nargin < 3, name = 'struct'; end
    
    if ~isstruct(s)
        error('assert_utils:notStruct', ...
            '%s must be a struct, got %s', name, class(s));
    end
    
    if ischar(requiredFields)
        requiredFields = {requiredFields};
    end
    
    missingFields = {};
    for i = 1:length(requiredFields)
        if ~isfield(s, requiredFields{i})
            missingFields{end+1} = requiredFields{i}; %#ok<AGROW>
        end
    end
    
    if ~isempty(missingFields)
        error('assert_utils:missingFields', ...
            '%s is missing required fields: %s', ...
            name, strjoin(missingFields, ', '));
    end
end

%% ======================== ARRAY VALIDATION ========================

function assertSize(x, expectedSize, name)
%ASSERTSIZE Validate array dimensions
%
%   Inputs:
%       x            - Array to validate
%       expectedSize - Expected size vector (use NaN for "any" dimension)
%       name         - Variable name for error messages
%
%   Example:
%       assertSize(acc, [NaN, 3], 'accelerometer')  % Nx3 array

    if nargin < 3, name = 'array'; end
    
    actualSize = size(x);
    
    % Pad sizes to match lengths
    if length(actualSize) < length(expectedSize)
        actualSize = [actualSize, ones(1, length(expectedSize) - length(actualSize))];
    elseif length(expectedSize) < length(actualSize)
        expectedSize = [expectedSize, ones(1, length(actualSize) - length(expectedSize))];
    end
    
    % Check each dimension (NaN means "any size is OK")
    for i = 1:length(expectedSize)
        if ~isnan(expectedSize(i)) && actualSize(i) ~= expectedSize(i)
            error('assert_utils:sizeMismatch', ...
                '%s has size [%s], expected [%s]', ...
                name, num2str(actualSize), num2str(expectedSize));
        end
    end
end

function assertNumeric(x, name)
%ASSERTNUMERIC Validate that input is numeric

    if nargin < 2, name = 'value'; end
    
    if ~isnumeric(x)
        error('assert_utils:notNumeric', ...
            '%s must be numeric, got %s', name, class(x));
    end
end

function assertFinite(x, name)
%ASSERTFINITE Check for NaN or Inf values

    if nargin < 2, name = 'array'; end
    
    assertNumeric(x, name);
    
    numNaN = sum(isnan(x(:)));
    numInf = sum(isinf(x(:)));
    
    if numNaN > 0 || numInf > 0
        error('assert_utils:nonFinite', ...
            '%s contains %d NaN and %d Inf values', name, numNaN, numInf);
    end
end

function assertRange(x, minVal, maxVal, name)
%ASSERTRANGE Validate values are within expected range

    if nargin < 4, name = 'value'; end
    
    assertNumeric(x, name);
    
    actualMin = min(x(:));
    actualMax = max(x(:));
    
    if actualMin < minVal || actualMax > maxVal
        error('assert_utils:outOfRange', ...
            '%s has range [%.4g, %.4g], expected [%.4g, %.4g]', ...
            name, actualMin, actualMax, minVal, maxVal);
    end
end

%% ======================== SPECIALIZED VALIDATION ========================

function assertQuaternion(q, name)
%ASSERTQUATERNION Validate quaternion format and normalization
%
%   Expects quaternions as Nx4 array in [w, x, y, z] format

    if nargin < 2, name = 'quaternion'; end
    
    assertNumeric(q, name);
    
    % Check dimensions
    if size(q, 2) ~= 4
        error('assert_utils:quaternionFormat', ...
            '%s must be Nx4, got Nx%d', name, size(q, 2));
    end
    
    % Check normalization
    norms = sqrt(sum(q.^2, 2));
    normError = max(abs(norms - 1));
    
    if normError > 0.01
        error('assert_utils:quaternionNorm', ...
            '%s not normalized (max error = %.4f)', name, normError);
    elseif normError > 0.001
        warning('assert_utils:quaternionNormWarn', ...
            '%s has slight normalization error (max = %.6f)', name, normError);
    end
    
    assertFinite(q, name);
end

function assertTimeSeries(t, name)
%ASSERTTIMESERIES Validate time vector properties

    if nargin < 2, name = 'time'; end
    
    assertNumeric(t, name);
    assertFinite(t, name);
    
    % Check monotonicity
    dt = diff(t);
    if any(dt <= 0)
        numNonMono = sum(dt <= 0);
        error('assert_utils:timeNotMonotonic', ...
            '%s must be strictly increasing (%d violations)', name, numNonMono);
    end
    
    % Warn about irregular sampling
    dtMean = mean(dt);
    dtStd = std(dt);
    if dtStd / dtMean > 0.1
        warning('assert_utils:irregularSampling', ...
            '%s has irregular sampling (CV = %.2f%%)', name, 100*dtStd/dtMean);
    end
end

function assertSampleRate(Fs, name)
%ASSERTSAMPLERATE Validate sampling rate is reasonable

    if nargin < 2, name = 'Fs'; end
    
    assertNumeric(Fs, name);
    
    if ~isscalar(Fs)
        error('assert_utils:notScalar', '%s must be scalar', name);
    end
    
    if Fs <= 0
        error('assert_utils:invalidFs', '%s must be positive', name);
    end
    
    if Fs < 10
        warning('assert_utils:lowFs', '%s = %.1f Hz is very low for IMU', name, Fs);
    elseif Fs > 1000
        warning('assert_utils:highFs', '%s = %.1f Hz is unusually high', name, Fs);
    end
end

%% ======================== PIPELINE STRUCT VALIDATION ========================

function assertDataStruct(data)
%ASSERTDATASTRUCT Validate raw data structure from read_phone_data

    assertStruct(data, {'t', 'acc', 'gyr'}, 'data');
    
    N = length(data.t);
    
    assertTimeSeries(data.t, 'data.t');
    assertSize(data.acc, [N, 3], 'data.acc');
    assertSize(data.gyr, [N, 3], 'data.gyr');
    assertFinite(data.acc, 'data.acc');
    assertFinite(data.gyr, 'data.gyr');
    
    % Magnetometer is optional
    if isfield(data, 'mag') && ~isempty(data.mag)
        assertSize(data.mag, [N, 3], 'data.mag');
        assertFinite(data.mag, 'data.mag');
    end
    
    % Sanity checks on values
    accMag = sqrt(sum(data.acc.^2, 2));
    if mean(accMag) < 5 || mean(accMag) > 15
        warning('assert_utils:accMagnitude', ...
            'Accelerometer magnitude (mean=%.2f) unusual, check units (expected m/s²)', ...
            mean(accMag));
    end
    
    gyrMag = sqrt(sum(data.gyr.^2, 2));
    if max(gyrMag) > 50
        warning('assert_utils:gyrMagnitude', ...
            'Gyroscope magnitude (max=%.2f) very high, check units (expected rad/s)', ...
            max(gyrMag));
    end
end

function assertImuStruct(imu)
%ASSERTIMUSTRUCT Validate preprocessed IMU structure

    assertStruct(imu, {'t', 'acc', 'gyr', 'Fs'}, 'imu');
    
    N = length(imu.t);
    
    assertTimeSeries(imu.t, 'imu.t');
    assertSampleRate(imu.Fs, 'imu.Fs');
    assertSize(imu.acc, [N, 3], 'imu.acc');
    assertSize(imu.gyr, [N, 3], 'imu.gyr');
    assertFinite(imu.acc, 'imu.acc');
    assertFinite(imu.gyr, 'imu.gyr');
    
    % Check dt consistency with Fs
    if isfield(imu, 'dt')
        expectedDt = 1 / imu.Fs;
        if abs(imu.dt - expectedDt) > 0.001
            warning('assert_utils:dtFsMismatch', ...
                'imu.dt (%.4f) inconsistent with imu.Fs (%.1f Hz)', imu.dt, imu.Fs);
        end
    end
    
    % Optional fields
    if isfield(imu, 'mag') && ~isempty(imu.mag)
        assertSize(imu.mag, [N, 3], 'imu.mag');
    end
    
    if isfield(imu, 'flags') && isfield(imu.flags, 'stationary')
        assertSize(imu.flags.stationary, [N, 1], 'imu.flags.stationary');
    end
end

function assertEstStruct(est)
%ASSERTESTSTRUCT Validate fusion estimate structure

    assertStruct(est, {'q'}, 'est');
    
    N = size(est.q, 1);
    
    assertQuaternion(est.q, 'est.q');
    
    % Optional fields
    if isfield(est, 'euler')
        assertSize(est.euler, [N, 3], 'est.euler');
        assertFinite(est.euler, 'est.euler');
    end
    
    if isfield(est, 'b_g')
        assertSize(est.b_g, [N, 3], 'est.b_g');
        assertFinite(est.b_g, 'est.b_g');
        
        % Check bias magnitude is reasonable
        biasMag = sqrt(sum(est.b_g.^2, 2));
        if max(biasMag) > 0.5
            warning('assert_utils:highGyroBias', ...
                'Gyro bias estimate (max=%.3f rad/s) seems high', max(biasMag));
        end
    end
end

%% ======================== SOFT ASSERTIONS & WARNINGS ========================

function warnIf(condition, msgId, msg)
%WARNIF Issue warning only if condition is true

    if condition
        warning(msgId, msg);
    end
end

function ok = softAssert(condition, msg)
%SOFTASSERT Non-fatal assertion that issues warning instead of error
%
%   Returns true if condition passed, false otherwise

    ok = condition;
    if ~condition
        warning('assert_utils:softAssertFailed', 'Soft assertion failed: %s', msg);
    end
end

%% ======================== PARAMETER VALIDATION ========================

function validateParams(params)
%VALIDATEPARAMS Validate configuration parameters structure

    % Check main sections exist
    requiredSections = {'sampling', 'preprocess', 'ekf', 'kf', 'segmentation', 'gestures'};
    assertStruct(params, requiredSections, 'params');
    
    % Sampling
    assertSampleRate(params.sampling.targetFs, 'params.sampling.targetFs');
    
    % EKF parameters
    if isfield(params.ekf, 'Q') && ~isempty(params.ekf.Q)
        Q = params.ekf.Q;
        if ~all(diag(Q) >= 0)
            error('assert_utils:invalidQ', 'EKF Q matrix must have non-negative diagonal');
        end
    end
    
    if isfield(params.ekf, 'R_acc') && ~isempty(params.ekf.R_acc)
        R = params.ekf.R_acc;
        if ~all(diag(R) > 0)
            error('assert_utils:invalidR', 'EKF R_acc matrix must have positive diagonal');
        end
    end
    
    % Segmentation thresholds
    if params.segmentation.threshold_low >= params.segmentation.threshold_high
        error('assert_utils:invalidThresholds', ...
            'segmentation.threshold_low must be < threshold_high');
    end
    
    fprintf('  [assert_utils] Parameters validated successfully\n');
end

%% ======================== MEMORY CHECK ========================

function checkMemory(N, numChannels, bytesPerElement)
%CHECKMEMORY Warn if processing may require excessive memory
%
%   Inputs:
%       N               - Number of samples
%       numChannels     - Number of data channels
%       bytesPerElement - Bytes per element (default: 8 for double)

    if nargin < 3, bytesPerElement = 8; end
    if nargin < 2, numChannels = 10; end
    
    estimatedMB = (N * numChannels * bytesPerElement) / 1e6;
    
    if estimatedMB > 1000
        warning('assert_utils:highMemory', ...
            'Processing %d samples may require ~%.0f MB memory', N, estimatedMB);
    elseif estimatedMB > 100
        fprintf('  [assert_utils] Estimated memory usage: %.1f MB\n', estimatedMB);
    end
end
