%% TEST_FEATURE_EXTRACTION - Unit tests for feature extraction
% Run with: runtests('test_feature_extraction')
%
% Tests cover:
%   - Feature naming conventions (both old and new)
%   - Feature vector dimensions
%   - Edge cases (empty data, single sample)
%   - Numeric stability
%
% Author: Sensor Fusion Demo (Test Suite)
% Date: 2026

classdef test_feature_extraction < matlab.unittest.TestCase

    properties
        params  % Configuration parameters
        tol     % Numerical tolerance
    end

    methods(TestMethodSetup)
        function setup(testCase)
            testCase.params = config_params();
            testCase.tol = 1e-6;
        end
    end

    %% ==================== FEATURE NAMING TESTS ====================

    methods(Test)
        function test_feature_names_exist(testCase)
            % Create minimal test data
            [imu, est, seg] = create_test_data(100, testCase.params);

            feat = extract_features(imu, est, seg, testCase.params);

            % Check that feature names exist
            testCase.verifyTrue(isfield(feat, 'names'));
            testCase.verifyTrue(~isempty(feat.names));
        end

        function test_feature_naming_conventions(testCase)
            % Both naming conventions should be supported
            [imu, est, seg] = create_test_data(100, testCase.params);

            feat = extract_features(imu, est, seg, testCase.params);

            % Check for new naming convention
            testCase.verifyTrue(any(strcmp(feat.names, 'peak_gyr_x')));
            testCase.verifyTrue(any(strcmp(feat.names, 'peak_gyr_y')));
            testCase.verifyTrue(any(strcmp(feat.names, 'peak_gyr_z')));

            % Check for backward compatibility aliases
            testCase.verifyTrue(any(strcmp(feat.names, 'gyr_peak_x')));
            testCase.verifyTrue(any(strcmp(feat.names, 'gyr_peak_y')));
            testCase.verifyTrue(any(strcmp(feat.names, 'gyr_peak_z')));
        end

        function test_rotation_units(testCase)
            % Both rotation units should be available
            [imu, est, seg] = create_test_data(100, testCase.params);

            feat = extract_features(imu, est, seg, testCase.params);

            % Check for both unit types
            testCase.verifyTrue(any(strcmp(feat.names, 'total_rotation')));
            testCase.verifyTrue(any(strcmp(feat.names, 'total_rotation_deg')));

            % Check that deg version is ~57x radians version
            idx_rad = strcmp(feat.names, 'total_rotation');
            idx_deg = strcmp(feat.names, 'total_rotation_deg');

            rotation_rad = feat.x(idx_rad);
            rotation_deg = feat.x(idx_deg);

            expected_ratio = 180/pi;
            actual_ratio = rotation_deg / rotation_rad;

            testCase.verifyEqual(actual_ratio, expected_ratio, 'RelTol', 0.01);
        end
    end

    %% ==================== FEATURE VECTOR TESTS ====================

    methods(Test)
        function test_feature_vector_dimensions(testCase)
            % Feature vector should match names length
            [imu, est, seg] = create_test_data(100, testCase.params);

            feat = extract_features(imu, est, seg, testCase.params);

            testCase.verifyEqual(length(feat.x), length(feat.names));
        end

        function test_feature_values_finite(testCase)
            % All features should be finite (no NaN/Inf)
            [imu, est, seg] = create_test_data(100, testCase.params);

            feat = extract_features(imu, est, seg, testCase.params);

            testCase.verifyTrue(all(isfinite(feat.x)));
        end

        function test_feature_values_struct(testCase)
            % Feature values struct should exist and have entries
            [imu, est, seg] = create_test_data(100, testCase.params);

            feat = extract_features(imu, est, seg, testCase.params);

            testCase.verifyTrue(isfield(feat, 'values'));
            testCase.verifyTrue(isstruct(feat.values));
        end
    end

    %% ==================== SPECIFIC FEATURE TESTS ====================

    methods(Test)
        function test_duration_feature(testCase)
            % Duration should match segment length
            [imu, est, seg] = create_test_data(100, testCase.params);

            feat = extract_features(imu, est, seg, testCase.params);

            idx = strcmp(feat.names, 'duration');
            duration = feat.x(idx);

            expected = (seg.winIdx(2) - seg.winIdx(1) + 1) / imu.Fs;
            testCase.verifyEqual(duration, expected, 'RelTol', 0.01);
        end

        function test_rms_features_positive(testCase)
            % RMS features should be non-negative
            [imu, est, seg] = create_test_data(100, testCase.params);

            feat = extract_features(imu, est, seg, testCase.params);

            rms_indices = contains(feat.names, 'rms');
            rms_values = feat.x(rms_indices);

            testCase.verifyTrue(all(rms_values >= 0));
        end

        function test_dominant_axis_valid(testCase)
            % Dominant axis should be X, Y, or Z
            [imu, est, seg] = create_test_data(100, testCase.params);

            feat = extract_features(imu, est, seg, testCase.params);

            testCase.verifyTrue(isfield(feat.values, 'dominant_axis'));
            dom_axis = feat.values.dominant_axis;
            testCase.verifyTrue(ismember(dom_axis, {'X', 'Y', 'Z'}));
        end
    end

    %% ==================== EDGE CASE TESTS ====================

    methods(Test)
        function test_short_segment(testCase)
            % Test with minimal segment length
            [imu, est, seg] = create_test_data(20, testCase.params);

            % Should not error
            feat = extract_features(imu, est, seg, testCase.params);
            testCase.verifyTrue(~isempty(feat.x));
        end

        function test_static_data(testCase)
            % Test with nearly zero motion
            [imu, est, seg] = create_test_data(100, testCase.params);

            % Set gyro to near-zero
            imu.gyr = imu.gyr * 0.001;

            feat = extract_features(imu, est, seg, testCase.params);

            % RMS should be very small
            idx = strcmp(feat.names, 'gyr_rms_total');
            testCase.verifyLessThan(feat.x(idx), 0.01);
        end
    end

end

%% ==================== HELPER FUNCTIONS ====================

function [imu, est, seg] = create_test_data(n, params)
%CREATE_TEST_DATA Create synthetic test data

    Fs = 100;  % Sample rate
    t = (0:n-1)' / Fs;

    % Create sinusoidal motion
    freq = 2;  % Hz
    amp = 1;   % rad/s

    gyr = amp * [sin(2*pi*freq*t), cos(2*pi*freq*t), 0.5*sin(2*pi*freq*t)];
    acc = [zeros(n,2), ones(n,1)*9.81] + 0.1*randn(n,3);

    imu.t = t;
    imu.dt = diff(t);
    imu.Fs = Fs;
    imu.gyr = gyr;
    imu.acc = acc;
    imu.mag = nan(n, 3);
    imu.flags.stationary = false(n, 1);

    % Simple attitude estimate (identity quaternion)
    est.q = repmat([1 0 0 0], n, 1);
    est.euler = zeros(n, 3);
    est.t = t;

    % Segmentation covering whole signal
    seg.winIdx = [1, n];
    seg.windows = [1, n];
    seg.primary = 1;
    seg.score = 1.0;
    seg.energy = ones(n, 1);
end
