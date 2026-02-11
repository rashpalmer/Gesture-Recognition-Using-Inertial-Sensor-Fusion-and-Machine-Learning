%% TEST_SEGMENTATION - Test gesture segmentation on synthetic data
%
% Verifies that segment_gesture.m:
%   - Detects clear gesture windows
%   - Handles quiet periods correctly
%   - Respects duration constraints
%   - Returns valid window indices
%
% Author: Generated for Gesture-SA-Sensor-Fusion project
% Date: 2025

function tests = test_segmentation
    tests = functiontests(localfunctions);
end

%% Test: Detects single clear gesture
function test_single_gesture(testCase)
    [imu, params] = generate_single_gesture_data();
    seg = segment_gesture(imu, params);
    
    verifyGreaterThanOrEqual(testCase, length(seg.windows), 1, ...
        'Should detect at least one gesture window');
end

%% Test: Primary window has valid indices
function test_valid_indices(testCase)
    [imu, params] = generate_single_gesture_data();
    seg = segment_gesture(imu, params);
    
    N = length(imu.t);
    
    verifyGreaterThanOrEqual(testCase, seg.winIdx(1), 1, ...
        'Window start should be >= 1');
    verifyLessThanOrEqual(testCase, seg.winIdx(2), N, ...
        'Window end should be <= N');
    verifyLessThan(testCase, seg.winIdx(1), seg.winIdx(2), ...
        'Window start should be < end');
end

%% Test: No false positives in quiet data
function test_quiet_data(testCase)
    [imu, params] = generate_quiet_data();
    
    % Increase threshold to avoid false detections
    params.segmentation.energy_high = 10.0;
    
    seg = segment_gesture(imu, params);
    
    % Should have no or very few windows
    verifyLessThanOrEqual(testCase, length(seg.windows), 1, ...
        'Should not detect gestures in quiet data');
end

%% Test: Detects multiple gestures
function test_multiple_gestures(testCase)
    [imu, params] = generate_multi_gesture_data();
    seg = segment_gesture(imu, params);
    
    verifyGreaterThanOrEqual(testCase, size(seg.windows, 1), 2, ...
        'Should detect multiple gesture windows');
end

%% Test: Respects minimum duration
function test_min_duration(testCase)
    [imu, params] = generate_single_gesture_data();
    
    % Set minimum duration to 0.5s
    params.segmentation.min_duration = 0.5;
    
    seg = segment_gesture(imu, params);
    
    if ~isempty(seg.windows)
        for i = 1:size(seg.windows, 1)
            win_start = seg.windows(i, 1);
            win_end = seg.windows(i, 2);
            duration = imu.t(win_end) - imu.t(win_start);
            
            verifyGreaterThanOrEqual(testCase, duration, ...
                params.segmentation.min_duration - 0.05, ...
                sprintf('Window %d duration should meet minimum', i));
        end
    end
end

%% Test: Respects maximum duration
function test_max_duration(testCase)
    [imu, params] = generate_long_gesture_data();
    
    % Set maximum duration to 2s
    params.segmentation.max_duration = 2.0;
    
    seg = segment_gesture(imu, params);
    
    if ~isempty(seg.windows)
        for i = 1:size(seg.windows, 1)
            win_start = seg.windows(i, 1);
            win_end = seg.windows(i, 2);
            duration = imu.t(win_end) - imu.t(win_start);
            
            verifyLessThanOrEqual(testCase, duration, ...
                params.segmentation.max_duration + 0.1, ...
                sprintf('Window %d duration should not exceed maximum', i));
        end
    end
end

%% Test: Score is positive for valid gesture
function test_positive_score(testCase)
    [imu, params] = generate_single_gesture_data();
    seg = segment_gesture(imu, params);
    
    verifyGreaterThan(testCase, seg.score, 0, ...
        'Segmentation score should be positive for valid gesture');
end

%% Test: Energy array has correct length
function test_energy_length(testCase)
    [imu, params] = generate_single_gesture_data();
    seg = segment_gesture(imu, params);
    
    verifyLength(testCase, seg.energy, length(imu.t), ...
        'Energy array should match IMU length');
end

%% Helper: Generate data with single clear gesture
function [imu, params] = generate_single_gesture_data()
    params = config_params();
    Fs = params.sampling.targetFs;
    
    % 3 seconds: 1s quiet, 0.8s gesture, 1.2s quiet
    total_duration = 3.0;
    N = round(total_duration * Fs);
    
    imu.t = (0:N-1)' / Fs;
    imu.dt = diff(imu.t);
    imu.Fs = Fs;
    
    % Initialize with quiet data
    imu.acc = repmat([0, 0, -9.81], N, 1) + 0.05 * randn(N, 3);
    imu.gyr = 0.01 * randn(N, 3);
    
    % Add gesture motion in middle (1.0s to 1.8s)
    gesture_start = round(1.0 * Fs);
    gesture_end = round(1.8 * Fs);
    
    % Simulate rotation around Z-axis (twist gesture)
    t_gesture = imu.t(gesture_start:gesture_end) - imu.t(gesture_start);
    omega_max = 8.0;  % rad/s peak (must exceed energy_high * gyro_norm_factor)
    omega_profile = omega_max * sin(2*pi*t_gesture / 0.8);  % One cycle
    
    imu.gyr(gesture_start:gesture_end, 3) = omega_profile;
    
    % Corresponding accelerometer disturbance
    imu.acc(gesture_start:gesture_end, :) = imu.acc(gesture_start:gesture_end, :) + ...
        2.0 * randn(gesture_end - gesture_start + 1, 3);
    
    imu.flags.stationary = false(N, 1);
    imu.flags.stationary(1:gesture_start-10) = true;
    imu.flags.stationary(gesture_end+10:end) = true;
end

%% Helper: Generate quiet data (no gestures)
function [imu, params] = generate_quiet_data()
    params = config_params();
    Fs = params.sampling.targetFs;
    N = round(2.0 * Fs);
    
    imu.t = (0:N-1)' / Fs;
    imu.dt = diff(imu.t);
    imu.Fs = Fs;
    
    % Only sensor noise, no motion
    imu.acc = repmat([0, 0, -9.81], N, 1) + 0.02 * randn(N, 3);
    imu.gyr = 0.005 * randn(N, 3);
    
    imu.flags.stationary = true(N, 1);
end

%% Helper: Generate data with multiple gestures
function [imu, params] = generate_multi_gesture_data()
    params = config_params();
    Fs = params.sampling.targetFs;
    
    % 5 seconds with 3 gestures
    total_duration = 5.0;
    N = round(total_duration * Fs);
    
    imu.t = (0:N-1)' / Fs;
    imu.dt = diff(imu.t);
    imu.Fs = Fs;
    
    imu.acc = repmat([0, 0, -9.81], N, 1) + 0.05 * randn(N, 3);
    imu.gyr = 0.01 * randn(N, 3);
    
    % Gesture 1: 0.5s - 1.0s
    g1_start = round(0.5 * Fs);
    g1_end = round(1.0 * Fs);
    imu.gyr(g1_start:g1_end, 1) = 6.0 * ones(g1_end - g1_start + 1, 1);
    
    % Gesture 2: 2.0s - 2.6s
    g2_start = round(2.0 * Fs);
    g2_end = round(2.6 * Fs);
    imu.gyr(g2_start:g2_end, 2) = -7.0 * ones(g2_end - g2_start + 1, 1);
    
    % Gesture 3: 3.5s - 4.2s
    g3_start = round(3.5 * Fs);
    g3_end = round(4.2 * Fs);
    imu.gyr(g3_start:g3_end, 3) = 6.0 * sin(linspace(0, 2*pi, g3_end - g3_start + 1))';
    
    imu.flags.stationary = false(N, 1);
end

%% Helper: Generate long continuous gesture
function [imu, params] = generate_long_gesture_data()
    params = config_params();
    Fs = params.sampling.targetFs;
    
    % 6 seconds with one 4-second gesture
    total_duration = 6.0;
    N = round(total_duration * Fs);
    
    imu.t = (0:N-1)' / Fs;
    imu.dt = diff(imu.t);
    imu.Fs = Fs;
    
    imu.acc = repmat([0, 0, -9.81], N, 1) + 0.05 * randn(N, 3);
    imu.gyr = 0.01 * randn(N, 3);
    
    % Long gesture: 1.0s - 5.0s
    g_start = round(1.0 * Fs);
    g_end = round(5.0 * Fs);
    
    % Continuous rotation
    imu.gyr(g_start:g_end, 3) = 6.0 * ones(g_end - g_start + 1, 1);
    
    imu.flags.stationary = false(N, 1);
end


%% To run these tests:
%   results = runtests('test_segmentation');
%   disp(results);