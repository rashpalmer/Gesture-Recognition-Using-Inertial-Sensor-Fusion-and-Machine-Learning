%% TEST_EKF_STATIC - Test EKF attitude estimation on static/synthetic data
%
% Verifies that the EKF:
%   - Converges to correct orientation from poor initial guess
%   - Maintains bounded drift during static conditions
%   - Properly estimates gyroscope bias
%   - Quaternion stays normalized
%
% Author: Generated for Gesture-SA-Sensor-Fusion project
% Date: 2025

function tests = test_ekf_static
    tests = functiontests(localfunctions);
end

%% Test: Quaternion normalization maintained
function test_quat_norm_maintained(testCase)
    [imu, params] = generate_static_imu(5.0);  % 5 seconds
    est = ekf_attitude_quat(imu, params);
    
    % Check quaternion norm at all time steps
    norms = sqrt(sum(est.q.^2, 2));
    
    verifyEqual(testCase, norms, ones(size(norms)), 'AbsTol', 0.001, ...
        'Quaternion norm should stay within [0.999, 1.001]');
end

%% Test: Gravity alignment from poor initial guess
function test_gravity_alignment(testCase)
    [imu, params] = generate_static_imu(3.0);
    
    % Give EKF a bad initial guess (30° tilted)
    qu = quat_utils();
    bad_init = qu.fromEuler(deg2rad(30), deg2rad(20), 0);
    params.fusion.ekf.init_q = bad_init;
    
    est = ekf_attitude_quat(imu, params);
    
    % After convergence, should align with gravity
    % True orientation is identity (phone flat)
    final_euler = est.euler(end, :);  % [roll, pitch, yaw]
    
    % Roll and pitch should converge to ~0
    verifyEqual(testCase, final_euler(1), 0, 'AbsTol', deg2rad(2), ...
        'Roll should converge to 0° (within 2°)');
    verifyEqual(testCase, final_euler(2), 0, 'AbsTol', deg2rad(2), ...
        'Pitch should converge to 0° (within 2°)');
end

%% Test: Gyro bias estimation
function test_gyro_bias_estimation(testCase)
    [imu, params] = generate_static_imu(10.0);  % 10 seconds for bias to settle
    
    % Add known gyro bias to measurements
    true_bias = [0.02, -0.01, 0.015];  % rad/s
    imu.gyr = imu.gyr + true_bias;
    
    est = ekf_attitude_quat(imu, params);
    
    % Check final bias estimate
    final_bias = est.b_g(end, :);
    
    % X/Y bias observable via gravity → tight tolerance
    verifyEqual(testCase, final_bias(1:2), true_bias(1:2), 'AbsTol', 0.005, ...
        'Roll/pitch gyro bias should converge (gravity-observable)');
    
    % Z bias only observable via magnetometer → looser tolerance
    verifyEqual(testCase, final_bias(3), true_bias(3), 'AbsTol', 0.02, ...
        'Yaw gyro bias convergence (magnetometer-limited)');
end

%% Test: Bounded orientation drift during static
function test_bounded_drift(testCase)
    [imu, params] = generate_static_imu(60.0);  % 1 minute
    est = ekf_attitude_quat(imu, params);
    
    % Check yaw drift (most susceptible without magnetometer updates)
    yaw = est.euler(:, 3);
    yaw_drift = abs(yaw(end) - yaw(1));
    
    % Should be less than 5° per minute in static conditions
    verifyLessThan(testCase, rad2deg(yaw_drift), 5.0, ...
        'Yaw drift should be < 5° per minute in static conditions');
end

%% Test: Covariance remains bounded
function test_covariance_bounded(testCase)
    [imu, params] = generate_static_imu(10.0);
    est = ekf_attitude_quat(imu, params);
    
    % P trace should not explode
    P_trace = est.Ptrace;
    
    verifyLessThan(testCase, max(P_trace), 1.0, ...
        'Covariance trace should remain bounded');
    verifyGreaterThan(testCase, min(P_trace), 0, ...
        'Covariance should remain positive');
end

%% Test: Handles accelerometer noise gracefully
function test_accel_noise_rejection(testCase)
    [imu, params] = generate_static_imu(5.0);
    
    % Add significant accelerometer noise
    noise_std = 0.5;  % m/s²
    imu.acc = imu.acc + noise_std * randn(size(imu.acc));
    
    est = ekf_attitude_quat(imu, params);
    
    % Should still converge to reasonable orientation
    final_euler = est.euler(end, :);
    
    verifyEqual(testCase, final_euler(1), 0, 'AbsTol', deg2rad(5), ...
        'Roll should be within 5° despite noise');
    verifyEqual(testCase, final_euler(2), 0, 'AbsTol', deg2rad(5), ...
        'Pitch should be within 5° despite noise');
end

%% Test: Innovation sequence is white
function test_innovation_whiteness(testCase)
    [imu, params] = generate_static_imu(10.0);
    est = ekf_attitude_quat(imu, params);
    
    if isfield(est, 'innov_acc') && ~isempty(est.innov_acc)
        % Check that innovations are approximately zero-mean
        innov_mean = mean(est.innov_acc, 1, 'omitnan');
        
        verifyEqual(testCase, innov_mean, zeros(1, 3), 'AbsTol', 0.1, ...
            'Accelerometer innovations should be zero-mean');
    end
end

%% Test: Tilted static orientation
function test_tilted_static(testCase)
    % Generate static data for phone tilted 45° forward
    true_roll = 0;
    true_pitch = deg2rad(45);
    true_yaw = 0;
    
    [imu, params] = generate_tilted_static_imu(5.0, true_roll, true_pitch, true_yaw);
    est = ekf_attitude_quat(imu, params);
    
    final_euler = est.euler(end, :);
    
    verifyEqual(testCase, final_euler(1), true_roll, 'AbsTol', deg2rad(3), ...
        'Roll should match true orientation');
    verifyEqual(testCase, final_euler(2), true_pitch, 'AbsTol', deg2rad(3), ...
        'Pitch should match true orientation');
end

%% Helper: Generate static IMU data (phone flat, stationary)
function [imu, params] = generate_static_imu(duration_sec)
    params = config_params();
    Fs = params.sampling.targetFs;
    N = round(duration_sec * Fs);
    
    % Time vector
    imu.t = (0:N-1)' / Fs;
    imu.dt = diff(imu.t);
    imu.Fs = Fs;
    
    % Static accelerometer: gravity pointing down (-Z in phone frame)
    g = norm(params.frames.gravity_world);  % 9.81 m/s²
    imu.acc = repmat([0, 0, -g], N, 1);
    
    % Add small sensor noise
    acc_noise = 0.05;  % m/s² std
    imu.acc = imu.acc + acc_noise * randn(N, 3);
    
    % Static gyroscope: zero (no rotation)
    gyr_noise = 0.001;  % rad/s std
    imu.gyr = gyr_noise * randn(N, 3);
    
    % Magnetometer: pointing North (X direction in world = East, Y = North in ENU)
    % For phone flat with screen up, mag should point mostly in +Y
    mag_strength = 45;  % µT typical
    imu.mag = repmat([0, mag_strength, 0], N, 1);
    mag_noise = 1.0;  % µT std
    imu.mag = imu.mag + mag_noise * randn(N, 3);
    
    % Static flags (all samples are static)
    imu.flags.stationary = true(N, 1);
    
    % Calibration (no bias)
    imu.calib.gyro_bias = [0, 0, 0];
    imu.calib.acc_scale = 1;
    imu.calib.mag_offset = [0, 0, 0];
end

%% Helper: Generate tilted static IMU data
function [imu, params] = generate_tilted_static_imu(duration_sec, roll, pitch, yaw)
    params = config_params();
    qu = quat_utils();
    Fs = params.sampling.targetFs;
    N = round(duration_sec * Fs);
    
    imu.t = (0:N-1)' / Fs;
    imu.dt = diff(imu.t);
    imu.Fs = Fs;
    
    % Rotation from world to body
    q = qu.fromEuler(roll, pitch, yaw);
    q_inv = qu.inverse(q);
    
    % Gravity in world frame: [0, 0, -g] (down)
    g = norm(params.frames.gravity_world);
    g_world = [0; 0; -g];
    
    % Rotate gravity to body frame (returns 3x1, transpose to row)
    g_body = qu.rotate(q_inv, g_world)';
    
    imu.acc = repmat(g_body, N, 1);
    imu.acc = imu.acc + 0.05 * randn(N, 3);
    
    imu.gyr = 0.001 * randn(N, 3);
    
    % Magnetometer in body frame
    mag_world = [0; 45; 0];  % North in ENU
    mag_body = qu.rotate(q_inv, mag_world)';
    imu.mag = repmat(mag_body, N, 1);
    imu.mag = imu.mag + 1.0 * randn(N, 3);
    
    imu.flags.stationary = true(N, 1);
    imu.calib.gyro_bias = [0, 0, 0];
    imu.calib.acc_scale = 1;
    imu.calib.mag_offset = [0, 0, 0];
end


%% To run these tests:
%   results = runtests('test_ekf_static');
%   disp(results);