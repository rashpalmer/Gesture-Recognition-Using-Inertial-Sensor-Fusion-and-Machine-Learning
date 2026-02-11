function data = generate_synthetic_gesture(gestureType, varargin)
%GENERATE_SYNTHETIC_GESTURE Generate synthetic IMU data for testing
%
%   Creates realistic synthetic accelerometer, gyroscope, and magnetometer
%   data for common gestures. Useful for testing the pipeline without
%   real sensor data.
%
%   Syntax:
%       data = generate_synthetic_gesture(gestureType)
%       data = generate_synthetic_gesture(gestureType, 'Name', Value)
%
%   Inputs:
%       gestureType - Gesture to generate:
%                     'flip_up'      - Phone flip towards user
%                     'flip_down'    - Phone flip away from user
%                     'shake'        - Side-to-side shake
%                     'twist'        - Rotation around long axis
%                     'push_forward' - Forward push motion
%                     'circle'       - Circular motion
%                     'static'       - Stationary (for calibration testing)
%                     'random_walk'  - Random drift (for testing)
%
%   Name-Value Parameters:
%       'Duration'     - Gesture duration in seconds (default: 1.0)
%       'Fs'           - Sampling frequency in Hz (default: 100)
%       'NoiseLevel'   - Noise multiplier 0-1 (default: 0.1)
%       'BiasGyro'     - Gyroscope bias [x,y,z] rad/s (default: [0.01, -0.02, 0.015])
%       'BiasAcc'      - Accelerometer bias [x,y,z] m/s² (default: [0.05, -0.03, 0.08])
%       'MagField'     - Magnetic field vector µT (default: [20, 5, 45])
%       'MagHardIron'  - Hard iron offset µT (default: [10, -5, 8])
%       'Intensity'    - Motion intensity multiplier (default: 1.0)
%       'PrePadding'   - Seconds of static before gesture (default: 0.5)
%       'PostPadding'  - Seconds of static after gesture (default: 0.5)
%       'Seed'         - Random seed for reproducibility (default: [])
%
%   Output:
%       data - Struct with fields:
%              .t     - Time vector (Nx1)
%              .acc   - Accelerometer data (Nx3, m/s²)
%              .gyr   - Gyroscope data (Nx3, rad/s)
%              .mag   - Magnetometer data (Nx3, µT)
%              .meta  - Generation metadata
%              .ground_truth - True orientation/motion (for validation)
%
%   Example:
%       % Generate a flip_up gesture
%       data = generate_synthetic_gesture('flip_up');
%
%       % Generate with custom parameters
%       data = generate_synthetic_gesture('shake', ...
%                   'Duration', 2.0, 'Intensity', 1.5, 'Seed', 42);
%
%       % Static data for EKF initialization testing
%       data = generate_synthetic_gesture('static', 'Duration', 5);
%
%   See also: read_phone_data, test_ekf_static
%
%   Author: Claude (Anthropic)
%   Date: January 2025

    %% Parse inputs
    p = inputParser;
    addRequired(p, 'gestureType', @(x) ismember(lower(x), ...
        {'flip_up', 'flip_down', 'shake', 'twist', 'push_forward', 'circle', 'static', 'random_walk'}));
    addParameter(p, 'Duration', 1.0, @isnumeric);
    addParameter(p, 'Fs', 100, @isnumeric);
    addParameter(p, 'NoiseLevel', 0.1, @isnumeric);
    addParameter(p, 'BiasGyro', [0.01, -0.02, 0.015], @isnumeric);
    addParameter(p, 'BiasAcc', [0.05, -0.03, 0.08], @isnumeric);
    addParameter(p, 'MagField', [20, 5, 45], @isnumeric);      % World frame magnetic field (µT)
    addParameter(p, 'MagHardIron', [10, -5, 8], @isnumeric);   % Hard iron offset
    addParameter(p, 'Intensity', 1.0, @isnumeric);
    addParameter(p, 'PrePadding', 0.5, @isnumeric);
    addParameter(p, 'PostPadding', 0.5, @isnumeric);
    addParameter(p, 'Seed', [], @isnumeric);
    
    parse(p, gestureType, varargin{:});
    opts = p.Results;
    opts.gestureType = lower(opts.gestureType);
    
    % Set random seed if specified
    if ~isempty(opts.Seed)
        rng(opts.Seed);
    end
    
    %% Generate time vector
    totalDuration = opts.PrePadding + opts.Duration + opts.PostPadding;
    dt = 1 / opts.Fs;
    t = (0 : dt : totalDuration - dt)';
    N = length(t);
    
    % Define gesture region
    gestureStart = opts.PrePadding;
    gestureEnd = opts.PrePadding + opts.Duration;
    
    %% Initialize arrays
    gyr_clean = zeros(N, 3);  % Angular velocity (rad/s)
    q = zeros(N, 4);          % Quaternion history
    q(1, :) = [1, 0, 0, 0];   % Start at identity
    
    %% Generate gesture-specific angular velocity profile
    switch opts.gestureType
        case 'flip_up'
            % Rotate about X axis (pitch forward towards user)
            gyr_clean = generateFlipProfile(t, gestureStart, gestureEnd, ...
                [1, 0, 0], pi/2 * opts.Intensity);
            
        case 'flip_down'
            % Rotate about X axis (pitch backward)
            gyr_clean = generateFlipProfile(t, gestureStart, gestureEnd, ...
                [-1, 0, 0], pi/2 * opts.Intensity);
            
        case 'shake'
            % Oscillating rotation about Z axis
            gyr_clean = generateShakeProfile(t, gestureStart, gestureEnd, ...
                [0, 0, 1], 4, pi/6 * opts.Intensity);
            
        case 'twist'
            % Rotate about Y axis (phone's long axis)
            gyr_clean = generateFlipProfile(t, gestureStart, gestureEnd, ...
                [0, 1, 0], pi * opts.Intensity);
            
        case 'push_forward'
            % Small rotation (pitch) during push
            gyr_clean = generatePushProfile(t, gestureStart, gestureEnd, ...
                [1, 0, 0], pi/8 * opts.Intensity);
            
        case 'circle'
            % Circular motion - rotating angular velocity
            gyr_clean = generateCircleProfile(t, gestureStart, gestureEnd, ...
                2, pi/4 * opts.Intensity);
            
        case 'static'
            % No motion (already zeros)
            
        case 'random_walk'
            % Random angular velocity (for testing drift)
            gyr_clean = generateRandomWalk(t, 0.1 * opts.Intensity);
    end
    
    %% Integrate angular velocity to get orientation quaternions
    for i = 2:N
        omega = gyr_clean(i-1, :)';
        q(i, :) = integrateQuaternion(q(i-1, :)', omega, dt)';
    end
    
    %% Generate accelerometer data
    % Gravity in world frame (ENU: Z up)
    g_world = [0; 0; -9.81];
    
    acc_clean = zeros(N, 3);
    for i = 1:N
        % Rotate gravity into body frame
        R = quat2rotm(q(i, :));
        acc_clean(i, :) = (R' * g_world)';  % Gravity as measured in body frame
    end
    
    % Add linear acceleration for push gesture
    if strcmp(opts.gestureType, 'push_forward')
        acc_linear = generateLinearAccProfile(t, gestureStart, gestureEnd, ...
            [0, 1, 0], 5 * opts.Intensity);  % Forward is +Y
        acc_clean = acc_clean + acc_linear;
    end
    
    %% Generate magnetometer data
    mag_world = opts.MagField(:);
    mag_clean = zeros(N, 3);
    for i = 1:N
        R = quat2rotm(q(i, :));
        mag_clean(i, :) = (R' * mag_world)';
    end
    
    %% Add noise and biases
    % Gyroscope noise (typically 0.01-0.05 rad/s RMS for smartphone)
    gyr_noise_std = 0.03 * opts.NoiseLevel;
    gyr = gyr_clean + opts.BiasGyro + gyr_noise_std * randn(N, 3);
    
    % Accelerometer noise (typically 0.05-0.2 m/s² RMS)
    acc_noise_std = 0.1 * opts.NoiseLevel;
    acc = acc_clean + opts.BiasAcc + acc_noise_std * randn(N, 3);
    
    % Magnetometer noise (typically 1-5 µT RMS)
    mag_noise_std = 2 * opts.NoiseLevel;
    mag = mag_clean + opts.MagHardIron + mag_noise_std * randn(N, 3);
    
    %% Package output
    data = struct();
    data.t = t;
    data.acc = acc;
    data.gyr = gyr;
    data.mag = mag;
    
    % Metadata
    data.meta = struct();
    data.meta.gesture = opts.gestureType;
    data.meta.duration = totalDuration;
    data.meta.gestureDuration = opts.Duration;
    data.meta.Fs = opts.Fs;
    data.meta.N = N;
    data.meta.synthetic = true;
    data.meta.noiseLevel = opts.NoiseLevel;
    data.meta.intensity = opts.Intensity;
    data.meta.gestureWindow = [round(gestureStart * opts.Fs) + 1, round(gestureEnd * opts.Fs)];
    
    % Ground truth for validation
    data.ground_truth = struct();
    data.ground_truth.q = q;
    data.ground_truth.euler = quat2euler(q);
    data.ground_truth.gyr_clean = gyr_clean;
    data.ground_truth.acc_clean = acc_clean;
    data.ground_truth.mag_clean = mag_clean;
    data.ground_truth.biasGyro = opts.BiasGyro;
    data.ground_truth.biasAcc = opts.BiasAcc;
    data.ground_truth.magHardIron = opts.MagHardIron;
    
    fprintf('  [synthetic] Generated %s gesture: %d samples, %.1f s\n', ...
        opts.gestureType, N, totalDuration);
end

%% ======================== PROFILE GENERATORS ========================

function gyr = generateFlipProfile(t, tStart, tEnd, axis, totalAngle)
%GENERATEFLIPPROFILE Smooth flip motion (single rotation)
%
%   Uses raised cosine profile for smooth acceleration/deceleration

    N = length(t);
    gyr = zeros(N, 3);
    axis = axis(:)' / norm(axis);  % Normalize axis
    
    duration = tEnd - tStart;
    
    for i = 1:N
        if t(i) >= tStart && t(i) <= tEnd
            % Normalized time within gesture [0, 1]
            tau = (t(i) - tStart) / duration;
            
            % Raised cosine angular velocity profile
            % Integral of (1 - cos(2*pi*tau)) from 0 to 1 equals 1
            omega_mag = (totalAngle / duration) * (1 - cos(2 * pi * tau));
            
            gyr(i, :) = omega_mag * axis;
        end
    end
end

function gyr = generateShakeProfile(t, tStart, tEnd, axis, numCycles, amplitude)
%GENERATESHAKEPROFILE Oscillating back-and-forth motion

    N = length(t);
    gyr = zeros(N, 3);
    axis = axis(:)' / norm(axis);
    
    duration = tEnd - tStart;
    freq = numCycles / duration;  % Hz
    
    for i = 1:N
        if t(i) >= tStart && t(i) <= tEnd
            tau = t(i) - tStart;
            
            % Envelope (ramp up and down)
            env = sin(pi * tau / duration);
            
            % Oscillating angular velocity
            omega_mag = amplitude * 2 * pi * freq * cos(2 * pi * freq * tau) * env;
            
            gyr(i, :) = omega_mag * axis;
        end
    end
end

function gyr = generatePushProfile(t, tStart, tEnd, axis, totalAngle)
%GENERATEPUSHPROFILE Small tilt during push motion

    N = length(t);
    gyr = zeros(N, 3);
    axis = axis(:)' / norm(axis);
    
    duration = tEnd - tStart;
    
    for i = 1:N
        if t(i) >= tStart && t(i) <= tEnd
            tau = (t(i) - tStart) / duration;
            
            % Tilt forward then back: two half-cosines
            if tau < 0.5
                % Tilt forward
                omega_mag = (2 * totalAngle / duration) * (1 - cos(4 * pi * tau));
            else
                % Tilt back
                omega_mag = -(2 * totalAngle / duration) * (1 - cos(4 * pi * tau));
            end
            
            gyr(i, :) = omega_mag * axis;
        end
    end
end

function gyr = generateCircleProfile(t, tStart, tEnd, numCircles, tiltAngle)
%GENERATECIRCLEPROFILE Circular motion (cone tracing)

    N = length(t);
    gyr = zeros(N, 3);
    
    duration = tEnd - tStart;
    freq = numCircles / duration;
    
    for i = 1:N
        if t(i) >= tStart && t(i) <= tEnd
            tau = t(i) - tStart;
            phase = 2 * pi * freq * tau;
            
            % Envelope
            env = sin(pi * tau / duration);
            
            % Angular velocity rotates in XY plane
            omega_x = tiltAngle * 2 * pi * freq * cos(phase) * env;
            omega_y = tiltAngle * 2 * pi * freq * sin(phase) * env;
            
            gyr(i, :) = [omega_x, omega_y, 0];
        end
    end
end

function gyr = generateRandomWalk(t, intensity)
%GENERATERANDOMWALK Random drift for testing

    N = length(t);
    dt = t(2) - t(1);
    
    % Random walk: integrate white noise
    noise = intensity * randn(N, 3);
    gyr = cumsum(noise) * sqrt(dt);
    
    % Smooth it a bit
    windowSize = max(1, round(0.1 / dt));
    for ax = 1:3
        gyr(:, ax) = movmean(gyr(:, ax), windowSize);
    end
end

function acc = generateLinearAccProfile(t, tStart, tEnd, direction, peakAcc)
%GENERATELINEARACCPROFILE Linear acceleration pulse

    N = length(t);
    acc = zeros(N, 3);
    direction = direction(:)' / norm(direction);
    
    duration = tEnd - tStart;
    
    for i = 1:N
        if t(i) >= tStart && t(i) <= tEnd
            tau = (t(i) - tStart) / duration;
            
            % Sinusoidal pulse (accelerate then decelerate)
            acc_mag = peakAcc * sin(pi * tau);
            
            acc(i, :) = acc_mag * direction;
        end
    end
end

%% ======================== QUATERNION UTILITIES ========================

function q_new = integrateQuaternion(q, omega, dt)
%INTEGRATEQUATERNION Integrate angular velocity to update quaternion

    omega_mag = norm(omega);
    
    if omega_mag < 1e-10
        q_new = q;
        return;
    end
    
    % Quaternion from angular velocity (small angle approximation or exact)
    axis = omega / omega_mag;
    angle = omega_mag * dt;
    
    q_delta = [cos(angle/2); axis * sin(angle/2)];
    
    % Quaternion multiplication: q_new = q * q_delta
    q_new = quatMultiply(q, q_delta);
    
    % Normalize
    q_new = q_new / norm(q_new);
end

function q_out = quatMultiply(q1, q2)
%QUATMULTIPLY Hamilton quaternion multiplication

    w1 = q1(1); x1 = q1(2); y1 = q1(3); z1 = q1(4);
    w2 = q2(1); x2 = q2(2); y2 = q2(3); z2 = q2(4);
    
    q_out = [
        w1*w2 - x1*x2 - y1*y2 - z1*z2;
        w1*x2 + x1*w2 + y1*z2 - z1*y2;
        w1*y2 - x1*z2 + y1*w2 + z1*x2;
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ];
end

function R = quat2rotm(q)
%QUAT2ROTM Convert quaternion to rotation matrix

    q = q(:)' / norm(q);  % Ensure normalized row vector
    
    w = q(1); x = q(2); y = q(3); z = q(4);
    
    R = [
        1-2*(y^2+z^2),  2*(x*y-w*z),    2*(x*z+w*y);
        2*(x*y+w*z),    1-2*(x^2+z^2),  2*(y*z-w*x);
        2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x^2+y^2)
    ];
end

function euler = quat2euler(q)
%QUAT2EULER Convert quaternion array to Euler angles (ZYX convention)
%
%   Output: [roll, pitch, yaw] in radians

    N = size(q, 1);
    euler = zeros(N, 3);
    
    for i = 1:N
        w = q(i,1); x = q(i,2); y = q(i,3); z = q(i,4);
        
        % Roll (X)
        sinr_cosp = 2 * (w*x + y*z);
        cosr_cosp = 1 - 2 * (x^2 + y^2);
        euler(i, 1) = atan2(sinr_cosp, cosr_cosp);
        
        % Pitch (Y)
        sinp = 2 * (w*y - z*x);
        if abs(sinp) >= 1
            euler(i, 2) = sign(sinp) * pi/2;  % Gimbal lock
        else
            euler(i, 2) = asin(sinp);
        end
        
        % Yaw (Z)
        siny_cosp = 2 * (w*z + x*y);
        cosy_cosp = 1 - 2 * (y^2 + z^2);
        euler(i, 3) = atan2(siny_cosp, cosy_cosp);
    end
end
