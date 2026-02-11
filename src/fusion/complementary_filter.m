function est = complementary_filter(imu, params)
%COMPLEMENTARY_FILTER Simple complementary filter for attitude estimation
%   est = complementary_filter(imu, params) estimates device orientation
%   using a complementary filter that fuses gyroscope and accelerometer.
%
%   This provides a simpler baseline for comparison with the EKF approach.
%
%   INPUTS:
%       imu     - Preprocessed IMU data from preprocess_imu()
%       params  - Configuration from config_params() (optional)
%
%   OUTPUT:
%       est - Estimation results struct:
%           .q          - Nx4 quaternion trajectory [w,x,y,z]
%           .euler      - Nx3 Euler angles [roll,pitch,yaw] (rad)
%           .euler_acc  - Nx3 Euler from accelerometer only
%           .euler_gyr  - Nx3 Euler from gyro integration only
%
%   IMPROVEMENTS (v2.0):
%       - ADDED: Input validation
%       - ADDED: Adaptive alpha based on motion detection
%       - OPTIMIZED: Vectorized quaternion operations where possible
%       - IMPROVED: Better handling of singularities
%
%   COMPLEMENTARY FILTER CONCEPT:
%       q_fused = alpha * q_gyro + (1-alpha) * q_accel
%
%       - alpha (typically 0.95-0.99): weight on gyroscope
%       - Gyro: good short-term, drifts long-term
%       - Accel: noisy short-term, stable long-term (gravity reference)
%
%   The filter acts as:
%       - High-pass filter on gyroscope (fast dynamics)
%       - Low-pass filter on accelerometer (slow/steady reference)
%
%   EXAMPLE:
%       imu = preprocess_imu(data, params);
%       est_cf = complementary_filter(imu, params);
%       est_ekf = ekf_attitude_quat(imu, params);
%
%       subplot(2,1,1); plot(imu.t, est_cf.euler*180/pi);
%       title('Complementary Filter');
%       subplot(2,1,2); plot(imu.t, est_ekf.euler*180/pi);
%       title('EKF');
%
%   Author: Sensor Fusion Demo (Improved)
%   Date: 2026

    %% Default parameters
    if nargin < 2
        params = config_params();
    end

    fprintf('Running complementary filter...\n');

    %% Input Validation
    if ~isstruct(imu) || ~isfield(imu, 'acc') || ~isfield(imu, 'gyr')
        error('complementary_filter:InvalidInput', 'imu must be a struct with acc and gyr fields');
    end

    %% Initialize
    qu = quat_utils();

    n = length(imu.t);
    dt_vec = [imu.dt; imu.dt(end)];

    % Filter coefficient
    alpha = params.comp.alpha;  % Gyro weight (typically 0.98)

    % Adaptive alpha parameters (NEW)
    use_adaptive = isfield(params.comp, 'adaptive') && params.comp.adaptive;
    if use_adaptive
        alpha_min = 0.90;
        alpha_max = 0.995;
        motion_threshold = 0.5;  % rad/s
    end

    % Allocate outputs
    est.q = zeros(n, 4);
    est.euler = zeros(n, 3);
    est.euler_acc = zeros(n, 3);
    est.euler_gyr = zeros(n, 3);

    %% Initialize orientation from first accelerometer reading
    acc_init = imu.acc(1, :)';
    [roll_init, pitch_init] = accel_to_tilt(acc_init);
    yaw_init = 0;  % Cannot determine from accelerometer alone

    q = qu.fromEuler(roll_init, pitch_init, yaw_init);
    q_gyro = q;  % Gyro-only tracking

    est.q(1, :) = q';
    est.euler(1, :) = [roll_init, pitch_init, yaw_init];
    est.euler_acc(1, :) = [roll_init, pitch_init, 0];
    est.euler_gyr(1, :) = [roll_init, pitch_init, yaw_init];

    %% Main loop
    for k = 2:n
        dt = dt_vec(k-1);

        %% Gyroscope integration
        omega = imu.gyr(k, :)';

        % Create rotation quaternion from angular velocity
        q_delta = qu.fromOmega(omega, dt);

        % Propagate quaternion
        q_gyro_new = qu.multiply(q, q_delta);
        q_gyro_new = qu.normalize(q_gyro_new);

        % Also track pure gyro for comparison
        q_gyro = qu.multiply(q_gyro, q_delta);
        q_gyro = qu.normalize(q_gyro);

        %% Accelerometer tilt estimation
        acc = imu.acc(k, :)';
        [roll_acc, pitch_acc] = accel_to_tilt(acc);

        % Store accelerometer-only estimate
        est.euler_acc(k, :) = [roll_acc, pitch_acc, 0];

        %% Adaptive alpha (NEW)
        if use_adaptive
            omega_mag = norm(omega);
            if omega_mag > motion_threshold
                % High motion: trust gyro more
                alpha_k = alpha_max;
            else
                % Low motion: trust accelerometer more
                alpha_k = alpha_min + (alpha_max - alpha_min) * (omega_mag / motion_threshold);
            end
        else
            alpha_k = alpha;
        end

        %% Complementary fusion

        % Get Euler angles from gyro prediction
        [roll_gyr, pitch_gyr, yaw_gyr] = qu.toEuler(q_gyro_new);

        % Store gyro-only estimate
        [roll_gyr_only, pitch_gyr_only, yaw_gyr_only] = qu.toEuler(q_gyro);
        est.euler_gyr(k, :) = [roll_gyr_only, pitch_gyr_only, yaw_gyr_only];

        % Complementary filter for roll and pitch
        roll_fused = alpha_k * roll_gyr + (1 - alpha_k) * roll_acc;
        pitch_fused = alpha_k * pitch_gyr + (1 - alpha_k) * pitch_acc;

        % Yaw comes only from gyro (no accelerometer reference)
        yaw_fused = yaw_gyr;

        % Optional: magnetometer correction for yaw
        if isfield(params.ekf, 'use_mag_update') && params.ekf.use_mag_update && ...
           isfield(imu, 'mag') && ~all(isnan(imu.mag(k,:)))
            mag = imu.mag(k, :)';
            yaw_mag = mag_to_yaw(mag, roll_fused, pitch_fused);

            % Blend magnetometer yaw
            beta = params.comp.beta;  % Magnetometer weight
            yaw_fused = alpha_k * yaw_gyr + (1 - alpha_k) * beta * yaw_mag + ...
                        (1 - alpha_k) * (1 - beta) * yaw_gyr;
        end

        % Convert back to quaternion
        q = qu.fromEuler(roll_fused, pitch_fused, yaw_fused);
        q = qu.normalize(q);  % Ensure unit quaternion

        % Store results
        est.q(k, :) = q';
        est.euler(k, :) = [roll_fused, pitch_fused, yaw_fused];
    end

    % Store time reference
    est.t = imu.t;

    %% Summary
    fprintf('Complementary filter complete:\n');
    fprintf('  Alpha (gyro weight): %.3f\n', alpha);
    if use_adaptive
        fprintf('  Adaptive alpha: enabled\n');
    end
    fprintf('  Final orientation: [%.1f, %.1f, %.1f] deg\n', ...
        est.euler(end,:) * 180/pi);

end

%% ==================== HELPER FUNCTIONS ====================

function [roll, pitch] = accel_to_tilt(acc)
%ACCEL_TO_TILT Estimate roll and pitch from accelerometer
%   Assumes accelerometer measures gravity when stationary

    EPS = 1e-10;

    % Normalize (handles different gravity magnitudes)
    a_norm = norm(acc);
    if a_norm < EPS
        roll = 0;
        pitch = 0;
        return;
    end
    a = acc / a_norm;

    % Roll (rotation about X axis)
    roll = atan2(-a(2), -a(3));

    % Pitch (rotation about Y axis)
    % Clamp to prevent NaN from sqrt of negative
    sqrt_arg = max(0, a(2)^2 + a(3)^2);
    pitch = atan2(a(1), sqrt(sqrt_arg));
end

function yaw = mag_to_yaw(mag, roll, pitch)
%MAG_TO_YAW Estimate yaw from magnetometer with tilt compensation

    EPS = 1e-10;

    % Tilt-compensated magnetic field components
    cos_roll = cos(roll);
    sin_roll = sin(roll);
    cos_pitch = cos(pitch);
    sin_pitch = sin(pitch);

    mx = mag(1);
    my = mag(2);
    mz = mag(3);

    % Rotate magnetic field to horizontal plane
    mx_h = mx * cos_pitch + my * sin_roll * sin_pitch + mz * cos_roll * sin_pitch;
    my_h = my * cos_roll - mz * sin_roll;

    % Handle degenerate case
    if abs(mx_h) < EPS && abs(my_h) < EPS
        yaw = 0;
        return;
    end

    % Yaw angle
    yaw = atan2(-my_h, mx_h);
end
