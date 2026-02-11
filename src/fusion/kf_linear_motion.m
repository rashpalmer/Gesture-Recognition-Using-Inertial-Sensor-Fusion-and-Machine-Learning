function motion = kf_linear_motion(imu, est, params)
%KF_LINEAR_MOTION Linear Kalman Filter for velocity and position estimation
%   motion = kf_linear_motion(imu, est, params) estimates velocity and
%   position by integrating world-frame acceleration with drift correction
%   using zero-velocity updates (ZUPT) during detected stationary periods.
%
%   INPUTS:
%       imu     - Preprocessed IMU data from preprocess_imu()
%       est     - Attitude estimation from ekf_attitude_quat()
%       params  - Configuration from config_params() (optional)
%
%   OUTPUT:
%       motion - Motion estimation struct:
%           .v          - Nx3 velocity estimate (m/s) in world frame
%           .p          - Nx3 position estimate (m) in world frame
%           .a_world    - Nx3 linear acceleration in world frame (m/s²)
%           .P          - 6x6xN covariance history
%           .Ptrace     - Nx6 diagonal covariance traces
%           .zupt_flag  - Nx1 logical (true = ZUPT applied)
%
%   IMPROVEMENTS (v2.0):
%       - FIXED: rms() calls with dimension argument
%       - ADDED: Input validation
%       - ADDED: Covariance symmetry enforcement
%       - OPTIMIZED: Batch quaternion rotation for a_world
%       - IMPROVED: Better ZUPT detection logic
%
%   STATE VECTOR (6 states):
%       x = [v_x, v_y, v_z, p_x, p_y, p_z]'
%
%   PROCESS MODEL (constant velocity with acceleration input):
%       v_{k+1} = v_k + a_world * dt
%       p_{k+1} = p_k + v_k * dt + 0.5 * a_world * dt²
%
%   MEASUREMENT (Zero-Velocity Update):
%       z = v = [0; 0; 0] when stationary detected
%
%   WHY LINEAR KF (NOT EKF)?
%       After rotating acceleration to world frame and removing gravity,
%       the dynamics are LINEAR in velocity and position:
%           x_{k+1} = F*x_k + B*u_k
%       This is a textbook linear state-space model, so standard KF
%       provides optimal estimation without linearization.
%
%   EXAMPLE:
%       imu = preprocess_imu(data, params);
%       est = ekf_attitude_quat(imu, params);
%       motion = kf_linear_motion(imu, est, params);
%       plot3(motion.p(:,1), motion.p(:,2), motion.p(:,3));
%
%   Author: Sensor Fusion Demo (Improved)
%   Date: 2026

    %% Default parameters
    if nargin < 3
        params = config_params();
    end

    fprintf('Running linear KF for motion estimation...\n');

    %% Input Validation
    if ~isstruct(imu) || ~isfield(imu, 'acc') || ~isfield(imu, 't')
        error('kf_linear_motion:InvalidInput', 'imu must be a struct with acc and t fields');
    end

    if ~isstruct(est) || ~isfield(est, 'q')
        error('kf_linear_motion:InvalidInput', 'est must be a struct with q field');
    end

    %% Initialize
    qu = quat_utils();

    n = length(imu.t);
    dt_vec = [imu.dt; imu.dt(end)];

    % State dimension
    n_states = 6;  % 3 velocity + 3 position

    % Allocate outputs
    motion.v = zeros(n, 3);
    motion.p = zeros(n, 3);
    motion.a_world = zeros(n, 3);
    motion.P = zeros(n_states, n_states, n);
    motion.Ptrace = zeros(n, n_states);
    motion.zupt_flag = false(n, 1);

    %% Transform acceleration to world frame and remove gravity
    % OPTIMIZED: Use batch rotation if available
    g_world = params.frames.gravity_world;

    if isfield(qu, 'rotateBatch') && n > 100
        % Batch rotation for efficiency
        a_body = imu.acc';  % 3xN
        motion.a_world = (qu.rotateBatch(est.q, a_body) - g_world)';
    else
        % Standard loop
        for k = 1:n
            % Get rotation matrix from estimated quaternion
            q = est.q(k, :)';
            R = qu.toRotMat(q);

            % Transform acceleration to world frame
            a_body = imu.acc(k, :)';
            a_world = R * a_body;

            % Remove gravity to get linear acceleration
            motion.a_world(k, :) = (a_world - g_world)';
        end
    end

    %% Initial state and covariance
    % Start at rest at origin
    x = zeros(n_states, 1);  % [v_x, v_y, v_z, p_x, p_y, p_z]

    P = diag([params.kf.P0_velocity * ones(3,1);
              params.kf.P0_position * ones(3,1)]);

    %% Process and measurement noise
    % Process noise Q
    Q_v = params.kf.Q_velocity;
    Q_p = params.kf.Q_position;

    % Measurement noise (ZUPT)
    R_zupt = params.kf.R_zupt * eye(3);

    %% ZUPT detection parameters
    zupt_gyro_th = params.kf.zupt_gyro_threshold;
    zupt_acc_th = params.kf.zupt_acc_threshold;
    zupt_window = params.kf.zupt_window;

    %% Main KF Loop
    for k = 1:n
        %% Store current estimates
        motion.v(k, :) = x(1:3)';
        motion.p(k, :) = x(4:6)';
        motion.P(:,:,k) = P;
        motion.Ptrace(k, :) = diag(P)';

        if k == n
            break;
        end

        %% Get measurements and dt
        dt = dt_vec(k);
        a = motion.a_world(k, :)';

        %% === PREDICTION STEP ===

        % State transition matrix F
        % v_{k+1} = v_k + a*dt
        % p_{k+1} = p_k + v_k*dt + 0.5*a*dt²
        F = [eye(3),    zeros(3);
             dt*eye(3), eye(3)];

        % Control input matrix B
        % Effect of acceleration on state
        B = [dt*eye(3);
             0.5*dt^2*eye(3)];

        % Predicted state
        x_pred = F * x + B * a;

        % Process noise covariance (discrete-time)
        Q = diag([Q_v*dt^2, Q_v*dt^2, Q_v*dt^2, ...
                  Q_p*dt^4, Q_p*dt^4, Q_p*dt^4]);

        % Predicted covariance
        P_pred = F * P * F' + Q;

        %% === ZUPT DETECTION ===

        % Check for stationary condition
        is_stationary = detect_zupt(imu, k, zupt_gyro_th, zupt_acc_th, zupt_window);

        %% === MEASUREMENT UPDATE (ZUPT) ===

        if is_stationary
            % Zero-velocity measurement
            z = [0; 0; 0];

            % Measurement matrix (observe velocity only)
            H = [eye(3), zeros(3)];

            % Innovation
            y = z - H * x_pred;

            % Innovation covariance
            S = H * P_pred * H' + R_zupt;

            % Kalman gain (use right division for numerical stability)
            K = (P_pred * H') / S;

            % State update
            x = x_pred + K * y;

            % Covariance update (Joseph form for numerical stability)
            I_KH = eye(n_states) - K * H;
            P = I_KH * P_pred * I_KH' + K * R_zupt * K';

            % FIXED: Enforce covariance symmetry
            P = (P + P') / 2;

            motion.zupt_flag(k) = true;
        else
            % No measurement update
            x = x_pred;
            P = P_pred;
        end

    end

    %% Summary
    fprintf('Linear KF complete:\n');
    fprintf('  ZUPT events: %d (%.1f%%)\n', ...
        sum(motion.zupt_flag), 100*sum(motion.zupt_flag)/n);
    fprintf('  Final velocity: [%.3f, %.3f, %.3f] m/s\n', motion.v(end,:));
    fprintf('  Total displacement: %.3f m\n', norm(motion.p(end,:)));

    % FIXED: rms() with dimension argument
    fprintf('  Velocity RMS: [%.3f, %.3f, %.3f] m/s\n', ...
        rms(motion.v(:,1), 1), rms(motion.v(:,2), 1), rms(motion.v(:,3), 1));

    % Store time reference
    motion.t = imu.t;

end

%% ==================== HELPER FUNCTIONS ====================

function is_stationary = detect_zupt(imu, k, gyro_th, acc_th, window)
%DETECT_ZUPT Detect zero-velocity condition
%   IMPROVED: Better handling of edge cases and multiple detection criteria

    n = length(imu.t);

    % Window around current sample
    k_start = max(1, k - floor(window/2));
    k_end = min(n, k + floor(window/2));

    % Gyroscope check: low angular rate
    gyr_window = imu.gyr(k_start:k_end, :);
    gyr_mag = sqrt(sum(gyr_window.^2, 2));
    gyro_condition = mean(gyr_mag) < gyro_th;

    % Accelerometer check: variance close to zero (after gravity removal)
    % Note: We use raw acc variance since a_world includes noise
    acc_window = imu.acc(k_start:k_end, :);
    acc_var = var(sqrt(sum(acc_window.^2, 2)));
    acc_condition = acc_var < acc_th;

    % Both conditions must be met
    is_stationary = gyro_condition && acc_condition;

    % Also check pre-computed stationary flag (from preprocessing)
    if isfield(imu, 'flags') && isfield(imu.flags, 'stationary')
        % Use OR - either detection method triggers ZUPT
        is_stationary = is_stationary || imu.flags.stationary(k);
    end
end
