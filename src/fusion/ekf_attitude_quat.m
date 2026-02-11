function est = ekf_attitude_quat(imu, params)
%EKF_ATTITUDE_QUAT Extended Kalman Filter for quaternion attitude estimation
%   est = ekf_attitude_quat(imu, params) estimates device orientation using
%   an EKF that fuses gyroscope, accelerometer, and magnetometer data.
%
%   INPUTS:
%       imu     - Preprocessed IMU data from preprocess_imu()
%       params  - Configuration from config_params() (optional)
%
%   OUTPUT:
%       est - Estimation results struct:
%           .q          - Nx4 quaternion trajectory [w,x,y,z]
%           .b_g        - Nx3 estimated gyro bias (rad/s)
%           .euler      - Nx3 Euler angles [roll,pitch,yaw] (rad) for plotting
%           .P          - 7x7xN covariance history (if store_full_covariance)
%           .Ptrace     - Nx7 diagonal covariance traces
%           .innov_acc  - Nx3 accelerometer innovations
%           .innov_mag  - Nx3 magnetometer innovations
%           .S_acc      - Nx3 innovation covariances (acc)
%           .params     - Parameters used
%
%   IMPROVEMENTS (v2.0):
%       - FIXED: Quaternion normalized BEFORE storing (not after)
%       - FIXED: Covariance symmetry enforced after update
%       - FIXED: Quaternion continuity check using dot product
%       - ADDED: Input validation with clear error messages
%       - ADDED: Memory optimization option (store only diagonal)
%       - OPTIMIZED: Cached rotation matrices
%       - OPTIMIZED: Pre-computed constant matrix structures
%       - ADDED: Better numerical stability with EPS constant
%
%   Author: Sensor Fusion Demo (Improved)
%   Date: 2026

    %% Input validation
    if nargin < 2 || isempty(params)
        params = config_params();
    end

    % Validate IMU struct
    validateattributes(imu, {'struct'}, {'nonempty'}, mfilename, 'imu', 1);
    assert(isfield(imu, 'acc'), 'IMU struct must contain acc field');
    assert(isfield(imu, 'gyr'), 'IMU struct must contain gyr field');
    assert(isfield(imu, 't'), 'IMU struct must contain t field');
    assert(all(size(imu.acc, 2) == 3), 'Accelerometer must be Nx3');
    assert(size(imu.gyr, 2) == 3, 'Gyroscope must be Nx3');

    if params.verbose
        fprintf('Running EKF attitude estimation...\n');
    end

    %% Initialize
    qu = quat_utils();  % Quaternion utilities
    EPS = params.EPS;   % Numerical stability constant

    n = length(imu.t);
    dt_vec = [imu.dt; imu.dt(end)];  % Extend to n samples

    % State dimension
    n_states = 7;  % 4 quaternion + 3 gyro bias

    % Memory optimization option
    store_full_P = params.ekf.store_full_covariance;

    % Allocate output arrays
    est.q = zeros(n, 4);
    est.b_g = zeros(n, 3);
    if store_full_P
        est.P = zeros(n_states, n_states, n);
    end
    est.Ptrace = zeros(n, n_states);
    est.innov_acc = zeros(n, 3);
    est.innov_mag = zeros(n, 3);
    est.S_acc = zeros(n, 3);

    %% Initialize state and covariance
    q0 = initialize_orientation(imu, params);

    % Initial gyro bias
    b_g0 = params.ekf.init_gyro_bias;
    if isfield(imu, 'calib') && isfield(imu.calib, 'gyro_bias')
        b_g0 = imu.calib.gyro_bias;
    end
    if size(b_g0, 1) == 1
        b_g0 = b_g0';  % Ensure column vector
    end

    % Initial state
    x = [q0; b_g0];

    % Initial covariance
    P = diag([params.ekf.P0_quat * ones(4,1);
              params.ekf.P0_bias * ones(3,1)]);

    %% Process and measurement noise covariances
    Q = diag([params.ekf.Q_quat * ones(4,1);
              params.ekf.Q_gyro_bias * ones(3,1)]);

    R_acc = params.ekf.R_acc * eye(3);
    R_mag = params.ekf.R_mag * eye(3);

    %% Reference vectors in world frame
    g_world = params.frames.gravity_world;
    m_world = params.frames.mag_ref_world;
    m_world = m_world / (norm(m_world) + EPS);

    %% Pre-compute identity matrix
    I7 = eye(n_states);

    % Track previous quaternion for continuity
    q_prev = q0;

    %% Main EKF Loop
    for k = 1:n
        %% FIXED: Normalize quaternion BEFORE storing
        x(1:4) = qu.normalize(x(1:4));

        % FIXED: Quaternion sign continuity check using dot product
        if dot(x(1:4), q_prev) < 0
            x(1:4) = -x(1:4);
        end
        q_prev = x(1:4);

        % Ensure positive scalar part (after continuity check)
        if x(1) < 0
            x(1:4) = -x(1:4);
            q_prev = -q_prev;
        end

        %% Store current estimates
        est.q(k, :) = x(1:4)';
        est.b_g(k, :) = x(5:7)';
        if store_full_P
            est.P(:,:,k) = P;
        end
        est.Ptrace(k, :) = diag(P)';

        if k == n
            break;  % No prediction after last sample
        end

        %% Get measurements and dt
        dt = dt_vec(k);
        omega_m = imu.gyr(k, :)';
        acc_m = imu.acc(k, :)';
        mag_m = imu.mag(k, :)';

        %% === PREDICTION STEP ===

        q = x(1:4);
        b_g = x(5:7);

        % Corrected angular velocity
        omega = omega_m - b_g;

        % Quaternion propagation using exponential map
        q_delta = qu.fromOmega(omega, dt);
        q_pred = qu.multiply(q, q_delta);
        q_pred = qu.normalize(q_pred);

        % Gyro bias prediction (random walk)
        b_g_pred = b_g;

        % Predicted state
        x_pred = [q_pred; b_g_pred];

        % State transition Jacobian
        F = compute_state_jacobian(q, omega, dt);

        % Predicted covariance
        P_pred = F * P * F' + Q;

        %% === MEASUREMENT UPDATE (Accelerometer) ===

        acc_mag = norm(acc_m);
        acc_window = params.ekf.acc_magnitude_window;

        if acc_mag > acc_window(1) && acc_mag < acc_window(2)
            % OPTIMIZED: Cache rotation matrix
            R_pred = qu.toRotMat(q_pred);
            g_pred = R_pred' * g_world;

            % Normalize acceleration
            acc_norm = acc_m / (acc_mag + EPS) * params.constants.g;

            % Innovation
            y_acc = acc_norm - g_pred;

            % Measurement Jacobian
            H_acc = compute_acc_jacobian(q_pred, g_world);

            % Innovation covariance
            S_acc = H_acc * P_pred * H_acc' + R_acc;

            % OPTIMIZED: Use right division instead of explicit inverse
            K_acc = (P_pred * H_acc') / S_acc;

            % State update
            x_upd = x_pred + K_acc * y_acc;

            % FIXED: Joseph form covariance update with symmetry enforcement
            I_KH = I7 - K_acc * H_acc;
            P_upd = I_KH * P_pred * I_KH' + K_acc * R_acc * K_acc';

            % FIXED: Enforce symmetry
            if params.ekf.enforce_symmetry
                P_upd = (P_upd + P_upd') / 2;
            end

            % Store innovations
            est.innov_acc(k, :) = y_acc';
            est.S_acc(k, :) = diag(S_acc)';
        else
            x_upd = x_pred;
            P_upd = P_pred;
        end

        %% === MEASUREMENT UPDATE (Magnetometer) ===

        if params.ekf.use_mag_update && ~all(isnan(mag_m)) && ...
           isfield(imu, 'flags') && isfield(imu.flags, 'mag_outlier') && ...
           ~imu.flags.mag_outlier(k)

            q_current = x_upd(1:4);
            R_current = qu.toRotMat(q_current);
            m_pred = R_current' * m_world;

            % Normalize measurement
            mag_norm = mag_m / (norm(mag_m) + EPS);

            % Innovation
            y_mag = mag_norm - m_pred;

            % Outlier check
            if norm(y_mag) < params.ekf.mag_rejection_threshold * sqrt(params.ekf.R_mag)

                H_mag = compute_mag_jacobian(q_current, m_world);
                S_mag = H_mag * P_upd * H_mag' + R_mag;
                K_mag = (P_upd * H_mag') / S_mag;

                x_upd = x_upd + K_mag * y_mag;

                I_KH = I7 - K_mag * H_mag;
                P_upd = I_KH * P_upd * I_KH' + K_mag * R_mag * K_mag';

                if params.ekf.enforce_symmetry
                    P_upd = (P_upd + P_upd') / 2;
                end

                est.innov_mag(k, :) = y_mag';
            end
        end

        %% Prepare for next iteration
        x = x_upd;
        P = P_upd;

    end

    %% Post-processing: Convert to Euler angles
    est.euler = zeros(n, 3);
    for k = 1:n
        [roll, pitch, yaw] = qu.toEuler(est.q(k,:)');
        est.euler(k, :) = [roll, pitch, yaw];
    end

    % Store parameters and time
    est.params = params.ekf;
    est.t = imu.t;

    %% Summary
    if params.verbose
        qNorms = sqrt(sum(est.q.^2, 2));
        fprintf('EKF complete:\n');
        fprintf('  Final quaternion: [%.3f, %.3f, %.3f, %.3f]\n', est.q(end,:));
        fprintf('  Final gyro bias:  [%.4f, %.4f, %.4f] rad/s\n', est.b_g(end,:));
        fprintf('  Quaternion norm range: [%.6f, %.6f]\n', min(qNorms), max(qNorms));
    end

end

%% ==================== HELPER FUNCTIONS ====================

function q0 = initialize_orientation(imu, params)
%INITIALIZE_ORIENTATION Estimate initial orientation from static data

    qu = quat_utils();

    % Find first static segment
    static_idx = [];
    if isfield(imu, 'flags') && isfield(imu.flags, 'stationary')
        static_idx = find(imu.flags.stationary, 50);
    end

    if length(static_idx) >= 10
        acc_init = mean(imu.acc(static_idx, :))';
        mag_init = mean(imu.mag(static_idx, :))';

        if ~all(isnan(acc_init)) && ~all(isnan(mag_init))
            q0 = triad_quaternion(acc_init, mag_init, params);
            return;
        end
    end

    % Fallback: use first sample
    acc_init = imu.acc(1, :)';

    if ~all(isnan(acc_init))
        acc_init = acc_init / (norm(acc_init) + 1e-10);
        roll = atan2(-acc_init(2), -acc_init(3));
        pitch = atan2(acc_init(1), sqrt(acc_init(2)^2 + acc_init(3)^2));
        yaw = 0;
        q0 = qu.fromEuler(roll, pitch, yaw);
    else
        q0 = [1; 0; 0; 0];
    end
end

function q = triad_quaternion(acc, mag, params)
%TRIAD_QUATERNION Compute orientation from accelerometer and magnetometer

    qu = quat_utils();
    EPS = 1e-10;

    v1b = -acc / (norm(acc) + EPS);
    v2b = mag / (norm(mag) + EPS);

    v1w = -params.frames.gravity_world / (norm(params.frames.gravity_world) + EPS);
    v2w = params.frames.mag_ref_world / (norm(params.frames.mag_ref_world) + EPS);

    w1b = v1b;
    w2b = cross(v1b, v2b);
    w2b = w2b / (norm(w2b) + EPS);
    w3b = cross(w1b, w2b);

    w1w = v1w;
    w2w = cross(v1w, v2w);
    w2w = w2w / (norm(w2w) + EPS);
    w3w = cross(w1w, w2w);

    Mb = [w1b, w2b, w3b];
    Mw = [w1w, w2w, w3w];

    R = Mw * Mb';
    q = qu.fromRotMat(R);
end

function F = compute_state_jacobian(q, omega, dt)
%COMPUTE_STATE_JACOBIAN Compute state transition Jacobian

    wx = omega(1); wy = omega(2); wz = omega(3);

    Omega = [0, -wx, -wy, -wz;
             wx,  0,  wz, -wy;
             wy, -wz,  0,  wx;
             wz,  wy, -wx,  0];

    F_q = eye(4) + 0.5 * dt * Omega;

    qw = q(1); qx = q(2); qy = q(3); qz = q(4);

    Gamma = [-qx, -qy, -qz;
              qw, -qz,  qy;
              qz,  qw, -qx;
             -qy,  qx,  qw];

    F_qb = -0.5 * dt * Gamma;
    F_b = eye(3);

    F = [F_q,       F_qb;
         zeros(3,4), F_b];
end

function H = compute_acc_jacobian(q, g_world)
%COMPUTE_ACC_JACOBIAN Measurement Jacobian for accelerometer

    qw = q(1); qx = q(2); qy = q(3); qz = q(4);
    gx = g_world(1); gy = g_world(2); gz = g_world(3);

    H_q = 2 * [qw*gx + qz*gy - qy*gz,  qx*gx + qy*gy + qz*gz, -qy*gx + qx*gy - qw*gz, -qz*gx + qw*gy + qx*gz;
               -qz*gx + qw*gy + qx*gz,  qy*gx - qx*gy + qw*gz,  qx*gx + qy*gy + qz*gz, -qw*gx - qz*gy + qy*gz;
               qy*gx - qx*gy + qw*gz,  qz*gx - qw*gy - qx*gz,  qw*gx + qz*gy - qy*gz,  qx*gx + qy*gy + qz*gz];

    H_b = zeros(3, 3);
    H = [H_q, H_b];
end

function H = compute_mag_jacobian(q, m_world)
%COMPUTE_MAG_JACOBIAN Measurement Jacobian for magnetometer

    qw = q(1); qx = q(2); qy = q(3); qz = q(4);
    mx = m_world(1); my = m_world(2); mz = m_world(3);

    H_q = 2 * [qw*mx + qz*my - qy*mz,  qx*mx + qy*my + qz*mz, -qy*mx + qx*my - qw*mz, -qz*mx + qw*my + qx*mz;
               -qz*mx + qw*my + qx*mz,  qy*mx - qx*my + qw*mz,  qx*mx + qy*my + qz*mz, -qw*mx - qz*my + qy*mz;
               qy*mx - qx*my + qw*mz,  qz*mx - qw*my - qx*mz,  qw*mx + qz*my - qy*mz,  qx*mx + qy*my + qz*mz];

    H_b = zeros(3, 3);
    H = [H_q, H_b];
end
