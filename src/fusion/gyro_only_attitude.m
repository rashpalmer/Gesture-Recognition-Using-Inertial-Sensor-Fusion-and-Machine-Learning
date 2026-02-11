function est = gyro_only_attitude(imu, params)
%GYRO_ONLY_ATTITUDE Dead-reckoning attitude from gyroscope integration
%   Fallback estimator when EKF fails (e.g. magnetometer anomaly).
%   Integrates bias-corrected gyroscope using quaternion kinematics.
%
%   SYNTAX:
%       est = gyro_only_attitude(imu)
%       est = gyro_only_attitude(imu, params)
%
%   INPUTS:
%       imu    - Preprocessed IMU struct (from preprocess_imu)
%       params - Configuration (optional, unused but accepted for API parity
%                with ekf_attitude_quat)
%
%   OUTPUTS:
%       est - Attitude estimate struct matching ekf_attitude_quat output:
%             .q     : Nx4 quaternion history [w x y z]
%             .euler : Nx3 Euler angles (rad) [roll pitch yaw]
%             .b_g   : Nx3 gyro bias (zeros — not estimated)
%             .t     : Nx1 time vector
%             .method: 'gyro_only'
%
%   NOTES:
%       - No sensor fusion: orientation will drift over time
%       - Suitable for short gesture windows (~1-3 s) where drift is small
%       - Bias correction uses preprocess_imu calibration if available
%
%   See also: ekf_attitude_quat, preprocess_imu

    n = length(imu.t);
    dt = median(diff(imu.t));

    % Get gyro bias from calibration if available
    gyro_bias = [0, 0, 0];
    if isfield(imu, 'calib') && isfield(imu.calib, 'gyro_bias')
        bias = imu.calib.gyro_bias;
        gyro_bias = bias(:)';  % Ensure row vector
    end

    % Pre-allocate output
    est = struct();
    est.q = zeros(n, 4);
    est.euler = zeros(n, 3);
    est.b_g = zeros(n, 3);
    est.t = imu.t;
    est.method = 'gyro_only';

    % Integrate gyroscope with first-order quaternion kinematics
    q = [1, 0, 0, 0];  % Identity — start with unknown orientation

    for k = 1:n
        est.q(k, :) = q;

        if k < n
            % Bias-corrected angular velocity
            w = imu.gyr(k, :) - gyro_bias;

            % Quaternion derivative: dq/dt = 0.5 * q ⊗ [0, ω]
            omega_quat = [0, w];
            dq = 0.5 * qmul(q, omega_quat);
            q = q + dq * dt;
            q = q / norm(q);
        end
    end

    % Convert to Euler angles (ZYX convention → [roll, pitch, yaw])
    for k = 1:n
        w = est.q(k,1); x = est.q(k,2); y = est.q(k,3); z = est.q(k,4);

        % Roll (X-axis rotation)
        sinr_cosp = 2 * (w*x + y*z);
        cosr_cosp = 1 - 2 * (x^2 + y^2);
        est.euler(k, 1) = atan2(sinr_cosp, cosr_cosp);

        % Pitch (Y-axis rotation)
        sinp = 2 * (w*y - z*x);
        if abs(sinp) >= 1
            est.euler(k, 2) = sign(sinp) * pi/2;
        else
            est.euler(k, 2) = asin(sinp);
        end

        % Yaw (Z-axis rotation)
        siny_cosp = 2 * (w*z + x*y);
        cosy_cosp = 1 - 2 * (y^2 + z^2);
        est.euler(k, 3) = atan2(siny_cosp, cosy_cosp);
    end
end

%% ==================== LOCAL QUATERNION MULTIPLY ====================
% Self-contained to avoid dependency on quat_utils or Aerospace Toolbox

function r = qmul(p, q)
%QMUL Hamilton quaternion product [w x y z] convention
    r = [
        p(1)*q(1) - p(2)*q(2) - p(3)*q(3) - p(4)*q(4), ...
        p(1)*q(2) + p(2)*q(1) + p(3)*q(4) - p(4)*q(3), ...
        p(1)*q(3) - p(2)*q(4) + p(3)*q(1) + p(4)*q(2), ...
        p(1)*q(4) + p(2)*q(3) - p(3)*q(2) + p(4)*q(1)
    ];
end
