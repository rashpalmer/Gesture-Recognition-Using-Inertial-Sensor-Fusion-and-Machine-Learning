classdef realtime_gesture_processor < handle
%REALTIME_GESTURE_PROCESSOR Real-time gesture recognition processor
%   Processes streaming IMU data for gesture recognition using a sliding
%   window approach with the EKF attitude estimator.
%
%   USAGE:
%       processor = realtime_gesture_processor();
%       processor.initialize();
%
%       % In streaming loop:
%       while streaming
%           [acc, gyr, mag] = read_sensors();
%           result = processor.process_sample(acc, gyr, mag, dt);
%           if result.gesture_detected
%               fprintf('Gesture: %s (%.1f%% confidence)\n', ...
%                   result.label, result.confidence*100);
%           end
%       end
%
%   FEATURES:
%       - Circular buffer for efficient memory usage
%       - Incremental EKF updates
%       - Energy-based gesture detection with hysteresis
%       - Configurable window sizes and thresholds
%       - Low latency (~10ms per sample on modern CPU)
%
%   MICROCONTROLLER NOTES:
%       This implementation uses MATLAB syntax but can be converted to
%       C/C++ using MATLAB Coder for embedded deployment:
%       - Replace handle class with struct + function pointers
%       - Pre-allocate all buffers
%       - Use fixed-point arithmetic where possible
%
%   Author: Sensor Fusion Demo
%   Date: 2026

    properties (SetAccess = private)
        % Configuration
        params          % Configuration parameters
        buffer_size     % Circular buffer size (samples)
        window_size     % Analysis window size (samples)
        Fs              % Expected sample rate (Hz)

        % State
        initialized     % Boolean flag
        sample_count    % Total samples processed
        gesture_state   % State machine: 'idle', 'detecting', 'active'

        % Circular buffers
        acc_buffer      % Nx3 accelerometer buffer
        gyr_buffer      % Nx3 gyroscope buffer
        mag_buffer      % Nx3 magnetometer buffer
        t_buffer        % Nx1 timestamp buffer
        energy_buffer   % Nx1 energy buffer

        % Buffer indices
        head_idx        % Write position (newest)
        fill_count      % Number of valid samples

        % EKF state (persistent across samples)
        q               % 4x1 current quaternion estimate
        b_g             % 3x1 current gyro bias estimate
        P               % 7x7 EKF covariance

        % Velocity/Position state
        v               % 3x1 velocity estimate
        p               % 3x1 position estimate
        P_motion        % 6x6 motion KF covariance

        % Gesture detection state
        in_gesture      % Boolean: currently in gesture
        gesture_start   % Start index of current gesture
        energy_smooth   % Smoothed energy value
        last_gesture    % Cache of last detected gesture
    end

    properties (Constant)
        % Default parameters
        DEFAULT_BUFFER_SIZE = 500;    % ~5 seconds at 100 Hz
        DEFAULT_WINDOW_SIZE = 100;    % 1 second at 100 Hz
        DEFAULT_FS = 100;             % Sample rate
    end

    methods
        %% ==================== CONSTRUCTOR ====================

        function obj = realtime_gesture_processor(params)
        %REALTIME_GESTURE_PROCESSOR Constructor
            if nargin < 1
                obj.params = config_params();
            else
                obj.params = params;
            end

            obj.buffer_size = obj.DEFAULT_BUFFER_SIZE;
            obj.window_size = obj.DEFAULT_WINDOW_SIZE;
            obj.Fs = obj.DEFAULT_FS;
            obj.initialized = false;
        end

        %% ==================== INITIALIZATION ====================

        function initialize(obj, Fs)
        %INITIALIZE Initialize processor state and buffers
            if nargin > 1
                obj.Fs = Fs;
            end

            % Pre-allocate circular buffers
            obj.acc_buffer = zeros(obj.buffer_size, 3);
            obj.gyr_buffer = zeros(obj.buffer_size, 3);
            obj.mag_buffer = nan(obj.buffer_size, 3);  % Optional
            obj.t_buffer = zeros(obj.buffer_size, 1);
            obj.energy_buffer = zeros(obj.buffer_size, 1);

            % Initialize indices
            obj.head_idx = 0;
            obj.fill_count = 0;
            obj.sample_count = 0;

            % Initialize EKF state
            obj.q = [1; 0; 0; 0];  % Identity quaternion
            obj.b_g = [0; 0; 0];   % Zero bias
            obj.P = diag([0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01]);

            % Initialize motion state
            obj.v = [0; 0; 0];
            obj.p = [0; 0; 0];
            obj.P_motion = diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01]);

            % Initialize gesture detection state
            obj.gesture_state = 'idle';
            obj.in_gesture = false;
            obj.gesture_start = 0;
            obj.energy_smooth = 0;
            obj.last_gesture = struct('label', 'unknown', 'confidence', 0);

            obj.initialized = true;
            fprintf('Real-time processor initialized (Fs=%.1f Hz, buffer=%d samples)\n', ...
                obj.Fs, obj.buffer_size);
        end

        %% ==================== MAIN PROCESSING ====================

        function result = process_sample(obj, acc, gyr, mag, dt)
        %PROCESS_SAMPLE Process a single IMU sample
        %   acc: 3x1 accelerometer (m/s²)
        %   gyr: 3x1 gyroscope (rad/s)
        %   mag: 3x1 magnetometer (µT) or empty
        %   dt:  time since last sample (seconds)

            if ~obj.initialized
                error('Processor not initialized. Call initialize() first.');
            end

            % Ensure column vectors
            acc = acc(:);
            gyr = gyr(:);
            if nargin < 4 || isempty(mag)
                mag = [NaN; NaN; NaN];
            else
                mag = mag(:);
            end
            if nargin < 5
                dt = 1 / obj.Fs;
            end

            % Update sample count and buffers
            obj.sample_count = obj.sample_count + 1;
            obj.update_buffers(acc, gyr, mag, dt);

            % Update EKF (attitude estimation)
            obj.ekf_update(acc, gyr, mag, dt);

            % Update motion estimation
            obj.motion_update(acc, dt);

            % Compute instantaneous energy
            energy = obj.compute_energy(gyr, acc);
            obj.energy_buffer(obj.head_idx) = energy;

            % Update smoothed energy (exponential moving average)
            alpha = 0.1;  % Smoothing factor
            obj.energy_smooth = alpha * energy + (1 - alpha) * obj.energy_smooth;

            % Gesture detection state machine
            result = obj.detect_gesture();

            % Add current state to result
            result.sample_count = obj.sample_count;
            result.quaternion = obj.q';
            result.energy = obj.energy_smooth;
        end

        %% ==================== BUFFER MANAGEMENT ====================

        function update_buffers(obj, acc, gyr, mag, dt)
        %UPDATE_BUFFERS Add sample to circular buffers
            obj.head_idx = mod(obj.head_idx, obj.buffer_size) + 1;
            obj.fill_count = min(obj.fill_count + 1, obj.buffer_size);

            obj.acc_buffer(obj.head_idx, :) = acc';
            obj.gyr_buffer(obj.head_idx, :) = gyr';
            obj.mag_buffer(obj.head_idx, :) = mag';

            if obj.head_idx == 1
                obj.t_buffer(1) = 0;
            else
                prev_idx = obj.head_idx - 1;
                obj.t_buffer(obj.head_idx) = obj.t_buffer(prev_idx) + dt;
            end
        end

        function [data, valid] = get_window(obj, window_size)
        %GET_WINDOW Get most recent window_size samples from buffer
            if nargin < 2
                window_size = obj.window_size;
            end

            if obj.fill_count < window_size
                valid = false;
                data = [];
                return;
            end

            valid = true;
            indices = obj.get_buffer_indices(window_size);

            data.acc = obj.acc_buffer(indices, :);
            data.gyr = obj.gyr_buffer(indices, :);
            data.mag = obj.mag_buffer(indices, :);
            data.t = obj.t_buffer(indices);
            data.energy = obj.energy_buffer(indices);
        end

        function indices = get_buffer_indices(obj, count)
        %GET_BUFFER_INDICES Get indices for most recent 'count' samples
            indices = zeros(count, 1);
            for i = 1:count
                idx = obj.head_idx - count + i;
                if idx <= 0
                    idx = idx + obj.buffer_size;
                end
                indices(i) = idx;
            end
        end

        %% ==================== EKF UPDATE ====================

        function ekf_update(obj, acc, gyr, mag, dt)
        %EKF_UPDATE Incremental EKF attitude update

            qu = quat_utils();

            % Bias-corrected gyroscope
            gyr_corrected = gyr - obj.b_g;

            % Prediction: propagate quaternion
            omega_norm = norm(gyr_corrected);
            if omega_norm > 1e-10
                q_delta = qu.fromOmega(gyr_corrected, dt);
                obj.q = qu.multiply(obj.q, q_delta);
                obj.q = qu.normalize(obj.q);
            end

            % Simplified measurement update (accelerometer gravity reference)
            % Full EKF linearization omitted for clarity - see ekf_attitude_quat.m
            acc_norm = norm(acc);
            if abs(acc_norm - 9.81) < 2.0  % Only update if near 1g
                % Compute expected gravity in body frame
                g_world = [0; 0; -9.81];
                R = qu.toRotMat(obj.q);
                g_body_expected = R' * g_world;

                % Innovation (simplified)
                g_body_measured = acc;
                innovation = g_body_measured - g_body_expected;

                % Simple correction (should use full Kalman gain)
                correction_scale = 0.01;
                correction = correction_scale * innovation;

                % Apply small rotation correction
                angle = norm(correction) * dt;
                if angle > 1e-10
                    axis = correction / norm(correction);
                    q_corr = qu.fromAxisAngle(axis, angle);
                    obj.q = qu.multiply(obj.q, q_corr);
                    obj.q = qu.normalize(obj.q);
                end
            end
        end

        %% ==================== MOTION UPDATE ====================

        function motion_update(obj, acc, dt)
        %MOTION_UPDATE Update velocity and position estimates

            qu = quat_utils();

            % Transform acceleration to world frame
            R = qu.toRotMat(obj.q);
            a_world = R * acc - [0; 0; 9.81];  % Remove gravity

            % Simple integration (could add ZUPT)
            obj.v = obj.v + a_world * dt;
            obj.p = obj.p + obj.v * dt;

            % Apply damping to prevent unbounded drift
            damping = 0.99;
            obj.v = obj.v * damping;
        end

        %% ==================== GESTURE DETECTION ====================

        function energy = compute_energy(obj, gyr, acc)
        %COMPUTE_ENERGY Compute instantaneous motion energy

            % Normalized energy (from segment_gesture.m improvements)
            gyr_norm_factor = 2.0;  % rad/s
            acc_norm_factor = 5.0;  % m/s²
            acc_weight = 0.3;

            gyr_mag = norm(gyr);
            acc_dev = abs(norm(acc) - 9.81);

            gyr_norm = gyr_mag / gyr_norm_factor;
            acc_norm = acc_dev / acc_norm_factor;

            energy = gyr_norm + acc_weight * acc_norm;
        end

        function result = detect_gesture(obj)
        %DETECT_GESTURE Gesture detection state machine

            result = struct();
            result.gesture_detected = false;
            result.gesture_ended = false;
            result.label = '';
            result.confidence = 0;

            % Thresholds
            onset_threshold = obj.params.segmentation.onset_threshold;
            offset_threshold = obj.params.segmentation.offset_threshold;
            min_duration = 0.1;  % seconds
            max_duration = 3.0;  % seconds

            min_samples = min_duration * obj.Fs;
            max_samples = max_duration * obj.Fs;

            switch obj.gesture_state
                case 'idle'
                    % Looking for gesture onset
                    if obj.energy_smooth > onset_threshold
                        obj.gesture_state = 'detecting';
                        obj.gesture_start = obj.sample_count;
                        obj.in_gesture = true;
                    end

                case 'detecting'
                    % In potential gesture, waiting for offset
                    duration_samples = obj.sample_count - obj.gesture_start;

                    if obj.energy_smooth < offset_threshold
                        % Gesture ended
                        if duration_samples >= min_samples
                            % Valid gesture - classify it
                            result = obj.classify_current_gesture();
                            result.gesture_detected = true;
                            result.gesture_ended = true;
                            obj.last_gesture = result;
                        end
                        obj.gesture_state = 'idle';
                        obj.in_gesture = false;

                    elseif duration_samples > max_samples
                        % Gesture too long - reset
                        obj.gesture_state = 'idle';
                        obj.in_gesture = false;
                    end

                otherwise
                    obj.gesture_state = 'idle';
            end

            result.in_gesture = obj.in_gesture;
        end

        function result = classify_current_gesture(obj)
        %CLASSIFY_CURRENT_GESTURE Classify the current gesture window

            result = struct();
            result.label = 'unknown';
            result.confidence = 0;

            % Get gesture window
            duration_samples = obj.sample_count - obj.gesture_start;
            window_size = min(duration_samples, obj.fill_count);

            [data, valid] = obj.get_window(window_size);
            if ~valid
                return;
            end

            % Create minimal IMU struct for feature extraction
            imu.t = data.t;
            imu.dt = diff(data.t);
            if isempty(imu.dt), imu.dt = 1/obj.Fs; end
            imu.Fs = obj.Fs;
            imu.acc = data.acc;
            imu.gyr = data.gyr;
            imu.mag = data.mag;
            imu.flags.stationary = false(size(data.t));

            % Create minimal est struct
            est.q = repmat(obj.q', length(data.t), 1);
            est.euler = zeros(length(data.t), 3);
            est.t = data.t;

            % Create segmentation struct
            seg.winIdx = [1, length(data.t)];
            seg.windows = seg.winIdx;
            seg.primary = 1;
            seg.score = 1.0;
            seg.energy = data.energy;

            % Extract features
            try
                feat = extract_features(imu, est, seg, obj.params);

                % Classify
                cls = classify_gesture_rules(feat, obj.params);

                result.label = cls.label;
                result.confidence = cls.score;
                result.reason = cls.reason;
            catch ME
                warning('Classification failed: %s', ME.message);
            end
        end

        %% ==================== UTILITY METHODS ====================

        function reset(obj)
        %RESET Reset processor state (keep configuration)
            obj.initialize(obj.Fs);
        end

        function stats = get_stats(obj)
        %GET_STATS Get processor statistics
            stats.sample_count = obj.sample_count;
            stats.fill_count = obj.fill_count;
            stats.gesture_state = obj.gesture_state;
            stats.current_energy = obj.energy_smooth;
            stats.last_gesture = obj.last_gesture;
            stats.quaternion = obj.q';
            stats.position = obj.p';
            stats.velocity = obj.v';
        end
    end
end
