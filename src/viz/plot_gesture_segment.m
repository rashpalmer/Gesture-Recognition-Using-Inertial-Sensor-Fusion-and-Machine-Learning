%% PLOT_GESTURE_SEGMENT - Visualize a Single Gesture Segment
% Plots detailed visualization of a single detected gesture segment,
% showing IMU signals, orientation changes, and extracted features.
%
% SYNTAX:
%   plot_gesture_segment(imu, est, seg, feat, params)
%   plot_gesture_segment(imu, est, seg, feat, params, windowIdx)
%   fig = plot_gesture_segment(...)
%
% INPUTS:
%   imu       - Preprocessed IMU data struct
%   est       - Attitude estimation struct
%   seg       - Segmentation struct
%   feat      - Features struct (optional, can be [])
%   params    - Configuration parameters
%   windowIdx - (optional) Which window to plot (default: primary)
%
% OUTPUTS:
%   fig - Figure handle
%
% See also: plot_diagnostics, segment_gesture, extract_features

function fig = plot_gesture_segment(imu, est, seg, feat, params, windowIdx)
    %% ========================================================================
    %  INITIALIZATION
    %  ========================================================================
    
    % Default to primary window
    if nargin < 6 || isempty(windowIdx)
        windowIdx = seg.primary;
    end
    
    % Validate window index
    if windowIdx < 1 || windowIdx > size(seg.windows, 1)
        error('Invalid window index: %d (available: 1-%d)', windowIdx, size(seg.windows, 1));
    end
    
    % Get window bounds
    iStart = seg.windows(windowIdx, 1);
    iEnd = seg.windows(windowIdx, 2);
    
    % Extract segment data
    idx = iStart:iEnd;
    t_seg = imu.t(idx) - imu.t(iStart);  % Time relative to segment start
    
    acc_seg = imu.acc(idx, :);
    gyr_seg = imu.gyr(idx, :);
    euler_seg = est.euler(idx, :);
    
    % Colors
    colors = struct();
    colors.x = [0.8500, 0.3250, 0.0980];
    colors.y = [0.4660, 0.6740, 0.1880];
    colors.z = [0.0000, 0.4470, 0.7410];
    
    %% ========================================================================
    %  CREATE FIGURE
    %  ========================================================================
    
    fig = figure('Name', sprintf('Gesture Segment %d', windowIdx), ...
                 'NumberTitle', 'off', 'Position', [100, 100, 1000, 700]);
    
    %% ========================================================================
    %  ACCELEROMETER
    %  ========================================================================
    
    subplot(3, 2, 1);
    hold on;
    plot(t_seg, acc_seg(:, 1), 'Color', colors.x, 'LineWidth', 1.5);
    plot(t_seg, acc_seg(:, 2), 'Color', colors.y, 'LineWidth', 1.5);
    plot(t_seg, acc_seg(:, 3), 'Color', colors.z, 'LineWidth', 1.5);
    ylabel('Acceleration (m/s^2)');
    legend('a_x', 'a_y', 'a_z', 'Location', 'best');
    title('Accelerometer');
    grid on;
    xlim([0, t_seg(end)]);
    
    %% ========================================================================
    %  GYROSCOPE
    %  ========================================================================
    
    subplot(3, 2, 3);
    hold on;
    plot(t_seg, rad2deg(gyr_seg(:, 1)), 'Color', colors.x, 'LineWidth', 1.5);
    plot(t_seg, rad2deg(gyr_seg(:, 2)), 'Color', colors.y, 'LineWidth', 1.5);
    plot(t_seg, rad2deg(gyr_seg(:, 3)), 'Color', colors.z, 'LineWidth', 1.5);
    ylabel('Angular Rate (deg/s)');
    legend('\omega_x', '\omega_y', '\omega_z', 'Location', 'best');
    title('Gyroscope');
    grid on;
    xlim([0, t_seg(end)]);
    
    %% ========================================================================
    %  EULER ANGLES
    %  ========================================================================
    
    subplot(3, 2, 5);
    hold on;
    
    % Show change from initial orientation
    euler_rel = euler_seg - euler_seg(1, :);
    
    plot(t_seg, rad2deg(euler_rel(:, 1)), 'Color', colors.x, 'LineWidth', 1.5);
    plot(t_seg, rad2deg(euler_rel(:, 2)), 'Color', colors.y, 'LineWidth', 1.5);
    plot(t_seg, rad2deg(euler_rel(:, 3)), 'Color', colors.z, 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Angle Change (deg)');
    legend('\Delta Roll', '\Delta Pitch', '\Delta Yaw', 'Location', 'best');
    title('Orientation Change');
    grid on;
    xlim([0, t_seg(end)]);
    
    %% ========================================================================
    %  MOTION ENERGY
    %  ========================================================================
    
    subplot(3, 2, 2);
    
    if isfield(seg, 'energy') && ~isempty(seg.energy)
        energy_seg = seg.energy(idx);
        plot(t_seg, energy_seg, 'b', 'LineWidth', 1.5);
        hold on;
        yline(params.segmentation.energy_high, 'r--', 'LineWidth', 1);
        yline(params.segmentation.energy_low, 'g--', 'LineWidth', 1);
        ylabel('Motion Energy');
        title('Motion Energy Profile');
    else
        % Compute gyro magnitude
        gyrMag = sqrt(sum(gyr_seg.^2, 2));
        plot(t_seg, rad2deg(gyrMag), 'b', 'LineWidth', 1.5);
        ylabel('Gyro Magnitude (deg/s)');
        title('Gyroscope Magnitude');
    end
    grid on;
    xlim([0, t_seg(end)]);
    
    %% ========================================================================
    %  3D ORIENTATION VISUALIZATION
    %  ========================================================================
    
    subplot(3, 2, 4);
    
    % Plot orientation trajectory in 3D (roll-pitch-yaw space)
    plot3(rad2deg(euler_seg(:, 1)), rad2deg(euler_seg(:, 2)), rad2deg(euler_seg(:, 3)), ...
          'b', 'LineWidth', 1.5);
    hold on;
    scatter3(rad2deg(euler_seg(1, 1)), rad2deg(euler_seg(1, 2)), rad2deg(euler_seg(1, 3)), ...
             100, 'g', 'filled');
    scatter3(rad2deg(euler_seg(end, 1)), rad2deg(euler_seg(end, 2)), rad2deg(euler_seg(end, 3)), ...
             100, 'r', 'filled');
    xlabel('Roll (deg)');
    ylabel('Pitch (deg)');
    zlabel('Yaw (deg)');
    legend('Trajectory', 'Start', 'End', 'Location', 'best');
    title('Orientation Trajectory');
    grid on;
    view(30, 30);
    
    %% ========================================================================
    %  FEATURE SUMMARY
    %  ========================================================================
    
    subplot(3, 2, 6);
    axis off;
    
    % Build feature text
    if ~isempty(feat) && isfield(feat, 'values')
        textLines = {'KEY FEATURES:', ''};
        
        % Duration
        if isfield(feat.values, 'duration')
            textLines{end+1} = sprintf('Duration: %.3f s', feat.values.duration);
        end
        
        % Total rotation
        if isfield(feat.values, 'total_rotation_deg')
            textLines{end+1} = sprintf('Total rotation: %.1f deg', feat.values.total_rotation_deg);
        end
        
        % Dominant axis
        if isfield(feat.values, 'dominant_axis')
            textLines{end+1} = sprintf('Dominant axis: %s', feat.values.dominant_axis);
        end
        
        % RMS values
        if isfield(feat.values, 'rms_gyr_total')
            textLines{end+1} = sprintf('Gyro RMS: %.1f deg/s', rad2deg(feat.values.rms_gyr_total));
        end
        if isfield(feat.values, 'rms_acc_total')
            textLines{end+1} = sprintf('Accel RMS: %.2f m/s^2', feat.values.rms_acc_total);
        end
        
        % Peak values
        if isfield(feat.values, 'gyr_peak_abs')
            textLines{end+1} = sprintf('Peak gyro: %.1f deg/s', rad2deg(feat.values.gyr_peak_abs));
        end
        
        % Orientation change
        if isfield(feat.values, 'delta_roll_deg')
            textLines{end+1} = sprintf('\\Delta Roll: %.1f deg', feat.values.delta_roll_deg);
        end
        if isfield(feat.values, 'delta_pitch_deg')
            textLines{end+1} = sprintf('\\Delta Pitch: %.1f deg', feat.values.delta_pitch_deg);
        end
        if isfield(feat.values, 'delta_yaw_deg')
            textLines{end+1} = sprintf('\\Delta Yaw: %.1f deg', feat.values.delta_yaw_deg);
        end
        
        text(0.1, 0.9, strjoin(textLines, '\n'), 'FontSize', 10, ...
             'VerticalAlignment', 'top', 'FontName', 'FixedWidth');
    else
        % Basic stats without features
        duration = t_seg(end);
        gyrMag = sqrt(sum(gyr_seg.^2, 2));
        
        textLines = {
            'SEGMENT STATISTICS:', ''
            sprintf('Duration: %.3f s', duration)
            sprintf('Samples: %d', length(idx))
            sprintf('Peak gyro: %.1f deg/s', rad2deg(max(gyrMag)))
            sprintf('Mean gyro: %.1f deg/s', rad2deg(mean(gyrMag)))
        };
        
        text(0.1, 0.9, strjoin(textLines, '\n'), 'FontSize', 10, ...
             'VerticalAlignment', 'top', 'FontName', 'FixedWidth');
    end
    
    %% ========================================================================
    %  TITLE
    %  ========================================================================
    
    segDuration = (iEnd - iStart) / imu.Fs;
    if isfield(seg, 'score')
        sgtitle(sprintf('Gesture Segment %d/%d (%.2f s, score=%.2f)', ...
                        windowIdx, size(seg.windows, 1), segDuration, seg.score));
    else
        sgtitle(sprintf('Gesture Segment %d/%d (%.2f s)', ...
                        windowIdx, size(seg.windows, 1), segDuration));
    end
end
