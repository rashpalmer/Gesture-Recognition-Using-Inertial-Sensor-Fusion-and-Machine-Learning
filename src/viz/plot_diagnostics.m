%% PLOT_DIAGNOSTICS - Comprehensive Visualization for Gesture Recognition Pipeline
% Generates diagnostic plots showing all stages of the gesture recognition pipeline:
% raw IMU data, preprocessed signals, attitude estimation, motion tracking,
% gesture segmentation, and classification results.
%
% SYNTAX:
%   plot_diagnostics(imu, est, motion, seg, feat, cls, params)
%   figs = plot_diagnostics(...)  % Returns figure handles
%
% INPUTS:
%   imu    - Preprocessed IMU data struct (from preprocess_imu)
%   est    - Attitude estimation struct (from ekf_attitude_quat or complementary_filter)
%   motion - Motion estimation struct (from kf_linear_motion), can be []
%   seg    - Segmentation struct (from segment_gesture), can be []
%   feat   - Features struct (from extract_features), can be []
%   cls    - Classification struct (from classify_gesture_rules or ml_predict), can be []
%   params - Configuration parameters (from config_params)
%
% OUTPUTS:
%   figs   - (optional) Array of figure handles
%
% GENERATED FIGURES:
%   Figure 1: Raw/Preprocessed IMU signals
%   Figure 2: Attitude Estimation (Euler angles, quaternion)
%   Figure 3: Motion Estimation (velocity, position, ZUPT)
%   Figure 4: Gesture Segmentation
%   Figure 5: EKF Diagnostics (covariance, innovations)
%
% See also: plot_gesture_segment, config_params

function figs = plot_diagnostics(imu, est, motion, seg, feat, cls, params)
    %% ========================================================================
    %  INITIALIZATION
    %  ========================================================================
    
    figs = [];
    
    % Check what data is available
    hasMotion = ~isempty(motion) && isfield(motion, 'v');
    hasSeg = ~isempty(seg) && isfield(seg, 'windows') && ~isempty(seg.windows);
    hasFeat = ~isempty(feat) && isfield(feat, 'values');
    hasCls = ~isempty(cls) && isfield(cls, 'label');
    hasEKFDiag = isfield(est, 'Ptrace') || isfield(est, 'innov_acc');
    
    % Time vector
    t = imu.t - imu.t(1);  % Start from zero
    
    % Colors
    colors = struct();
    colors.x = [0.8500, 0.3250, 0.0980];  % Orange-red
    colors.y = [0.4660, 0.6740, 0.1880];  % Green
    colors.z = [0.0000, 0.4470, 0.7410];  % Blue
    colors.seg = [1.0, 0.9, 0.8];         % Light orange for segments
    colors.zupt = [0.8, 1.0, 0.8];        % Light green for ZUPT
    
    %% ========================================================================
    %  FIGURE 1: IMU SIGNALS
    %  ========================================================================
    
    fig1 = figure('Name', 'IMU Signals', 'NumberTitle', 'off', ...
                  'Position', [50, 400, 1200, 600]);
    figs = [figs, fig1];
    
    % Accelerometer
    subplot(3, 1, 1);
    hold on;
    plot(t, imu.acc(:, 1), 'Color', colors.x, 'LineWidth', 1);
    plot(t, imu.acc(:, 2), 'Color', colors.y, 'LineWidth', 1);
    plot(t, imu.acc(:, 3), 'Color', colors.z, 'LineWidth', 1);
    
    % Mark gesture segments
    if hasSeg
        for w = 1:size(seg.windows, 1)
            xPatch = t([seg.windows(w,1), seg.windows(w,2), seg.windows(w,2), seg.windows(w,1)]);
            yLim = ylim;
            yPatch = [yLim(1), yLim(1), yLim(2), yLim(2)];
            patch(xPatch, yPatch, colors.seg, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        end
    end
    
    ylabel('Acceleration (m/s^2)');
    legend('a_x', 'a_y', 'a_z', 'Location', 'eastoutside');
    title('Accelerometer');
    grid on;
    xlim([0, t(end)]);
    
    % Gyroscope
    subplot(3, 1, 2);
    hold on;
    plot(t, rad2deg(imu.gyr(:, 1)), 'Color', colors.x, 'LineWidth', 1);
    plot(t, rad2deg(imu.gyr(:, 2)), 'Color', colors.y, 'LineWidth', 1);
    plot(t, rad2deg(imu.gyr(:, 3)), 'Color', colors.z, 'LineWidth', 1);
    
    if hasSeg
        for w = 1:size(seg.windows, 1)
            xPatch = t([seg.windows(w,1), seg.windows(w,2), seg.windows(w,2), seg.windows(w,1)]);
            yLim = ylim;
            yPatch = [yLim(1), yLim(1), yLim(2), yLim(2)];
            patch(xPatch, yPatch, colors.seg, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        end
    end
    
    ylabel('Angular Rate (deg/s)');
    legend('\omega_x', '\omega_y', '\omega_z', 'Location', 'eastoutside');
    title('Gyroscope');
    grid on;
    xlim([0, t(end)]);
    
    % Magnetometer (if available)
    subplot(3, 1, 3);
    if ~isempty(imu.mag)
        hold on;
        plot(t, imu.mag(:, 1), 'Color', colors.x, 'LineWidth', 1);
        plot(t, imu.mag(:, 2), 'Color', colors.y, 'LineWidth', 1);
        plot(t, imu.mag(:, 3), 'Color', colors.z, 'LineWidth', 1);
        ylabel('Magnetic Field (\muT)');
        legend('m_x', 'm_y', 'm_z', 'Location', 'eastoutside');
        title('Magnetometer');
    else
        % Show stationary detection instead
        plot(t, imu.flags.stationary, 'k', 'LineWidth', 1.5);
        ylabel('Stationary Flag');
        title('Stationary Detection');
        ylim([-0.1, 1.1]);
    end
    grid on;
    xlim([0, t(end)]);
    xlabel('Time (s)');
    
    sgtitle('Preprocessed IMU Signals');
    
    %% ========================================================================
    %  FIGURE 2: ATTITUDE ESTIMATION
    %  ========================================================================
    
    fig2 = figure('Name', 'Attitude Estimation', 'NumberTitle', 'off', ...
                  'Position', [100, 350, 1200, 600]);
    figs = [figs, fig2];
    
    % Euler angles
    subplot(2, 2, [1, 3]);
    hold on;
    plot(t, rad2deg(est.euler(:, 1)), 'Color', colors.x, 'LineWidth', 1.5);
    plot(t, rad2deg(est.euler(:, 2)), 'Color', colors.y, 'LineWidth', 1.5);
    plot(t, rad2deg(est.euler(:, 3)), 'Color', colors.z, 'LineWidth', 1.5);
    
    if hasSeg
        for w = 1:size(seg.windows, 1)
            xPatch = t([seg.windows(w,1), seg.windows(w,2), seg.windows(w,2), seg.windows(w,1)]);
            yLim = ylim;
            yPatch = [yLim(1), yLim(1), yLim(2), yLim(2)];
            patch(xPatch, yPatch, colors.seg, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        end
    end
    
    xlabel('Time (s)');
    ylabel('Angle (deg)');
    legend('Roll', 'Pitch', 'Yaw', 'Location', 'best');
    title('Euler Angles (from EKF)');
    grid on;
    xlim([0, t(end)]);
    
    % Quaternion components
    subplot(2, 2, 2);
    hold on;
    plot(t, est.q(:, 1), 'k', 'LineWidth', 1.5);
    plot(t, est.q(:, 2), 'Color', colors.x, 'LineWidth', 1);
    plot(t, est.q(:, 3), 'Color', colors.y, 'LineWidth', 1);
    plot(t, est.q(:, 4), 'Color', colors.z, 'LineWidth', 1);
    ylabel('Quaternion');
    legend('q_w', 'q_x', 'q_y', 'q_z', 'Location', 'best');
    title('Quaternion Components');
    grid on;
    xlim([0, t(end)]);
    
    % Quaternion norm (should be ~1)
    subplot(2, 2, 4);
    qNorm = sqrt(sum(est.q.^2, 2));
    plot(t, qNorm, 'k', 'LineWidth', 1.5);
    hold on;
    yline(1, 'r--', 'LineWidth', 1);
    ylabel('Norm');
    xlabel('Time (s)');
    title(sprintf('Quaternion Norm (should be 1) - range: [%.6f, %.6f]', min(qNorm), max(qNorm)));
    grid on;
    xlim([0, t(end)]);
    ylim([0.99, 1.01]);
    
    % Gyro bias (if available)
    if isfield(est, 'b_g')
        figure('Name', 'Gyro Bias Estimation', 'NumberTitle', 'off', ...
               'Position', [150, 300, 800, 300]);
        figs = [figs, gcf];
        
        hold on;
        plot(t, rad2deg(est.b_g(:, 1)), 'Color', colors.x, 'LineWidth', 1.5);
        plot(t, rad2deg(est.b_g(:, 2)), 'Color', colors.y, 'LineWidth', 1.5);
        plot(t, rad2deg(est.b_g(:, 3)), 'Color', colors.z, 'LineWidth', 1.5);
        xlabel('Time (s)');
        ylabel('Bias (deg/s)');
        legend('b_{gx}', 'b_{gy}', 'b_{gz}', 'Location', 'best');
        title('Gyroscope Bias Estimation');
        grid on;
        xlim([0, t(end)]);
    end
    
    %% ========================================================================
    %  FIGURE 3: MOTION ESTIMATION
    %  ========================================================================
    
    if hasMotion
        fig3 = figure('Name', 'Motion Estimation', 'NumberTitle', 'off', ...
                      'Position', [150, 300, 1200, 600]);
        figs = [figs, fig3];
        
        % Velocity
        subplot(2, 2, 1);
        hold on;
        plot(t, motion.v(:, 1), 'Color', colors.x, 'LineWidth', 1.5);
        plot(t, motion.v(:, 2), 'Color', colors.y, 'LineWidth', 1.5);
        plot(t, motion.v(:, 3), 'Color', colors.z, 'LineWidth', 1.5);
        
        % Mark ZUPT corrections
        zupt_idx = find(motion.zupt_flag);
        if ~isempty(zupt_idx)
            scatter(t(zupt_idx), zeros(size(zupt_idx)), 10, 'g', 'filled', 'MarkerFaceAlpha', 0.5);
        end
        
        xlabel('Time (s)');
        ylabel('Velocity (m/s)');
        legend('v_x', 'v_y', 'v_z', 'ZUPT', 'Location', 'best');
        title('Velocity (World Frame)');
        grid on;
        xlim([0, t(end)]);
        
        % Position
        subplot(2, 2, 2);
        hold on;
        plot(t, motion.p(:, 1), 'Color', colors.x, 'LineWidth', 1.5);
        plot(t, motion.p(:, 2), 'Color', colors.y, 'LineWidth', 1.5);
        plot(t, motion.p(:, 3), 'Color', colors.z, 'LineWidth', 1.5);
        xlabel('Time (s)');
        ylabel('Position (m)');
        legend('p_x', 'p_y', 'p_z', 'Location', 'best');
        title('Position (World Frame)');
        grid on;
        xlim([0, t(end)]);
        
        % World-frame acceleration
        subplot(2, 2, 3);
        hold on;
        plot(t, motion.a_world(:, 1), 'Color', colors.x, 'LineWidth', 1);
        plot(t, motion.a_world(:, 2), 'Color', colors.y, 'LineWidth', 1);
        plot(t, motion.a_world(:, 3), 'Color', colors.z, 'LineWidth', 1);
        xlabel('Time (s)');
        ylabel('Acceleration (m/s^2)');
        legend('a_x', 'a_y', 'a_z', 'Location', 'best');
        title('World-Frame Acceleration (gravity removed)');
        grid on;
        xlim([0, t(end)]);
        
        % 3D trajectory
        subplot(2, 2, 4);
        plot3(motion.p(:, 1), motion.p(:, 2), motion.p(:, 3), 'b', 'LineWidth', 1.5);
        hold on;
        scatter3(motion.p(1, 1), motion.p(1, 2), motion.p(1, 3), 100, 'g', 'filled');
        scatter3(motion.p(end, 1), motion.p(end, 2), motion.p(end, 3), 100, 'r', 'filled');
        xlabel('X (m)');
        ylabel('Y (m)');
        zlabel('Z (m)');
        legend('Trajectory', 'Start', 'End', 'Location', 'best');
        title('3D Trajectory');
        grid on;
        axis equal;
        view(30, 30);
        
        sgtitle('Motion Estimation (Linear Kalman Filter)');
    end
    
    %% ========================================================================
    %  FIGURE 4: GESTURE SEGMENTATION
    %  ========================================================================
    
    if hasSeg && isfield(seg, 'energy')
        fig4 = figure('Name', 'Gesture Segmentation', 'NumberTitle', 'off', ...
                      'Position', [200, 250, 1000, 500]);
        figs = [figs, fig4];
        
        subplot(2, 1, 1);
        hold on;
        
        % Motion energy
        plot(t, seg.energy, 'b', 'LineWidth', 1.5);
        
        % Thresholds
        yline(params.segmentation.energy_high, 'r--', 'High', 'LineWidth', 1);
        yline(params.segmentation.energy_low, 'g--', 'Low', 'LineWidth', 1);
        
        % Detected windows
        for w = 1:size(seg.windows, 1)
            xPatch = t([seg.windows(w,1), seg.windows(w,2), seg.windows(w,2), seg.windows(w,1)]);
            yMax = max(seg.energy) * 1.1;
            yPatch = [0, 0, yMax, yMax];
            
            if w == seg.primary
                patch(xPatch, yPatch, [1, 0.8, 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'r', 'LineWidth', 2);
            else
                patch(xPatch, yPatch, colors.seg, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            end
        end
        
        ylabel('Motion Energy');
        title(sprintf('Segmentation (%d windows detected, primary=%d)', ...
                      size(seg.windows, 1), seg.primary));
        legend('Energy', 'Threshold (high)', 'Threshold (low)', 'Location', 'best');
        grid on;
        xlim([0, t(end)]);
        
        % State machine
        subplot(2, 1, 2);
        if isfield(seg, 'state')
            stairs(t, seg.state, 'k', 'LineWidth', 1.5);
            ylabel('State');
            yticks([0, 1]);
            yticklabels({'Quiet', 'Active'});
            ylim([-0.2, 1.2]);
        else
            % Show gyro magnitude instead
            gyrMag = sqrt(sum(imu.gyr.^2, 2));
            plot(t, rad2deg(gyrMag), 'k', 'LineWidth', 1);
            ylabel('Gyro Magnitude (deg/s)');
        end
        xlabel('Time (s)');
        title('Segmentation State Machine');
        grid on;
        xlim([0, t(end)]);
        
        sgtitle('Gesture Segmentation');
    end
    
    %% ========================================================================
    %  FIGURE 5: EKF DIAGNOSTICS
    %  ========================================================================
    
    if hasEKFDiag
        fig5 = figure('Name', 'EKF Diagnostics', 'NumberTitle', 'off', ...
                      'Position', [250, 200, 1000, 600]);
        figs = [figs, fig5];
        
        nPlots = 0;
        
        % Covariance trace
        if isfield(est, 'Ptrace') && ~isempty(est.Ptrace)
            nPlots = nPlots + 1;
            subplot(2, 2, nPlots);
            
            if size(est.Ptrace, 2) >= 7
                hold on;
                % Quaternion uncertainty (first 4 states)
                plot(t, sqrt(est.Ptrace(:, 1)), 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
                plot(t, sqrt(est.Ptrace(:, 2)), 'Color', colors.x, 'LineWidth', 1);
                plot(t, sqrt(est.Ptrace(:, 3)), 'Color', colors.y, 'LineWidth', 1);
                plot(t, sqrt(est.Ptrace(:, 4)), 'Color', colors.z, 'LineWidth', 1);
                legend('\sigma_{qw}', '\sigma_{qx}', '\sigma_{qy}', '\sigma_{qz}', 'Location', 'best');
                title('Quaternion Uncertainty (1\sigma)');
            else
                plot(t, est.Ptrace, 'LineWidth', 1);
                title('Covariance Trace');
            end
            ylabel('Std Dev');
            xlabel('Time (s)');
            grid on;
            xlim([0, t(end)]);
        end
        
        % Gyro bias uncertainty
        if isfield(est, 'Ptrace') && size(est.Ptrace, 2) >= 7
            nPlots = nPlots + 1;
            subplot(2, 2, nPlots);
            hold on;
            plot(t, rad2deg(sqrt(est.Ptrace(:, 5))), 'Color', colors.x, 'LineWidth', 1);
            plot(t, rad2deg(sqrt(est.Ptrace(:, 6))), 'Color', colors.y, 'LineWidth', 1);
            plot(t, rad2deg(sqrt(est.Ptrace(:, 7))), 'Color', colors.z, 'LineWidth', 1);
            legend('\sigma_{bx}', '\sigma_{by}', '\sigma_{bz}', 'Location', 'best');
            ylabel('Std Dev (deg/s)');
            xlabel('Time (s)');
            title('Gyro Bias Uncertainty (1\sigma)');
            grid on;
            xlim([0, t(end)]);
        end
        
        % Accelerometer innovations
        if isfield(est, 'innov_acc') && ~isempty(est.innov_acc)
            nPlots = nPlots + 1;
            subplot(2, 2, nPlots);
            hold on;
            plot(t, est.innov_acc(:, 1), 'Color', colors.x, 'LineWidth', 1);
            plot(t, est.innov_acc(:, 2), 'Color', colors.y, 'LineWidth', 1);
            plot(t, est.innov_acc(:, 3), 'Color', colors.z, 'LineWidth', 1);
            legend('x', 'y', 'z', 'Location', 'best');
            ylabel('Innovation');
            xlabel('Time (s)');
            title('Accelerometer Innovations (should be zero-mean)');
            grid on;
            xlim([0, t(end)]);
        end
        
        % Magnetometer innovations
        if isfield(est, 'innov_mag') && ~isempty(est.innov_mag)
            nPlots = nPlots + 1;
            subplot(2, 2, nPlots);
            hold on;
            plot(t, est.innov_mag(:, 1), 'Color', colors.x, 'LineWidth', 1);
            plot(t, est.innov_mag(:, 2), 'Color', colors.y, 'LineWidth', 1);
            plot(t, est.innov_mag(:, 3), 'Color', colors.z, 'LineWidth', 1);
            legend('x', 'y', 'z', 'Location', 'best');
            ylabel('Innovation');
            xlabel('Time (s)');
            title('Magnetometer Innovations');
            grid on;
            xlim([0, t(end)]);
        end
        
        sgtitle('Extended Kalman Filter Diagnostics');
    end
    
    %% ========================================================================
    %  FIGURE 6: CLASSIFICATION SUMMARY
    %  ========================================================================
    
    if hasCls
        fig6 = figure('Name', 'Classification Result', 'NumberTitle', 'off', ...
                      'Position', [300, 150, 600, 400]);
        figs = [figs, fig6];
        
        % Text summary
        subplot(2, 1, 1);
        axis off;
        
        resultText = sprintf([...
            'CLASSIFICATION RESULT\n\n' ...
            'Gesture: %s\n' ...
            'Confidence: %.1f%%\n' ...
            'Method: %s\n' ...
            'Reason: %s'], ...
            upper(cls.label), cls.score * 100, cls.method, cls.reason);
        
        text(0.5, 0.5, resultText, 'FontSize', 14, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        
        % Key features (if available)
        if hasFeat
            subplot(2, 1, 2);
            
            % Select key features to display
            keyFeatNames = {'duration', 'gyr_rms_total', 'total_rotation_deg', ...
                           'dominant_axis', 'peak_gyr_x', 'peak_gyr_y', 'peak_gyr_z'};
            
            displayNames = {};
            displayValues = [];
            
            for i = 1:length(keyFeatNames)
                fname = keyFeatNames{i};
                if isfield(feat.values, fname)
                    val = feat.values.(fname);
                    if isnumeric(val)
                        displayNames{end+1} = strrep(fname, '_', ' ');
                        displayValues(end+1) = val;
                    end
                end
            end
            
            if ~isempty(displayValues)
                barh(displayValues);
                yticks(1:length(displayNames));
                yticklabels(displayNames);
                xlabel('Value');
                title('Key Features');
                grid on;
            else
                axis off;
                text(0.5, 0.5, 'No numeric features to display', ...
                     'HorizontalAlignment', 'center');
            end
        end
    end
    
    %% ========================================================================
    %  SAVE FIGURES (if enabled)
    %  ========================================================================
    
    if isfield(params, 'viz') && isfield(params.viz, 'saveFigures') && params.viz.saveFigures
        % Determine output directory
        thisFile = mfilename('fullpath');
        [thisDir, ~, ~] = fileparts(thisFile);
        srcDir = fileparts(thisDir);
        repoDir = fileparts(srcDir);
        figDir = fullfile(repoDir, 'outputs', 'figures');
        
        if ~exist(figDir, 'dir')
            mkdir(figDir);
        end
        
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        
        figNames = {'imu_signals', 'attitude', 'motion', 'segmentation', 'ekf_diagnostics', 'classification'};
        
        for i = 1:length(figs)
            if i <= length(figNames)
                fname = sprintf('%s_%s.png', figNames{i}, timestamp);
            else
                fname = sprintf('figure%d_%s.png', i, timestamp);
            end
            
            saveas(figs(i), fullfile(figDir, fname));
            fprintf('Saved: %s\n', fname);
        end
    end
end
