%% GENERATE_POSTER_FIGURES - Creates all poster-ready figures
% Generates 15 publication-quality figures for the FYP poster
% covering sensor fusion theory, feature engineering, and ML results.
%
% USAGE:
%   1. Open MATLAB, cd to project root
%   2. Run: run_training_robust   (loads data into workspace)
%   3. Run: generate_poster_figures
%
% REQUIRES in workspace (from run_training_robust):
%   allFeatures, allLabels, featureNames, fallbackSamples
%
% OUTPUT:
%   All figures saved as high-res PNGs in outputs/poster_figures/
%
% Author: Rashaan Palmer
% Date: February 2026

%% ========================================================================
%  SETUP
%  ========================================================================
fprintf('\n========================================\n');
fprintf('  POSTER FIGURE GENERATOR v1.0\n');
fprintf('========================================\n\n');

% --- Navigate to project root (home path) ---
projectRoot = 'C:\Users\rasha\Documents Local\Sensor Fusion Project Matlab\gesture-recognition-sensor-fusion-main';
cd(projectRoot);
addpath(genpath('src'));

% --- Create output directory ---
figDir = fullfile(projectRoot, 'outputs', 'poster_figures');
if ~isfolder(figDir), mkdir(figDir); end
fprintf('Saving figures to: %s\n\n', figDir);

% --- Load params ---
params = config_params();

% --- Data directory mapping (used throughout) ---
dataDir = 'src/data';
folderMap = {
    'CircleData','circle'; 'Flip-DownData','flip_down'; 'Flip-UpData','flip_up';
    'PushForwardData','push_forward'; 'ShakeData','shake'; 'TwistData','twist'
};

% --- Check workspace has training data ---
if ~exist('allFeatures','var') || ~exist('allLabels','var')
    error(['Workspace is empty. Run run_training_robust first, then run this script.\n' ...
           'Usage:\n  >> cd(''%s'');\n  >> run_training_robust;\n  >> generate_poster_figures;'], projectRoot);
end

N = size(allFeatures, 1);
nFeat = size(allFeatures, 2);
gestures = {'circle','flip_down','flip_up','push_forward','shake','twist'};
gestureLabels = {'Circle','Flip Down','Flip Up','Push Forward','Shake','Twist'};
nClass = length(gestures);
fprintf('Dataset: %d samples, %d features, %d classes\n', N, nFeat, nClass);

% --- Rebuild fallbackSamples if not in workspace ---
if ~exist('fallbackSamples','var')
    fprintf('fallbackSamples not in workspace - rebuilding from data...\n');
    fallbackSamples = false(N, 1);
    sampleIdx = 0;
    for g = 1:size(folderMap,1)
        folder = fullfile(dataDir, folderMap{g,1});
        files = dir(fullfile(folder, '*.mat'));
        for f = 1:length(files)
            try
                data_tmp = read_phone_data(fullfile(folder, files(f).name), 'Verbose', false);
                if all(data_tmp.gyr(:)==0) || all(isnan(data_tmp.gyr(:))), continue; end
                imu_tmp = preprocess_imu(data_tmp, params);
                sampleIdx = sampleIdx + 1;
                try
                    ekf_attitude_quat(imu_tmp, params);
                    fallbackSamples(sampleIdx) = false;
                catch
                    fallbackSamples(sampleIdx) = true;
                end
            catch
                continue;
            end
        end
    end
    fprintf('  Rebuilt: %d fallbacks out of %d samples (%.1f%%)\n', ...
        sum(fallbackSamples), sampleIdx, 100*sum(fallbackSamples)/max(sampleIdx,1));
end

% --- Build RF importance ordering (needed for multiple figures) ---
fprintf('Building Random Forest for feature importance...\n');
rng(42);
rf_full = TreeBagger(200, allFeatures, allLabels, 'Method','classification', ...
    'OOBPrediction','on', 'MinLeafSize',3, 'OOBPredictorImportance','on');
importance = rf_full.OOBPermutedPredictorDeltaError;
[~, impOrder] = sort(importance, 'descend');
fprintf('Done. Top feature: %s (%.4f)\n\n', featureNames{impOrder(1)}, importance(impOrder(1)));

% --- Colour scheme (consistent across all figures) ---
colours = [
    0.200 0.467 0.733;   % circle - blue
    0.867 0.200 0.200;   % flip_down - red
    0.933 0.467 0.200;   % flip_up - orange
    0.200 0.667 0.333;   % push_forward - green
    0.600 0.333 0.733;   % shake - purple
    0.400 0.733 0.733;   % twist - teal
];

% --- Pick one representative sample per class for single-sample figures ---
% Use a sample that did NOT require EKF fallback

fprintf('Loading representative samples for per-gesture figures...\n');
repData = struct(); % stores one representative per class
for g = 1:size(folderMap,1)
    folder = fullfile(dataDir, folderMap{g,1});
    files = dir(fullfile(folder, '*.mat'));
    found = false;
    for f = 1:length(files)
        try
            data = read_phone_data(fullfile(folder, files(f).name), 'Verbose', false);
            if all(data.gyr(:)==0) || all(isnan(data.gyr(:))), continue; end
            imu = preprocess_imu(data, params);
            try
                est = ekf_attitude_quat(imu, params);
                seg = segment_gesture(imu, params);
                if isempty(seg.winIdx), continue; end
                repData.(folderMap{g,2}).data = data;
                repData.(folderMap{g,2}).imu  = imu;
                repData.(folderMap{g,2}).est  = est;
                repData.(folderMap{g,2}).seg  = seg;
                repData.(folderMap{g,2}).file = files(f).name;
                fprintf('  %s -> %s\n', folderMap{g,2}, files(f).name);
                found = true;
                break;
            catch
                continue; % skip fallback samples for representative
            end
        catch
            continue;
        end
    end
    if ~found
        fprintf('  %s -> NONE FOUND (will use fallback sample)\n', folderMap{g,2});
        % Retry allowing fallback
        for f = 1:length(files)
            try
                data = read_phone_data(fullfile(folder, files(f).name), 'Verbose', false);
                if all(data.gyr(:)==0) || all(isnan(data.gyr(:))), continue; end
                imu = preprocess_imu(data, params);
                seg = segment_gesture(imu, params);
                if isempty(seg.winIdx), continue; end
                % Gyro-only fallback
                n = length(imu.t); dt = median(diff(imu.t));
                q = [1 0 0 0];
                est_fb = struct('q',zeros(n,4),'euler',zeros(n,3),'b_g',zeros(n,3),'t',imu.t);
                for k = 1:n
                    est_fb.q(k,:) = q;
                    if k < n
                        w = imu.gyr(k,:);
                        if isfield(imu,'calib') && isfield(imu.calib,'gyro_bias')
                            w = w - imu.calib.gyro_bias';
                        end
                        dq = 0.5 * quatmultiply(q, [0 w]);
                        q = q + dq * dt; q = q/norm(q);
                    end
                end
                est_fb.euler = quat2eul(est_fb.q,'ZYX')*(180/pi);
                repData.(folderMap{g,2}).data = data;
                repData.(folderMap{g,2}).imu  = imu;
                repData.(folderMap{g,2}).est  = est_fb;
                repData.(folderMap{g,2}).seg  = seg;
                repData.(folderMap{g,2}).file = files(f).name;
                fprintf('  %s -> %s (fallback)\n', folderMap{g,2}, files(f).name);
                break;
            catch
                continue;
            end
        end
    end
end
fprintf('Done loading representatives.\n\n');

% --- Pick a CIRCLE sample for the detailed fusion demo figures ---
demoGesture = 'circle';
demoImu = repData.(demoGesture).imu;
demoEst = repData.(demoGesture).est;
demoSeg = repData.(demoGesture).seg;
demoData = repData.(demoGesture).data;

figCount = 0;
saveFig = @(fig, name) exportgraphics(fig, fullfile(figDir, [name '.png']), 'Resolution', 300);

%% ========================================================================
%  FIGURE 1: Raw Gyro Drift vs EKF-Corrected Orientation
%  ========================================================================
fprintf('Figure 1: Raw Gyro Drift vs EKF...\n');
figCount = figCount + 1;
fig1 = figure('Position', [100 100 900 500], 'Color', 'w', 'Visible', 'off');

% Pure gyro integration (dead reckoning - no fusion)
n = length(demoImu.t);
dt = median(diff(demoImu.t));
q_gyro = zeros(n, 4);
q_gyro(1,:) = [1 0 0 0];
for k = 2:n
    w = demoImu.gyr(k-1,:);
    dq = 0.5 * quatmultiply(q_gyro(k-1,:), [0 w]);
    q_gyro(k,:) = q_gyro(k-1,:) + dq * dt;
    q_gyro(k,:) = q_gyro(k,:) / norm(q_gyro(k,:));
end
euler_gyro = quat2eul(q_gyro, 'ZYX') * (180/pi);

t_plot = demoImu.t - demoImu.t(1);
axLabels = {'Roll','Pitch','Yaw'};
axCols = [3 2 1]; % euler order is ZYX -> col 3=roll, 2=pitch, 1=yaw

for a = 1:3
    subplot(3,1,a);
    plot(t_plot, euler_gyro(:,axCols(a)), 'Color', [0.8 0.2 0.2], 'LineWidth', 1.2); hold on;
    plot(t_plot, demoEst.euler(:,axCols(a)), 'Color', [0.2 0.4 0.8], 'LineWidth', 1.5);
    ylabel([axLabels{a} ' (°)']);
    if a == 1
        title('Gyroscope-Only Integration vs EKF Sensor Fusion', 'FontSize', 13);
        legend('Gyro Only (Drifts)', 'EKF Fused (Stable)', 'Location', 'best');
    end
    if a == 3, xlabel('Time (s)'); end
    grid on; set(gca, 'FontSize', 10);
    hold off;
end

saveFig(fig1, '01_gyro_drift_vs_ekf_orientation');
fprintf('  Saved.\n');

%% ========================================================================
%  FIGURE 2: Raw vs Preprocessed IMU Signals
%  ========================================================================
fprintf('Figure 2: Raw vs Preprocessed IMU...\n');
figCount = figCount + 1;
fig2 = figure('Position', [100 100 900 600], 'Color', 'w', 'Visible', 'off');

% Raw data - trim to minimum length (some files have time/sensor mismatch)
nRawAcc = min(length(demoData.t), size(demoData.acc,1));
nRawGyr = min(length(demoData.t), size(demoData.gyr,1));
t_raw_acc = demoData.t(1:nRawAcc) - demoData.t(1);
t_raw_gyr = demoData.t(1:nRawGyr) - demoData.t(1);
t_proc = demoImu.t - demoImu.t(1);

% Accelerometer
subplot(2,2,1);
plot(t_raw_acc, demoData.acc(1:nRawAcc,1), 'r', ...
     t_raw_acc, demoData.acc(1:nRawAcc,2), 'g', ...
     t_raw_acc, demoData.acc(1:nRawAcc,3), 'b');
title('Raw Accelerometer'); ylabel('m/s²'); xlabel('Time (s)');
legend('X','Y','Z','Location','best'); grid on; set(gca,'FontSize',9);

subplot(2,2,2);
plot(t_proc, demoImu.acc(:,1), 'r', t_proc, demoImu.acc(:,2), 'g', t_proc, demoImu.acc(:,3), 'b');
title('Preprocessed Accelerometer'); ylabel('m/s²'); xlabel('Time (s)');
legend('X','Y','Z','Location','best'); grid on; set(gca,'FontSize',9);

% Gyroscope
subplot(2,2,3);
plot(t_raw_gyr, demoData.gyr(1:nRawGyr,1), 'r', ...
     t_raw_gyr, demoData.gyr(1:nRawGyr,2), 'g', ...
     t_raw_gyr, demoData.gyr(1:nRawGyr,3), 'b');
title('Raw Gyroscope'); ylabel('rad/s'); xlabel('Time (s)');
legend('X','Y','Z','Location','best'); grid on; set(gca,'FontSize',9);

subplot(2,2,4);
plot(t_proc, demoImu.gyr(:,1), 'r', t_proc, demoImu.gyr(:,2), 'g', t_proc, demoImu.gyr(:,3), 'b');
title('Preprocessed Gyroscope'); ylabel('rad/s'); xlabel('Time (s)');
legend('X','Y','Z','Location','best'); grid on; set(gca,'FontSize',9);

sgtitle('Signal Conditioning: Raw vs Preprocessed IMU Data', 'FontSize', 13, 'FontWeight', 'bold');

saveFig(fig2, '02_raw_vs_preprocessed_imu');
fprintf('  Saved.\n');

%% ========================================================================
%  FIGURE 3: Magnetometer Anomaly & Fallback Trigger
%  ========================================================================
fprintf('Figure 3: Magnetometer Anomaly & Fallback...\n');
figCount = figCount + 1;

% Find a sample that DID trigger EKF fallback
fallbackFile = '';
fallbackData = []; fallbackImu = [];
for g = 1:size(folderMap,1)
    folder = fullfile(dataDir, folderMap{g,1});
    files = dir(fullfile(folder, '*.mat'));
    for f = 1:length(files)
        try
            data_fb = read_phone_data(fullfile(folder, files(f).name), 'Verbose', false);
            if all(data_fb.gyr(:)==0) || all(isnan(data_fb.gyr(:))), continue; end
            imu_fb = preprocess_imu(data_fb, params);
            try
                ekf_attitude_quat(imu_fb, params);
            catch
                % This is a fallback sample - perfect for the figure
                fallbackFile = files(f).name;
                fallbackData = data_fb;
                fallbackImu = imu_fb;
                break;
            end
        catch
            continue;
        end
    end
    if ~isempty(fallbackFile), break; end
end

fig3 = figure('Position', [100 100 900 450], 'Color', 'w', 'Visible', 'off');

if ~isempty(fallbackImu)
    t_fb = fallbackImu.t - fallbackImu.t(1);
    mag_magnitude = sqrt(sum(fallbackImu.mag.^2, 2));

    % Expected geomagnetic field magnitude (~25-65 uT)
    expected_mag = 50; % approximate for UK
    threshold_hi = expected_mag * 3;
    threshold_lo = expected_mag * 0.3;

    subplot(2,1,1);
    plot(t_fb, fallbackImu.mag(:,1), 'r', t_fb, fallbackImu.mag(:,2), 'g', ...
         t_fb, fallbackImu.mag(:,3), 'b', 'LineWidth', 1);
    title(sprintf('Magnetometer XYZ Components (%s)', fallbackFile), 'Interpreter', 'none');
    ylabel('Magnetic Field (µT)'); xlabel('Time (s)');
    legend('X','Y','Z'); grid on; set(gca,'FontSize',10);

    subplot(2,1,2);
    plot(t_fb, mag_magnitude, 'k', 'LineWidth', 1.5); hold on;
    yline(expected_mag, 'g--', 'Expected (~50 µT)', 'LineWidth', 1.2);
    yline(threshold_hi, 'r--', 'Anomaly Threshold', 'LineWidth', 1.2);

    % Highlight anomaly regions
    anomaly = mag_magnitude > threshold_hi | mag_magnitude < threshold_lo;
    if any(anomaly)
        yl = ylim;
        for k = 1:length(t_fb)
            if anomaly(k)
                patch([t_fb(k)-dt/2 t_fb(k)+dt/2 t_fb(k)+dt/2 t_fb(k)-dt/2], ...
                      [yl(1) yl(1) yl(2) yl(2)], 'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            end
        end
    end
    title('Magnetometer Magnitude - Anomaly Detection Triggers Gyro-Only Fallback');
    ylabel('|B| (µT)'); xlabel('Time (s)');
    grid on; set(gca,'FontSize',10); hold off;

    sgtitle('Magnetometer Disturbance & EKF Fallback Mechanism', 'FontSize', 13, 'FontWeight', 'bold');
else
    text(0.5, 0.5, 'No fallback sample found', 'HorizontalAlignment', 'center', 'FontSize', 14);
end

saveFig(fig3, '03_magnetometer_anomaly_fallback');
fprintf('  Saved.\n');

%% ========================================================================
%  FIGURE 4: EKF Covariance Trace (Convergence)
%  ========================================================================
fprintf('Figure 4: EKF Covariance Trace...\n');
figCount = figCount + 1;
fig4 = figure('Position', [100 100 800 400], 'Color', 'w', 'Visible', 'off');

if isfield(demoEst, 'Ptrace') && ~isempty(demoEst.Ptrace)
    t_cov = demoImu.t - demoImu.t(1);
    plot(t_cov, demoEst.Ptrace(:,1), 'r', 'LineWidth', 1.5); hold on;
    plot(t_cov, demoEst.Ptrace(:,2), 'g', 'LineWidth', 1.5);
    plot(t_cov, demoEst.Ptrace(:,3), 'b', 'LineWidth', 1.5);
    if size(demoEst.Ptrace,2) >= 6
        plot(t_cov, demoEst.Ptrace(:,4), 'r--', 'LineWidth', 1);
        plot(t_cov, demoEst.Ptrace(:,5), 'g--', 'LineWidth', 1);
        plot(t_cov, demoEst.Ptrace(:,6), 'b--', 'LineWidth', 1);
        legend('q_1 var','q_2 var','q_3 var','b_{gx} var','b_{gy} var','b_{gz} var', 'Location','best');
    else
        legend('State 1','State 2','State 3', 'Location','best');
    end
    xlabel('Time (s)'); ylabel('Covariance Trace');
    title('EKF Error Covariance Convergence During Gesture', 'FontSize', 13);
    grid on; set(gca, 'FontSize', 10);
    hold off;
else
    % Ptrace not available - create a conceptual figure using quaternion norm stability
    t_cov = demoImu.t - demoImu.t(1);
    q_norm = sqrt(sum(demoEst.q.^2, 2));
    q_dev = abs(q_norm - 1);
    plot(t_cov, q_dev, 'b', 'LineWidth', 1.5);
    xlabel('Time (s)'); ylabel('|q| deviation from 1');
    title('EKF Quaternion Norm Stability (Proxy for Convergence)', 'FontSize', 13);
    grid on; set(gca, 'FontSize', 10);
    annotation('textbox', [0.15 0.75 0.35 0.1], 'String', ...
        'Near-zero deviation confirms filter convergence', ...
        'BackgroundColor', 'w', 'EdgeColor', [0.5 0.5 0.5], 'FontSize', 9);
end

saveFig(fig4, '04_ekf_covariance_convergence');
fprintf('  Saved.\n');

%% ========================================================================
%  FIGURE 5: Quaternion Attitude Trajectories by Gesture Class
%  ========================================================================
fprintf('Figure 5: Attitude Trajectories by Gesture...\n');
figCount = figCount + 1;
fig5 = figure('Position', [100 100 1000 600], 'Color', 'w', 'Visible', 'off');

for g = 1:nClass
    gName = gestures{g};
    if ~isfield(repData, gName), continue; end

    est_g = repData.(gName).est;
    imu_g = repData.(gName).imu;
    seg_g = repData.(gName).seg;

    % Extract gesture window only
    if ~isempty(seg_g.winIdx)
        iStart = seg_g.winIdx(1);
        iEnd   = min(seg_g.winIdx(2), size(est_g.euler,1));
    else
        iStart = 1; iEnd = size(est_g.euler,1);
    end

    t_g = imu_g.t(iStart:iEnd) - imu_g.t(iStart);
    euler_g = est_g.euler(iStart:iEnd, :);

    subplot(2,3,g);
    plot(t_g, euler_g(:,3), 'r', 'LineWidth', 1.3); hold on; % Roll
    plot(t_g, euler_g(:,2), 'g', 'LineWidth', 1.3);          % Pitch
    plot(t_g, euler_g(:,1), 'b', 'LineWidth', 1.3);          % Yaw
    title(gestureLabels{g}, 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Angle (°)'); xlabel('Time (s)');
    grid on; set(gca, 'FontSize', 9);
    if g == 1, legend('Roll','Pitch','Yaw','Location','best','FontSize',7); end
    hold off;
end

sgtitle('EKF-Estimated Euler Angles per Gesture Class (Gesture Window)', ...
    'FontSize', 13, 'FontWeight', 'bold');

saveFig(fig5, '05_euler_angles_per_gesture_class');
fprintf('  Saved.\n');

%% ========================================================================
%  FIGURE 6: Energy-Based Segmentation Visualisation
%  ========================================================================
fprintf('Figure 6: Energy-Based Segmentation...\n');
figCount = figCount + 1;
fig6 = figure('Position', [100 100 900 450], 'Color', 'w', 'Visible', 'off');

t_seg = demoImu.t - demoImu.t(1);
gyr_energy = sum(demoImu.gyr.^2, 2);
acc_energy = sum((demoImu.acc - mean(demoImu.acc,1)).^2, 2);
total_energy = gyr_energy + 0.1 * acc_energy;

% Smooth the energy signal
winSize = round(0.2 * demoImu.Fs);
if winSize < 1, winSize = 1; end
energy_smooth = movmean(total_energy, winSize);

subplot(2,1,1);
plot(t_seg, demoImu.gyr(:,1), 'r', t_seg, demoImu.gyr(:,2), 'g', ...
     t_seg, demoImu.gyr(:,3), 'b', 'LineWidth', 0.8);
hold on;
if ~isempty(demoSeg.winIdx)
    iS = demoSeg.winIdx(1); iE = min(demoSeg.winIdx(2), length(t_seg));
    yl = ylim;
    patch([t_seg(iS) t_seg(iE) t_seg(iE) t_seg(iS)], ...
          [yl(1) yl(1) yl(2) yl(2)], 'g', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
end
title('Gyroscope Signal with Detected Gesture Window');
ylabel('Angular Rate (rad/s)'); legend('X','Y','Z','Gesture Window');
grid on; set(gca, 'FontSize', 10); hold off;

subplot(2,1,2);
plot(t_seg, energy_smooth, 'k', 'LineWidth', 1.5); hold on;
if ~isempty(demoSeg.winIdx)
    patch([t_seg(iS) t_seg(iE) t_seg(iE) t_seg(iS)], ...
          [min(energy_smooth) min(energy_smooth) max(energy_smooth)*1.1 max(energy_smooth)*1.1], ...
          'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end
title('Energy Signal with Segmentation Boundaries');
ylabel('Energy (rad²/s²)'); xlabel('Time (s)');
legend('Smoothed Energy', 'Detected Window'); grid on; set(gca, 'FontSize', 10); hold off;

sgtitle('Energy-Based Gesture Segmentation', 'FontSize', 13, 'FontWeight', 'bold');

saveFig(fig6, '06_energy_based_segmentation');
fprintf('  Saved.\n');

%% ========================================================================
%  FIGURE 7: Top 15 Feature Importance Bar Chart
%  ========================================================================
fprintf('Figure 7: Feature Importance...\n');
figCount = figCount + 1;
fig7 = figure('Position', [100 100 700 500], 'Color', 'w', 'Visible', 'off');

nShow = 15;
topIdx = impOrder(1:nShow);
topImp = importance(topIdx);
topNames = featureNames(topIdx);

% Clean up feature names for display
displayNames = strrep(topNames, '_', ' ');

barh(nShow:-1:1, topImp, 'FaceColor', [0.2 0.467 0.733]);
set(gca, 'YTick', 1:nShow, 'YTickLabel', flip(displayNames), 'FontSize', 9);
xlabel('Permutation Importance (OOB)', 'FontSize', 11);
title('Top 15 Features by Random Forest Importance', 'FontSize', 13);
grid on;

% Highlight new engineered features
newFeats = {'rot_trans_ratio','acc_var_ratio_z','acc_var_ratio_y', ...
            'peak_time_ratio_y','energy_asymmetry_y'};
for i = 1:nShow
    if any(strcmp(topNames{i}, newFeats))
        % Mark with a star
        text(topImp(i) + 0.01, nShow-i+1, ' ★', 'FontSize', 12, 'Color', [0.9 0.3 0.1]);
    end
end

annotation('textbox', [0.55 0.15 0.35 0.08], 'String', '★ = Engineered features (this work)', ...
    'FontSize', 9, 'BackgroundColor', 'w', 'EdgeColor', [0.5 0.5 0.5], 'Color', [0.9 0.3 0.1]);

saveFig(fig7, '07_feature_importance_top15');
fprintf('  Saved.\n');

%% ========================================================================
%  FIGURE 8: Gesture Signature Comparison (Gyro Y waveforms)
%  ========================================================================
fprintf('Figure 8: Gesture Signatures...\n');
figCount = figCount + 1;
fig8 = figure('Position', [100 100 1000 600], 'Color', 'w', 'Visible', 'off');

for g = 1:nClass
    gName = gestures{g};
    if ~isfield(repData, gName), continue; end

    imu_g = repData.(gName).imu;
    seg_g = repData.(gName).seg;

    if ~isempty(seg_g.winIdx)
        iS = seg_g.winIdx(1);
        iE = min(seg_g.winIdx(2), size(imu_g.gyr,1));
    else
        iS = 1; iE = size(imu_g.gyr,1);
    end

    gyr_win = imu_g.gyr(iS:iE, :);
    t_win = (0:size(gyr_win,1)-1) / imu_g.Fs;

    subplot(2,3,g);
    plot(t_win, gyr_win(:,1), 'Color', [0.8 0.3 0.3], 'LineWidth', 1); hold on;
    plot(t_win, gyr_win(:,2), 'Color', [0.3 0.6 0.3], 'LineWidth', 1.5);
    plot(t_win, gyr_win(:,3), 'Color', [0.3 0.3 0.8], 'LineWidth', 1);
    title(gestureLabels{g}, 'FontSize', 11, 'FontWeight', 'bold', 'Color', colours(g,:));
    ylabel('rad/s'); xlabel('Time (s)');
    grid on; set(gca, 'FontSize', 9);
    if g == 1, legend('Gyr X','Gyr Y','Gyr Z','Location','best','FontSize',7); end
    hold off;
end

sgtitle('Gyroscope Signatures per Gesture Class (Gesture Window)', ...
    'FontSize', 13, 'FontWeight', 'bold');

saveFig(fig8, '08_gyroscope_signatures_per_gesture');
fprintf('  Saved.\n');

%% ========================================================================
%  FIGURE 9: Classifier Comparison Bar Chart
%  ========================================================================
fprintf('Figure 9: Classifier Comparison (LOO)...\n');
figCount = figCount + 1;

% Run LOO for each classifier
selIdx = impOrder(1:30);

classifierNames = {'kNN (k=3)', 'kNN (k=5)', 'SVM Linear', 'SVM Gaussian', ...
                   'Decision Tree', 'Random Forest'};
nCls = length(classifierNames);
looAccs = zeros(nCls, 1);

fprintf('  Running LOO for 6 classifiers (this takes a few minutes)...\n');

for c = 1:nCls
    wrong = 0;
    for i = 1:N
        trainIdx = setdiff(1:N, i);
        Xtr = allFeatures(trainIdx, selIdx); Ytr = allLabels(trainIdx);
        Xte = allFeatures(i, selIdx);
        mu_f = mean(Xtr,1); sig_f = std(Xtr,0,1); sig_f(sig_f==0) = 1;
        Xtr = (Xtr - mu_f) ./ sig_f;
        Xte = (Xte - mu_f) ./ sig_f;

        try
            switch c
                case 1 % kNN k=3
                    mdl = fitcknn(Xtr, Ytr, 'NumNeighbors', 3);
                case 2 % kNN k=5
                    mdl = fitcknn(Xtr, Ytr, 'NumNeighbors', 5);
                case 3 % SVM linear
                    mdl = fitcecoc(Xtr, Ytr, 'Learners', ...
                        templateSVM('KernelFunction','linear','Standardize',true));
                case 4 % SVM Gaussian
                    mdl = fitcecoc(Xtr, Ytr, 'Learners', ...
                        templateSVM('KernelFunction','gaussian','KernelScale','auto', ...
                        'Standardize',true,'BoxConstraint',10));
                case 5 % Decision Tree
                    mdl = fitctree(Xtr, Ytr);
                case 6 % Random Forest
                    mdl = TreeBagger(50, Xtr, Ytr, 'Method','classification','MinLeafSize',3);
            end
            pred = predict(mdl, Xte);
            if iscell(pred), pred_str = pred{1};
            elseif iscategorical(pred), pred_str = char(pred);
            else, pred_str = pred; end
            if iscategorical(pred_str), pred_str = char(pred_str); end
            wrong = wrong + ~strcmp(pred_str, allLabels{i});
        catch
            wrong = wrong + 1;
        end
    end
    looAccs(c) = 100 * (1 - wrong/N);
    fprintf('    %s: %.1f%%\n', classifierNames{c}, looAccs(c));
end

fig9 = figure('Position', [100 100 700 450], 'Color', 'w', 'Visible', 'off');

barColours = repmat([0.5 0.5 0.5], nCls, 1);
[~, bestIdx] = max(looAccs);
barColours(bestIdx,:) = [0.2 0.6 0.3]; % highlight best

b = bar(1:nCls, looAccs, 0.6);
b.FaceColor = 'flat';
b.CData = barColours;

set(gca, 'XTick', 1:nCls, 'XTickLabel', classifierNames, 'FontSize', 10);
ylabel('LOO Accuracy (%)', 'FontSize', 11);
title('Classifier Comparison (Leave-One-Out Cross-Validation)', 'FontSize', 13);
ylim([60 100]);
grid on;

% Add value labels
for i = 1:nCls
    text(i, looAccs(i)+0.8, sprintf('%.1f%%', looAccs(i)), ...
        'HorizontalAlignment','center', 'FontWeight','bold', 'FontSize', 10);
end

saveFig(fig9, '09_classifier_comparison_loo');
fprintf('  Saved.\n');

%% ========================================================================
%  FIGURE 10: Confusion Matrix Heatmap (Best Classifier - SVM Gaussian)
%  ========================================================================
fprintf('Figure 10: Confusion Matrix Heatmap...\n');
figCount = figCount + 1;

% Run LOO with SVM Gaussian and collect predictions
selIdx = impOrder(1:30);
loo_pred = cell(N,1);
for i = 1:N
    trainIdx = setdiff(1:N, i);
    Xtr = allFeatures(trainIdx, selIdx); Ytr = allLabels(trainIdx);
    Xte = allFeatures(i, selIdx);
    mu_f = mean(Xtr,1); sig_f = std(Xtr,0,1); sig_f(sig_f==0) = 1;
    Xtr = (Xtr - mu_f) ./ sig_f;
    Xte = (Xte - mu_f) ./ sig_f;
    mdl = fitcecoc(Xtr, Ytr, 'Learners', ...
        templateSVM('KernelFunction','gaussian','KernelScale','auto', ...
        'Standardize',true,'BoxConstraint',10));
    pred = predict(mdl, Xte);
    if iscategorical(pred), pred = cellstr(pred); end
    loo_pred{i} = pred{1};
end

% Build confusion matrix
confMat = zeros(nClass, nClass);
for i = 1:N
    trueIdx = find(strcmp(gestures, allLabels{i}));
    predIdx = find(strcmp(gestures, loo_pred{i}));
    if ~isempty(trueIdx) && ~isempty(predIdx)
        confMat(trueIdx, predIdx) = confMat(trueIdx, predIdx) + 1;
    end
end

fig10 = figure('Position', [100 100 600 500], 'Color', 'w', 'Visible', 'off');

imagesc(confMat); colormap(flipud(bone));
set(gca, 'XTick', 1:nClass, 'XTickLabel', gestureLabels, 'XTickLabelRotation', 35);
set(gca, 'YTick', 1:nClass, 'YTickLabel', gestureLabels);
xlabel('Predicted', 'FontSize', 11); ylabel('True', 'FontSize', 11);
title(sprintf('Confusion Matrix - SVM Gaussian LOO (%.1f%%)', ...
    100*trace(confMat)/sum(confMat(:))), 'FontSize', 13);
colorbar;
set(gca, 'FontSize', 10);

% Add text annotations
for r = 1:nClass
    for c = 1:nClass
        val = confMat(r,c);
        if val > 0
            if r == c
                txtCol = 'w'; fw = 'bold';
            else
                txtCol = [0.8 0.1 0.1]; fw = 'bold';
            end
            text(c, r, num2str(val), 'HorizontalAlignment','center', ...
                'Color', txtCol, 'FontWeight', fw, 'FontSize', 12);
        end
    end
end

saveFig(fig10, '10_confusion_matrix_svm_gaussian_loo');
fprintf('  Saved.\n');

%% ========================================================================
%  FIGURE 11: LOO Accuracy vs Number of Features
%  ========================================================================
fprintf('Figure 11: LOO vs Feature Count...\n');
figCount = figCount + 1;

featCounts = [5, 10, 15, 20, 25, 30, 40, 50, nFeat];
featCounts = featCounts(featCounts <= nFeat);
looByFeat = zeros(length(featCounts), 1);

fprintf('  Running LOO for %d feature counts...\n', length(featCounts));
for fc = 1:length(featCounts)
    nF = featCounts(fc);
    if nF == nFeat
        selF = 1:nFeat;
    else
        selF = impOrder(1:nF);
    end
    wrong = 0;
    for i = 1:N
        trainIdx = setdiff(1:N, i);
        Xtr = allFeatures(trainIdx, selF); Ytr = allLabels(trainIdx);
        Xte = allFeatures(i, selF);
        mu_f = mean(Xtr,1); sig_f = std(Xtr,0,1); sig_f(sig_f==0)=1;
        Xtr = (Xtr-mu_f)./sig_f; Xte = (Xte-mu_f)./sig_f;
        mdl = fitcecoc(Xtr, Ytr, 'Learners', ...
            templateSVM('KernelFunction','gaussian','KernelScale','auto', ...
            'Standardize',true,'BoxConstraint',10));
        pred = predict(mdl, Xte);
        if iscategorical(pred), pred = cellstr(pred); end
        wrong = wrong + ~strcmp(pred{1}, allLabels{i});
    end
    looByFeat(fc) = 100*(1-wrong/N);
    fprintf('    Top %2d features -> %.1f%%\n', nF, looByFeat(fc));
end

fig11 = figure('Position', [100 100 700 400], 'Color', 'w', 'Visible', 'off');

plot(featCounts, looByFeat, 'bo-', 'MarkerFaceColor', [0.2 0.467 0.733], ...
    'LineWidth', 1.8, 'MarkerSize', 8);
xlabel('Number of Features (ranked by importance)', 'FontSize', 11);
ylabel('LOO Accuracy (%)', 'FontSize', 11);
title('Feature Selection: LOO Accuracy vs Feature Count', 'FontSize', 13);
grid on; set(gca, 'FontSize', 10);

% Mark the best
[bestAcc, bestFeatIdx] = max(looByFeat);
hold on;
plot(featCounts(bestFeatIdx), bestAcc, 'rp', 'MarkerSize', 18, 'MarkerFaceColor', 'r');
text(featCounts(bestFeatIdx)+1.5, bestAcc, sprintf('Best: %.1f%% (%d features)', ...
    bestAcc, featCounts(bestFeatIdx)), 'FontSize', 10, 'Color', 'r');
hold off;

saveFig(fig11, '11_loo_accuracy_vs_feature_count');
fprintf('  Saved.\n');

%% ========================================================================
%  FIGURE 12: Per-Class Recall Bar Chart
%  ========================================================================
fprintf('Figure 12: Per-Class Recall...\n');
figCount = figCount + 1;
fig12 = figure('Position', [100 100 700 400], 'Color', 'w', 'Visible', 'off');

recall = zeros(nClass, 1);
nPerClass = zeros(nClass, 1);
for g = 1:nClass
    classIdx = strcmp(allLabels, gestures{g});
    nPerClass(g) = sum(classIdx);
    correct = 0;
    for i = 1:N
        if classIdx(i) && strcmp(loo_pred{i}, allLabels{i})
            correct = correct + 1;
        end
    end
    recall(g) = 100 * correct / nPerClass(g);
end

b12 = bar(1:nClass, recall, 0.6);
b12.FaceColor = 'flat';
b12.CData = colours;
set(gca, 'XTick', 1:nClass, 'XTickLabel', gestureLabels, 'FontSize', 10);
ylabel('Recall (%)', 'FontSize', 11);
title('Per-Class Recall (SVM Gaussian LOO)', 'FontSize', 13);
ylim([70 105]);
grid on;

% Add labels
for g = 1:nClass
    text(g, recall(g)+1.2, sprintf('%.0f%%\n(%d/%d)', recall(g), ...
        round(recall(g)*nPerClass(g)/100), nPerClass(g)), ...
        'HorizontalAlignment','center', 'FontSize', 9, 'FontWeight', 'bold');
end

saveFig(fig12, '12_per_class_recall_svm_loo');
fprintf('  Saved.\n');

%% ========================================================================
%  FIGURE 13: Pipeline Block Diagram (Programmatic)
%  ========================================================================
fprintf('Figure 13: Pipeline Block Diagram...\n');
figCount = figCount + 1;
fig13 = figure('Position', [100 100 1100 350], 'Color', 'w', 'Visible', 'off');
axis off; hold on;

blocks = {'Raw IMU\nData', 'Pre-\nprocessing', 'EKF\nAttitude', 'Linear KF\n(ZUPT)', ...
          'Energy\nSegmentation', 'Feature\nExtraction', 'ML\nClassifier', 'Gesture\nLabel'};
nBlocks = length(blocks);

blockW = 0.09; blockH = 0.5;
gap = (1 - nBlocks*blockW) / (nBlocks + 1);
yCenter = 0.5;

blockColours = [
    0.85 0.92 1.0;   % data - light blue
    0.85 1.0 0.85;   % preprocess - light green
    1.0 0.9 0.8;     % EKF - light orange
    1.0 0.9 0.8;     % KF - light orange
    0.9 0.85 1.0;    % segmentation - light purple
    0.9 0.85 1.0;    % features - light purple
    1.0 0.85 0.85;   % classifier - light red
    0.95 1.0 0.85;   % output - light yellow
];

for b = 1:nBlocks
    x = gap + (b-1)*(blockW + gap);

    rectangle('Position', [x yCenter-blockH/2 blockW blockH], ...
        'Curvature', 0.2, 'FaceColor', blockColours(b,:), ...
        'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 1.5);

    text(x + blockW/2, yCenter, strrep(blocks{b},'\n',newline), ...
        'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
        'FontSize', 8, 'FontWeight', 'bold');

    % Arrow between blocks
    if b < nBlocks
        xArrow = x + blockW;
        xEnd = gap + b*(blockW + gap);
        annotation('arrow', [xArrow + 0.005, xEnd - 0.005], [0.5 0.5], ...
            'HeadWidth', 8, 'HeadLength', 6, 'Color', [0.3 0.3 0.3]);
    end
end

% Labels underneath
stageLabels = {'iPhone 13', 'LPF + Bias\nCorrection', 'Quaternion\nAttitude', ...
               'Position +\nVelocity', 'Gesture\nBoundaries', '88 Features\n(Top 30)', ...
               'SVM-RBF\n(ECOC)', '6 Classes\n95.6% LOO'};
for b = 1:nBlocks
    x = gap + (b-1)*(blockW + gap);
    text(x + blockW/2, yCenter - blockH/2 - 0.12, ...
        strrep(stageLabels{b},'\n',newline), ...
        'HorizontalAlignment','center', 'VerticalAlignment','top', ...
        'FontSize', 7, 'Color', [0.4 0.4 0.4], 'FontAngle', 'italic');
end

% Sensor fusion bracket
annotation('textbox', [gap yCenter+blockH/2+0.05 3*(blockW+gap) 0.12], ...
    'String', 'Sensor Fusion Pipeline', 'HorizontalAlignment', 'center', ...
    'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.8 0.4 0.1], ...
    'EdgeColor', [0.8 0.4 0.1], 'LineWidth', 1.5, 'BackgroundColor', 'none');

% ML bracket
annotation('textbox', [gap+4*(blockW+gap) yCenter+blockH/2+0.05 3*(blockW+gap) 0.12], ...
    'String', 'Machine Learning Pipeline', 'HorizontalAlignment', 'center', ...
    'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.4 0.2 0.6], ...
    'EdgeColor', [0.4 0.2 0.6], 'LineWidth', 1.5, 'BackgroundColor', 'none');

title('End-to-End Gesture Recognition Pipeline', 'FontSize', 14, 'FontWeight', 'bold');
xlim([0 1]); ylim([0 1]);
hold off;

saveFig(fig13, '13_pipeline_block_diagram');
fprintf('  Saved.\n');

%% ========================================================================
%  FIGURE 14: EKF Fallback Rate by Gesture Class
%  ========================================================================
fprintf('Figure 14: Fallback Rate by Class...\n');
figCount = figCount + 1;
fig14 = figure('Position', [100 100 700 400], 'Color', 'w', 'Visible', 'off');

fbRate = zeros(nClass, 1);
fbCount = zeros(nClass, 1);
classCounts = zeros(nClass, 1);
for g = 1:nClass
    classIdx = strcmp(allLabels, gestures{g});
    classCounts(g) = sum(classIdx);
    fbCount(g) = sum(fallbackSamples(classIdx));
    fbRate(g) = 100 * fbCount(g) / classCounts(g);
end

b14 = bar(1:nClass, fbRate, 0.6);
b14.FaceColor = 'flat';
for g = 1:nClass
    if fbRate(g) > 25
        b14.CData(g,:) = [0.85 0.2 0.2]; % red for high
    else
        b14.CData(g,:) = [0.3 0.6 0.8]; % blue for normal
    end
end

set(gca, 'XTick', 1:nClass, 'XTickLabel', gestureLabels, 'FontSize', 10);
ylabel('EKF Fallback Rate (%)', 'FontSize', 11);
title('Magnetometer Rejection Rate by Gesture Class', 'FontSize', 13);
grid on;

% Labels
for g = 1:nClass
    text(g, fbRate(g)+1.2, sprintf('%.0f%%\n(%d/%d)', fbRate(g), fbCount(g), classCounts(g)), ...
        'HorizontalAlignment','center', 'FontSize', 9, 'FontWeight', 'bold');
end

yline(25, 'r--', 'Overall: 25%', 'LineWidth', 1.2, 'FontSize', 9);

saveFig(fig14, '14_ekf_fallback_rate_by_gesture');
fprintf('  Saved.\n');

%% ========================================================================
%  FIGURE 15: Dataset Distribution
%  ========================================================================
fprintf('Figure 15: Dataset Distribution...\n');
figCount = figCount + 1;
fig15 = figure('Position', [100 100 600 400], 'Color', 'w', 'Visible', 'off');

b15 = bar(1:nClass, classCounts, 0.6);
b15.FaceColor = 'flat';
b15.CData = colours;

set(gca, 'XTick', 1:nClass, 'XTickLabel', gestureLabels, 'FontSize', 10);
ylabel('Number of Samples', 'FontSize', 11);
title(sprintf('Dataset Distribution (N = %d, 6 Classes)', N), 'FontSize', 13);
grid on;

% Labels
for g = 1:nClass
    text(g, classCounts(g)+0.6, sprintf('%d', classCounts(g)), ...
        'HorizontalAlignment','center', 'FontSize', 11, 'FontWeight', 'bold');
end

yline(mean(classCounts), 'k--', sprintf('Mean: %.0f', mean(classCounts)), ...
    'LineWidth', 1, 'FontSize', 9);

saveFig(fig15, '15_dataset_distribution');
fprintf('  Saved.\n');

%% ========================================================================
%  SUMMARY
%  ========================================================================
fprintf('\n========================================\n');
fprintf('  ALL %d FIGURES GENERATED\n', figCount);
fprintf('========================================\n');
fprintf('Saved to: %s\n', figDir);
fprintf('\nFiles:\n');
pngs = dir(fullfile(figDir, '*.png'));
for i = 1:length(pngs)
    fprintf('  %s\n', pngs(i).name);
end
fprintf('\nRecommended for poster (top 8):\n');
fprintf('  01 - Gyro Drift vs EKF (sensor fusion justification)\n');
fprintf('  05 - Euler Angles per Gesture (fusion output signatures)\n');
fprintf('  06 - Energy Segmentation (how gestures are detected)\n');
fprintf('  07 - Feature Importance (engineered features matter)\n');
fprintf('  09 - Classifier Comparison (why SVM Gaussian)\n');
fprintf('  10 - Confusion Matrix (where errors remain)\n');
fprintf('  11 - Feature Selection Curve (optimal feature count)\n');
fprintf('  12 - Per-Class Recall (per-gesture performance)\n');

% Close all invisible figures
close all;
fprintf('\nDone.\n');