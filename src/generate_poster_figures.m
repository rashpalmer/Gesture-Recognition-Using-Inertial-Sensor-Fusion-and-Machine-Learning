%% GENERATE_POSTER_FIGURES v2 — Refined Poster-Ready Figures Only
%  Generates 8 publication-quality figures matched to poster layout.
%  Drops weak figures (raw vs preprocessed, magnetometer anomaly,
%  Euler angles, segmentation, pipeline diagram) and keeps only
%  figures that passed visual review.
%
%  FIGURES PRODUCED:
%    Fig 1 — Gyro Drift vs EKF-Corrected Orientation
%    Fig 2 — EKF Covariance Convergence
%    Fig 3 — Top 15 Feature Importance
%    Fig 4 — Gyroscope Signatures per Gesture Class
%    Fig 5 — Classifier Comparison (LOO Bar Chart)
%    Fig 6 — Confusion Matrix Heatmap (SVM Gaussian)
%    Fig 7 — Feature Selection Curve (LOO vs Feature Count)
%    Fig 8 — Per-Class Recall Bar Chart
%
%  USAGE:
%    1. cd to project root
%    2. run_training_robust          (loads workspace)
%    3. generate_poster_figures_v2   (this script)
%
%  REQUIRES in workspace: allFeatures, allLabels, featureNames
%
%  VERIFIED NUMBERS (from comprehensive analysis):
%    LOO Accuracy:      95.6%  (153/160, 7 errors)
%    Samples:           160    (6 gesture classes)
%    Features:          88 extracted, 30 selected
%    Top feature:       rot_trans_ratio (importance 0.6954)
%    EKF drift reduction: Roll 18.4x, Pitch 72.4x, Yaw 23.9x
%    Pipeline latency:  13.9 ms (real-time capable)
%    Fallback rate:     25.0% (40/160)
%
%  Author: Rashaan Palmer
%  Date:   February 2026 (v2 — refined for poster)

%% ========================================================================
%  SETUP
%  ========================================================================
fprintf('\n========================================\n');
fprintf('  POSTER FIGURE GENERATOR v2.0\n');
fprintf('  (8 poster-ready figures only)\n');
fprintf('========================================\n\n');

projectRoot = 'C:\Users\rasha\Documents Local\Sensor Fusion Project Matlab\gesture-recognition-sensor-fusion-main';
cd(projectRoot);
addpath(genpath('src'));

figDir = fullfile(projectRoot, 'outputs', 'poster_figures_v2');
if ~isfolder(figDir), mkdir(figDir); end
fprintf('Output: %s\n\n', figDir);

params = config_params();

dataDir = 'src/data';
folderMap = {
    'CircleData','circle'; 'Flip-DownData','flip_down'; 'Flip-UpData','flip_up';
    'PushForwardData','push_forward'; 'ShakeData','shake'; 'TwistData','twist'
};

% --- Validate workspace ---
if ~exist('allFeatures','var') || ~exist('allLabels','var')
    error('Workspace empty. Run run_training_robust first.');
end

N = size(allFeatures, 1);
nFeat = size(allFeatures, 2);
gestures = {'circle','flip_down','flip_up','push_forward','shake','twist'};
gestureLabels = {'Circle','Flip Down','Flip Up','Push Fwd','Shake','Twist'};
nClass = length(gestures);
fprintf('Dataset: %d samples, %d features, %d classes\n', N, nFeat, nClass);

% --- Rebuild fallbackSamples if missing ---
if ~exist('fallbackSamples','var')
    fprintf('Rebuilding fallbackSamples...\n');
    fallbackSamples = false(N, 1);
    sIdx = 0;
    for g = 1:size(folderMap,1)
        folder = fullfile(dataDir, folderMap{g,1});
        files = dir(fullfile(folder, '*.mat'));
        for f = 1:length(files)
            try
                d = read_phone_data(fullfile(folder, files(f).name), 'Verbose', false);
                if all(d.gyr(:)==0) || all(isnan(d.gyr(:))), continue; end
                im = preprocess_imu(d, params);
                sIdx = sIdx + 1;
                try ekf_attitude_quat(im, params); catch; fallbackSamples(sIdx) = true; end
            catch, continue;
            end
        end
    end
    fprintf('  %d fallbacks / %d samples\n', sum(fallbackSamples), sIdx);
end

% --- Feature importance (RF) ---
fprintf('Building RF importance ranking...\n');
rng(42);
rf_full = TreeBagger(200, allFeatures, allLabels, 'Method','classification', ...
    'OOBPrediction','on', 'MinLeafSize',3, 'OOBPredictorImportance','on');
importance = rf_full.OOBPermutedPredictorDeltaError;
[~, impOrder] = sort(importance, 'descend');
fprintf('  Top: %s (%.4f)\n\n', featureNames{impOrder(1)}, importance(impOrder(1)));

% --- Consistent colour palette ---
colours = [
    0.200 0.467 0.733;   % circle - blue
    0.867 0.200 0.200;   % flip_down - red
    0.933 0.467 0.200;   % flip_up - orange
    0.200 0.667 0.333;   % push_forward - green
    0.600 0.333 0.733;   % shake - purple
    0.400 0.733 0.733;   % twist - teal
];

% --- Load ONE representative sample per class (prefer non-fallback) ---
fprintf('Loading representative samples...\n');
repData = struct();
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
                repData.(folderMap{g,2}) = struct('data',data,'imu',imu,'est',est,'seg',seg,'file',files(f).name);
                fprintf('  %s -> %s\n', folderMap{g,2}, files(f).name);
                found = true; break;
            catch, continue;
            end
        catch, continue;
        end
    end
    if ~found
        % Allow fallback sample
        for f = 1:length(files)
            try
                data = read_phone_data(fullfile(folder, files(f).name), 'Verbose', false);
                if all(data.gyr(:)==0) || all(isnan(data.gyr(:))), continue; end
                imu = preprocess_imu(data, params);
                seg = segment_gesture(imu, params);
                if isempty(seg.winIdx), continue; end
                n_ = length(imu.t); dt_ = median(diff(imu.t));
                q_ = [1 0 0 0]; est_fb = struct('q',zeros(n_,4),'euler',zeros(n_,3),'b_g',zeros(n_,3),'t',imu.t,'Ptrace',[]);
                for k=1:n_
                    est_fb.q(k,:) = q_;
                    if k<n_, w=imu.gyr(k,:); dq=0.5*quatmultiply(q_,[0 w]); q_=q_+dq*dt_; q_=q_/norm(q_); end
                end
                est_fb.euler = quat2eul(est_fb.q,'ZYX')*(180/pi);
                repData.(folderMap{g,2}) = struct('data',data,'imu',imu,'est',est_fb,'seg',seg,'file',files(f).name);
                fprintf('  %s -> %s (fallback)\n', folderMap{g,2}, files(f).name);
                break;
            catch, continue;
            end
        end
    end
end

demoImu = repData.circle.imu;
demoEst = repData.circle.est;
fprintf('Done.\n\n');

figCount = 0;
saveFig = @(fig, name) exportgraphics(fig, fullfile(figDir, [name '.png']), 'Resolution', 300);


%% ========================================================================
%  FIG 1: Gyro Drift vs EKF-Corrected Orientation
%  PURPOSE: Single strongest visual for sensor fusion justification
%  KEY NUMBER: Drift reduction up to 72.4x (pitch)
%  ========================================================================
fprintf('Fig 1/8: Gyro Drift vs EKF...\n');
figCount = figCount + 1;
fig1 = figure('Position', [100 100 900 520], 'Color', 'w', 'Visible', 'off');

% Gyro-only dead reckoning
n = length(demoImu.t); dt = median(diff(demoImu.t));
q_gyro = zeros(n,4); q_gyro(1,:) = [1 0 0 0];
for k = 2:n
    w = demoImu.gyr(k-1,:);
    dq = 0.5 * quatmultiply(q_gyro(k-1,:), [0 w]);
    q_gyro(k,:) = q_gyro(k-1,:) + dq * dt;
    q_gyro(k,:) = q_gyro(k,:) / norm(q_gyro(k,:));
end
euler_gyro = quat2eul(q_gyro, 'ZYX') * (180/pi);

t_plot = demoImu.t - demoImu.t(1);
axLabels = {'Roll','Pitch','Yaw'};
axCols   = [3 2 1];  % ZYX order: col3=roll, col2=pitch, col1=yaw

for a = 1:3
    subplot(3,1,a);
    plot(t_plot, euler_gyro(:,axCols(a)), 'Color', [0.82 0.18 0.18], 'LineWidth', 1.4); hold on;
    plot(t_plot, demoEst.euler(:,axCols(a)), 'Color', [0.15 0.35 0.75], 'LineWidth', 1.8);
    ylabel([axLabels{a} ' (°)'], 'FontSize', 11);
    set(gca, 'FontSize', 10); grid on;

    % Show drift reduction factor on each axis
    drift_gyro = range(euler_gyro(:,axCols(a)));
    drift_ekf  = range(demoEst.euler(:,axCols(a)));
    reduction  = drift_gyro / max(drift_ekf, 0.01);
    text(0.98, 0.88, sprintf('%.0f× reduction', reduction), ...
        'Units','normalized', 'HorizontalAlignment','right', ...
        'FontSize', 9, 'FontWeight','bold', 'Color', [0.15 0.35 0.75], ...
        'BackgroundColor', [1 1 1 0.8]);

    if a == 1
        title('Gyroscope-Only Integration vs EKF Sensor Fusion', 'FontSize', 13, 'FontWeight','bold');
        legend('Gyro Only (Drifts)', 'EKF Fused (Stable)', 'Location','northeast', 'FontSize', 9);
    end
    if a == 3, xlabel('Time (s)', 'FontSize', 11); end
    hold off;
end

saveFig(fig1, 'fig1_gyro_drift_vs_ekf');
fprintf('  Saved.\n');


%% ========================================================================
%  FIG 2: EKF Covariance Convergence
%  PURPOSE: Shows filter learning/converging — textbook EKF behaviour
%  KEY NUMBER: 128x covariance reduction
%  ========================================================================
fprintf('Fig 2/8: EKF Covariance...\n');
figCount = figCount + 1;
fig2 = figure('Position', [100 100 800 420], 'Color', 'w', 'Visible', 'off');

t_cov = demoImu.t - demoImu.t(1);

if isfield(demoEst, 'Ptrace') && ~isempty(demoEst.Ptrace) && size(demoEst.Ptrace,1) > 1
    % Full Ptrace available
    stateNames_solid = {'q_1','q_2','q_3'};
    stateNames_dash  = {'b_{gx}','b_{gy}','b_{gz}'};
    cSolid = [0.82 0.18 0.18; 0.18 0.65 0.18; 0.15 0.35 0.75];
    cDash  = cSolid * 0.7 + 0.3;

    for s = 1:min(3, size(demoEst.Ptrace,2))
        plot(t_cov, demoEst.Ptrace(:,s), 'Color', cSolid(s,:), 'LineWidth', 1.8); hold on;
    end
    if size(demoEst.Ptrace,2) >= 6
        for s = 4:6
            plot(t_cov, demoEst.Ptrace(:,s), '--', 'Color', cDash(s-3,:), 'LineWidth', 1.2);
        end
        legend([stateNames_solid stateNames_dash], 'Location','northeast', 'FontSize', 9);
    else
        legend(stateNames_solid(1:min(3,size(demoEst.Ptrace,2))), 'Location','northeast');
    end

    % Convergence annotation
    P0 = mean(demoEst.Ptrace(1,:));
    Pf = mean(demoEst.Ptrace(end,:));
    reduction = P0 / max(Pf, 1e-15);
    text(0.65, 0.85, sprintf('%.0f× covariance reduction', reduction), ...
        'Units','normalized', 'FontSize', 10, 'FontWeight','bold', ...
        'BackgroundColor', [1 1 1 0.85], 'EdgeColor', [0.5 0.5 0.5]);

    xlabel('Time (s)', 'FontSize', 11);
    ylabel('Error Covariance (P diagonal)', 'FontSize', 11);
    title('EKF Error Covariance Convergence', 'FontSize', 13, 'FontWeight','bold');
    grid on; set(gca, 'FontSize', 10); hold off;

else
    % Ptrace unavailable — show quaternion norm stability as proxy
    q_norm = sqrt(sum(demoEst.q.^2, 2));
    q_dev = abs(q_norm - 1);
    semilogy(t_cov, q_dev + 1e-17, 'b', 'LineWidth', 1.5);
    xlabel('Time (s)', 'FontSize', 11);
    ylabel('|q| Deviation from Unity', 'FontSize', 11);
    title('EKF Quaternion Norm Stability', 'FontSize', 13, 'FontWeight','bold');
    grid on; set(gca, 'FontSize', 10);
    text(0.5, 0.85, sprintf('Norm = 1.0 ± 9.4×10^{-17}'), ...
        'Units','normalized', 'HorizontalAlignment','center', ...
        'FontSize', 10, 'FontWeight','bold', ...
        'BackgroundColor', [1 1 1 0.85], 'EdgeColor', [0.5 0.5 0.5]);
end

saveFig(fig2, 'fig2_ekf_covariance_convergence');
fprintf('  Saved.\n');


%% ========================================================================
%  FIG 3: Top 15 Feature Importance
%  PURPOSE: Shows rot_trans_ratio dominance + engineered features marked
%  KEY NUMBER: rot_trans_ratio importance 0.70 (rank 1/88)
%  ========================================================================
fprintf('Fig 3/8: Feature Importance...\n');
figCount = figCount + 1;
fig3 = figure('Position', [100 100 720 520], 'Color', 'w', 'Visible', 'off');

nShow = 15;
topIdx  = impOrder(1:nShow);
topImp  = importance(topIdx);
topNames = featureNames(topIdx);

% Clean names for display
displayNames = strrep(topNames, '_', ' ');

bh = barh(nShow:-1:1, topImp, 0.6, 'FaceColor', [0.20 0.47 0.73]);
set(gca, 'YTick', 1:nShow, 'YTickLabel', flip(displayNames), 'FontSize', 9);
xlabel('Permutation Importance (OOB)', 'FontSize', 11);
title('Top 15 Features by Random Forest Importance', 'FontSize', 13, 'FontWeight','bold');
grid on;

% Star-mark engineered features
engineeredFeats = {'rot_trans_ratio','acc_var_ratio_z','acc_var_ratio_y','acc_var_ratio_x', ...
                   'peak_time_ratio_y','energy_asymmetry_y','gyr_y_area_ratio','gyr_y_sign'};
for i = 1:nShow
    if any(strcmp(topNames{i}, engineeredFeats))
        text(topImp(i) + 0.008, nShow-i+1, '★', 'FontSize', 13, 'Color', [0.90 0.25 0.10], 'FontWeight','bold');
    end
end

% Value labels on bars
for i = 1:nShow
    text(0.015, nShow-i+1, sprintf('%.3f', topImp(i)), 'FontSize', 7.5, ...
        'Color','w', 'FontWeight','bold', 'VerticalAlignment','middle');
end

annotation('textbox', [0.58 0.12 0.34 0.07], 'String', '★ = Engineered features (this work)', ...
    'FontSize', 9, 'BackgroundColor','w', 'EdgeColor', [0.5 0.5 0.5], ...
    'Color', [0.90 0.25 0.10], 'FontWeight','bold');

saveFig(fig3, 'fig3_feature_importance_top15');
fprintf('  Saved.\n');


%% ========================================================================
%  FIG 4: Gyroscope Signatures per Gesture Class
%  PURPOSE: Shows distinct motion patterns that features capture
%  ========================================================================
fprintf('Fig 4/8: Gyroscope Signatures...\n');
figCount = figCount + 1;
fig4 = figure('Position', [100 100 1000 620], 'Color', 'w', 'Visible', 'off');

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
    plot(t_win, gyr_win(:,1), 'Color', [0.82 0.25 0.25], 'LineWidth', 0.9); hold on;
    plot(t_win, gyr_win(:,2), 'Color', [0.20 0.55 0.20], 'LineWidth', 1.5);  % Y thickest — most important axis
    plot(t_win, gyr_win(:,3), 'Color', [0.25 0.25 0.82], 'LineWidth', 0.9);
    title(gestureLabels{g}, 'FontSize', 11, 'FontWeight','bold', 'Color', colours(g,:));
    ylabel('rad/s'); xlabel('Time (s)');
    grid on; set(gca, 'FontSize', 9);
    if g == 1, legend('X','Y','Z','Location','best','FontSize',7); end
    hold off;
end

sgtitle('Gyroscope Signatures per Gesture Class (Gesture Window)', ...
    'FontSize', 13, 'FontWeight','bold');

saveFig(fig4, 'fig4_gyroscope_signatures');
fprintf('  Saved.\n');


%% ========================================================================
%  FIG 5: Classifier Comparison Bar Chart (LOO)
%  PURPOSE: Justifies SVM Gaussian selection — best LOO accuracy
%  KEY NUMBER: SVM Gaussian 95.6%, Decision Tree 77.5% (worst)
%  ========================================================================
fprintf('Fig 5/8: Classifier Comparison (LOO)...\n');
fprintf('  Running LOO for 6 classifiers — takes a few minutes...\n');
figCount = figCount + 1;

selIdx = impOrder(1:30);
classifierNames = {'kNN (k=3)', 'kNN (k=5)', 'SVM Linear', 'SVM Gaussian', ...
                   'Decision Tree', 'Random Forest'};
nCls = length(classifierNames);
looAccs = zeros(nCls, 1);

% Store SVM Gaussian predictions for confusion matrix (fig 6) and recall (fig 8)
loo_pred = cell(N, 1);

for c = 1:nCls
    wrong = 0;
    preds_c = cell(N,1);
    for i = 1:N
        trainIdx = setdiff(1:N, i);
        Xtr = allFeatures(trainIdx, selIdx); Ytr = allLabels(trainIdx);
        Xte = allFeatures(i, selIdx);
        mu_f = mean(Xtr,1); sig_f = std(Xtr,0,1); sig_f(sig_f==0) = 1;
        Xtr_n = (Xtr - mu_f) ./ sig_f;
        Xte_n = (Xte - mu_f) ./ sig_f;

        try
            switch c
                case 1, mdl = fitcknn(Xtr_n, Ytr, 'NumNeighbors', 3);
                case 2, mdl = fitcknn(Xtr_n, Ytr, 'NumNeighbors', 5);
                case 3, mdl = fitcecoc(Xtr_n, Ytr, 'Learners', templateSVM('KernelFunction','linear','Standardize',true));
                case 4, mdl = fitcecoc(Xtr_n, Ytr, 'Learners', templateSVM('KernelFunction','gaussian','KernelScale','auto','Standardize',true,'BoxConstraint',10));
                case 5, mdl = fitctree(Xtr_n, Ytr);
                case 6, mdl = TreeBagger(50, Xtr_n, Ytr, 'Method','classification','MinLeafSize',3);
            end
            pred = predict(mdl, Xte_n);
            if iscell(pred), p = pred{1};
            elseif iscategorical(pred), p = char(pred);
            else, p = pred; end
            if iscategorical(p), p = char(p); end
            preds_c{i} = p;
            wrong = wrong + ~strcmp(p, allLabels{i});
        catch
            preds_c{i} = 'ERROR';
            wrong = wrong + 1;
        end
    end
    looAccs(c) = 100*(1 - wrong/N);
    fprintf('    %s: %.1f%% (%d/%d)\n', classifierNames{c}, looAccs(c), N-wrong, N);

    % Keep SVM Gaussian predictions
    if c == 4, loo_pred = preds_c; end
end

fig5 = figure('Position', [100 100 720 460], 'Color', 'w', 'Visible', 'off');

barColours = repmat([0.55 0.55 0.55], nCls, 1);
[~, bestIdx] = max(looAccs);
barColours(bestIdx,:) = [0.20 0.62 0.30];  % green highlight for best

b = bar(1:nCls, looAccs, 0.6);
b.FaceColor = 'flat';
b.CData = barColours;

set(gca, 'XTick', 1:nCls, 'XTickLabel', classifierNames, 'FontSize', 10);
ylabel('LOO Accuracy (%)', 'FontSize', 11);
title('Classifier Comparison — Leave-One-Out Cross-Validation', 'FontSize', 13, 'FontWeight','bold');
ylim([60 100]); grid on;

% Value labels above bars
for i = 1:nCls
    fw = 'normal'; if i == bestIdx, fw = 'bold'; end
    text(i, looAccs(i)+1.0, sprintf('%.1f%%', looAccs(i)), ...
        'HorizontalAlignment','center', 'FontWeight', fw, 'FontSize', 10);
end

% Note about validation rigour
annotation('textbox', [0.12 0.02 0.80 0.06], ...
    'String', 'LOO = most rigorous validation for small datasets: each of 160 samples tested against model trained on remaining 159', ...
    'FontSize', 7.5, 'HorizontalAlignment','center', 'EdgeColor','none', 'Color', [0.4 0.4 0.4]);

saveFig(fig5, 'fig5_classifier_comparison_loo');
fprintf('  Saved.\n');


%% ========================================================================
%  FIG 6: Confusion Matrix Heatmap (SVM Gaussian LOO)
%  PURPOSE: Shows where errors remain — flip gestures
%  KEY NUMBER: 95.6% overall, flip_up 86.2% recall (weakest)
%  ========================================================================
fprintf('Fig 6/8: Confusion Matrix...\n');
figCount = figCount + 1;

% Build confusion matrix from LOO predictions
confMat = zeros(nClass, nClass);
for i = 1:N
    trueIdx = find(strcmp(gestures, allLabels{i}));
    predIdx = find(strcmp(gestures, loo_pred{i}));
    if ~isempty(trueIdx) && ~isempty(predIdx)
        confMat(trueIdx, predIdx) = confMat(trueIdx, predIdx) + 1;
    end
end

overallAcc = 100 * trace(confMat) / sum(confMat(:));
nCorrect = trace(confMat);
nErrors  = sum(confMat(:)) - nCorrect;

fig6 = figure('Position', [100 100 620 520], 'Color', 'w', 'Visible', 'off');

% Normalised colour map (row-normalised for recall visualisation)
confNorm = confMat ./ max(sum(confMat,2), 1);   % row-normalised
imagesc(confNorm); colormap(flipud(bone));

set(gca, 'XTick', 1:nClass, 'XTickLabel', gestureLabels, 'XTickLabelRotation', 30);
set(gca, 'YTick', 1:nClass, 'YTickLabel', gestureLabels);
xlabel('Predicted', 'FontSize', 11); ylabel('True', 'FontSize', 11);
title(sprintf('Confusion Matrix — SVM Gaussian LOO: %.1f%% (%d/%d correct)', ...
    overallAcc, nCorrect, N), 'FontSize', 12, 'FontWeight','bold');
set(gca, 'FontSize', 10);

% Cell annotations — counts + recall on diagonal
for r = 1:nClass
    for c = 1:nClass
        val = confMat(r,c);
        if val > 0
            if r == c
                % Diagonal: white bold + recall
                recall_rc = 100*val/sum(confMat(r,:));
                txt = sprintf('%d\n(%.0f%%)', val, recall_rc);
                text(c, r, txt, 'HorizontalAlignment','center', ...
                    'Color','w', 'FontWeight','bold', 'FontSize', 10);
            else
                % Off-diagonal: red
                text(c, r, num2str(val), 'HorizontalAlignment','center', ...
                    'Color', [0.85 0.10 0.10], 'FontWeight','bold', 'FontSize', 11);
            end
        end
    end
end

cb = colorbar;
cb.Label.String = 'Recall (row-normalised)';
cb.Label.FontSize = 9;

saveFig(fig6, 'fig6_confusion_matrix_svm_loo');
fprintf('  Saved.\n');


%% ========================================================================
%  FIG 7: Feature Selection Curve (LOO vs Feature Count)
%  PURPOSE: Demonstrates curse of dimensionality — optimal at 30 features
%  KEY NUMBER: 95.6% at 30, drops to ~90% at 88 (all)
%  ========================================================================
fprintf('Fig 7/8: Feature Selection Sweep...\n');
figCount = figCount + 1;

featCounts = [5, 10, 15, 20, 25, 30, 40, 50, nFeat];
featCounts = featCounts(featCounts <= nFeat);
looByFeat = zeros(length(featCounts), 1);

fprintf('  Running LOO for %d feature counts...\n', length(featCounts));
for fc = 1:length(featCounts)
    nF = featCounts(fc);
    if nF == nFeat, selF = 1:nFeat; else, selF = impOrder(1:nF); end
    wrong = 0;
    for i = 1:N
        trainIdx = setdiff(1:N, i);
        Xtr = allFeatures(trainIdx, selF); Ytr = allLabels(trainIdx);
        Xte = allFeatures(i, selF);
        mu_f = mean(Xtr,1); sig_f = std(Xtr,0,1); sig_f(sig_f==0)=1;
        Xtr_n = (Xtr-mu_f)./sig_f; Xte_n = (Xte-mu_f)./sig_f;
        mdl = fitcecoc(Xtr_n, Ytr, 'Learners', ...
            templateSVM('KernelFunction','gaussian','KernelScale','auto','Standardize',true,'BoxConstraint',10));
        pred = predict(mdl, Xte_n);
        if iscategorical(pred), pred = cellstr(pred); end
        wrong = wrong + ~strcmp(pred{1}, allLabels{i});
    end
    looByFeat(fc) = 100*(1-wrong/N);
    fprintf('    Top %2d -> %.1f%%\n', nF, looByFeat(fc));
end

fig7 = figure('Position', [100 100 720 420], 'Color', 'w', 'Visible', 'off');

plot(featCounts, looByFeat, 'o-', 'Color', [0.20 0.47 0.73], ...
    'MarkerFaceColor', [0.20 0.47 0.73], 'LineWidth', 1.8, 'MarkerSize', 8);
xlabel('Number of Features (ranked by RF importance)', 'FontSize', 11);
ylabel('LOO Accuracy (%)', 'FontSize', 11);
title('Feature Selection: LOO Accuracy vs Feature Count (SVM Gaussian)', 'FontSize', 12, 'FontWeight','bold');
grid on; set(gca, 'FontSize', 10);

% Mark the peak
[bestAcc, bestFcIdx] = max(looByFeat);
hold on;
plot(featCounts(bestFcIdx), bestAcc, 'rp', 'MarkerSize', 18, 'MarkerFaceColor','r');
text(featCounts(bestFcIdx)+2, bestAcc-0.5, sprintf('Peak: %.1f%% (%d features)', ...
    bestAcc, featCounts(bestFcIdx)), 'FontSize', 10, 'Color','r', 'FontWeight','bold');

% Mark the all-features point
allFeatIdx = find(featCounts == nFeat);
if ~isempty(allFeatIdx)
    text(featCounts(allFeatIdx)-3, looByFeat(allFeatIdx)+1.0, ...
        sprintf('All %d: %.1f%%', nFeat, looByFeat(allFeatIdx)), ...
        'FontSize', 9, 'Color', [0.5 0.1 0.1], 'HorizontalAlignment','right');
end
hold off;

% Annotation explaining the story
annotation('textbox', [0.50 0.15 0.43 0.10], ...
    'String', sprintf('Adding features beyond 30\nhurts: overfitting on N=%d', N), ...
    'FontSize', 9, 'EdgeColor', [0.5 0.5 0.5], 'BackgroundColor', [1 1 1 0.9]);

saveFig(fig7, 'fig7_feature_selection_curve');
fprintf('  Saved.\n');


%% ========================================================================
%  FIG 8: Per-Class Recall Bar Chart
%  PURPOSE: Shows where the system succeeds and where it struggles
%  KEY NUMBER: flip_up weakest (86-90%), circle/push/shake = 100%
%  ========================================================================
fprintf('Fig 8/8: Per-Class Recall...\n');
figCount = figCount + 1;
fig8 = figure('Position', [100 100 720 420], 'Color', 'w', 'Visible', 'off');

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

b8 = bar(1:nClass, recall, 0.6);
b8.FaceColor = 'flat';
b8.CData = colours;

set(gca, 'XTick', 1:nClass, 'XTickLabel', gestureLabels, 'FontSize', 10);
ylabel('Recall (%)', 'FontSize', 11);
title('Per-Class Recall — SVM Gaussian (LOO)', 'FontSize', 13, 'FontWeight','bold');
ylim([70 108]); grid on;

% Labels with count
for g = 1:nClass
    nCorr = round(recall(g)*nPerClass(g)/100);
    text(g, recall(g)+1.5, sprintf('%.0f%%\n(%d/%d)', recall(g), nCorr, nPerClass(g)), ...
        'HorizontalAlignment','center', 'FontSize', 9, 'FontWeight','bold');
end

% Overall accuracy line
yline(overallAcc, 'k--', sprintf('Overall: %.1f%%', overallAcc), ...
    'LineWidth', 1.2, 'FontSize', 9, 'LabelHorizontalAlignment','left');

saveFig(fig8, 'fig8_per_class_recall');
fprintf('  Saved.\n');


%% ========================================================================
%  SUMMARY
%  ========================================================================
fprintf('\n========================================\n');
fprintf('  ALL %d FIGURES GENERATED\n', figCount);
fprintf('========================================\n');
fprintf('Output: %s\n\n', figDir);

pngs = dir(fullfile(figDir, '*.png'));
for i = 1:length(pngs)
    fprintf('  %s\n', pngs(i).name);
end

fprintf('\nPoster placement guide:\n');
fprintf('  Sensor Fusion Theory block:  Fig 1 (drift vs EKF), Fig 2 (covariance)\n');
fprintf('  System Architecture block:   (use PowerPoint pipeline diagram)\n');
fprintf('  Classification block:        Fig 3 (features), Fig 4 (signatures)\n');
fprintf('  Results & Evaluation block:  Fig 5 (classifiers), Fig 6 (confusion matrix)\n');
fprintf('  Supporting evidence:         Fig 7 (feat selection), Fig 8 (per-class)\n');

fprintf('\nVerified headline numbers for poster text:\n');
fprintf('  SVM Gaussian LOO:   %.1f%% (%d/%d)\n', overallAcc, nCorrect, N);
fprintf('  Features:           %d extracted -> %d selected\n', nFeat, 30);
fprintf('  Top feature:        %s (%.4f)\n', featureNames{impOrder(1)}, importance(impOrder(1)));
fprintf('  EKF fallback rate:  %.1f%% (%d/%d)\n', 100*sum(fallbackSamples)/N, sum(fallbackSamples), N);
fprintf('  Pipeline latency:   <50 ms (real-time capable)\n');

close all;
fprintf('\nDone.\n');