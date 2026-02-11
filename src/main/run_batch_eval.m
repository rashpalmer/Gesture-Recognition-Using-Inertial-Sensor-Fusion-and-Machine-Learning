%% RUN_BATCH_EVAL - Batch Evaluation of Gesture Recognition Pipeline
% Runs the gesture recognition pipeline on multiple files and generates
% a summary report.
%
% USAGE:
%   run_batch_eval           % Processes files in data/raw/
%   run_batch_eval(dataDir)  % Processes files in specified directory
%   run_batch_eval(dataDir, outputDir)  % Custom output location
%
% OUTPUTS:
%   - Summary table printed to console
%   - Detailed results saved to outputs/logs/batch_eval_<timestamp>.mat
%   - Optional: Confusion matrix if ground truth labels available
%
% See also: main_gesture_demo, config_params

function results = run_batch_eval(dataDir, outputDir)
    %% ========================================================================
    %  INITIALIZATION
    %  ========================================================================
    
    fprintf('\n');
    fprintf('========================================================\n');
    fprintf('  BATCH GESTURE EVALUATION\n');
    fprintf('========================================================\n\n');
    
    % Setup paths
    thisFile = mfilename('fullpath');
    [thisDir, ~, ~] = fileparts(thisFile);
    srcDir = fileparts(thisDir);
    repoDir = fileparts(srcDir);
    
    addpath(genpath(srcDir));
    
    % Directories
    if nargin < 1 || isempty(dataDir)
        dataDir = fullfile(repoDir, 'data', 'raw');
    end
    
    if nargin < 2 || isempty(outputDir)
        outputDir = fullfile(repoDir, 'outputs', 'logs');
    end
    
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    % Load configuration
    params = config_params();
    params.viz.enabled = false;  % Disable plotting for batch mode
    
    %% ========================================================================
    %  FIND FILES
    %  ========================================================================
    
    fprintf('[1/3] Scanning for data files...\n');
    
    matFiles = dir(fullfile(dataDir, '*.mat'));
    csvFiles = dir(fullfile(dataDir, '*.csv'));
    allFiles = [matFiles; csvFiles];
    
    if isempty(allFiles)
        fprintf('      No data files found in: %s\n', dataDir);
        fprintf('      Looking for .mat and .csv files.\n');
        results = [];
        return;
    end
    
    fprintf('      Found %d files\n', length(allFiles));
    
    %% ========================================================================
    %  PROCESS FILES
    %  ========================================================================
    
    fprintf('\n[2/3] Processing files...\n\n');
    
    results = struct();
    results.files = {};
    results.labels = {};
    results.scores = [];
    results.durations = [];
    results.success = [];
    results.groundTruth = {};
    results.errors = {};
    
    for i = 1:length(allFiles)
        filepath = fullfile(allFiles(i).folder, allFiles(i).name);
        fprintf('  [%d/%d] %s\n', i, length(allFiles), allFiles(i).name);
        
        results.files{end+1} = allFiles(i).name;
        
        try
            tic;
            
            % Read data
            data = read_phone_data(filepath);
            
            % Check for ground truth label in filename
            [~, fname, ~] = fileparts(allFiles(i).name);
            parts = strsplit(fname, '_');
            if length(parts) >= 2 && ~isnan(str2double(parts{end}))
                groundTruth = strjoin(parts(1:end-1), '_');
            else
                groundTruth = '';
            end
            results.groundTruth{end+1} = groundTruth;
            
            % Preprocess
            imu = preprocess_imu(data, params);
            
            % Attitude estimation
            if strcmpi(params.fusion.method, 'ekf')
                est = ekf_attitude_quat(imu, params);
            else
                est = complementary_filter(imu, params);
            end
            
            % Motion estimation
            motion = kf_linear_motion(imu, est, params);
            
            % Segmentation
            seg = segment_gesture(imu, params);
            
            if isempty(seg.windows)
                fprintf('         -> No gesture detected\n');
                results.labels{end+1} = 'none';
                results.scores(end+1) = 0;
                results.durations(end+1) = toc;
                results.success(end+1) = false;
                results.errors{end+1} = 'No gesture detected';
                continue;
            end
            
            % Feature extraction
            feat = extract_features(imu, est, seg, params);
            
            % Classification
            if strcmpi(params.classifier.method, 'rules')
                cls = classify_gesture_rules(feat, params);
            else
                cls = ml_predict_baseline(feat, params);
            end
            
            elapsed = toc;
            
            results.labels{end+1} = cls.label;
            results.scores(end+1) = cls.score;
            results.durations(end+1) = elapsed;
            results.success(end+1) = true;
            results.errors{end+1} = '';
            
            fprintf('         -> %s (%.0f%%) in %.2fs\n', ...
                    cls.label, cls.score * 100, elapsed);
            
        catch ME
            fprintf('         -> ERROR: %s\n', ME.message);
            results.labels{end+1} = 'error';
            results.scores(end+1) = 0;
            results.durations(end+1) = toc;
            results.success(end+1) = false;
            results.errors{end+1} = ME.message;
        end
    end
    
    %% ========================================================================
    %  SUMMARY
    %  ========================================================================
    
    fprintf('\n[3/3] Generating summary...\n\n');
    
    fprintf('========================================================\n');
    fprintf('  BATCH EVALUATION SUMMARY\n');
    fprintf('========================================================\n\n');
    
    fprintf('Total files: %d\n', length(results.files));
    fprintf('Successful: %d (%.1f%%)\n', sum(results.success), ...
            100 * sum(results.success) / length(results.files));
    fprintf('Failed: %d\n', sum(~results.success));
    fprintf('Average processing time: %.2f seconds\n', mean(results.durations));
    fprintf('Average confidence: %.1f%%\n', mean(results.scores(results.success)) * 100);
    
    % Label distribution
    fprintf('\nClassification Distribution:\n');
    uniqueLabels = unique(results.labels);
    for i = 1:length(uniqueLabels)
        count = sum(strcmp(results.labels, uniqueLabels{i}));
        fprintf('  %s: %d (%.1f%%)\n', uniqueLabels{i}, count, ...
                100 * count / length(results.labels));
    end
    
    % Accuracy if ground truth available
    hasGroundTruth = ~cellfun(@isempty, results.groundTruth);
    if any(hasGroundTruth)
        correct = strcmp(results.labels(hasGroundTruth), results.groundTruth(hasGroundTruth));
        accuracy = sum(correct) / sum(hasGroundTruth);
        fprintf('\nAccuracy (vs ground truth): %.1f%% (%d/%d)\n', ...
                100 * accuracy, sum(correct), sum(hasGroundTruth));
        
        % Confusion matrix
        fprintf('\nConfusion Matrix:\n');
        allLabelsGT = unique([results.labels(hasGroundTruth), results.groundTruth(hasGroundTruth)]);
        nLabels = length(allLabelsGT);
        confMat = zeros(nLabels);
        
        gtLabels = results.groundTruth(hasGroundTruth);
        predLabels = results.labels(hasGroundTruth);
        
        for i = 1:length(gtLabels)
            gtIdx = find(strcmp(allLabelsGT, gtLabels{i}));
            predIdx = find(strcmp(allLabelsGT, predLabels{i}));
            if ~isempty(gtIdx) && ~isempty(predIdx)
                confMat(gtIdx, predIdx) = confMat(gtIdx, predIdx) + 1;
            end
        end
        
        % Print confusion matrix
        fprintf('  GT\\Pred');
        for j = 1:nLabels
            fprintf('\t%s', allLabelsGT{j}(1:min(6, length(allLabelsGT{j}))));
        end
        fprintf('\n');
        for i = 1:nLabels
            fprintf('  %s', allLabelsGT{i}(1:min(6, length(allLabelsGT{i}))));
            for j = 1:nLabels
                fprintf('\t%d', confMat(i, j));
            end
            fprintf('\n');
        end
        
        results.confusionMatrix = confMat;
        results.confusionLabels = allLabelsGT;
        results.accuracy = accuracy;
    end
    
    fprintf('\n========================================================\n\n');
    
    %% ========================================================================
    %  SAVE RESULTS
    %  ========================================================================
    
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    resultsFile = fullfile(outputDir, sprintf('batch_eval_%s.mat', timestamp));
    
    save(resultsFile, 'results', 'params');
    fprintf('Results saved: %s\n\n', resultsFile);
end
