%% RUN_TRAINING - Train ML Gesture Classifier
% Orchestrator script for training the ML-based gesture classifier.
% Loads labeled data, extracts features, and trains/saves a model.
%
% USAGE:
%   run_training           % Uses data from data/labeled/
%   run_training(dataDir)  % Uses specified directory
%
% EXPECTED DATA FORMAT:
%   Each .mat file in the labeled directory should contain:
%   - Sensor data (same format as MATLAB Mobile export)
%   - Variable 'label' (string) with the gesture name
%   OR
%   - Filename pattern: <label>_<number>.mat (e.g., flip_up_001.mat)
%
% See also: ml_train_baseline, extract_features, config_params

function run_training(dataDir)
    %% ========================================================================
    %  INITIALIZATION
    %  ========================================================================
    
    fprintf('\n');
    fprintf('========================================================\n');
    fprintf('  ML GESTURE CLASSIFIER TRAINING\n');
    fprintf('========================================================\n\n');
    
    % Setup paths
    thisFile = mfilename('fullpath');
    [thisDir, ~, ~] = fileparts(thisFile);
    srcDir = fileparts(thisDir);
    repoDir = fileparts(srcDir);
    
    addpath(genpath(srcDir));
    
    % Load configuration
    params = config_params();
    
    % Data directory
    if nargin < 1 || isempty(dataDir)
        dataDir = fullfile(repoDir, 'data', 'labeled');
    end
    
    if ~exist(dataDir, 'dir')
        error('Labeled data directory not found: %s', dataDir);
    end
    
    %% ========================================================================
    %  LOAD AND PROCESS LABELED DATA
    %  ========================================================================
    
    fprintf('[1/4] Loading labeled data from: %s\n', dataDir);
    
    % Find all .mat files
    files = dir(fullfile(dataDir, '*.mat'));
    
    if isempty(files)
        error('No .mat files found in %s\n  Please add labeled gesture recordings.', dataDir);
    end
    
    fprintf('      Found %d files\n', length(files));
    
    % Process each file
    allFeatures = [];
    allLabels = {};
    
    for i = 1:length(files)
        filepath = fullfile(files(i).folder, files(i).name);
        fprintf('      Processing %d/%d: %s\n', i, length(files), files(i).name);
        
        try
            % Load file
            loaded = load(filepath);
            
            % Determine label
            if isfield(loaded, 'label')
                label = loaded.label;
            else
                % Extract from filename (format: label_number.mat)
                [~, fname, ~] = fileparts(files(i).name);
                parts = strsplit(fname, '_');
                if length(parts) >= 2
                    % Join all parts except the last (which is the number)
                    if ~isnan(str2double(parts{end}))
                        label = strjoin(parts(1:end-1), '_');
                    else
                        label = fname;
                    end
                else
                    label = fname;
                end
            end
            
            % Read sensor data
            data = read_phone_data(filepath);
            
            % Preprocess
            imu = preprocess_imu(data, params);
            
            % Run EKF
            est = ekf_attitude_quat(imu, params);
            
            % Segment gesture
            seg = segment_gesture(imu, params);
            
            if isempty(seg.windows)
                fprintf('        WARNING: No gesture detected, skipping\n');
                continue;
            end
            
            % Extract features
            feat = extract_features(imu, est, seg, params);
            
            % Store
            allFeatures = [allFeatures; feat.x];
            allLabels{end+1} = label;
            
        catch ME
            fprintf('        ERROR: %s\n', ME.message);
            continue;
        end
    end
    
    if isempty(allFeatures)
        error('No valid gesture samples extracted.');
    end
    
    fprintf('\n      Extracted %d samples with %d features\n', ...
            size(allFeatures, 1), size(allFeatures, 2));
    
    % Label summary
    uniqueLabels = unique(allLabels);
    fprintf('      Labels: ');
    for i = 1:length(uniqueLabels)
        count = sum(strcmp(allLabels, uniqueLabels{i}));
        fprintf('%s(%d) ', uniqueLabels{i}, count);
    end
    fprintf('\n');
    
    %% ========================================================================
    %  PREPARE TRAINING DATA
    %  ========================================================================
    
    fprintf('\n[2/4] Preparing training data...\n');
    
    X = allFeatures;
    Y = categorical(allLabels);
    
    % Feature names (from last extraction)
    featureNames = feat.names;
    
    % Check for NaN/Inf
    badRows = any(~isfinite(X), 2);
    if any(badRows)
        fprintf('      WARNING: Removing %d rows with NaN/Inf values\n', sum(badRows));
        X = X(~badRows, :);
        Y = Y(~badRows);
    end
    
    % Normalize features
    featureMean = mean(X, 1);
    featureStd = std(X, 0, 1);
    featureStd(featureStd == 0) = 1;  % Prevent division by zero
    X_norm = (X - featureMean) ./ featureStd;
    
    fprintf('      Training samples: %d\n', size(X_norm, 1));
    fprintf('      Features: %d\n', size(X_norm, 2));
    
    %% ========================================================================
    %  TRAIN MODEL
    %  ========================================================================
    
    fprintf('\n[3/4] Training classifier...\n');
    fprintf('      Method: %s\n', params.ml.method);
    
    trainData = struct();
    trainData.X = X_norm;
    trainData.Y = Y;
    trainData.featureNames = featureNames;
    trainData.featureMean = featureMean;
    trainData.featureStd = featureStd;
    
    model = ml_train_baseline(trainData, params);
    
    %% ========================================================================
    %  SAVE MODEL
    %  ========================================================================
    
    fprintf('\n[4/4] Saving model...\n');
    
    modelDir = fullfile(repoDir, 'models');
    if ~exist(modelDir, 'dir')
        mkdir(modelDir);
    end
    
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    modelFile = fullfile(modelDir, sprintf('gesture_model_%s.mat', timestamp));
    
    % Also save as 'latest' for easy loading
    latestFile = fullfile(modelDir, 'gesture_model_latest.mat');
    
    save(modelFile, 'model', 'featureNames', 'featureMean', 'featureStd', 'params');
    save(latestFile, 'model', 'featureNames', 'featureMean', 'featureStd', 'params');
    
    fprintf('      Saved: %s\n', modelFile);
    fprintf('      Saved: %s (latest)\n', latestFile);
    
    fprintf('\n========================================================\n');
    fprintf('  TRAINING COMPLETE\n');
    fprintf('  Model accuracy (training): %.1f%%\n', model.accuracy * 100);
    fprintf('========================================================\n\n');
end
