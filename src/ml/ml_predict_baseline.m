%% ML_PREDICT_BASELINE - Predict Gesture Using Trained ML Model
% Loads a trained model and predicts the gesture label for extracted features.
%
% SYNTAX:
%   cls = ml_predict_baseline(feat, params)
%
% INPUTS:
%   feat   - Feature struct from extract_features()
%            .x     : 1xM feature vector
%            .names : 1xM cell array of feature names
%   params - Configuration struct from config_params()
%
% OUTPUTS:
%   cls    - Classification result struct
%            .label  : Predicted gesture name (string)
%            .score  : Confidence score (0-1)
%            .method : "ml"
%            .reason : Explanation of prediction
%            .probs  : Per-class probabilities (if available)
%
% IMPROVEMENTS (v2.0):
%   - FIXED: Handles both naming conventions (peak_gyr_* and gyr_peak_*)
%   - FIXED: Looks for BOTH gesture_model.mat and gesture_model_latest.mat
%   - ADDED: Better error messages and diagnostics
%   - ADDED: Model validation before prediction
%   - OPTIMIZED: Faster feature normalization
%
% NOTES:
%   - Looks for model files in models/ directory
%   - Falls back to rule-based classifier if no model found
%   - Applies same normalization used during training
%
% See also: ml_train_baseline, classify_gesture_rules, extract_features

function cls = ml_predict_baseline(feat, params)
    %% ========================================================================
    %  INITIALIZATION
    %  ========================================================================

    cls = struct();
    cls.label = 'unknown';
    cls.score = 0;
    cls.method = 'ml';
    cls.reason = '';
    cls.probs = [];

    %% ========================================================================
    %  INPUT VALIDATION
    %  ========================================================================

    if nargin < 2
        params = config_params();
    end

    if ~isstruct(feat)
        warning('Invalid feature input: expected struct, got %s', class(feat));
        cls = classify_gesture_rules(feat, params);
        cls.reason = [cls.reason, ' (invalid feature format, used rules)'];
        return;
    end

    if ~isfield(feat, 'x') || isempty(feat.x)
        warning('Feature vector is empty');
        cls = classify_gesture_rules(feat, params);
        cls.reason = [cls.reason, ' (empty features, used rules)'];
        return;
    end

    %% ========================================================================
    %  LOAD MODEL
    %  ========================================================================

    % Find model file - check multiple possible paths and filenames
    thisFile = mfilename('fullpath');
    [thisDir, ~, ~] = fileparts(thisFile);
    srcDir = fileparts(thisDir);
    repoDir = fileparts(srcDir);

    % List of potential model files (in order of preference)
    modelCandidates = {};

    % 1. Custom path from params (highest priority)
    if isfield(params, 'ml') && isfield(params.ml, 'modelFile') && ~isempty(params.ml.modelFile)
        modelCandidates{end+1} = params.ml.modelFile;
    end

    % 2. Standard locations with both filename conventions
    modelCandidates{end+1} = fullfile(repoDir, 'models', 'gesture_model_latest.mat');
    modelCandidates{end+1} = fullfile(repoDir, 'models', 'gesture_model.mat');
    modelCandidates{end+1} = fullfile(thisDir, '..', 'models', 'gesture_model_latest.mat');
    modelCandidates{end+1} = fullfile(thisDir, '..', 'models', 'gesture_model.mat');
    modelCandidates{end+1} = fullfile(pwd, 'models', 'gesture_model_latest.mat');
    modelCandidates{end+1} = fullfile(pwd, 'models', 'gesture_model.mat');

    % Find first existing model file
    modelFile = '';
    for i = 1:length(modelCandidates)
        if exist(modelCandidates{i}, 'file')
            modelFile = modelCandidates{i};
            break;
        end
    end

    if isempty(modelFile)
        warning('ML model not found in any of:\n%s\nFalling back to rule-based classifier.', ...
                strjoin(modelCandidates, '\n'));
        cls = classify_gesture_rules(feat, params);
        cls.reason = [cls.reason, ' (ML model not found, used rules)'];
        return;
    end

    % Load model
    try
        loaded = load(modelFile);
    catch ME
        warning('Failed to load ML model: %s\nFalling back to rule-based classifier.', ME.message);
        cls = classify_gesture_rules(feat, params);
        cls.reason = [cls.reason, ' (ML model load failed, used rules)'];
        return;
    end

    if ~isfield(loaded, 'model')
        warning('Invalid model file format (no "model" field).\nFalling back to rule-based classifier.');
        cls = classify_gesture_rules(feat, params);
        cls.reason = [cls.reason, ' (invalid model format, used rules)'];
        return;
    end

    model = loaded.model;

    %% ========================================================================
    %  PREPARE FEATURES
    %  ========================================================================

    X = feat.x;

    % Ensure row vector
    if size(X, 1) > size(X, 2)
        X = X';
    end

    % Check feature dimensions
    if isfield(loaded, 'featureNames')
        expectedFeatures = length(loaded.featureNames);
        if length(X) ~= expectedFeatures
            % Try to align features by name matching
            if isfield(feat, 'names') && ~isempty(feat.names)
                [X_aligned, alignSuccess] = align_features(X, feat.names, loaded.featureNames);
                if alignSuccess
                    X = X_aligned;
                    % Note: if alignment worked, we continue
                else
                    warning('Feature dimension mismatch: expected %d, got %d.\nFalling back to rule-based classifier.', ...
                            expectedFeatures, length(X));
                    cls = classify_gesture_rules(feat, params);
                    cls.reason = [cls.reason, ' (feature mismatch, used rules)'];
                    return;
                end
            else
                warning('Feature dimension mismatch: expected %d, got %d.\nFalling back to rule-based classifier.', ...
                        expectedFeatures, length(X));
                cls = classify_gesture_rules(feat, params);
                cls.reason = [cls.reason, ' (feature mismatch, used rules)'];
                return;
            end
        end
    end

    % Apply normalization (same as training)
    if isfield(loaded, 'featureMean') && isfield(loaded, 'featureStd')
        % OPTIMIZED: Vectorized normalization
        featureStd = loaded.featureStd;
        featureStd(featureStd == 0) = 1;  % Prevent division by zero
        X_norm = (X - loaded.featureMean) ./ featureStd;
    else
        X_norm = X;
    end

    % Handle NaN/Inf
    nanInf_mask = ~isfinite(X_norm);
    if any(nanInf_mask)
        X_norm(nanInf_mask) = 0;
        % Count for diagnostics
        n_bad = sum(nanInf_mask);
        if n_bad > length(X_norm) * 0.1
            warning('%.1f%% of features are NaN/Inf', 100*n_bad/length(X_norm));
        end
    end

    %% ========================================================================
    %  PREDICT
    %  ========================================================================

    try
        % Check model type and predict accordingly
        if isstruct(model) && isfield(model, 'type')
            % Custom struct-based model (from ml_train_baseline)
            switch model.type
                case 'knn'
                    % Manual kNN prediction
                    distances = sqrt(sum((model.X - X_norm).^2, 2));
                    [~, sortIdx] = sort(distances);
                    kNearest = sortIdx(1:min(model.k, length(sortIdx)));
                    nearestLabels = model.Y(kNearest);

                    % Majority vote
                    uniqueLabels = categories(nearestLabels);
                    if isempty(uniqueLabels)
                        uniqueLabels = unique(cellstr(nearestLabels));
                    end
                    votes = zeros(length(uniqueLabels), 1);
                    for i = 1:length(uniqueLabels)
                        if iscategorical(nearestLabels)
                            votes(i) = sum(nearestLabels == uniqueLabels{i});
                        else
                            votes(i) = sum(strcmp(nearestLabels, uniqueLabels{i}));
                        end
                    end
                    [maxVotes, maxIdx] = max(votes);

                    cls.label = char(uniqueLabels{maxIdx});
                    cls.score = maxVotes / length(kNearest);
                    cls.probs = votes / sum(votes);
                    cls.reason = sprintf('kNN (k=%d): %d/%d neighbors voted %s', ...
                                        model.k, maxVotes, length(kNearest), cls.label);

                case 'tree'
                    % Decision tree (MATLAB fitctree model)
                    [label, scores] = predict(model.classifier, X_norm);
                    cls.label = char(label);
                    cls.score = max(scores);
                    cls.probs = scores;
                    cls.reason = sprintf('Decision tree: %.1f%% confidence', cls.score * 100);

                case 'svm'
                    % SVM (MATLAB fitcsvm or fitcecoc model)
                    [label, scores] = predict(model.classifier, X_norm);
                    cls.label = char(label);
                    if size(scores, 2) > 1
                        % Multi-class: scores are per-class
                        cls.score = max(scores);
                        cls.probs = scores;
                    else
                        % Binary: convert decision value to pseudo-probability
                        cls.score = 1 / (1 + exp(-scores));
                    end
                    cls.reason = sprintf('SVM: %.1f%% confidence', cls.score * 100);

                case 'ensemble'
                    % Random Forest / Ensemble
                    [label, scores] = predict(model.classifier, X_norm);
                    cls.label = char(label);
                    cls.score = max(scores);
                    cls.probs = scores;
                    cls.reason = sprintf('Ensemble (Random Forest): %.1f%% confidence', cls.score * 100);

                otherwise
                    error('Unknown model type: %s', model.type);
            end

        elseif isa(model, 'ClassificationKNN')
            % MATLAB's built-in kNN classifier
            [label, scores, ~] = predict(model, X_norm);
            cls.label = char(label);
            cls.score = max(scores);
            cls.probs = scores;
            cls.reason = sprintf('kNN classifier: %.1f%% posterior', cls.score * 100);

        elseif isa(model, 'ClassificationTree')
            % MATLAB's built-in decision tree
            [label, scores] = predict(model, X_norm);
            cls.label = char(label);
            cls.score = max(scores);
            cls.probs = scores;
            cls.reason = sprintf('Decision tree: %.1f%% confidence', cls.score * 100);

        elseif isa(model, 'ClassificationSVM') || isa(model, 'ClassificationECOC')
            % MATLAB's built-in SVM
            [label, scores] = predict(model, X_norm);
            cls.label = char(label);
            cls.score = max(scores);
            cls.probs = scores;
            cls.reason = sprintf('SVM: %.1f%% confidence', cls.score * 100);

        elseif isa(model, 'ClassificationEnsemble')
            % Ensemble classifier (e.g., Random Forest)
            [label, scores] = predict(model, X_norm);
            cls.label = char(label);
            cls.score = max(scores);
            cls.probs = scores;
            cls.reason = sprintf('Ensemble: %.1f%% confidence', cls.score * 100);

        else
            % Try generic predict
            try
                [label, scores] = predict(model, X_norm);
                cls.label = char(label);
                cls.score = max(scores);
                cls.probs = scores;
                cls.reason = sprintf('ML classifier: %.1f%% confidence', cls.score * 100);
            catch
                error('Unsupported model type: %s', class(model));
            end
        end

    catch ME
        warning('ML prediction failed: %s\nFalling back to rule-based classifier.', ME.message);
        cls = classify_gesture_rules(feat, params);
        cls.reason = [cls.reason, sprintf(' (ML prediction failed: %s)', ME.message)];
        return;
    end

    %% ========================================================================
    %  POST-PROCESSING
    %  ========================================================================

    % Get minimum confidence threshold
    minConfidence = 0.5;  % Default
    if isfield(params, 'ml') && isfield(params.ml, 'minConfidence')
        minConfidence = params.ml.minConfidence;
    end

    % Confidence threshold check
    if cls.score < minConfidence
        originalLabel = cls.label;
        originalScore = cls.score;

        % Fall back to rules for low-confidence predictions
        clsRules = classify_gesture_rules(feat, params);

        if clsRules.score > cls.score
            cls = clsRules;
            cls.reason = sprintf('ML confidence too low (%.1f%% for %s), used rules instead', ...
                                originalScore * 100, originalLabel);
        else
            cls.reason = [cls.reason, sprintf(' (low confidence %.1f%%, but better than rules)', cls.score*100)];
        end
    end

    % Validate label against known gestures
    if isfield(params, 'gestures') && isfield(params.gestures, 'labels')
        knownLabels = params.gestures.labels;
        if ~any(strcmpi(cls.label, knownLabels)) && ~strcmpi(cls.label, 'unknown')
            cls.reason = [cls.reason, ' (warning: label not in known gesture set)'];
        end
    end
end

%% ==================== HELPER FUNCTIONS ====================

function [X_aligned, success] = align_features(X, srcNames, targetNames)
%ALIGN_FEATURES Attempt to align feature vectors by name matching
%   Handles both naming conventions (peak_gyr_* vs gyr_peak_*)

    success = false;
    X_aligned = zeros(1, length(targetNames));

    % Create mapping of alternative names
    altNames = create_alt_name_map();

    for i = 1:length(targetNames)
        targetName = targetNames{i};

        % Direct match
        idx = find(strcmpi(srcNames, targetName), 1);

        if isempty(idx)
            % Try alternative names
            if isKey(altNames, targetName)
                alts = altNames(targetName);
                for j = 1:length(alts)
                    idx = find(strcmpi(srcNames, alts{j}), 1);
                    if ~isempty(idx)
                        break;
                    end
                end
            end
        end

        if ~isempty(idx)
            X_aligned(i) = X(idx);
        else
            % Feature not found - this is okay if it's only a few
            X_aligned(i) = 0;
        end
    end

    % Success if we found at least 80% of features
    matchedCount = sum(X_aligned ~= 0);
    success = (matchedCount >= 0.8 * length(targetNames));
end

function altNames = create_alt_name_map()
%CREATE_ALT_NAME_MAP Create mapping between alternative feature names

    altNames = containers.Map('KeyType', 'char', 'ValueType', 'any');

    % peak_gyr_* <-> gyr_peak_*
    altNames('peak_gyr_x') = {'gyr_peak_x'};
    altNames('peak_gyr_y') = {'gyr_peak_y'};
    altNames('peak_gyr_z') = {'gyr_peak_z'};
    altNames('gyr_peak_x') = {'peak_gyr_x'};
    altNames('gyr_peak_y') = {'peak_gyr_y'};
    altNames('gyr_peak_z') = {'peak_gyr_z'};

    % total_rotation vs total_rotation_deg
    altNames('total_rotation') = {'total_rotation_deg', 'rotation_total'};
    altNames('total_rotation_deg') = {'total_rotation', 'rotation_total_deg'};

    % rms variations
    altNames('gyr_rms_x') = {'rms_gyr_x', 'gyro_rms_x'};
    altNames('gyr_rms_y') = {'rms_gyr_y', 'gyro_rms_y'};
    altNames('gyr_rms_z') = {'rms_gyr_z', 'gyro_rms_z'};
    altNames('acc_rms_x') = {'rms_acc_x', 'accel_rms_x'};
    altNames('acc_rms_y') = {'rms_acc_y', 'accel_rms_y'};
    altNames('acc_rms_z') = {'rms_acc_z', 'accel_rms_z'};
end

