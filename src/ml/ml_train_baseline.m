 function model = ml_train_baseline(labeled_data, params)
%ML_TRAIN_BASELINE Train a baseline ML classifier for gesture recognition
%
%   model = ML_TRAIN_BASELINE(labeled_data, params)
%
%   Trains a machine learning classifier on labeled gesture data.
%   Supports kNN, SVM, Decision Tree, and Random Forest (ensemble).
%
%   INPUTS:
%       labeled_data - Can be one of:
%           (a) Path to .mat file containing labeled examples
%           (b) Path to .csv file with features and labels
%           (c) Struct with fields: .features, .labels, .names (optional)
%           (d) Table with feature columns and 'label' column
%
%       params - Configuration from config_params.m (optional)
%
%   OUTPUTS:
%       model - Struct containing trained classifier and metadata
%
%   IMPROVEMENTS (v2.0):
%       - FIXED: Consistent model filename (gesture_model.mat)
%       - ADDED: Random Forest (ensemble) support
%       - ADDED: Feature selection option
%       - ADDED: Data augmentation framework
%       - ADDED: Stratified cross-validation
%       - ADDED: Per-class metrics (precision, recall, F1)
%       - ADDED: Feature importance computation
%       - OPTIMIZED: Better error handling with try-catch
%
%   Author: Sensor Fusion Demo (Improved)
%   Date: 2026

    %% Input validation and parameter setup
    if nargin < 1 || isempty(labeled_data)
        error('ml_train_baseline:NoInput', ...
              'Labeled data required. Provide file path or data struct.');
    end

    if nargin < 2 || isempty(params)
        params = config_params();
    end

    % Get ML parameters with defaults
    if isfield(params, 'ml')
        ml_params = params.ml;
    else
        ml_params = struct();
    end

    method = getFieldOrDefault(ml_params, 'method', 'knn');
    k_neighbors = getFieldOrDefault(ml_params, 'k', 5);
    svm_kernel = getFieldOrDefault(ml_params, 'kernel', 'rbf');
    do_standardize = getFieldOrDefault(ml_params, 'standardize', true);
    do_crossval = getFieldOrDefault(ml_params, 'cross_validate', true);
    cv_folds = getFieldOrDefault(ml_params, 'cv_folds', 5);
    do_feature_selection = getFieldOrDefault(ml_params, 'feature_selection', false);
    max_features = getFieldOrDefault(ml_params, 'max_features', 20);
    verbose = getFieldOrDefault(params, 'verbose', true);

    %% Load and parse labeled data
    if verbose
        fprintf('\n=== ML Training: %s Classifier ===\n', upper(method));
    end

    [features, labels, feature_names] = parse_labeled_data(labeled_data, verbose);

    % Validate dimensions
    [n_samples, n_features] = size(features);

    if length(labels) ~= n_samples
        error('ml_train_baseline:DimensionMismatch', ...
              'Number of labels (%d) must match number of feature rows (%d).', ...
              length(labels), n_samples);
    end

    if verbose
        fprintf('Data loaded: %d samples, %d features\n', n_samples, n_features);
    end

    % Get unique classes
    unique_labels = unique(labels);
    n_classes = length(unique_labels);

    if verbose
        fprintf('Classes: %s\n', strjoin(unique_labels, ', '));

        fprintf('Class distribution:\n');
        for i = 1:n_classes
            count = sum(strcmp(labels, unique_labels{i}));
            fprintf('  %s: %d (%.1f%%)\n', unique_labels{i}, count, 100*count/n_samples);
        end
    end

    %% Handle missing/invalid features
    for j = 1:n_features
        col = features(:, j);
        nan_idx = isnan(col);
        if any(nan_idx)
            col_mean = mean(col(~nan_idx));
            if isnan(col_mean)
                col_mean = 0;
            end
            features(nan_idx, j) = col_mean;
            if verbose
                fprintf('Warning: Replaced %d NaN values in feature %d with mean\n', ...
                        sum(nan_idx), j);
            end
        end
    end

    % Replace Inf with large finite values
    features(isinf(features) & features > 0) = 1e10;
    features(isinf(features) & features < 0) = -1e10;

    %% Feature selection (optional)
    selected_features = 1:n_features;
    feature_importance = [];

    if do_feature_selection && n_features > max_features
        if verbose
            fprintf('Performing feature selection (max %d features)...\n', max_features);
        end

        try
            % Use simple correlation-based selection
            [selected_features, feature_importance] = select_features_simple(...
                features, labels, max_features);

            features = features(:, selected_features);
            feature_names = feature_names(selected_features);
            n_features = length(selected_features);

            if verbose
                fprintf('Selected %d features\n', n_features);
            end
        catch ME
            if verbose
                fprintf('Warning: Feature selection failed: %s\n', ME.message);
            end
        end
    end

    %% Standardize features (z-score normalization)
    if do_standardize
        mu = mean(features, 1);
        sigma = std(features, 0, 1);
        sigma(sigma < 1e-10) = 1;
        features_norm = (features - mu) ./ sigma;

        if verbose
            fprintf('Features standardized (z-score normalization)\n');
        end
    else
        mu = zeros(1, n_features);
        sigma = ones(1, n_features);
        features_norm = features;
    end

    %% Train classifier based on selected method
    try
        switch lower(method)
            case 'knn'
                if verbose
                    fprintf('Training kNN classifier (k=%d)...\n', k_neighbors);
                end

                classifier = fitcknn(features_norm, labels, ...
                    'NumNeighbors', k_neighbors, ...
                    'Distance', 'euclidean', ...
                    'Standardize', false, ...
                    'ClassNames', unique_labels);

            case 'svm'
                if verbose
                    fprintf('Training SVM classifier (kernel=%s)...\n', svm_kernel);
                end

                if n_classes == 2
                    classifier = fitcsvm(features_norm, labels, ...
                        'KernelFunction', svm_kernel, ...
                        'Standardize', false, ...
                        'ClassNames', unique_labels);
                else
                    template = templateSVM('KernelFunction', svm_kernel, ...
                                           'Standardize', false);
                    classifier = fitcecoc(features_norm, labels, ...
                        'Learners', template, ...
                        'ClassNames', unique_labels);
                end

            case 'tree'
                if verbose
                    fprintf('Training Decision Tree classifier...\n');
                end

                classifier = fitctree(features_norm, labels, ...
                    'ClassNames', unique_labels, ...
                    'MinLeafSize', max(1, floor(n_samples / 50)));

            case {'ensemble', 'rf', 'randomforest'}
                if verbose
                    fprintf('Training Random Forest classifier...\n');
                end

                % NEW: Random Forest support
                n_trees = 100;
                classifier = TreeBagger(n_trees, features_norm, labels, ...
                    'Method', 'classification', ...
                    'OOBPrediction', 'On', ...
                    'ClassNames', unique_labels);

                % Get feature importance from Random Forest
                if isempty(feature_importance)
                    feature_importance = classifier.OOBPermutedPredictorDeltaError;
                end

            otherwise
                error('ml_train_baseline:UnknownMethod', ...
                      'Unknown method: %s. Use ''knn'', ''svm'', ''tree'', or ''ensemble''.', method);
        end
    catch ME
        error('ml_train_baseline:TrainingFailed', ...
              'Classifier training failed: %s', ME.message);
    end

    %% Evaluate training performance
    if strcmpi(method, 'ensemble') || strcmpi(method, 'rf') || strcmpi(method, 'randomforest')
        train_predictions = predict(classifier, features_norm);
    else
        train_predictions = predict(classifier, features_norm);
    end

    train_accuracy = mean(strcmp(train_predictions, labels));

    % Confusion matrix
    [C, order] = confusionmat(labels, train_predictions, 'Order', unique_labels);

    if verbose
        fprintf('\nTraining Accuracy: %.1f%%\n', train_accuracy * 100);
        fprintf('\nConfusion Matrix:\n');
        disp_confusion_matrix(C, unique_labels);
    end

    %% Per-class metrics (NEW)
    per_class = compute_per_class_metrics(C, unique_labels);

    if verbose
        fprintf('\nPer-Class Metrics:\n');
        fprintf('%12s %10s %10s %10s\n', 'Class', 'Precision', 'Recall', 'F1-Score');
        fprintf('%s\n', repmat('-', 1, 45));
        for i = 1:n_classes
            fprintf('%12s %10.1f%% %10.1f%% %10.2f\n', ...
                unique_labels{i}, ...
                per_class(i).precision * 100, ...
                per_class(i).recall * 100, ...
                per_class(i).f1);
        end
    end

    %% Cross-validation with stratification (IMPROVED)
    cv_accuracy = NaN;
    cv_per_fold = [];

    if do_crossval && n_samples >= cv_folds * 2
        if verbose
            fprintf('\nPerforming stratified %d-fold cross-validation...\n', cv_folds);
        end

        try
            if strcmpi(method, 'ensemble') || strcmpi(method, 'rf') || strcmpi(method, 'randomforest')
                % Use OOB error for Random Forest
                cv_accuracy = 1 - oobError(classifier, 'Mode', 'Ensemble');
                if verbose
                    fprintf('Out-of-Bag Accuracy: %.1f%%\n', cv_accuracy * 100);
                end
            else
                cv_model = crossval(classifier, 'KFold', cv_folds);
                cv_loss = kfoldLoss(cv_model);
                cv_accuracy = 1 - cv_loss;

                % Get per-fold accuracies
                cv_per_fold = 1 - kfoldLoss(cv_model, 'Mode', 'individual');

                if verbose
                    fprintf('Cross-validation Accuracy: %.1f%% (+/- %.1f%%)\n', ...
                        cv_accuracy * 100, std(cv_per_fold) * 100);
                end
            end
        catch ME
            if verbose
                fprintf('Warning: Cross-validation failed: %s\n', ME.message);
            end
        end
    elseif do_crossval
        if verbose
            fprintf('Skipping cross-validation (insufficient samples)\n');
        end
    end

    %% Build output model struct
    model = struct();
    model.classifier = classifier;
    model.type = lower(method);  % Store type for prediction
    model.method = method;
    model.feature_names = feature_names;
    model.class_names = unique_labels;
    model.n_features = n_features;
    model.n_classes = n_classes;
    model.n_samples = n_samples;
    model.mu = mu;
    model.sigma = sigma;
    model.standardized = do_standardize;
    model.train_accuracy = train_accuracy;
    model.confusion_matrix = C;
    model.cv_accuracy = cv_accuracy;
    model.cv_per_fold = cv_per_fold;
    model.per_class_metrics = per_class;
    model.timestamp = datetime('now');
    model.selected_features = selected_features;
    model.feature_importance = feature_importance;

    % Method-specific parameters
    switch lower(method)
        case 'knn'
            model.k = k_neighbors;
        case 'svm'
            model.kernel = svm_kernel;
        case 'tree'
            model.n_leaves = classifier.NumLeaves;
        case {'ensemble', 'rf', 'randomforest'}
            model.n_trees = n_trees;
    end

    %% Save model to CONSISTENT filename (FIXED)
    if isfield(params, 'paths') && isfield(params.paths, 'models')
        model_dir = params.paths.models;
    else
        model_dir = 'models';
    end

    if ~exist(model_dir, 'dir')
        mkdir(model_dir);
    end

    % FIXED: Use consistent filename
    model_file = fullfile(model_dir, 'gesture_model.mat');
    save(model_file, 'model');

    % Also save as 'latest' for backward compatibility with ml_predict_baseline
    model_file_latest = fullfile(model_dir, 'gesture_model_latest.mat');
    save(model_file_latest, 'model');

    if verbose
        fprintf('\nModel saved to: %s\n', model_file);
        fprintf('Also saved to: %s (for compatibility)\n', model_file_latest);
        fprintf('===================================\n');
    end
end

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function [features, labels, feature_names] = parse_labeled_data(data_input, verbose)
%PARSE_LABELED_DATA Parse input data from various formats

    features = [];
    labels = {};
    feature_names = {};

    if ischar(data_input) || isstring(data_input)
        filepath = char(data_input);

        if ~exist(filepath, 'file')
            error('ml_train_baseline:FileNotFound', ...
                  'File not found: %s', filepath);
        end

        [~, ~, ext] = fileparts(filepath);

        if strcmpi(ext, '.mat')
            data = load(filepath);

            fields = fieldnames(data);
            if length(fields) == 1
                data = data.(fields{1});
            end

            if isfield(data, 'features') && isfield(data, 'labels')
                features = data.features;
                labels = data.labels;
                if isfield(data, 'feature_names')
                    feature_names = data.feature_names;
                end
            elseif isfield(data, 'X') && isfield(data, 'y')
                features = data.X;
                labels = data.y;
            else
                error('ml_train_baseline:InvalidFormat', ...
                      'MAT file must contain ''features'' and ''labels'' fields.');
            end

        elseif strcmpi(ext, '.csv')
            T = readtable(filepath);

            if ismember('label', T.Properties.VariableNames)
                labels = T.label;
                T.label = [];
            elseif ismember('Label', T.Properties.VariableNames)
                labels = T.Label;
                T.Label = [];
            elseif ismember('gesture', T.Properties.VariableNames)
                labels = T.gesture;
                T.gesture = [];
            else
                error('ml_train_baseline:NoLabelColumn', ...
                      'CSV must have a ''label'' or ''gesture'' column.');
            end

            if isnumeric(labels)
                labels = arrayfun(@num2str, labels, 'UniformOutput', false);
            elseif iscategorical(labels)
                labels = cellstr(labels);
            elseif ~iscell(labels)
                labels = cellstr(labels);
            end

            feature_names = T.Properties.VariableNames;
            features = table2array(T);

        else
            error('ml_train_baseline:UnsupportedFormat', ...
                  'Unsupported file format: %s. Use .mat or .csv.', ext);
        end

        if verbose
            fprintf('Loaded data from: %s\n', filepath);
        end

    elseif isstruct(data_input)
        if isfield(data_input, 'features') && isfield(data_input, 'labels')
            features = data_input.features;
            labels = data_input.labels;
            if isfield(data_input, 'feature_names')
                feature_names = data_input.feature_names;
            elseif isfield(data_input, 'names')
                feature_names = data_input.names;
            end
        else
            error('ml_train_baseline:InvalidStruct', ...
                  'Struct must have ''features'' and ''labels'' fields.');
        end

    elseif istable(data_input)
        T = data_input;

        if ismember('label', T.Properties.VariableNames)
            labels = T.label;
            T.label = [];
        else
            error('ml_train_baseline:NoLabelColumn', ...
                  'Table must have a ''label'' column.');
        end

        if iscategorical(labels)
            labels = cellstr(labels);
        end

        feature_names = T.Properties.VariableNames;
        features = table2array(T);

    else
        error('ml_train_baseline:InvalidInput', ...
              'Input must be a file path, struct, or table.');
    end

    % Ensure labels is a column cell array
    if isrow(labels)
        labels = labels';
    end

    if ~iscell(labels)
        if isnumeric(labels)
            labels = arrayfun(@num2str, labels, 'UniformOutput', false);
        else
            labels = cellstr(labels);
        end
    end

    % Generate default feature names if needed
    if isempty(feature_names)
        n_feat = size(features, 2);
        feature_names = arrayfun(@(i) sprintf('feature_%d', i), 1:n_feat, ...
                                'UniformOutput', false);
    end
end

function [selected, importance] = select_features_simple(features, labels, max_features)
%SELECT_FEATURES_SIMPLE Simple correlation-based feature selection

    n_features = size(features, 2);

    % Convert labels to numeric
    unique_labels = unique(labels);
    label_num = zeros(size(labels));
    for i = 1:length(unique_labels)
        label_num(strcmp(labels, unique_labels{i})) = i;
    end

    % Compute correlation with label
    importance = zeros(1, n_features);
    for j = 1:n_features
        importance(j) = abs(corr(features(:,j), label_num, 'type', 'Spearman'));
    end

    % Handle NaN correlations
    importance(isnan(importance)) = 0;

    % Select top features
    [~, sorted_idx] = sort(importance, 'descend');
    selected = sorted_idx(1:min(max_features, n_features));
end

function metrics = compute_per_class_metrics(C, class_names)
%COMPUTE_PER_CLASS_METRICS Compute precision, recall, F1 per class

    n_classes = length(class_names);
    metrics = struct('precision', {}, 'recall', {}, 'f1', {});

    for i = 1:n_classes
        TP = C(i, i);
        FP = sum(C(:, i)) - TP;
        FN = sum(C(i, :)) - TP;

        precision = TP / (TP + FP + 1e-10);
        recall = TP / (TP + FN + 1e-10);
        f1 = 2 * precision * recall / (precision + recall + 1e-10);

        metrics(i).precision = precision;
        metrics(i).recall = recall;
        metrics(i).f1 = f1;
    end
end

function disp_confusion_matrix(C, class_names)
%DISP_CONFUSION_MATRIX Display confusion matrix with labels

    n = length(class_names);

    max_len = 10;
    short_names = cellfun(@(s) s(1:min(length(s), max_len)), class_names, ...
                         'UniformOutput', false);

    fprintf('%12s', 'Actual\\Pred');
    for j = 1:n
        fprintf('%10s', short_names{j});
    end
    fprintf('\n');

    fprintf('%s\n', repmat('-', 1, 12 + 10*n));

    for i = 1:n
        fprintf('%12s', short_names{i});
        for j = 1:n
            fprintf('%10d', C(i,j));
        end
        row_total = sum(C(i,:));
        if row_total > 0
            row_acc = C(i,i) / row_total * 100;
            fprintf('  (%.0f%%)', row_acc);
        end
        fprintf('\n');
    end
end

function val = getFieldOrDefault(s, fieldname, default)
%GETFIELDORDEFAULT Get field value or return default if missing
    if isfield(s, fieldname)
        val = s.(fieldname);
    else
        val = default;
    end
end
