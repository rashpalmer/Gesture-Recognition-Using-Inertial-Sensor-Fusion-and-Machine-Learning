%% run_training_robust.m (v2.0)
% Optimized ML training pipeline with:
%   - k=3 kNN (reduced from k=5 for small dataset)
%   - Multi-classifier comparison (kNN, Random Forest, SVM)
%   - Feature correlation analysis
%   - Gyro-only EKF fallback for bad mag data
%
% Save to: src/main/run_training_robust.m

clear allFeatures allLabels featureNames

% Fix working directory (handles run() from subdirectory)
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));  % up from src/main to root
cd(projectRoot);

%% ==================== CONFIGURATION ====================
dataDir = 'src/data';
folderMap = {
    'CircleData',       'circle';
    'Flip-DownData',    'flip_down';
    'Flip-UpData',      'flip_up';
    'PushForwardData',  'push_forward';
    'ShakeData',        'shake';
    'TwistData',        'twist'
};

params = config_params ();
params.verbose = false;

% Training options
KNN_K = 3;                    % Reduced from 5 for better local decisions
COMPARE_CLASSIFIERS = true;   % Compare kNN, RF, SVM
REMOVE_CORRELATED = false;    % Remove features with >0.95 correlation
CORRELATION_THRESHOLD = 0.95;

%% ==================== LOAD DATA ====================
allFeatures = [];
allLabels = {};
featureNames = {};
skippedFiles = {};
loadedCount = 0;
ekfFallbackCount = 0;
fallbackByGesture = struct();

fprintf('=== Loading Gesture Data (v2.0 with Gyro-Only Fallback) ===\n\n');

for g = 1:size(folderMap, 1)
    folderName = folderMap{g,1};
    gestureLabel = folderMap{g,2};
    folder = fullfile(dataDir, folderName);
    files = dir(fullfile(folder, '*.mat'));
    
    fallbackByGesture.(gestureLabel) = 0;
    
    fprintf('[%s] %d files: ', upper(gestureLabel), length(files));
    
    for f = 1:length(files)
        filepath = fullfile(folder, files(f).name);
        try
            data = read_phone_data(filepath, 'Verbose', false);
            
            if all(data.gyr(:) == 0) || all(isnan(data.gyr(:)))
                skippedFiles{end+1} = [files(f).name ' (no gyro)'];
                fprintf('S');
                continue;
            end
            
            imu = preprocess_imu(data, params);
            
            % TRY EKF, FALLBACK TO GYRO-ONLY IF MAG DATA IS BAD
            try
                est = ekf_attitude_quat(imu, params);
                fprintf('.');
            catch ME
                ekfFallbackCount = ekfFallbackCount + 1;
                fallbackByGesture.(gestureLabel) = fallbackByGesture.(gestureLabel) + 1;
                
                est = gyro_only_attitude(imu, params);
                fprintf('F');
            end
            
            seg = segment_gesture(imu, params);
            feat = extract_features(imu, est, seg, params);
            
            if isempty(featureNames)
                featureNames = feat.names;
            end
            allFeatures(end+1, :) = feat.x;
            allLabels{end+1} = gestureLabel;
            loadedCount = loadedCount + 1;
            
        catch ME
            skippedFiles{end+1} = [files(f).name ' (' ME.message ')'];
            fprintf('X');
        end
    end
    fprintf('\n');
end

%% ==================== LOADING SUMMARY ====================
fprintf('\n=== LOADING SUMMARY ===\n');
fprintf('Total loaded: %d samples\n', loadedCount);
fprintf('Features per sample: %d\n', length(featureNames));
fprintf('EKF fallbacks: %d (%.1f%%)\n', ekfFallbackCount, 100*ekfFallbackCount/loadedCount);
fprintf('Skipped: %d files\n', length(skippedFiles));

fprintf('\nPer-class distribution:\n');
for g = 1:size(folderMap, 1)
    gesture = folderMap{g,2};
    count = sum(strcmp(allLabels, gesture));
    fallbacks = fallbackByGesture.(gesture);
    fprintf('  %12s: %3d samples (%d fallbacks, %.0f%%)\n', ...
        gesture, count, fallbacks, 100*fallbacks/max(count,1));
end

if loadedCount < 10
    error('Not enough data loaded for training.');
end

%% ==================== FEATURE ANALYSIS ====================
fprintf('\n=== FEATURE ANALYSIS ===\n');

% Check for constant features
feature_std = std(allFeatures, 0, 1);
constant_features = find(feature_std < 1e-10);
if ~isempty(constant_features)
    fprintf('WARNING: %d constant features detected (will cause issues)\n', length(constant_features));
    for i = 1:min(5, length(constant_features))
        fprintf('  - %s\n', featureNames{constant_features(i)});
    end
end

% Correlation analysis
if REMOVE_CORRELATED
    fprintf('\nRemoving highly correlated features (threshold: %.2f)...\n', CORRELATION_THRESHOLD);
    corrMatrix = corrcoef(allFeatures);
    corrMatrix(isnan(corrMatrix)) = 0;
    
    % Find highly correlated pairs
    [row, col] = find(triu(abs(corrMatrix), 1) > CORRELATION_THRESHOLD);
    featuresToRemove = unique(col);  % Remove the second of each pair
    
    if ~isempty(featuresToRemove)
        fprintf('Removing %d correlated features:\n', length(featuresToRemove));
        for i = 1:min(10, length(featuresToRemove))
            fprintf('  - %s\n', featureNames{featuresToRemove(i)});
        end
        if length(featuresToRemove) > 10
            fprintf('  ... and %d more\n', length(featuresToRemove) - 10);
        end
        
        % Remove features
        keepIdx = setdiff(1:length(featureNames), featuresToRemove);
        allFeatures = allFeatures(:, keepIdx);
        featureNames = featureNames(keepIdx);
        fprintf('Features reduced: %d → %d\n', length(keepIdx) + length(featuresToRemove), length(keepIdx));
    else
        fprintf('No highly correlated features found.\n');
    end
end

%% ==================== TRAIN/TEST SPLIT ====================
allLabels = allLabels(:);
rng(42);  % Reproducibility

labelsCat = categorical(allLabels);
cv = cvpartition(labelsCat, 'HoldOut', 0.2);
trainIdx = find(cv.training);
testIdx = find(cv.test);

X_train_raw = allFeatures(trainIdx, :);
Y_train = allLabels(trainIdx);
X_test_raw = allFeatures(testIdx, :);
Y_test = allLabels(testIdx);

fprintf('\n=== TRAIN/TEST SPLIT ===\n');
fprintf('Training: %d samples\n', length(trainIdx));
fprintf('Testing:  %d samples\n', length(testIdx));

% Normalize
mu = mean(X_train_raw, 1);
sigma = std(X_train_raw, 0, 1);
sigma(sigma == 0) = 1;

X_train = (X_train_raw - mu) ./ sigma;
X_test = (X_test_raw - mu) ./ sigma;

%% ==================== CROSS-VALIDATION ====================
fprintf('\n=== CROSS-VALIDATION (5-fold) ===\n');

cvModel = fitcknn(X_train, Y_train, 'NumNeighbors', KNN_K, ...
    'Distance', 'euclidean', 'Standardize', false);
cvResults = crossval(cvModel, 'KFold', 5);
cvLoss = kfoldLoss(cvResults);
cvAcc = (1 - cvLoss) * 100;

% Get per-fold accuracy for variance
foldLosses = zeros(5, 1);
for fold = 1:5
    foldLosses(fold) = kfoldLoss(cvResults, 'Folds', fold);
end
foldAccs = (1 - foldLosses) * 100;
cvStd = std(foldAccs);

fprintf('kNN (k=%d) CV Accuracy: %.1f%% ± %.1f%%\n', KNN_K, cvAcc, cvStd);

%% ==================== TRAIN PRIMARY MODEL (kNN k=3) ====================
fprintf('\n=== TRAINING PRIMARY MODEL (kNN k=%d) ===\n', KNN_K);

model_knn = fitcknn(X_train, Y_train, ...
    'NumNeighbors', KNN_K, ...
    'Distance', 'euclidean', ...
    'Standardize', false);

% Training accuracy
train_pred = predict(model_knn, X_train);
if iscategorical(train_pred), train_pred = cellstr(train_pred); end
trainAcc = 100 * sum(strcmp(train_pred, Y_train)) / length(Y_train);
fprintf('Training Accuracy: %.1f%%\n', trainAcc);

% Test accuracy
test_pred_knn = predict(model_knn, X_test);
if iscategorical(test_pred_knn), test_pred_knn = cellstr(test_pred_knn); end
testAcc_knn = 100 * sum(strcmp(test_pred_knn, Y_test)) / length(Y_test);

%% ==================== COMPARE CLASSIFIERS ====================
if COMPARE_CLASSIFIERS
    fprintf('\n=== CLASSIFIER COMPARISON ===\n');
    fprintf('%20s  %10s  %10s\n', 'Classifier', 'Train Acc', 'Test Acc');
    fprintf('%s\n', repmat('-', 1, 44));
    
    % kNN k=3 (already trained)
    fprintf('%20s  %9.1f%%  %9.1f%%\n', sprintf('kNN (k=%d)', KNN_K), trainAcc, testAcc_knn);
    
    % kNN k=5 (for comparison)
    model_knn5 = fitcknn(X_train, Y_train, 'NumNeighbors', 5, 'Distance', 'euclidean');
    pred_knn5_train = predict(model_knn5, X_train);
    pred_knn5_test = predict(model_knn5, X_test);
    if iscategorical(pred_knn5_train), pred_knn5_train = cellstr(pred_knn5_train); end
    if iscategorical(pred_knn5_test), pred_knn5_test = cellstr(pred_knn5_test); end
    acc_knn5_train = 100 * sum(strcmp(pred_knn5_train, Y_train)) / length(Y_train);
    acc_knn5_test = 100 * sum(strcmp(pred_knn5_test, Y_test)) / length(Y_test);
    fprintf('%20s  %9.1f%%  %9.1f%%\n', 'kNN (k=5)', acc_knn5_train, acc_knn5_test);
    
    % Random Forest
    try
        model_rf = TreeBagger(50, X_train, Y_train, 'Method', 'classification', ...
            'OOBPrediction', 'on', 'MinLeafSize', 3);
        pred_rf_train = predict(model_rf, X_train);
        pred_rf_test = predict(model_rf, X_test);
        acc_rf_train = 100 * sum(strcmp(pred_rf_train, Y_train)) / length(Y_train);
        acc_rf_test = 100 * sum(strcmp(pred_rf_test, Y_test)) / length(Y_test);
        fprintf('%20s  %9.1f%%  %9.1f%%\n', 'Random Forest (50)', acc_rf_train, acc_rf_test);
        
        % Feature importance from RF
        importance = model_rf.OOBPermutedPredictorDeltaError;
        [sortedImp, sortIdx] = sort(importance, 'descend');
        fprintf('\nTop 10 features (RF importance):\n');
        for i = 1:min(10, length(sortIdx))
            fprintf('  %2d. %s (%.3f)\n', i, featureNames{sortIdx(i)}, sortedImp(i));
        end
    catch ME
        fprintf('%20s  %s\n', 'Random Forest', 'FAILED');
    end
    
    % SVM (One-vs-All)
    try
        model_svm = fitcecoc(X_train, Y_train, 'Learners', 'linear');
        pred_svm_train = predict(model_svm, X_train);
        pred_svm_test = predict(model_svm, X_test);
        if iscategorical(pred_svm_train), pred_svm_train = cellstr(pred_svm_train); end
        if iscategorical(pred_svm_test), pred_svm_test = cellstr(pred_svm_test); end
        acc_svm_train = 100 * sum(strcmp(pred_svm_train, Y_train)) / length(Y_train);
        acc_svm_test = 100 * sum(strcmp(pred_svm_test, Y_test)) / length(Y_test);
        fprintf('%20s  %9.1f%%  %9.1f%%\n', 'SVM (linear)', acc_svm_train, acc_svm_test);
    catch ME
        fprintf('%20s  %s\n', 'SVM', 'FAILED');
    end
    
    % Decision Tree
    try
        model_tree = fitctree(X_train, Y_train, 'MinLeafSize', 5);
        pred_tree_train = predict(model_tree, X_train);
        pred_tree_test = predict(model_tree, X_test);
        if iscategorical(pred_tree_train), pred_tree_train = cellstr(pred_tree_train); end
        if iscategorical(pred_tree_test), pred_tree_test = cellstr(pred_tree_test); end
        acc_tree_train = 100 * sum(strcmp(pred_tree_train, Y_train)) / length(Y_train);
        acc_tree_test = 100 * sum(strcmp(pred_tree_test, Y_test)) / length(Y_test);
        fprintf('%20s  %9.1f%%  %9.1f%%\n', 'Decision Tree', acc_tree_train, acc_tree_test);
    catch ME
        fprintf('%20s  %s\n', 'Decision Tree', 'FAILED');
    end
end

%% ==================== PRIMARY MODEL RESULTS ====================
fprintf('\n');
fprintf('========================================\n');
fprintf('*** PRIMARY: kNN (k=%d) ***\n', KNN_K);
fprintf('*** TEST ACCURACY: %.1f%% (%d/%d) ***\n', testAcc_knn, ...
    sum(strcmp(test_pred_knn, Y_test)), length(Y_test));
fprintf('*** CV ACCURACY: %.1f%% ± %.1f%% ***\n', cvAcc, cvStd);
fprintf('========================================\n');

%% ==================== PER-CLASS BREAKDOWN ====================
fprintf('\nPer-class test results (kNN k=%d):\n', KNN_K);
fprintf('  %12s  %8s  %8s\n', 'Gesture', 'Correct', 'Accuracy');
fprintf('  %s\n', repmat('-', 1, 32));

gestures = folderMap(:,2);
for g = 1:length(gestures)
    gesture = gestures{g};
    mask = strcmp(Y_test, gesture);
    if sum(mask) > 0
        classCorrect = sum(strcmp(test_pred_knn(mask), Y_test(mask)));
        classTotal = sum(mask);
        classAcc = 100 * classCorrect / classTotal;
        fprintf('  %12s  %4d/%-3d  %5.0f%%\n', gesture, classCorrect, classTotal, classAcc);
    end
end

%% ==================== CONFUSION MATRIX ====================
fprintf('\nTest Confusion Matrix:\n');
testConfMat = zeros(6);
for i = 1:length(Y_test)
    trueIdx = find(strcmp(gestures, Y_test{i}));
    predIdx = find(strcmp(gestures, test_pred_knn{i}));
    if ~isempty(trueIdx) && ~isempty(predIdx)
        testConfMat(trueIdx, predIdx) = testConfMat(trueIdx, predIdx) + 1;
    end
end

fprintf('%12s', 'True\\Pred');
for g = 1:6
    fprintf('%8s', gestures{g}(1:min(7,end)));
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 60));

for i = 1:6
    fprintf('%12s', gestures{i});
    for j = 1:6
        if i == j
            fprintf('%8d', testConfMat(i,j));
        elseif testConfMat(i,j) > 0
            fprintf('   [%d]  ', testConfMat(i,j));  % Highlight errors
        else
            fprintf('%8d', 0);
        end
    end
    fprintf('\n');
end

%% ==================== ERROR ANALYSIS ====================
fprintf('\nMisclassification details:\n');
errorCount = 0;
for i = 1:length(Y_test)
    if ~strcmp(test_pred_knn{i}, Y_test{i})
        errorCount = errorCount + 1;
        fprintf('  %d. True: %s → Predicted: %s\n', errorCount, Y_test{i}, test_pred_knn{i});
    end
end
if errorCount == 0
    fprintf('  (No errors!)\n');
end

%% ==================== SAVE MODEL ====================
model = struct();
model.classifier = model_knn;
model.mu = mu;
model.sigma = sigma;
model.feature_names = featureNames;
model.k = KNN_K;
model.trainAcc = trainAcc;
model.testAcc = testAcc_knn;
model.cvAcc = cvAcc;
model.cvStd = cvStd;
model.gestures = gestures;
model.confusionMatrix = testConfMat;
model.version = '2.0';
model.timestamp = datetime('now');

save('models/gesture_model.mat', 'model');
fprintf('\nModel saved to models/gesture_model.mat\n');

% Save full results
results = struct();
results.accuracy = testAcc_knn;
results.predictions = test_pred_knn;
results.testLabels = Y_test;
results.confusionMatrix = testConfMat;
results.fallbackCount = ekfFallbackCount;
results.fallbackByGesture = fallbackByGesture;
results.featureCount = length(featureNames);
results.timestamp = datetime('now');

save('models/training_results.mat', 'results');
fprintf('Results saved to models/training_results.mat\n');

fprintf('\n=== TRAINING COMPLETE ===\n');