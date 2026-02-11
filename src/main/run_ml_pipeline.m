%% COMPLETE ML TRAINING & EVALUATION PIPELINE
% Run from: gesture-recognition-sensor-fusion-main folder

clear; clc; close all;

%% 1. Setup
cd('C:\Users\rasha\Documents Local\Sensor Fusion Project\gesture-recognition-sensor-fusion-main');
addpath(genpath('src'));

params = config_params();
params.verbose = false;  % Reduce console noise

%% 2. Define gesture folders and labels
gestureFolders = {
    'data/CircleData',       'circle'
    'data/Flip-UpData',      'flip_up'
    'data/Flip-DownData',    'flip_down'
    'data/ShakeData',        'shake'
    'data/TwistData',        'twist'
    'data/PushForwardData',  'push_forward'
};

%% 3. Process all files and extract features
allFeatures = [];
allLabels = {};
featureNames = {};
fileList = {};

fprintf('========================================\n');
fprintf('  LOADING & PROCESSING GESTURE DATA\n');
fprintf('========================================\n\n');

for g = 1:size(gestureFolders, 1)
    folder = gestureFolders{g, 1};
    label = gestureFolders{g, 2};
    
    % Find all .mat files in folder
    files = dir(fullfile(folder, '*.mat'));
    
    fprintf('[%s] Found %d files\n', upper(label), length(files));
    
    for i = 1:length(files)
        filepath = fullfile(folder, files(i).name);
        
        try
            % Load and process
            data = convert_mobile_data(filepath);
            imu = preprocess_imu(data, params);
            
            % EKF with gyro-only fallback
            try
                est = ekf_attitude_quat(imu, params);
            catch
                est = gyro_only_attitude(imu, params);
            end
            
            seg = segment_gesture(imu, params);
            feat = extract_features(imu, est, seg, params);
            
            % Store results
            allFeatures(end+1, :) = feat.x;
            allLabels{end+1, 1} = label;
            fileList{end+1, 1} = files(i).name;
            
            if isempty(featureNames)
                featureNames = feat.names;
            end
            
        catch ME
            fprintf('  ERROR in %s: %s\n', files(i).name, ME.message);
        end
    end
end

%% 4. Dataset Summary
fprintf('\n========================================\n');
fprintf('  DATASET SUMMARY\n');
fprintf('========================================\n');
fprintf('Total samples: %d\n', size(allFeatures, 1));
fprintf('Features per sample: %d\n', size(allFeatures, 2));
fprintf('\nClass distribution:\n');

uniqueLabels = unique(allLabels);
for i = 1:length(uniqueLabels)
    count = sum(strcmp(allLabels, uniqueLabels{i}));
    fprintf('  %-15s: %3d samples\n', uniqueLabels{i}, count);
end

%% 5. Check if we have enough data
if length(uniqueLabels) < 2
    error('Need at least 2 gesture types to train classifier! You only have: %s', ...
          strjoin(uniqueLabels, ', '));
end

if size(allFeatures, 1) < 10
    error('Need at least 10 samples total. You only have %d.', size(allFeatures, 1));
end

%% 6. Split into Train/Test (80/20)
fprintf('\n========================================\n');
fprintf('  SPLITTING DATA (80%% train, 20%% test)\n');
fprintf('========================================\n');

rng(42);  % For reproducibility

nSamples = size(allFeatures, 1);
shuffleIdx = randperm(nSamples);

splitPoint = round(0.8 * nSamples);
trainIdx = shuffleIdx(1:splitPoint);
testIdx = shuffleIdx(splitPoint+1:end);

X_train = allFeatures(trainIdx, :);
y_train = allLabels(trainIdx);
X_test = allFeatures(testIdx, :);
y_test = allLabels(testIdx);

fprintf('Training samples: %d\n', length(trainIdx));
fprintf('Test samples: %d\n', length(testIdx));

%% 7. Train ML Model
fprintf('\n========================================\n');
fprintf('  TRAINING ML MODEL\n');
fprintf('========================================\n');

% Prepare training data struct
trainingData.features = X_train;
trainingData.labels = y_train;
trainingData.feature_names = featureNames;

% Train (uses kNN by default, can change in config_params)
params.verbose = true;
model = ml_train_baseline(trainingData, params);

%% 8. Evaluate on Test Set
fprintf('\n========================================\n');
fprintf('  EVALUATING ON TEST SET\n');
fprintf('========================================\n');

% Predict each test sample
predictions = {};
scores = [];

for i = 1:length(testIdx)
    % Create feature struct for prediction
    feat_test.x = X_test(i, :);
    feat_test.names = featureNames;
    feat_test.values = struct();  % Empty, not needed for ML
    
    % Predict
    cls = ml_predict_baseline(feat_test, params);
    predictions{i, 1} = cls.label;
    scores(i, 1) = cls.score;
end

% Calculate accuracy
correct = strcmp(predictions, y_test);
accuracy = mean(correct) * 100;

fprintf('\n*** TEST ACCURACY: %.1f%% ***\n', accuracy);
fprintf('Correct: %d / %d\n', sum(correct), length(correct));

%% 9. Confusion Matrix
fprintf('\n--- Confusion Matrix ---\n');
fprintf('%15s', 'Actual\\Pred');
for j = 1:length(uniqueLabels)
    fprintf('%12s', uniqueLabels{j}(1:min(10,end)));
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 15 + 12*length(uniqueLabels)));

for i = 1:length(uniqueLabels)
    fprintf('%15s', uniqueLabels{i});
    for j = 1:length(uniqueLabels)
        count = sum(strcmp(y_test, uniqueLabels{i}) & strcmp(predictions, uniqueLabels{j}));
        fprintf('%12d', count);
    end
    fprintf('\n');
end

%% 10. Show Misclassified Samples
fprintf('\n--- Misclassified Samples ---\n');
if any(~correct)
    wrongIdx = find(~correct);
    for i = 1:length(wrongIdx)
        idx = wrongIdx(i);
        origFileIdx = testIdx(idx);
        fprintf('  %s: Actual=%s, Predicted=%s (%.0f%%)\n', ...
                fileList{origFileIdx}, y_test{idx}, predictions{idx}, scores(idx)*100);
    end
else
    fprintf('  None! Perfect classification.\n');
end

%% 11. Save Results
fprintf('\n========================================\n');
fprintf('  SAVING RESULTS\n');
fprintf('========================================\n');

results.accuracy = accuracy;
results.predictions = predictions;
results.actual = y_test;
results.confusion = confusionmat(y_test, predictions);
results.featureNames = featureNames;
results.trainSize = length(trainIdx);
results.testSize = length(testIdx);
results.date = datetime('now');

save('outputs/evaluation_results.mat', 'results');
fprintf('Results saved to: outputs/evaluation_results.mat\n');

%% 12. Quick Test - Classify a Single New File
fprintf('\n========================================\n');
fprintf('  QUICK TEST: Single File Classification\n');
fprintf('========================================\n');

% Pick a random test file
testFile = fullfile(gestureFolders{1,1}, dir(fullfile(gestureFolders{1,1},'*.mat')).name);
fprintf('Testing: %s\n', testFile);

data = convert_mobile_data(testFile);
imu = preprocess_imu(data, params);
try
    est = ekf_attitude_quat(imu, params);
catch
    est = gyro_only_attitude(imu, params);
end
seg = segment_gesture(imu, params);
feat = extract_features(imu, est, seg, params);

params.classifier.method = 'ml';  % Use ML model
cls = ml_predict_baseline(feat, params);

fprintf('\nResult: %s (%.0f%% confidence)\n', upper(cls.label), cls.score*100);
fprintf('Method: %s\n', cls.method);

fprintf('\n========================================\n');
fprintf('  COMPLETE!\n');
fprintf('========================================\n');
```

---

## What This Script Does

| Step | Action |
|------|--------|
| 1-2 | Setup paths and define gesture folders |
| 3 | Load ALL .mat files from each gesture folder |
| 4 | Show dataset summary (samples per class) |
| 5 | Validate you have enough data |
| 6 | Split 80% train / 20% test |
| 7 | Train ML model (kNN by default) |
| 8 | Predict on test set |
| 9 | Show confusion matrix |
| 10 | List misclassified files |
| 11 | Save results |
| 12 | Demo single file classification |

---

## Expected Output
```
========================================
  LOADING & PROCESSING GESTURE DATA
========================================

[CIRCLE] Found 20 files
[FLIP_UP] Found 15 files
[FLIP_DOWN] Found 12 files
...

========================================
  DATASET SUMMARY
========================================
Total samples: 87
Features per sample: 42

Class distribution:
  circle         :  20 samples
  flip_up        :  15 samples
  ...

*** TEST ACCURACY: 78.5% ***