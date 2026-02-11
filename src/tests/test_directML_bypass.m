%% Direct ML Test - Correct Version
fprintf('\n=== Direct ML Model Test ===\n');

% Load everything from model file
data = load('models/gesture_model.mat');
model = data.model;

% Get test data
testIdx = idx(split+1:end);
testFeatures = allFeatures(testIdx, :);
testLabels = allLabels(testIdx);

% Find normalization parameters (check different possible locations)
if isfield(data, 'featureMean') && isfield(data, 'featureStd')
    mu = data.featureMean;
    sigma = data.featureStd;
elseif isfield(model, 'normalization')
    mu = model.normalization.mu;
    sigma = model.normalization.sigma;
elseif isfield(model, 'mu')
    mu = model.mu;
    sigma = model.sigma;
else
    warning('No normalization found, using training data stats');
    trainFeatures = allFeatures(idx(1:split), :);
    mu = mean(trainFeatures, 1);
    sigma = std(trainFeatures, 0, 1);
end
sigma(sigma == 0) = 1;  % Prevent div by zero

% Normalize test data
X_test = (testFeatures - mu) ./ sigma;

% Predict using the classifier directly
if isfield(model, 'classifier')
    % Model has a classifier object
    predictions = predict(model.classifier, X_test);
elseif isa(model, 'ClassificationKNN') || isa(model, 'ClassificationTree')
    % Model IS the classifier
    predictions = predict(model, X_test);
else
    error('Cannot find classifier in model');
end

% Convert to cell if needed
if iscategorical(predictions)
    predictions = cellstr(predictions);
end

% Calculate accuracy
correct = sum(strcmp(predictions, testLabels));
total = length(testLabels);

fprintf('\n*** ML TEST ACCURACY: %.1f%% (%d/%d) ***\n', ...
    100*correct/total, correct, total);

% Show per-class breakdown
classes = unique(testLabels);
fprintf('\nPer-class results:\n');
for i = 1:length(classes)
    mask = strcmp(testLabels, classes{i});
    classCorrect = sum(strcmp(predictions(mask), testLabels(mask)));
    classTotal = sum(mask);
    fprintf('  %12s: %d/%d (%.0f%%)\n', classes{i}, classCorrect, classTotal, 100*classCorrect/classTotal);
end

% List misclassifications
fprintf('\nMisclassifications:\n');
wrong = find(~strcmp(predictions, testLabels));
for i = 1:length(wrong)
    fprintf('  #%2d: Actual=%-12s Predicted=%s\n', wrong(i), testLabels{wrong(i)}, predictions{wrong(i)});
end