% Test with correct field name
fprintf('\nTesting on held-out data...\n');
correct = 0;
testLabels = allLabels(idx(split+1:end));

for i = 1:length(testLabels)
    % Use uppercase X to match what ML model expects
    feat.X = allFeatures(idx(split+i), :);      % <-- UPPERCASE X
    feat.x = allFeatures(idx(split+i), :);      % Keep lowercase too for fallback
    feat.names = featureNames;
    feat.values = struct();
    
    cls = ml_predict_baseline(feat, params);
    
    if strcmp(cls.label, testLabels{i})
        correct = correct + 1;
    else
        fprintf('  WRONG: Actual=%s, Predicted=%s\n', testLabels{i}, cls.label);
    end
end

fprintf('\n*** TEST ACCURACY: %.1f%% (%d/%d) ***\n', ...
    100*correct/length(testLabels), correct, length(testLabels));