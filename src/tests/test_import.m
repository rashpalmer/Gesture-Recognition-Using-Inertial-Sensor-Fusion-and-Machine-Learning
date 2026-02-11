%% TEST_IMPORT - Test data import functionality
%
% Verifies that read_phone_data.m correctly handles:
%   - Different file formats (.mat, .csv)
%   - Various MATLAB Mobile export conventions
%   - Missing/optional fields
%   - Data validation
%
% Author: Generated for Gesture-SA-Sensor-Fusion project
% Date: 2025

function tests = test_import
    tests = functiontests(localfunctions);
end

%% Test: Synthetic .mat file with standard naming
function test_mat_standard_naming(testCase)
    % Create temporary test file
    test_file = create_temp_mat_standard();
    cleanup = onCleanup(@() delete(test_file));
    
    data = read_phone_data(test_file);
    
    % Verify required fields exist
    verifyTrue(testCase, isfield(data, 't'), 'Should have time field');
    verifyTrue(testCase, isfield(data, 'acc'), 'Should have acc field');
    verifyTrue(testCase, isfield(data, 'gyr'), 'Should have gyr field');
    verifyTrue(testCase, isfield(data, 'meta'), 'Should have meta field');
    
    % Verify dimensions
    N = length(data.t);
    verifySize(testCase, data.acc, [N, 3], 'acc should be Nx3');
    verifySize(testCase, data.gyr, [N, 3], 'gyr should be Nx3');
end

%% Test: Time is monotonic
function test_time_monotonic(testCase)
    test_file = create_temp_mat_standard();
    cleanup = onCleanup(@() delete(test_file));
    
    data = read_phone_data(test_file);
    
    dt = diff(data.t);
    verifyGreaterThan(testCase, min(dt), 0, ...
        'Time should be strictly increasing');
end

%% Test: Handles missing magnetometer gracefully
function test_missing_mag(testCase)
    test_file = create_temp_mat_no_mag();
    cleanup = onCleanup(@() delete(test_file));
    
    data = read_phone_data(test_file);
    
    % Should still work without mag
    verifyTrue(testCase, isfield(data, 't'), 'Should have time field');
    verifyTrue(testCase, isfield(data, 'acc'), 'Should have acc field');
    verifyTrue(testCase, isfield(data, 'gyr'), 'Should have gyr field');
    
    % mag might be empty or NaN
    if isfield(data, 'mag') && ~isempty(data.mag)
        verifyTrue(testCase, all(isnan(data.mag(:))) || isempty(data.mag), ...
            'Mag should be NaN or empty when not present');
    end
end

%% Test: Handles timetable format
function test_timetable_format(testCase)
    test_file = create_temp_mat_timetable();
    cleanup = onCleanup(@() delete(test_file));
    
    data = read_phone_data(test_file);
    
    verifyTrue(testCase, isfield(data, 't'), 'Should extract time from timetable');
    verifyTrue(testCase, isfield(data, 'acc'), 'Should extract acc from timetable');
    verifyGreaterThan(testCase, length(data.t), 0, 'Should have data');
end

%% Test: Detects reasonable sampling rate
function test_sampling_rate(testCase)
    test_file = create_temp_mat_standard();
    cleanup = onCleanup(@() delete(test_file));
    
    data = read_phone_data(test_file);
    
    % Should detect ~100 Hz
    verifyGreaterThan(testCase, data.meta.Fs, 50, ...
        'Should detect reasonable sampling rate > 50 Hz');
    verifyLessThan(testCase, data.meta.Fs, 200, ...
        'Should detect reasonable sampling rate < 200 Hz');
end

%% Test: Alternative variable naming (Acceleration vs acc)
function test_alternative_naming(testCase)
    test_file = create_temp_mat_matlab_mobile();
    cleanup = onCleanup(@() delete(test_file));
    
    data = read_phone_data(test_file);
    
    verifyTrue(testCase, isfield(data, 'acc'), ...
        'Should recognize "Acceleration" as acc');
    verifyTrue(testCase, isfield(data, 'gyr'), ...
        'Should recognize "AngularVelocity" as gyr');
end

%% Test: CSV import
function test_csv_import(testCase)
    test_file = create_temp_csv();
    cleanup = onCleanup(@() delete(test_file));
    
    data = read_phone_data(test_file);
    
    verifyTrue(testCase, isfield(data, 't'), 'Should import time from CSV');
    verifyTrue(testCase, isfield(data, 'acc'), 'Should import acc from CSV');
    verifyGreaterThan(testCase, length(data.t), 0, 'Should have data');
end

%% Test: Data validation catches obvious errors
function test_validation(testCase)
    test_file = create_temp_mat_standard();
    cleanup = onCleanup(@() delete(test_file));
    
    data = read_phone_data(test_file);
    
    % Accelerometer should be reasonable (not millions of g)
    verifyLessThan(testCase, max(abs(data.acc(:))), 1000, ...
        'Accelerometer values should be reasonable');
    
    % Gyroscope should be reasonable (not thousands of rad/s)
    verifyLessThan(testCase, max(abs(data.gyr(:))), 100, ...
        'Gyroscope values should be reasonable');
end

%% Helper: Create standard .mat test file
function filepath = create_temp_mat_standard()
    Fs = 100;
    duration = 2;  % seconds
    N = Fs * duration;
    
    t = (0:N-1)' / Fs;
    acc = randn(N, 3) * 0.1 + [0, 0, -9.81];
    gyr = randn(N, 3) * 0.01;
    mag = randn(N, 3) * 2 + [20, 5, 40];
    
    filepath = [tempname, '.mat'];
    save(filepath, 't', 'acc', 'gyr', 'mag');
end

%% Helper: Create .mat without magnetometer
function filepath = create_temp_mat_no_mag()
    Fs = 100;
    N = 200;
    
    t = (0:N-1)' / Fs;
    acc = randn(N, 3) * 0.1 + [0, 0, -9.81];
    gyr = randn(N, 3) * 0.01;
    
    filepath = [tempname, '.mat'];
    save(filepath, 't', 'acc', 'gyr');
end

%% Helper: Create .mat with timetable format
function filepath = create_temp_mat_timetable()
    Fs = 100;
    N = 200;
    
    Time = seconds((0:N-1)' / Fs);
    X = randn(N, 1);
    Y = randn(N, 1);
    Z = randn(N, 1) - 9.81;
    
    Acceleration = timetable(Time, X, Y, Z);
    
    Time = seconds((0:N-1)' / Fs);
    X = randn(N, 1) * 0.01;
    Y = randn(N, 1) * 0.01;
    Z = randn(N, 1) * 0.01;
    
    AngularVelocity = timetable(Time, X, Y, Z);
    
    filepath = [tempname, '.mat'];
    save(filepath, 'Acceleration', 'AngularVelocity');
end

%% Helper: Create .mat with MATLAB Mobile naming convention
function filepath = create_temp_mat_matlab_mobile()
    Fs = 100;
    N = 200;
    
    Timestamp = (0:N-1)' / Fs;
    
    % MATLAB Mobile style: Acceleration, AngularVelocity
    Acceleration = randn(N, 3) * 0.1 + [0, 0, -9.81];
    AngularVelocity = randn(N, 3) * 0.01;
    MagneticField = randn(N, 3) * 2 + [20, 5, 40];
    
    filepath = [tempname, '.mat'];
    save(filepath, 'Timestamp', 'Acceleration', 'AngularVelocity', 'MagneticField');
end

%% Helper: Create CSV test file
function filepath = create_temp_csv()
    Fs = 100;
    N = 200;
    
    t = (0:N-1)' / Fs;
    acc_x = randn(N, 1) * 0.1;
    acc_y = randn(N, 1) * 0.1;
    acc_z = randn(N, 1) * 0.1 - 9.81;
    gyr_x = randn(N, 1) * 0.01;
    gyr_y = randn(N, 1) * 0.01;
    gyr_z = randn(N, 1) * 0.01;
    
    T = table(t, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, ...
        'VariableNames', {'time', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z'});
    
    filepath = [tempname, '.csv'];
    writetable(T, filepath);
end


%% To run these tests:
%   results = runtests('test_import');
%   disp(results);
