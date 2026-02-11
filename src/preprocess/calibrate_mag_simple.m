function [mag_cal, calib] = calibrate_mag_simple(mag_raw, varargin)
%CALIBRATE_MAG_SIMPLE Simple magnetometer calibration for hard/soft iron effects
%
%   Calibrates magnetometer data to remove hard iron offsets and optionally
%   soft iron distortions. Designed for practical use with smartphone sensors.
%
%   Syntax:
%       [mag_cal, calib] = calibrate_mag_simple(mag_raw)
%       [mag_cal, calib] = calibrate_mag_simple(mag_raw, 'Name', Value)
%       mag_cal = calibrate_mag_simple(mag_raw, calib_prev)  % Apply existing calibration
%
%   Inputs:
%       mag_raw  - Raw magnetometer data (Nx3, typically in µT)
%       calib_prev - (optional) Previous calibration struct to apply
%
%   Name-Value Parameters:
%       'Method'      - Calibration method:
%                       'minmax'    - Simple min/max averaging (default, fast)
%                       'sphere'    - Sphere fitting (more accurate)
%                       'ellipsoid' - Full ellipsoid fitting (soft + hard iron)
%       'Normalize'   - Normalize output to unit sphere (default: false)
%       'ExpectedMag' - Expected field magnitude for validation (default: 50 µT)
%       'OutlierPct'  - Percentile for outlier rejection (default: 2)
%       'Verbose'     - Print diagnostic info (default: true)
%
%   Outputs:
%       mag_cal - Calibrated magnetometer data (Nx3)
%       calib   - Calibration parameters struct:
%                 .offset    - Hard iron offset [3x1]
%                 .scale     - Soft iron scale factors [3x1]
%                 .transform - Full 3x3 transform matrix (for ellipsoid)
%                 .magnitude - Estimated field magnitude
%                 .method    - Method used
%                 .residual  - Calibration residual error
%
%   Background:
%       Hard Iron: Permanent magnetic fields from device (speakers, magnets)
%                  Creates offset in measurements
%                  Correction: mag_cal = mag_raw - offset
%
%       Soft Iron: Ferromagnetic materials that distort external field
%                  Creates ellipsoidal distortion (axis scaling/rotation)
%                  Correction: mag_cal = A * (mag_raw - offset)
%
%   Example:
%       % Basic hard iron calibration
%       [mag_cal, calib] = calibrate_mag_simple(mag_raw);
%
%       % Full ellipsoid calibration with normalization
%       [mag_cal, calib] = calibrate_mag_simple(mag_raw, ...
%                              'Method', 'ellipsoid', 'Normalize', true);
%
%       % Apply existing calibration to new data
%       mag_cal_new = calibrate_mag_simple(mag_new, calib);
%
%   Notes:
%       - Best results when data covers all orientations (figure-8 motion)
%       - 'minmax' is robust but less accurate than fitting methods
%       - 'ellipsoid' requires good spatial coverage and more samples
%       - Smartphone magnetometers are easily disturbed - collect data away
%         from computers, speakers, and metal objects
%
%   See also: preprocess_imu, ekf_attitude_quat
%
%   Author: Claude (Anthropic)
%   Date: January 2025

    %% Handle case where previous calibration is provided
    if nargin >= 2 && isstruct(varargin{1})
        % Apply existing calibration
        calib = varargin{1};
        mag_cal = applyCalibration(mag_raw, calib);
        return;
    end
    
    %% Parse options
    p = inputParser;
    addParameter(p, 'Method', 'minmax', @(x) ismember(lower(x), {'minmax', 'sphere', 'ellipsoid'}));
    addParameter(p, 'Normalize', false, @islogical);
    addParameter(p, 'ExpectedMag', 50, @isnumeric);  % µT, typical Earth field
    addParameter(p, 'OutlierPct', 2, @isnumeric);
    addParameter(p, 'Verbose', true, @islogical);
    parse(p, varargin{:});
    opts = p.Results;
    opts.Method = lower(opts.Method);
    
    %% Validate input
    if size(mag_raw, 2) ~= 3
        error('calibrate_mag_simple:invalidInput', 'mag_raw must be Nx3');
    end
    
    N = size(mag_raw, 1);
    if N < 10
        error('calibrate_mag_simple:tooFewSamples', 'Need at least 10 samples');
    end
    
    if opts.Verbose
        fprintf('\n=== Magnetometer Calibration ===\n');
        fprintf('  Method: %s\n', opts.Method);
        fprintf('  Samples: %d\n', N);
    end
    
    %% Remove outliers
    mag_clean = removeOutliers(mag_raw, opts.OutlierPct, opts.Verbose);
    
    %% Compute calibration based on method
    switch opts.Method
        case 'minmax'
            calib = calibrateMinMax(mag_clean, opts);
        case 'sphere'
            calib = calibrateSphere(mag_clean, opts);
        case 'ellipsoid'
            calib = calibrateEllipsoid(mag_clean, opts);
    end
    
    %% Apply calibration
    mag_cal = applyCalibration(mag_raw, calib);
    
    %% Compute residual error
    mag_cal_clean = applyCalibration(mag_clean, calib);
    magnitudes = sqrt(sum(mag_cal_clean.^2, 2));
    calib.residual = std(magnitudes) / mean(magnitudes) * 100;  % CV in %
    
    %% Optionally normalize to unit sphere
    if opts.Normalize
        mag_cal = mag_cal ./ sqrt(sum(mag_cal.^2, 2));
        calib.normalized = true;
    else
        calib.normalized = false;
    end
    
    %% Validate calibration quality
    if opts.Verbose
        fprintf('\n  Calibration Results:\n');
        fprintf('    Hard iron offset: [%.2f, %.2f, %.2f] µT\n', calib.offset);
        fprintf('    Scale factors:    [%.3f, %.3f, %.3f]\n', calib.scale);
        fprintf('    Est. magnitude:   %.2f µT\n', calib.magnitude);
        fprintf('    Residual (CV):    %.2f%%\n', calib.residual);
        
        % Quality assessment
        if calib.residual < 5
            fprintf('    Quality: GOOD\n');
        elseif calib.residual < 15
            fprintf('    Quality: ACCEPTABLE\n');
        else
            fprintf('    Quality: POOR - consider recollecting with better coverage\n');
        end
        
        % Check magnitude
        if abs(calib.magnitude - opts.ExpectedMag) > 20
            fprintf('    WARNING: Magnitude (%.1f) differs from expected (%.1f)\n', ...
                calib.magnitude, opts.ExpectedMag);
        end
        
        fprintf('================================\n\n');
    end
end

%% ======================== CALIBRATION METHODS ========================

function calib = calibrateMinMax(mag, opts)
%CALIBRATEMINMAX Simple min/max averaging for hard iron offset
%
%   Assumes ideal case where data spans full sphere:
%   offset = (max + min) / 2 for each axis

    minVals = min(mag, [], 1);
    maxVals = max(mag, [], 1);
    
    % Hard iron offset
    offset = (minVals + maxVals) / 2;
    
    % Estimate scale from axis ranges
    ranges = maxVals - minVals;
    avgRange = mean(ranges);
    scale = avgRange ./ ranges;  % Normalize to average
    
    % Estimated field magnitude
    magnitude = avgRange / 2;
    
    calib = struct();
    calib.offset = offset(:)';
    calib.scale = scale(:)';
    calib.transform = diag(scale);
    calib.magnitude = magnitude;
    calib.method = 'minmax';
    
    if opts.Verbose
        fprintf('  Min/Max ranges: [%.1f, %.1f, %.1f]\n', ranges);
        coverage = assessCoverage(mag - offset);
        fprintf('  Spatial coverage: %.0f%%\n', coverage);
    end
end

function calib = calibrateSphere(mag, opts)
%CALIBRATESPHERE Fit sphere to find center (hard iron offset)
%
%   Minimizes: sum((|mag - center| - radius)^2)

    if opts.Verbose
        fprintf('  Fitting sphere...\n');
    end
    
    % Initial guess from minmax
    initial = calibrateMinMax(mag, struct('Verbose', false));
    x0 = [initial.offset, initial.magnitude];
    
    % Objective function
    objective = @(x) sphereResidual(x, mag);
    
    % Optimize
    options = optimset('Display', 'off', 'TolFun', 1e-8, 'TolX', 1e-8);
    if exist('fminsearch', 'file')
        x_opt = fminsearch(objective, x0, options);
    else
        % Fallback if optimization toolbox not available
        x_opt = x0;
        warning('calibrate_mag_simple:noOptim', 'fminsearch not available, using minmax');
    end
    
    calib = struct();
    calib.offset = x_opt(1:3);
    calib.magnitude = abs(x_opt(4));
    calib.scale = [1, 1, 1];  % No scale correction for sphere
    calib.transform = eye(3);
    calib.method = 'sphere';
end

function residual = sphereResidual(x, mag)
%SPHERERESIDUAL Compute sum of squared distance from sphere surface

    center = x(1:3);
    radius = abs(x(4));
    
    centered = mag - center;
    distances = sqrt(sum(centered.^2, 2));
    residual = sum((distances - radius).^2);
end

function calib = calibrateEllipsoid(mag, opts)
%CALIBRATEELLIPSOID Full ellipsoid fitting for hard and soft iron
%
%   Fits ellipsoid: (x-c)'*A*(x-c) = 1
%   Soft iron correction transforms ellipsoid to sphere

    if opts.Verbose
        fprintf('  Fitting ellipsoid...\n');
    end
    
    N = size(mag, 1);
    
    % Need sufficient samples for ellipsoid fitting
    if N < 100
        warning('calibrate_mag_simple:fewSamples', ...
            'Ellipsoid fitting works best with 100+ samples, got %d', N);
    end
    
    % Build design matrix for general quadric
    % ax^2 + by^2 + cz^2 + 2dxy + 2exz + 2fyz + 2gx + 2hy + 2iz = 1
    x = mag(:,1); y = mag(:,2); z = mag(:,3);
    
    D = [x.^2, y.^2, z.^2, 2*x.*y, 2*x.*z, 2*y.*z, 2*x, 2*y, 2*z, ones(N,1)];
    
    % Solve constrained least squares (constraint: ellipsoid, not hyperboloid)
    % Simplified approach: assume aligned ellipsoid (d=e=f=0)
    D_simple = [x.^2, y.^2, z.^2, 2*x, 2*y, 2*z, ones(N,1)];
    
    % Least squares: D * v ≈ 1
    v = D_simple \ ones(N, 1);
    
    % Extract parameters
    % v = [a, b, c, g, h, i, j] where ax^2 + by^2 + cz^2 + 2gx + 2hy + 2iz + j = 1
    a = v(1); b = v(2); c = v(3);
    g = v(4); h = v(5); i = v(6); j = v(7);
    
    % Center: from completing the square
    % a(x + g/a)^2 = ax^2 + 2gx + g^2/a
    if a > 0 && b > 0 && c > 0  % Valid ellipsoid
        offset = [-g/a, -h/b, -i/c];
        
        % Radius in each direction
        r_sq = [g^2/a + (1-j)/a, h^2/b + (1-j)/b, i^2/c + (1-j)/c];
        r_sq = max(r_sq, 0.01);  % Ensure positive
        radii = sqrt(r_sq);
        
        % Scale factors to make sphere
        avgRadius = mean(radii);
        scale = avgRadius ./ radii;
        
        calib = struct();
        calib.offset = offset;
        calib.scale = scale;
        calib.transform = diag(scale);
        calib.magnitude = avgRadius;
        calib.method = 'ellipsoid';
        calib.radii = radii;
    else
        % Fallback to sphere if ellipsoid fitting fails
        warning('calibrate_mag_simple:ellipsoidFail', ...
            'Ellipsoid fitting failed (non-positive axes), falling back to sphere');
        calib = calibrateSphere(mag, opts);
        calib.method = 'ellipsoid_fallback';
    end
end

%% ======================== HELPER FUNCTIONS ========================

function mag_cal = applyCalibration(mag_raw, calib)
%APPLYCALIBRATION Apply calibration parameters to raw magnetometer data

    % Remove hard iron offset
    mag_centered = mag_raw - calib.offset;
    
    % Apply soft iron correction (scale or full transform)
    if isfield(calib, 'transform') && ~isequal(calib.transform, eye(3))
        mag_cal = (calib.transform * mag_centered')';
    else
        mag_cal = mag_centered .* calib.scale;
    end
end

function mag_clean = removeOutliers(mag, pct, verbose)
%REMOVEOUTLIERS Remove outlier samples based on magnitude

    magnitudes = sqrt(sum(mag.^2, 2));
    
    lowThresh = prctile(magnitudes, pct);
    highThresh = prctile(magnitudes, 100 - pct);
    
    validMask = (magnitudes >= lowThresh) & (magnitudes <= highThresh);
    mag_clean = mag(validMask, :);
    
    numRemoved = sum(~validMask);
    if verbose && numRemoved > 0
        fprintf('  Removed %d outliers (%.1f%%)\n', numRemoved, 100*numRemoved/length(magnitudes));
    end
end

function coverage = assessCoverage(mag_centered)
%ASSESSCOVERAGE Estimate how well data covers the sphere surface (0-100%)
%
%   Divides sphere into bins and counts how many have data

    % Normalize to unit sphere
    magnitudes = sqrt(sum(mag_centered.^2, 2));
    mag_unit = mag_centered ./ magnitudes;
    
    % Convert to spherical coordinates
    theta = atan2(mag_unit(:,2), mag_unit(:,1));  % azimuth
    phi = asin(mag_unit(:,3));  % elevation
    
    % Bin into grid (e.g., 12x6 = 72 bins)
    nAz = 12; nEl = 6;
    azBins = linspace(-pi, pi, nAz + 1);
    elBins = linspace(-pi/2, pi/2, nEl + 1);
    
    % Count occupied bins
    occupied = zeros(nAz, nEl);
    for i = 1:length(theta)
        azIdx = find(theta(i) >= azBins(1:end-1) & theta(i) < azBins(2:end), 1);
        elIdx = find(phi(i) >= elBins(1:end-1) & phi(i) < elBins(2:end), 1);
        if ~isempty(azIdx) && ~isempty(elIdx)
            occupied(azIdx, elIdx) = 1;
        end
    end
    
    coverage = 100 * sum(occupied(:)) / numel(occupied);
end

%% ======================== VISUALIZATION ========================

function plotCalibration(mag_raw, mag_cal, calib)
%PLOTCALIBRATION Visualize calibration results
%
%   Creates 3D scatter plots of raw and calibrated data

    figure('Name', 'Magnetometer Calibration', 'Position', [100 100 1200 500]);
    
    % Raw data
    subplot(1, 2, 1);
    scatter3(mag_raw(:,1), mag_raw(:,2), mag_raw(:,3), 3, 'b', 'filled');
    hold on;
    plot3(calib.offset(1), calib.offset(2), calib.offset(3), 'ro', ...
        'MarkerSize', 15, 'LineWidth', 3);
    xlabel('X (µT)'); ylabel('Y (µT)'); zlabel('Z (µT)');
    title('Raw Magnetometer Data');
    axis equal; grid on;
    legend('Data', 'Offset', 'Location', 'best');
    
    % Calibrated data
    subplot(1, 2, 2);
    scatter3(mag_cal(:,1), mag_cal(:,2), mag_cal(:,3), 3, 'g', 'filled');
    hold on;
    
    % Plot reference sphere
    [xs, ys, zs] = sphere(20);
    xs = xs * calib.magnitude;
    ys = ys * calib.magnitude;
    zs = zs * calib.magnitude;
    surf(xs, ys, zs, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'FaceColor', 'r');
    
    xlabel('X (µT)'); ylabel('Y (µT)'); zlabel('Z (µT)');
    title(sprintf('Calibrated Data (residual = %.1f%%)', calib.residual));
    axis equal; grid on;
    legend('Data', 'Reference sphere', 'Location', 'best');
    
    sgtitle(sprintf('Magnetometer Calibration (%s method)', calib.method));
end

%% ======================== BATCH CALIBRATION ========================

function calib_combined = calibrateMultipleDatasets(datasets, varargin)
%CALIBRATEMULTIPLEDATASETS Combine multiple datasets for better calibration
%
%   datasets - Cell array of Nx3 magnetometer data arrays
%   Combines all data for more complete coverage

    % Concatenate all datasets
    mag_all = vertcat(datasets{:});
    
    fprintf('Combining %d datasets (%d total samples)\n', length(datasets), size(mag_all, 1));
    
    % Run calibration
    [~, calib_combined] = calibrate_mag_simple(mag_all, varargin{:});
end
