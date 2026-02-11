function export_helpers()
%EXPORT_HELPERS Export utilities for gesture recognition pipeline
%
%   This file contains helper functions for exporting processed data,
%   results, reports, and visualizations.
%
%   Functions:
%       exportResults(results, filename)          - Export classification results
%       exportFeatures(feat, filename)            - Export feature matrix to CSV
%       exportTimeSeries(data, filename)          - Export time series to CSV/MAT
%       exportCalibration(calib, filename)        - Save calibration parameters
%       exportRunLog(logData, outputDir)          - Save run configuration + metrics
%       generateReport(results, outputDir)        - Generate text summary report
%       exportForPython(data, filename)           - Export in Python-friendly format
%       exportForExcel(data, filename)            - Export formatted Excel file
%
%   Usage:
%       exp = export_helpers();
%       exp.exportResults(results, 'outputs/results.csv');
%       exp.exportRunLog(logData, 'outputs/logs/');
%
%   Author: Claude (Anthropic)
%   Date: January 2025

    % Return struct of function handles
    export_helpers = struct();
    export_helpers.exportResults = @exportResults;
    export_helpers.exportFeatures = @exportFeatures;
    export_helpers.exportTimeSeries = @exportTimeSeries;
    export_helpers.exportCalibration = @exportCalibration;
    export_helpers.exportRunLog = @exportRunLog;
    export_helpers.generateReport = @generateReport;
    export_helpers.exportForPython = @exportForPython;
    export_helpers.exportForExcel = @exportForExcel;
    export_helpers.createOutputDir = @createOutputDir;
    export_helpers.getTimestamp = @getTimestamp;
    export_helpers.exportFigure = @exportFigure;
    export_helpers.exportAll = @exportAll;
    
    assignin('caller', 'ans', export_helpers);
end

%% ======================== RESULTS EXPORT ========================

function filepath = exportResults(results, filename)
%EXPORTRESULTS Export classification results to CSV or MAT
%
%   Inputs:
%       results  - Struct or table with classification results
%       filename - Output filename (.csv or .mat)
%
%   Results struct expected fields:
%       .label       - Predicted label (string)
%       .score       - Confidence score
%       .method      - Classification method
%       .timestamp   - When recorded
%       .filename    - Source data file

    if nargin < 2
        filename = sprintf('results_%s.csv', getTimestamp());
    end
    
    % Ensure output directory exists
    [outDir, ~, ext] = fileparts(filename);
    if ~isempty(outDir)
        createOutputDir(outDir);
    end
    
    % Convert to table if needed
    if isstruct(results) && ~istable(results)
        if isfield(results, 'label')
            % Single result
            T = struct2table(results, 'AsArray', true);
        else
            % Array of results
            T = struct2table(results);
        end
    elseif istable(results)
        T = results;
    else
        error('export_helpers:invalidResults', 'Results must be struct or table');
    end
    
    % Export based on extension
    if strcmpi(ext, '.csv')
        writetable(T, filename);
        fprintf('  [export] Results saved to: %s\n', filename);
    elseif strcmpi(ext, '.mat')
        save(filename, 'results', 'T');
        fprintf('  [export] Results saved to: %s\n', filename);
    else
        % Default to CSV
        filename = [filename, '.csv'];
        writetable(T, filename);
        fprintf('  [export] Results saved to: %s\n', filename);
    end
    
    filepath = filename;
end

%% ======================== FEATURE EXPORT ========================

function filepath = exportFeatures(feat, filename, varargin)
%EXPORTFEATURES Export feature matrix to CSV with headers
%
%   Inputs:
%       feat     - Feature struct with .x (NxM matrix) and .names (1xM cell)
%       filename - Output filename
%
%   Options:
%       'IncludeLabels', labels - Add label column (Nx1 cell or string array)
%       'IncludeTime', times    - Add time column

    p = inputParser;
    addParameter(p, 'IncludeLabels', {}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'IncludeTime', [], @isnumeric);
    parse(p, varargin{:});
    opts = p.Results;
    
    if nargin < 2
        filename = sprintf('features_%s.csv', getTimestamp());
    end
    
    % Validate feature struct
    if ~isfield(feat, 'x') || ~isfield(feat, 'names')
        error('export_helpers:invalidFeatures', ...
            'Feature struct must have .x and .names fields');
    end
    
    % Build table
    X = feat.x;
    names = feat.names;
    
    % Ensure names are valid variable names
    validNames = matlab.lang.makeValidName(names);
    
    T = array2table(X, 'VariableNames', validNames);
    
    % Add optional columns
    if ~isempty(opts.IncludeLabels)
        T.Label = opts.IncludeLabels(:);
        T = movevars(T, 'Label', 'Before', 1);
    end
    
    if ~isempty(opts.IncludeTime)
        T.Time = opts.IncludeTime(:);
        T = movevars(T, 'Time', 'Before', 1);
    end
    
    % Ensure output directory exists
    [outDir, ~, ~] = fileparts(filename);
    if ~isempty(outDir)
        createOutputDir(outDir);
    end
    
    writetable(T, filename);
    fprintf('  [export] Features (%d samples x %d features) saved to: %s\n', ...
        size(X, 1), size(X, 2), filename);
    
    filepath = filename;
end

%% ======================== TIME SERIES EXPORT ========================

function filepath = exportTimeSeries(data, filename, varargin)
%EXPORTTIMESERIES Export time series data to CSV or MAT
%
%   Inputs:
%       data     - Struct with .t, .acc, .gyr, .mag, etc.
%       filename - Output filename
%
%   Options:
%       'Format'  - 'csv' (default), 'mat', or 'both'
%       'Fields'  - Cell array of fields to export (default: all numeric)

    p = inputParser;
    addParameter(p, 'Format', 'csv', @(x) ismember(lower(x), {'csv', 'mat', 'both'}));
    addParameter(p, 'Fields', {}, @iscell);
    parse(p, varargin{:});
    opts = p.Results;
    
    if nargin < 2
        filename = sprintf('timeseries_%s', getTimestamp());
    end
    
    % Remove extension if present
    [outDir, baseName, ~] = fileparts(filename);
    if isempty(baseName)
        baseName = filename;
    end
    
    % Determine fields to export
    if isempty(opts.Fields)
        % Export all numeric array fields
        fields = fieldnames(data);
        exportFields = {};
        for i = 1:length(fields)
            val = data.(fields{i});
            if isnumeric(val) && ~isscalar(val)
                exportFields{end+1} = fields{i}; %#ok<AGROW>
            end
        end
    else
        exportFields = opts.Fields;
    end
    
    % Ensure output directory exists
    if ~isempty(outDir)
        createOutputDir(outDir);
    else
        outDir = '.';
    end
    
    filepath = '';
    
    % Export CSV
    if ismember(lower(opts.Format), {'csv', 'both'})
        % Build column data
        columns = {};
        colNames = {};
        
        % Time first
        if isfield(data, 't')
            columns{end+1} = data.t(:);
            colNames{end+1} = 'time_s';
        end
        
        % Add each field
        for i = 1:length(exportFields)
            field = exportFields{i};
            if ~isfield(data, field) || strcmp(field, 't')
                continue;
            end
            
            val = data.(field);
            numCols = size(val, 2);
            
            if numCols == 1
                columns{end+1} = val(:); %#ok<AGROW>
                colNames{end+1} = field; %#ok<AGROW>
            else
                % Multi-column (e.g., acc_x, acc_y, acc_z)
                axisLabels = {'x', 'y', 'z', 'w'};
                for c = 1:numCols
                    columns{end+1} = val(:, c); %#ok<AGROW>
                    if c <= length(axisLabels)
                        colNames{end+1} = sprintf('%s_%s', field, axisLabels{c}); %#ok<AGROW>
                    else
                        colNames{end+1} = sprintf('%s_%d', field, c); %#ok<AGROW>
                    end
                end
            end
        end
        
        % Create table and write
        if ~isempty(columns)
            T = table(columns{:}, 'VariableNames', matlab.lang.makeValidName(colNames));
            csvPath = fullfile(outDir, [baseName, '.csv']);
            writetable(T, csvPath);
            fprintf('  [export] Time series saved to: %s\n', csvPath);
            filepath = csvPath;
        end
    end
    
    % Export MAT
    if ismember(lower(opts.Format), {'mat', 'both'})
        matPath = fullfile(outDir, [baseName, '.mat']);
        save(matPath, '-struct', 'data');
        fprintf('  [export] Time series saved to: %s\n', matPath);
        if isempty(filepath)
            filepath = matPath;
        end
    end
end

%% ======================== CALIBRATION EXPORT ========================

function filepath = exportCalibration(calib, filename)
%EXPORTCALIBRATION Save calibration parameters to MAT and readable JSON-like format
%
%   Inputs:
%       calib    - Calibration struct from calibrate_mag_simple or preprocess_imu
%       filename - Output base filename (will create .mat and .txt)

    if nargin < 2
        filename = sprintf('calibration_%s', getTimestamp());
    end
    
    [outDir, baseName, ~] = fileparts(filename);
    if isempty(baseName)
        baseName = filename;
    end
    if isempty(outDir)
        outDir = '.';
    end
    
    createOutputDir(outDir);
    
    % Save MAT file
    matPath = fullfile(outDir, [baseName, '.mat']);
    save(matPath, '-struct', 'calib');
    
    % Save human-readable text file
    txtPath = fullfile(outDir, [baseName, '.txt']);
    fid = fopen(txtPath, 'w');
    fprintf(fid, '# Calibration Parameters\n');
    fprintf(fid, '# Generated: %s\n\n', datestr(now));
    
    fields = fieldnames(calib);
    for i = 1:length(fields)
        field = fields{i};
        val = calib.(field);
        
        if isnumeric(val)
            if isscalar(val)
                fprintf(fid, '%s = %.6g\n', field, val);
            elseif isvector(val)
                fprintf(fid, '%s = [%s]\n', field, num2str(val, '%.6g '));
            elseif ismatrix(val)
                fprintf(fid, '%s = [\n', field);
                for r = 1:size(val, 1)
                    fprintf(fid, '  %s\n', num2str(val(r,:), '%.6g '));
                end
                fprintf(fid, ']\n');
            end
        elseif ischar(val) || isstring(val)
            fprintf(fid, '%s = "%s"\n', field, val);
        elseif islogical(val)
            fprintf(fid, '%s = %s\n', field, mat2str(val));
        end
    end
    
    fclose(fid);
    
    fprintf('  [export] Calibration saved to: %s (.mat + .txt)\n', fullfile(outDir, baseName));
    filepath = matPath;
end

%% ======================== RUN LOG EXPORT ========================

function filepath = exportRunLog(logData, outputDir)
%EXPORTRUNLOG Save run configuration, parameters, and metrics
%
%   Creates timestamped log file with:
%   - Configuration parameters
%   - Input file info
%   - Processing metrics
%   - Classification results
%
%   Inputs:
%       logData   - Struct with run information
%       outputDir - Output directory (default: 'outputs/logs/')

    if nargin < 2
        outputDir = 'outputs/logs';
    end
    
    createOutputDir(outputDir);
    
    timestamp = getTimestamp();
    filename = fullfile(outputDir, sprintf('run_%s.mat', timestamp));
    
    % Add timestamp to log
    logData.timestamp = timestamp;
    logData.datetime = datestr(now);
    logData.matlabVersion = version;
    
    % Save MAT file
    save(filename, '-struct', 'logData');
    
    % Also save human-readable summary
    txtFile = fullfile(outputDir, sprintf('run_%s.txt', timestamp));
    fid = fopen(txtFile, 'w');
    
    fprintf(fid, '========================================\n');
    fprintf(fid, 'GESTURE RECOGNITION RUN LOG\n');
    fprintf(fid, '========================================\n');
    fprintf(fid, 'Timestamp: %s\n', logData.datetime);
    fprintf(fid, 'MATLAB: %s\n\n', logData.matlabVersion);
    
    % Write each section
    sections = fieldnames(logData);
    for i = 1:length(sections)
        section = sections{i};
        val = logData.(section);
        
        if isstruct(val)
            fprintf(fid, '\n--- %s ---\n', upper(section));
            writeStructToFile(fid, val, '  ');
        elseif ~ismember(section, {'timestamp', 'datetime', 'matlabVersion'})
            fprintf(fid, '%s: %s\n', section, formatValue(val));
        end
    end
    
    fprintf(fid, '\n========================================\n');
    fclose(fid);
    
    fprintf('  [export] Run log saved to: %s\n', filename);
    filepath = filename;
end

function writeStructToFile(fid, s, indent)
%WRITESTRUCTTOFILE Recursively write struct to file
    fields = fieldnames(s);
    for i = 1:length(fields)
        field = fields{i};
        val = s.(field);
        
        if isstruct(val)
            fprintf(fid, '%s%s:\n', indent, field);
            writeStructToFile(fid, val, [indent, '  ']);
        else
            fprintf(fid, '%s%s: %s\n', indent, field, formatValue(val));
        end
    end
end

function str = formatValue(val)
%FORMATVALUE Convert value to display string
    if ischar(val) || isstring(val)
        str = char(val);
    elseif isnumeric(val)
        if isscalar(val)
            str = sprintf('%.6g', val);
        elseif isvector(val) && length(val) <= 10
            str = sprintf('[%s]', num2str(val(:)', '%.4g '));
        else
            str = sprintf('[%dx%d array]', size(val, 1), size(val, 2));
        end
    elseif islogical(val)
        str = mat2str(val);
    elseif iscell(val)
        str = sprintf('{%d elements}', numel(val));
    else
        str = class(val);
    end
end

%% ======================== REPORT GENERATION ========================

function filepath = generateReport(results, outputDir, varargin)
%GENERATEREPORT Generate human-readable text summary report
%
%   Inputs:
%       results   - Struct with classification results and metrics
%       outputDir - Output directory
%
%   Options:
%       'Title'    - Report title
%       'Verbose'  - Include detailed metrics

    p = inputParser;
    addParameter(p, 'Title', 'Gesture Recognition Report', @ischar);
    addParameter(p, 'Verbose', true, @islogical);
    parse(p, varargin{:});
    opts = p.Results;
    
    if nargin < 2
        outputDir = 'outputs';
    end
    
    createOutputDir(outputDir);
    
    timestamp = getTimestamp();
    filename = fullfile(outputDir, sprintf('report_%s.txt', timestamp));
    
    fid = fopen(filename, 'w');
    
    % Header
    fprintf(fid, '╔══════════════════════════════════════════════════════════════╗\n');
    fprintf(fid, '║  %s\n', opts.Title);
    fprintf(fid, '║  Generated: %s\n', datestr(now));
    fprintf(fid, '╚══════════════════════════════════════════════════════════════╝\n\n');
    
    % Classification result
    if isfield(results, 'cls')
        fprintf(fid, '┌─ CLASSIFICATION RESULT ─────────────────────────────────────┐\n');
        fprintf(fid, '│  Predicted Gesture: %s\n', upper(results.cls.label));
        fprintf(fid, '│  Confidence Score:  %.2f%%\n', results.cls.score * 100);
        fprintf(fid, '│  Method:            %s\n', results.cls.method);
        if isfield(results.cls, 'reason')
            fprintf(fid, '│  Reason:            %s\n', results.cls.reason);
        end
        fprintf(fid, '└─────────────────────────────────────────────────────────────┘\n\n');
    end
    
    % Data summary
    if isfield(results, 'data')
        fprintf(fid, '┌─ DATA SUMMARY ──────────────────────────────────────────────┐\n');
        if isfield(results.data, 'N')
            fprintf(fid, '│  Samples:     %d\n', results.data.N);
        end
        if isfield(results.data, 'duration')
            fprintf(fid, '│  Duration:    %.2f s\n', results.data.duration);
        end
        if isfield(results.data, 'Fs')
            fprintf(fid, '│  Sample Rate: %.1f Hz\n', results.data.Fs);
        end
        if isfield(results.data, 'filename')
            fprintf(fid, '│  Source:      %s\n', results.data.filename);
        end
        fprintf(fid, '└─────────────────────────────────────────────────────────────┘\n\n');
    end
    
    % Segmentation info
    if isfield(results, 'seg')
        fprintf(fid, '┌─ GESTURE SEGMENTATION ─────────────────────────────────────-┐\n');
        if isfield(results.seg, 'windows')
            fprintf(fid, '│  Windows Found: %d\n', size(results.seg.windows, 1));
        end
        if isfield(results.seg, 'winIdx')
            duration = (results.seg.winIdx(2) - results.seg.winIdx(1)) / results.data.Fs;
            fprintf(fid, '│  Primary Window: samples %d-%d (%.2f s)\n', ...
                results.seg.winIdx(1), results.seg.winIdx(2), duration);
        end
        if isfield(results.seg, 'score')
            fprintf(fid, '│  Segmentation Score: %.2f\n', results.seg.score);
        end
        fprintf(fid, '└─────────────────────────────────────────────────────────────┘\n\n');
    end
    
    % Feature highlights
    if opts.Verbose && isfield(results, 'feat')
        fprintf(fid, '┌─ KEY FEATURES ──────────────────────────────────────────────┐\n');
        if isfield(results.feat, 'values')
            vals = results.feat.values;
            if isfield(vals, 'duration')
                fprintf(fid, '│  Duration:       %.3f s\n', vals.duration);
            end
            if isfield(vals, 'peakGyr')
                fprintf(fid, '│  Peak Gyro:      %.2f rad/s\n', vals.peakGyr);
            end
            if isfield(vals, 'totalRotation')
                fprintf(fid, '│  Total Rotation: %.1f deg\n', rad2deg(vals.totalRotation));
            end
            if isfield(vals, 'dominantAxis')
                fprintf(fid, '│  Dominant Axis:  %s\n', vals.dominantAxis);
            end
        end
        fprintf(fid, '└─────────────────────────────────────────────────────────────┘\n\n');
    end
    
    % Fusion quality
    if opts.Verbose && isfield(results, 'fusion')
        fprintf(fid, '┌─ SENSOR FUSION QUALITY ────────────────────────────────────-┐\n');
        if isfield(results.fusion, 'quatNormError')
            fprintf(fid, '│  Quaternion Norm Error: %.6f\n', results.fusion.quatNormError);
        end
        if isfield(results.fusion, 'biasStability')
            fprintf(fid, '│  Gyro Bias Stability:   %.4f rad/s\n', results.fusion.biasStability);
        end
        fprintf(fid, '└─────────────────────────────────────────────────────────────┘\n\n');
    end
    
    fclose(fid);
    
    fprintf('  [export] Report saved to: %s\n', filename);
    filepath = filename;
end

%% ======================== PYTHON/EXCEL EXPORT ========================

function filepath = exportForPython(data, filename)
%EXPORTFORPYTHON Export in NumPy-friendly format (CSV with metadata)
%
%   Creates a CSV file with a companion .json metadata file

    if nargin < 2
        filename = sprintf('data_for_python_%s', getTimestamp());
    end
    
    [outDir, baseName, ~] = fileparts(filename);
    if isempty(outDir)
        outDir = '.';
    end
    createOutputDir(outDir);
    
    % Export main data as CSV
    csvPath = fullfile(outDir, [baseName, '.csv']);
    exportTimeSeries(data, csvPath, 'Format', 'csv');
    
    % Create metadata file
    meta = struct();
    meta.columns = {};
    meta.shapes = struct();
    
    fields = fieldnames(data);
    for i = 1:length(fields)
        field = fields{i};
        val = data.(field);
        if isnumeric(val) && ~isscalar(val)
            meta.shapes.(field) = size(val);
        end
    end
    
    if isfield(data, 'Fs')
        meta.sample_rate_hz = data.Fs;
    end
    if isfield(data, 't')
        meta.duration_s = data.t(end) - data.t(1);
        meta.num_samples = length(data.t);
    end
    
    % Save metadata as simple text (pseudo-JSON)
    metaPath = fullfile(outDir, [baseName, '_meta.txt']);
    fid = fopen(metaPath, 'w');
    fprintf(fid, '# Metadata for %s.csv\n', baseName);
    fprintf(fid, '# Load in Python: data = np.loadtxt("%s.csv", delimiter=",", skiprows=1)\n\n', baseName);
    
    metaFields = fieldnames(meta);
    for i = 1:length(metaFields)
        fprintf(fid, '%s: %s\n', metaFields{i}, formatValue(meta.(metaFields{i})));
    end
    fclose(fid);
    
    fprintf('  [export] Python-compatible files saved: %s + _meta.txt\n', csvPath);
    filepath = csvPath;
end

function filepath = exportForExcel(data, filename)
%EXPORTFOREXCEL Export with Excel-friendly formatting
%
%   Creates .xlsx with multiple sheets if data has multiple arrays

    if nargin < 2
        filename = sprintf('data_%s.xlsx', getTimestamp());
    end
    
    [outDir, baseName, ext] = fileparts(filename);
    if isempty(ext)
        ext = '.xlsx';
    end
    if isempty(outDir)
        outDir = '.';
    end
    createOutputDir(outDir);
    
    xlsPath = fullfile(outDir, [baseName, ext]);
    
    % Write time series to main sheet
    if isfield(data, 't')
        mainData = data.t(:);
        headers = {'Time_s'};
        
        for field = {'acc', 'gyr', 'mag'}
            if isfield(data, field{1}) && ~isempty(data.(field{1}))
                vals = data.(field{1});
                mainData = [mainData, vals]; %#ok<AGROW>
                for ax = {'X', 'Y', 'Z'}
                    headers{end+1} = sprintf('%s_%s', field{1}, ax{1}); %#ok<AGROW>
                end
            end
        end
        
        T = array2table(mainData, 'VariableNames', headers);
        writetable(T, xlsPath, 'Sheet', 'TimeSeries');
    end
    
    % Write quaternions to separate sheet if present
    if isfield(data, 'q')
        T_quat = array2table(data.q, 'VariableNames', {'q_w', 'q_x', 'q_y', 'q_z'});
        if isfield(data, 't')
            T_quat.Time_s = data.t(:);
            T_quat = movevars(T_quat, 'Time_s', 'Before', 1);
        end
        writetable(T_quat, xlsPath, 'Sheet', 'Quaternions');
    end
    
    fprintf('  [export] Excel file saved: %s\n', xlsPath);
    filepath = xlsPath;
end

%% ======================== UTILITY FUNCTIONS ========================

function createOutputDir(dirPath)
%CREATEOUTPUTDIR Create directory if it doesn't exist
    if ~exist(dirPath, 'dir')
        mkdir(dirPath);
    end
end

function ts = getTimestamp()
%GETTIMESTAMP Get formatted timestamp for filenames
    ts = datestr(now, 'yyyymmdd_HHMMSS');
end

function filepath = exportFigure(fig, filename, varargin)
%EXPORTFIGURE Save figure in multiple formats
%
%   Options:
%       'Formats' - Cell array of formats: 'png', 'pdf', 'fig', 'svg'
%       'Resolution' - DPI for raster formats (default: 300)

    p = inputParser;
    addParameter(p, 'Formats', {'png'}, @iscell);
    addParameter(p, 'Resolution', 300, @isnumeric);
    parse(p, varargin{:});
    opts = p.Results;
    
    [outDir, baseName, ~] = fileparts(filename);
    if isempty(outDir)
        outDir = '.';
    end
    createOutputDir(outDir);
    
    filepath = '';
    
    for i = 1:length(opts.Formats)
        fmt = lower(opts.Formats{i});
        outFile = fullfile(outDir, [baseName, '.', fmt]);
        
        switch fmt
            case 'png'
                print(fig, outFile, '-dpng', sprintf('-r%d', opts.Resolution));
            case 'pdf'
                print(fig, outFile, '-dpdf', '-bestfit');
            case 'svg'
                print(fig, outFile, '-dsvg');
            case 'fig'
                savefig(fig, outFile);
            case 'eps'
                print(fig, outFile, '-depsc2');
        end
        
        if isempty(filepath)
            filepath = outFile;
        end
    end
    
    fprintf('  [export] Figure saved: %s\n', fullfile(outDir, baseName));
end

function exportAll(results, outputDir)
%EXPORTALL Export all results, features, and generate report
%
%   Convenience function to export everything at once

    if nargin < 2
        outputDir = 'outputs';
    end
    
    createOutputDir(outputDir);
    timestamp = getTimestamp();
    
    fprintf('\n=== Exporting All Results ===\n');
    
    % Export results
    if isfield(results, 'cls')
        exportResults(results.cls, fullfile(outputDir, sprintf('results_%s.csv', timestamp)));
    end
    
    % Export features
    if isfield(results, 'feat')
        exportFeatures(results.feat, fullfile(outputDir, sprintf('features_%s.csv', timestamp)));
    end
    
    % Export calibration
    if isfield(results, 'calib')
        exportCalibration(results.calib, fullfile(outputDir, sprintf('calibration_%s', timestamp)));
    end
    
    % Generate report
    generateReport(results, outputDir);
    
    fprintf('=== Export Complete ===\n\n');
end
