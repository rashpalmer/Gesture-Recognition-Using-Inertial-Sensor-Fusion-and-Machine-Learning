function data = convert_mobile_data(filepath)
%CONVERT_MOBILE_DATA Convert MATLAB Mobile timetable format to standard struct
%
%   data = convert_mobile_data('circle_001.mat')

    raw = load(filepath);
    
    %% Extract acceleration
    if isfield(raw, 'Acceleration') && ~isempty(raw.Acceleration)
        acc_table = raw.Acceleration;
        t_acc = seconds(acc_table.Timestamp - acc_table.Timestamp(1));
        data.acc = [acc_table.X, acc_table.Y, acc_table.Z];
        data.t = t_acc;
    else
        error('No Acceleration data found');
    end
    
    %% Extract gyroscope (angular velocity)
    if isfield(raw, 'AngularVelocity') && ~isempty(raw.AngularVelocity)
        gyr_table = raw.AngularVelocity;
        data.gyr = [gyr_table.X, gyr_table.Y, gyr_table.Z];
    else
        data.gyr = zeros(size(data.acc));
        warning('No AngularVelocity data, using zeros');
    end
    
    %% Extract magnetometer
    if isfield(raw, 'MagneticField') && ~isempty(raw.MagneticField)
        mag_table = raw.MagneticField;
        data.mag = [mag_table.X, mag_table.Y, mag_table.Z];
    else
        data.mag = zeros(size(data.acc));
        warning('No MagneticField data, using zeros');
    end
    
    %% Extract orientation (if available)
    if isfield(raw, 'Orientation') && ~isempty(raw.Orientation)
        ori_table = raw.Orientation;
        data.ori = [ori_table.X, ori_table.Y, ori_table.Z];
    end
    
    %% Resample to common length (sensors may have different counts)
    n_acc = size(data.acc, 1);
    n_gyr = size(data.gyr, 1);
    n_mag = size(data.mag, 1);
    n_min = min([n_acc, n_gyr, n_mag]);
    
    data.t = data.t(1:n_min);
    data.acc = data.acc(1:n_min, :);
    data.gyr = data.gyr(1:n_min, :);
    data.mag = data.mag(1:n_min, :);
    if isfield(data, 'ori')
        data.ori = data.ori(1:min(n_min, size(data.ori,1)), :);
    end
    
    %% Compute metadata
    dt = diff(data.t);
    data.meta.Fs = 1 / mean(dt);
    data.meta.duration = data.t(end) - data.t(1);
    data.meta.n_samples = n_min;
    data.meta.source = filepath;
    
    fprintf('Loaded: %d samples, %.1f Hz, %.2f seconds\n', ...
        n_min, data.meta.Fs, data.meta.duration);
end