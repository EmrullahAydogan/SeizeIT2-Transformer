function [data_cell, labels] = load_patient_data(patient_id, data_type, cfg)
% LOAD_PATIENT_DATA - Load windowed data for a specific patient
%
% Inputs:
%   patient_id - Patient identifier (e.g., 'sub-015')
%   data_type  - 'train', 'test', 'all'
%   cfg        - Configuration struct from config()
%
% Outputs:
%   data_cell  - Cell array of data windows [channels x time]
%   labels     - Vector of labels (0=normal, 1=seizure)
%
% Example:
%   cfg = config();
%   [data, labels] = load_patient_data('sub-015', 'test', cfg);

    data_cell = {};
    labels = [];

    % Determine which directories to search
    switch lower(data_type)
        case 'train'
            search_dirs = {cfg.paths.model_data_train};
        case 'test'
            search_dirs = {cfg.paths.model_data_test};
        case 'all'
            search_dirs = {cfg.paths.model_data_train, cfg.paths.model_data_test};
        otherwise
            error('Invalid data_type: %s (use train/test/all)', data_type);
    end

    % Load data from all matching files
    for d = 1:length(search_dirs)
        dir_path = search_dirs{d};

        % Find all files for this patient
        files = dir(fullfile(dir_path, sprintf('%s*.mat', patient_id)));

        for f = 1:length(files)
            file_path = fullfile(files(f).folder, files(f).name);

            try
                loaded = load(file_path);

                % Extract X and Y
                if isfield(loaded, 'X')
                    X = squeeze(loaded.X);
                else
                    % Fallback: use first variable
                    vars = fieldnames(loaded);
                    X = squeeze(loaded.(vars{1}));
                end

                if isfield(loaded, 'Y')
                    Y = loaded.Y;
                else
                    % Assume all normal if no labels
                    Y = zeros(size(X, 3), 1);
                end

                % Convert to cell array
                num_windows = size(X, 3);
                for w = 1:num_windows
                    data_cell{end+1} = X(:, :, w); %#ok<AGROW>
                    labels(end+1) = Y(w); %#ok<AGROW>
                end

            catch ME
                warning('Failed to load %s: %s', files(f).name, ME.message);
            end
        end
    end

    % Convert to column vector
    labels = labels(:);

    if isempty(data_cell)
        warning('No data found for patient %s in %s', patient_id, data_type);
    end
end
