function save_results(results, filename, cfg, varargin)
% SAVE_RESULTS - Save results in multiple formats
%
% Inputs:
%   results   - Struct or table containing results
%   filename  - Base filename (without extension)
%   cfg       - Configuration struct
%   varargin  - Optional: 'Format', {'mat', 'csv', 'json'} (default: all)
%
% Example:
%   save_results(metrics, 'evaluation_results', cfg, 'Format', {'mat', 'csv'});

    p = inputParser;
    addParameter(p, 'Format', {'mat', 'csv'}, @iscell);
    parse(p, varargin{:});

    formats = p.Results.Format;

    % Add timestamp
    timestamp = datestr(now, cfg.log.timestamp_format);

    for i = 1:length(formats)
        fmt = lower(formats{i});

        switch fmt
            case 'mat'
                % Save as .mat file
                filepath = fullfile(cfg.paths.results, [filename '.mat']);
                save(filepath, 'results', 'timestamp', '-v7.3');
                fprintf('Saved: %s\n', filepath);

            case 'csv'
                % Save as CSV (if struct can be converted to table)
                try
                    if isstruct(results)
                        T = struct2table(results);
                    elseif istable(results)
                        T = results;
                    else
                        warning('Cannot convert to table, skipping CSV');
                        continue;
                    end

                    filepath = fullfile(cfg.paths.tables, [filename '.csv']);
                    writetable(T, filepath);
                    fprintf('Saved: %s\n', filepath);
                catch ME
                    warning('CSV save failed: %s', ME.message);
                end

            case 'json'
                % Save as JSON
                try
                    filepath = fullfile(cfg.paths.results, [filename '.json']);
                    json_str = jsonencode(results);
                    fid = fopen(filepath, 'w');
                    fprintf(fid, '%s', json_str);
                    fclose(fid);
                    fprintf('Saved: %s\n', filepath);
                catch ME
                    warning('JSON save failed: %s', ME.message);
                end

            otherwise
                warning('Unknown format: %s', fmt);
        end
    end
end
