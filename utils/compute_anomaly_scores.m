function scores = compute_anomaly_scores(net, data_cell, varargin)
% COMPUTE_ANOMALY_SCORES - Calculate reconstruction errors for anomaly detection
%
% Inputs:
%   net        - Trained autoencoder network
%   data_cell  - Cell array of input samples [channels x time]
%   varargin   - Optional parameters:
%                'MiniBatchSize', batch_size (default: 16)
%                'ErrorMetric', 'mse'|'mae'|'rmse' (default: 'mse')
%
% Outputs:
%   scores     - Vector of anomaly scores (one per sample)
%
% Example:
%   scores = compute_anomaly_scores(net, test_data, 'MiniBatchSize', 8);

    % Parse inputs
    p = inputParser;
    addParameter(p, 'MiniBatchSize', 16, @isnumeric);
    addParameter(p, 'ErrorMetric', 'mse', @ischar);
    addParameter(p, 'Verbose', false, @islogical);
    parse(p, varargin{:});

    batch_size = p.Results.MiniBatchSize;
    error_metric = p.Results.ErrorMetric;
    verbose = p.Results.Verbose;

    % Predict reconstructions
    if verbose
        fprintf('Computing anomaly scores for %d samples... ', length(data_cell));
    end

    reconstructions = predict(net, data_cell, 'MiniBatchSize', batch_size);

    % Compute errors
    num_samples = length(data_cell);
    scores = zeros(num_samples, 1);
    is_output_cell = iscell(reconstructions);

    for i = 1:num_samples
        original = data_cell{i};

        if is_output_cell
            reconstructed = reconstructions{i};
        else
            reconstructed = reconstructions(:, :, i);
        end

        % Compute error
        diff = original - reconstructed;

        switch lower(error_metric)
            case 'mse'
                scores(i) = mean(diff.^2, 'all');
            case 'mae'
                scores(i) = mean(abs(diff), 'all');
            case 'rmse'
                scores(i) = sqrt(mean(diff.^2, 'all'));
            otherwise
                error('Unknown error metric: %s', error_metric);
        end
    end

    if verbose
        fprintf('[Done]\n');
    end
end
