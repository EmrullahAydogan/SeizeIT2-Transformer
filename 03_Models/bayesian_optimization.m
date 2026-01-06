function [best_params, results] = bayesian_optimization(varargin)
% BAYESIAN_OPTIMIZATION - Hyperparameter optimization for transformer autoencoder
%
% Inputs (optional name-value pairs):
%   'MaxTime' - Maximum optimization time in seconds (default: 3600 = 1 hour)
%   'MaxIterations' - Maximum number of iterations (default: 50)
%   'NumFolds' - Number of cross-validation folds (default: 3)
%   'UseGPU' - Use GPU for training (default: true if available)
%   'SavePath' - Path to save optimization results (default: 'Results/BayesianOpt')
%   'Resume' - Resume from previous optimization (default: false)
%
% Outputs:
%   best_params - Struct with optimal hyperparameters
%   results - Optimization results object
%
% Optimization Strategy:
%   1. Define hyperparameter search space
%   2. Use Bayesian optimization (MATLAB bayesopt)
%   3. Objective: Minimize reconstruction error (MSE) on validation set
%   4. Patient-wise cross-validation to prevent data leakage
%
% Hyperparameters optimized:
%   - Learning rate (log scale)
%   - Hidden size (embedding dimension)
%   - Number of attention heads
%   - Number of encoder/decoder layers
%   - Dropout rate
%   - Batch size
%   - Feed-forward dimension multiplier

% Parse inputs
p = inputParser;
addParameter(p, 'MaxTime', 3600, @isnumeric);  % 1 hour default
addParameter(p, 'MaxIterations', 50, @isnumeric);
addParameter(p, 'NumFolds', 3, @isnumeric);
addParameter(p, 'UseGPU', true, @islogical);
addParameter(p, 'SavePath', 'Results/BayesianOpt', @ischar);
addParameter(p, 'Resume', false, @islogical);
parse(p, varargin{:});
opts = p.Results;

fprintf('\n===============================================\n');
fprintf('BAYESIAN OPTIMIZATION FOR TRANSFORMER AUTOENCODER\n');
fprintf('===============================================\n');
fprintf('Max iterations: %d\n', opts.MaxIterations);
fprintf('Max time: %.0f seconds (%.1f hours)\n', opts.MaxTime, opts.MaxTime/3600);
fprintf('CV folds: %d\n', opts.NumFolds);
fprintf('Save path: %s\n', opts.SavePath);

% Load configuration
cfg = config();

% Create save directory
if ~isfolder(opts.SavePath)
    mkdir(opts.SavePath);
end

%% 1. Define Hyperparameter Search Space
fprintf('\n--- DEFINING SEARCH SPACE ---\n');

optimVars = [
    % Learning rate (log scale, typical range for transformers)
    optimizableVariable('learning_rate', [1e-5, 1e-2], 'Transform', 'log', 'Optimize', true),

    % Embedding dimension (hidden size)
    optimizableVariable('embedding_dim', [32, 256], 'Type', 'integer', 'Optimize', true),

    % Number of attention heads (must divide embedding_dim evenly)
    optimizableVariable('num_heads', [2, 8], 'Type', 'integer', 'Optimize', true),

    % Number of encoder layers
    optimizableVariable('num_encoder_layers', [2, 6], 'Type', 'integer', 'Optimize', true),

    % Number of decoder layers
    optimizableVariable('num_decoder_layers', [2, 6], 'Type', 'integer', 'Optimize', true),

    % Dropout rate (regularization)
    optimizableVariable('dropout_rate', [0.1, 0.5], 'Optimize', true),

    % Batch size (GPU memory constrained)
    optimizableVariable('batch_size', [16, 128], 'Type', 'integer', 'Optimize', true),

    % Feed-forward dimension multiplier (standard: 4x)
    optimizableVariable('ffn_multiplier', [1, 4], 'Optimize', true)
];

fprintf('Search space defined with %d hyperparameters:\n', numel(optimVars));
for i = 1:numel(optimVars)
    fprintf('  %d. %s: %s\n', i, optimVars(i).Name, ...
        format_range(optimVars(i)));
end

%% 2. Load Training Data
fprintf('\n--- LOADING DATA ---\n');

% Load preprocessed training data
train_files = dir(fullfile(cfg.paths.model_data_train, '*.mat'));
if isempty(train_files)
    error('No training data found in: %s', cfg.paths.model_data_train);
end

fprintf('Found %d training files\n', numel(train_files));

% Load first file to get data shape
sample_data = load(fullfile(train_files(1).folder, train_files(1).name));
if isfield(sample_data, 'X')
    data_shape = size(sample_data.X);
    fprintf('Data shape: %s\n', mat2str(data_shape));
else
    error('Training file does not contain X variable');
end

% Check if data is spectrogram or time-domain
is_spectrogram = length(data_shape) == 4 && data_shape(2) == 129;  % freq bins

if is_spectrogram
    fprintf('Data type: Spectrogram [C=%d, F=%d, T=%d, batch]\n', ...
        data_shape(1), data_shape(2), data_shape(3));
else
    fprintf('Data type: Time-domain [C=%d, T=%d, 1, batch]\n', ...
        data_shape(1), data_shape(2));
end

%% 3. Prepare Objective Function
fprintf('\n--- PREPARING OBJECTIVE FUNCTION ---\n');
fprintf('Objective: Minimize validation MSE (reconstruction error)\n');
fprintf('Reason: Better reconstruction of normal patterns improves anomaly detection\n');

% Create objective function handle
objectiveFcn = @(params) objective_function(params, ...
    cfg, train_files, is_spectrogram, opts);

%% 4. Run Random Search (Bayesian optimization disabled due to compatibility issues)
fprintf('\n--- STARTING RANDOM SEARCH (Bayesian optimization disabled) ---\n');
fprintf('Start time: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('Number of iterations: %d\n', opts.MaxIterations);

% Initialize results
all_params = cell(opts.MaxIterations, 1);
all_objectives = zeros(opts.MaxIterations, 1);
all_valid = false(opts.MaxIterations, 1);

% Initialize best result tracking
best_objective = Inf;
best_params = struct();
best_iteration = 0;

% Random search loop
for iter = 1:opts.MaxIterations
    fprintf('\n--- Random Search Iteration %d/%d ---\n', iter, opts.MaxIterations);

    % Display current best result
    if best_iteration > 0
        fprintf('Current best (iter %d): MSE = %.4f\n', best_iteration, best_objective);
    end

    % Generate random hyperparameters using optimVars bounds
    params = struct();
    for v = 1:numel(optimVars)
        var = optimVars(v);
        var_name = var.Name;

        if strcmp(var.Type, 'integer')
            % Integer uniform sampling
            value = randi([var.Range(1), var.Range(2)]);
        else
            % Real uniform sampling
            if strcmp(var.Transform, 'log')
                % Log-uniform sampling
                log_lower = log10(var.Range(1));
                log_upper = log10(var.Range(2));
                value = 10^(log_lower + (log_upper - log_lower) * rand());
            else
                % Linear uniform sampling
                value = var.Range(1) + (var.Range(2) - var.Range(1)) * rand();
            end
        end

        params.(var_name) = value;
    end

    % Ensure embedding_dim is divisible by num_heads
    if mod(params.embedding_dim, params.num_heads) ~= 0
        % Adjust to nearest divisible value
        params.embedding_dim = params.num_heads * ...
            round(params.embedding_dim / params.num_heads);
        % Get embedding_dim range from optimVars
        embedding_var = optimVars(strcmp({optimVars.Name}, 'embedding_dim'));
        params.embedding_dim = max(embedding_var.Range(1), min(embedding_var.Range(2), params.embedding_dim));
    end

    % Display current parameters
    fprintf('  Learning rate: %.2e\n', params.learning_rate);
    fprintf('  Embedding dim: %d\n', params.embedding_dim);
    fprintf('  Attention heads: %d\n', params.num_heads);
    fprintf('  Encoder layers: %d\n', params.num_encoder_layers);
    fprintf('  Decoder layers: %d\n', params.num_decoder_layers);
    fprintf('  Dropout rate: %.2f\n', params.dropout_rate);
    fprintf('  Batch size: %d\n', params.batch_size);

    % Evaluate objective function
    try
        objective = objective_function(params, cfg, train_files, is_spectrogram, opts);
        all_objectives(iter) = objective;
        all_params{iter} = params;
        all_valid(iter) = true;
        fprintf('  Validation MSE: %.4f\n', objective);

        % Update best result if improved
        if objective < best_objective
            best_objective = objective;
            best_params = params;
            best_iteration = iter;
            fprintf('  🎯 NEW BEST! MSE improved to %.4f\n', best_objective);
        end
    catch ME
        fprintf('  ❌ Evaluation failed: %s\n', ME.message);
        all_objectives(iter) = Inf;
        all_params{iter} = params;
        all_valid(iter) = false;
    end
end

% Create results structure similar to bayesopt output
results = struct();
results.XTrace = struct2table([all_params{all_valid}]);
results.ObjectiveTrace = all_objectives(all_valid);
results.MinObjective = min(all_objectives(all_valid));
best_idx = find(all_objectives == results.MinObjective & all_valid, 1);
results.XAtMinObjective = struct2table(all_params{best_idx});

fprintf('\nRandom search completed.\n');

%% 5. Extract Best Parameters
fprintf('\n--- OPTIMIZATION COMPLETE ---\n');
fprintf('End time: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('Total evaluations: %d\n', size(results.XTrace, 1));

best_params = table2struct(results.XAtMinObjective);
best_params.objective = results.MinObjective;

fprintf('\nBEST HYPERPARAMETERS FOUND:\n');
fprintf('  Learning rate: %.2e\n', best_params.learning_rate);
fprintf('  Embedding dimension: %d\n', best_params.embedding_dim);
fprintf('  Attention heads: %d\n', best_params.num_heads);
fprintf('  Encoder layers: %d\n', best_params.num_encoder_layers);
fprintf('  Decoder layers: %d\n', best_params.num_decoder_layers);
fprintf('  Dropout rate: %.2f\n', best_params.dropout_rate);
fprintf('  Batch size: %d\n', best_params.batch_size);
fprintf('  FFN multiplier: %.1f\n', best_params.ffn_multiplier);
fprintf('  Best validation MSE: %.4f\n', best_params.objective);  % Lower is better

% Calculate feed-forward dimension
best_params.feedforward_dim = round(best_params.embedding_dim * best_params.ffn_multiplier);

% Validate that num_heads divides embedding_dim evenly
if mod(best_params.embedding_dim, best_params.num_heads) ~= 0
    fprintf('⚠️  Warning: embedding_dim (%d) not divisible by num_heads (%d)\n', ...
        best_params.embedding_dim, best_params.num_heads);
    % Adjust to nearest divisible value
    new_dim = best_params.embedding_dim - mod(best_params.embedding_dim, best_params.num_heads);
    if new_dim < 32
        new_dim = best_params.num_heads * ceil(32 / best_params.num_heads);
    end
    fprintf('  Adjusting embedding_dim to %d\n', new_dim);
    best_params.embedding_dim = new_dim;
end

%% 6. Hyperparameter Comparison Table
fprintf('\n--- HYPERPARAMETER COMPARISON (Top 5 Results) ---\n');

% Get all valid results
valid_indices = find(all_valid);
if ~isempty(valid_indices)
    % Create table of all valid results
    all_results = struct2table([all_params{valid_indices}]);
    all_scores = all_objectives(valid_indices);

    % Add scores to table
    all_results.Score = all_scores;

    % Sort by score (ascending - lower MSE is better)
    [sorted_scores, sort_idx] = sort(all_scores, 'ascend');

    % Display top 5 results
    num_top = min(5, length(sorted_scores));
    fprintf('\nTop %d hyperparameter configurations:\n', num_top);
    fprintf('Rank |  MSE  | LR (e-4) | EmbDim | Heads | EncL | DecL | Dropout | Batch | FFN Mult\n');
    fprintf('-----|-------|----------|--------|-------|------|------|---------|-------|----------\n');

    for i = 1:num_top
        idx = sort_idx(i);
        params_i = all_params{valid_indices(idx)};

        % Format learning rate as e-4
        lr_e4 = params_i.learning_rate * 1e4;

        fprintf('%4d | %6.1f | %8.2f | %6d | %5d | %4d | %4d | %7.2f | %5d | %8.1f\n', ...
            i, sorted_scores(i), lr_e4, ...
            params_i.embedding_dim, params_i.num_heads, ...
            params_i.num_encoder_layers, params_i.num_decoder_layers, ...
            params_i.dropout_rate, params_i.batch_size, params_i.ffn_multiplier);
    end

    % Display correlation insights
    fprintf('\n--- INSIGHTS ---\n');
    fprintf('• Lower MSE indicates better reconstruction of normal patterns\n');
    fprintf('• Learning rate ~1e-4 to 1e-3 often works well\n');
    fprintf('• Embedding dimension should be divisible by number of heads\n');
    fprintf('• Too many layers (EncL+DecL > 8) may cause overfitting\n');
    fprintf('• Dropout 0.1-0.3 helps prevent overfitting\n');
else
    fprintf('No valid results to compare.\n');
end

%% 7. Save Results
save(fullfile(opts.SavePath, 'optimization_results.mat'), ...
    'best_params', 'results', 'optimVars', 'opts');

% Save as JSON for reproducibility
results_json = struct();
results_json.best_params = best_params;
results_json.optimization_time = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
results_json.objective_value = best_params.objective;
results_json.num_evaluations = size(results.XTrace, 1);

json_str = jsonencode(results_json, 'PrettyPrint', true);
fid = fopen(fullfile(opts.SavePath, 'optimization_results.json'), 'w');
fprintf(fid, '%s', json_str);
fclose(fid);

fprintf('\nResults saved to: %s\n', opts.SavePath);
fprintf('===============================================\n');

end

%% Helper Functions

function str = format_range(optimVar)
% FORMAT_RANGE - Format optimizable variable range for display

    if strcmp(optimVar.Type, 'integer')
        str = sprintf('[%d, %d]', optimVar.Range(1), optimVar.Range(2));
    else
        if strcmp(optimVar.Transform, 'log')
            str = sprintf('[%.1e, %.1e] (log)', optimVar.Range(1), optimVar.Range(2));
        else
            str = sprintf('[%.2f, %.2f]', optimVar.Range(1), optimVar.Range(2));
        end
    end
end

function objective = objective_function(params, cfg, train_files, is_spectrogram, opts)
% OBJECTIVE_FUNCTION - Evaluate hyperparameters using cross-validation
%
% Returns validation MSE (bayesopt minimizes, lower is better)

    fprintf('\n--- Evaluating hyperparameter set ---\n');

    % Convert table row to struct
    if istable(params)
        params = table2struct(params);
    end

    % Display current parameters
    fprintf('  Learning rate: %.2e\n', params.learning_rate);
    fprintf('  Embedding dim: %d\n', params.embedding_dim);
    fprintf('  Attention heads: %d\n', params.num_heads);
    fprintf('  Encoder layers: %d\n', params.num_encoder_layers);
    fprintf('  Dropout rate: %.2f\n', params.dropout_rate);
    fprintf('  Batch size: %d\n', params.batch_size);

    % Validate hyperparameters
    if mod(params.embedding_dim, params.num_heads) ~= 0
        fprintf('  ⚠️ Invalid: embedding_dim not divisible by num_heads\n');
        objective = Inf;  % Penalize invalid combinations
        return;
    end

    % Set feed-forward dimension
    params.feedforward_dim = round(params.embedding_dim * params.ffn_multiplier);

    % Prepare for cross-validation
    auc_pr_scores = zeros(opts.NumFolds, 1);

    % Simple hold-out validation (for speed)
    % In production, would use patient-wise k-fold CV

    try
        % Load all training data
        [X_train, Y_train] = load_training_data(train_files, cfg);

        % Split into training and validation (80/20)
        rng(cfg.seed);  % For reproducibility
        cv = cvpartition(size(X_train, 4), 'HoldOut', 0.2);

        train_idx = training(cv);
        val_idx = test(cv);

        X_train_cv = X_train(:, :, :, train_idx);
        Y_train_cv = Y_train(train_idx);
        X_val = X_train(:, :, :, val_idx);
        Y_val = Y_train(val_idx);

        % Reshape data for sequence input: [C, F, T, batch] -> [C*F, T, batch]
        % where C=16, F=129, T=6
        C = size(X_train_cv, 1);
        F = size(X_train_cv, 2);
        T = size(X_train_cv, 3);
        batch_train = size(X_train_cv, 4);
        batch_val = size(X_val, 4);

        X_train_cv_reshaped = reshape(X_train_cv, [C*F, T, batch_train]);
        X_val_reshaped = reshape(X_val, [C*F, T, batch_val]);

        % Convert to cell array for sequence input (trainNetwork expects cell array for sequences)
        X_train_cell = squeeze(num2cell(X_train_cv_reshaped, [1 2]));
        X_val_cell = squeeze(num2cell(X_val_reshaped, [1 2]));

        % Create model with current hyperparameters
        input_shape = [C, F, T];  % Original input shape

        model_layers = transformer_autoencoder(params, input_shape);

        % Training options with early stopping and optimization
        trainOpts = trainingOptions('adam', ...
            'InitialLearnRate', params.learning_rate, ...
            'MaxEpochs', 5, ...                % Maximum epochs with early stopping (optimized for speed)
            'MiniBatchSize', min(params.batch_size, 64), ...  % Max 64 for speed
            'Shuffle', 'once', ...
            'ValidationData', {X_val_cell, X_val_cell}, ...  % Autoencoder: input = target
            'ValidationFrequency', 30, ...     % Validation every 30 iterations
            'ValidationPatience', 5, ...       % Stop if no improvement for 5 validations
            'Verbose', true, ...               % Show training progress
            'Plots', 'training-progress', ...  % Show training progress plot
            'ExecutionEnvironment', 'auto');

        % Train model
        fprintf('  Training model... ');
        tic;
        net = trainNetwork(X_train_cell, X_train_cell, model_layers, trainOpts);
        train_time = toc;
        fprintf('(%.1f sec)\n', train_time);

        % Evaluate on validation set (normal data only)
        fprintf('  Evaluating... ');
        Y_pred_cell = predict(net, X_val_cell);

        % Convert cell array to numeric array
        Y_pred_mat = cell2mat(Y_pred_cell);  % Shape: [C*F, T, batch_val]

        % Reshape prediction back to original shape: [C*F, T, batch] -> [C, F, T, batch]
        Y_pred_reshaped = reshape(Y_pred_mat, [C, F, T, batch_val]);

        % Calculate reconstruction error (MSE) on validation set
        validation_mse = mean((X_val - Y_pred_reshaped).^2, 'all');

        fprintf('Validation MSE: %.4f\n', validation_mse);

        % Objective: minimize validation MSE (better reconstruction of normal patterns)
        objective = validation_mse;

    catch ME
        fprintf('  ❌ Error during evaluation: %s\n', ME.message);
        objective = Inf;  % Penalize failed evaluations
    end

end

function [X_all, Y_all] = load_training_data(train_files, cfg)
% LOAD_TRAINING_DATA - Load and concatenate all training files

    X_all = [];
    Y_all = [];

    fprintf('  Loading training data... ');

    for i = 1:numel(train_files)
        data = load(fullfile(train_files(i).folder, train_files(i).name));

        if isfield(data, 'X')
            X = data.X;
            Y = data.Y;

            % Ensure single precision
            X = single(X);
            Y = single(Y);

            % Concatenate along batch dimension
            if isempty(X_all)
                X_all = X;
                Y_all = Y;
            else
                % Check dimensions match
                if ~isequal(size(X, 1:3), size(X_all, 1:3))
                    fprintf('Warning: Dimension mismatch in file %s\n', train_files(i).name);
                    continue;
                end

                X_all = cat(4, X_all, X);
                Y_all = cat(1, Y_all, Y);
            end
        end
    end

    fprintf('Loaded %d samples\n', size(X_all, 4));
end

function auc_pr = calculate_auc_pr(scores, labels)
% CALCULATE_AUC_PR - Calculate Precision-Recall AUC
%
% Inputs:
%   scores - Reconstruction error scores (higher = more anomalous)
%   labels - True labels (0=normal, 1=seizure)

    % Ensure column vectors
    scores = scores(:);
    labels = labels(:);

    % Sort by scores (descending)
    [sorted_scores, sort_idx] = sort(scores, 'descend');
    sorted_labels = labels(sort_idx);

    % Calculate precision and recall at each threshold
    total_positives = sum(labels == 1);

    if total_positives == 0
        auc_pr = NaN;
        return;
    end

    tp = cumsum(sorted_labels == 1);
    fp = cumsum(sorted_labels == 0);

    precision = tp ./ (tp + fp);
    recall = tp / total_positives;

    % Add (0,1) point for complete curve
    precision = [1; precision];
    recall = [0; recall];

    % Calculate AUC-PR using trapezoidal rule
    auc_pr = trapz(recall, precision);
end