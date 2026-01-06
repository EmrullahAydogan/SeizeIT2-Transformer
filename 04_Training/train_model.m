% train_model.m
% SeizeIT2 - Transformer Autoencoder Training (Refactored v2.1)
%
% IMPROVEMENTS OVER LEGACY:
%   - Centralized config system
%   - 50 epochs (vs 5 in legacy)
%   - Early stopping with patience
%   - Validation split (20%)
%   - Gradient clipping
%   - Learning rate schedule
%   - Optional Bayesian optimization
%   - Better logging and visualization
%   - Modular architecture
%
% Usage:
%   % Basic training (uses config parameters)
%   [net, trainInfo] = train_model();
%
%   % Override specific parameters
%   [net, trainInfo] = train_model('MaxEpochs', 100, 'MiniBatchSize', 16);
%
%   % With Bayesian Optimization (finds optimal hyperparameters first)
%   [net, trainInfo] = train_model('UseBayesianOpt', true);
%   [net, trainInfo] = train_model('UseBayesianOpt', true, 'BayesOptIterations', 20);
%
% Parameters:
%   'MaxEpochs'           - Maximum training epochs (default: 50)
%   'MiniBatchSize'       - Batch size (default: 32)
%   'InitialLearningRate' - Learning rate (default: 5.3e-4)
%   'Verbose'             - Verbose output (default: true)
%   'UseBayesianOpt'      - Run Bayesian optimization first (default: false)
%   'BayesOptIterations'  - Number of Bayesian opt iterations (default: 10)
%
% Author: SeizeIT2 Project
% Date: January 2025
% Version: 2.1

function [net, trainInfo] = train_model(varargin)
    clc;

    %% === CONFIGURATION ===
    % Load central configuration
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'config'));
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'utils'));
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '03_Models'));

    cfg = config();

    % Parse optional arguments (can override config)
    p = inputParser;
    addParameter(p, 'MaxEpochs', cfg.train.max_epochs, @isnumeric);
    addParameter(p, 'MiniBatchSize', cfg.train.min_batch_size, @isnumeric);
    addParameter(p, 'InitialLearningRate', cfg.train.initial_lr, @isnumeric);
    addParameter(p, 'Verbose', cfg.train.verbose, @islogical);
    addParameter(p, 'UseBayesianOpt', false, @islogical);  % Enable Bayesian optimization
    addParameter(p, 'BayesOptIterations', 10, @isnumeric); % Number of Bayesian opt iterations
    parse(p, varargin{:});

    use_bayesopt = p.Results.UseBayesianOpt;
    bayesopt_iters = p.Results.BayesOptIterations;
    max_epochs = p.Results.MaxEpochs;
    batch_size = p.Results.MiniBatchSize;
    initial_lr = p.Results.InitialLearningRate;
    verbose = p.Results.Verbose;

    fprintf('=== TRANSFORMER AUTOENCODER TRAINING ===\n');
    fprintf('Configuration: %s v%s\n', cfg.meta.project_name, cfg.meta.version);

    %% === BAYESIAN OPTIMIZATION (OPTIONAL) ===
    if use_bayesopt
        fprintf('\n========== BAYESIAN OPTIMIZATION ==========\n');
        fprintf('Running hyperparameter optimization...\n');
        fprintf('Iterations: %d\n', bayesopt_iters);
        fprintf('This may take 1-2 hours...\n\n');

        % Run Bayesian optimization
        try
            [best_params, ~] = bayesian_optimization(...
                'MaxIterations', bayesopt_iters, ...
                'UseGPU', cfg.bayesopt.use_gpu, ...
                'SavePath', fullfile(cfg.paths.results, 'BayesianOpt_AutoTrain'));

            % Override parameters with optimized values
            fprintf('\n========== OPTIMIZATION COMPLETE ==========\n');
            fprintf('Optimal hyperparameters found:\n');
            fprintf('  Learning Rate:      %.6f\n', best_params.learning_rate);
            fprintf('  Embedding Dim:      %d\n', best_params.embedding_dim);
            fprintf('  Attention Heads:    %d\n', best_params.num_heads);
            fprintf('  Encoder Layers:     %d\n', best_params.num_encoder_layers);
            fprintf('  Decoder Layers:     %d\n', best_params.num_decoder_layers);
            fprintf('  Dropout Rate:       %.4f\n', best_params.dropout_rate);
            fprintf('  Batch Size:         %d\n', best_params.batch_size);
            fprintf('  FFN Multiplier:     %.2f\n', best_params.ffn_multiplier);
            fprintf('  Objective (Error):  %.4f\n\n', best_params.objective);

            % Update config with optimal parameters
            cfg.model.embedding_dim = best_params.embedding_dim;
            cfg.model.num_heads = best_params.num_heads;
            cfg.model.num_encoder_layers = best_params.num_encoder_layers;
            cfg.model.num_decoder_layers = best_params.num_decoder_layers;
            cfg.model.dropout_rate = best_params.dropout_rate;
            cfg.model.feedforward_dim_multiplier = best_params.ffn_multiplier;
            cfg.train.initial_lr = best_params.learning_rate;
            cfg.train.min_batch_size = best_params.batch_size;

            % Update local variables
            initial_lr = best_params.learning_rate;
            batch_size = best_params.batch_size;

        catch ME
            fprintf('\n========== OPTIMIZATION FAILED ==========\n');
            fprintf('Error: %s\n', ME.message);
            fprintf('Falling back to config parameters...\n\n');
            % Continue with config parameters
        end
    else
        fprintf('Bayesian Optimization: DISABLED (using config parameters)\n');
    end

    fprintf('\nTraining Parameters:\n');
    fprintf('  Max Epochs: %d\n', max_epochs);
    fprintf('  Batch Size: %d\n', batch_size);
    fprintf('  Learning Rate: %.2e\n', initial_lr);
    fprintf('  Hardware: %s (%dGB)\n\n', cfg.hardware.gpu_name, cfg.hardware.gpu_memory_gb);

    %% === DATA LOADING ===
    fprintf('Loading training data...\n');

    trainDir = cfg.paths.model_data_train;

    % Create file datastore
    fds = fileDatastore(fullfile(trainDir, "*.mat"), ...
                        'ReadFcn', @readMatAsTable);

    % Transform for autoencoder (Input = Response)
    allDS = transform(fds, @addResponse);

    % Preview data to get dimensions
    try
        previewData = preview(allDS);
        inputCell = previewData{1, 1};
        inputSample = inputCell{1};
        [num_channels, num_timesteps] = size(inputSample);
        fprintf('Input shape: %d channels x %d timesteps\n', num_channels, num_timesteps);
    catch ME
        error('Data preview failed: %s', ME.message);
    end

    % VALIDATION SPLIT
    fprintf('Splitting data for validation...\n');
    numFiles = numpartitions(allDS);
    trainRatio = 1 - cfg.train.validation_fraction;
    trainIdx = floor(numFiles * trainRatio);

    trainDS = partition(allDS, 1, trainIdx);
    valDS = partition(allDS, trainIdx + 1, numFiles);

    fprintf('  Train files: %d (%.1f%%)\n', trainIdx, trainRatio * 100);
    fprintf('  Validation files: %d (%.1f%%)\n', numFiles - trainIdx, cfg.train.validation_fraction * 100);

    %% === MODEL ARCHITECTURE ===
    fprintf('\nBuilding Transformer-Autoencoder architecture...\n');

    embedding_dim = cfg.model.embedding_dim;
    num_heads = cfg.model.num_heads;
    conv_kernel = cfg.model.conv_kernel_size;

    fprintf('  - Embedding dim: %d\n', embedding_dim);
    fprintf('  - Attention heads: %d\n', num_heads);
    fprintf('  - Conv kernel size: %d\n', conv_kernel);

    lgraph = layerGraph();

    layers = [
        % INPUT
        sequenceInputLayer(num_channels, 'MinLength', num_timesteps, ...
                          'Name', 'input', 'Normalization', 'zscore')

        % ENCODER: Embedding
        convolution1dLayer(conv_kernel, embedding_dim, ...
                          'Padding', 'same', 'Name', 'embed_conv')
        layerNormalizationLayer('Name', 'ln1')
        reluLayer('Name', 'relu1')

        % TRANSFORMER: Self-Attention
        selfAttentionLayer(num_heads, embedding_dim, 'Name', 'attention')
        layerNormalizationLayer('Name', 'ln2')
        reluLayer('Name', 'relu2')

        % DECODER: Reconstruction
        convolution1dLayer(conv_kernel, num_channels, ...
                          'Padding', 'same', 'Name', 'decode_conv')

        % OUTPUT
        regressionLayer('Name', 'output')
    ];

    lgraph = addLayers(lgraph, layers);

    % Verify architecture
    if verbose
        try
            analyzeNetwork(lgraph);
            fprintf('Architecture validated.\n');
        catch
            fprintf('Warning: Cannot display network analyzer (no GUI).\n');
        end
    end

    %% === TRAINING OPTIONS ===
    fprintf('\nConfiguring training...\n');

    % Base options
    options = trainingOptions(cfg.train.optimizer, ...
        'MaxEpochs', max_epochs, ...
        'MiniBatchSize', batch_size, ...
        'InitialLearnRate', initial_lr, ...
        'Shuffle', cfg.train.shuffle, ...
        'Plots', cfg.train.plots, ...
        'Verbose', verbose, ...
        'ExecutionEnvironment', cfg.train.execution_env, ...
        'ValidationData', valDS, ...
        'ValidationFrequency', 50, ...
        'CheckpointPath', cfg.paths.checkpoints, ...
        'OutputNetwork', 'best-validation-loss', ...
        'GradientThreshold', cfg.train.gradient_threshold, ...
        'GradientThresholdMethod', 'l2norm');

    % Early stopping callback
    if cfg.train.early_stopping
        options.OutputFcn = @early_stopping_callback;
        fprintf('  Early stopping: ENABLED (patience=%d epochs, min_delta=%.6f)\n', ...
            cfg.train.patience, cfg.train.min_delta);
    else
        fprintf('  Early stopping: DISABLED\n');
    end

    % Learning rate schedule
    if ~strcmp(cfg.train.lr_schedule, 'none')
        options.LearnRateSchedule = cfg.train.lr_schedule;
        options.LearnRateDropPeriod = cfg.train.lr_drop_period;
        options.LearnRateDropFactor = cfg.train.lr_drop_factor;
        fprintf('  Learning rate schedule: %s (drop %.1f%% every %d epochs)\n', ...
            cfg.train.lr_schedule, (1-cfg.train.lr_drop_factor)*100, cfg.train.lr_drop_period);
    end

    %% === TRAINING ===
    fprintf('\n========== TRAINING STARTED ==========\n');
    fprintf('Monitor training progress in the plot window.\n');
    fprintf('Checkpoints will be saved to: %s\n\n', cfg.paths.checkpoints);

    tic;
    try
        [net, trainInfo] = trainNetwork(trainDS, lgraph, options);
        elapsed = toc;

        fprintf('\n========== TRAINING COMPLETED ==========\n');
        fprintf('Total time: %.2f minutes\n', elapsed/60);
        fprintf('Final training loss: %.4f\n', trainInfo.TrainingLoss(end));

        % Display validation results if available
        if isfield(trainInfo, 'ValidationLoss') && ~isempty(trainInfo.ValidationLoss)
            fprintf('Final validation loss: %.4f\n', trainInfo.ValidationLoss(end));
            fprintf('Best validation loss: %.4f\n', min(trainInfo.ValidationLoss));
        end

        %% === SAVE MODEL ===
        timestamp = datestr(now, cfg.log.timestamp_format);

        % Add optimization info to filename if used
        if use_bayesopt
            model_filename = sprintf('Trained_Transformer_BayesOpt_%s.mat', timestamp);
        else
            model_filename = sprintf('Trained_Transformer_%s.mat', timestamp);
        end

        model_path = fullfile(cfg.paths.models, model_filename);

        % Also save as "Latest" for easy access
        latest_path = fullfile(cfg.paths.models, 'Trained_Transformer_Latest.mat');

        % Store optimization metadata
        training_metadata = struct();
        training_metadata.used_bayesopt = use_bayesopt;
        training_metadata.bayesopt_iterations = bayesopt_iters;
        training_metadata.final_learning_rate = initial_lr;
        training_metadata.final_batch_size = batch_size;
        training_metadata.timestamp = timestamp;

        save(model_path, 'net', 'trainInfo', 'cfg', 'training_metadata', '-v7.3');
        save(latest_path, 'net', 'trainInfo', 'cfg', 'training_metadata', '-v7.3');

        fprintf('\nModel saved:\n');
        fprintf('  - %s\n', model_filename);
        fprintf('  - Trained_Transformer_Latest.mat\n');

        %% === TRAINING CURVES ===
        % Save training and validation curves
        fig = figure('Name', 'Training Progress', 'Visible', 'off', 'Position', [100, 100, 1200, 400]);

        % Training loss
        subplot(1, 2, 1);
        plot(trainInfo.TrainingLoss, 'b-', 'LineWidth', 2);
        xlabel('Iteration');
        ylabel('Loss');
        title(sprintf('Training Loss (Final: %.4f)', trainInfo.TrainingLoss(end)));
        grid on;

        % Validation loss (if available)
        if isfield(trainInfo, 'ValidationLoss') && ~isempty(trainInfo.ValidationLoss)
            subplot(1, 2, 2);
            plot(trainInfo.ValidationLoss, 'r-', 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Loss');
            title(sprintf('Validation Loss (Best: %.4f)', min(trainInfo.ValidationLoss)));
            grid on;
        end

        curve_path = fullfile(cfg.paths.figures, sprintf('Training_Curve_%s.png', timestamp));
        saveas(fig, curve_path);
        close(fig);

        fprintf('  - Training curve: %s\n', curve_path);

        %% === SAVE FINAL SUMMARY ===
        save_training_summary(trainInfo, cfg, model_path, timestamp, use_bayesopt, false);

    catch ME
        fprintf('\n========== TRAINING FAILED/INTERRUPTED ==========\n');
        fprintf('Error: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('Location: %s (Line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end

        % Try to save partial results if training was interrupted
        try
            if exist('trainInfo', 'var') && exist('net', 'var')
                fprintf('\n[Recovery] Saving partial training results...\n');

                timestamp = datestr(now, cfg.log.timestamp_format);
                partial_filename = sprintf('Partial_Training_%s.mat', timestamp);
                partial_path = fullfile(cfg.paths.models, partial_filename);

                training_metadata = struct();
                training_metadata.used_bayesopt = use_bayesopt;
                training_metadata.training_interrupted = true;
                training_metadata.error_message = ME.message;
                training_metadata.timestamp = timestamp;

                save(partial_path, 'net', 'trainInfo', 'cfg', 'training_metadata', '-v7.3');
                fprintf('[Recovery] Partial model saved: %s\n', partial_filename);

                % Save summary for partial results
                save_training_summary(trainInfo, cfg, partial_path, timestamp, use_bayesopt, true);
            end
        catch
            fprintf('[Recovery] Could not save partial results\n');
        end

        rethrow(ME);
    end
end

%% === HELPER FUNCTION: SAVE TRAINING SUMMARY ===
function save_training_summary(trainInfo, cfg, model_path, timestamp, used_bayesopt, is_partial)
    % SAVE_TRAINING_SUMMARY - Save comprehensive training results

    summary_filename = sprintf('training_summary_%s.txt', timestamp);
    summary_path = fullfile(cfg.paths.results, summary_filename);

    fid = fopen(summary_path, 'w');
    fprintf(fid, '==========================================================\n');
    if is_partial
        fprintf(fid, 'PARTIAL TRAINING SUMMARY (Interrupted/Early Stopped)\n');
    else
        fprintf(fid, 'TRAINING SUMMARY - COMPLETE\n');
    end
    fprintf(fid, '==========================================================\n\n');

    % Timestamp and duration
    fprintf(fid, 'Timestamp: %s\n', timestamp);
    fprintf(fid, 'Model Path: %s\n\n', model_path);

    % Bayesian optimization info
    if used_bayesopt
        fprintf(fid, 'Bayesian Optimization: USED\n');
    else
        fprintf(fid, 'Bayesian Optimization: NOT USED (config parameters)\n');
    end
    fprintf(fid, '\n');

    % Training configuration
    fprintf(fid, '--- CONFIGURATION ---\n');
    fprintf(fid, 'Max Epochs: %d\n', cfg.train.max_epochs);
    fprintf(fid, 'Batch Size: %d\n', cfg.train.min_batch_size);
    fprintf(fid, 'Initial Learning Rate: %.6f\n', cfg.train.initial_lr);
    fprintf(fid, 'Embedding Dim: %d\n', cfg.model.embedding_dim);
    fprintf(fid, 'Attention Heads: %d\n', cfg.model.num_heads);
    fprintf(fid, 'Dropout Rate: %.4f\n', cfg.model.dropout_rate);
    fprintf(fid, '\n');

    % Training results
    fprintf(fid, '--- TRAINING RESULTS ---\n');
    fprintf(fid, 'Total Epochs: %d\n', numel(trainInfo.TrainingLoss));
    fprintf(fid, 'Final Training Loss: %.8f\n', trainInfo.TrainingLoss(end));

    if isfield(trainInfo, 'ValidationLoss') && ~isempty(trainInfo.ValidationLoss)
        fprintf(fid, 'Final Validation Loss: %.8f\n', trainInfo.ValidationLoss(end));
        fprintf(fid, 'Best Validation Loss: %.8f\n', min(trainInfo.ValidationLoss));
        [~, best_epoch] = min(trainInfo.ValidationLoss);
        fprintf(fid, 'Best Epoch: %d\n', best_epoch);
    end

    fprintf(fid, '\n');

    % Loss progression (every 10%%)
    fprintf(fid, '--- LOSS PROGRESSION ---\n');
    num_epochs = numel(trainInfo.TrainingLoss);
    milestone_epochs = unique(round(linspace(1, num_epochs, min(11, num_epochs))));

    fprintf(fid, 'Epoch | Train Loss | Val Loss\n');
    fprintf(fid, '------|------------|----------\n');
    for i = 1:length(milestone_epochs)
        ep = milestone_epochs(i);
        train_loss = trainInfo.TrainingLoss(ep);

        if isfield(trainInfo, 'ValidationLoss') && ~isempty(trainInfo.ValidationLoss) && ep <= length(trainInfo.ValidationLoss)
            val_loss = trainInfo.ValidationLoss(ep);
            fprintf(fid, '%5d | %10.6f | %8.6f\n', ep, train_loss, val_loss);
        else
            fprintf(fid, '%5d | %10.6f | N/A\n', ep, train_loss);
        end
    end

    fprintf(fid, '\n');

    % Model info
    fprintf(fid, '--- MODEL INFORMATION ---\n');
    fprintf(fid, 'Network Type: DAGNetwork\n');
    fprintf(fid, 'Architecture: Transformer Autoencoder\n');

    if is_partial
        fprintf(fid, '\nNOTE: Training was interrupted or early stopped.\n');
        fprintf(fid, 'This model may not have reached full convergence.\n');
    end

    fprintf(fid, '\n==========================================================\n');
    fclose(fid);

    fprintf('\n[Summary] Training summary saved: %s\n', summary_filename);
end

%% === HELPER FUNCTIONS ===

function T = readMatAsTable(filename)
    % Load .mat file and convert to table format for datastore
    d = load(filename);
    rawX = d.X;  % [channels, timesteps, 1, batch]
    rawX = squeeze(rawX);  % [channels, timesteps, batch]

    % Convert to cell array
    num_samples = size(rawX, 3);
    dataCell = cell(num_samples, 1);
    for i = 1:num_samples
        dataCell{i} = rawX(:, :, i);
    end

    T = table(dataCell, 'VariableNames', {'Input'});
end

function T_out = addResponse(T_in)
    % For autoencoder: Response = Input
    T_out = T_in;
    T_out.Response = T_in.Input;
end
