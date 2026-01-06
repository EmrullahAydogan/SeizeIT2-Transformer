function cfg = config()
% CONFIG - Central configuration file for SeizeIT2 project
% Returns a struct containing all project parameters
%
% Usage:
%   cfg = config();
%   cfg.data.fs  % Access sampling frequency
%
% Author: SeizeIT2 Project
% Date: January 2025

    cfg = struct();

    %% ========== PATH CONFIGURATION ==========
    % Base directory (auto-detect)
    cfg.paths.base = fileparts(fileparts(mfilename('fullpath')));

    % Data paths
    cfg.paths.raw_dataset = '/home/developer/Desktop/SeizeIT2/dataset';
    cfg.paths.processed = fullfile(cfg.paths.base, 'Data', 'Processed');
    cfg.paths.model_data_train = fullfile(cfg.paths.base, 'Data', 'ModelData', 'Train');
    cfg.paths.model_data_test = fullfile(cfg.paths.base, 'Data', 'ModelData', 'Test');
    cfg.paths.models = fullfile(cfg.paths.base, 'Data', 'ModelData', 'Models');
    cfg.paths.metadata = fullfile(cfg.paths.base, 'Data', 'Metadata');

    % Results paths
    cfg.paths.results = fullfile(cfg.paths.base, 'Results');
    cfg.paths.figures = fullfile(cfg.paths.base, 'Results', 'Figures');
    cfg.paths.tables = fullfile(cfg.paths.base, 'Results', 'Tables');
    cfg.paths.checkpoints = fullfile(cfg.paths.base, 'Results', 'Checkpoints');

    % Create directories if they don't exist
    dirs_to_create = {cfg.paths.processed, cfg.paths.model_data_train, ...
                      cfg.paths.model_data_test, cfg.paths.models, ...
                      cfg.paths.metadata, cfg.paths.figures, ...
                      cfg.paths.tables, cfg.paths.checkpoints};
    for i = 1:length(dirs_to_create)
        if ~isfolder(dirs_to_create{i})
            mkdir(dirs_to_create{i});
        end
    end

    %% ========== DATA PROCESSING PARAMETERS ==========
    cfg.data.fs = 250;              % Target sampling frequency (Hz)
    cfg.data.window_size_sec = 4;   % Window size in seconds
    cfg.data.stride_sec = 2;        % Stride in seconds (50% overlap)
    cfg.data.window_size = cfg.data.fs * cfg.data.window_size_sec;
    cfg.data.stride = cfg.data.fs * cfg.data.stride_sec;

    % Seizure labeling threshold
    cfg.data.seizure_threshold = 0.2;  % 20% of window must be seizure

    % Data split ratio
    cfg.data.test_ratio = 0.2;      % 20% for test (normal data only)

    % Transformer-specific preprocessing
    cfg.data.transformer_preprocessing = struct();
    cfg.data.transformer_preprocessing.enable = true;  % Enable transformer-specific steps
    cfg.data.transformer_preprocessing.add_positional_encoding = true; % Add positional encoding to windows
    cfg.data.transformer_preprocessing.normalization_type = 'zscore';  % 'zscore', 'minmax', 'robust'
    cfg.data.transformer_preprocessing.spectrogram_enable = true;  % Convert to spectrogram
    cfg.data.transformer_preprocessing.spectrogram_window = 256;  % Window size for spectrogram
    cfg.data.transformer_preprocessing.spectrogram_overlap = 128; % Overlap for spectrogram
    cfg.data.transformer_preprocessing.data_augmentation = true;  % Enable data augmentation
    cfg.data.transformer_preprocessing.augmentation_methods = {'time_warp', 'jitter', 'scaling'};

    %% ========== PATIENT SELECTION CRITERIA ==========
    cfg.patient.min_duration_hours = 18;
    cfg.patient.min_seizures = 5;
    cfg.patient.require_mov = true;
    cfg.patient.preferred_vigilance = 'MIXED';  % MIXED > Wake Only > Sleep Only

    % Quality score parameters for patient selection
    cfg.quality_score.weights = struct(...
        'duration', 25, ...
        'seizure_count', 25, ...
        'vigilance', 30, ...
        'modality', 20);

    cfg.quality_score.thresholds = struct(...
        'duration', [10, 15, 18, 20], ...  % hours
        'seizure_count', [3, 5, 7, 10]);   % number of seizures

    % Currently selected patients (based on analysis - updated 2026-01-04)
    cfg.patient.selected = ["sub-039", "sub-015", "sub-022"];

    %% ========== MODEL ARCHITECTURE ==========
    cfg.model.name = 'TransformerAutoencoder';

    % OPTIMIZED HYPERPARAMETERS (from Bayesian Optimization - GPU Constrained)
    % Based on BayesianOpt_Final_10iter results (Objective: 22.9641)
    % Adjusted for RTX 4070 8GB memory constraints
    cfg.model.embedding_dim = 128;              % Optimal: 186, reduced for GPU (was 64)
    cfg.model.num_heads = 6;                    % Optimal: 6 (was 4)
    cfg.model.num_encoder_layers = 4;           % Optimal: 4 (was 3)
    cfg.model.num_decoder_layers = 5;           % Optimal: 6, reduced to 5 for GPU (was 3)
    cfg.model.dropout_rate = 0.30;              % Optimal: 0.3606, rounded to 0.30 (was 0.1)
    cfg.model.feedforward_dim_multiplier = 3.0; % Optimal: 3.17, rounded to 3.0 (was 4)
    cfg.model.conv_kernel_size = 5;             % 1D convolution kernel size (~20ms @ 250Hz)
    cfg.model.activation = 'relu';              % 'relu' for better stability
    cfg.model.positional_encoding_type = 'sinusoidal';  % 'sinusoidal' or 'learned'
    cfg.model.use_skip_connections = false;

    % Input shape based on preprocessing
    if cfg.data.transformer_preprocessing.spectrogram_enable
        % Spectrogram shape: [channels=16, freq_bins=129, time_frames=7]
        cfg.model.input_channels = 16;
        cfg.model.input_freq_bins = 129;
        cfg.model.input_time_frames = 7;
        cfg.model.input_shape = [16, 129, 7];
        cfg.model.is_spectrogram = true;
    else
        % Time-domain shape: [channels=16, timesteps=1000, 1]
        cfg.model.input_channels = 16;
        cfg.model.input_timesteps = 1000;
        cfg.model.input_shape = [16, 1000, 1];
        cfg.model.is_spectrogram = false;
    end

    % Model variants for ablation study
    cfg.model.variants = struct();
    cfg.model.variants.baseline_lstm = struct('hidden_size', 128, 'num_layers', 2);
    cfg.model.variants.small_transformer = struct('embedding_dim', 32, 'num_heads', 2);
    cfg.model.variants.deep_transformer = struct('embedding_dim', 128, 'num_heads', 8, ...
        'num_encoder_layers', 6, 'num_decoder_layers', 6);

    %% ========== BAYESIAN OPTIMIZATION ==========
    cfg.bayesopt.max_iterations = 50;
    cfg.bayesopt.max_time_seconds = 3600;  % 1 hour
    cfg.bayesopt.num_cv_folds = 3;
    cfg.bayesopt.use_gpu = true;
    cfg.bayesopt.save_path = fullfile(cfg.paths.results, 'BayesianOpt');
    cfg.bayesopt.metric = 'AUC-PR';  % Primary optimization metric

    % Hyperparameter search ranges
    cfg.bayesopt.search_space = struct();
    cfg.bayesopt.search_space.learning_rate = [1e-5, 1e-2];  % log scale
    cfg.bayesopt.search_space.embedding_dim = [32, 256];     % integer
    cfg.bayesopt.search_space.num_heads = [2, 8];           % integer
    cfg.bayesopt.search_space.num_encoder_layers = [2, 6];  % integer
    cfg.bayesopt.search_space.num_decoder_layers = [2, 6];  % integer
    cfg.bayesopt.search_space.dropout_rate = [0.1, 0.5];
    cfg.bayesopt.search_space.batch_size = [16, 128];       % integer
    cfg.bayesopt.search_space.ffn_multiplier = [1, 4];

    %% ========== TRAINING PARAMETERS ==========
    cfg.train.optimizer = 'adam';
    cfg.train.initial_lr = 5.3e-4;      % Optimal: 0.00053 (was 1e-4)
    cfg.train.max_epochs = 50;          % Increased from 5
    cfg.train.min_batch_size = 32;      % Optimal: 117, reduced to 32 for GPU (was 8)
    cfg.train.validation_fraction = 0.2; % 20% of data for validation
    cfg.train.shuffle = 'every-epoch';
    cfg.train.verbose = true;
    cfg.train.plots = 'training-progress';
    cfg.train.execution_env = 'auto';   % 'auto', 'gpu', 'cpu'
    cfg.train.gradient_threshold = 1;   % Gradient clipping to prevent explosion

    % Early stopping
    cfg.train.early_stopping = true;
    cfg.train.patience = 10;            % Stop if no improvement for 10 epochs
    cfg.train.min_delta = 1e-4;         % Minimum change to qualify as improvement

    % Learning rate schedule
    cfg.train.lr_schedule = 'piecewise'; % 'none', 'piecewise', 'exponential'
    cfg.train.lr_drop_period = 20;       % Drop LR every 20 epochs
    cfg.train.lr_drop_factor = 0.5;      % Multiply LR by 0.5 at drop

    %% ========== EVALUATION PARAMETERS ==========
    cfg.eval.metrics = {'AUC', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'Accuracy'};
    cfg.eval.threshold_method = 'youden';  % 'youden', 'fixed', 'optimal_f1'
    cfg.eval.cross_validation = 'LOPO';    % Leave-One-Patient-Out
    cfg.eval.statistical_test = 'ranksum'; % Mann-Whitney U test
    cfg.eval.alpha = 0.05;                 % Significance level

    %% ========== VISUALIZATION PARAMETERS ==========
    cfg.viz.figure_format = 'png';
    cfg.viz.figure_dpi = 300;
    cfg.viz.colormap = 'jet';
    cfg.viz.font_size = 12;

    %% ========== REPRODUCIBILITY ==========
    cfg.seed = 42;                  % Random seed for reproducibility
    rng(cfg.seed);                  % Set random seed

    % Version control and reproducibility
    cfg.reproducibility.git_commit = get_git_commit();  % Function to get git hash
    cfg.reproducibility.timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    cfg.reproducibility.matlab_version = version();
    cfg.reproducibility.platform = computer();

    %% ========== HARDWARE CONSTRAINTS ==========
    cfg.hardware.gpu_name = 'RTX 4070';
    cfg.hardware.gpu_memory_gb = 8;
    cfg.hardware.max_patients = 3;  % Due to GPU memory limitation

    %% ========== LOGGING ==========
    cfg.log.verbose = true;
    cfg.log.save_intermediate = true;
    cfg.log.timestamp_format = 'yyyy-mm-dd_HH-MM-SS';

    %% ========== PROJECT METADATA ==========
    cfg.meta.project_name = 'SeizeIT2-Transformer';
    cfg.meta.version = '2.0.0';     % Refactored version
    cfg.meta.author = 'Emrullah Aydogan';
    cfg.meta.institution = '[Your Institution]';
    cfg.meta.date_created = '2025-01-04';

    fprintf('Configuration loaded successfully.\n');
    fprintf('Project: %s v%s\n', cfg.meta.project_name, cfg.meta.version);
    fprintf('Base path: %s\n', cfg.paths.base);
end

%% ========== LOCAL FUNCTIONS ==========
function commit_hash = get_git_commit()
    % GET_GIT_COMMIT - Get current git commit hash for reproducibility
    % Returns git hash if available, otherwise 'unknown'

    commit_hash = 'unknown';

    % Try to get git commit hash
    try
        [status, result] = system('git rev-parse --short HEAD');
        if status == 0
            commit_hash = strtrim(result);
        else
            % If not in git repo, try to get from .git folder
            if exist('.git', 'dir')
                [status, result] = system('git log -1 --format="%H"');
                if status == 0
                    commit_hash = strtrim(result);
                end
            end
        end
    catch
        % If any error, return 'unknown'
        commit_hash = 'unknown';
    end
end
