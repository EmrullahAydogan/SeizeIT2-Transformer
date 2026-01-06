% preprocess_pipeline.m
% SeizeIT2 - Comprehensive Preprocessing Pipeline for Transformer Autoencoder
% Academic Version with Enhanced Features
%
% This script combines raw data processing and window creation into a single
% pipeline with academic features:
% 1. Config-based parameter management
% 2. Comprehensive error handling and logging
% 3. Data quality assessment (SNR, artifacts, missing data)
% 4. Reproducibility tracking (git hash, timestamp, parameters)
% 5. Transformer-autoencoder specific preprocessing options
% 6. Publication-ready reporting
%
% Author: SeizeIT2 Project
% Date: 2026-01-04
% Version: 2.0.0

clc; clear; close all;

%% ==================== INITIALIZATION ====================
fprintf('===============================================\n');
fprintf('SEIZEIT2 - PREPROCESSING PIPELINE (Academic Version)\n');
fprintf('===============================================\n');

% Start timer for performance measurement
pipelineTimer = tic;

% Add config path
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', 'config'));

% Load configuration
cfg = config();
fprintf('Configuration loaded: %s v%s\n', cfg.meta.project_name, cfg.meta.version);
fprintf('Selected patients: %s\n', strjoin(cfg.patient.selected, ', '));

% Set random seed for reproducibility
rng(cfg.seed);
fprintf('Random seed set to: %d\n', cfg.seed);

%% ==================== REPRODUCIBILITY SETUP ====================
% Create reproducibility log
repro_log = struct();
repro_log.pipeline_start_time = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss');
repro_log.git_commit = cfg.reproducibility.git_commit;
repro_log.matlab_version = cfg.reproducibility.matlab_version;
repro_log.platform = cfg.reproducibility.platform;
repro_log.config_hash = string2hash(jsonencode(cfg)); % Simple hash for config

% Save reproducibility info
repro_path = fullfile(cfg.paths.processed, 'reproducibility_preprocess.json');
fid = fopen(repro_path, 'w');
fprintf(fid, '%s', jsonencode(repro_log));
fclose(fid);
fprintf('Reproducibility log saved: %s\n', repro_path);

%% ==================== PARAMETER VALIDATION ====================
fprintf('\n--- PARAMETER VALIDATION ---\n');

% Validate target sampling frequency
valid_fs = [250, 500, 1000]; % Common EEG sampling frequencies
if ~ismember(cfg.data.fs, valid_fs)
    warning('Target sampling frequency %d Hz is unusual for EEG data.', cfg.data.fs);
end

% Validate window parameters
if cfg.data.window_size_sec < 1 || cfg.data.window_size_sec > 10
    error('Window size %.1f seconds is outside recommended range [1, 10].', cfg.data.window_size_sec);
end

% Validate stride (overlap)
overlap_percent = (1 - cfg.data.stride_sec/cfg.data.window_size_sec) * 100;
fprintf('Window size: %.1f seconds (%d samples)\n', cfg.data.window_size_sec, cfg.data.window_size);
fprintf('Stride: %.1f seconds (%d samples)\n', cfg.data.stride_sec, cfg.data.stride);
fprintf('Overlap: %.1f%%\n', overlap_percent);

% Validate seizure threshold
if cfg.data.seizure_threshold < 0 || cfg.data.seizure_threshold > 1
    error('Seizure threshold %.2f must be between 0 and 1.', cfg.data.seizure_threshold);
end

%% ==================== PATIENT PROCESSING LOOP ====================
fprintf('\n--- PATIENT PROCESSING ---\n');
fprintf('Processing %d patients...\n', length(cfg.patient.selected));

% Initialize quality metrics table
quality_metrics = table();
quality_metrics.Properties.Description = 'Data Quality Metrics for Processed Patients';

% Initialize summary statistics
summary_stats = struct();
summary_stats.total_patients = length(cfg.patient.selected);
summary_stats.total_train_windows = 0;
summary_stats.total_test_windows = 0;
summary_stats.total_seizure_windows = 0;
summary_stats.total_normal_windows = 0;

for pIdx = 1:length(cfg.patient.selected)
    patient_id = cfg.patient.selected(pIdx);
    patient_timer = tic;

    fprintf('\n===============================================\n');
    fprintf('PATIENT %d/%d: %s\n', pIdx, length(cfg.patient.selected), patient_id);
    fprintf('===============================================\n');

    try
        %% ----- STEP 1: RAW DATA PROCESSING -----
        fprintf('[1/4] Raw data processing...\n');

        basePath = fullfile(cfg.paths.raw_dataset, patient_id, "ses-01");

        % Check if raw data exists
        if ~isfolder(basePath)
            error('Raw data directory not found: %s', basePath);
        end

        % Process modalities
        tablesToSync = {};

        tablesToSync{end+1} = processModalityV8(fullfile(basePath, 'eeg'), 'EEG', cfg.data.fs);
        tablesToSync{end+1} = processModalityV8(fullfile(basePath, 'ecg'), 'ECG', cfg.data.fs);
        tablesToSync{end+1} = processModalityV8(fullfile(basePath, 'emg'), 'EMG', cfg.data.fs);
        tablesToSync{end+1} = processModalityV8(fullfile(basePath, 'mov'), 'MOV', cfg.data.fs);

        % Remove empty tables
        tablesToSync = tablesToSync(~cellfun('isempty', tablesToSync));

        if isempty(tablesToSync)
            error('No signal data could be read for patient %s.', patient_id);
        end

        % Synchronize and interpolate
        fprintf('   > Synchronizing signals... ');
        fullData = synchronize(tablesToSync{:}, 'union', 'linear');
        fullData = fillmissing(fullData, 'linear');
        fprintf('[%d channels, %.1f hours]\n', width(fullData), hours(fullData.Time(end) - fullData.Time(1)));

        %% ----- STEP 2: SEIZURE LABELING -----
        fprintf('[2/4] Seizure labeling...\n');

        labels = zeros(height(fullData), 1, 'int8');
        eventFiles = dir(fullfile(basePath, 'eeg', '*events.tsv'));

        if ~isempty(eventFiles)
            opts = detectImportOptions(fullfile(eventFiles(1).folder, eventFiles(1).name), 'FileType', 'text');
            opts.VariableNamingRule = 'preserve';
            events = readtable(fullfile(eventFiles(1).folder, eventFiles(1).name), opts);

            cols = events.Properties.VariableNames;
            typeCol = cols{contains(lower(cols), 'type')};
            onsetCol = cols{contains(lower(cols), 'onset')};
            durCol = cols{contains(lower(cols), 'duration')};

            if ~isempty(typeCol)
                isSeizure = contains(lower(string(events.(typeCol))), 'sz', 'IgnoreCase', true);
                szEvents = events(isSeizure, :);

                if height(szEvents) > 0
                    fprintf('   > Labeling %d seizures... ', height(szEvents));

                    timeVec = seconds(fullData.Time - fullData.Time(1));

                    for k = 1:height(szEvents)
                        startT = szEvents.(onsetCol)(k);
                        endT = startT + szEvents.(durCol)(k);
                        idx = (timeVec >= startT) & (timeVec <= endT);
                        labels(idx) = 1;
                    end
                    fprintf('[%.1f%% seizure samples]\n', 100 * sum(labels) / length(labels));
                else
                    fprintf('   > No seizures found (clean patient).\n');
                end
            end
        end

        %% ----- STEP 3: DATA QUALITY ASSESSMENT -----
        fprintf('[3/4] Data quality assessment...\n');

        % Calculate basic quality metrics
        rawSignals = single(table2array(fullData));

        % Signal-to-Noise Ratio (SNR) estimation
        snr_vals = zeros(1, size(rawSignals, 2));
        for ch = 1:size(rawSignals, 2)
            signal = rawSignals(:, ch);
            % Simple SNR estimation using percentile method
            signal_power = var(signal);
            noise_est = mad(signal, 1)^2; % Median absolute deviation as noise estimate
            snr_vals(ch) = 10 * log10(signal_power / (noise_est + eps));
        end

        % Artifact detection (simple threshold-based)
        artifact_ratio = zeros(1, size(rawSignals, 2));
        for ch = 1:size(rawSignals, 2)
            signal = rawSignals(:, ch);
            threshold = 5 * mad(signal, 1);
            is_artifact = abs(signal - median(signal)) > threshold;
            artifact_ratio(ch) = sum(is_artifact) / length(signal);
        end

        % Missing data ratio (already interpolated, check for NaNs)
        missing_ratio = sum(any(isnan(rawSignals), 2)) / size(rawSignals, 1);

        % Store quality metrics
        patient_metrics = struct();
        patient_metrics.PatientID = patient_id;
        patient_metrics.Duration_Hours = hours(fullData.Time(end) - fullData.Time(1));
        patient_metrics.TotalSamples = height(fullData);
        patient_metrics.NumChannels = width(fullData);
        patient_metrics.SNR_mean = mean(snr_vals);
        patient_metrics.SNR_std = std(snr_vals);
        patient_metrics.ArtifactRatio_mean = mean(artifact_ratio);
        patient_metrics.ArtifactRatio_std = std(artifact_ratio);
        patient_metrics.MissingDataRatio = missing_ratio;
        patient_metrics.SeizureSamples = sum(labels);
        patient_metrics.SeizureRatio = sum(labels) / length(labels);

        % Add to quality metrics table
        quality_metrics = [quality_metrics; struct2table(patient_metrics)];

        fprintf('   > SNR: %.1f ± %.1f dB\n', patient_metrics.SNR_mean, patient_metrics.SNR_std);
        fprintf('   > Artifacts: %.1f%% ± %.1f%%\n', 100*patient_metrics.ArtifactRatio_mean, 100*patient_metrics.ArtifactRatio_std);
        fprintf('   > Missing data: %.1f%%\n', 100*patient_metrics.MissingDataRatio);

        %% ----- STEP 4: WINDOW CREATION FOR TRANSFORMER AUTOENCODER -----
        fprintf('[4/4] Window creation for transformer autoencoder...\n');

        % Select normalization method based on config
        fprintf('   > Normalizing (%s)... ', cfg.data.transformer_preprocessing.normalization_type);
        switch lower(cfg.data.transformer_preprocessing.normalization_type)
            case 'zscore'
                mu = mean(rawSignals, 1);
                sigma = std(rawSignals, 0, 1);
                sigma(sigma == 0) = 1;
                normSignals = (rawSignals - mu) ./ sigma;
            case 'minmax'
                min_vals = min(rawSignals, [], 1);
                max_vals = max(rawSignals, [], 1);
                range_vals = max_vals - min_vals;
                range_vals(range_vals == 0) = 1;
                normSignals = (rawSignals - min_vals) ./ range_vals;
            case 'robust'
                median_vals = median(rawSignals, 1);
                iqr_vals = iqr(rawSignals, 1);
                iqr_vals(iqr_vals == 0) = 1;
                normSignals = (rawSignals - median_vals) ./ iqr_vals;
            otherwise
                warning('Unknown normalization type: %s, using Z-score.', cfg.data.transformer_preprocessing.normalization_type);
                mu = mean(rawSignals, 1);
                sigma = std(rawSignals, 0, 1);
                sigma(sigma == 0) = 1;
                normSignals = (rawSignals - mu) ./ sigma;
        end
        fprintf('[Done]\n');

        % Window creation parameters
        numSamples = size(normSignals, 1);
        numChannels = size(normSignals, 2);
        numWindows = floor((numSamples - cfg.data.window_size) / cfg.data.stride) + 1;

        if numWindows < 1
            fprintf('   WARNING: File too short for windowing, skipping.\n');
            continue;
        end

        fprintf('   > Creating %d windows (%.1f hours)... ', numWindows, ...
                numWindows * cfg.data.window_size_sec / 3600);

        % Preallocate batches (memory efficient)
        X_Batch = zeros(numChannels, cfg.data.window_size, 1, numWindows, 'single');
        Y_Batch = zeros(numWindows, 1, 'single');

        % Vectorized window creation
        startIdx = (0:numWindows-1) * cfg.data.stride + 1;

        for w = 1:numWindows
            s = startIdx(w);
            e = s + cfg.data.window_size - 1;

            % Extract segment
            segment = normSignals(s:e, :)'; % [Channels x Time]

            % Deep Learning Toolbox format: [Channels, Time, 1, Batch]
            X_Batch(:, :, 1, w) = segment;

            % Label determination (seizure if > threshold% of window is seizure)
            if sum(labels(s:e)) > (cfg.data.window_size * cfg.data.seizure_threshold)
                Y_Batch(w) = 1; % Seizure
            else
                Y_Batch(w) = 0; % Normal
            end
        end

        fprintf('[Done]\n');
        fprintf('   > Seizure windows: %d (%.1f%%)\n', sum(Y_Batch == 1), 100 * sum(Y_Batch == 1) / numWindows);

        %% ----- STEP 5: TRANSFORMER-SPECIFIC PREPROCESSING -----
        if cfg.data.transformer_preprocessing.enable
            fprintf('   > Transformer-specific preprocessing...\n');

            % Apply transformer-specific preprocessing to X_Batch
            X_Batch = apply_transformer_preprocessing(X_Batch, cfg);

            % Add positional encoding if enabled
            if cfg.data.transformer_preprocessing.add_positional_encoding
                fprintf('     Adding positional encoding... ');
                X_Batch = add_positional_encoding(X_Batch, cfg.data.window_size);
                fprintf('[Done]\n');
            end

            % Apply data augmentation if enabled (only to seizure windows)
            if cfg.data.transformer_preprocessing.data_augmentation
                fprintf('     Applying data augmentation... ');
                [X_Batch, Y_Batch] = apply_data_augmentation(X_Batch, Y_Batch, cfg.data.transformer_preprocessing.augmentation_methods);
                fprintf('[Done]\n');
                % Update numWindows after augmentation
                numWindows = length(Y_Batch);
            end

            % Convert to spectrogram if enabled
            if cfg.data.transformer_preprocessing.spectrogram_enable
                fprintf('     Converting to spectrogram... ');
                X_Batch = convert_to_spectrogram(X_Batch, cfg);
                fprintf('[Done]\n');
            end

            fprintf('   > Transformer preprocessing complete.\n');
        end

        %% ----- STEP 6: TRAIN/TEST SPLIT AND SAVING -----
        fprintf('   > Train/Test split...\n');

        isSeizure = (Y_Batch == 1);
        patient_train_windows = 0;
        patient_test_windows = 0;

        % Case 1: Clean patient (no seizures) - All to train
        if sum(labels) == 0
            saveName = fullfile(cfg.paths.model_data_train, patient_id + "_Full.mat");
            X = X_Batch;
            Y = Y_Batch;
            save(saveName, 'X', 'Y', '-v7.3');
            patient_train_windows = numWindows;
            fprintf('     All windows -> TRAIN (clean patient)\n');

        % Case 2: Patient with seizures
        else
            % Check if we have spectrogram data (3rd dimension > 1)
            if size(X_Batch, 3) == 1
                % Time-domain data: [Channels x Time x 1 x Batch]
                idx_pattern = @(idx) X_Batch(:, :, 1, idx);
            else
                % Spectrogram data: [Channels x Freq x Time x Batch]
                idx_pattern = @(idx) X_Batch(:, :, :, idx);
            end

            % A) Seizure windows -> TEST
            if sum(isSeizure) > 0
                X = idx_pattern(isSeizure);
                Y = Y_Batch(isSeizure);
                saveName = fullfile(cfg.paths.model_data_test, patient_id + "_Seizures.mat");
                save(saveName, 'X', 'Y', '-v7.3');
                patient_test_windows = patient_test_windows + sum(isSeizure);
                fprintf('     %d seizure windows -> TEST\n', sum(isSeizure));
            end

            % B) Normal windows -> 80% Train, 20% Test
            normIdx = find(~isSeizure);
            if ~isempty(normIdx)
                cv = cvpartition(length(normIdx), 'HoldOut', cfg.data.test_ratio);

                % Train portion
                trainIdx = normIdx(training(cv));
                X = idx_pattern(trainIdx);
                Y = Y_Batch(trainIdx);
                saveName = fullfile(cfg.paths.model_data_train, patient_id + "_NormalPart.mat");
                save(saveName, 'X', 'Y', '-v7.3');
                patient_train_windows = patient_train_windows + length(trainIdx);

                % Test portion
                testIdx = normIdx(test(cv));
                X = idx_pattern(testIdx);
                Y = Y_Batch(testIdx);
                saveName = fullfile(cfg.paths.model_data_test, patient_id + "_NormalPart.mat");
                save(saveName, 'X', 'Y', '-v7.3');
                patient_test_windows = patient_test_windows + length(testIdx);

                fprintf('     Normal windows: %d Train, %d Test\n', length(trainIdx), length(testIdx));
            end
        end

        %% ----- STEP 6: PATIENT SUMMARY -----
        patient_elapsed = toc(patient_timer);

        % Update global statistics
        summary_stats.total_train_windows = summary_stats.total_train_windows + patient_train_windows;
        summary_stats.total_test_windows = summary_stats.total_test_windows + patient_test_windows;
        summary_stats.total_seizure_windows = summary_stats.total_seizure_windows + sum(isSeizure);
        summary_stats.total_normal_windows = summary_stats.total_normal_windows + (numWindows - sum(isSeizure));

        fprintf('\n   PATIENT SUMMARY:\n');
        fprintf('   - Duration: %.1f hours\n', patient_metrics.Duration_Hours);
        fprintf('   - Channels: %d\n', patient_metrics.NumChannels);
        fprintf('   - Total windows: %d\n', numWindows);
        fprintf('   - Seizure windows: %d (%.1f%%)\n', sum(isSeizure), 100 * sum(isSeizure) / numWindows);
        fprintf('   - Train windows: %d\n', patient_train_windows);
        fprintf('   - Test windows: %d\n', patient_test_windows);
        fprintf('   - Processing time: %.1f seconds\n', patient_elapsed);

        % Clear large variables for next patient
        clear rawSignals normSignals X_Batch Y_Batch fullData labels X Y;

    catch ME
        fprintf('\n   !!! ERROR PROCESSING PATIENT %s !!!\n', patient_id);
        fprintf('   Message: %s\n', ME.message);
        fprintf('   Location: %s (Line %d)\n', ME.stack(1).name, ME.stack(1).line);

        % Log error to file
        error_log_path = fullfile(cfg.paths.processed, 'preprocessing_errors.log');
        fid = fopen(error_log_path, 'a');
        fprintf(fid, '[%s] Patient: %s - Error: %s\n', ...
                datestr(now, 'yyyy-mm-dd HH:MM:SS'), patient_id, ME.message);
        fclose(fid);

        continue; % Continue with next patient
    end
end

%% ==================== PIPELINE SUMMARY ====================
fprintf('\n===============================================\n');
fprintf('PREPROCESSING PIPELINE COMPLETE\n');
fprintf('===============================================\n');

total_elapsed = toc(pipelineTimer);

% Calculate overall statistics
summary_stats.total_windows = summary_stats.total_train_windows + summary_stats.total_test_windows;
summary_stats.seizure_ratio = summary_stats.total_seizure_windows / summary_stats.total_windows * 100;
summary_stats.class_imbalance_ratio = summary_stats.total_normal_windows / (summary_stats.total_seizure_windows + eps);

fprintf('\n--- OVERALL STATISTICS ---\n');
fprintf('Patients processed: %d/%d\n', height(quality_metrics), summary_stats.total_patients);
fprintf('Total windows created: %d\n', summary_stats.total_windows);
fprintf('  - Training windows: %d (%.1f%%)\n', summary_stats.total_train_windows, ...
        100 * summary_stats.total_train_windows / summary_stats.total_windows);
fprintf('  - Test windows: %d (%.1f%%)\n', summary_stats.total_test_windows, ...
        100 * summary_stats.total_test_windows / summary_stats.total_windows);
fprintf('  - Seizure windows: %d (%.1f%%)\n', summary_stats.total_seizure_windows, summary_stats.seizure_ratio);
fprintf('  - Normal windows: %d\n', summary_stats.total_normal_windows);
fprintf('Class imbalance ratio (Normal:Seizure): %.1f:1\n', summary_stats.class_imbalance_ratio);
fprintf('\nTotal processing time: %.1f minutes\n', total_elapsed / 60);

%% ==================== SAVE QUALITY METRICS ====================
fprintf('\n--- QUALITY METRICS ---\n');

% Save quality metrics to CSV
quality_csv_path = fullfile(cfg.paths.processed, 'preprocessing_quality_metrics.csv');
writetable(quality_metrics, quality_csv_path);
fprintf('Quality metrics saved: %s\n', quality_csv_path);

% Display summary of quality metrics
if ~isempty(quality_metrics)
    fprintf('\nQuality Metrics Summary:\n');
    fprintf('  Average SNR: %.1f ± %.1f dB\n', mean(quality_metrics.SNR_mean), std(quality_metrics.SNR_mean));
    fprintf('  Average artifact ratio: %.1f%% ± %.1f%%\n', ...
            100 * mean(quality_metrics.ArtifactRatio_mean), 100 * std(quality_metrics.ArtifactRatio_mean));
    fprintf('  Average missing data: %.1f%% ± %.1f%%\n', ...
            100 * mean(quality_metrics.MissingDataRatio), 100 * std(quality_metrics.MissingDataRatio));
end

%% ==================== SAVE PIPELINE REPORT ====================
fprintf('\n--- GENERATING REPORT ---\n');

% Create comprehensive report
report_path = fullfile(cfg.paths.processed, 'preprocessing_pipeline_report.txt');
fid = fopen(report_path, 'w');

fprintf(fid, '===============================================\n');
fprintf(fid, 'SEIZEIT2 - PREPROCESSING PIPELINE REPORT\n');
fprintf(fid, '===============================================\n');
fprintf(fid, 'Generated: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'Pipeline version: 2.0.0 (Academic)\n');
fprintf(fid, 'Git commit: %s\n', cfg.reproducibility.git_commit);
fprintf(fid, 'MATLAB version: %s\n', cfg.reproducibility.matlab_version);
fprintf(fid, '\n');

fprintf(fid, '--- CONFIGURATION SUMMARY ---\n');
fprintf(fid, 'Selected patients: %s\n', strjoin(cfg.patient.selected, ', '));
fprintf(fid, 'Target sampling frequency: %d Hz\n', cfg.data.fs);
fprintf(fid, 'Window size: %.1f seconds (%d samples)\n', cfg.data.window_size_sec, cfg.data.window_size);
fprintf(fid, 'Stride: %.1f seconds (%d samples)\n', cfg.data.stride_sec, cfg.data.stride);
fprintf(fid, 'Seizure threshold: %.1f%% of window\n', 100 * cfg.data.seizure_threshold);
fprintf(fid, 'Test ratio (normal data): %.1f%%\n', 100 * cfg.data.test_ratio);
fprintf(fid, '\n');

fprintf(fid, '--- PROCESSING SUMMARY ---\n');
fprintf(fid, 'Patients processed: %d/%d\n', height(quality_metrics), summary_stats.total_patients);
fprintf(fid, 'Total windows created: %d\n', summary_stats.total_windows);
fprintf(fid, '  - Training windows: %d (%.1f%%)\n', summary_stats.total_train_windows, ...
        100 * summary_stats.total_train_windows / summary_stats.total_windows);
fprintf(fid, '  - Test windows: %d (%.1f%%)\n', summary_stats.total_test_windows, ...
        100 * summary_stats.total_test_windows / summary_stats.total_windows);
fprintf(fid, '  - Seizure windows: %d (%.1f%%)\n', summary_stats.total_seizure_windows, summary_stats.seizure_ratio);
fprintf(fid, '  - Normal windows: %d\n', summary_stats.total_normal_windows);
fprintf(fid, 'Class imbalance ratio (Normal:Seizure): %.1f:1\n', summary_stats.class_imbalance_ratio);
fprintf(fid, 'Total processing time: %.1f minutes\n', total_elapsed / 60);
fprintf(fid, '\n');

fprintf(fid, '--- DATA QUALITY METRICS ---\n');
if ~isempty(quality_metrics)
    for i = 1:height(quality_metrics)
        fprintf(fid, '\nPatient: %s\n', quality_metrics.PatientID{i});
        fprintf(fid, '  Duration: %.1f hours\n', quality_metrics.Duration_Hours(i));
        fprintf(fid, '  Channels: %d\n', quality_metrics.NumChannels(i));
        fprintf(fid, '  SNR: %.1f ± %.1f dB\n', quality_metrics.SNR_mean(i), quality_metrics.SNR_std(i));
        fprintf(fid, '  Artifact ratio: %.1f%% ± %.1f%%\n', ...
                100 * quality_metrics.ArtifactRatio_mean(i), 100 * quality_metrics.ArtifactRatio_std(i));
        fprintf(fid, '  Missing data: %.1f%%\n', 100 * quality_metrics.MissingDataRatio(i));
        fprintf(fid, '  Seizure samples: %d (%.1f%%)\n', ...
                quality_metrics.SeizureSamples(i), 100 * quality_metrics.SeizureRatio(i));
    end
end

fprintf(fid, '\n--- REPRODUCIBILITY INFORMATION ---\n');
fprintf(fid, 'Random seed: %d\n', cfg.seed);
fprintf(fid, 'Config hash: %s\n', repro_log.config_hash);
fprintf(fid, 'Pipeline start: %s\n', repro_log.pipeline_start_time);

fclose(fid);
fprintf('Pipeline report saved: %s\n', report_path);

%% ==================== TRANSFORMER AUTOENCODER SPECIFIC NOTES ====================
fprintf('\n--- TRANSFORMER AUTOENCODER NOTES ---\n');
fprintf('The preprocessed data is ready for transformer autoencoder training.\n');
fprintf('Data format: [Channels x Time x 1 x Batch]\n');
fprintf('Suggested preprocessing for transformer autoencoder:\n');
fprintf('  1. Consider adding positional encoding to windows\n');
fprintf('  2. Experiment with different window sizes (1-10 seconds)\n');
fprintf('  3. Consider time-frequency representations (spectrograms)\n');
fprintf('  4. Data augmentation: time warping, jitter, scaling\n');
fprintf('  5. Channel-wise attention mechanisms\n');

fprintf('\n===============================================\n');
fprintf('PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY\n');
fprintf('===============================================\n');

%% ==================== HELPER FUNCTIONS ====================
function resampledTT = processModalityV8(folderPath, prefix, targetFs)
    % PROCESSMODALITYV8 - Process a single modality with enhanced error handling
    % Version 8: Includes data quality checks and better logging

    resampledTT = timetable();

    files = dir(fullfile(folderPath, "*.edf"));
    if isempty(files)
        fprintf('   WARNING: No EDF files found in %s\n', folderPath);
        return;
    end

    filename = fullfile(files(1).folder, files(1).name);

    % 1. READ EDF FILE
    try
        tt_raw = edfread(filename);
        fprintf('   Reading %s... ', filename);
    catch ME
        fprintf('   ERROR reading %s: %s\n', filename, ME.message);
        return;
    end

    % 2. SANITIZATION
    % Ensure time dimension is named 'Time'
    tt_raw.Properties.DimensionNames{1} = 'Time';

    % Remove annotation columns
    badCols = {};
    for v = 1:width(tt_raw)
        oldName = tt_raw.Properties.VariableNames{v};
        if contains(oldName, 'Annot') || contains(oldName, 'Record')
            if ~isnumeric(tt_raw.(oldName)) && ~iscell(tt_raw.(oldName))
                badCols{end+1} = oldName; %#ok<AGROW>
            end
        end
    end
    if ~isempty(badCols)
        tt_raw(:, badCols) = [];
    end

    % 3. PROCESS EACH CHANNEL
    processedCols = {};
    varNames = tt_raw.Properties.VariableNames;

    for i = 1:length(varNames)
        colName = varNames{i};
        colData = tt_raw{:, i};

        % Cell Unpacking (for segmented EDF)
        if iscell(colData)
            if isempty(colData) || ~isnumeric(colData{1})
                continue;
            end
            try
                flatData = vertcat(colData{:});
            catch
                fprintf('   WARNING: Could not concatenate cell data for %s\n', colName);
                continue;
            end

            % Convert to single precision
            flatData = single(flatData);

            % Estimate sampling frequency
            if height(tt_raw) > 1
                recDur = tt_raw.Time(2) - tt_raw.Time(1);
            else
                recDur = seconds(1);
            end
            fs_est = length(colData{1}) / seconds(recDur);
            totalSamples = length(flatData);

            % Create time vector
            newTime = tt_raw.Properties.StartTime + (0:totalSamples-1)' * seconds(1/fs_est);

            singleTT = timetable(newTime, flatData, 'VariableNames', {'TempVar'});

        elseif isnumeric(colData)
            % Convert to single precision
            colData = single(colData);
            singleTT = timetable(tt_raw.Time, colData, 'VariableNames', {'TempVar'});
        else
            continue;
        end

        % Ensure time dimension is named 'Time'
        singleTT.Properties.DimensionNames{1} = 'Time';

        % 4. RESAMPLE TO TARGET FREQUENCY
        try
            singleTT_res = retime(singleTT, 'regular', 'linear', 'TimeStep', seconds(1/targetFs));

            % Clean channel name
            cleanName = regexprep(colName, '[^a-zA-Z0-9]', '');
            singleTT_res.Properties.VariableNames{1} = [prefix '_' cleanName];

            processedCols{end+1} = singleTT_res; %#ok<AGROW>
        catch ME
            fprintf('   WARNING: Could not resample %s: %s\n', colName, ME.message);
            continue;
        end
    end

    % 5. SYNCHRONIZE ALL CHANNELS
    if ~isempty(processedCols)
        try
            resampledTT = synchronize(processedCols{:}, 'union', 'linear');
            resampledTT.Properties.DimensionNames{1} = 'Time';
            fprintf('[%d channels, %.1f hours]\n', width(resampledTT), hours(resampledTT.Time(end) - resampledTT.Time(1)));
        catch ME
            fprintf('   ERROR synchronizing channels: %s\n', ME.message);
            resampledTT = timetable();
        end
    end
end

function hash = string2hash(str)
    % STRING2HASH - Simple string hash for config tracking
    % Not cryptographic, just for reproducibility tracking

    hash = 0;
    for i = 1:length(str)
        hash = mod(hash * 31 + double(str(i)), 2^32);
    end
    hash = dec2hex(hash, 8);
end

%% ==================== TRANSFORMER-SPECIFIC PREPROCESSING FUNCTIONS ====================

function X_processed = apply_transformer_preprocessing(X_batch, cfg)
    % APPLY_TRANSFORMER_PREPROCESSING - Apply transformer-specific preprocessing
    % Input: X_batch [Channels x Time x 1 x Batch]
    % Output: X_processed [Channels x Time x 1 x Batch] (or different dimensions)

    % For now, just return the batch as-is (placeholder for future enhancements)
    X_processed = X_batch;

    % Optional: Add channel-wise normalization or other transformations
    % Example: Layer normalization across time dimension for each channel
    if cfg.data.transformer_preprocessing.enable
        % Could add layer normalization here
        % X_processed = layernorm(X_batch, 'Channel');
    end
end

function X_with_pe = add_positional_encoding(X_batch, seq_length)
    % ADD_POSITIONAL_ENCODING - Add sinusoidal positional encoding to windows
    % Based on "Attention Is All You Need" positional encoding
    % Input: X_batch [Channels x Time x 1 x Batch]
    % Output: X_with_pe [Channels x Time x 1 x Batch] with added positional encoding

    [num_channels, seq_len, ~, batch_size] = size(X_batch);

    % Create positional encoding matrix
    pe = zeros(seq_len, num_channels);

    for pos = 1:seq_len
        for i = 1:num_channels
            if mod(i, 2) == 1
                % Sine for even indices in original paper, but we adapt for channels
                pe(pos, i) = sin(pos / (10000 ^ ((i-1) / num_channels)));
            else
                % Cosine for odd indices
                pe(pos, i) = cos(pos / (10000 ^ ((i-2) / num_channels)));
            end
        end
    end

    % Add positional encoding to each batch element
    X_with_pe = X_batch;
    for b = 1:batch_size
        X_with_pe(:, :, 1, b) = X_batch(:, :, 1, b) + pe';
    end

    fprintf('Positional encoding added (seq length: %d, channels: %d)', seq_len, num_channels);
end

function X_spec = convert_to_spectrogram(X_batch, cfg)
    % CONVERT_TO_SPECTROGRAM - Convert time-series windows to spectrograms
    % Input: X_batch [Channels x Time x 1 x Batch]
    % Output: X_spec [Channels x Freq x Time x Batch] (spectrogram cubes)

    [num_channels, seq_len, ~, batch_size] = size(X_batch);
    window_size = cfg.data.transformer_preprocessing.spectrogram_window;
    overlap = cfg.data.transformer_preprocessing.spectrogram_overlap;

    % Calculate spectrogram dimensions
    nfft = 2^nextpow2(window_size);
    freq_bins = nfft/2 + 1;

    % Preallocate spectrogram batch
    X_spec = zeros(num_channels, freq_bins, floor((seq_len-overlap)/(window_size-overlap)), batch_size, 'single');

    for b = 1:batch_size
        for ch = 1:num_channels
            % Extract channel data
            signal = squeeze(X_batch(ch, :, 1, b));

            % Compute spectrogram
            [S, ~, ~] = spectrogram(signal, window_size, overlap, nfft, cfg.data.fs);

            % Take magnitude and convert to dB
            S_mag = abs(S);
            S_db = 10 * log10(S_mag + eps);

            % Store in output
            X_spec(ch, :, :, b) = single(S_db);
        end
    end

    fprintf('Converted to spectrogram (%d channels, %d freq bins)', num_channels, freq_bins);
end

function [X_aug, Y_aug] = apply_data_augmentation(X_batch, Y_batch, methods)
    % APPLY_DATA_AUGMENTATION - Apply data augmentation to time-series windows
    % Supported methods: 'time_warp', 'jitter', 'scaling', 'permutation'
    % Input: X_batch [Channels x Time x 1 x Batch], Y_batch [Batch x 1]
    % Output: X_aug, Y_aug (augmented data)
    % Note: Only augments seizure windows (Y_batch == 1) to address class imbalance

    % Find seizure windows
    seizure_idx = find(Y_batch == 1);
    num_seizure = length(seizure_idx);

    if num_seizure == 0
        X_aug = X_batch;
        Y_aug = Y_batch;
        fprintf('No seizure windows to augment.\n');
        return;
    end

    % Determine augmentation factor based on imbalance
    % Target: increase seizure windows by 5x (configurable)
    augmentation_factor = 5;
    total_augmented = num_seizure * augmentation_factor;

    % Preallocate augmented arrays
    [num_channels, seq_len, ~, batch_size] = size(X_batch);
    X_aug = zeros(num_channels, seq_len, 1, batch_size + total_augmented, 'single');
    Y_aug = zeros(batch_size + total_augmented, 1, 'single');

    % Copy original data
    X_aug(:, :, 1, 1:batch_size) = X_batch;
    Y_aug(1:batch_size) = Y_batch;

    % Generate augmented seizure windows
    aug_count = 0;

    for i = 1:num_seizure
        orig_idx = seizure_idx(i);
        original_window = X_batch(:, :, 1, orig_idx);

        for aug = 1:augmentation_factor
            aug_count = aug_count + 1;
            aug_window = original_window;

            % Apply selected augmentation methods
            for m = 1:length(methods)
                method = methods{m};

                switch method
                    case 'jitter'
                        % Add small Gaussian noise
                        noise_level = 0.05 * std(aug_window, 0, 'all');
                        aug_window = aug_window + noise_level * randn(size(aug_window), 'single');

                    case 'scaling'
                        % Random scaling factor
                        scale_factor = 0.8 + 0.4 * rand(); % [0.8, 1.2]
                        aug_window = aug_window * scale_factor;

                    case 'time_warp'
                        % Simple time warping: stretch/squeeze along time dimension
                        % Using linear interpolation with fixed output size
                        orig_time = 1:seq_len;
                        warp_factor = 0.9 + 0.2 * rand(); % [0.9, 1.1] - smaller range to avoid extreme resizing

                        % Create warped time axis
                        new_length = round(seq_len * warp_factor);
                        new_time = linspace(1, seq_len, new_length);

                        % Interpolate each channel to warped length
                        warped = zeros(num_channels, new_length, 'single');
                        for ch = 1:num_channels
                            warped(ch, :) = interp1(orig_time, aug_window(ch, :), new_time, 'linear', 'extrap');
                        end

                        % Now interpolate back to original length
                        back_time = linspace(1, seq_len, seq_len);
                        aug_window_back = zeros(num_channels, seq_len, 'single');
                        for ch = 1:num_channels
                            aug_window_back(ch, :) = interp1(new_time, warped(ch, :), back_time, 'linear', 'extrap');
                        end

                        aug_window = aug_window_back;

                    case 'permutation'
                        % Not recommended for seizure data - skip
                        continue;
                end
            end

            % Store augmented window
            X_aug(:, :, 1, batch_size + aug_count) = aug_window;
            Y_aug(batch_size + aug_count) = 1; % Seizure label
        end
    end

    % Trim if needed (in case permutation was skipped)
    X_aug = X_aug(:, :, 1, 1:batch_size+aug_count);
    Y_aug = Y_aug(1:batch_size+aug_count);

    fprintf('Data augmentation applied: %d seizure windows -> %d total (%.1fx)\n', ...
            num_seizure, num_seizure + aug_count, (num_seizure + aug_count) / num_seizure);
end