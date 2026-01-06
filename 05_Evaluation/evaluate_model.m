% evaluate_model.m
% SeizeIT2 - Academic-Grade Model Evaluation (Refactored v2.0)
%
% IMPROVEMENTS OVER LEGACY:
%   - Uses centralized config
%   - Uses utility functions (compute_metrics, find_optimal_threshold, etc.)
%   - Automatic patient detection
%   - Better error handling
%   - Cleaner code structure
%
% Usage:
%   evaluate_model()  % Uses latest model
%   evaluate_model('ModelPath', 'path/to/model.mat')
%
% Author: SeizeIT2 Project
% Date: January 2025

function Results = evaluate_model(varargin)
    clc;

    %% === CONFIGURATION ===
    % Add paths
    script_dir = fileparts(mfilename('fullpath'));
    addpath(fullfile(script_dir, '..', 'config'));
    addpath(fullfile(script_dir, '..', 'utils'));

    cfg = config();

    % Parse arguments
    p = inputParser;
    addParameter(p, 'ModelPath', fullfile(cfg.paths.models, 'Trained_Transformer_Latest.mat'), @ischar);
    parse(p, varargin{:});

    model_path = p.Results.ModelPath;

    fprintf('=== ACADEMIC EVALUATION ===\n');
    fprintf('Model: %s\n\n', model_path);

    %% === LOAD MODEL ===
    if ~isfile(model_path)
        error('Model not found: %s\nTrain a model first using train_model()', model_path);
    end

    fprintf('Loading model...\n');
    loaded = load(model_path, 'net');
    net = loaded.net;

    %% === DETECT PATIENTS ===
    fprintf('Detecting available patients...\n');
    patients = cfg.patient.selected;
    Results = struct();
    result_idx = 1;

    %% === EVALUATE EACH PATIENT ===
    for p = 1:length(patients)
        patient_id = patients(p);
        fprintf('\n[%d/%d] Evaluating: %s\n', p, length(patients), patient_id);

        % Load test data
        [seizure_data, seizure_labels] = load_patient_data(patient_id, 'test', cfg);
        seizure_data = seizure_data(seizure_labels == 1);  % Only seizures

        [normal_data, normal_labels] = load_patient_data(patient_id, 'test', cfg);
        normal_data = normal_data(normal_labels == 0);  % Only normals

        if isempty(seizure_data)
            fprintf('   WARNING: No seizure data found, skipping.\n');
            continue;
        end

        if isempty(normal_data)
            fprintf('   WARNING: No normal data found, skipping.\n');
            continue;
        end

        fprintf('   Data: %d Seizure + %d Normal windows\n', ...
                length(seizure_data), length(normal_data));

        % Compute anomaly scores
        fprintf('   Computing anomaly scores... ');
        sz_scores = compute_anomaly_scores(net, seizure_data, 'MiniBatchSize', cfg.train.min_batch_size);
        norm_scores = compute_anomaly_scores(net, normal_data, 'MiniBatchSize', cfg.train.min_batch_size);
        fprintf('[Done]\n');

        % Combine
        all_scores = [sz_scores; norm_scores];
        true_labels = [ones(length(sz_scores), 1); zeros(length(norm_scores), 1)];

        % Find optimal threshold
        [threshold, roc_data] = find_optimal_threshold(true_labels, all_scores, 'youden');

        fprintf('   AUC: %.4f | Optimal Threshold: %.4f\n', roc_data.auc, threshold);

        % Compute metrics at optimal threshold
        predictions = all_scores >= threshold;
        metrics = compute_metrics(true_labels, predictions, all_scores, threshold);

        fprintf('   Sens: %.3f | Spec: %.3f | F1: %.3f | Acc: %.3f\n', ...
                metrics.sensitivity, metrics.specificity, metrics.f1_score, metrics.accuracy);

        % Statistical test
        [p_value, ~] = ranksum(norm_scores, sz_scores);
        if p_value < 0.001
            sig_str = '***';
        elseif p_value < 0.01
            sig_str = '**';
        elseif p_value < 0.05
            sig_str = '*';
        else
            sig_str = '(ns)';
        end
        fprintf('   Mann-Whitney U: p = %.2e %s\n', p_value, sig_str);

        % Store results
        Results(result_idx).PatientID = char(patient_id);
        Results(result_idx).NumSeizure = length(sz_scores);
        Results(result_idx).NumNormal = length(norm_scores);
        Results(result_idx).AUC = metrics.auc;
        Results(result_idx).OptimalThreshold = threshold;
        Results(result_idx).Sensitivity = metrics.sensitivity;
        Results(result_idx).Specificity = metrics.specificity;
        Results(result_idx).Precision = metrics.precision;
        Results(result_idx).F1_Score = metrics.f1_score;
        Results(result_idx).Accuracy = metrics.accuracy;
        Results(result_idx).PValue = p_value;
        Results(result_idx).TP = metrics.TP;
        Results(result_idx).TN = metrics.TN;
        Results(result_idx).FP = metrics.FP;
        Results(result_idx).FN = metrics.FN;
        Results(result_idx).ROC_X = roc_data.roc_fpr;
        Results(result_idx).ROC_Y = roc_data.roc_tpr;
        Results(result_idx).SeizureScores = sz_scores;
        Results(result_idx).NormalScores = norm_scores;

        result_idx = result_idx + 1;
    end

    if isempty(Results)
        error('No patients evaluated successfully!');
    end

    %% === SUMMARY STATISTICS ===
    fprintf('\n=== SUMMARY (n=%d Patients) ===\n', length(Results));

    T = struct2table(Results);
    T = T(:, {'PatientID', 'AUC', 'Sensitivity', 'Specificity', 'F1_Score', 'Accuracy', 'PValue'});
    disp(T);

    fprintf('\n--- Mean ± SD ---\n');
    fprintf('AUC:         %.3f ± %.3f\n', mean([Results.AUC]), std([Results.AUC]));
    fprintf('Sensitivity: %.3f ± %.3f\n', mean([Results.Sensitivity]), std([Results.Sensitivity]));
    fprintf('Specificity: %.3f ± %.3f\n', mean([Results.Specificity]), std([Results.Specificity]));
    fprintf('F1-Score:    %.3f ± %.3f\n', mean([Results.F1_Score]), std([Results.F1_Score]));
    fprintf('Accuracy:    %.3f ± %.3f\n', mean([Results.Accuracy]), std([Results.Accuracy]));

    %% === SAVE RESULTS ===
    save_results(Results, 'EvaluationResults', cfg, 'Format', {'mat'});
    writetable(T, fullfile(cfg.paths.tables, 'PerPatient_Performance.csv'));

    fprintf('\n=== Results saved to %s ===\n', cfg.paths.results);

    %% === VISUALIZATIONS ===
    generate_evaluation_figures(Results, cfg);
end

%% === VISUALIZATION HELPER ===
function generate_evaluation_figures(Results, cfg)
    fprintf('\nGenerating figures...\n');

    % ROC Curves
    fig1 = figure('Name', 'ROC Curves', 'Color', 'w', 'Position', [100, 100, 900, 700]);
    hold on; grid on;

    colors = lines(length(Results));
    for p = 1:length(Results)
        plot(Results(p).ROC_X, Results(p).ROC_Y, '-', 'Color', colors(p,:), 'LineWidth', 2, ...
             'DisplayName', sprintf('%s (AUC=%.3f)', Results(p).PatientID, Results(p).AUC));
    end

    plot([0 1], [0 1], 'k--', 'LineWidth', 1.5, 'DisplayName', 'Chance');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('ROC Curves: Per-Patient Performance');
    legend('Location', 'southeast');
    axis square; xlim([0 1]); ylim([0 1]);

    saveas(fig1, fullfile(cfg.paths.figures, 'ROC_Curves_PerPatient.png'));
    close(fig1);

    % Anomaly Score Distributions
    fig2 = figure('Name', 'Distributions', 'Color', 'w', 'Position', [150, 150, 1200, 400]);
    for p = 1:length(Results)
        subplot(1, length(Results), p);
        hold on; grid on;

        histogram(Results(p).NormalScores, 30, 'FaceColor', 'g', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        histogram(Results(p).SeizureScores, 30, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        xline(Results(p).OptimalThreshold, 'b--', 'LineWidth', 2);

        title(sprintf('%s (AUC=%.3f)', Results(p).PatientID, Results(p).AUC));
        xlabel('Anomaly Score');
        ylabel('Frequency');
        legend({'Normal', 'Seizure', 'Threshold'}, 'Location', 'best');
    end

    saveas(fig2, fullfile(cfg.paths.figures, 'Anomaly_Scores_Distribution.png'));
    close(fig2);

    % Confusion Matrices
    fig3 = figure('Name', 'Confusion', 'Color', 'w', 'Position', [200, 200, 1200, 400]);
    for p = 1:length(Results)
        subplot(1, length(Results), p);

        cm = [Results(p).TN, Results(p).FP; Results(p).FN, Results(p).TP];
        imagesc(cm);
        colormap(flipud(gray));
        colorbar;

        textStrings = num2str(cm(:), '%d');
        textStrings = strtrim(cellstr(textStrings));
        [x, y] = meshgrid(1:2);
        text(x(:), y(:), textStrings, 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');

        title(sprintf('%s\nAcc=%.2f%%', Results(p).PatientID, Results(p).Accuracy*100));
        xlabel('Predicted'); ylabel('Actual');
        set(gca, 'XTick', 1:2, 'XTickLabel', {'Normal', 'Seizure'});
        set(gca, 'YTick', 1:2, 'YTickLabel', {'Normal', 'Seizure'});
        axis square;
    end

    saveas(fig3, fullfile(cfg.paths.figures, 'Confusion_Matrices.png'));
    close(fig3);

    fprintf('Figures saved to: %s\n', cfg.paths.figures);
end
