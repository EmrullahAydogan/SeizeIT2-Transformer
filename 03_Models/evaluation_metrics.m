function metrics = evaluation_metrics(scores, labels, varargin)
% EVALUATION_METRICS - Comprehensive evaluation metrics for imbalanced seizure detection
%
% Inputs:
%   scores - Reconstruction error scores (higher = more anomalous)
%   labels - True labels (0=normal, 1=seizure)
% Optional name-value pairs:
%   'ThresholdMethod' - Method for threshold selection:
%       'youden' (default), 'optimal_f1', 'fixed', 'median', 'percentile'
%   'FixedThreshold' - Fixed threshold value (if method='fixed')
%   'Percentile' - Percentile for threshold (if method='percentile')
%   'ShowPlots' - Display ROC and PR curves (default: true)
%   'SavePath' - Path to save plots (default: '')
%
% Outputs:
%   metrics - Struct containing all evaluation metrics:
%     .auc_roc - Area under ROC curve
%     .auc_pr - Area under Precision-Recall curve
%     .optimal_threshold - Selected threshold
%     .confusion_matrix - [TN, FP; FN, TP]
%     .accuracy - Overall accuracy
%     .sensitivity (recall) - TP / (TP + FN)
%     .specificity - TN / (TN + FP)
%     .precision - TP / (TP + FP)
%     .f1_score - 2 * (precision * recall) / (precision + recall)
%     .mcc - Matthews Correlation Coefficient
%     .gmean - Geometric mean of sensitivity and specificity
%     .roc_curve - [FPR, TPR] points
%     .pr_curve - [Recall, Precision] points
%
% Notes:
%   - Designed for severe class imbalance (Normal:Seizure = 198.6:1)
%   - Primary metric: AUC-PR (more informative than AUC-ROC for imbalance)
%   - Includes robust metrics: MCC, G-mean for imbalanced data

% Parse inputs
p = inputParser;
addParameter(p, 'ThresholdMethod', 'youden', @ischar);
addParameter(p, 'FixedThreshold', NaN, @isnumeric);
addParameter(p, 'Percentile', 95, @isnumeric);
addParameter(p, 'ShowPlots', true, @islogical);
addParameter(p, 'SavePath', '', @ischar);
parse(p, varargin{:});
opts = p.Results;

% Validate inputs
scores = scores(:);
labels = labels(:);

if length(scores) ~= length(labels)
    error('scores and labels must have same length');
end

% Ensure binary labels (0=normal, 1=seizure)
if ~all(ismember(labels, [0, 1]))
    error('labels must be binary (0 or 1)');
end

% Count classes
n_normal = sum(labels == 0);
n_seizure = sum(labels == 1);
imbalance_ratio = n_normal / max(n_seizure, 1);

fprintf('\n=== EVALUATION METRICS ===\n');
fprintf('Total samples: %d\n', length(labels));
fprintf('  Normal: %d (%.1f%%)\n', n_normal, n_normal/length(labels)*100);
fprintf('  Seizure: %d (%.1f%%)\n', n_seizure, n_seizure/length(labels)*100);
fprintf('  Imbalance ratio: %.1f:1 (Normal:Seizure)\n', imbalance_ratio);

%% 1. Calculate ROC Curve and AUC
[fpr, tpr, roc_thresholds, auc_roc] = perfcurve(labels, scores, 1);

% Store ROC curve
metrics.roc_curve = [fpr, tpr];
metrics.roc_thresholds = roc_thresholds;
metrics.auc_roc = auc_roc;

%% 2. Calculate Precision-Recall Curve and AUC
% MATLAB's perfcurve for PR curve
[recall, precision, ~, auc_pr] = perfcurve(labels, scores, 1, ...
    'XCrit', 'reca', 'YCrit', 'prec');

% Store PR curve
metrics.pr_curve = [recall, precision];
metrics.auc_pr = auc_pr;

fprintf('\n--- AUC Scores ---\n');
fprintf('AUC-ROC:  %.4f\n', auc_roc);
fprintf('AUC-PR:   %.4f\n', auc_pr);

%% 3. Select Optimal Threshold
threshold = select_threshold(scores, labels, opts);

% Apply threshold to get binary predictions
predictions = scores >= threshold;

%% 4. Calculate Confusion Matrix
TP = sum(predictions == 1 & labels == 1);
FP = sum(predictions == 1 & labels == 0);
TN = sum(predictions == 0 & labels == 0);
FN = sum(predictions == 0 & labels == 1);

metrics.confusion_matrix = [TN, FP; FN, TP];

%% 5. Calculate Standard Metrics
% Sensitivity (Recall, True Positive Rate)
sensitivity = TP / (TP + FN + eps);

% Specificity (True Negative Rate)
specificity = TN / (TN + FP + eps);

% Precision (Positive Predictive Value)
precision_val = TP / (TP + FP + eps);

% Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN);

% F1 Score
f1_score = 2 * (precision_val * sensitivity) / (precision_val + sensitivity + eps);

% Matthews Correlation Coefficient (robust for imbalance)
mcc_numerator = (TP * TN) - (FP * FN);
mcc_denominator = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps);
mcc = mcc_numerator / mcc_denominator;

% Geometric Mean (G-mean)
gmean = sqrt(sensitivity * specificity);

% Store metrics
metrics.optimal_threshold = threshold;
metrics.sensitivity = sensitivity;
metrics.specificity = specificity;
metrics.precision = precision_val;
metrics.accuracy = accuracy;
metrics.f1_score = f1_score;
metrics.mcc = mcc;
metrics.gmean = gmean;

%% 6. Additional Statistical Measures
% Youden's J statistic
youden_j = sensitivity + specificity - 1;
metrics.youden_j = youden_j;

% Balanced Accuracy
balanced_accuracy = (sensitivity + specificity) / 2;
metrics.balanced_accuracy = balanced_accuracy;

% Cohen's Kappa (chance-corrected agreement)
p0 = accuracy;
pe = (((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / ...
    (TP + TN + FP + FN)^2) + eps;
kappa = (p0 - pe) / (1 - pe);
metrics.kappa = kappa;

%% 7. Display Results
fprintf('\n--- Threshold Selection ---\n');
fprintf('Method: %s\n', opts.ThresholdMethod);
fprintf('Optimal threshold: %.4f\n', threshold);
fprintf('  (%.1f%% of scores below threshold)\n', mean(scores < threshold) * 100);

fprintf('\n--- Performance at Optimal Threshold ---\n');
fprintf('Confusion Matrix:\n');
fprintf('            Predicted\n');
fprintf('            Normal   Seizure\n');
fprintf('Actual Normal %6d   %6d\n', TN, FP);
fprintf('       Seizure %6d   %6d\n', FN, TP);

fprintf('\nPrimary Metrics:\n');
fprintf('  Sensitivity (Recall):  %.4f\n', sensitivity);
fprintf('  Specificity:           %.4f\n', specificity);
fprintf('  Precision:             %.4f\n', precision_val);
fprintf('  F1 Score:              %.4f\n', f1_score);
fprintf('  Accuracy:              %.4f\n', accuracy);

fprintf('\nRobust Metrics (for imbalance):\n');
fprintf('  MCC:                   %.4f\n', mcc);
fprintf('  G-mean:                %.4f\n', gmean);
fprintf('  Balanced Accuracy:     %.4f\n', balanced_accuracy);

%% 8. Generate Plots
if opts.ShowPlots
    generate_evaluation_plots(scores, labels, metrics, opts);
end

%% 9. Save Results if requested
if ~isempty(opts.SavePath)
    save_metrics_to_file(metrics, opts.SavePath);
end

fprintf('\nEvaluation complete.\n');

end

%% Helper Functions

function threshold = select_threshold(scores, labels, opts)
% SELECT_THRESHOLD - Select optimal threshold based on method

    switch lower(opts.ThresholdMethod)
        case 'youden'
            % Youden's J statistic: max(sensitivity + specificity - 1)
            [fpr, tpr, thresholds] = perfcurve(labels, scores, 1);
            youden_j = tpr + (1 - fpr) - 1;  % sensitivity + specificity - 1
            [~, idx] = max(youden_j);
            threshold = thresholds(idx);

        case 'optimal_f1'
            % Maximize F1 score
            [~, ~, ~, auc_roc, thresholds] = perfcurve(labels, scores, 1);
            % Calculate F1 at each threshold
            f1_scores = zeros(length(thresholds), 1);
            for i = 1:length(thresholds)
                pred = scores >= thresholds(i);
                TP = sum(pred == 1 & labels == 1);
                FP = sum(pred == 1 & labels == 0);
                FN = sum(pred == 0 & labels == 1);

                precision = TP / (TP + FP + eps);
                recall = TP / (TP + FN + eps);
                f1_scores(i) = 2 * (precision * recall) / (precision + recall + eps);
            end
            [~, idx] = max(f1_scores);
            threshold = thresholds(idx);

        case 'fixed'
            if isnan(opts.FixedThreshold)
                error('FixedThreshold must be specified for method=''fixed''');
            end
            threshold = opts.FixedThreshold;

        case 'median'
            % Median of normal scores (for anomaly detection)
            normal_scores = scores(labels == 0);
            threshold = median(normal_scores);

        case 'percentile'
            % Percentile of normal scores
            normal_scores = scores(labels == 0);
            threshold = prctile(normal_scores, opts.Percentile);

        case 'mean_std'
            % Mean + k*std of normal scores
            normal_scores = scores(labels == 0);
            threshold = mean(normal_scores) + 2 * std(normal_scores);

        otherwise
            error('Unknown threshold method: %s', opts.ThresholdMethod);
    end
end

function generate_evaluation_plots(scores, labels, metrics, opts)
% GENERATE_EVALUATION_PLOTS - Create ROC and PR curves

    % Create figure with two subplots
    fig = figure('Position', [100, 100, 1200, 500]);

    % 1. ROC Curve
    subplot(1, 2, 1);
    plot(metrics.roc_curve(:,1), metrics.roc_curve(:,2), 'b-', 'LineWidth', 2);
    hold on;
    plot([0, 1], [0, 1], 'k--', 'LineWidth', 1);  % Random classifier
    xlabel('False Positive Rate (1 - Specificity)', 'FontSize', 12);
    ylabel('True Positive Rate (Sensitivity)', 'FontSize', 12);
    title(sprintf('ROC Curve (AUC = %.4f)', metrics.auc_roc), 'FontSize', 14);
    grid on;
    axis equal;
    xlim([0, 1]);
    ylim([0, 1]);

    % Mark optimal threshold point
    optimal_idx = find(metrics.roc_thresholds == metrics.optimal_threshold, 1);
    if ~isempty(optimal_idx)
        plot(metrics.roc_curve(optimal_idx,1), metrics.roc_curve(optimal_idx,2), ...
            'ro', 'MarkerSize', 10, 'LineWidth', 2);
        legend({'ROC Curve', 'Random', 'Optimal Threshold'}, 'Location', 'southeast');
    else
        legend({'ROC Curve', 'Random'}, 'Location', 'southeast');
    end

    % 2. Precision-Recall Curve
    subplot(1, 2, 2);
    plot(metrics.pr_curve(:,1), metrics.pr_curve(:,2), 'r-', 'LineWidth', 2);
    xlabel('Recall (Sensitivity)', 'FontSize', 12);
    ylabel('Precision', 'FontSize', 12);
    title(sprintf('Precision-Recall Curve (AUC = %.4f)', metrics.auc_pr), 'FontSize', 14);
    grid on;
    xlim([0, 1]);
    ylim([0, 1]);

    % Add no-skill line (precision = positive class ratio)
    positive_ratio = sum(labels == 1) / length(labels);
    line([0, 1], [positive_ratio, positive_ratio], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
    legend({'PR Curve', 'No-Skill'}, 'Location', 'southeast');

    % Save figure if requested
    if ~isempty(opts.SavePath)
        saveas(fig, fullfile(opts.SavePath, 'evaluation_curves.png'));
        saveas(fig, fullfile(opts.SavePath, 'evaluation_curves.fig'));
        fprintf('Plots saved to: %s\n', opts.SavePath);
    end

    % 3. Score distributions (separate figure)
    fig2 = figure('Position', [100, 100, 800, 600]);

    % Histogram of scores by class
    normal_scores = scores(labels == 0);
    seizure_scores = scores(labels == 1);

    subplot(2, 1, 1);
    histogram(normal_scores, 50, 'FaceColor', 'blue', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    hold on;
    histogram(seizure_scores, 50, 'FaceColor', 'red', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    xlabel('Reconstruction Error Score', 'FontSize', 12);
    ylabel('Frequency', 'FontSize', 12);
    title('Score Distributions by Class', 'FontSize', 14);
    legend({'Normal', 'Seizure'}, 'FontSize', 10);
    grid on;

    % Add threshold line
    y_limits = ylim;
    line([metrics.optimal_threshold, metrics.optimal_threshold], y_limits, ...
        'Color', 'black', 'LineStyle', '--', 'LineWidth', 2);
    text(metrics.optimal_threshold, y_limits(2)*0.9, ...
        sprintf('Threshold = %.3f', metrics.optimal_threshold), ...
        'HorizontalAlignment', 'right', 'FontSize', 10);

    % Box plot
    subplot(2, 1, 2);
    boxplot([normal_scores; seizure_scores], [zeros(size(normal_scores)); ones(size(seizure_scores))], ...
        'Labels', {'Normal', 'Seizure'});
    ylabel('Reconstruction Error Score', 'FontSize', 12);
    title('Score Distribution (Box Plot)', 'FontSize', 14);
    grid on;

    % Add threshold line
    hold on;
    plot(xlim, [metrics.optimal_threshold, metrics.optimal_threshold], ...
        'k--', 'LineWidth', 1.5);

    if ~isempty(opts.SavePath)
        saveas(fig2, fullfile(opts.SavePath, 'score_distributions.png'));
        saveas(fig2, fullfile(opts.SavePath, 'score_distributions.fig'));
    end
end

function save_metrics_to_file(metrics, save_path)
% SAVE_METRICS_TO_FILE - Save metrics to text and JSON files

    % Create directory if it doesn't exist
    if ~isfolder(save_path)
        mkdir(save_path);
    end

    % Save as text file
    txt_file = fullfile(save_path, 'evaluation_metrics.txt');
    fid = fopen(txt_file, 'w');

    fprintf(fid, 'EVALUATION METRICS REPORT\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

    fprintf(fid, '=== AUC SCORES ===\n');
    fprintf(fid, 'AUC-ROC: %.4f\n', metrics.auc_roc);
    fprintf(fid, 'AUC-PR:  %.4f\n\n', metrics.auc_pr);

    fprintf(fid, '=== OPTIMAL THRESHOLD ===\n');
    fprintf(fid, 'Threshold: %.4f\n\n', metrics.optimal_threshold);

    fprintf(fid, '=== CONFUSION MATRIX ===\n');
    fprintf(fid, '            Predicted\n');
    fprintf(fid, '            Normal   Seizure\n');
    fprintf(fid, 'Actual Normal %6d   %6d\n', metrics.confusion_matrix(1,1), metrics.confusion_matrix(1,2));
    fprintf(fid, '       Seizure %6d   %6d\n\n', metrics.confusion_matrix(2,1), metrics.confusion_matrix(2,2));

    fprintf(fid, '=== PERFORMANCE METRICS ===\n');
    fprintf(fid, 'Sensitivity (Recall):  %.4f\n', metrics.sensitivity);
    fprintf(fid, 'Specificity:           %.4f\n', metrics.specificity);
    fprintf(fid, 'Precision:             %.4f\n', metrics.precision);
    fprintf(fid, 'F1 Score:              %.4f\n', metrics.f1_score);
    fprintf(fid, 'Accuracy:              %.4f\n\n', metrics.accuracy);

    fprintf(fid, '=== ROBUST METRICS (for imbalance) ===\n');
    fprintf(fid, 'MCC:                   %.4f\n', metrics.mcc);
    fprintf(fid, 'G-mean:                %.4f\n', metrics.gmean);
    fprintf(fid, 'Balanced Accuracy:     %.4f\n', metrics.balanced_accuracy);
    fprintf(fid, 'Youden''s J:           %.4f\n', metrics.youden_j);
    fprintf(fid, 'Cohen''s Kappa:        %.4f\n', metrics.kappa);

    fclose(fid);
    fprintf('Metrics saved to text file: %s\n', txt_file);

    % Save as JSON
    json_file = fullfile(save_path, 'evaluation_metrics.json');
    json_str = jsonencode(metrics, 'PrettyPrint', true);
    fid = fopen(json_file, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
    fprintf('Metrics saved to JSON file: %s\n', json_file);
end