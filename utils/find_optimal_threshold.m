function [threshold, metrics] = find_optimal_threshold(y_true, y_scores, method)
% FIND_OPTIMAL_THRESHOLD - Find optimal decision threshold
%
% Inputs:
%   y_true  - True binary labels (0=normal, 1=seizure)
%   y_scores - Anomaly scores (continuous values)
%   method  - 'youden', 'f1', 'fixed' (default: 'youden')
%
% Outputs:
%   threshold - Optimal threshold value
%   metrics   - Struct containing ROC curve data and optimal point
%
% Methods:
%   'youden'  - Maximize Youden's Index (Sensitivity + Specificity - 1)
%   'f1'      - Maximize F1-Score
%   'fixed'   - Use median of seizure scores
%
% Example:
%   [threshold, info] = find_optimal_threshold(labels, scores, 'youden');

    if nargin < 3
        method = 'youden';
    end

    % Compute ROC curve
    [X, Y, T, AUC] = perfcurve(y_true, y_scores, 1);

    metrics.roc_fpr = X;
    metrics.roc_tpr = Y;
    metrics.roc_thresholds = T;
    metrics.auc = AUC;

    switch lower(method)
        case 'youden'
            % Youden's Index: J = Sensitivity + Specificity - 1
            youden_index = Y - X;  % TPR - FPR
            [~, opt_idx] = max(youden_index);
            threshold = T(opt_idx);
            metrics.sensitivity = Y(opt_idx);
            metrics.specificity = 1 - X(opt_idx);
            metrics.method = 'Youden Index';

        case 'f1'
            % Maximize F1-Score
            f1_scores = zeros(size(T));
            for i = 1:length(T)
                y_pred = y_scores >= T(i);
                tp = sum(y_pred == 1 & y_true == 1);
                fp = sum(y_pred == 1 & y_true == 0);
                fn = sum(y_pred == 0 & y_true == 1);

                precision = tp / (tp + fp + eps);
                recall = tp / (tp + fn + eps);
                f1_scores(i) = 2 * precision * recall / (precision + recall + eps);
            end
            [~, opt_idx] = max(f1_scores);
            threshold = T(opt_idx);
            metrics.f1_score = f1_scores(opt_idx);
            metrics.method = 'Optimal F1';

        case 'fixed'
            % Use median of seizure scores
            seizure_scores = y_scores(y_true == 1);
            threshold = median(seizure_scores);
            metrics.method = 'Fixed (Median Seizure)';

        otherwise
            error('Unknown method: %s', method);
    end

    metrics.optimal_threshold = threshold;
end
