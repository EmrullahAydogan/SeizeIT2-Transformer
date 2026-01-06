function metrics = compute_metrics(y_true, y_pred, y_scores, threshold)
% COMPUTE_METRICS - Calculate classification performance metrics
%
% Inputs:
%   y_true    - True labels (binary: 0=normal, 1=seizure)
%   y_pred    - Predicted labels (binary)
%   y_scores  - Anomaly scores (continuous)
%   threshold - Decision threshold (optional, uses y_pred if not provided)
%
% Outputs:
%   metrics   - Struct containing performance metrics
%
% Example:
%   metrics = compute_metrics(true_labels, predictions, scores, 0.5);

    % Apply threshold if provided
    if nargin >= 4
        y_pred = y_scores >= threshold;
    end

    % Confusion matrix elements
    TP = sum(y_pred == 1 & y_true == 1);
    TN = sum(y_pred == 0 & y_true == 0);
    FP = sum(y_pred == 1 & y_true == 0);
    FN = sum(y_pred == 0 & y_true == 1);

    % Basic metrics
    metrics.TP = TP;
    metrics.TN = TN;
    metrics.FP = FP;
    metrics.FN = FN;

    % Performance metrics
    metrics.sensitivity = TP / (TP + FN);  % Recall, TPR
    metrics.specificity = TN / (TN + FP);  % TNR
    metrics.precision = TP / (TP + FP);    % PPV
    metrics.accuracy = (TP + TN) / (TP + TN + FP + FN);

    % F1-Score
    if metrics.precision + metrics.sensitivity > 0
        metrics.f1_score = 2 * metrics.precision * metrics.sensitivity / ...
                          (metrics.precision + metrics.sensitivity);
    else
        metrics.f1_score = 0;
    end

    % ROC-AUC (if scores provided)
    if nargin >= 3 && ~isempty(y_scores)
        try
            [~, ~, ~, metrics.auc] = perfcurve(y_true, y_scores, 1);
        catch
            metrics.auc = NaN;
        end
    else
        metrics.auc = NaN;
    end

    % False positive rate / False negative rate
    metrics.fpr = FP / (FP + TN);
    metrics.fnr = FN / (FN + TP);

    % Matthews Correlation Coefficient
    mcc_denominator = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
    if mcc_denominator > 0
        metrics.mcc = (TP*TN - FP*FN) / mcc_denominator;
    else
        metrics.mcc = 0;
    end

    % Balanced accuracy
    metrics.balanced_accuracy = (metrics.sensitivity + metrics.specificity) / 2;
end
