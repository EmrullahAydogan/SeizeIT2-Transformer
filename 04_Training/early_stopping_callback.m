function stop = early_stopping_callback(info)
% EARLY_STOPPING_CALLBACK - Enhanced output function with monitoring
%
% Features:
%   - Early stopping with patience
%   - Real-time logging to file
%   - Epoch-wise metrics tracking
%   - Training progress visualization
%
% Usage:
%   Add to trainingOptions: 'OutputFcn', @early_stopping_callback

    persistent best_val_loss
    persistent epochs_no_improve
    persistent patience
    persistent min_delta
    persistent log_file
    persistent metrics_file
    persistent start_time
    persistent cfg

    stop = false;

    % Initialize on first iteration
    if info.Iteration == 0
        cfg = config();
        best_val_loss = inf;
        epochs_no_improve = 0;
        patience = cfg.train.patience;
        min_delta = cfg.train.min_delta;
        start_time = datetime('now');

        % Create log files
        timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
        log_file = fullfile(cfg.paths.results, sprintf('training_log_%s.txt', timestamp));
        metrics_file = fullfile(cfg.paths.results, sprintf('training_metrics_%s.csv', timestamp));

        % Initialize log file
        fid = fopen(log_file, 'w');
        fprintf(fid, '==========================================================\n');
        fprintf(fid, 'TRAINING LOG - %s\n', timestamp);
        fprintf(fid, '==========================================================\n');
        fprintf(fid, 'Start Time: %s\n', start_time);
        fprintf(fid, 'Early Stopping: Enabled (Patience: %d, Min Delta: %.6f)\n', patience, min_delta);
        fprintf(fid, '==========================================================\n\n');
        fclose(fid);

        % Initialize metrics CSV
        fid = fopen(metrics_file, 'w');
        fprintf(fid, 'Epoch,Iteration,TrainingLoss,ValidationLoss,LearningRate,ElapsedTime,BestValLoss,EpochsNoImprove\n');
        fclose(fid);

        fprintf('\n[Monitoring] Log files created:\n');
        fprintf('  - Training log: %s\n', log_file);
        fprintf('  - Metrics CSV: %s\n', metrics_file);
        fprintf('[Monitoring] You can monitor progress with: tail -f %s\n\n', log_file);
        fprintf('[Early Stopping] Initialized - Patience: %d epochs, Min Delta: %.6f\n', ...
                patience, min_delta);
        return;
    end

    % Only check at end of epoch
    if info.State ~= "iteration"
        return;
    end

    % Get current metrics
    current_train_loss = info.TrainingLoss;
    current_val_loss = info.ValidationLoss;
    elapsed_time = seconds(datetime('now') - start_time);

    % Log to console and file
    log_msg = sprintf('[Epoch %d] Iter: %d | Train Loss: %.6f | Val Loss: %.6f | Time: %.1fs', ...
                      info.Epoch, info.Iteration, current_train_loss, current_val_loss, elapsed_time);
    fprintf('%s\n', log_msg);

    % Write to log file
    fid = fopen(log_file, 'a');
    fprintf(fid, '%s\n', log_msg);
    fclose(fid);

    % Write to metrics CSV
    fid = fopen(metrics_file, 'a');
    fprintf(fid, '%d,%d,%.8f,%.8f,%.8f,%.2f,%.8f,%d\n', ...
            info.Epoch, info.Iteration, current_train_loss, current_val_loss, ...
            info.BaseLearnRate, elapsed_time, best_val_loss, epochs_no_improve);
    fclose(fid);

    % Check if we have validation loss
    if isempty(current_val_loss)
        return;
    end

    % Check for improvement
    if current_val_loss < (best_val_loss - min_delta)
        % Improvement detected
        improvement = best_val_loss - current_val_loss;
        best_val_loss = current_val_loss;
        epochs_no_improve = 0;

        msg = sprintf('[Early Stopping] ✓ Improvement! Val loss: %.6f (↓ %.6f)', ...
                      best_val_loss, improvement);
        fprintf('%s\n', msg);

        fid = fopen(log_file, 'a');
        fprintf(fid, '%s\n', msg);
        fclose(fid);
    else
        % No improvement
        epochs_no_improve = epochs_no_improve + 1;

        msg = sprintf('[Early Stopping] ⚠ No improvement: %d/%d epochs (current: %.6f, best: %.6f)', ...
                      epochs_no_improve, patience, current_val_loss, best_val_loss);
        fprintf('%s\n', msg);

        fid = fopen(log_file, 'a');
        fprintf(fid, '%s\n', msg);
        fclose(fid);

        % Check if patience exceeded
        if epochs_no_improve >= patience
            stop_msg = sprintf('\n[Early Stopping] 🛑 STOPPING - No improvement for %d epochs!\nBest validation loss: %.6f\n', ...
                              patience, best_val_loss);
            fprintf('%s\n', stop_msg);

            fid = fopen(log_file, 'a');
            fprintf(fid, '%s\n', stop_msg);
            fprintf(fid, '\n==========================================================\n');
            fprintf(fid, 'TRAINING STOPPED BY EARLY STOPPING\n');
            fprintf(fid, 'Total Time: %.2f minutes\n', elapsed_time/60);
            fprintf(fid, 'Final Epoch: %d\n', info.Epoch);
            fprintf(fid, 'Best Validation Loss: %.6f\n', best_val_loss);
            fprintf(fid, '==========================================================\n');
            fclose(fid);

            stop = true;
        end
    end
end
