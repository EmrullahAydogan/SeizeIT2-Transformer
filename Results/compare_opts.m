% Compare all Bayesian optimization results

files = {
    'BayesianOpt_QuickTest/optimization_results.mat'
    'BayesianOpt_QuickTest2/optimization_results.mat'
    'BayesianOpt_QuickTest3/optimization_results.mat'
    'BayesianOpt_QuickTest4/optimization_results.mat'
    'BayesianOpt_Final_10iter/optimization_results.mat'
};

fprintf('\n========================================\n');
fprintf('BAYESIAN OPTIMIZATION COMPARISON\n');
fprintf('========================================\n\n');

comp = struct([]);
for i = 1:length(files)
    try
        data = load(files{i});
        comp(i).name = files{i};
        comp(i).params = data.best_params;

        fprintf('[%d] %s\n', i, strrep(files{i}, '/optimization_results.mat', ''));
        fprintf('    Objective: %.4f (Lower is better)\n', data.best_params.objective);
        fprintf('    LR: %.6f | Embed: %d | Heads: %d\n', ...
            data.best_params.learning_rate, data.best_params.embedding_dim, data.best_params.num_heads);
        fprintf('    Enc: %d | Dec: %d | Dropout: %.4f\n', ...
            data.best_params.num_encoder_layers, data.best_params.num_decoder_layers, data.best_params.dropout_rate);
        fprintf('    Batch: %d | FFN Mult: %.2f\n\n', ...
            data.best_params.batch_size, data.best_params.ffn_multiplier);
    catch ME
        fprintf('[%d] ERROR: %s\n\n', i, ME.message);
    end
end

% Find best
if ~isempty(comp)
    objs = arrayfun(@(x) x.params.objective, comp);
    [best_obj, best_idx] = min(objs);

    fprintf('========================================\n');
    fprintf('🏆 BEST RESULT\n');
    fprintf('========================================\n');
    fprintf('File: %s\n', strrep(comp(best_idx).name, '/optimization_results.mat', ''));
    fprintf('Objective: %.4f\n\n', best_obj);

    fprintf('OPTIMAL HYPERPARAMETERS:\n');
    fprintf('------------------------\n');
    fprintf('  Learning Rate:      %.6f\n', comp(best_idx).params.learning_rate);
    fprintf('  Embedding Dim:      %d\n', comp(best_idx).params.embedding_dim);
    fprintf('  Attention Heads:    %d\n', comp(best_idx).params.num_heads);
    fprintf('  Encoder Layers:     %d\n', comp(best_idx).params.num_encoder_layers);
    fprintf('  Decoder Layers:     %d\n', comp(best_idx).params.num_decoder_layers);
    fprintf('  Dropout Rate:       %.4f\n', comp(best_idx).params.dropout_rate);
    fprintf('  Batch Size:         %d\n', comp(best_idx).params.batch_size);
    fprintf('  FFN Multiplier:     %.2f\n', comp(best_idx).params.ffn_multiplier);
    fprintf('  Feedforward Dim:    %d\n', round(comp(best_idx).params.embedding_dim * comp(best_idx).params.ffn_multiplier));
    fprintf('\n');
end
