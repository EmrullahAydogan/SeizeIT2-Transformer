function [fig_handles, attention_data] = visualize_attention(net, X_sample, varargin)
% VISUALIZE_ATTENTION - Visualize attention weights for explainable AI
%
% Inputs:
%   net - Trained transformer autoencoder network
%   X_sample - Input sample(s) for visualization [C, F, T, batch]
% Optional name-value pairs:
%   'SampleIdx' - Index of sample to visualize (default: 1)
%   'LayerNames' - Specific attention layers to visualize (default: all)
%   'AttentionType' - 'self' (encoder), 'cross' (decoder), or 'all' (default)
%   'HeadIdx' - Specific attention head to visualize (default: all)
%   'SavePath' - Path to save figures (default: '')
%   'ShowPlots' - Display plots (default: true)
%   'ColorMap' - Colormap for heatmaps (default: 'parula')
%
% Outputs:
%   fig_handles - Array of figure handles
%   attention_data - Struct containing extracted attention weights
%
% Purpose:
%   Provides interpretability for transformer autoencoder predictions by:
%   1. Extracting attention weights from all attention layers
%   2. Visualizing attention patterns across time-frequency dimensions
%   3. Identifying which input features contribute to reconstruction
%   4. Enabling analysis of seizure detection decisions
%
% Note: Requires network to have attention layers with identifiable names

% Parse inputs
p = inputParser;
addParameter(p, 'SampleIdx', 1, @isnumeric);
addParameter(p, 'LayerNames', {}, @iscell);
addParameter(p, 'AttentionType', 'all', @ischar);
addParameter(p, 'HeadIdx', 'all', @(x) isnumeric(x) || strcmp(x, 'all'));
addParameter(p, 'SavePath', '', @ischar);
addParameter(p, 'ShowPlots', true, @islogical);
addParameter(p, 'ColorMap', 'parula', @ischar);
parse(p, varargin{:});
opts = p.Results;

fprintf('\n=== ATTENTION VISUALIZATION ===\n');
fprintf('Sample index: %d\n', opts.SampleIdx);
fprintf('Attention type: %s\n', opts.AttentionType);

% Ensure sample index is valid
if opts.SampleIdx > size(X_sample, 4)
    error('SampleIdx %d exceeds batch size %d', opts.SampleIdx, size(X_sample, 4));
end

% Extract single sample for visualization
X_single = X_sample(:, :, :, opts.SampleIdx);

fprintf('Input shape: %s\n', mat2str(size(X_single)));

%% 1. Extract Attention Weights
fprintf('\n--- Extracting Attention Weights ---\n');

attention_data = extract_attention_weights(net, X_single, opts);

if isempty(attention_data)
    error('No attention weights found in network. Check layer names.');
end

fprintf('Extracted attention weights from %d layers:\n', length(attention_data));
for i = 1:length(attention_data)
    layer_info = attention_data(i);
    fprintf('  %d. %s: %s attention, %d heads, shape %s\n', ...
        i, layer_info.layer_name, layer_info.attention_type, ...
        layer_info.num_heads, mat2str(size(layer_info.weights{1})));
end

%% 2. Visualize Attention Weights
fig_handles = [];

if opts.ShowPlots
    fprintf('\n--- Generating Visualizations ---\n');

    % Create main summary figure
    fig_summary = visualize_attention_summary(attention_data, X_single, opts);
    fig_handles = [fig_handles, fig_summary];

    % Create individual layer figures
    for i = 1:length(attention_data)
        fig_layer = visualize_attention_layer(attention_data(i), X_single, i, opts);
        fig_handles = [fig_handles, fig_layer];
    end

    % Create time-frequency analysis figure
    fig_tf = visualize_time_frequency_attention(attention_data, X_single, opts);
    fig_handles = [fig_handles, fig_tf];
end

%% 3. Save Results
if ~isempty(opts.SavePath)
    save_attention_results(attention_data, fig_handles, opts);
end

fprintf('\nAttention visualization complete.\n');

end

%% Helper Functions

function attention_data = extract_attention_weights(net, X_sample, opts)
% EXTRACT_ATTENTION_WEIGHTS - Extract attention weights from network

    attention_data = [];

    % Get network layers
    layers = net.Layers;

    % Check if network has UserData with attention layer info
    if isfield(net.UserData, 'attention_layers')
        attention_layer_names = net.UserData.attention_layers;
        fprintf('Found %d attention layers in network UserData\n', length(attention_layer_names));
    else
        % Try to identify attention layers by name pattern
        attention_layer_names = {};
        for i = 1:length(layers)
            layer_name = layers(i).Name;
            if contains(layer_name, 'attention', 'IgnoreCase', true)
                attention_layer_names{end+1} = layer_name;
            end
        end
        fprintf('Found %d attention layers by name pattern\n', length(attention_layer_names));
    end

    % Filter by layer names if specified
    if ~isempty(opts.LayerNames)
        attention_layer_names = intersect(attention_layer_names, opts.LayerNames);
        fprintf('Filtered to %d specified layers\n', length(attention_layer_names));
    end

    % Filter by attention type
    filtered_names = {};
    for i = 1:length(attention_layer_names)
        layer_name = attention_layer_names{i};

        % Determine attention type from name
        if contains(layer_name, 'self', 'IgnoreCase', true)
            attn_type = 'self';
        elseif contains(layer_name, 'cross', 'IgnoreCase', true)
            attn_type = 'cross';
        else
            attn_type = 'unknown';
        end

        % Apply filter
        if strcmp(opts.AttentionType, 'all') || strcmp(opts.AttentionType, attn_type)
            filtered_names{end+1} = layer_name;
        end
    end
    attention_layer_names = filtered_names;

    % Extract weights for each attention layer
    for i = 1:length(attention_layer_names)
        layer_name = attention_layer_names{i};

        try
            % Create activation network for this layer
            act_net = activations(net, X_sample, layer_name, 'OutputAs', 'channels');

            % Determine attention type from name
            if contains(layer_name, 'self', 'IgnoreCase', true)
                attn_type = 'self';
            elseif contains(layer_name, 'cross', 'IgnoreCase', true)
                attn_type = 'cross';
            else
                attn_type = 'unknown';
            end

            % Parse layer name to get layer index
            tokens = regexp(layer_name, '(\d+)$', 'tokens');
            if ~isempty(tokens)
                layer_idx = str2double(tokens{1}{1});
            else
                layer_idx = i;
            end

            % Store attention data
            layer_data = struct();
            layer_data.layer_name = layer_name;
            layer_data.layer_idx = layer_idx;
            layer_data.attention_type = attn_type;
            layer_data.weights = {act_net};  % Cell array for multiple heads

            % Determine number of attention heads
            % For multiHeadAttentionLayer, weights shape depends on implementation
            if ndims(act_net) == 4
                % Assuming shape: [seq_len, seq_len, heads, 1]
                layer_data.num_heads = size(act_net, 3);
                fprintf('  Layer %s: %d attention heads\n', layer_name, layer_data.num_heads);
            else
                layer_data.num_heads = 1;
            end

            attention_data = [attention_data, layer_data];

        catch ME
            fprintf('Warning: Could not extract weights from layer %s: %s\n', ...
                layer_name, ME.message);
        end
    end
end

function fig = visualize_attention_summary(attention_data, X_sample, opts)
% VISUALIZE_ATTENTION_SUMMARY - Create summary visualization

    fig = figure('Position', [100, 100, 1400, 800], 'Name', 'Attention Summary');

    num_layers = length(attention_data);

    % Determine subplot grid
    rows = ceil(sqrt(num_layers));
    cols = ceil(num_layers / rows);

    % Plot each layer
    for i = 1:num_layers
        subplot(rows, cols, i);

        layer_info = attention_data(i);

        % Get attention weights (average across heads if multiple)
        weights = layer_info.weights{1};

        if ndims(weights) == 4
            % Average across heads
            weights_mean = mean(weights, 3);
            weights_mean = squeeze(weights_mean);  % Remove singleton dim
        else
            weights_mean = weights;
        end

        % Plot attention matrix
        imagesc(weights_mean);
        colorbar;
        colormap(opts.ColorMap);

        title_str = sprintf('%s\n%s', ...
            strrep(layer_info.layer_name, '_', '\_'), ...
            layer_info.attention_type);
        title(title_str, 'FontSize', 10);

        xlabel('Key Position', 'FontSize', 8);
        ylabel('Query Position', 'FontSize', 8);

        axis equal tight;
    end

    % Add overall title
    sgtitle('Attention Weight Matrices (Averaged Across Heads)', 'FontSize', 14, 'FontWeight', 'bold');

    % Save if requested
    if ~isempty(opts.SavePath)
        saveas(fig, fullfile(opts.SavePath, 'attention_summary.png'));
    end
end

function fig = visualize_attention_layer(layer_data, X_sample, layer_idx, opts)
% VISUALIZE_ATTENTION_LAYER - Detailed visualization for a single layer

    fig = figure('Position', [100, 100, 1200, 800], ...
        'Name', sprintf('Attention Layer: %s', layer_data.layer_name));

    weights = layer_data.weights{1};
    num_heads = layer_data.num_heads;

    % Plot individual attention heads
    if num_heads > 1
        rows = ceil(sqrt(num_heads));
        cols = ceil(num_heads / rows);

        for h = 1:num_heads
            subplot(rows, cols, h);

            if ndims(weights) == 4
                head_weights = squeeze(weights(:, :, h, :));
            else
                head_weights = weights;
            end

            imagesc(head_weights);
            colorbar;
            colormap(opts.ColorMap);

            title(sprintf('Head %d', h), 'FontSize', 10);
            xlabel('Key Position', 'FontSize', 8);
            ylabel('Query Position', 'FontSize', 8);

            axis equal tight;
        end

        sgtitle(sprintf('%s - Individual Attention Heads', ...
            strrep(layer_data.layer_name, '_', '\_')), 'FontSize', 12);

    else
        % Single head
        subplot(2, 2, 1);
        imagesc(squeeze(weights));
        colorbar;
        colormap(opts.ColorMap);
        title('Attention Weights', 'FontSize', 10);
        xlabel('Key Position');
        ylabel('Query Position');
        axis equal tight;

        % Plot attention distribution
        subplot(2, 2, 2);
        hist(squeeze(weights(:)), 50);
        xlabel('Attention Weight');
        ylabel('Frequency');
        title('Attention Weight Distribution', 'FontSize', 10);
        grid on;

        % Plot max attention per query
        subplot(2, 2, 3);
        max_attention = max(squeeze(weights), [], 2);
        plot(1:length(max_attention), max_attention, 'b-', 'LineWidth', 2);
        xlabel('Query Position');
        ylabel('Max Attention Weight');
        title('Maximum Attention per Query', 'FontSize', 10);
        grid on;

        % Plot attention entropy (uncertainty)
        subplot(2, 2, 4);
        weights_normalized = squeeze(weights) + eps;
        weights_normalized = weights_normalized ./ sum(weights_normalized, 2);
        attention_entropy = -sum(weights_normalized .* log2(weights_normalized), 2);
        plot(1:length(attention_entropy), attention_entropy, 'r-', 'LineWidth', 2);
        xlabel('Query Position');
        ylabel('Attention Entropy (bits)');
        title('Attention Uncertainty', 'FontSize', 10);
        grid on;
    end

    % Save if requested
    if ~isempty(opts.SavePath)
        save_name = sprintf('attention_layer_%s.png', layer_data.layer_name);
        save_name = regexprep(save_name, '[<>:"/\\|?*]', '_');  % Remove invalid chars
        saveas(fig, fullfile(opts.SavePath, save_name));
    end
end

function fig = visualize_time_frequency_attention(attention_data, X_sample, opts)
% VISUALIZE_TIME_FREQUENCY_ATTENTION - Map attention to time-frequency domain

    fig = figure('Position', [100, 100, 1000, 800], 'Name', 'Time-Frequency Attention Analysis');

    % Get input dimensions
    [C, F, T] = size(X_sample);  % Channels, Freq bins, Time frames
    seq_len = F * T;  % Total sequence length

    % Find encoder self-attention layers (most relevant for input analysis)
    encoder_layers = [];
    for i = 1:length(attention_data)
        if strcmp(attention_data(i).attention_type, 'self') && ...
           contains(attention_data(i).layer_name, 'encoder', 'IgnoreCase', true)
            encoder_layers = [encoder_layers, i];
        end
    end

    if isempty(encoder_layers)
        fprintf('No encoder self-attention layers found for time-frequency analysis\n');
        return;
    end

    % Use first encoder layer
    layer_idx = encoder_layers(1);
    layer_data = attention_data(layer_idx);

    % Get attention weights (average across heads)
    weights = layer_data.weights{1};
    if ndims(weights) == 4
        weights_mean = mean(weights, 3);
        weights_mean = squeeze(weights_mean);
    else
        weights_mean = weights;
    end

    % Reshape attention to time-frequency grid
    % attention_matrix is [seq_len, seq_len] where seq_len = F * T
    attention_matrix = weights_mean;

    % Plot 1: Attention matrix with time-frequency labels
    subplot(2, 2, 1);
    imagesc(attention_matrix);
    colorbar;
    colormap(opts.ColorMap);
    title(sprintf('%s\nAttention Matrix', strrep(layer_data.layer_name, '_', '\_')), 'FontSize', 10);
    xlabel('Key Position (F × T)');
    ylabel('Query Position (F × T)');
    axis equal tight;

    % Plot 2: Average attention to each frequency bin
    subplot(2, 2, 2);

    % Reshape to [F, T, F, T]
    attention_4d = reshape(attention_matrix, [F, T, F, T]);

    % Average across time dimensions: attention from all queries to each frequency
    freq_attention = squeeze(mean(mean(attention_4d, [2, 4]), 1));  % [F, 1]

    bar(1:F, freq_attention);
    xlabel('Frequency Bin Index');
    ylabel('Average Attention');
    title('Attention to Frequency Bins', 'FontSize', 10);
    grid on;

    % Add EEG frequency band labels if available
    if F >= 129  % Our spectrogram has 129 frequency bins (0-125 Hz)
        freq_bands = {
            'Delta (0.5-4 Hz)', 1, 4;
            'Theta (4-8 Hz)', 4, 8;
            'Alpha (8-13 Hz)', 8, 13;
            'Beta (13-30 Hz)', 13, 30;
            'Gamma (30-100 Hz)', 30, 100;
        };

        hold on;
        colors = {'r', 'g', 'b', 'm', 'c'};
        for b = 1:size(freq_bands, 1)
            band_name = freq_bands{b, 1};
            band_start = freq_bands{b, 2};
            band_end = freq_bands{b, 3};

            % Convert Hz to bin indices (assuming linear spacing 0-125 Hz)
            bin_start = round(band_start / 125 * F) + 1;
            bin_end = round(band_end / 125 * F) + 1;

            % Plot band region
            x_band = [bin_start, bin_end, bin_end, bin_start, bin_start];
            y_band = [0, 0, max(freq_attention)*1.1, max(freq_attention)*1.1, 0];
            fill(x_band, y_band, colors{b}, 'FaceAlpha', 0.1, 'EdgeColor', colors{b});

            % Add label
            text(mean([bin_start, bin_end]), max(freq_attention)*0.9, ...
                band_name, 'HorizontalAlignment', 'center', 'FontSize', 8);
        end
    end

    % Plot 3: Average attention to each time frame
    subplot(2, 2, 3);

    % Average across frequency dimensions: attention from all queries to each time frame
    time_attention = squeeze(mean(mean(attention_4d, [1, 3]), 1));  % [T, 1]

    bar(1:T, time_attention);
    xlabel('Time Frame Index');
    ylabel('Average Attention');
    title('Attention to Time Frames', 'FontSize', 10);
    grid on;

    % Each time frame is ~0.57 seconds (4s window / 7 frames)
    time_labels = arrayfun(@(t) sprintf('%.1fs', t * 0.57), 1:T, 'UniformOutput', false);
    xticks(1:T);
    xticklabels(time_labels);

    % Plot 4: Attention heatmap in time-frequency space
    subplot(2, 2, 4);

    % Compute which frequency-time positions receive most attention overall
    total_attention = sum(attention_matrix, 1);  % Attention received by each key position
    attention_map = reshape(total_attention, [F, T]);

    imagesc(1:T, 1:F, attention_map);
    colorbar;
    colormap(opts.ColorMap);
    xlabel('Time Frame');
    ylabel('Frequency Bin');
    title('Total Attention Received (Time-Frequency Map)', 'FontSize', 10);

    % Add time labels
    xticks(1:T);
    xticklabels(time_labels);

    % Add frequency band lines
    if F >= 129
        hold on;
        band_boundaries = [4, 8, 13, 30, 100];
        for b = 1:length(band_boundaries)
            bin_idx = round(band_boundaries(b) / 125 * F) + 1;
            line(xlim, [bin_idx, bin_idx], 'Color', 'w', 'LineStyle', '--', 'LineWidth', 1);
        end
    end

    % Overall title
    sgtitle('Time-Frequency Attention Analysis', 'FontSize', 14, 'FontWeight', 'bold');

    % Save if requested
    if ~isempty(opts.SavePath)
        saveas(fig, fullfile(opts.SavePath, 'time_frequency_attention.png'));
    end
end

function save_attention_results(attention_data, fig_handles, opts)
% SAVE_ATTENTION_RESULTS - Save attention data and figures

    if isempty(opts.SavePath)
        return;
    end

    % Create directory
    if ~isfolder(opts.SavePath)
        mkdir(opts.SavePath);
    end

    fprintf('\n--- Saving Results to %s ---\n', opts.SavePath);

    % Save attention data as MAT file
    save(fullfile(opts.SavePath, 'attention_data.mat'), 'attention_data');

    % Save figures
    for i = 1:length(fig_handles)
        fig = fig_handles(i);
        fig_name = get(fig, 'Name');
        if isempty(fig_name)
            fig_name = sprintf('figure_%d', i);
        end

        % Clean filename
        fig_name = regexprep(fig_name, '[<>:"/\\|?*]', '_');
        fig_name = strrep(fig_name, ' ', '_');

        saveas(fig, fullfile(opts.SavePath, [fig_name '.png']));
        saveas(fig, fullfile(opts.SavePath, [fig_name '.fig']));
    end

    % Save summary text file
    summary_file = fullfile(opts.SavePath, 'attention_summary.txt');
    fid = fopen(summary_file, 'w');

    fprintf(fid, 'ATTENTION VISUALIZATION SUMMARY\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

    fprintf(fid, '=== ATTENTION LAYERS ===\n');
    for i = 1:length(attention_data)
        layer = attention_data(i);
        fprintf(fid, 'Layer %d: %s\n', i, layer.layer_name);
        fprintf(fid, '  Type: %s\n', layer.attention_type);
        fprintf(fid, '  Index: %d\n', layer.layer_idx);
        fprintf(fid, '  Heads: %d\n', layer.num_heads);
        fprintf(fid, '  Weight shape: %s\n\n', mat2str(size(layer.weights{1})));
    end

    fprintf(fid, '\n=== ANALYSIS ===\n');
    fprintf(fid, 'Total layers: %d\n', length(attention_data));
    fprintf(fid, 'Self-attention layers: %d\n', ...
        sum(strcmp({attention_data.attention_type}, 'self')));
    fprintf(fid, 'Cross-attention layers: %d\n', ...
        sum(strcmp({attention_data.attention_type}, 'cross')));

    fclose(fid);

    fprintf('Results saved successfully.\n');
end