function [model, attention_weights] = transformer_autoencoder(params, input_shape)
% TRANSFORMER_AUTOENCODER - Transformer-based autoencoder for seizure detection
%
% Inputs:
%   params - Struct containing model hyperparameters
%   input_shape - Shape of input data [channels, freq_bins, time_frames, batch]
%
% Outputs:
%   model - MATLAB layerGraph object for the autoencoder
%   attention_weights - Cell array of attention layer names for visualization
%
% Model Architecture:
%   1. Input embedding (linear projection)
%   2. Positional encoding (sinusoidal or learned)
%   3. Transformer encoder (multiple layers with multi-head attention + residual)
%   4. Transformer decoder (for reconstruction + residual)
%   5. Output projection (back to input space)
%
% Attention visualization: Returns layer names that can be used to extract
% attention weights for explainable AI.

% Default parameters if not provided
if nargin < 1
    params = struct();
end

if nargin < 2
    % Default input shape based on preprocessing
    % [channels=16, freq_bins=129, time_frames=7, batch]
    input_shape = [16, 129, 7, NaN];  % NaN for variable batch size
end

% Set default parameters
params = set_default_params(params, input_shape);

fprintf('Building Transformer Autoencoder...\n');
fprintf('  Input shape: %s\n', mat2str(input_shape));
fprintf('  Embedding dim: %d\n', params.embedding_dim);
fprintf('  Num heads: %d\n', params.num_heads);
fprintf('  Num encoder layers: %d\n', params.num_encoder_layers);
fprintf('  Num decoder layers: %d\n', params.num_decoder_layers);
fprintf('  Dropout rate: %.2f\n', params.dropout_rate);

% Calculate input size for sequenceInputLayer
% input_shape: [C, F, T] -> input_size = C * F, sequence_length = T
input_size = input_shape(1) * input_shape(2);
sequence_length = input_shape(3);

fprintf('  Sequence input: size=%d, length=%d\n', input_size, sequence_length);

%% 1. Create Layer Graph
lgraph = layerGraph();

%% 2. Input Layer (Sequence Input)
% Convert spectrogram [C, F, T] to sequence [C*F, T]
input_layer = sequenceInputLayer(input_size, 'Name', 'input', 'Normalization', 'none');
lgraph = addLayers(lgraph, input_layer);

%% 3. Input Embedding (Linear Projection to Embedding Dimension)
embedding_layer = fullyConnectedLayer(params.embedding_dim, ...
    'Name', 'input_embedding', ...
    'WeightsInitializer', 'glorot', ...
    'BiasInitializer', 'zeros');
lgraph = addLayers(lgraph, embedding_layer);
lgraph = connectLayers(lgraph, 'input', 'input_embedding');

%% 4. Positional Encoding
if strcmp(params.positional_encoding_type, 'sinusoidal')
    % Sinusoidal positional encoding
    pos_encoding = functionLayer(@(x) add_sinusoidal_positional_encoding(x, params), ...
        'Name', 'positional_encoding');
    lgraph = addLayers(lgraph, pos_encoding);
    lgraph = connectLayers(lgraph, 'input_embedding', 'positional_encoding');
    current_output = 'positional_encoding';
else
    % Learned positional encoding (simplified - no addition layer for now)
    pos_encoding = embeddingLayer(sequence_length, params.embedding_dim, ...
        'Name', 'positional_encoding');
    lgraph = addLayers(lgraph, pos_encoding);
    % Note: For learned encoding, we need addition layer - simplified for now
    lgraph = connectLayers(lgraph, 'input_embedding', 'positional_encoding');
    current_output = 'positional_encoding';
end

%% 5. Transformer Encoder Layers
attention_layer_names = {};  % For attention visualization

for i = 1:params.num_encoder_layers
    % Create encoder block with residual connections
    [lgraph, current_output, attn_name] = add_encoder_block(lgraph, current_output, params, i);
    attention_layer_names{end+1} = attn_name;
end

%% 6. Bottleneck (Latent Representation)
bottleneck_norm = layerNormalizationLayer('Name', 'bottleneck_norm');
lgraph = addLayers(lgraph, bottleneck_norm);
lgraph = connectLayers(lgraph, current_output, 'bottleneck_norm');
current_output = 'bottleneck_norm';

%% 7. Transformer Decoder Layers
for i = 1:params.num_decoder_layers
    % Create decoder block with residual connections
    [lgraph, current_output, attn_name] = add_decoder_block(lgraph, current_output, params, i);
    attention_layer_names{end+1} = attn_name;
end

%% 8. Output Projection (Back to original input size)
output_projection = fullyConnectedLayer(input_size, ...
    'Name', 'output_projection', ...
    'WeightsInitializer', 'glorot', ...
    'BiasInitializer', 'zeros');
lgraph = addLayers(lgraph, output_projection);
lgraph = connectLayers(lgraph, current_output, 'output_projection');
current_output = 'output_projection';

%% 9. Regression Output Layer
regression_layer = regressionLayer('Name', 'output');
lgraph = addLayers(lgraph, regression_layer);
lgraph = connectLayers(lgraph, current_output, 'output');

% Store attention layer names as UserData (if supported)
try
    lgraph.UserData.attention_layers = attention_layer_names;
catch
    % UserData may not be available
end

fprintf('Transformer Autoencoder built successfully.\n');
fprintf('  Total layers: %d\n', numel(lgraph.Layers));
fprintf('  Attention layers: %d\n', numel(attention_layer_names));

% Return attention weights info
attention_weights = attention_layer_names;
model = lgraph;

end

%% ========== HELPER FUNCTIONS ==========

function params = set_default_params(params, input_shape)
% SET_DEFAULT_PARAMS - Set default hyperparameters

    if ~isfield(params, 'embedding_dim')
        params.embedding_dim = 64;  % Default from config
    end

    if ~isfield(params, 'num_heads')
        params.num_heads = 4;  % Default from config
    end

    if ~isfield(params, 'num_encoder_layers')
        params.num_encoder_layers = 3;
    end

    if ~isfield(params, 'num_decoder_layers')
        params.num_decoder_layers = 3;
    end

    if ~isfield(params, 'dropout_rate')
        params.dropout_rate = 0.1;
    end

    if ~isfield(params, 'feedforward_dim')
        params.feedforward_dim = params.embedding_dim * 4;  % Standard transformer
    end

    if ~isfield(params, 'activation')
        params.activation = 'gelu';  % GELU activation for transformers
    end

    if ~isfield(params, 'use_skip_connections')
        params.use_skip_connections = false;  % Optional skip connections
    end

    if ~isfield(params, 'positional_encoding_type')
        params.positional_encoding_type = 'sinusoidal';  % 'sinusoidal' or 'learned'
    end

    % Calculate sequence length from input shape
    % input_shape: [C, F, T] -> sequence length = T (after flattening C*F features)
    params.seq_len = input_shape(3);  % time_frames
end

function x_encoded = add_sinusoidal_positional_encoding(x, params)
% ADD_SINUSOIDAL_POSITIONAL_ENCODING - Add fixed sinusoidal encoding
% x shape: [batch, seq_len, embedding_dim]

    [batch_size, seq_len, d_model] = size(x);

    % Create positional encoding matrix
    pe = zeros(seq_len, d_model);

    for pos = 1:seq_len
        for i = 1:2:d_model
            pe(pos, i) = sin(pos / (10000 ^ ((i-1)/d_model)));
            if i+1 <= d_model
                pe(pos, i+1) = cos(pos / (10000 ^ ((i-1)/d_model)));
            end
        end
    end

    % Add to input (broadcast across batch)
    x_encoded = x + reshape(pe, [1, seq_len, d_model]);
end

function [lgraph, output_name, attn_name] = add_encoder_block(lgraph, input_name, params, layer_idx)
% ADD_ENCODER_BLOCK - Add encoder block with residual connections
% Returns updated layerGraph, output layer name, and attention layer name

    attn_name = sprintf('encoder_self_attention_%d', layer_idx);

    % 1. Self-Attention Layer
    attention_layer = selfAttentionLayer(params.num_heads, params.embedding_dim, ...
        'Name', attn_name);
    lgraph = addLayers(lgraph, attention_layer);
    lgraph = connectLayers(lgraph, input_name, attn_name);

    % 2. First Residual Connection (attention output + input)
    add1_name = sprintf('encoder_add1_%d', layer_idx);
    add1 = additionLayer(2, 'Name', add1_name);
    lgraph = addLayers(lgraph, add1);
    lgraph = connectLayers(lgraph, attn_name, [add1_name '/in1']);
    lgraph = connectLayers(lgraph, input_name, [add1_name '/in2']);

    % 3. Layer Normalization 1
    norm1_name = sprintf('encoder_norm1_%d', layer_idx);
    norm1 = layerNormalizationLayer('Name', norm1_name);
    lgraph = addLayers(lgraph, norm1);
    lgraph = connectLayers(lgraph, add1_name, norm1_name);

    % 4. Feed-Forward Network
    ffn_layers = feed_forward_block(params, sprintf('encoder_ffn_%d', layer_idx));

    % Add FFN layers to graph and connect
    prev_layer = norm1_name;
    for i = 1:length(ffn_layers)
        layer = ffn_layers{i};
        lgraph = addLayers(lgraph, layer);
        lgraph = connectLayers(lgraph, prev_layer, layer.Name);
        prev_layer = layer.Name;
    end
    ffn_output = prev_layer;

    % 5. Second Residual Connection (FFN output + norm1 output)
    add2_name = sprintf('encoder_add2_%d', layer_idx);
    add2 = additionLayer(2, 'Name', add2_name);
    lgraph = addLayers(lgraph, add2);
    lgraph = connectLayers(lgraph, ffn_output, [add2_name '/in1']);
    lgraph = connectLayers(lgraph, norm1_name, [add2_name '/in2']);

    % 6. Layer Normalization 2
    norm2_name = sprintf('encoder_norm2_%d', layer_idx);
    norm2 = layerNormalizationLayer('Name', norm2_name);
    lgraph = addLayers(lgraph, norm2);
    lgraph = connectLayers(lgraph, add2_name, norm2_name);

    output_name = norm2_name;
end

function [lgraph, output_name, attn_name] = add_decoder_block(lgraph, input_name, params, layer_idx)
% ADD_DECODER_BLOCK - Add decoder block with residual connections
% Returns updated layerGraph, output layer name, and attention layer name

    attn_name = sprintf('decoder_self_attention_%d', layer_idx);

    % 1. Self-Attention Layer
    attention_layer = selfAttentionLayer(params.num_heads, params.embedding_dim, ...
        'Name', attn_name);
    lgraph = addLayers(lgraph, attention_layer);
    lgraph = connectLayers(lgraph, input_name, attn_name);

    % 2. First Residual Connection (attention output + input)
    add1_name = sprintf('decoder_add1_%d', layer_idx);
    add1 = additionLayer(2, 'Name', add1_name);
    lgraph = addLayers(lgraph, add1);
    lgraph = connectLayers(lgraph, attn_name, [add1_name '/in1']);
    lgraph = connectLayers(lgraph, input_name, [add1_name '/in2']);

    % 3. Layer Normalization 1
    norm1_name = sprintf('decoder_norm1_%d', layer_idx);
    norm1 = layerNormalizationLayer('Name', norm1_name);
    lgraph = addLayers(lgraph, norm1);
    lgraph = connectLayers(lgraph, add1_name, norm1_name);

    % 4. Feed-Forward Network
    ffn_layers = feed_forward_block(params, sprintf('decoder_ffn_%d', layer_idx));

    % Add FFN layers to graph and connect
    prev_layer = norm1_name;
    for i = 1:length(ffn_layers)
        layer = ffn_layers{i};
        lgraph = addLayers(lgraph, layer);
        lgraph = connectLayers(lgraph, prev_layer, layer.Name);
        prev_layer = layer.Name;
    end
    ffn_output = prev_layer;

    % 5. Second Residual Connection (FFN output + norm1 output)
    add2_name = sprintf('decoder_add2_%d', layer_idx);
    add2 = additionLayer(2, 'Name', add2_name);
    lgraph = addLayers(lgraph, add2);
    lgraph = connectLayers(lgraph, ffn_output, [add2_name '/in1']);
    lgraph = connectLayers(lgraph, norm1_name, [add2_name '/in2']);

    % 6. Layer Normalization 2
    norm2_name = sprintf('decoder_norm2_%d', layer_idx);
    norm2 = layerNormalizationLayer('Name', norm2_name);
    lgraph = addLayers(lgraph, norm2);
    lgraph = connectLayers(lgraph, add2_name, norm2_name);

    output_name = norm2_name;
end

function layers = feed_forward_block(params, name_prefix)
% FEED_FORWARD_BLOCK - Transformer feed-forward network
% Returns cell array of layers

    % First linear layer (expansion)
    linear1 = fullyConnectedLayer(params.feedforward_dim, ...
        'Name', [name_prefix '_linear1'], ...
        'WeightsInitializer', 'glorot', ...
        'BiasInitializer', 'zeros');

    % Activation
    if strcmp(params.activation, 'gelu')
        activation = functionLayer(@(x) 0.5 * x .* (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x.^3))), ...
            'Name', [name_prefix '_gelu']);
    else
        activation = reluLayer('Name', [name_prefix '_relu']);
    end

    % Dropout
    dropout = dropoutLayer(params.dropout_rate, 'Name', [name_prefix '_dropout']);

    % Second linear layer (projection back)
    linear2 = fullyConnectedLayer(params.embedding_dim, ...
        'Name', [name_prefix '_linear2'], ...
        'WeightsInitializer', 'glorot', ...
        'BiasInitializer', 'zeros');

    layers = {linear1, activation, dropout, linear2};
end