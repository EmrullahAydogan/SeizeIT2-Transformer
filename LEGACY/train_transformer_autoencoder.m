% train_transformer_final_v3.m
% SeizeIT2 - Transformer Autoencoder (Table Access Fix)
% Düzeltme: Tablo erişim hatası ve Datastore transform yapısı onarıldı.

clc; clear; close all;

% --- AYARLAR ---
dataDir = "/home/developer/Desktop/SeizeIT2/MatlabProject/ModelData";
trainDir = fullfile(dataDir, "Train");
checkpointDir = "/home/developer/Desktop/SeizeIT2/MatlabProject/Checkpoints";

if ~isfolder(checkpointDir), mkdir(checkpointDir); end

% 1. VERİ OKUYUCU (Data Pipeline)
fprintf('Veri kaynağı hazırlanıyor...\n');

% Dosyaları okuyan datastore
fds = fileDatastore(fullfile(trainDir, "*.mat"), 'ReadFcn', @readMatAsTable);

% Autoencoder Dönüşümü:
% trainNetwork, [Giriş, Hedef] şeklinde 2 sütunlu bir tablo ister.
% Bizim hedefimiz girişin aynısı (Autoencoder).
trainDS = transform(fds, @addResponse);

% Boyut Kontrolü (Hata düzeltildi)
try
    previewData = preview(trainDS);
    % previewData bir tablodur. İlk sütun (Input) ilk satırın içindeki hücreyi alalım.
    inputCell = previewData{1, 1}; % {1,1} = 1. Satır, 1. Sütun (Hücre Dizisi)
    inputSample = inputCell{1};    % Hücre dizisinin içindeki ilk matris
    [c, t] = size(inputSample);
    fprintf('Giriş Sinyali: %d Kanal x %d Zaman Adımı\n', c, t);
catch ME
    error('Veri önizleme hatası: %s', ME.message);
end

% 2. MİMARİ (Transformer-CNN)
% ---------------------------------------------------------
lgraph = layerGraph();

embeddingDim = 64; 
numHeads = 4;      

layers = [
    % GİRİŞ
    sequenceInputLayer(c, 'MinLength', t, 'Name', 'input', 'Normalization', 'zscore')
    
    % EMBEDDING (Kanal Genişletme)
    convolution1dLayer(5, embeddingDim, 'Padding', 'same', 'Name', 'embed_conv')
    layerNormalizationLayer('Name', 'ln1')
    reluLayer('Name', 'relu1')
    
    % TRANSFORMER (Attention)
    selfAttentionLayer(numHeads, embeddingDim, 'Name', 'attention')
    layerNormalizationLayer('Name', 'ln2')
    reluLayer('Name', 'relu2')
    
    % DECODER (Reconstruction)
    convolution1dLayer(5, c, 'Padding', 'same', 'Name', 'decode_conv')
    
    % ÇIKIŞ
    regressionLayer('Name', 'output')
];

lgraph = addLayers(lgraph, layers);

% Modeli Kontrol Et
fprintf('Model mimarisi kontrol ediliyor...\n');
try
    analyzeNetwork(lgraph);
    fprintf('Mimari onaylandı.\n');
catch
    fprintf('UYARI: Grafik arayüzü açılamadı, eğitime devam ediliyor.\n');
end

% 3. EĞİTİM AYARLARI
options = trainingOptions('adam', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 16, ...       % Güvenli batch size
    'InitialLearnRate', 1e-4, ...  
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ExecutionEnvironment', 'auto', ...
    'CheckpointPath', checkpointDir);

% 4. EĞİTİMİ BAŞLAT
fprintf('Eğitim Başlatılıyor... (Grafik penceresini izleyin)\n');
try
    [net, trainInfo] = trainNetwork(trainDS, lgraph, options);
    
    % Kaydet
    save(fullfile(dataDir, 'Trained_Transformer_Final.mat'), 'net', 'trainInfo');
    fprintf('BAŞARILI! Model kaydedildi.\n');
    
catch ME
    fprintf('\nEĞİTİM HATASI: %s\n', ME.message);
    fprintf('Detay: %s\n', ME.stack(1).name);
end


% ---------------------------------------------------------
% YARDIMCI FONKSİYONLAR
% ---------------------------------------------------------

function T = readMatAsTable(filename)
    % Dosyayı oku ve tek sütunlu tablo yap: table(CellArray, 'VariableNames', {'Input'})
    d = load(filename);
    rawX = d.X; % [16, 1000, 1, Batch]
    
    rawX = squeeze(rawX); % [16, 1000, Batch]
    
    % Cell Array'e çevir
    numSamples = size(rawX, 3);
    dataCell = cell(numSamples, 1);
    for i = 1:numSamples
        dataCell{i} = rawX(:, :, i);
    end
    
    T = table(dataCell, 'VariableNames', {'Input'});
end

function T_out = addResponse(T_in)
    % Datastore transform fonksiyonu
    % Giriş tablosunu al, 'Response' sütunu ekle (Input ile aynı)
    % Çıktı: [Input, Response]
    T_out = T_in;
    T_out.Response = T_in.Input;
end