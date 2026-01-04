% visualize_explainability.m
% SeizeIT2 - Açıklanabilirlik (Kanal Bazlı Isı Haritası)
% Amaç: Nöbet anındaki yüksek hatanın hangi kanallardan kaynaklandığını göstermek.

clc; clear; close all;

% --- AYARLAR ---
dataDir = "/home/developer/Desktop/SeizeIT2/MatlabProject/ModelData";
testDir = fullfile(dataDir, "Test");
modelPath = fullfile(dataDir, "Trained_Transformer_Final.mat");

% 1. MODELİ YÜKLE
fprintf('Model yükleniyor...\n');
loaded = load(modelPath, 'net');
net = loaded.net;

% 2. NÖBETLİ DOSYAYI SEÇ (sub-015)
files = dir(fullfile(testDir, "*_Seizures.mat"));
seizureFile = fullfile(files(1).folder, files(1).name);
fprintf('Analiz Edilen Dosya: %s\n', files(1).name);

% Veriyi Yükle
d = load(seizureFile);
if isfield(d, 'X'), rawX = squeeze(d.X); else, vars=fieldnames(d); rawX=squeeze(d.(vars{1})); end

% 3. TÜM PENCERELERİ TAHMİN ET
numWindows = size(rawX, 3);
dataCell = cell(numWindows, 1);
for k = 1:numWindows
    dataCell{k} = rawX(:, :, k);
end

fprintf('Model tahmin yapıyor... ');
reconstructions = predict(net, dataCell, 'MiniBatchSize', 16);
fprintf('[Tamam]\n');

% 4. EN YÜKSEK HATALI PENCEREYİ BUL
errors = zeros(numWindows, 1);
isOutputCell = iscell(reconstructions);

for k = 1:numWindows
    original = dataCell{k};
    if isOutputCell, rec = reconstructions{k}; else, rec = reconstructions(:,:,k); end
    
    diff = original - rec;
    errors(k) = mean(diff.^2, 'all');
end

[maxErr, targetIdx] = max(errors);
fprintf('>>> En şiddetli nöbet anı: Pencere %d (Hata: %.2f)\n', targetIdx, maxErr);

% 5. DETAYLI ANALİZ (Giriş vs Çıkış)
originalSignal = dataCell{targetIdx}; % [16 x 1000]
if isOutputCell, reconstructedSignal = reconstructions{targetIdx}; else, reconstructedSignal = reconstructions(:,:,targetIdx); end

% Kanal Bazlı Fark (Mutlak Hata)
% [16 Kanal x 1000 Zaman Adımı]
absErrorMap = abs(originalSignal - reconstructedSignal);

% Kanal İsimleri (Temsili)
channelNames = arrayfun(@(x) sprintf('CH%02d', x), 1:16, 'UniformOutput', false);

% 6. GÖRSELLEŞTİRME (HEATMAP & SİNYAL)
figure('Name', 'Seizure X-Ray (Heatmap)', 'Color', 'w', 'Position', [100 50 1000 900]);

% A) Isı Haritası (Hangi Kanal, Hangi Saniye?)
subplot(3,1,1);
imagesc(absErrorMap);
colormap(jet); % Kırmızı = Yüksek Hata
colorbar;
title(sprintf('Nöbetin Anatomisi (Isı Haritası) - Pencere %d', targetIdx), 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Zaman (Örnek)');
ylabel('Kanal (Sensör)');
yticks(1:16);
yticklabels(channelNames);
clim([0, max(absErrorMap(:)) * 0.7]); % Kontrastı artırarak detayları göster

% B) Kanal Bazlı Ortalama Hata (Bar Grafiği)
channelErrors = mean(absErrorMap, 2); % Her kanalın ortalama hatası
[~, worstCh] = max(channelErrors);

subplot(3,1,2);
bar(channelErrors, 'FaceColor', [0.8 0.2 0.2]);
title('Hangi Kanal En Çok Etkilendi?', 'FontSize', 11);
xlabel('Kanal No');
ylabel('Ortalama Hata');
grid on;
xticks(1:16);
xticklabels(channelNames);

% C) En Kötü Kanalın Sinyali (Kanıt)
subplot(3,1,3);
plot(originalSignal(worstCh, :), 'b', 'LineWidth', 1.5, 'DisplayName', 'Gerçek (Nöbet)');
hold on;
plot(reconstructedSignal(worstCh, :), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Model (Normal)');
title(sprintf('En Sorunlu Kanal: %s (Sinyal Karşılaştırması)', channelNames{worstCh}), 'FontSize', 11, 'Color', 'r');
legend;
grid on;
xlim([1, 1000]);

fprintf('\n=== TANI RAPORU ===\n');
fprintf('Tespit Edilen En Riskli Kanal: CH%02d\n', worstCh);
fprintf('Bu kanalın hata skoru diğerlerinin %.1f katı.\n', max(channelErrors) / mean(channelErrors));