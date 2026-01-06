% test_model_final.m
% SeizeIT2 - Model Test (Final Fix for Predict)
% Düzeltme: 'predict' fonksiyonuna Table yerine Cell Array veriyoruz.

clc; clear; close all;

% --- AYARLAR ---
dataDir = "/home/developer/Desktop/SeizeIT2/MatlabProject/ModelData";
testDir = fullfile(dataDir, "Test");
modelPath = fullfile(dataDir, "Trained_Transformer_Final.mat");

% 1. MODELİ YÜKLE
fprintf('Model yükleniyor...\n');
if ~isfile(modelPath)
    error('Model bulunamadı: %s', modelPath);
end
loaded = load(modelPath, 'net');
net = loaded.net;

% 2. TEST DOSYASI SEÇ (Nöbet Dosyası)
files = dir(fullfile(testDir, "*_Seizures.mat"));
if isempty(files)
    fprintf('Uyarı: Nöbet dosyası yok, normal dosya deneniyor.\n');
    files = dir(fullfile(testDir, "*.mat"));
end
testFile = fullfile(files(1).folder, files(1).name);
fprintf('Test Dosyası: %s\n', files(1).name);

% 3. VERİYİ OKU VE HAZIRLA
d = load(testFile); 
rawX = d.X; 
rawX = squeeze(rawX); % [16, 1000, Batch]

% Cell Array'e çevir (Predict için en güvenli format)
numSamples = size(rawX, 3);
dataCell = cell(numSamples, 1);
for i = 1:numSamples
    dataCell{i} = rawX(:, :, i);
end

% 4. TAHMİN YAP (Table yerine Cell Array veriyoruz)
fprintf('Model tahmin yapıyor (%d pencere)... ', numSamples);
% 'MiniBatchSize' ekleyelim ki hafıza şişmesin
reconstructions = predict(net, dataCell, 'MiniBatchSize', 16); 
fprintf('[Tamam]\n');

% 5. HATA HESAPLA (MSE - Anomaly Score)
errors = zeros(numSamples, 1);

% Predict çıktısı bazen Cell bazen Matris dönebilir, kontrol edelim
isOutputCell = iscell(reconstructions);

for i = 1:numSamples
    % Gerçek Sinyal
    original = dataCell{i};       
    
    % Modelin Çizimi (Format kontrolü)
    if isOutputCell
        reconstructed = reconstructions{i};
    else
        reconstructed = reconstructions(:, :, i);
    end
    
    % Hata Hesapla (Mean Squared Error)
    diff = original - reconstructed;
    mse = mean(diff.^2, 'all'); 
    errors(i) = mse;
end

% 6. GÖRSELLEŞTİRME
figure('Name', 'Final Seizure Detection', 'Color', 'w', 'Position', [100 100 1000 700]);

% A) Hata Grafiği (Anomaly Score)
subplot(2,1,1);
plot(errors, 'r.-', 'LineWidth', 1.5, 'MarkerSize', 10);
title(['Nöbet Tespiti (Hata Skoru) - ' files(1).name], 'Interpreter', 'none', 'FontSize', 12);
xlabel('Pencere No');
ylabel('Reconstruction Error (MSE)');
grid on;

% Ortalama hata çizgisini ekle
avgErr = mean(errors);
yline(avgErr, 'k--', sprintf('Ortalama: %.2f', avgErr), 'LineWidth', 2);

% B) En Kötü An (En Yüksek Hata)
[maxErr, idx] = max(errors);
subplot(2,1,2);

% Orijinal vs Tahmin
plot(dataCell{idx}(1,:), 'b', 'LineWidth', 1.5, 'DisplayName', 'Gerçek Sinyal (Nöbet)');
hold on;

if isOutputCell
    recPlot = reconstructions{idx};
else
    recPlot = reconstructions(:,:,idx);
end
plot(recPlot(1,:), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Modelin Çizimi (Normal)');

title(sprintf('Modelin Çuvalladığı An (Pencere %d) - Hata: %.2f', idx, maxErr), 'FontSize', 12, 'Color', 'r');
legend;
grid on;
xlabel('Zaman (Örnek)');

% SONUÇ RAPORU
fprintf('\n=== SONUÇ RAPORU ===\n');
fprintf('Dosya: %s\n', files(1).name);
fprintf('Toplam Pencere: %d\n', numSamples);
fprintf('Ortalama Hata:  %.4f\n', avgErr);
fprintf('Maksimum Hata:  %.4f\n', maxErr);
fprintf('---------------------------------\n');
if maxErr > 5 % Eşik değeri tahmini (Eğitimde loss 0.3 idi)
    fprintf('✅ TESPİT BAŞARILI: Yüksek hata oranları nöbeti işaret ediyor.\n');
else
    fprintf('⚠️ DİKKAT: Hata oranları düşük. Model nöbeti normal sanıyor olabilir.\n');
end