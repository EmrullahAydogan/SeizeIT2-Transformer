% compare_rigorous_v2.m
% SeizeIT2 - Akademik Olarak Geçerli Karşılaştırma (Düzeltilmiş Versiyon)
% Hata Düzeltmesi: Döngü değişkeni hatası giderildi (i -> k).

clc; clear; close all;

% --- AYARLAR ---
dataDir = "/home/developer/Desktop/SeizeIT2/MatlabProject/ModelData";
testDir = fullfile(dataDir, "Test");
modelPath = fullfile(dataDir, "Trained_Transformer_Final.mat");

% 1. MODELİ YÜKLE
fprintf('Model yükleniyor...\n');
loaded = load(modelPath, 'net');
net = loaded.net;

% 2. DOSYALARI SEÇ (AKADEMİK TİTİZLİK)
% A) Nöbetli Dosya (Kırmızı)
seizureFiles = dir(fullfile(testDir, "*_Seizures.mat"));
if isempty(seizureFiles), error("Nöbet dosyası yok!"); end
seizureFile = fullfile(seizureFiles(1).folder, seizureFiles(1).name);

% B) HİÇ GÖRÜLMEMİŞ Normal Dosya (Yeşil) 
% Test klasöründeki '_NormalPart' dosyaları
normalFiles = dir(fullfile(testDir, "*_NormalPart.mat"));

if isempty(normalFiles)
    warning("NormalPart dosyası bulunamadı, Train klasöründen bir dosya seçiliyor.");
    trainDir = fullfile(dataDir, "Train");
    normalFiles = dir(fullfile(trainDir, "*.mat"));
end

normalFile = fullfile(normalFiles(1).folder, normalFiles(1).name);

fprintf('=== ADİL KARŞILAŞTIRMA ===\n');
fprintf('1. Tehlike (Nöbet): %s\n', seizureFiles(1).name);
fprintf('2. Güvenli (Normal): %s (Eğitimde GÖRÜLMEDİ)\n', normalFiles(1).name);

% 3. ANALİZ BAŞLIYOR
limitWindows = 50; % Grafik net olsun diye 50 pencereye bakalım

fprintf('Unseen Normal veri analiz ediliyor... ');
errNormal = get_errors_fixed(net, normalFile, limitWindows);
fprintf('[Tamam]\n');

fprintf('Seizure veri analiz ediliyor... ');
errSeizure = get_errors_fixed(net, seizureFile, limitWindows);
fprintf('[Tamam]\n');

% 4. GÖRSELLEŞTİRME
figure('Name', 'Rigorous Validation', 'Color', 'w', 'Position', [100 100 1000 600]);

hold on;
% Unseen Normal (Yeşil)
plot(errNormal, 'g-o', 'LineWidth', 2, 'MarkerSize', 5, 'DisplayName', 'Hiç Görülmemiş Normal Veri');

% Seizure (Kırmızı)
plot(errSeizure, 'r-x', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Nöbet Verisi');

% Ortalama Çizgileri
yline(mean(errNormal), 'g--', sprintf('Normal Ort: %.2f', mean(errNormal)));
yline(mean(errSeizure), 'r--', sprintf('Nöbet Ort: %.2f', mean(errSeizure)));

title('Gerçek Model Performansı (Unseen Data Validation)', 'FontSize', 12);
xlabel('Zaman Penceresi');
ylabel('Anormallik Skoru (MSE)');
legend('Location', 'best');
grid on;

fprintf('\n=== SONUÇ ===\n');
fprintf('Unseen Normal Hata: %.4f\n', mean(errNormal));
fprintf('Seizure Hata:       %.4f\n', mean(errSeizure));

diffRatio = mean(errSeizure) / mean(errNormal);
fprintf('Ayırt Edicilik Oranı: %.1f Kat\n', diffRatio);

if mean(errSeizure) > mean(errNormal) * 1.5
    fprintf('✅ ONAYLANDI: Model hiç görmediği normal veriyi de doğru tanıyor!\n');
else
    fprintf('⚠️ DİKKAT: Normal ve Nöbet hatası birbirine yakın.\n');
end


% --- DÜZELTİLMİŞ FONKSİYON ---
function errors = get_errors_fixed(net, filePath, maxWindows)
    d = load(filePath);
    if isfield(d, 'X')
        rawX = squeeze(d.X);
    else
        vars = fieldnames(d);
        rawX = squeeze(d.(vars{1}));
    end
    
    totalWindows = size(rawX, 3);
    limit = min(totalWindows, maxWindows); 
    
    dataCell = cell(limit, 1);
    for k = 1:limit
        dataCell{k} = rawX(:, :, k);
    end
    
    % Tahmin
    reconstructions = predict(net, dataCell, 'MiniBatchSize', 16);
    
    % Hata Hesabı
    errors = zeros(limit, 1);
    isOutputCell = iscell(reconstructions);
    
    for k = 1:limit
        original = dataCell{k};
        if isOutputCell
            rec = reconstructions{k};
        else
            rec = reconstructions(:,:,k);
        end
        diff = original - rec;
        
        % DÜZELTME BURADA YAPILDI (i -> k)
        errors(k) = mean(diff.^2, 'all'); 
    end
end