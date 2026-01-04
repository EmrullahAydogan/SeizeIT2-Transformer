% create_model_data_v2.m
% SeizeIT2 - AI Model Verisi Hazırlama (RAM Dostu - Çökme Önleyici)
% Yöntem: Her hastayı işleyip anında diske kaydeder. Birleştirme yapmaz.

clc; clear; close all;

% --- AYARLAR ---
inputDir = "/home/developer/Desktop/SeizeIT2/MatlabProject/ProcessedData";
baseOutputDir = "/home/developer/Desktop/SeizeIT2/MatlabProject/ModelData";

% Eğitim ve Test için ayrı klasörler açalım (Datastore için gerekli)
trainDir = fullfile(baseOutputDir, "Train");
testDir = fullfile(baseOutputDir, "Test");

if ~isfolder(trainDir), mkdir(trainDir); end
if ~isfolder(testDir), mkdir(testDir); end

% Parametreler
fs = 250;
winSizeSec = 4; % 4 saniye
strideSec = 2;  % 2 saniye kaydırma
winSize = fs * winSizeSec;
stride = fs * strideSec;

files = dir(fullfile(inputDir, "*_processed.mat"));
fprintf('=== RAM DOSTU VERİ HAZIRLAMA (%d Dosya) ===\n', length(files));

totalTrainWindows = 0;
totalTestWindows = 0;

for i = 1:length(files)
    fileName = files(i).name;
    filePath = fullfile(files(i).folder, fileName);
    [~, nameNoExt, ~] = fileparts(fileName);
    
    fprintf('\n[%d/%d] İşleniyor: %s\n', i, length(files), fileName);
    
    % 1. Yükle
    loaded = load(filePath); 
    rawSignals = single(table2array(loaded.fullData)); % Bellek için single
    labels = loaded.labels;
    
    % 2. Normalize Et (Z-Score)
    mu = mean(rawSignals, 1);
    sigma = std(rawSignals, 0, 1);
    sigma(sigma == 0) = 1;
    normSignals = (rawSignals - mu) ./ sigma;
    
    % 3. Pencereleme (Loop ile değil, vectorization ile hızlı yapalım)
    numSamples = size(normSignals, 1);
    numChannels = size(normSignals, 2);
    numWindows = floor((numSamples - winSize) / stride) + 1;
    
    if numWindows < 1
        fprintf('   UYARI: Dosya çok kısa, atlanıyor.\n');
        continue;
    end
    
    % İndeks matrisi oluştur (Hızlı Pencereleme)
    % indices: [WindowSize x NumWindows]
    startIdx = (0:numWindows-1) * stride + 1;
    indices = startIdx + (0:winSize-1)'; 
    
    % 3 Boyutlu Küpü Oluştur: [WindowSize, NumWindows, Channels]
    % Not: MATLAB'da reshape ile kanal boyutunu yönetmek biraz triklidir.
    % Basit döngü RAM şişirmemesi için en güvenlisidir bu boyutta.
    
    X_Batch = zeros(numChannels, winSize, 1, numWindows, 'single');
    Y_Batch = zeros(numWindows, 1, 'single');
    
    fprintf('   > Pencereleniyor (%d adet)... ', numWindows);
    
    % RAM Şişmesini önlemek için 1000'erli paketler halinde de yapabiliriz ama
    % Tek hasta 32GB RAM'e sığar. Sorun hepsini birleştirmekti.
    for w = 1:numWindows
        s = startIdx(w);
        e = s + winSize - 1;
        
        % Segmenti al
        segment = normSignals(s:e, :)'; % Transpose: [Channels x Time]
        
        % Deep Learning Toolbox Formatı: [Channels, Time, 1, Batch]
        X_Batch(:, :, 1, w) = segment;
        
        % Etiket
        if sum(labels(s:e)) > (winSize * 0.2)
            Y_Batch(w) = 1; % Nöbet
        else
            Y_Batch(w) = 0; % Normal
        end
    end
    fprintf('[Tamam]\n');
    
    % 4. Kaydetme (Train/Test Ayrımı)
    if contains(fileName, 'sub-022')
        % Temiz hasta -> Hepsi Train
        saveName = fullfile(trainDir, nameNoExt + "_Full.mat");
        X = X_Batch;
        Y = Y_Batch;
        save(saveName, 'X', 'Y', '-v7.3');
        totalTrainWindows = totalTrainWindows + numWindows;
        fprintf('   > Kaydedildi (TRAIN): %s\n', saveName);
        
    else
        % Nöbetli Hasta -> Nöbetler Test, Normaller Train/Test
        isSeizure = (Y_Batch == 1);
        
        % A) NÖBETLER (Test Klasörüne)
        if sum(isSeizure) > 0
            X = X_Batch(:, :, 1, isSeizure);
            Y = Y_Batch(isSeizure);
            saveName = fullfile(testDir, nameNoExt + "_Seizures.mat");
            save(saveName, 'X', 'Y', '-v7.3');
            totalTestWindows = totalTestWindows + sum(isSeizure);
            fprintf('   > Kaydedildi (TEST - Nöbetler): %d adet\n', sum(isSeizure));
        end
        
        % B) NORMALLER (%80 Train, %20 Test)
        normIdx = find(~isSeizure);
        if ~isempty(normIdx)
            cv = cvpartition(length(normIdx), 'HoldOut', 0.2);
            
            % Train Parçası
            trainIdx = normIdx(training(cv));
            X = X_Batch(:, :, 1, trainIdx);
            Y = Y_Batch(trainIdx);
            saveName = fullfile(trainDir, nameNoExt + "_NormalPart.mat");
            save(saveName, 'X', 'Y', '-v7.3');
            totalTrainWindows = totalTrainWindows + length(trainIdx);
            
            % Test Parçası
            testIdx = normIdx(test(cv));
            X = X_Batch(:, :, 1, testIdx);
            Y = Y_Batch(testIdx);
            saveName = fullfile(testDir, nameNoExt + "_NormalPart.mat");
            save(saveName, 'X', 'Y', '-v7.3');
            totalTestWindows = totalTestWindows + length(testIdx);
            
            fprintf('   > Kaydedildi (TRAIN/TEST - Normaller Dağıtıldı)\n');
        end
    end
    
    % RAM TEMİZLİĞİ (En Önemli Kısım)
    clear X_Batch Y_Batch normSignals rawSignals loaded X Y indices
end

fprintf('\n=== İŞLEM TAMAMLANDI ===\n');
fprintf('Toplam Eğitim Penceresi: %d\n', totalTrainWindows);
fprintf('Toplam Test Penceresi:   %d\n', totalTestWindows);
fprintf('Veriler "%s" klasöründe hazır.\n', baseOutputDir);