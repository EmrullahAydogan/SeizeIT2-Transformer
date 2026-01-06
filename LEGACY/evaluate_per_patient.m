% evaluate_per_patient.m
% SeizeIT2 - Academic-Grade Per-Patient Evaluation
%
% Amaç: Her hasta için ayrı ayrı:
%   - ROC Curve + AUC
%   - Optimal Threshold (Youden's Index)
%   - Confusion Matrix
%   - Sensitivity, Specificity, Precision, F1-Score
%   - Statistical Significance (Mann-Whitney U Test)
%
% Çıktı:
%   - Per-patient performance table
%   - Grafik: ROC curves (3 hasta + ortalama)
%   - Grafik: Anomaly score distribution
%   - Results.mat: Tüm sonuçlar kaydedilir

clc; clear; close all;

%% === AYARLAR ===
dataDir = "/home/developer/Desktop/SeizeIT2/MatlabProject/ModelData";
testDir = fullfile(dataDir, "Test");
modelPath = fullfile(dataDir, "Trained_Transformer_Final.mat");
outputDir = "/home/developer/Desktop/SeizeIT2/MatlabProject/Results";

if ~isfolder(outputDir), mkdir(outputDir); end

% Hasta bilgileri (Manuel tanım - academic_selection_matrix.csv'den)
patients = struct();
patients(1).id = 'sub-015';
patients(1).seizures = 9;
patients(1).duration = 20.9;
patients(2).id = 'sub-103';
patients(2).seizures = 15;
patients(2).duration = 18.4;
patients(3).id = 'sub-022';
patients(3).seizures = 7;
patients(3).duration = 21.5;

%% === 1. MODEL YÜKLEME ===
fprintf('=== AKADEMİK DEĞERLENDİRME BAŞLIYOR ===\n');
fprintf('Model yükleniyor...\n');
if ~isfile(modelPath)
    error('Model bulunamadı: %s\nÖnce train_transformer_autoencoder.m çalıştırın.', modelPath);
end
loaded = load(modelPath, 'net');
net = loaded.net;
fprintf('Model yüklendi: %s\n\n', modelPath);

%% === 2. HER HASTA İÇİN EVALUATION ===
Results = struct();

for p = 1:length(patients)
    patientID = patients(p).id;
    fprintf('[%d/%d] Değerlendiriliyor: %s\n', p, length(patients), patientID);
    fprintf('   Klinik Bilgi: %d nöbet, %.1f saat kayıt\n', patients(p).seizures, patients(p).duration);

    % --- 2.1. VERİ YÜKLEME (Seizure + Normal) ---
    % Nöbet verisi
    seizureFile = dir(fullfile(testDir, sprintf('%s*_Seizures.mat', patientID)));
    normalFile = dir(fullfile(testDir, sprintf('%s*_NormalPart.mat', patientID)));

    if isempty(seizureFile)
        fprintf('   UYARI: Nöbet verisi bulunamadı, atlanıyor.\n\n');
        continue;
    end

    % Nöbet windows
    szData = load(fullfile(seizureFile(1).folder, seizureFile(1).name));
    szX = squeeze(szData.X); % [16, 1000, N]
    numSeizure = size(szX, 3);

    % Normal windows
    if ~isempty(normalFile)
        normData = load(fullfile(normalFile(1).folder, normalFile(1).name));
        normX = squeeze(normData.X);
        numNormal = size(normX, 3);
    else
        fprintf('   UYARI: Normal test verisi yok, Train''den alınıyor.\n');
        trainNormFile = dir(fullfile(dataDir, 'Train', sprintf('%s*_NormalPart.mat', patientID)));
        if isempty(trainNormFile)
            fprintf('   HATA: Hiç normal veri yok!\n\n');
            continue;
        end
        normData = load(fullfile(trainNormFile(1).folder, trainNormFile(1).name));
        normX = squeeze(normData.X);
        numNormal = min(size(normX, 3), 200); % Train'den max 200 pencere al
        normX = normX(:, :, 1:numNormal);
    end

    fprintf('   Veri: %d Nöbet + %d Normal pencere\n', numSeizure, numNormal);

    % --- 2.2. TAHMIN (Anomaly Score Hesaplama) ---
    fprintf('   Model tahmin yapıyor... ');

    % Seizure anomaly scores
    szCell = cell(numSeizure, 1);
    for i = 1:numSeizure
        szCell{i} = szX(:, :, i);
    end
    szRec = predict(net, szCell, 'MiniBatchSize', 16);
    szScores = compute_mse(szCell, szRec);

    % Normal anomaly scores
    normCell = cell(numNormal, 1);
    for i = 1:numNormal
        normCell{i} = normX(:, :, i);
    end
    normRec = predict(net, normCell, 'MiniBatchSize', 16);
    normScores = compute_mse(normCell, normRec);

    fprintf('[Tamam]\n');

    % --- 2.3. ROC ANALİZİ ---
    % Labels: 1=Seizure, 0=Normal
    trueLabels = [ones(numSeizure, 1); zeros(numNormal, 1)];
    allScores = [szScores; normScores];

    % ROC Curve (MATLAB perfcurve)
    [X_roc, Y_roc, T_roc, AUC] = perfcurve(trueLabels, allScores, 1);

    % Optimal Threshold (Youden's Index: max(Sensitivity + Specificity - 1))
    youden = Y_roc - X_roc; % TPR - FPR
    [~, optIdx] = max(youden);
    optThreshold = T_roc(optIdx);
    optSens = Y_roc(optIdx);
    optSpec = 1 - X_roc(optIdx);

    fprintf('   AUC: %.4f | Optimal Threshold: %.4f\n', AUC, optThreshold);

    % --- 2.4. CONFUSION MATRIX (Optimal Threshold ile) ---
    predictions = allScores >= optThreshold;

    TP = sum(predictions == 1 & trueLabels == 1);
    TN = sum(predictions == 0 & trueLabels == 0);
    FP = sum(predictions == 1 & trueLabels == 0);
    FN = sum(predictions == 0 & trueLabels == 1);

    % Metrikler
    sensitivity = TP / (TP + FN);
    specificity = TN / (TN + FP);
    precision = TP / (TP + FP);
    f1_score = 2 * precision * sensitivity / (precision + sensitivity);
    accuracy = (TP + TN) / (TP + TN + FP + FN);

    fprintf('   Sens: %.3f | Spec: %.3f | F1: %.3f | Acc: %.3f\n', ...
        sensitivity, specificity, f1_score, accuracy);

    % --- 2.5. STATISTICAL TEST (Mann-Whitney U) ---
    [pValue, ~] = ranksum(normScores, szScores);
    fprintf('   Mann-Whitney U Test: p = %.2e', pValue);
    if pValue < 0.001
        fprintf(' ***\n');
    elseif pValue < 0.01
        fprintf(' **\n');
    elseif pValue < 0.05
        fprintf(' *\n');
    else
        fprintf(' (ns)\n');
    end

    % --- 2.6. SONUÇLARI KAYDET ---
    Results(p).PatientID = patientID;
    Results(p).NumSeizure = numSeizure;
    Results(p).NumNormal = numNormal;
    Results(p).AUC = AUC;
    Results(p).OptimalThreshold = optThreshold;
    Results(p).Sensitivity = sensitivity;
    Results(p).Specificity = specificity;
    Results(p).Precision = precision;
    Results(p).F1_Score = f1_score;
    Results(p).Accuracy = accuracy;
    Results(p).PValue = pValue;
    Results(p).TP = TP;
    Results(p).TN = TN;
    Results(p).FP = FP;
    Results(p).FN = FN;
    Results(p).ROC_X = X_roc;
    Results(p).ROC_Y = Y_roc;
    Results(p).SeizureScores = szScores;
    Results(p).NormalScores = normScores;

    fprintf('\n');
end

%% === 3. ÖZET İSTATİSTİKLER ===
fprintf('=== ÖZET PERFORMANS (n=%d Hasta) ===\n', length(Results));

% Tablo oluştur
T = struct2table(Results);
T = T(:, {'PatientID', 'AUC', 'Sensitivity', 'Specificity', 'F1_Score', 'Accuracy', 'PValue'});

fprintf('\nPer-Patient Performance:\n');
disp(T);

% Ortalama ± Std
fprintf('\n--- Genel Performans (Mean ± SD) ---\n');
fprintf('AUC:         %.3f ± %.3f\n', mean([Results.AUC]), std([Results.AUC]));
fprintf('Sensitivity: %.3f ± %.3f\n', mean([Results.Sensitivity]), std([Results.Sensitivity]));
fprintf('Specificity: %.3f ± %.3f\n', mean([Results.Specificity]), std([Results.Specificity]));
fprintf('F1-Score:    %.3f ± %.3f\n', mean([Results.F1_Score]), std([Results.F1_Score]));
fprintf('Accuracy:    %.3f ± %.3f\n', mean([Results.Accuracy]), std([Results.Accuracy]));

%% === 4. GÖRSELLEŞTİRME ===

% --- FIGURE 1: ROC Curves (Per-Patient + Mean) ---
fig1 = figure('Name', 'Per-Patient ROC Curves', 'Color', 'w', 'Position', [100, 100, 900, 700]);
hold on; grid on;

colors = lines(length(Results));
legendEntries = cell(length(Results)+1, 1);

for p = 1:length(Results)
    plot(Results(p).ROC_X, Results(p).ROC_Y, '-', 'Color', colors(p,:), 'LineWidth', 2);
    legendEntries{p} = sprintf('%s (AUC=%.3f)', Results(p).PatientID, Results(p).AUC);
end

% Chance line
plot([0 1], [0 1], 'k--', 'LineWidth', 1.5);
legendEntries{end} = 'Chance';

xlabel('False Positive Rate (1-Specificity)', 'FontSize', 12);
ylabel('True Positive Rate (Sensitivity)', 'FontSize', 12);
title('ROC Curves: Per-Patient Performance', 'FontSize', 14, 'FontWeight', 'bold');
legend(legendEntries, 'Location', 'southeast');
axis square;
xlim([0 1]); ylim([0 1]);

% Kaydet
saveas(fig1, fullfile(outputDir, 'ROC_Curves_PerPatient.png'));
fprintf('\nGrafik kaydedildi: ROC_Curves_PerPatient.png\n');

% --- FIGURE 2: Anomaly Score Distributions ---
fig2 = figure('Name', 'Anomaly Score Distribution', 'Color', 'w', 'Position', [150, 150, 1200, 400]);

for p = 1:length(Results)
    subplot(1, length(Results), p);
    hold on; grid on;

    % Histogram
    histogram(Results(p).NormalScores, 30, 'FaceColor', 'g', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    histogram(Results(p).SeizureScores, 30, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none');

    % Threshold line
    xline(Results(p).OptimalThreshold, 'b--', 'LineWidth', 2);

    title(sprintf('%s (AUC=%.3f)', Results(p).PatientID, Results(p).AUC), 'FontSize', 11);
    xlabel('Anomaly Score (MSE)');
    ylabel('Frequency');
    legend({'Normal', 'Seizure', 'Threshold'}, 'Location', 'best');
end

saveas(fig2, fullfile(outputDir, 'Anomaly_Scores_Distribution.png'));
fprintf('Grafik kaydedildi: Anomaly_Scores_Distribution.png\n');

% --- FIGURE 3: Confusion Matrices ---
fig3 = figure('Name', 'Confusion Matrices', 'Color', 'w', 'Position', [200, 200, 1200, 400]);

for p = 1:length(Results)
    subplot(1, length(Results), p);

    % Confusion matrix
    cm = [Results(p).TN, Results(p).FP; Results(p).FN, Results(p).TP];

    imagesc(cm);
    colormap(flipud(gray));
    colorbar;

    % Anotasyonlar
    textStrings = num2str(cm(:), '%d');
    textStrings = strtrim(cellstr(textStrings));
    [x, y] = meshgrid(1:2);
    text(x(:), y(:), textStrings, 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');

    title(sprintf('%s\nAcc=%.2f%%', Results(p).PatientID, Results(p).Accuracy*100), 'FontSize', 11);
    xlabel('Predicted');
    ylabel('Actual');
    set(gca, 'XTick', 1:2, 'XTickLabel', {'Normal', 'Seizure'});
    set(gca, 'YTick', 1:2, 'YTickLabel', {'Normal', 'Seizure'});
    axis square;
end

saveas(fig3, fullfile(outputDir, 'Confusion_Matrices.png'));
fprintf('Grafik kaydedildi: Confusion_Matrices.png\n');

%% === 5. SONUÇLARI KAYDET ===
save(fullfile(outputDir, 'EvaluationResults.mat'), 'Results', 'T');
writetable(T, fullfile(outputDir, 'PerPatient_Performance.csv'));

fprintf('\n=== DEĞERLENDİRME TAMAMLANDI ===\n');
fprintf('Sonuçlar "%s" klasörüne kaydedildi.\n', outputDir);
fprintf('\nDosyalar:\n');
fprintf('  - EvaluationResults.mat\n');
fprintf('  - PerPatient_Performance.csv\n');
fprintf('  - ROC_Curves_PerPatient.png\n');
fprintf('  - Anomaly_Scores_Distribution.png\n');
fprintf('  - Confusion_Matrices.png\n');

%% === YARDIMCI FONKSİYON ===
function scores = compute_mse(originals, reconstructions)
    % Anomaly score = Mean Squared Error (MSE)
    numSamples = length(originals);
    scores = zeros(numSamples, 1);
    isOutputCell = iscell(reconstructions);

    for i = 1:numSamples
        orig = originals{i};

        if isOutputCell
            rec = reconstructions{i};
        else
            rec = reconstructions(:, :, i);
        end

        diff = orig - rec;
        scores(i) = mean(diff.^2, 'all');
    end
end
