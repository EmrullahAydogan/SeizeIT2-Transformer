% check_final_details.m
% SeizeIT2 - Seçilen 3 Hastanın Demografik ve Zamansal Analizi
% Amaç: Cinsiyet dengesi ve Nöbet dağılımını (Clustering) görselleştirmek.

clc; clear; close all;

datasetDir = "/home/developer/Desktop/SeizeIT2/dataset"; 
targets = ["sub-103", "sub-015", "sub-022"];

% 1. DEMOGRAFİK BİLGİ OKUMA (participants.tsv)
% ---------------------------------------------------------
participantsFile = fullfile(datasetDir, 'participants.tsv');
fprintf('=== DEMOGRAFİK KONTROL ===\n');

if exist(participantsFile, 'file')
    opts = detectImportOptions(participantsFile, 'FileType', 'text');
    opts.VariableNamingRule = 'preserve';
    P = readtable(participantsFile, opts);
    
    % Sadece seçilenleri filtrele
    % (participant_id sütununda sub-XXX yazar)
    targetRows = ismember(string(P.participant_id), targets);
    selectedP = P(targetRows, :);
    
    disp(selectedP);
    
    % Hızlı Yorum
    genders = string(selectedP.sex);
    if length(unique(genders)) > 1
        fprintf('>>> YORUM: Cinsiyet çeşitliliği VAR. (Süper)\n');
    else
        fprintf('>>> YORUM: Cinsiyet çeşitliliği YOK. (Sorun değil ama not edilmeli)\n');
    end
else
    fprintf('UYARI: participants.tsv dosyası bulunamadı.\n');
end

% 2. NÖBET DAĞILIMI GÖRSELLEŞTİRME
% ---------------------------------------------------------
fprintf('\n=== NÖBET DAĞILIMI ANALİZİ ===\n');
figure('Name', 'Nöbet Dağılımı', 'Position', [100, 100, 1000, 400]);
hold on;

colors = lines(length(targets));

for i = 1:length(targets)
    subID = targets(i);
    eventFiles = dir(fullfile(datasetDir, subID, 'ses-01', 'eeg', '*events.tsv'));
    
    if ~isempty(eventFiles)
        opts = detectImportOptions(fullfile(eventFiles(1).folder, eventFiles(1).name), 'FileType', 'text');
        opts.VariableNamingRule = 'preserve';
        events = readtable(fullfile(eventFiles(1).folder, eventFiles(1).name), opts);
        
        % Sütun bul
        cols = events.Properties.VariableNames;
        typeCol = cols{contains(lower(cols), 'type')};
        onsetCol = cols{contains(lower(cols), 'onset')};
        
        % Nöbet zamanlarını al (Saniye cinsinden)
        isSz = contains(lower(string(events.(typeCol))), 'sz');
        szTimes = events.(onsetCol)(isSz);
        
        % Grafiğe nokta koy (Y ekseni: Hasta ID, X ekseni: Zaman)
        % Zamanı saate çevirelim
        scatter(szTimes/3600, repmat(i, length(szTimes), 1), 100, colors(i,:), 'filled', 'DisplayName', subID);
    end
end

yticks(1:3);
yticklabels(targets);
xlabel('Zaman (Saat)');
title('Seçilen Hastaların Nöbet Dağılımı (Dağınık olması iyidir)');
grid on;
legend;
hold off;

fprintf('Grafik çizildi. Lütfen nöbetlerin üst üste binip binmediğini kontrol edin.\n');