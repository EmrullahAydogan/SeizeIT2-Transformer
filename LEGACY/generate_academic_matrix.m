% generate_academic_matrix.m
% SeizeIT2 - Akademik Proje İçin Detaylı Hasta Seçim Matrisi
% Amaç: 125 Hastayı 7 farklı kriterde puanlayıp CSV olarak kaydetmek.

clc; clear; close all;

% --- AYARLAR ---
datasetDir = "/home/developer/Desktop/SeizeIT2/dataset"; 
outputFile = "academic_selection_matrix.csv";
% ---------------

fprintf('AKADEMİK ANALİZ BAŞLATILIYOR (125 HASTA)...\n');
fprintf('Bu işlem tüm meta-verileri okuyacağı için birkaç dakika sürebilir.\n\n');

% Sonuçları Tutacak Değişkenler
subList = strings(0);
durations = [];
szCounts = [];
szDensity = []; % Nöbet / Saat
hasMOV = logical([]);
vigilanceScore = strings(0); % "Sleep Only", "Wake Only", "Mixed", "Unknown"
lateralization = strings(0); % "Left", "Right", "Bilateral", "Mixed"
dominantType = strings(0);   % En sık görülen nöbet tipi

hWait = waitbar(0, 'Analiz ediliyor...');

% Tüm hastaları bul
folders = dir(fullfile(datasetDir, 'sub-*'));
folders = folders([folders.isdir]);

for i = 1:length(folders)
    subID = folders(i).name;
    sesPath = fullfile(datasetDir, subID, 'ses-01');
    waitbar(i/length(folders), hWait, sprintf('%s (%d/%d)', subID, i, length(folders)));
    
    % -- 1. TEMEL KONTROLLER --
    % Dosya varlıkları
    eegFiles = dir(fullfile(sesPath, 'eeg', '*.edf'));
    movFiles = dir(fullfile(sesPath, 'mov', '*.edf'));
    
    has_mov = ~isempty(movFiles);
    
    % -- 2. SÜRE HESAPLAMA --
    totalDur = 0;
    if ~isempty(eegFiles)
        try
            % Genelde run-01 ana dosyadır, toplam süreyi kabaca tahmin edelim
            info = edfinfo(fullfile(eegFiles(1).folder, eegFiles(1).name));
            totalDur = seconds(info.NumDataRecords * info.DataRecordDuration) / 3600; % Saat
        catch
            totalDur = NaN;
        end
    end
    
    % -- 3. EVENT ANALİZİ (TSV) --
    eventFiles = dir(fullfile(sesPath, 'eeg', '*events.tsv'));
    sz_count = 0;
    vig_status = "No Seizure";
    lat_status = "No Seizure";
    dom_type = "None";
    
    if ~isempty(eventFiles)
        try
            % Tüm run dosyalarını birleştir
            allEvents = table();
            for k = 1:length(eventFiles)
                opts = detectImportOptions(fullfile(eventFiles(k).folder, eventFiles(k).name), 'FileType', 'text');
                opts.VariableNamingRule = 'preserve';
                T = readtable(fullfile(eventFiles(k).folder, eventFiles(k).name), opts);
                allEvents = [allEvents; T]; %#ok<AGROW>
            end
            
            % Sütun İsimlerini Bul
            cols = allEvents.Properties.VariableNames;
            typeCol = cols{contains(lower(cols), 'type')}; 
            
            % Nöbetleri Filtrele
            isSz = contains(lower(string(allEvents.(typeCol))), 'sz');
            szRows = allEvents(isSz, :);
            sz_count = height(szRows);
            
            if sz_count > 0
                % A. VIGILANCE (Uyanıklık) Analizi
                vigColIdx = find(contains(lower(cols), 'vigilance'));
                if ~isempty(vigColIdx)
                    vigs = lower(string(szRows.(cols{vigColIdx(1)})));
                    hasWake = any(contains(vigs, 'awake'));
                    hasSleep = any(contains(vigs, 'sleep'));
                    
                    if hasWake && hasSleep, vig_status = "MIXED (Best)";
                    elseif hasWake, vig_status = "Wake Only";
                    elseif hasSleep, vig_status = "Sleep Only";
                    else, vig_status = "Unknown";
                    end
                else
                    vig_status = "Not Recorded";
                end
                
                % B. LATERALIZATION (Lob) Analizi
                % Genelde 'trial_type' içinde veya ayrı sütunda olur. 
                % SeizeIT2'de tip isminde geçer: "sz_foc_ia_left..." gibi
                types = lower(string(szRows.(typeCol)));
                hasLeft = any(contains(types, 'left'));
                hasRight = any(contains(types, 'right'));
                hasBi = any(contains(types, 'bi') | contains(types, 'bilateral'));
                
                if hasLeft && hasRight, lat_status = "Mixed L/R";
                elseif hasLeft, lat_status = "Left";
                elseif hasRight, lat_status = "Right";
                elseif hasBi, lat_status = "Bilateral";
                else, lat_status = "Unspecified";
                end
                
                % C. Nöbet Tipi (Dominant)
                dom_type = mode(categorical(types));
            end
            
        catch
            dom_type = "Error Reading";
        end
    end
    
    % Verileri Listeye Ekle
    subList(end+1) = subID; %#ok<AGROW>
    durations(end+1) = totalDur; %#ok<AGROW>
    szCounts(end+1) = sz_count; %#ok<AGROW>
    if totalDur > 0, szDensity(end+1) = sz_count / totalDur; else, szDensity(end+1) = 0; end %#ok<AGROW>
    hasMOV(end+1) = has_mov; %#ok<AGROW>
    vigilanceScore(end+1) = vig_status; %#ok<AGROW>
    lateralization(end+1) = lat_status; %#ok<AGROW>
    dominantType(end+1) = string(dom_type); %#ok<AGROW>
end
close(hWait);

% Tabloyu Oluştur
Results = table(subList', durations', szCounts', szDensity', hasMOV', vigilanceScore', lateralization', dominantType', ...
    'VariableNames', {'SubjectID', 'Duration_Hours', 'SeizureCount', 'Seizures_Per_Hour', 'HasMOV', 'Vigilance', 'Lateralization', 'DominantType'});

% CSV Kaydet
writetable(Results, outputFile);
fprintf('Analiz Tamamlandı. "%s" dosyası kaydedildi.\n', outputFile);

% --- AKADEMİK FİLTRELEME (ÖNERİ MOTORU) ---
% Kriterler:
% 1. MOV var
% 2. Süre > 18 Saat (Sirkadiyen ritim için)
% 3. Nöbet Sayısı > 5
% 4. Vigilance = "MIXED" (Hem uyku hem uyanık nöbeti olanlar - EN DEĞERLİSİ)

fprintf('\n=== AKADEMİK SEÇİM ÖNERİLERİ ===\n');
Candidates = Results(Results.HasMOV & Results.Duration_Hours > 18 & Results.SeizureCount >= 5, :);

% En değerlileri (Mixed Vigilance) öne al
MixedVig = Candidates(Candidates.Vigilance == "MIXED (Best)", :);

if ~isempty(MixedVig)
    fprintf('>>> ALTIN STANDART ADAYLAR (Hem Uyku Hem Uyanık Nöbeti Olanlar):\n');
    disp(MixedVig(:, {'SubjectID', 'Duration_Hours', 'SeizureCount', 'Vigilance', 'Lateralization'}));
else
    fprintf('Mixed Vigilance adayı bulunamadı. Diğer uzun süreli kayıtlara bakılıyor:\n');
    disp(Candidates(:, {'SubjectID', 'Duration_Hours', 'SeizureCount', 'Vigilance', 'Lateralization'}));
end

fprintf('\nÖneri: Tezinde/Makalende kullanmak için farklı Lateralizasyonlara sahip (Sağ/Sol) hastaları seçmelisin.\n');