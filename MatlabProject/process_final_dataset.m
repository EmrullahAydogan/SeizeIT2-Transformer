% process_final_dataset_v7.m
% SeizeIT2 - Pilot Veri Seti Hazırlama (V7 - FINAL STABLE)
% Düzeltmeler:
% 1. DimensionNames hatası giderildi (Explicit naming).
% 2. Büyük dosya kaydı için '-v7.3' bayrağı eklendi.
% 3. RAM ve Disk tasarrufu için veriler 'single' (float32) formatına çevrildi.

clc; clear; close all;

% --- AYARLAR ---
datasetDir = "/home/developer/Desktop/SeizeIT2/dataset"; 
outputDir = "/home/developer/Desktop/SeizeIT2/MatlabProject/ProcessedData";
targetSubjects = ["sub-103", "sub-015", "sub-022"]; 
targetFs = 250; 

if ~isfolder(outputDir), mkdir(outputDir); end

fprintf('=== VERİ ÖN İŞLEME BAŞLIYOR (V7 - Large File Support) ===\n');

for i = 1:length(targetSubjects)
    subID = targetSubjects(i);
    fprintf('\n[%d/%d] İşleniyor: %s\n', i, length(targetSubjects), subID);
    
    basePath = fullfile(datasetDir, subID, "ses-01");
    timerStart = tic;
    
    try
        % 1. Modaliteleri Oku ve Temizle
        tablesToSync = {};
        
        tablesToSync{end+1} = processModalityV7(fullfile(basePath, 'eeg'), 'EEG', targetFs);
        tablesToSync{end+1} = processModalityV7(fullfile(basePath, 'ecg'), 'ECG', targetFs);
        tablesToSync{end+1} = processModalityV7(fullfile(basePath, 'emg'), 'EMG', targetFs);
        tablesToSync{end+1} = processModalityV7(fullfile(basePath, 'mov'), 'MOV', targetFs);
        
        % Boş tabloları çıkar
        tablesToSync = tablesToSync(~cellfun('isempty', tablesToSync));
        
        if isempty(tablesToSync)
            error("Hiçbir sinyal verisi okunamadı.");
        end
        
        % 2. Birleştir (Synchronize)
        fprintf("   > Sinyaller birleştiriliyor... ");
        fullData = synchronize(tablesToSync{:}, 'union', 'linear');
        fullData = fillmissing(fullData, 'linear');
        fprintf("[Tamam]\n");
        
        % 3. Nöbet Etiketlerini Ekle (Labels)
        labels = zeros(height(fullData), 1, 'int8'); % Bellek için int8 yaptık
        eventFiles = dir(fullfile(basePath, 'eeg', '*events.tsv'));
        
        if ~isempty(eventFiles)
            opts = detectImportOptions(fullfile(eventFiles(1).folder, eventFiles(1).name), 'FileType', 'text');
            opts.VariableNamingRule = 'preserve'; 
            events = readtable(fullfile(eventFiles(1).folder, eventFiles(1).name), opts);
            
            cols = events.Properties.VariableNames;
            typeCol = cols{contains(lower(cols), 'type')};
            onsetCol = cols{contains(lower(cols), 'onset')};
            durCol = cols{contains(lower(cols), 'duration')};
            
            if ~isempty(typeCol)
                isSeizure = contains(lower(string(events.(typeCol))), 'sz', 'IgnoreCase', true);
                szEvents = events(isSeizure, :);
                
                if height(szEvents) > 0
                     fprintf("   > %d nöbet etiketleniyor... ", height(szEvents));
                     
                     % Zaman vektörünü al (DimensionName ne olursa olsun .Time ile erişiriz artık)
                     timeVec = seconds(fullData.Time - fullData.Time(1));
                     
                     for k = 1:height(szEvents)
                        startT = szEvents.(onsetCol)(k);
                        endT = startT + szEvents.(durCol)(k);
                        idx = (timeVec >= startT) & (timeVec <= endT);
                        labels(idx) = 1;
                     end
                     fprintf("[Tamam]\n");
                else
                     fprintf("   > Nöbet bulunamadı (Temiz).\n");
                end
            end
        end
        
        % 4. Kaydet (Optimize Edilmiş)
        savePath = fullfile(outputDir, subID + "_processed.mat");
        fprintf("   > Kaydediliyor (Bu işlem boyut nedeniyle 1-2 dk sürebilir)... ");
        
        % -v7.3 bayrağı 2GB+ dosyalar için ZORUNLUDUR
        save(savePath, 'fullData', 'labels', 'targetFs', '-v7.3');
        
        elapsed = toc(timerStart);
        fileInfo = dir(savePath);
        fprintf("[Bitti]\n   > BAŞARILI: %s (%.2f MB)\n", subID, fileInfo.bytes/1024/1024);
        
    catch ME
        fprintf("\n   !!! HATA: %s\n", ME.message);
        fprintf("   Yer: %s (Satır %d)\n", ME.stack(1).name, ME.stack(1).line);
    end
end
fprintf('\n=== İŞLEM BİTTİ ===\n');


% ---------------------------------------------------------
% YARDIMCI FONKSİYON (V7 - Single Precision & Dim Fix)
% ---------------------------------------------------------
function resampledTT = processModalityV7(folderPath, prefix, targetFs)
    resampledTT = timetable();
    
    files = dir(fullfile(folderPath, "*.edf"));
    if isempty(files), return; end
    
    filename = fullfile(files(1).folder, files(1).name);
    
    % 1. OKUMA
    try
        tt_raw = edfread(filename);
    catch
        return;
    end
    
    % 2. SANITIZATION
    % Zaman boyutunun ismini zorla 'Time' yap
    tt_raw.Properties.DimensionNames{1} = 'Time'; 
    
    % Hatalı sütunları temizle
    badCols = {};
    for v = 1:width(tt_raw)
        oldName = tt_raw.Properties.VariableNames{v};
        if contains(oldName, 'Annot') || contains(oldName, 'Record')
             if ~isnumeric(tt_raw.(oldName)) && ~iscell(tt_raw.(oldName))
                 badCols{end+1} = oldName; %#ok<AGROW>
             end
        end
    end
    if ~isempty(badCols)
        tt_raw(:, badCols) = [];
    end
    
    % 3. İŞLEME
    processedCols = {};
    varNames = tt_raw.Properties.VariableNames;
    
    for i = 1:length(varNames)
        colName = varNames{i};
        colData = tt_raw{:, i};
        
        % Cell Unpacking
        if iscell(colData)
            if isempty(colData) || ~isnumeric(colData{1})
                continue; 
            end
            try
                flatData = vertcat(colData{:});
            catch
                continue; 
            end
            
            % SINGLE'a çevir (Boyut Tasarrufu)
            flatData = single(flatData); 
            
            if height(tt_raw) > 1
                recDur = tt_raw.Time(2) - tt_raw.Time(1);
            else
                recDur = seconds(1); 
            end
            fs_est = length(colData{1}) / seconds(recDur);
            totalSamples = length(flatData);
            
            newTime = tt_raw.Properties.StartTime + (0:totalSamples-1)' * seconds(1/fs_est);
            
            singleTT = timetable(newTime, flatData, 'VariableNames', {'TempVar'});
            
        elseif isnumeric(colData)
            % SINGLE'a çevir
            colData = single(colData);
            singleTT = timetable(tt_raw.Time, colData, 'VariableNames', {'TempVar'});
        else
            continue; 
        end
        
        % KRİTİK DÜZELTME: Oluşturulan her tablonun zaman ismi 'Time' olsun
        singleTT.Properties.DimensionNames{1} = 'Time';
        
        % Resample
        try
            singleTT_res = retime(singleTT, 'regular', 'linear', 'TimeStep', seconds(1/targetFs));
            
            cleanName = regexprep(colName, '[^a-zA-Z0-9]', '');
            singleTT_res.Properties.VariableNames{1} = [prefix '_' cleanName];
            
            processedCols{end+1} = singleTT_res; %#ok<AGROW>
        catch
            continue;
        end
    end
    
    if ~isempty(processedCols)
        resampledTT = synchronize(processedCols{:}, 'union', 'linear');
        % Birleşim sonrası da isim Time kalsın
        resampledTT.Properties.DimensionNames{1} = 'Time';
    end
end