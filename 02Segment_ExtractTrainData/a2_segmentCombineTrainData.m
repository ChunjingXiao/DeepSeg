%% Chunjing Xiao <ChunjingXiao@gmail.com> 20200530
%% DeepSeg: Deep Learning based Motion Segmentation Framework for Activity Recognition using WiFi
%% IEEE Internet of Things Journal 2020
%
% combine 5 files (one data per user) into segmentBaseTrainCsi (1,2,3,4,5) and test (6)
%
    clear   
    
    global segmentBaseCsi;
    global segmentBaseLab;    
    global segmentTrainCsi;
    global segmentTrainLab;    
    global segmentTestCsi;
    global segmentTestLab;
    
    segmentBaseCsi = [];
    segmentBaseLab = [];    
    segmentTrainCsi = [];
    segmentTrainLab = [];    
    segmentTestCsi = [];
    segmentTestLab = [];
    currentDir = 'TrainingDataForSegment'  %currentDir = '20191220SegmentTrainNew'
    dataDir      = [currentDir, '/user1_data_label/']; 
    combineCsiLabel(dataDir);
    dataDir      = [currentDir, '/user2_data_label/']; 
    combineCsiLabel(dataDir);
    dataDir      = [currentDir, '/user3_data_label/']; 
    combineCsiLabel(dataDir);
    dataDir      = [currentDir, '/user4_data_label/']; 
    combineCsiLabel(dataDir);
    dataDir      = [currentDir, '/user5_data_label/']; 
    combineCsiLabel(dataDir);

    %save([currentDir, '/segmentBaseCsi'],'segmentBaseCsi'); 
    %save([currentDir, '/segmentBaseLab'],'segmentBaseLab'); 
    %save([currentDir, '/segmentTrainCsi'],'segmentTrainCsi'); 
    %save([currentDir, '/segmentTrainLab'],'segmentTrainLab'); 
    save([currentDir, '/segmentTestCsi'],'segmentTestCsi'); 
    save([currentDir, '/segmentTestLab'],'segmentTestLab'); 
    
    segmentBaseTrainCsi = cat(4,segmentBaseCsi, segmentTrainCsi);
    segmentBaseTrainLab = [segmentBaseLab;segmentTrainLab];
    save([currentDir, '/segmentBaseTrainCsi'],'segmentBaseTrainCsi'); 
    save([currentDir, '/segmentBaseTrainLab'],'segmentBaseTrainLab'); 
    
    fprintf('size(segmentBaseTrainCsi)         : %s\n', num2str(size(segmentBaseTrainCsi)))
    fprintf('size(segmentBaseTrainLab)         : %s\n', num2str(size(segmentBaseTrainLab)))
    fprintf('size(segmentTestCsi)         : %s\n', num2str(size(segmentTestCsi)))
    fprintf('size(segmentTestLab)         : %s\n', num2str(size(segmentTestLab)))
    
 function combineCsiLabel(dataDir)
    fileList = dir(strcat(dataDir,'*.mat'));
    numberFiles = length(fileList);
    global segmentBaseCsi;
    global segmentBaseLab;    
    global segmentTrainCsi;
    global segmentTrainLab;    
    global segmentTestCsi;
    global segmentTestLab;
    
    for i=1:numberFiles
        fprintf('i    : %s -- fieName: %s\n',  num2str(i),fileList(i).name)        
        %fprintf('size(data_)         : %s\n', num2str(size(data_)))
        if ~isempty(strfind(fileList(i).name,'_1.mat'))  || ~isempty(strfind(fileList(i).name,'_2.mat'))
            load([dataDir,fileList(i).name]);
            segmentBaseCsi = cat(4,segmentBaseCsi, data_);
            load([dataDir,strrep(fileList(i).name, '.mat', '_label.mat')]);
            segmentBaseLab = [segmentBaseLab;label_];
            fprintf('size(segmentBaseCsi)         : %s\n', num2str(size(segmentBaseCsi)))
        end
        
        %if ~isempty(strfind(fileList(i).name,'_3.mat'))  || ~isempty(strfind(fileList(i).name,'_4.mat'))
        if ~isempty(strfind(fileList(i).name,'_3.mat'))  || ~isempty(strfind(fileList(i).name,'_4.mat')) || ...
                ~isempty(strfind(fileList(i).name,'_5.mat'))
            load([dataDir,fileList(i).name]);
            segmentTrainCsi = cat(4,segmentTrainCsi, data_);
            load([dataDir,strrep(fileList(i).name, '.mat', '_label.mat')]);
            segmentTrainLab = [segmentTrainLab;label_];
        end

        %if ~isempty(strfind(fileList(i).name,'_5.mat'))  || ~isempty(strfind(fileList(i).name,'_6.mat'))
        if ~isempty(strfind(fileList(i).name,'_6.mat'))
            load([dataDir,fileList(i).name]);
            segmentTestCsi = cat(4,segmentTestCsi, data_);
            load([dataDir,strrep(fileList(i).name, '.mat', '_label.mat')]);
            segmentTestLab = [segmentTestLab;label_];
        end
    end
 end

