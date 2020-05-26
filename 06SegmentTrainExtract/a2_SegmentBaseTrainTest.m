%%  Chunjing Xiao <ChunjingXiao@gmail.com>
%
%
% divide 5 files (one data per user) into base (1,2) train (3,4) and test (5,6)
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
    currentDir = '20200115SegmentTrainNew'  %currentDir = '20191220SegmentTrainNew'
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
    fprintf('size(segmentBaseCsi)         : %s\n', num2str(size(segmentBaseCsi)))
    fprintf('size(segmentBaseLab)         : %s\n', num2str(size(segmentBaseLab)))
    fprintf('size(segmentTrainCsi)         : %s\n', num2str(size(segmentTrainCsi)))
    fprintf('size(segmentTrainLab)         : %s\n', num2str(size(segmentTrainLab)))
    fprintf('size(segmentTestCsi)         : %s\n', num2str(size(segmentTestCsi)))
    fprintf('size(segmentTestLab)         : %s\n', num2str(size(segmentTestLab)))
    save([currentDir, '/segmentBaseCsi'],'segmentBaseCsi'); 
    save([currentDir, '/segmentBaseLab'],'segmentBaseLab'); 
    save([currentDir, '/segmentTrainCsi'],'segmentTrainCsi'); 
    save([currentDir, '/segmentTrainLab'],'segmentTrainLab'); 
    save([currentDir, '/segmentTestCsi'],'segmentTestCsi'); 
    save([currentDir, '/segmentTestLab'],'segmentTestLab'); 
    
    segmentBaseTrainCsi = cat(4,segmentBaseCsi, segmentTrainCsi);
    segmentBaseTrainLab = [segmentBaseLab;segmentTrainLab];
    save([currentDir, '/segmentBaseTrainCsi'],'segmentBaseTrainCsi'); 
    save([currentDir, '/segmentBaseTrainLab'],'segmentBaseTrainLab'); 
    
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

