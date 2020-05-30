%%  Chunjing Xiao <ChunjingXiao@gmail.com>
%
%% combine 5 files (one data per user) into baseTrain ( 1,2, 3,4,5) and test (6 )
%


    clear
    
    
    global actionBaseCsi;
    global actionBaseLab;    
    global actionTrainCsi;
    global actionTrainLab;    
    global actionTestCsi;
    global actionTestLab;
    
    actionBaseCsi = [];
    actionBaseLab = [];    
    actionTrainCsi = [];
    actionTrainLab = [];    
    actionTestCsi = [];
    actionTestLab = [];
    
    currentDir = 'ExtractedActivitySample\'
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

    %save([currentDir, '/actionBaseCsi'],'actionBaseCsi'); 
    %save([currentDir, '/actionBaseLab'],'actionBaseLab'); 
    %save([currentDir, '/actionTrainCsi'],'actionTrainCsi'); 
    %save([currentDir, '/actionTrainLab'],'actionTrainLab'); 
    save([currentDir, '/actionTestCsi'],'actionTestCsi'); 
    save([currentDir, '/actionTestLab'],'actionTestLab'); 
    
    actionBaseTrainCsi = cat(4,actionBaseCsi, actionTrainCsi);
    actionBaseTrainLab = [actionBaseLab;actionTrainLab];
    save([currentDir, '/actionBaseTrainCsi'],'actionBaseTrainCsi'); 
    save([currentDir, '/actionBaseTrainLab'],'actionBaseTrainLab'); 
    %fprintf('size(actionBaseCsi)         : %s\n', num2str(size(actionBaseCsi)))
    %fprintf('size(actionBaseLab)         : %s\n', num2str(size(actionBaseLab)))
    fprintf('size(actionBaseTrainCsi)    : %s\n', num2str(size(actionBaseTrainCsi)))
    fprintf('size(actionBaseTrainLab)    : %s\n', num2str(size(actionBaseTrainLab)))
    fprintf('size(actionTestCsi)         : %s\n', num2str(size(actionTestCsi)))
    fprintf('size(actionTestLab)         : %s\n', num2str(size(actionTestLab)))
    
 function combineCsiLabel(dataDir)
    fileList = dir(strcat(dataDir,'*.mat'));
    numberFiles = length(fileList);
    global actionBaseCsi;
    global actionBaseLab;    
    global actionTrainCsi;
    global actionTrainLab;    
    global actionTestCsi;
    global actionTestLab;
    
    for i=1:numberFiles
        fprintf('i    : %s -- fieName: %s\n',  num2str(i),fileList(i).name)
        
        %fprintf('size(data_)         : %s\n', num2str(size(data_)))
        if ~isempty(strfind(fileList(i).name,'_1.mat'))  || ~isempty(strfind(fileList(i).name,'_2.mat'))
            load([dataDir,fileList(i).name]);
            actionBaseCsi = cat(4,actionBaseCsi, data_);
            load([dataDir,strrep(fileList(i).name, '.mat', '_label.mat')]);
            actionBaseLab = [actionBaseLab;label_];
            fprintf('size(actionBaseCsi)         : %s\n', num2str(size(actionBaseCsi)))
        end
        
        %if ~isempty(strfind(fileList(i).name,'_3.mat'))  || ~isempty(strfind(fileList(i).name,'_4.mat'))
        if ~isempty(strfind(fileList(i).name,'_3.mat')) || ~isempty(strfind(fileList(i).name,'_4.mat')) || ...
                ~isempty(strfind(fileList(i).name,'_5.mat'))
            
            load([dataDir,fileList(i).name]);
            actionTrainCsi = cat(4,actionTrainCsi, data_);
            load([dataDir,strrep(fileList(i).name, '.mat', '_label.mat')]);
            actionTrainLab = [actionTrainLab;label_];  
        end

        % if ~isempty(strfind(fileList(i).name,'_5.mat'))  || ~isempty(strfind(fileList(i).name,'_6.mat'))
         if ~isempty(strfind(fileList(i).name,'_6.mat'))
            load([dataDir,fileList(i).name]);
            actionTestCsi = cat(4,actionTestCsi, data_);
            load([dataDir,strrep(fileList(i).name, '.mat', '_label.mat')]);
            actionTestLab = [actionTestLab;label_];
        end
    end
 end
    
