%% Yongsen Ma <yma@cs.wm.edu>
% Computer Science Department, The College of William & Mary
%
% This is an example for the following paper
% Yongsen Ma, Gang Zhou, Shuangquan Wang, Hongyang Zhao, and Woosub Jung. 2018.
% SignFi: Sign Language Recognition Using WiFi.
% Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 2, 1, Article 23 (March 2018), 21 pages.
% DOI: https://doi.org/10.1145/3191755

% 12CnnClassifyActivity can outperform this code.

    clear
    dataDir = 'ExtractedActivitySample'; %dataDir = '20200115ActionMatPerFileSmall'
    csiData1 = load([dataDir,'/actionBaseTrainCsi.mat']);  % train: user1_data; test: user2_data -- 16%; 
    csiLabel1 = load([dataDir,'/actionBaseTrainLab.mat']);% train: user1_data; test: user3_data -- 21%; 
    csiName =fieldnames(csiData1);                    % train: user1_data; test: user4_data -- 18%; 
    labelName =fieldnames(csiLabel1);                 % train: user1_data; test: user5_data -- 15%; 
    dataBigSmallTrainCut = csiData1.(csiName{1});
    labelBigSmallTrain = csiLabel1.(labelName{1}); 
                    
    csiData1 = load([dataDir,'/actionTestCsi.mat']);  % train: user2_data; test: user1_data -- 20%; 
    csiLabel1 = load([dataDir,'/actionTestLab.mat']);% train: user2_data; test: user3_data -- 13%; 
    csiName =fieldnames(csiData1);                    % train: user2_data; test: user4_data -- 15%; 
    labelName =fieldnames(csiLabel1);                 % train: user2_data; test: user5_data -- 15%; 
    dataBigSmallTestCut = csiData1.(csiName{1});
    labelBigSmallTest = csiLabel1.(labelName{1});
    if(0)
        dataBigSmallTrainCut = dataBigSmallTrainCut(:,:,:,151:end);
        labelBigSmallTrain = labelBigSmallTrain(151:end);
        dataBigSmallTestCut = dataBigSmallTestCut(:,:,:,151:end);
        labelBigSmallTest = labelBigSmallTest(151:end);
    end
    fprintf('size(dataBigSmallTrainCut)        : %s\n', num2str(size(dataBigSmallTrainCut)))
    fprintf('size(labelBigSmallTrain)          : %s\n', num2str(size(labelBigSmallTrain)))
    fprintf('size(dataBigSmallTestCut)         : %s\n', num2str(size(dataBigSmallTestCut)))    
    fprintf('size(labelBigSmallTest)           : %s\n', num2str(size(labelBigSmallTest)))

    
    Nw = 10;
    n_epoch=60; %n_epoch=150;
    %[net_info,perf]=signfi_cnn_train_test(dataBigSmallTrain,dataBigSmallTest,labelBigSmallTrain,labelBigSmallTest,Nw,n_epoch)
    [net_info,perf]=signfi_cnn_train_test(dataBigSmallTrainCut,dataBigSmallTestCut,labelBigSmallTrain,labelBigSmallTest,Nw,n_epoch)
    
