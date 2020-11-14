%% Chunjing Xiao <ChunjingXiao@gmail.com> 20200530
%% DeepSeg: Deep Learning based Motion Segmentation Framework for Activity Recognition using WiFi
%% IEEE Internet of Things Journal 2020

% This is just for test. Our paper does not use this code.

% This is used to test performance of training data for state inference models.
% This will call signfi_cnn_train_test.m


    clear
    
    csiData1 = load('TrainingDataForSegment/segmentBaseTrainCsi.mat'); 
    csiLabel1 = load('TrainingDataForSegment/segmentBaseTrainLab.mat');     
    %csiData1 = load('TrainingDataForSegment/segmentTrainCsi.mat'); 
    %csiLabel1 = load('TrainingDataForSegment/segmentTrainLab.mat');  
    csiName =fieldnames(csiData1); 
    labelName =fieldnames(csiLabel1);    
    dataBigSmallTrainCut = csiData1.(csiName{1});
    labelBigSmallTrain = csiLabel1.(labelName{1}); 
                    
    csiData1 = load('TrainingDataForSegment/segmentTestCsi.mat');  
    csiLabel1 = load('TrainingDataForSegment/segmentTestLab.mat');
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

    
    Nw = 4;
    n_epoch=20;
    %[net_info,perf]=signfi_cnn_train_test(dataBigSmallTrain,dataBigSmallTest,labelBigSmallTrain,labelBigSmallTest,Nw,n_epoch)
    [net_info,perf]=signfi_cnn_train_test(dataBigSmallTrainCut,dataBigSmallTestCut,labelBigSmallTrain,labelBigSmallTest,Nw,n_epoch)
    
