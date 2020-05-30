%%  Chunjing Xiao <ChunjingXiao@gmail.com> 20200530
%
% This is to discreize continuous CSI data into bins  for segmentation
% DeepSeg: Deep Learning based Motion Segmentation Framework for Activity Recognition using WiFi
%
clear
sampleLen = 120;
%sampleCategory = 10;
userNum = 'user1';
%userNum = 'user2';
%userNum = 'user3';
%userNum = 'user4';
%userNum = 'user5';
saveDir = ['DataDiscretizeCsi/',userNum,'_test_data']; %'20191220SegmentTrainNew/user2_data_label';

dirMat = ['DataCsiAmplitudeCut\',userNum];  %'20191211OriginalMatData\user2'
SegmentFiles = dir([dirMat,'/','*.mat']); % 55user1_iw_1.mat
numberFiles = length(SegmentFiles);

for whichFile =1:numberFiles
    %fprintf('seectFile  : %s, matFileName: %s\n', num2str(whichFile), SegmentFiles(whichFile).name)

    
    fprintf('seectFile  : %s, matFileName: %s\n', num2str(whichFile), SegmentFiles(whichFile).name)
    data = load([dirMat,'/',SegmentFiles(whichFile).name]);
    lowpass = data.data_;
    lowpass = lowpass(1:20:end,:,:);%数据集缩小了20倍
    
    lenData = size(lowpass);
    lenData = lenData(1);
    lowpassDiff = diff(lowpass); 
    %lowpassDiff = lowpass;

    
    startSampleNum = sampleLen;
    endAampleNum = lenData - sampleLen;
    sampleNumPerFile = endAampleNum - startSampleNum + 1; %每个文件有10个样本
    data_=zeros(sampleLen,30,3,sampleNumPerFile);
    
    for i= startSampleNum+1:1:endAampleNum  %1:1:10 
        
        data_(:,:,:,i)= lowpassDiff(i:i+sampleLen-1,:,:);
    end
    
    saveName = strrep(SegmentFiles(whichFile).name,'55','');
    %fprintf('size(data_)         : %s\n', num2str(size(data_)))
    save([saveDir,'\',saveName], 'data_')

end
  