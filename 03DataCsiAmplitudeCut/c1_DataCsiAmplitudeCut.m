% this will reduce the rows of data, because the original data is too large
% and after reduceing, the performance almost keep the same.
% In other words, this will extract 1 row every 20 rows
% For example, if the the original data has the size if 1000*30*3, this
% will output 50*30*3
clear
userNum = 'user1';
%userNum = 'user2';
%userNum = 'user3';
%userNum = 'user4';
%userNum = 'user5';
saveDir = ['DataCsiAmplitudeCut/',userNum]; %'20191220SegmentTrainNew/user2_data_label';
dirMat = ['DataCsiAmplitude\',userNum];  %'20191211OriginalMatData\user2'
SegmentFiles = dir([dirMat,'/','*.mat']); % 55user1_iw_1.mat
numberFiles = length(SegmentFiles);

for whichFile =1:numberFiles
    %fprintf('seectFile  : %s, matFileName: %s\n', num2str(whichFile), SegmentFiles(whichFile).name)
    
    fprintf('seectFile  : %s, matFileName: %s\n', num2str(whichFile), SegmentFiles(whichFile).name)
    data = load([dirMat,'/',SegmentFiles(whichFile).name]);
    data_ = data.data;
    
    %lowpassDiff = lowpass;
    
    
    data_ = data_(1:20:end,:,:,:);%数据集缩小了20倍

    %saveName = strrep(SegmentFiles(whichFile).name,'55','');
    %fprintf('size(data_)         : %s\n', num2str(size(data_)))
    save([saveDir,'\',SegmentFiles(whichFile).name], 'data_')

    

  end
    