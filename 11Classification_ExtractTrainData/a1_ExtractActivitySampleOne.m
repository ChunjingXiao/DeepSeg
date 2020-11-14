
clear
sampleLen = 200; %sampleLen = 4000;
%sampleCategory = 10;
userNum = 'user1';
saveDir = ['ExtractedActivitySample/',userNum,'_data_label']; %'20191220SegmentTrainNew/user2_data_label';
%whetherPlot = 0;
%selectFile = 1;  % for user1, diff(lowpass) of 14 15 17 20 is not obvious,
startSampleNum = 1;
endAampleNum = 10;
sampleNumPerFile = endAampleNum - startSampleNum + 1; %每个文件有10个样本
%load('20191212ManualSegment/user1ManualSegment.mat'); cvsSegment = QQ;% use Duan YuCheng
%cvsSegment = csvread(['Label_CsiAmplitudeCut/',userNum,'ManualSegment.csv']);% 'ManualSegment/user1ManualSegment.csv'
fid = fopen(['Label_CsiAmplitudeCut/',userNum,'ManualSegment.csv']);
dcells = textscan(fid, '%f,%f,%f,%f,%s', 'HeaderLines', 1, 'EndOfLine', '\r\n');
fclose(fid); 
dcellneeds = dcells(1:4);
cvsSegment = cell2mat(dcellneeds);

dirMat = ['Data_CsiAmplitudeCut\',userNum];  %'20191211OriginalMatData\user2'
SegmentFiles = dir([dirMat,'/','*.mat']); % 55user1_iw_1.mat
numberFiles = length(SegmentFiles);

for whichFile =1:numberFiles
    %fprintf('seectFile  : %s, matFileName: %s\n', num2str(whichFile), SegmentFiles(whichFile).name)

    data_=zeros(sampleLen,30,3,sampleNumPerFile);
    label_=zeros(sampleNumPerFile,1);
    
    
    fprintf('seectFile  : %s, matFileName: %s\n', num2str(whichFile), SegmentFiles(whichFile).name)
    data = load([dirMat,'/',SegmentFiles(whichFile).name]);
    lowpass = data.data_;  %lowpass = data.data;
    lenData = size(lowpass);
    lenData = lenData(1);
    %lowpassDiff = diff(lowpass); 
    lowpassDiff = lowpass;
    startRow = whichFile*10-9;
    endRow = whichFile*10;
    %cvsOneFile = cvsSegment(:,:,whichFile);       % use Duan YuCheng   
    cvsOneFile = cvsSegment(startRow:endRow,3:4); 


    for i= startSampleNum:1:endAampleNum  %1:1:10 
        rightActionStart = cvsOneFile(i,1);
        rightActionEnd    = cvsOneFile(i,2);
        
        move_start =round(floor((rightActionStart+rightActionEnd)/2)-(sampleLen/2)+1);
        move_end   =round(floor((rightActionStart+rightActionEnd)/2)+(sampleLen/2));
       
        data_(:,:,:,i)= lowpassDiff(move_start:move_end,:,:);
        label_(i,1) = getCategory(SegmentFiles(whichFile).name,i) ;
    end
    %hold off
    
    %data_ = data_(1:20:end,:,:,:);%数据集缩小了20倍

    saveName = strrep(SegmentFiles(whichFile).name,'55','');
    %fprintf('size(data_)         : %s\n', num2str(size(data_)))
    save([saveDir,'\',saveName], 'data_')
    save([saveDir,'\',strrep(saveName,'.mat','_label.mat')], 'label_')

end
  
 function [categoryNum] = getCategory(readFileName,ii) 
    %readFileName = SegmentFiles(whichFile).name;
    fn = [readFileName(9),readFileName(10)];
    categoryNum = 0;
    if(ii<=5)
          if(fn == 'iw')
              categoryNum = 1;
          elseif(fn == 'ph')
              categoryNum = 3;
          elseif(fn == 'rp')
              categoryNum = 5;
          elseif(fn == 'sd')
              categoryNum = 7;
          elseif(fn == 'wd')
              categoryNum = 9;
          end
    else
          if(fn == 'iw')
              categoryNum = 2;
          elseif(fn == 'ph')
              categoryNum = 4;
          elseif(fn == 'rp')
              categoryNum = 6;
          elseif(fn == 'sd')
              categoryNum = 8;
          elseif(fn == 'wd')
              categoryNum = 10;
          end
    end
 end
    