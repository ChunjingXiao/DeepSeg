%% Chunjing Xiao <ChunjingXiao@gmail.com> 20200530
%% DeepSeg: Deep Learning based Motion Segmentation Framework for Activity Recognition using WiFi
%% IEEE Internet of Things Journal 2020

% This is to extract training data for training the state inference model for segmentation
% This extracts four bins (traingin sampes) from each activity data. These bins are used to train the state inference model.
% static-state: the bin is full of CSI data in the absence of activities; 
% start-state: the bin contains the start point of an activity; 
% motion-state: the bin is full of CSI data in the presence of activities; 
% end-state: the bin contains the end point of an activity.

% start-state contains half of the non-action part and half of the action part, 
% and end-state also contains both them, but the action part is in front. 
% Instead, static-state only contains the non-action part, 
% and motion-state only contains the action part.

clear
sampleLen = 2400/20;
sampleCategory = 4;
%userNum = 'user1';
for userSelect = {'user1' 'user2' 'user3' 'user4' 'user5'}
    userNum = userSelect{1,1}

saveDir = ['TrainingDataForSegment/',userNum,'_data_label']; %'20191220SegmentTrainNew/user2_data_label';
whetherPlot = 0;
selectFile = 1;  % for user1, diff(lowpass) of 14 15 17 20 is not obvious,
startSampleNum = 1;
endAampleNum = 10;
sampleNumPerFile = endAampleNum - startSampleNum + 1; %每个文件有10个样本
%cvsSegment = csvread(['Label_ActivityCategoryStartEnd/',userNum,'ManualSegment.csv']);
% 'Label_ActivityCategoryStartEnd/user1ManualSegment.csv'
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
    if(whichFile ~= selectFile)
        if(whetherPlot)
            continue;
        end
    end
    data_=zeros(sampleLen,30,3,sampleNumPerFile*sampleCategory);
    label_=zeros(sampleNumPerFile*sampleCategory,1);
    
    
    fprintf('seectFile  : %s, matFileName: %s\n', num2str(whichFile), SegmentFiles(whichFile).name)
    data = load([dirMat,'/',SegmentFiles(whichFile).name]);
    lowpass = data.data_;
    %lowpass = lowpass(1:20:end,:,:,:);%数据集缩小了20倍
    
    lenData = size(lowpass);
    lenData = lenData(1);
    lowpassDiff = diff(lowpass); 
    %lowpassDiff = lowpass;
    startRow = whichFile*10-9;
    endRow = whichFile*10;
    %cvsOneFile = cvsSegment(:,:,whichFile);       % use Duan YuCheng   
    cvsOneFile = cvsSegment(startRow:endRow,3:4); %round(cvsSegment(startRow:endRow,3:4)/20)
    if(whetherPlot)
         figure(1)
         dcm_obj = datacursormode(gcf); % change Value of 数据游标
         set(dcm_obj,'UpdateFcn',@NewCallback)
         plot(lowpassDiff(:,1,1))
    end
    k = 1;
    for i= startSampleNum:1:endAampleNum  %1:1:10 
        rightActionStart = cvsOneFile(i,1);
        rightActionEnd    = cvsOneFile(i,2);

        actionBegin_start =round(rightActionStart-(sampleLen/2)+1);  
        actionBegin_end   =round(rightActionStart+(sampleLen/2));  
        move_start =round(floor((rightActionStart+rightActionEnd)/2)-(sampleLen/2)+1);
        move_end   =round(floor((rightActionStart+rightActionEnd)/2)+(sampleLen/2));
        %move_start        =round(rightActionStart+400+1);
        %move_end          =round(rightActionStart+400 + sampleLen);
        
        actionEnd_start =round(rightActionEnd-(sampleLen/2)+1);  
        actionEnd_end   =round(rightActionEnd+(sampleLen/2)); 
        
        static_start = round(rightActionEnd+(sampleLen/2)) + 1;
        static_end   = round(rightActionEnd+(sampleLen/2)) + sampleLen;        
        % for seectFile:8, matFileName: 55user1_ph_2.mat 索引超出数组范围(不能超过 118919)
        if( static_end>lenData) % 取动作开始前面的作为静态数据
            static_start = round(rightActionStart-(sampleLen/2)) -sampleLen + 1;
            static_end   = round(rightActionStart-(sampleLen/2));
        end
         

        data_(:,:,:,k+2:k+2)= lowpassDiff(static_start:static_end,:,:);
        label_(k+2:k+2,1) = 1;   
        data_(:,:,:,k:k)= lowpassDiff(actionBegin_start:actionBegin_end,:,:);
        label_(k:k,1) = 2;
        data_(:,:,:,k+1:k+1)= lowpassDiff(move_start:move_end,:,:);
        label_(k+1:k+1,1) = 3;
        data_(:,:,:,k+3:k+3)= lowpassDiff(actionEnd_start:actionEnd_end,:,:);
        label_(k+3:k+3,1) = 4;        
        k=k+sampleCategory;
        

    
        if(whetherPlot)
            hold on
            lineAmp = 0.18;
            plot([rightActionStart,rightActionStart],[-lineAmp,lineAmp],'c');
            plot([rightActionEnd,rightActionEnd],[-lineAmp,lineAmp],'c');
            text(cvsOneFile(i,1),lineAmp-0.03,num2str(i),'Color','red');
            lineAmp = 0.08;
            plot([actionBegin_start,actionBegin_start],[-lineAmp,lineAmp],'y');
            plot([actionBegin_end,actionBegin_end],[-lineAmp,lineAmp],'y');

            plot([move_start,move_start],[-lineAmp,lineAmp],'m');
            plot([move_end,move_end],[-lineAmp,lineAmp],'m');

            plot([static_start,static_start],[-lineAmp,lineAmp],'g');
            plot([static_end,static_end],[-lineAmp,lineAmp],'g');
            set(gca,'xticklabel',get(gca,'xtick'),'yticklabel',get(gca,'ytick'));
            hold off
        end
    end
    %hold off
    
    %data_ = data_(1:20:end,:,:,:);%数据集缩小了20倍

    saveName = strrep(SegmentFiles(whichFile).name,'55','');
    %fprintf('size(data_)         : %s\n', num2str(size(data_)))
    save([saveDir,'\',saveName], 'data_')
    save([saveDir,'\',strrep(saveName,'.mat','_label.mat')], 'label_')
  end
  end