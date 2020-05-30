% This plot the figure for manually segmenting activities.
clear
selectFile = 1;  % for user1, diff(lowpass) of 14 15 17 20 is not obvious,
userNum = 'user1';
%load('20191212ManualSegment/user1ManualSegment.mat'); cvsSegment = QQ;% use Duan YuCheng
cvsSegment = csvread(['ManualSegmentStartEnd/',userNum,'ManualSegment.csv']);% '20191212ManualSegment/user1ManualSegment.csv'
dirMat = ['DataCsiAmplitudeCut\',userNum];  %'20191211OriginalMatData\user2'
SegmentFiles = dir([dirMat,'/','*.mat']); % 55user1_iw_1.mat
numberFiles = length(SegmentFiles);
for whichFile =1:numberFiles
    %fprintf('seectFile  : %s, matFileName: %s\n', num2str(whichFile), SegmentFiles(whichFile).name)
    if(whichFile ~= selectFile)
        continue;
    end
    fprintf('seectFile  : %s, matFileName: %s\n', num2str(whichFile), SegmentFiles(whichFile).name)
    data = load([dirMat,'/',SegmentFiles(whichFile).name]);
    
    %originalMatFile = 'OriginalMatData\55user1_ph_6.mat';
    %originalMatFile = 'OriginalMatData\55user1_iw_5.mat'
    %data = load(originalMatFile);
    lowpass = data.data_;
    lenData = size(lowpass);
    lenData = lenData(1);
    %fprintf('before size(lowpass)   : %s\n', num2str(size(lowpass)))
    %lowpass = lowpass(1:20:lenData,:,:);%数据集缩小了20倍
    lowpassDiff = diff(lowpass);
    figure(2)
    dcm_obj = datacursormode(gcf); % change Value of 数据游标
    set(dcm_obj,'UpdateFcn',@NewCallback) 
    plot(lowpassDiff(:,1,1))
    
    startRow = selectFile*10-9;
    endRow = selectFile*10;
    %cvsOneFile = cvsSegment(:,:,whichFile);       % use Duan YuCheng 
    cvsOneFile = cvsSegment(startRow:endRow,3:4);
    hold on
    lineAmp = 1.5;
    for i=1:1:10
        plot([cvsOneFile(i,1),cvsOneFile(i,1)],[-lineAmp,lineAmp],'r');
        plot([cvsOneFile(i,2),cvsOneFile(i,2)],[-lineAmp,lineAmp],'r');
        text(cvsOneFile(i,1),lineAmp-0.03,num2str(i),'Color','red');
    end
    hold off
    set(gca,'xticklabel',get(gca,'xtick'),'yticklabel',get(gca,'ytick'));
    
    
    figure(3)
    dcm_obj = datacursormode(gcf); % change Value of 数据游标
    set(dcm_obj,'UpdateFcn',@NewCallback) 
    plot(lowpass(:,1,1))
    hold on
    lineAmp = 05.08;
    for i=1:1:10
        plot([cvsOneFile(i,1),cvsOneFile(i,1)],[-lineAmp,lineAmp],'r');
        plot([cvsOneFile(i,2),cvsOneFile(i,2)],[-lineAmp,lineAmp],'r');
        text(cvsOneFile(i,1),lineAmp-0.03,num2str(i),'Color','red');
    end
    hold off
    set(gca,'xticklabel',get(gca,'xtick'),'yticklabel',get(gca,'ytick'));
    
    
    if(0)        
        lenDataDiff = size(lowpassDiff);
        lenDataDiff = lenDataDiff(1); % length of lowpass is different from lenDataDiff
        lowpassPlus = lowpass(1:lenDataDiff,:,:).* abs(lowpassDiff);
        figure(4)
        ddd = lowpass(:,1,1);
        [lowpassNorm,pp] = mapminmax(ddd',0,0.5);
        %plot(lowpassPlus(:,1,1))
        lowpassNorm = lowpassNorm';
        lowpassPlus = 0.3*lowpassNorm(1:lenDataDiff)+ 0.6*(lowpassDiff(1:lenDataDiff,1,1));
        plot(lowpassPlus)
        set(gca,'xticklabel',get(gca,'xtick'),'yticklabel',get(gca,'ytick'));
        
        hold on
        lineAmp = 0.08;
        for i=1:1:10
            plot([cvsOneFile(i,1),cvsOneFile(i,1)],[-lineAmp,lineAmp],'r');
            plot([cvsOneFile(i,2),cvsOneFile(i,2)],[-lineAmp,lineAmp],'r');
            text(cvsOneFile(i,1),lineAmp-0.03,num2str(i),'Color','red');
        end
        hold off
    end
end
    