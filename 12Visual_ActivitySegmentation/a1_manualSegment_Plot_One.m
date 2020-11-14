% This plots the figure for manually checking segmentation results.
clear
userNum = 'user1'; % Which user
selectFile = 1;    % Which file of this user
% 'Label_ActivityStartEndForSegment/user1ManualSegment.csv'
%cvsSegment = csvread(['Label_ActivityStartEndForSegment/',userNum,'ManualSegment.csv']);
fid = fopen(['Label_CsiAmplitudeCut/',userNum,'ManualSegment.csv']);
dcells = textscan(fid, '%f,%f,%f,%f,%s', 'HeaderLines', 1, 'EndOfLine', '\r\n');
fclose(fid); 
dcellneeds = dcells(1:4);
cvsSegment = cell2mat(dcellneeds);
%disp(cvsSegment);
dirMat = ['Data_CsiAmplitudeCut\',userNum];  %'Data_CsiAmplitude_Cut\user1'
SegmentFiles = dir([dirMat,'/','*.mat']); % 55user1_iw_1.mat
numberFiles = length(SegmentFiles);
for whichFile =1:numberFiles
    %fprintf('seectFile  : %s, matFileName: %s\n', num2str(whichFile), SegmentFiles(whichFile).name)
    if(whichFile ~= selectFile)
        continue;
    end
    fprintf('seectFile  : %s, matFileName: %s\n', num2str(whichFile), SegmentFiles(whichFile).name)
    data = load([dirMat,'/',SegmentFiles(whichFile).name]);
    
    originalData = data.data_;
    sizeOfData = size(originalData);
    lenData = sizeOfData(1);    
    lowpassDiff = diff(originalData);
    %fprintf('before size(lowpass)   : %s\n', num2str(size(lowpass)))
    
    startRow = selectFile*10-9;  % read start and end points of activities
    endRow = selectFile*10;
    cvsOneFile = cvsSegment(startRow:endRow,3:4);
    
    
    figure(1) % for original data
    dcm_obj = datacursormode(gcf); % change Value of 数据游标
    set(dcm_obj,'UpdateFcn',@NewCallback) 
    plot(originalData(:,1,1))     % plot wave figure of amplitude 
    hold on
    lineAmp = 05.08;              
    for i=1:1:10        % mark the start and end points of activities
        plot([cvsOneFile(i,1),cvsOneFile(i,1)],[-lineAmp,lineAmp],'r');
        plot([cvsOneFile(i,2),cvsOneFile(i,2)],[-lineAmp,lineAmp],'r');
        text(cvsOneFile(i,1),lineAmp-0.03,num2str(i),'Color','red');
    end
    hold off
    set(gca,'xticklabel',get(gca,'xtick'),'yticklabel',get(gca,'ytick'));
    
    
    figure(2) % for lowpass data
    dcm_obj = datacursormode(gcf); % change Value of 数据游标
    set(dcm_obj,'UpdateFcn',@NewCallback) 
    plot(lowpassDiff(:,1,1)) % plot wave figure of amplitude   

    hold on
    lineAmp = 1.5;
    for i=1:1:10  % mark the start and end points of activities
        plot([cvsOneFile(i,1),cvsOneFile(i,1)],[-lineAmp,lineAmp],'r');
        plot([cvsOneFile(i,2),cvsOneFile(i,2)],[-lineAmp,lineAmp],'r');
        text(cvsOneFile(i,1),lineAmp-0.03,num2str(i),'Color','red');
    end
    hold off
    set(gca,'xticklabel',get(gca,'xtick'),'yticklabel',get(gca,'ytick'));
    
    
    
end
    