% this extract amplitudes from raw CSI .dat files, and save as .mat files
clear
datDir = 'DataRawCSIDat/user1_Bai_ShiQing_dat/';      matUser = 'user1';
%datDir = 'DataRawCSIDat/user2_He_HaiSheng_dat/';     matUser = 'user2';
%datDir = 'DataRawCSIDat/user3_Lei_Yue_dat/';         matUser = 'user3';
%datDir = 'DataRawCSIDat/user4_Shao_HongTao_dat/';    matUser = 'user4';
%datDir = 'DataRawCSIDat/user5_Yan_Di_dat/';          matUser = 'user5';
matDir = ['DataCsiAmplitude/',matUser,'/']; %matDir = 'AmplitudeMat/user5/';
action_files = dir(fullfile(datDir,'*.dat'));
addpath(genpath('matlabCsiTool/'));
%addpath('C:\Program Files\MATLAB\R2018a\bin\dyc');
for i_text = 1:length(action_files)
    fprintf('read dat  : %s -- fieName: %s\n',  num2str(i_text),action_files(i_text).name)   
    file_name = action_files(i_text).name;
    data_file = [datDir,file_name];  
    csi_trace=read_bf_file(data_file);%2个发送端，3个接收端，每一条回路上有30个子载波
        [l k]=size(csi_trace);
    for idx=1:l
        if isempty(csi_trace{idx})==1
            g=idx-1;
        else
            g=idx;
        end
    end
    csi_mat = zeros(g,3*30);
    % form csi_stream
    for a = 1:g  % take num csi packets
        csia = get_scaled_csi(csi_trace{a}); % get scaled_csi,val(:,:,1)----val(:,:,30)
        for k = 1:3  %3
            for m = 1:30  %30
                 B = csia(1,k,m);  %取一个CSI点
                 %csi_phase=angle(B);
                 %csi_mat(a,m+(k-1)*30) = csi_phase; 
                 csi_amplitude = abs(B);
                 csi_mat(a,m+(k-1)*30) = csi_amplitude; 
            end
        end
    end	
    
    [a, b] = butter(5, 0.05, 'low');
    lowpass_ = filter(a, b,csi_mat);
    data_Y{i_text} = lowpass_;
end
%save('data_Y.mat','data_Y')
%save file to mat
for i_text = 1:1:length(action_files)
    fprintf('save mat  : %s -- fieName: %s\n',  num2str(i_text),action_files(i_text).name)      
    file_name = action_files(i_text).name;
    fn = [file_name(4),file_name(5)];
    name = ['55',matUser,'_',fn,'_',file_name(7)];%name = ['55','user5','_',fn,'_',file_name(7)];
    
    lowpass = data_Y{1,i_text};
    data = zeros(length(lowpass(:,1)),30,3);
    for i = 1:length(lowpass(:,1))  
    
        data(i,:,1) = lowpass(i,1:30);
        data(i,:,2) = lowpass(i,31:60);
        data(i,:,3) = lowpass(i,61:90);
    end
    %save([path, name], 'data')
    save([matDir,name], 'data')
end
