
    lenData = size(data);
    lenData = lenData(1);
    lowpass = data(1:20:lenData,:,:);%数据集缩小了20倍
    lowpassDiff = diff(lowpass);
    figure(1)
    plot(lowpassDiff(:,1,1))
    figure(3)
    plot(lowpass(:,1,1))

    set(gca,'xticklabel',get(gca,'xtick'),'yticklabel',get(gca,'ytick'));
    
    