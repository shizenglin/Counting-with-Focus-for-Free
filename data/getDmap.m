image_path='./test/img';
position_head_path='./test/ann';
dmap_path='./test/dmap';
for i=1:334
    i
    imgPath=fullfile(image_path,num2str(i,'img_%04d.jpg'));
    posPath=fullfile(position_head_path,num2str(i,'img_%04d_ann.mat'));
    dmapPath=fullfile(dmap_path,num2str(i,'DMAP_%04d.mat'));
    dmap=density(imgPath, posPath);
    save(dmapPath,'dmap');
    imagesc(dmap);
    %set(gca,'XTick',[]);
    %set(gca,'YTick',[]);
    %set(gca,'Position',[0 0 1 1]);
    %saveas(gcf,['./DMAP_',num2str(i),'.jpg'])
end
