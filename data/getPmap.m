image_path='./test/img';
position_head_path='./test/ann';
pmap_path='./';
for i=1:300
    i
    imgPath=fullfile(image_path,num2str(i,'IMG_%d.jpg'));
    posPath=fullfile(position_head_path,num2str(i,'GT_IMG_%d.mat'));
    pmapPath=fullfile(pmap_path,num2str(i,'PMAP_%d.mat'));
    pmap=segmentation(imgPath, posPath);
    save(pmapPath,'pmap');
    imagesc(pmap);
    %set(gca,'XTick',[]);
    %set(gca,'YTick',[]);
    %set(gca,'Position',[0 0 1 1]);
    %saveas(gcf,['./PMAP_',num2str(i),'.jpg'])
end
