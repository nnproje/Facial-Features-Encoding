FilePath = 'F:\College\Training\FacialExpressionDatasets\AR modified Grey\'; %%insert 
Files = dir(FilePath);
for k=3:length(Files)
   ImgName = strcat(FilePath,Files(k).name);
   Img = imread(ImgName);
   ImgVect = Img';
   ImgVect = ImgVect(:);
   AllImgs(:,k-2) = ImgVect;
end
ALLImgX = AllImgs';
dlmwrite('AR modified (test set)',ALLImgX)