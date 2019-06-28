%Read entire folder ==> convert to greyscale ==> save newImgs
RGB_FilePath = 'F:\College\Training\FacialExpressionDatasets\CFEED modified\';
GREY_FilePath = 'F:\College\Training\FacialExpressionDatasets\CFEED modified Grey\';
Files = dir(RGB_FilePath);
for k=3:length(Files)
   RGB_ImgName = strcat(RGB_FilePath,Files(k).name);
   RGB = imread(RGB_ImgName);
   GREY = rgb2gray(RGB);
   GREY_ImgName = strcat(GREY_FilePath,Files(k).name);
   imwrite(uint8(GREY), GREY_ImgName);
end