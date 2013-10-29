I = imread('data/MVI_3177_0127.png');
figure, imshow(I)

% color to three colors
[I2,map2] = rgb2ind(I,3,'nodither');
figure, imshow(I2,map2)
figure, imshow(I2,hsv(3))

% color to two colors
[I2,map2] = rgb2ind(I,2,'nodither');
figure, imshow(I2,map2)

% color components seems not to give something usefull
figure, imshow(I(:,:,1)), title('R component');
figure, imshow(I(:,:,2)), title('G component');
figure, imshow(I(:,:,3)), title('B component');
figure, imhist(I(:,:,1)), title('R hist');
imcontrast;

% threshold
iGray = rgb2gray(I);
imshow(iGray)
thr1 = graythresh(iGray);
iBw = im2bw(iGray, thr1);
figure, imshow(iBw), title('global Otsu');

% black-white each component
% NOTE: red component gives many information about bodies
comp=I(:,:,1);
thr=graythresh(comp);
bw = im2bw(comp, thr);
figure, imshow(bw), title('global Red');

comp=I(:,:,2);
thr=graythresh(comp);
bw = im2bw(comp, thr);
figure, imshow(bw), title('global Green');

comp=I(:,:,3);
thr=graythresh(comp);
bw = im2bw(comp, thr);
figure, imshow(bw), title('global Blue');

% extract water+bodies=couple of lanes and analyze histogram
m1=roipoly(I); % 1 lane
m2=roipoly(I); % 2 lane
m3=roipoly(I); % 3 lane
m4=roipoly(I); % 4 lane
m42=roipoly(I); % 4b lane
m5=roipoly(I); % 5 lane
m=m1 | m2 | m3 | m4 | m42 | m5;
%save('Mask_Water1.mat', 'm')
load('Mask_Water1.mat','m');

mint=im2uint8(m);
mintRgb=cat(3,mint,mint,mint);
waterMask=bitand(I,mintRgb);
imshow(waterMask)
imhist(waterMask(waterMask > 0))
waterMask1=waterMask(:,:,1);
waterMask2=waterMask(:,:,2);
waterMask3=waterMask(:,:,3);
figure, imhist(waterMask1(waterMask1>0)) %red
figure, imhist(waterMask2(waterMask2>0)) %green
figure, imhist(waterMask3(waterMask3>0)) %blue

% find average water color
waterRgb=zeros(1,3);
waterRgb(1)=median(waterMask1(waterMask1>0)); % 83
waterRgb(2)=median(waterMask2(waterMask2>0)); % 129
waterRgb(3)=median(waterMask3(waterMask3>0)); % 138

% image of water color
av2=reshape(waterRgb, [1,1,3]);
av2=cast(av2,'uint8');
avWat=repmat(av2,[size(I,1) size(I,2)]);
avWat=cast(avWat,'uint8');
imshow(avWat) % 'average' water color

% find distances from pixels to 'water' cluster
Icol=reshape(I,size(I,1)*size(I,2),3);
avWater = repmat(waterRgb, size(I,1)*size(I,2), 1);
avWater = cast(avWater, 'uint8');
distFromWat=sum((Icol - avWater ).^2, 2);
hist(distFromWat)

% =no separation Water-Other
distFromWat2d=sum((I-avWat).^2, 3);
imshow(distFromWat2d,[]);

skinRgb=uint8([191 118 84]);
%sum(([208 114 124]-double(skinRgb)).^2)
skinIm=repmat(reshape(skinRgb, [1 1 3]), [size(I,1) size(I,2)]);
distFromSkin = sum((double(I)-double(skinIm)).^2, 3);
%distFromSkin = prod((double(I)-double(skinIm)).^2, 3).^(1/3);
imshow(distFromSkin, [])
%imshow(imcomplement(distFromSkin), [])
min(distFromSkin(:))
max(distFromSkin(:))