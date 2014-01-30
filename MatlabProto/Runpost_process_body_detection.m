originalI=I;
bodyI=Iskin2;
imshow(bodyI), title('body');

sel=strel('disk',1,0);
%sel=strel('diamond',2); % good
%sel=strel('square',4);
%i2=imerode(bodyI,sel);
%i2=imclose(bodyI, sel);
bodyI2=imopen(bodyI, sel);
imshow(bodyI2);

% Gaussian
% gauFilt = fspecial('gaussian',9,2.5);
% %bar3(fn_gau,'b');
% %contour(fn_gau);
% bodyI2 = imfilter(bodyI,gauFilt);

% median (works on single channel eg: gray image)
% medWnd=[7 7];
% i2R = medfilt2(bodyI(:,:,1),medWnd);
% i2G = medfilt2(bodyI(:,:,2),medWnd);
% i2B = medfilt2(bodyI(:,:,3),medWnd);
% bodyI2=cat(3, i2R, i2G, i2B);

% min filter
%filtDomain=ones(4,4);
% filtDomain=[0 1 0; 1 1 1; 0 1 0];
% ord=1;
% i2R = ordfilt2(bodyI(:,:,1),ord, filtDomain);
% i2G = ordfilt2(bodyI(:,:,2),ord, filtDomain);
% i2B = ordfilt2(bodyI(:,:,3),ord, filtDomain);
% bodyI2=cat(3, i2R, i2G, i2B);
% imshow(bodyI2);

% remove islands of pixels
bodyGray=bodyI2(:,:,1);
imshow(bodyGray);
bodyConnComps=bwconncomp(bodyGray, 8);
bodyConnCompAreas=regionprops(bodyConnComps,'Area');
bodyConnCompAreasAr=struct2array(bodyConnCompAreas);
islandsInds=bodyConnCompAreasAr<200 | bodyConnCompAreasAr > 10000;
islands=bodyConnComps.PixelIdxList(islandsInds);
bodyNoIslands=bodyGray;

for k=1:length(islands)
    oneInds=islands{k};
    islandMask = ind2sub(size(bodyNoIslands), oneInds);
    bodyNoIslands(islandMask)=0; % remove island's pixel
end
imshow(bodyNoIslands);

% enlarge slightly bodies
sel=strel('disk',3,0);
bodyNoIslandsPad=imclose(bodyNoIslands, sel);
bodyNoIslandsPad(bodyNoIslandsPad > 0) = 255; % make a BW mask
imshow(bodyNoIslandsPad);

% TODO: take pool boundary only
% temporarily, in result better track all bodies around pool
bodyNoIslandsPad = bitand(im2uint8(poolMask), bodyNoIslandsPad);
imshow(bodyNoIslandsPad);

% take corresponding RGB image
bodyNoIslandsMask=im2uint8(bodyNoIslandsPad > 0);
bodyNoIslandsRGB = cat(3, bitand(originalI(:,:,1), bodyNoIslandsMask),...
                          bitand(originalI(:,:,2), bodyNoIslandsMask),...
                          bitand(originalI(:,:,3), bodyNoIslandsMask));
imshow(bodyNoIslandsRGB);

% TODO: what to do with big marker sign?

% put numbers/boundaries for each tracked objects
bodyiesConnComps = bwconncomp(bodyNoIslandsMask, 4);
bodyiesConnCompsProps = regionprops(bodyiesConnComps,'BoundingBox','Centroid');
hold on
for k=1:size(bodyiesConnCompsProps,1)
    box=bodyiesConnCompsProps(k).BoundingBox
    boxXs = [box(1), box(1) + box(3), box(1) + box(3), box(1), box(1)];
    boxYs = [box(2), box(2), box(2) + box(4), box(2) + box(4), box(2)];
    plot(boxXs, boxYs, 'g');
    plot(bodyiesConnCompsProps(k).Centroid(1), bodyiesConnCompsProps(k).Centroid(2), 'g.');
    text(max(box(1),box(1) + box(3) - 26), box(2) - 13, int2str(k), 'Color', 'g');
end
hold off
