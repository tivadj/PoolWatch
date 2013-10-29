classdef RunHumanDetector < handle
methods(Static)

function obj = create()
    obj = utils.TypeErasedClass;
end

function run(obj)
    debug = 1;
   
    RunHumanDetector.initHumanDetector(obj,debug);

    RunHumanDetector.testHumanDetectorOnImage(obj,debug);
    %RunHumanDetector.testWaterClassifier(obj, debug);
    %RunHumanDetector.testCombiningSkinAndWaterClassifiers(obj,debug);
    %RunHumanDetector.testGluingBodyParts(obj, debug);
end

% build human detector
function initHumanDetector(obj, debug)
    if ~isfield(obj.v, 'humanDetector')
        %
        distanceCompensator = CameraDistanceCompensator.create;

        %
        cl2=SkinClassifierStatics.create;
        SkinClassifierStatics.populateSurfPixels(cl2);
        SkinClassifierStatics.prepareTrainingDataMakeNonoverlappingHulls(cl2, debug);
        SkinClassifierStatics.findSkinPixelsConvexHullWithMinError(cl2, debug);

        %
        skinHullClassifierFun=@(XByRow) utils.inhull(XByRow, cl2.v.skinHullClassifHullPoints, cl2.v.skinHullClassifHullTriInds, 0.2);
        skinClassifier=skinHullClassifierFun;
        obj.v.skinHullClassifierFun = skinHullClassifierFun;
        
        % water classifier
        RunHumanDetector.initWaterClassifier(obj,debug);
        waterClassifierFun = obj.v.watClassifFun;

        %
        humanDetector = HumanDetector(skinClassifier, waterClassifierFun, distanceCompensator);
        obj.v.humanDetector = humanDetector;
    end
end

function initWaterClassifier(obj, debug)
    I = imread('../dinosaur/MVI_3177_0127.png');
    imshow(I);

    % clean water
    % cleanWat1=roipoly(I);
    % cleanWatMask = cleanWat1;
    %save('data/Mask_CleanWat1.mat', 'cleanWatMask');
    load('data/Mask_CleanWat1.mat','cleanWatMask');
    imshow(cleanWatMask);
    
    cleanWatImg = utils.applyMask(I, cleanWatMask);
    imshow(cleanWatImg);
    
    cleanWatPixs = reshape(cleanWatImg, [], 3);
    cleanWatPixs = unique(cleanWatPixs, 'rows');
    cleanWatPixs = cleanWatPixs(cleanWatPixs(:,1) > 0 & cleanWatPixs(:,2) > 0 & cleanWatPixs(:,3) > 0, :); % remove 'black' (on axes) pixels
    plot3(cleanWatPixs(:,1), cleanWatPixs(:,2),cleanWatPixs(:,3), 'b.');

    cleanWatPixsDbl = double(cleanWatPixs);
    cleanWatHullTri = convhulln(cleanWatPixsDbl);
    cleanWatSurf = trisurf(cleanWatHullTri, cleanWatPixs(:,1),cleanWatPixs(:,2),cleanWatPixs(:,3), 'FaceColor', 'c');
    alpha(cleanWatSurf, 0.8);

    cleanWatClassifFun = utils.PixelClassifier.getConvexHullClassifier(cleanWatPixsDbl, cleanWatHullTri);
    obj.v.cleanWatClassifFun = cleanWatClassifFun;
    cleanWatCfMask = utils.PixelClassifier.applyToImage(I, cleanWatClassifFun);
    cleanWatImg = utils.applyMask(I, cleanWatCfMask);
    imshow(cleanWatImg);
    %utils.inhull([143 127 117], cleanWatPixsDbl, cleanWatK222)
    xlabel('R'); ylabel('G'); zlabel('B');

    % extract water+bodies=couple of lanes and analyze histogram
    %{
    m1=roipoly(I); % 1 lane
    m2=roipoly(I); % 2 lane
    m3=roipoly(I); % 3 lane
    m4=roipoly(I); % 4 lane
    m42=roipoly(I); % 4b lane
    m5=roipoly(I); % 5 lane
    waterMask=m1 | m2 | m3 | m4 | m42 | m5;
    %save(fullfile('data/Mask_Water1.mat'), 'waterMask')
    %}
    load('data/Mask_Water1.mat','waterMask');
    watImg = utils.applyMask(I, waterMask);
    imshow(watImg);
    watPixs = reshape(watImg, [], 3);
    watPixs = unique(watPixs, 'rows');
    watPixs = watPixs(watPixs(:,1) > 0 & watPixs(:,2) > 0 & watPixs(:,3) > 0, :); % remove black
    plot3(watPixs(:,1), watPixs(:,2),watPixs(:,3), 'b.');

    watPixsDbl = double(watPixs);
    watHullTri = convhulln(watPixsDbl);
    hold on
    watSurf = trisurf(watHullTri, watPixs(:,1),watPixs(:,2),watPixs(:,3), 'FaceColor', [0 0 1]);
    alpha(watSurf, 0.3);
    hold off

    % 
    watClassifFun = utils.PixelClassifier.getConvexHullClassifier(watPixsDbl, watHullTri);
    obj.v.watClassifFun = watClassifFun;
    watMask = utils.PixelClassifier.applyToImage(I, watClassifFun);
    watImg = utils.applyMask(I, watMask);
    imshow(watImg);

    % collect convex hull free boundary pixels
    cleanWatHullPixsDbl = cleanWatPixsDbl(unique(cleanWatHullTri(:)), :);
    watHullPixsDbl = watPixsDbl(unique(watHullTri(:)), :);
    waterHullInfl=SkinClassifierStatics.inflateConvexHull(cleanWatHullPixsDbl, watHullPixsDbl, 1, 1);

    % free boundary pixels may be shifted so the convex hull is no longer valid
    % compute convex hull again
    waterHullInflTri = convhulln(waterHullInfl);
    hold on
    waterHullInflSurf = trisurf(waterHullInflTri, waterHullInfl(:,1),waterHullInfl(:,2),waterHullInfl(:,3), 'FaceColor', [0 1 0]);
    alpha(waterHullInflSurf, 0.3);
    hold off

    % apply inflated convex hull classifier to image
    waterHullInflClassifFun = utils.PixelClassifier.getConvexHullClassifier(waterHullInfl, waterHullInflTri);
    obj.v.waterHullInflClassifFun = waterHullInflClassifFun;
end

function testHumanDetectorOnImage(obj, debug)
    debug = 0;
    
    RunHumanDetector.initHumanDetector(obj,debug);
    
    %I = imread('data/MVI_3177_0127.png');
    %I = imread('data/mvi3177_47sec.png');
    videoFilePath = fullfile('../output/mvi3177_blueWomanLane3.avi');
    videoReader = VideoReader(videoFilePath);
    
    I = read(videoReader, 100);
    %I = blueWomanLane3(:,:,:,40);
    load('data/Mask_lane3Mask.mat', 'lane3Mask')
    I = utils.applyMask(I,lane3Mask);
    imshow(I)

    bodyDescrs = obj.v.humanDetector.GetHumanBodies(I, debug);

    % put numbers/boundaries for each tracked objects
    %figure(1);
    hold on
    for k=1:length(bodyDescrs)
        box=bodyDescrs(k).BoundingBox
        boxXs = [box(1), box(1) + box(3), box(1) + box(3), box(1), box(1)];
        boxYs = [box(2), box(2), box(2) + box(4), box(2) + box(4), box(2)];
        plot(boxXs, boxYs, 'g');
        plot(bodyDescrs(k).Centroid(1), bodyDescrs(k).Centroid(2), 'g.');
        text(max(box(1),box(1) + box(3) - 26), box(2) - 13, int2str(k), 'Color', 'g');
    end
    hold off
end

function testWaterClassifier(obj, debug)
    assert(isfield(obj.v, 'watClassifFun'));
    
    %
    %i1 = utils.VideoHelper.readFrameSingle('data/mvi3177_blueWomanLane3_16frames.avi', 16);
    i1 = imread('../dinosaur/close_swimmer1.png');
    imshow(i1)
    
    i1WaterMask = utils.PixelClassifier.applyToImage(i1, obj.v.watClassifFun,debug);
    figure, imshow(~i1WaterMask), title('no water mask');
    
    i1NoWater = utils.applyMask(i1, ~i1WaterMask);
    imshow(i1NoWater), title('no water');

%     
%     
%     i1Both = i1Skin & i1Water;
%     figure, imshow(i1Both);
%     
%     i1ImgBoth = utils.applyMask(i1, i1Both);
%     figure, imshow(i1ImgBoth);
%     load('data/Mask_Pool1.mat','poolMask');
%     poolMask = imresize(poolMask, [size(i1NoWater,1) size(i1NoWater,2)]); 
%     load('data/Mask_Water1.mat','waterMask');
%     waterMask = imresize(waterMask, [size(i1NoWater,1) size(i1NoWater,2)]); 
%     i1NoWaterPoolOnly = utils.applyMask(i1NoWater, poolMask&waterMask);
%     imshow(i1NoWaterPoolOnly);
end

function testCombiningSkinAndWaterClassifiers(obj, debug)
    assert(isfield(obj.v, 'skinHullClassifierFun'));
    assert(isfield(obj.v, 'watClassifFun'));
    
    %
    i1 = imread('../dinosaur/close_swimmer1.png');
    imshow(i1)
    
    i1WaterMask = utils.PixelClassifier.applyToImage(i1, obj.v.watClassifFun,debug);
    figure, imshow(~i1WaterMask), title('no water mask');
    
    i1NoWater = utils.applyMask(i1, ~i1WaterMask);
    figure, imshow(i1NoWater), title('no water');

    i1Skin = utils.PixelClassifier.applyAndGetImage(i1NoWater, obj.v.skinHullClassifierFun,debug);
    figure, imshow(i1Skin), title('no water and skin');
%     
%     
%     i1Both = i1Skin & i1Water;
%     figure, imshow(i1Both);
%     
%     i1ImgBoth = utils.applyMask(i1, i1Both);
%     figure, imshow(i1ImgBoth);
    load('data/Mask_Pool1.mat','poolMask');
    poolMask = imresize(poolMask, [size(i1NoWater,1) size(i1NoWater,2)]); 
    load('data/Mask_Water1.mat','waterMask');
    waterMask = imresize(waterMask, [size(i1NoWater,1) size(i1NoWater,2)]); 
    i1NoWaterPoolOnly = utils.applyMask(i1NoWater, poolMask&waterMask);
    imshow(i1NoWaterPoolOnly);
    bodyDescrs = obj.v.humanDetector.GetHumanBodies(i1NoWaterPoolOnly, true);
end


function testGluingBodyParts(obj, debug)
    i1 = imread('../artefacts/humanBodyRecognizer/elem_two_blocks.png');
    imshow(i1);
    i1Gray = rgb2gray(i1);
    imshow(i1Gray);
    
    connComps=bwconncomp(i1Gray, 8);
    connCompProps=regionprops(connComps,'Area','Centroid','PixelList');

    i2 = utils.drawRegionProps(i1Gray, connComps, 50);
    imshow(i2);
    
    bnd=bwboundaries(i1Gray);
    
    bothPoints = [bnd{1}; bnd{2}];
    bndInds = convhulln(bothPoints);
    
    newBndPoints = bothPoints(bndInds(:,1),:);
    newBndPoints = circshift(newBndPoints,[0 1]);
    i3 = cv.polylines(i1, {num2cell(newBndPoints,2)}, 'Color', [255 255 255],'Closed',true);
    figure(gcf+1), imshow(i3);
end

end
end