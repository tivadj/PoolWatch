classdef RunHumanDetector < handle
methods(Static)

function obj = create()
    obj = utils.TypeErasedClass;
end

function run(obj)
    debug = 1;
   
    %RunHumanDetector.initHumanDetector(obj,debug);
    RunHumanDetector.initWaterClassifier(obj, debug);

    %RunHumanDetector.testHumanDetectorOnImage(obj,debug);
    %RunHumanDetector.testWaterClassifier(obj, debug);
    %RunHumanDetector.testCombiningSkinAndWaterClassifiers(obj,debug);
    %RunHumanDetector.testGluingBodyParts(obj, debug);
    %RunHumanDetector.visualizeHeadPixelsAsVolume(obj, debug);
    RunHumanDetector.testSegmentPoolBoundary(obj,debug);
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
    if isfield(obj.v, 'watClassifFun')
        return;
    end
    
    I = imread('../dinosaur/MVI_3177_0127.png');
    imshow(I);

    % clean water
    % cleanWat1=roipoly(I);
    % cleanWatMask = cleanWat1;
    %save('data/Mask_CleanWat1.mat', 'cleanWatMask');
    load('data/Mask_CleanWat1.mat','cleanWatMask');
    figure(1), imshow(cleanWatMask);
    
    cleanWatImg = utils.applyMask(I, cleanWatMask);
    figure(1), imshow(cleanWatImg);
    
    cleanWatPixs = reshape(cleanWatImg, [], 3);
    cleanWatPixs = unique(cleanWatPixs, 'rows');
    cleanWatPixs = cleanWatPixs(cleanWatPixs(:,1) > 0 & cleanWatPixs(:,2) > 0 & cleanWatPixs(:,3) > 0, :); % remove 'black' (on axes) pixels
    figure(2), plot3(cleanWatPixs(:,1), cleanWatPixs(:,2),cleanWatPixs(:,3), 'b.');

    cleanWatPixsDbl = double(cleanWatPixs);
    cleanWatHullTri = convhulln(cleanWatPixsDbl);
    figure(3);
    cleanWatSurf = trisurf(cleanWatHullTri, cleanWatPixs(:,1),cleanWatPixs(:,2),cleanWatPixs(:,3), 'FaceColor', 'c');
    alpha(cleanWatSurf, 0.8);

    cleanWatClassifFun = utils.PixelClassifier.getConvexHullClassifier(cleanWatPixsDbl, cleanWatHullTri);
    obj.v.cleanWatClassifFun = cleanWatClassifFun;
    cleanWatCfMask = utils.PixelClassifier.applyToImage(I, cleanWatClassifFun);
    cleanWatImg = utils.applyMask(I, cleanWatCfMask);
    figure(1), imshow(cleanWatImg);
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
    figure(1), imshow(watImg);
    watPixs = reshape(watImg, [], 3);
    watPixs = unique(watPixs, 'rows');
    watPixs = watPixs(watPixs(:,1) > 0 & watPixs(:,2) > 0 & watPixs(:,3) > 0, :); % remove black
    figure(2), plot3(watPixs(:,1), watPixs(:,2),watPixs(:,3), 'b.');

    watPixsDbl = double(watPixs);
    watHullTri = convhulln(watPixsDbl);
    figure(3)
    hold on
    watSurf = trisurf(watHullTri, watPixs(:,1),watPixs(:,2),watPixs(:,3), 'FaceColor', [0 0 1]);
    alpha(watSurf, 0.3);
    hold off

    % 
    watClassifFun = utils.PixelClassifier.getConvexHullClassifier(watPixsDbl, watHullTri);
    obj.v.watClassifFun = watClassifFun;
    watMask = utils.PixelClassifier.applyToImage(I, watClassifFun);
    watImg = utils.applyMask(I, watMask);
    figure(1), imshow(watImg);

    % collect convex hull free boundary pixels
    cleanWatHullPixsDbl = cleanWatPixsDbl(unique(cleanWatHullTri(:)), :);
    watHullPixsDbl = watPixsDbl(unique(watHullTri(:)), :);
    waterHullInfl=SkinClassifierStatics.inflateConvexHull(cleanWatHullPixsDbl, watHullPixsDbl, 1, 1);

    % free boundary pixels may be shifted so the convex hull is no longer valid
    % compute convex hull again
    waterHullInflTri = convhulln(waterHullInfl);
    figure(3)
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
    %i1 = imread('../dinosaur/close_swimmer1.png');
    i1 = utils.VideoHelper.readFrameSingle(fullfile('../output/mvi3177_blueWomanLane3.avi'),741)
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

% Show where head (swimmer's cap) pixels are located.
function visualizeHeadPixelsAsVolume(obj, debug)
    i1=utils.VideoHelper.readFrameSingle(fullfile('../output/mvi3177_blueWomanLane3.avi'),741);
    headMask=roipoly(i1);
    imgHead = utils.applyMask(i1, headMask);
    imshow(imgHead)
    headPixs = unique(reshape(imgHead,[],3),'rows');
    headPixs(1,:) = []; % remove black
    
    headPixsDbl = double(headPixs);
    
    % draw volume around pixels
    triInds = convhulln(headPixsDbl);
    hold on
    headSurf = trisurf(triInds, headPixsDbl(:,1),headPixsDbl(:,2),headPixsDbl(:,3), 'FaceColor', 'r');
    alpha(headSurf,0.6)
    hold off
end

function tryFindLaneDividers(imgNoHoles, lines2, vanishPoint, bndPolyline)
    % extract divider polygons

    numLines = size(lines2,1);
    for i=1:numLines
    for i1=i+1:numLines
        p1 = [lines2(i,1) lines2(i,3)];
        p2 = [lines2(i,2) lines2(i,4)];
        p3 = [lines2(i1,1) lines2(i1,3)];
        p4 = [lines2(i1,2) lines2(i1,4)];
        dividerWidth = RunHumanDetector.estimateLaneDividerWidth(p1,p2, p3,p4, vanishPoint);
        fprintf('#%d(%.2f,%.2f,%.2f,%.2f)-#%d(%.2f,%.2f,%.2f,%.2f) wid=%.2f\n', i, p1(1),p1(2),p2(1),p2(2), i1, p3(1),p3(2),p4(1),p4(2), dividerWidth);
    end
    end
end

function testSegmentPoolBoundary(obj, debug)
    i1=utils.VideoHelper.readFrameSingle(fullfile('../output/mvi3177_blueWomanLane3.avi'),741);
    %i1=imread(fullfile('../dinosaur/poolBoundary/MVI_3177_frame1.png'));
    imshow(i1);

    imageCalibPnts = PoolBoundaryDetector.getCalibrationPoints(i1, obj.v.watClassifFun);
    
    %RunHumanDetector.tryFindLaneDividers(imgNoHoles, lines2, vanishPoint, bndPolyline);
end

% Find cost of gluing segment(p1,p2) and segment(p3,p4).
function cost = getTwoSegmentsGroupingCost(p1,p2, p3,p4)
    % find shortest distance
    allPoints = [p1; p2; p3; p4];
    distInds = [1 3; 1 4; 2 3; 2 4]; % distances to calculate
    
    dists = arrayfun(@(seg1Ind, seg2Ind) norm(allPoints(seg1Ind,:)-allPoints(seg2Ind,:)), distInds(:,1), distInds(:,2));
    
    [minDist,minInd] = min(dists);
    
    closestPointPairInds = distInds(minInd,:);
    
    % segment1 = (closest1,further1)
    closest1 = allPoints(closestPointPairInds(1),:);
    further1 = allPoints(3 - closestPointPairInds(1),:);
    
    % segment2 = (closest2,further2)
    closest2 = allPoints(closestPointPairInds(2),:);
    further2 = allPoints(7 - closestPointPairInds(2),:);
    
    v1 = closest1 - further1;
    connector = closest2 - closest1;
    
    ang1 = acos(v1 * connector' / (norm(v1)*norm(connector)));
    
    v2 = further2 - closest2;
    ang2 = acos(v2 * connector' / (norm(v2)*norm(connector)));
    
    angCostFun = @(ang) -0.99+exp(1.47*ang);
    
    angCost1 = angCostFun(ang1);
    angCost2 = angCostFun(ang2);
    angCost = angCost1 + angCost2
    
    distCostFun = @(d) -0.99+exp(0.01*d);
    
    distCost = distCostFun(minDist)
    
    if minDist
    end
    cost = angCost * distCost;
end

% Finds the midpoint between closest points of two segments and returns
% angle cost between two segments, approximately connected and midpoint.
function cost = getTwoSegmentsAngleCostMidPoint(p1,p2, p3,p4)
    % find shortest distance
    allPoints = [p1; p2; p3; p4];
    distInds = [1 3; 1 4; 2 3; 2 4]; % distances to calculate
    
    dists = arrayfun(@(seg1Ind, seg2Ind) norm(allPoints(seg1Ind,:)-allPoints(seg2Ind,:)), distInds(:,1), distInds(:,2));
    
    [~,minInd] = min(dists);
    
    closestPointPairInds = distInds(minInd,:);
    
    % segment1 = (closest1,further1)
    closest1 = allPoints(closestPointPairInds(1),:);
    further1 = allPoints(3 - closestPointPairInds(1),:);
    
    % segment2 = (closest2,further2)
    closest2 = allPoints(closestPointPairInds(2),:);
    further2 = allPoints(7 - closestPointPairInds(2),:);
    
    midPoint = mean([closest1; closest2], 1);
    
    v1 = midPoint - further1;
    v2 = further2 - midPoint;
    
    ang = RunHumanDetector.angleTwoVectors(v1,v2);

    angCostFun = @(ang) -0.99+exp(1.47*ang);
    
    angCost = angCostFun(ang);
    cost = angCost;
end

% Calculate the width between line (p1,p2) and line (p3,p4) if vanishing
% point vanishPoint is known.
function width = estimateLaneDividerWidth(p1,p2, p3,p4, vanishPoint)
    % find farthest point for each line segment
    [maxDist1,line1PointInd] = max([norm(p1 - vanishPoint) norm(p2 - vanishPoint)]);
    [maxDist2,line2PointInd] = max([norm(p3 - vanishPoint) norm(p4 - vanishPoint)]);
    
    if line1PointInd == 1
        line1Point = p1;
    else
        line1Point = p2;
    end
    if line2PointInd == 1
        line2Point = p3;
    else
        line2Point = p4;
    end
    
    [maxDist,maxPointInd] = max([norm(line1Point - vanishPoint) norm(line2Point - vanishPoint)]);
    if maxPointInd == 1
        maxPoint1 = line1Point;
        otherDirPoint = line2Point;
    else
        maxPoint1 = line2Point;
        otherDirPoint = line1Point;
    end
    
    % offset otherDirPoint to distance maxDist
    dir1 = otherDirPoint - vanishPoint;
    dir1 = dir1 ./ norm(dir1);
    
    maxPoint2 = vanishPoint + dir1 .* maxDist;
    
    width = norm(maxPoint2 - maxPoint1);
end

end
end