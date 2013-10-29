classdef SkinClassifierStatics < handle

% Properties
%{
skinPixels
surfaceSkinPixels
nonSkinPixelsSample
trainInputBal
trainOutputBal
%}

methods(Static)
    
function obj = create()
    obj = utils.TypeErasedClass;
end

function populateSurfPixels(obj, ~)
    resFolder = fullfile(cd, 'data/SkinMarkup/SkinPatch');
    svgPattern = fullfile(resFolder, '*.svg');
    skinPixels = utils.getPixelsDistinct(svgPattern, false, '#00FF00'); % color=green
    obj.v.skinPixels = skinPixels;
    fprintf(1, 'skin pixels count=%d\n', length(obj.v.skinPixels));

    surfaceSkin = utils.getPixelsDistinct(svgPattern, false, '#FFFF00'); % color=yellow
    obj.v.surfaceSkinPixels = setdiff(surfaceSkin, obj.v.skinPixels, 'rows');
    fprintf(1, 'surface skin pixels count=%d\n', length(obj.v.surfaceSkinPixels));

    fprintf(1, 'loading non skin pixels\n');
    nonSkinPixelsAll = utils.getPixelsDistinct(fullfile('data/SkinMarkup/MVI_3177_0127_640x476_nonskinmask.svg'), true);
    
    % nonSkin = All - Skin - SurfaceSkin
    nonSkinPixels = nonSkinPixelsAll;
    nonSkinPixels = setdiff(nonSkinPixels, skinPixels, 'rows');
    nonSkinPixels = setdiff(nonSkinPixels, surfaceSkin, 'rows');
    obj.v.nonSkinPixels = nonSkinPixels;
end

function visualizePoints(obj,alph)
    if ~exist('alph', 'var')
        alph=1;
    end
    
    o = obj.v;
%     plot3(o.nonSkinPixels(:,1), o.nonSkinPixels(:,2), o.nonSkinPixels(:,3), 'b.')
%     hold on
%     plot3(o.surfaceSkinPixels(:,1), o.surfaceSkinPixels(:,2), o.surfaceSkinPixels(:,3), 'g.')
%     plot3(o.skinPixels(:,1), o.skinPixels(:,2), o.skinPixels(:,3), 'r.')
%     hold off
    
    plot3(o.skinPixels(:,1), o.skinPixels(:,2), o.skinPixels(:,3), 'r.')
    hold on
    plot3(o.surfaceSkinPixels(:,1), o.surfaceSkinPixels(:,2), o.surfaceSkinPixels(:,3), 'g.')
    plot3(o.nonSkinPixels(:,1), o.nonSkinPixels(:,2), o.nonSkinPixels(:,3), 'b.')
    hold off
end

function visualizePointsSample(obj,sampleSize)
    if ~exist('sampleSize', 'var')
        sampleSize=length(obj.v.trainInputBal);
    end
    
    o = obj.v;
    
    allPixels = [o.skinPixels; o.surfaceSkinPixels; o.nonSkinPixels];
    allPixelsGroup = [1+zeros(length(o.skinPixels),1); 2+zeros(length(o.surfaceSkinPixels),1); 3+zeros(length(o.nonSkinPixels),1)];
%     allPixelsColor = [repmat([1 0 0],length(o.skinPixels),1);...
%                       repmat([0 1 0],length(o.surfaceSkinPixels),1);...
%                       repmat([0 0 1],length(o.nonSkinPixels),1)];
    
    sampleInds = randperm(length(allPixels), sampleSize);
    pixelsSample = allPixels(sampleInds, :);
    pixelsGroup = allPixelsGroup(sampleInds, :);
    
    %scatter3(allPixels(:,1), allPixels(:,2), allPixels(:,3), 2, allPixelsColor);
    plot3(pixelsSample(pixelsGroup==1,1), pixelsSample(pixelsGroup==1,2), pixelsSample(pixelsGroup==1,3), 'r.')
    hold on
    plot3(pixelsSample(pixelsGroup==2,1), pixelsSample(pixelsGroup==2,2), pixelsSample(pixelsGroup==2,3), 'g.')
    plot3(pixelsSample(pixelsGroup==3,1), pixelsSample(pixelsGroup==3,2), pixelsSample(pixelsGroup==3,3), 'b.')
    hold off
end


function visualizeTrainingPoints(obj)
    o = obj.v;
    plot3(o.nonSkinPixelsForTraining(:,1), o.nonSkinPixelsForTraining(:,2), o.nonSkinPixelsForTraining(:,3), 'b.')
    hold on
    plot3(o.surfaceSkinPixelsForTraining(:,1), o.surfaceSkinPixelsForTraining(:,2), o.surfaceSkinPixelsForTraining(:,3), 'g.')
    plot3(o.skinPixelsForTraining(:,1), o.skinPixelsForTraining(:,2), o.skinPixelsForTraining(:,3), 'r.')
    hold off
end

function visualizePointsRgb(obj)
    o = obj.v;

    allCatPixels = [o.nonSkinPixels; o.surfaceSkinPixels; o.skinPixels];
    %allCatPixels = allCatPixels(randperm(length(allCatPixels), 5000), :);
    scatter3(allCatPixels(:,1), allCatPixels(:,2), allCatPixels(:,3), 3, double(allCatPixels)/255)
    
    %axis([1 255 1 255 1 255])
    xlabel('R');
    ylabel('G');
    zlabel('B');
end

function prepareTrainingData(obj)
    % get data from skin_classifier_get_skin_pixels_and_visualize
    obj.v.trainInputBal = [obj.v.surfaceSkinPixels; obj.v.skinPixels; obj.v.nonSkinPixels];
    obj.v.trainOutputBal = [...
         1+zeros(length(obj.v.surfaceSkinPixels), 1);...
         1+zeros(length(obj.v.skinPixels), 1);...
        -1+zeros(length(obj.v.nonSkinPixels), 1)];
    
    % shuffle training data
    trainShuffleInds=randperm(length(obj.v.trainOutputBal));
    obj.v.trainInputBal(:,:) = obj.v.trainInputBal(trainShuffleInds,:);
    obj.v.trainOutputBal(:,:) = obj.v.trainOutputBal(trainShuffleInds,:);
    
    % svm requires input points of double type
    obj.v.trainInputBal = double(obj.v.trainInputBal);
end

function prepareTrainingDataSmallAndBalanced(obj, bothSkinToNonSkinFactor)
    if ~exist('bothSkinToNonSkinFactor', 'var')
        bothSkinToNonSkinFactor = 10;
    end
    % get data from skin_classifier_get_skin_pixels_and_visualize
    groupLen = min([length(obj.v.skinPixels), length(obj.v.surfaceSkinPixels), length(obj.v.nonSkinPixels)]);
    
    % balance each group's size
%     skinPix = obj.v.skinPixels(randperm(length(obj.v.skinPixels), groupLen), :);
%     surfSkinPix = obj.v.surfaceSkinPixels(randperm(length(obj.v.surfaceSkinPixels), groupLen), :);
%     nonSkinPix = obj.v.nonSkinPixels(randperm(length(obj.v.nonSkinPixels), groupLen*2), :);

    skinPix = obj.v.skinPixels;
    surfSkinPix = obj.v.surfaceSkinPixels;
    skinBoth = length(skinPix) + length(surfSkinPix);
    nonSkinCount = min([skinBoth * bothSkinToNonSkinFactor, length(obj.v.nonSkinPixels)]);
    nonSkinPix = obj.v.nonSkinPixels(randperm(length(obj.v.nonSkinPixels), nonSkinCount), :);

    %
    
    obj.v.skinPixelsForTraining = skinPix;
    obj.v.surfaceSkinPixelsForTraining = surfSkinPix;
    obj.v.nonSkinPixelsForTraining = nonSkinPix;

    obj.v.trainInputBal = double([obj.v.skinPixelsForTraining; obj.v.surfaceSkinPixelsForTraining; obj.v.nonSkinPixelsForTraining]);
    obj.v.trainOutputBal = [...
         1+zeros(length(obj.v.skinPixels), 1);...
         1+zeros(length(obj.v.surfaceSkinPixels), 1);...
        -1+zeros(length(obj.v.nonSkinPixelsForTraining), 1)];

end

function prepareTrainingDataGatherNearBoundary(obj, gatherRadius)
    % how much pixels to gather
    takeCount = sum([length(obj.v.skinPixels) length(obj.v.surfaceSkinPixels)]);

    %
    skinOrSurfPix = double([obj.v.skinPixels; obj.v.surfaceSkinPixels]);
    skinOrSurfPix = sortrows(skinOrSurfPix, [1 2 3]);

    fprintf(1, 'randomly pick boundary non skin pixels\n');
    tic;
    boundSkinPixels =  SkinClassifierStatics.chooseBoundaryByDensity(skinOrSurfPix, obj.v.nonSkinPixels, gatherRadius, takeCount, 1000);
    fprintf(1, 'completed gathering non skin pixels in %d sec\n', toc);
    
%     fixed radius for all skin pixels    
%     pixIndToIncl = zeros(length(obj.v.nonSkinPixels),1,'int32');
%     fprintf(1, 'calculating distances...');
%     tic;
%     parfor nonSkinInd = 1:length(obj.v.nonSkinPixels)
%         nonSkinPix = obj.v.nonSkinPixels(nonSkinInd,:);
%         
%         % find nearest pixel
%         nonSkinPixMat = double(repmat(nonSkinPix, length(skinOrSurfPix),1));
%         dists = sqrt(sum((nonSkinPixMat - skinOrSurfPix).^2, 2));
%         [minDist,minDistInd] = min(dists, [], 1);
%         
%         if minDist < gatherRadius
%             pixIndToIncl(nonSkinInd,1) = nonSkinInd;
%         end
%     end
%     fprintf(1, ' took %d sec\n', toc);
% 
%     boundSkinPixels = obj.v.nonSkinPixels(pixIndToIncl > 0,:);
    
    %
    fprintf(1, 'found %d pixels on the boundary with R=%d\n',length(boundSkinPixels),gatherRadius);
    
    %takeCountFixed = min([takeCount, length(boundSkinPixels)]);
    takeCountFixed = takeCount;
    nonSkinPixelsToTake = boundSkinPixels(randperm(length(boundSkinPixels), takeCountFixed),:);

    obj.v.skinPixelsForTraining = obj.v.skinPixels;
    obj.v.surfaceSkinPixelsForTraining = obj.v.surfaceSkinPixels;
    obj.v.nonSkinPixelsForTraining = nonSkinPixelsToTake;

    obj.v.trainInputBal = double([obj.v.skinPixelsForTraining; obj.v.surfaceSkinPixelsForTraining; obj.v.nonSkinPixelsForTraining]);
    obj.v.trainOutputBal = [...
         1+zeros(length(obj.v.skinPixels), 1);...
         1+zeros(length(obj.v.surfaceSkinPixels), 1);...
        -1+zeros(length(obj.v.nonSkinPixelsForTraining), 1)];
end

function prepareTrainingDataAttenuateSurfSkinPixels(obj, gatherRadius, surfPixCount, nonSkinToOtherFactor)
    if ~exist('nonSkinToOtherFactor', 'var')
        nonSkinToOtherFactor = 1;
    end
    
    % thin out surface skin pixels
    surfSkinPixels =  chooseBoundaryByDensity(obj.v.skinPixels, obj.v.surfaceSkinPixels, gatherRadius, surfPixCount, 1000);

    % how many non skin pixels to gather
    takeCount = nonSkinToOtherFactor * sum([length(obj.v.skinPixels) length(surfSkinPixels)]);
    takeCountFix = min([takeCount length(obj.v.nonSkinPixels)]);

    %
    skinOrSurfPix = double([obj.v.skinPixels; surfSkinPixels]);
    skinOrSurfPix = sortrows(skinOrSurfPix, [1 2 3]);

    fprintf(1, 'randomly pick boundary non skin pixels\n');
    tic;
    boundaryNonSkinPixels =  chooseBoundaryByDensity(skinOrSurfPix, obj.v.nonSkinPixels, gatherRadius, takeCountFix, 1000);
    fprintf(1, 'completed gathering non skin pixels in %d sec\n', toc);

    obj.v.skinPixelsForTraining = obj.v.skinPixels;
    obj.v.surfaceSkinPixelsForTraining = surfSkinPixels;
    obj.v.nonSkinPixelsForTraining = boundaryNonSkinPixels;

    obj.v.trainInputBal = double([obj.v.skinPixelsForTraining; obj.v.surfaceSkinPixelsForTraining; obj.v.nonSkinPixelsForTraining]);
    obj.v.trainOutputBal = [...
         1+zeros(length(obj.v.skinPixelsForTraining), 1);...
         1+zeros(length(obj.v.surfaceSkinPixelsForTraining), 1);...
        -1+zeros(length(obj.v.nonSkinPixelsForTraining), 1)];
end

function prepareTrainingDataSurfSkinAsPadding(obj)
    obj.v.skinPixelsForTraining = obj.v.skinPixels;
    obj.v.surfaceSkinPixelsForTraining = zeros(0,3,class(obj.v.surfaceSkinPixels));
    obj.v.nonSkinPixelsForTraining = obj.v.nonSkinPixels;

    % surfSkin points are excluded from training set
    obj.v.trainInputBal = [obj.v.skinPixelsForTraining; obj.v.nonSkinPixels];
    obj.v.trainOutputBal = [...
         1+zeros(length(obj.v.skinPixels), 1);...
        -1+zeros(length(obj.v.nonSkinPixels), 1)];
    
    % svm requires input points of double type
    obj.v.trainInputBal = double(obj.v.trainInputBal);
end

function prepareTrainingDataMakeNonoverlappingHulls(obj,debug)
    if ~exist('debug', 'var')
        debug = false;
    end
    
    % inhull works with double types
    skinPixelsDbl = double(obj.v.skinPixels);
    surfaceSkinPixelsDbl = double(obj.v.surfaceSkinPixels);
    nonSkinPixelsDbl = double(obj.v.nonSkinPixels);
    
    hullTolerance = 0.2;

    % skinPixels hull: remove surfSkin and nonSkin pixels
    skinPixelsHullTriInds = convhulln(skinPixelsDbl);
    skinPixelsHull = skinPixelsDbl(unique(skinPixelsHullTriInds(:)),:);
    obj.v.skinPixelsHullTriInds = skinPixelsHullTriInds;
    obj.v.skinPixelsHullInside = skinPixelsDbl;
    obj.v.skinPixelsHullPixels = skinPixelsHull;
%     
    if debug
        hold on
        t1=trisurf(skinPixelsHullTriInds,skinPixelsDbl(:,1),skinPixelsDbl(:,2),skinPixelsDbl(:,3),'FaceColor','y')
        hold off
        alpha(t1, 0.2);
    end
    
    surfSkinPixInSkinHull = utils.inhull(surfaceSkinPixelsDbl, skinPixelsHull, [], hullTolerance);
    fprintf(1, 'skinPixels hull contains %d surfPixels\n', sum(surfSkinPixInSkinHull));
    surfaceSkinPixelsDbl = surfaceSkinPixelsDbl(~surfSkinPixInSkinHull,:);

    nonSkinPixInSkinHull = utils.inhull(nonSkinPixelsDbl, skinPixelsHull, [], hullTolerance);
    fprintf(1, 'skinPixels hull contains %d nonSkinPixels\n', sum(nonSkinPixInSkinHull));
    nonSkinPixelsDbl = nonSkinPixelsDbl(~nonSkinPixInSkinHull,:);


    % surfPixels hull: remove surfSkin and nonSkin pixels
    surfaceSkinPixelsDbl2 = double(obj.v.surfaceSkinPixels);
    surfaceSkinPixelsHullTriInds = convhulln(surfaceSkinPixelsDbl2);
    surfaceSkinPixelsHull = surfaceSkinPixelsDbl2(unique(surfaceSkinPixelsHullTriInds(:)),:);
    obj.v.surfaceSkinPixelsHullTriInds = surfaceSkinPixelsHullTriInds;
    obj.v.surfaceSkinPixelsHullInside = surfaceSkinPixelsDbl2;
    obj.v.surfaceSkinPixelsHullPixels = surfaceSkinPixelsHull;
    
    if debug
        SkinClassifierStatics.visualizePointsSample(obj,2000)
        hold on
        t2=trisurf(surfaceSkinPixelsHullTriInds,surfaceSkinPixelsDbl2(:,1),surfaceSkinPixelsDbl2(:,2),surfaceSkinPixelsDbl2(:,3),'FaceColor','c')
        hold off
        alpha(t2, 0.2);
    end

    
    nonSkinPixInSurfSkinHull = utils.inhull(nonSkinPixelsDbl, surfaceSkinPixelsHull, [], hullTolerance);
    fprintf(1, 'surfSkinPixels hull contains %d nonSkinPixels\n', sum(nonSkinPixInSurfSkinHull));
    nonSkinPixelsDbl = nonSkinPixelsDbl(~nonSkinPixInSurfSkinHull,:);
    
    % post condition checks
    in1 = utils.inhull(surfaceSkinPixelsDbl, skinPixelsHull, [], hullTolerance);
    assert(0 == sum(in1)); % should be 0
    in2 = utils.inhull(nonSkinPixelsDbl, skinPixelsHull, [], hullTolerance);
    assert(0 == sum(in2)); % should be 0
    in3 = utils.inhull(nonSkinPixelsDbl, surfaceSkinPixelsHull, [], hullTolerance);
    assert(0 == sum(in3)); % should be 0
    
    %
    
    obj.v.skinPixelsForTraining = skinPixelsDbl;
    obj.v.surfaceSkinPixelsForTraining = surfaceSkinPixelsDbl;
    obj.v.nonSkinPixelsForTraining = nonSkinPixelsDbl;

    % surfSkin points are excluded from training set
%     obj.v.trainInputBal = [obj.v.skinPixelsForTraining; obj.v.surfaceSkinPixelsForTraining; obj.v.nonSkinPixels];
%     obj.v.trainOutputBal = [...
%          1+zeros(length(obj.v.skinPixels), 1);...
%          1+zeros(length(obj.v.surfaceSkinPixelsForTraining), 1);...
%         -1+zeros(length(obj.v.nonSkinPixels), 1)];

    % case2: skinPixels~nonSkinPixels (surfSkinPixels are not used)
    obj.v.trainInputBal = [obj.v.skinPixelsForTraining; obj.v.nonSkinPixels];
    obj.v.trainOutputBal = [...
         1+zeros(length(obj.v.skinPixelsForTraining), 1);...
        -1+zeros(length(obj.v.nonSkinPixels), 1)];

    % svm requires input points of double type
    obj.v.trainInputBal = double(obj.v.trainInputBal);
end

% linearFactor determines how much to inflate the smaller hull. linearFactor=[0;1]
% linearFactor=0 take smaller hull
% linearFactor=1 take larger hull
function shiftedHull = inflateConvexHull(smallHullPixels,largerHullPixels, linearFactor, debug)
    if ~isa(smallHullPixels,'double')
        error('smallHullPixels must be of "double" type (required by delaunayTriangulation');
    end
    
    hull1Pixels = smallHullPixels;
    hull2Pixels = largerHullPixels;
    
    mapHullPixelsCost = zeros(length(hull1Pixels), length(hull2Pixels));
    
    dt = delaunayTriangulation(hull1Pixels);
    [Tfb, Xfb] = freeBoundary(dt);
    TR = triangulation(Tfb, Xfb);
    
    for row=1:length(hull1Pixels)
        q1=hull1Pixels(row,:);
        for col=1:length(hull2Pixels)
            p=hull2Pixels(col,:);

            %dist1 = pdist([q1; q2],'euclidean');
            n1 = vertexNormal(TR, row);

            cosASign = dot(p-q1, n1);
            if (cosASign <= 0)
                dist1 = 9999; % normal points in opposite direction
            else
                dist1 = norm(cross(p-q1, n1));
            end
            mapHullPixelsCost(row,col) = dist1;
        end
    end
    
    % shift internal convex hull by T in the direction of external hull
    shiftedHull = hull1Pixels;

    unassignCost = 10;
    [assignment, unassignedHull1, unassignedHull2] = assignDetectionsToTracks(mapHullPixelsCost, unassignCost);
    
    if debug
        fprintf(1, 'unassigned %d and %d\n', length(unassignedHull1), length(unassignedHull2));
    end

    for i=1:size(assignment,1)
        hull1PixInd = assignment(i,1);
        pix = hull1Pixels(hull1PixInd,:) + linearFactor*(hull2Pixels(assignment(i,2),:) - hull1Pixels(hull1PixInd,:));
        shiftedHull(hull1PixInd,:) = pix;
    end        
end

function findSkinPixelsConvexHullWithMinError(obj, debug)
    if debug
        SkinClassifierStatics.visualizePointsSample(obj,2000)
    end
    
    % pdist2 doesn't allow two sets of different size
    %hull1Inds = sort(unique(obj.v.skinPixelsHullTriInds(:)));
    %hull1obj.v.skinPixelsHullInside(hull1Inds,:);
    
    hull1Pixels = obj.v.skinPixelsHullPixels;
    hull2Pixels = obj.v.surfaceSkinPixelsHullPixels;
    mapHullPixelsCost = zeros(length(hull1Pixels), length(hull2Pixels));
    
    dt = delaunayTriangulation(hull1Pixels);
    [Tfb, Xfb] = freeBoundary(dt);
    TR = triangulation(Tfb, Xfb);
    
    for row=1:length(hull1Pixels)
        q1=hull1Pixels(row,:);
        for col=1:length(hull2Pixels)
            p=hull2Pixels(col,:);

            %dist1 = pdist([q1; q2],'euclidean');
            n1 = vertexNormal(TR, row);

            cosASign = dot(p-q1, n1);
            if (cosASign <= 0)
                dist1 = 9999; % normal points in opposite direction
            else
                dist1 = norm(cross(p-q1, n1));
            end
            mapHullPixelsCost(row,col) = dist1;
        end
    end
    
    unassignCost = 10;
    perfHist=[];
    %for t=[0.3 0.7]
    %for t=linspace(0,1,10)
    for t=[1]
        % shift internal convex hull by T in the direction of external hull
        shiftedHull = hull1Pixels;
        [assignment, unassignedHull1, unassignedHull2] = assignDetectionsToTracks(mapHullPixelsCost, unassignCost);
        fprintf(1, 'unassigned %d and %d\n', length(unassignedHull1), length(unassignedHull2));
        
        for i=1:size(assignment,1)
            hull1PixInd = assignment(i,1);
            pix = hull1Pixels(hull1PixInd,:) + t*(hull2Pixels(assignment(i,2),:) - hull1Pixels(hull1PixInd,:));
            shiftedHull(hull1PixInd,:) = pix;
        end
        
        % convex hull may be distorted; construct new one
        hullTriInds = convhulln(shiftedHull);
        
%         hold on
%         surf1=trisurf(hullTriInds,shiftedHull(:,1),shiftedHull(:,2),shiftedHull(:,3),'FaceColor','y');
%         hold off
%         alpha(surf1, 0.3);
%         drawnow
        
        if debug
            figure
        end
        
        pixelClassif = @(X) utils.inhull(X, shiftedHull, [], 0.2);
        %obj.v.skinHullClassif = ConvexHullSkinClassifier(obj,shiftedHull, hullTriInds);
        obj.v.skinHullClassifHullPoints = shiftedHull;
        obj.v.skinHullClassifHullTriInds = hullTriInds;

        if debug
            SkinClassifierStatics.testOnImage(obj,fullfile('data/MVI_3177_0127_640x476.png'), 1000, 0.1, pixelClassif);
            title(sprintf('t=%d\n', t));
        end
        
        % check performance for this hull
%         trainData = [obj.v.skinPixels; obj.v.surfaceSkinPixels; obj.v.nonSkinPixels];
%         expectOut = [1 + zeros(length(obj.v.skinPixels),1);...
%                      1 + zeros(length(obj.v.surfaceSkinPixels),1);...
%                      0 + zeros(length(obj.v.nonSkinPixels),1)];
        
        hullToler = 0.2;
        in1 = utils.inhull(double(obj.v.skinPixels), shiftedHull, hullTriInds, hullToler);
        errs1 = sum(in1 ~= ones(length(obj.v.skinPixels),1));
        in2 = utils.inhull(double(obj.v.surfaceSkinPixels), shiftedHull, hullTriInds, hullToler);
        errs2 = sum(in2 ~= ones(length(obj.v.surfaceSkinPixels),1));
        in3 = utils.inhull(double(obj.v.nonSkinPixels), shiftedHull, hullTriInds, hullToler);
        errs3 = sum(in3 ~= zeros(length(obj.v.nonSkinPixels),1));
        perf = (errs1+errs2+errs3) / sum([length(obj.v.skinPixels) length(obj.v.surfaceSkinPixels) length(obj.v.nonSkinPixels)]);
        perfHist = [perfHist; t perf errs1 errs2 errs3];
    end
    if debug
        figure, plot(perfHist(:,1), perfHist(:,2));
    end
end

function trimTrainingData(obj, trainSize)
    trainSizeFix = min([trainSize length(obj.v.trainInputBal)]);
    
    sampInds = randperm(length(obj.v.trainInputBal), trainSizeFix);
    obj.v.trainInputBal = obj.v.trainInputBal(sampInds, :);
    obj.v.trainOutputBal = obj.v.trainOutputBal(sampInds, :);
end

function shuffleTrainingData(obj)
    % shuffle training data
    trainShuffleInds=randperm(length(obj.v.trainOutputBal));
    obj.v.trainInputBal(:,:) = obj.v.trainInputBal(trainShuffleInds,:);
    obj.v.trainOutputBal(:,:) = obj.v.trainOutputBal(trainShuffleInds,:);
end

function perf = trainSurfSkinSvm(obj, svmClassifyBatchSize, kktviolationlevel)
    if ~exist('kktviolationlevel', 'var')
        kktviolationlevel = 0;
    end
    
    % SVM
    opts=statset('Display','iter', 'MaxIter', 15000); % display training log
    % observation per row
    %'kernel_function','rbf',
    %'rbf_sigma', 50,
    %'kernel_function','mlp',...
    %'kktviolationlevel', 0.5,
    obj.v.skinClassif=svmtrain(obj.v.trainInputBal, obj.v.trainOutputBal, 'kernel_function','rbf','rbf_sigma', 0.4, 'kktviolationlevel', kktviolationlevel, 'method','SMO', 'options', opts);
    %obj.skinClassif=svmtrain(obj.trainInputBal, obj.trainOutputBal, 'kernel_function','linear', 'options', opts);
    
    % mlp
    %obj.v.skinClassif=svmtrain(obj.v.trainInputBal, obj.v.trainOutputBal, 'kernel_function','mlp','kktviolationlevel', 0.3,'kernelcachelimit',10000, 'options', opts);
    
    % calc performance
    trainSimRes = utils.SvmClassifyHelper(obj.v.skinClassif, obj.v.trainInputBal, svmClassifyBatchSize);
    err = sum(trainSimRes ~= obj.v.trainOutputBal);
    perf = err / length(trainSimRes);
end

function trainSurfSkinNet(obj)
    % 2-layer NN
    obj.v.net=feedforwardnet([3 3]); % trainlm
    %net=feedforwardnet([5],'traingda');
    %n1.trainParam.max_perf_inc = 1.004; % if gradient ratio is greater then learning rate is dicreased
    %net.trainParam.lr_dec = 0.9; % decrease factor
    %view(net);
    [obj.v.net,tr]=train(obj.v.net, obj.v.trainInputBal', obj.v.trainOutputBal');

    % check classifier
    trainSimRes = obj.v.net(obj.v.trainInputBal')';
    err = sum((trainSimRes - obj.v.trainOutputBal).^2);
    fprintf('training perf=%f (errors %d of %d)\n', err/length(obj.v.trainOutputBal), err, length(obj.v.trainOutputBal));
end

function checkSvmOrNetTraining(obj, svmClassifyBatchSize)
    o = obj.v;
    
    %trainIn = o.trainInputBal;
    trainIn = [o.skinPixels; o.surfaceSkinPixels; o.nonSkinPixels];
    trainOut = [1+zeros(length(o.skinPixels),1),1+zeros(length(o.surfaceSkinPixels),1),-1+zeros(length(o.nonSkinPixels),1)];
    
    % check classifier
    % !net
    trainSimRes = utils.SvmClassifyHelper(o.skinClassif, trainIn, svmClassifyBatchSize);
    %trainSimRes = o.net(o.trainInputBal')';
    err = sum((trainSimRes - trainOut).^2);
    fprintf('train err=%d of %d\n', err, length(trainOut));
    
    correctInd = trainSimRes == trainOut;
    
    skinPixelsDet = trainIn(correctInd & (trainOut == 1),:);
    nonSkinPixelsDet = trainIn(correctInd & (trainOut == -1),:);

    skinPixelsUndet = trainIn(~correctInd & (trainOut == 1),:);
    nonSkinPixelsUndet = trainIn(~correctInd & (trainOut == -1),:);

    % visualize errors in learning
    plot3(nonSkinPixelsDet(:,1), nonSkinPixelsDet(:,2), nonSkinPixelsDet(:,3), 'b.')
    hold on
    plot3(skinPixelsDet(:,1), skinPixelsDet(:,2), skinPixelsDet(:,3), 'r.')
    
    plot3(nonSkinPixelsUndet(:,1), nonSkinPixelsUndet(:,2), nonSkinPixelsUndet(:,3), 'b.')
    plot3(nonSkinPixelsUndet(:,1), nonSkinPixelsUndet(:,2), nonSkinPixelsUndet(:,3), 'mo')
    
    plot3(skinPixelsUndet(:,1), skinPixelsUndet(:,2), skinPixelsUndet(:,3), 'r.')
    plot3(skinPixelsUndet(:,1), skinPixelsUndet(:,2), skinPixelsUndet(:,3), 'ko')
    hold off
end

function visualizeSkinConvexHull(obj, seedPointsCount, svmClassifyBatchSize)
%     cubeSide=255;
%     pixs = zeros(cubeSide*cubeSide*cubeSide, 3, 'uint8');
%     for r=1:cubeSide
%         for g=1:cubeSide
%             for b=1:cubeSide
%                 index = (r-1)*cubeSide*cubeSide+ (g-1)*cubeSide + (b-1) + 1;
%                 pixs(index,:) = [r g b];
%             end
%         end
%     end
    fprintf(1, 'generating random points\n');
    pixs = unique(floor(255*rand(seedPointsCount, 3)),'rows');
    %!net
    classifRes = utils.SvmClassifyHelper(obj.v.skinClassif, double(pixs), svmClassifyBatchSize);
    %classifRes = obj.v.net(double(pixs'))';
    %pixsDet = pixs(classifRes == 1, :);
    cmpThresh = 0.5; % net
    pixsDet = pixs(classifRes > cmpThresh, :);
    %scatter3(pixsDet(:,1), pixsDet(:,2), pixsDet(:,3), '.');
    
    fprintf(1, 'making Delaunay triangulation\n');
    DT = DelaunayTri(pixsDet);
    %tetramesh(DT);
    
    fprintf(1, 'making convex hull\n');
    hullFacets = convexHull(DT);
    trisurf(hullFacets,DT.X(:,1),DT.X(:,2),DT.X(:,3),'FaceColor','c')
end

function analyzeSurfSkinPoints(obj)
    figure
    % show surface skin pixels in the low corner of cube
    lowCorn(:,:) = obj.v.surfaceSkinPixelsAll(obj.v.surfaceSkinPixelsAll(:,3)<120, :); % blue < 120
    %lowCorn(:,:) = obj.v.surfaceSkinPixels; % blue < 120
    %lowCorn = sortrows(lowCorn);
    %lowCorn = lowCorn(randperm(length(lowCorn)),:);
    
%     % show them in order of decreased surfSkin classification incre
%     lowCornSurfSkin = obj.v.net(lowCorn')';
%     [lowCornSurfSkinSorted,skinIndex] = sortrows(lowCornSurfSkin);
%     lowCorn = lowCorn(skinIndex,:);
%     
     dim = floor(sqrt(length(lowCorn)));
     imLowCorn = reshape(lowCorn(1:(dim*dim),:), dim, dim, 3);
     figure, imshow(imLowCorn)
    
%     o = obj.v;
%     plot3(o.nonSkinPixelsSample(:,1), o.nonSkinPixelsSample(:,2), o.nonSkinPixelsSample(:,3), 'b.')
%     hold on
%     plot3(lowCorn(:,1), lowCorn(:,2), lowCorn(:,3), 'g.')
%     plot3(o.skinPixels(:,1), o.skinPixels(:,2), o.skinPixels(:,3), 'r.')
%     hold off
%     
%     %axis([1 255 1 255 1 255])
%     xlabel('R');
%     ylabel('G');
%     zlabel('B');
end

function projectOn2D(obj)
    % limit size because of out of memory error
    trainSize = min(10000, length(obj.v.trainInputBal));
    pixels=obj.v.trainInputBal(randperm(length(obj.v.trainInputBal),trainSize), :);
    [U,S,V] = svd(pixels);
    obj.v.svdV = V;
    pixelsTest = U*S*V';
    delta = pixelsTest-pixels;
    err = sum(delta(:).^2); % should be 0
    
    r=2;
    
    % 2d projected pixels
    skinPix=double(obj.v.skinPixelsForTraining) * V(:,1:r);
    surfSkinPix=double(obj.v.surfaceSkinPixelsForTraining) * V(:,1:r);
    nonSkinPix=double(obj.v.nonSkinPixelsForTraining) * V(:,1:r);

    % draw 2d
    figure,plot(nonSkinPix(:,1), nonSkinPix(:,2), 'b.')
    title('svd axes 1-2')    
    hold on
    plot(surfSkinPix(:,1), surfSkinPix(:,2), 'g.')
    plot(skinPix(:,1), skinPix(:,2), 'r.')
    hold off

    % 2d projected pixels: secondary
    skinPix2=double(obj.v.skinPixelsForTraining) * V(:,[1 3]);
    surfSkinPix2=double(obj.v.surfaceSkinPixelsForTraining) * V(:,[1 3]);
    nonSkinPix2=double(obj.v.nonSkinPixelsForTraining) * V(:,[1 3]);

    % draw 2d
    figure,plot(nonSkinPix2(:,1), nonSkinPix2(:,2), 'b.')
    title('svd axes 1-3');
    hold on
    plot(surfSkinPix2(:,1), surfSkinPix2(:,2), 'g.')
    plot(skinPix2(:,1), skinPix2(:,2), 'r.')
    hold off
  
    skinPixInt = int32(skinPix);
    surfSkinPixInt = int32(surfSkinPix);
    nonSkinPixInt = int32(nonSkinPix);
    
    allPixInt = [skinPixInt; surfSkinPixInt; nonSkinPixInt];
    offsetXY = min(allPixInt);
    upRight = max(allPixInt);
    imgSize = circshift(upRight + int32([1 1]) - offsetXY, [0 1]); % XY coords to matrix
    skinMapImg = zeros(imgSize,  'uint8');
    offset = - offsetXY + int32([1 1]); % 1-based indices
    
    skinPixInt = skinPixInt + repmat(offset, length(skinPixInt), 1);
    surfSkinPixInt = surfSkinPixInt + repmat(offset, length(surfSkinPixInt), 1);
    nonSkinPixInt = nonSkinPixInt + repmat(offset, length(nonSkinPixInt), 1);

    skinMapImg(sub2ind(size(skinMapImg), nonSkinPixInt(:,2), nonSkinPixInt(:,1))) = 3;
    skinMapImg(sub2ind(size(skinMapImg), surfSkinPixInt(:,2), surfSkinPixInt(:,1))) = 2;
    skinMapImg(sub2ind(size(skinMapImg), skinPixInt(:,2), skinPixInt(:,1))) = 1;
    obj.v.skinMapImg = skinMapImg;
    
    imshow(skinMapImg, [0 0 0; 1 0 0; 0 1 0; 0 0 1]);
    
    % make random order of pixels to prevent bias over skin color types
    allPix = [skinPixInt, 1 + zeros(length(skinPixInt),1);...
              surfSkinPixInt, 2 + zeros(length(surfSkinPixInt),1);...
              nonSkinPixInt, 3 + zeros(length(nonSkinPixInt),1)];
          
    allPix(:,:) = allPix(randperm(length(allPix)), :);
    skinMapImgRnd = zeros(imgSize,'uint8');
    skinMapImgRnd(sub2ind(size(skinMapImgRnd), allPix(:,2), allPix(:,1))) = allPix(:,3);
    figure, imshow(skinMapImgRnd, [0 0 0; 1 0 0; 0 1 0; 0 0 1]);
    obj.v.skinMapImgRnd = skinMapImgRnd;
    
    %
    imgSkinRnd=medfilt2(skinMapImgRnd,[7 7]);
    imshow(imgSkinRnd,[0 0 0; 1 0 0; 0 1 0; 0 0 1])

    % svm
    skinOtherOutput = [...
         1 + zeros(length(skinPixInt),1);...
         1 + zeros(length(surfSkinPixInt),1);...
        -1 + zeros(length(nonSkinPixInt),1)];

    opts=statset('Display','iter'); % display training log
    allPixDouble=[skinPix; surfSkinPix; nonSkinPix];
    obj.v.skin2DClassif=svmtrain(allPixDouble, skinOtherOutput, 'kernel_function','rbf','rbf_sigma', 1,'showplot',true,'kktviolationlevel',0.3,'options', opts);
    
    % check SVM for skin only classifier
    % learn the same for
    % get area boundaries

%     % V(:,1) = OX
%     % V(:,2) = OY
%     pixels2 = pixels*V(:,1:r);
%     % invert OX
%     %pixels2(:,1) = -pixels2(:,1);
%     hold on
%     scatter(pixels2(:,1),pixels2(:,2),5, double(pixels)/255)
%     hold off
end

function testOnImage(obj, imagePath, classifThresh, svmClassifyBatchSize, pixelClassif)
    if ~exist('pixelClassif', 'var')
        pixelClassif = [];
    end
    I = imread(imagePath);
    
    fprintf(1, 'simulating classifier\n');
    % apply classification result to input image
    pixelTriples=reshape(I, size(I,1)*size(I,2), 3);
    pixelTriplesDbl = double(pixelTriples);
    % ! net
    if ~isempty(pixelClassif)
        classifRes = pixelClassif(pixelTriplesDbl);
    else
    %classifRes = utils.inhull(pixelTriplesDbl, obj.v.skinPixelsHullPixels, [], 0.2);
    classifRes = utils.inhull(pixelTriplesDbl, obj.v.surfaceSkinPixelsHullPixels, [], 0.2);
    %classifRes = utils.SvmClassifyHelper(obj.v.skinClassif, pixelTriplesDbl, svmClassifyBatchSize);
    %classifRes = obj.v.net(double(pixelTriples'))';
    end
    hist(classifRes);
    
    classifThresh=0.5; % 0.03
    classifMaskOnes=classifRes > classifThresh;
    
    classifMask = classifRes;
    classifMask( classifMaskOnes)=1;
    classifMask(~classifMaskOnes)=0;
    classifMask=im2uint8(classifMask);
    pixelTriples = bitand(pixelTriples, cat(2, classifMask, classifMask, classifMask)); % clear background
    imageSkin = reshape(pixelTriples, size(I,1), size(I,2), 3);
    imshow(imageSkin)
end

function testOnImage2DClassifier(obj, imagePath, classifThresh)
    I = imread(imagePath);
    
    fprintf(1, 'simulating classifier\n');
    % apply classification result to input image
    pixelTriples=reshape(I, size(I,1)*size(I,2), 3);
    
    pixelsXY = double(pixelTriples)*obj.v.svdV(:,1:2);
    
    %classif2DRes = utils.SvmClassifyHelper(obj.v.skin2DClassif, pixelsXY, 1000);
    classif2DRes = obj.v.classif2DRes;
    hist(classif2DRes);
    
    classifThresh=0.5; % 0.03
    classifMaskOnes=classif2DRes > classifThresh;
    
    classifMask = classif2DRes;
    classifMask( classifMaskOnes)=1;
    classifMask(~classifMaskOnes)=0;
    classifMask=im2uint8(classifMask);
    pixelTriples = bitand(pixelTriples, cat(2, classifMask, classifMask, classifMask)); % clear background
    imageSkin = reshape(pixelTriples, size(I,1), size(I,2), 3);
    imshow(imageSkin)
end

function Main(obj)
    svmClassifyBatchSize = 1000;
%     figure(1);
%     SkinClassifierStatics.populateSurfPixels(obj,1);
    SkinClassifierStatics.prepareTrainingDataSmallAndBalanced(obj,1000);
    %SkinClassifierStatics.prepareTrainingData(obj);
    SkinClassifierStatics.visualizePoints(obj);
    
    figure(2);
    SkinClassifierStatics.trainSurfSkinSvm(obj, svmClassifyBatchSize);
    SkinClassifierStatics.checkSvmOrNetTraining(obj, svmClassifyBatchSize);

    figure(1);
    hold on;
    SkinClassifierStatics.visualizeSkinConvexHull(obj, 500000, svmClassifyBatchSize);
    hold off;

    figure(2);
    SkinClassifierStatics.testOnImage(obj,fullfile('data/MVI_3177_0127_640x480.png'), 0.1, svmClassifyBatchSize);
end

function analyzeProportionsOfSkinColors(obj)
    svmClassifyBatchSize = 1000;
    bothSkinToNonSkinFactorArray = 1:10;
    perfHist = [];
    for i=1:length(bothSkinToNonSkinFactorArray)
        bothSkinToNonSkinFactor = bothSkinToNonSkinFactorArray(i);
        fprintf(1, 'bothSkinToNonSkinFactor=%d\n', bothSkinToNonSkinFactor);
        SkinClassifierStatics.prepareTrainingDataSmallAndBalanced(obj,999000,bothSkinToNonSkinFactor);

        try
            perf = SkinClassifierStatics.trainSurfSkinSvm(obj, svmClassifyBatchSize);
        catch err
            perf = -1;
        end
        perfHist(end+1,1:2) = [bothSkinToNonSkinFactor perf];
        title1=sprintf('bothSkinToNonSkinFactor=%d perf=%d\n', bothSkinToNonSkinFactor, perf);
        fprintf(1, title1);
        
        if perf ~= -1
            figure,SkinClassifierStatics.testOnImage(obj,fullfile('data/MVI_3177_0127_640x480.png'), 0.1, svmClassifyBatchSize);
            title(title1);
        end

    end
    
    figure, plot(perfHist(:,1), perfHist(:,2));
end

function testSurfSkinAttenuationOnSvmPerf(obj)
    svmClassifyBatchSize = 1000;
    perfHist = [];
    %for attLevel = linspace(0,1,10)
    for attLevel = [1]
    %for kktviolationlevel=linspace(0,1,10)
    for kktviolationlevel=[0]
    for nonSkinToOtherFactor=[10]
    %for nonSkinToOtherFactor=[1]
    %for trimTrainSizeFactor=linspace(0.1, 1, 10)
    for trimTrainSizeFactor=[1]
        fprintf(1, 'trimTrainSizeFactor=%d\n', trimTrainSizeFactor);
        surfPixCount = floor(length(obj.v.skinPixels) * attLevel);
        %SkinClassifierStatics.prepareTrainingDataAttenuateSurfSkinPixels(obj, 30, surfPixCount,nonSkinToOtherFactor);
        SkinClassifierStatics.prepareTrainingDataSmallAndBalanced(obj, nonSkinToOtherFactor);
        
        trainSize=floor(trimTrainSizeFactor * length(obj.v.trainInputBal));
        SkinClassifierStatics.trimTrainingData(obj, trainSize);
        
        try
            perf = SkinClassifierStatics.trainSurfSkinSvm(obj, svmClassifyBatchSize,kktviolationlevel);
        catch err
            getReport(err)
            perf = -1;
        end
        perfHist = [perfHist; nonSkinToOtherFactor perf];
        fprintf(1, 'nonSkinToOtherFactor=%d perf=%d\n', nonSkinToOtherFactor, perf);
        
        if perf ~= -1
            figure;
            SkinClassifierStatics.testOnImage(obj,fullfile('data/MVI_3177_0127_640x480.png'), 0.1, svmClassifyBatchSize);
            title(sprintf('trainSize=%d perf=%d\n', trainSize, perf));
        end
    end
    end
    end
    end
    obj.v.perfHist = perfHist;
    plot(perfHist(:,1), perfHist(:,2));
end

function testConstructSkinHullClassifier(obj)
    %SkinClassifierStatics.populateSurfPixels(obj);
    %SkinClassifierStatics.prepareTrainingDataMakeNonoverlappingHulls(obj);
    SkinClassifierStatics.findSkinPixelsConvexHullWithMinError(obj);
    
    % result=[obj.v.skinHullClassifHullPoints obj.v.skinHullClassifHullTriInds];
end


% filter image with given skin classifier
function skinImage = applySkinClassifierToImage(image, skinClassifFun)
    pixelTriples=reshape(image, size(image,1)*size(image,2), 3);
    
    pixelTriplesDbl = double(pixelTriples);
    classifRes=skinClassifFun(pixelTriplesDbl);
    %hist(classifRes);
    
    classifResOne=classifRes > 0.9; % 0.03
    classifRes( classifResOne)=1;
    classifRes(~classifResOne)=0;
    classifRes=im2uint8(classifRes);
    pixelTriples = bitand(pixelTriples, repmat(classifRes,1,3)); % clear background
    skinImage = reshape(pixelTriples, size(image,1), size(image,2), 3);
end

function videoUpd = applySkinClassifierToVideo(videoPath, skinClassifFun)
    obj = mmreader(videoPath);
    frameRate = get(obj,'FrameRate')
    procFrames=obj.NumberOfFrames;
    %framesToTake=1:30:procFrames;
    framesToTake=1:procFrames;
    %framesToTake=1;
    framesCount=length(framesToTake);
    videoUpd=zeros(obj.Height, obj.Width, floor(obj.BitsPerPixel/8), framesCount, 'uint8');

    % find bodies in video
    parfor i = 1:length(framesToTake)
        num=framesToTake(i);
        fprintf(1, 'processing frame %d', num);

        image = read(obj, num);
        skinImg = SkinClassifierStatics.applySkinClassifierToImage(image, skinClassifFun);
        %imshow(image)
        videoUpd(:,:,:,i) = skinImg;
    end
    
    %{
    writerObj = VideoWriter('output/mvi3177_skinConvexHull_int_to_ext_t1_0.avi');
    writerObj.open();
    writerObj.writeVideo(videoUpd);
    writerObj.close();
    %}
end

end
end
