% Detects human's body shape in the image.
classdef HumanDetector < handle

properties
  skinDetector;
  skinClassifierFun;
  distanceCompensator;
  labTransformation;
  v;
end

methods
    
function obj = HumanDetector(skinClassifierFun, waterClassifierFun, distanceCompensator) % constructor
    if ~exist('distanceCompensator', 'var')
        error('Argument "distanceCompensator" is not initialized');
    end
    
    if exist('skinClassifierFun', 'var')
        obj.skinClassifierFun = skinClassifierFun;
    else
        load('data/SkinClassifier_nn31.mat', 'net');
        obj.skinDetector = net;
    end
    
    if ~exist('waterClassifierFun', 'var')
        error('Argument "watClassifFun" is not initialized');
    end
    obj.v.waterClassifierFun = waterClassifierFun;
    
    obj.distanceCompensator = distanceCompensator;
    obj.labTransformation = makecform('srgb2lab');
end

% Parameter frameId is used in testing to identify camera image.
function BodyDescr = GetHumanBodies(this, frameId, image, waterMask, debug)
    if debug
        imshow(image), title('Original image');
    end
    
    % isolate body shapes
    
    %imageBody = obj.IsolateBodyShapes(image, debug);
    
    imageBody = utils.applyMask(image, ~waterMask);
    %imageBody = utils.PixelClassifier.applyAndGetImage(image, obj.v.watClassifFun, debug);
    
    if debug
        imshow(imageBody);
    end    
    
    imageInputGray = imageBody(:,:,1); % TODO: make gray?
    
    if debug
        imshow(imageInputGray);
    end
    
    % cut tenuous bridges between connected components
    % =1 dashed lines
    % =2 too match, some parts flying around
    narrowRad = 1;     
    sel=strel('disk',narrowRad,0);
    noTenuousBridges=imopen(imageInputGray, sel);
    
    if debug
        imshow(noTenuousBridges);
    end

    % find islands of pixels 
    connComp=bwconncomp(noTenuousBridges, 8);
    connCompProps=regionprops(connComp,'Area','MinorAxisLength','MajorAxisLength','Extent','Centroid');

    if debug
        imgBlobs = utils.drawRegionProps(noTenuousBridges, connComp, 120);
        imshow(imgBlobs);
    end
    
    % remove noise on a pixel level (too small/large components)
    % expect swimmer shape island to be in range
    shapeMinAreaPixels = 6; % (in pixels) any patch below is noise
    swimmerShapeAreaMax = 5000; % (in pixels) anything greater can't be a swimmer
    
    noiseIslands = [connCompProps(:).Area] < shapeMinAreaPixels;
    imageNoNoise = HumanDetector.removeIslands(connComp, noTenuousBridges, noiseIslands);

    if debug
        imshow(imageNoNoise), title('small islands removed (pixel level)');
    end

    noiseIslands = [connCompProps(:).Area] > swimmerShapeAreaMax;
    imageNoBigIslands = HumanDetector.removeIslands(connComp, imageNoNoise, noiseIslands);

    if debug
        imshow(imageNoBigIslands), title('large islands removed');
    end
    
    
    % [MinorAxisLength MajorAxisLength]
    % [37.6252 144.0715]=0.261 = human
    % [10.7585 242.3401]=0.044 = light reflection
    % remove alongated stripes created by lane markers and water 'blique' 
    noiseIslands = [connCompProps(:).MinorAxisLength]./[connCompProps(:).MajorAxisLength]<0.1;
    imageNoSticks = HumanDetector.removeIslands(connComp, imageNoBigIslands, noiseIslands);

    if debug
        imshow(imageNoSticks), title('minor/major axis');
    end
    
    % remove 'outline' objects
    noiseIslands = [connCompProps(:).Extent] < 0.1;
    imageNoOutlines = HumanDetector.removeIslands(connComp, imageNoSticks, noiseIslands);

    if debug
        imshow(imageNoOutlines), title('outline islands removed');
    end
    
    imageNoOutlines(imageNoOutlines > 0) = 255;

    % merge body parts (close blobs of similar color)
    % which can be disconnected by appearance of swimming clothes, lane markers etc.
    % TODO: gluing radius should depend on distance from camera (further objects
    % should be glued with smaller radius
    
    bodyApartMaxDistPix=20;
    maxBlobColorMergeDist = 15; % may be 15 for EMD impl and 27 for euclidian impl
    mergedBlobsMask = this.mergeBlobsWithSimilarColor(connComp, connCompProps, image, bodyApartMaxDistPix, maxBlobColorMergeDist, debug);
    
    imageMergedParts = bitor(imageNoOutlines, mergedBlobsMask);
    
    if debug
        imshow(imageMergedParts), title('merged body parts');
    end
        
    connComp = bwconncomp(imageMergedParts, 8);
    connCompProps = regionprops(connComp, 'Area','Centroid');
    
    if debug
        imgBlobs = utils.drawRegionProps(imageMergedParts, connComp, 120);
        imshow(imgBlobs);
    end
    
    % remove swimmer shapes of unfeasible area
    % depends: parts of shape must be glued
    imageFeasAreaShapes = imageMergedParts;
    
    blocksCount = connComp.NumObjects;
    if blocksCount > 0
        % 0.3 is too large - far away head is undetected 
        % eg. head 20 cm x 20 cm = 0.04
        bodyAreaMin = 0.04;
        bodyAreaMax = 2; % man of size 1m x 2m

        noiseIslands = zeros(1, blocksCount,'uint8');
        for i=1:blocksCount
            centroid = connCompProps(i).Centroid;
            centroidWorld = this.distanceCompensator.cameraToWorld(centroid);

            actualArea = connCompProps(i).Area;

            expectAreaMax = this.distanceCompensator.worldAreaToCamera(centroidWorld, bodyAreaMax);
            isNoise = actualArea > expectAreaMax;

            if ~isNoise
                expectAreaMim = this.distanceCompensator.worldAreaToCamera(centroidWorld, bodyAreaMin);
                isNoise = actualArea < expectAreaMim;
            end

            %isNoise = actualArea < expectAreaMax/100;
            noiseIslands(i) = isNoise;
        end
        noiseIslands = cast(noiseIslands,'logical'); % indices must be logical

        imageFeasAreaShapes = HumanDetector.removeIslands(connComp, imageFeasAreaShapes, noiseIslands);

        if debug
            imshow(imageFeasAreaShapes), title('shape area (world)');
        end
    end
    
    resultImage = imageFeasAreaShapes;
        
    if debug
        % take corresponding RGB image
        bodyNoIslandsMask=im2uint8(resultImage);
        bodyNoIslandsRGB=utils.applyMask(image, bodyNoIslandsMask);
        imshow(bodyNoIslandsRGB);
        %imwrite(imageBody, sprintf('../dinosaur/appear1/mvi3177_whaleMan_%s.png', utils.PW.timeStampNow))
    end

    % return info about swimmer's shapes
    connComp = bwconncomp(resultImage, 8); % TODO: 8 or 4?
    connCompProps = regionprops(connComp, 'BoundingBox','Centroid','FilledImage');
    resultCells = struct(DetectedBlob);
    resultCells(1) = []; % make 1x0 struct
    for i=1:length(connCompProps)
        props = connCompProps(i);
        imgFill = props.FilledImage;
        
        imgFillBoundaries = bwboundaries(imgFill,4,'noholes');
        imgOutlinePixelsLocal=imgFillBoundaries{1};
        
        bndBox = floor(props.BoundingBox);
        imgOutlinePixels = [imgOutlinePixelsLocal(:,1) + bndBox(2), imgOutlinePixelsLocal(:,2) + bndBox(1)];
        
        blob = DetectedBlob;
        blob.Id = int32(i);
        blob.Centroid = single(props.Centroid);
        blob.BoundingBox = single(props.BoundingBox);
        blob.OutlinePixels = int32(imgOutlinePixels);
        blob.FilledImage = imgFill;
        resultCells(i) = struct(blob);
    end
    BodyDescr = resultCells;
end

% Max distance from real center of the swimmer to corresponding center of detected shape.
function distError = shapeCentroidNoise(this)
    distError = 0.5;
end

function SetWeight(obj, weight)
   obj.Weight = weight;
end

end

methods (Access = public)
function bodyI = IsolateBodyShapes(obj,image, debug)
    % apply skin classifier to input image

    pixelTriples = reshape(image, size(image,1)*size(image,2), 3);

    if ~isempty(obj.skinClassifierFun)
        classifRes=obj.skinClassifierFun(double(pixelTriples));
    else
        pixelTriples = pixelTriples';
        classifRes=obj.skinDetector(pixelTriples);
    end

    if debug
        hist(classifRes);
    end

    % construct mask
    classifThrMask=classifRes > 0.8; % 0.03 % tune: skin threshold

    classifThr = classifRes;
    classifThr( classifThrMask)=1;
    classifThr(~classifThrMask)=0;
    classifThr=im2uint8(classifThr);

    bgMask = [];
    if ~isempty(obj.skinDetector)
        bgMask = repmat(classifThr,3,1);; % by rows
    else
        bgMask = repmat(classifThr,1,3); % by columns
    end

    % clear background
    pixelTriplesBody = bitand(pixelTriples, bgMask);

    if ~isempty(obj.skinDetector)
        pixelTriplesBody = pixelTriplesBody';
    end

    bodyI = reshape(pixelTriplesBody, size(image,1), size(image,2), 3);
    if debug
        imshow(bodyI)
    end
end

function mergeBlobsRecipe = mergeBlobsRecipeViaColorSimilarity(this, connComp, connCompPropsOrNull, imageRgb, maxDistBetweenBlobs, maxBlobColorMergeDist, debug)
    mergeBlobsRecipe = zeros(0,2,'int32');
    blobsCount = connComp.NumObjects;
    
    pixelsRgbByRow = reshape(imageRgb, [], 3);
    
    % learn GMM of colors for each blob
    mixGaussList = cell(1,blobsCount);
    for blobInd=1:blobsCount
        pixInds = connComp.PixelIdxList{blobInd};
        blobPixs = pixelsRgbByRow(pixInds, :);
        blobPixsLab = applycform(blobPixs, this.labTransformation);

        % limit number of clusters to be at least the number of pixels
        nclust = min([8 size(blobPixsLab,1)]);
        
        mixGauss = cv.EM('Nclusters', nclust, 'CovMatType', 'Spherical');
        mixGauss.train(blobPixsLab);
        mixGaussList{blobInd} = mixGauss;
    end

    for blobInd1=1:blobsCount

        blob1Center = [];
        if ~isempty(connCompPropsOrNull)
            blob1Center = connCompPropsOrNull(blobInd1).Centroid;
        end
        
        for blobInd2=blobInd1+1:blobsCount

            % skip distant blobs
            
            if ~isempty(connCompPropsOrNull)
                assert(~isempty(blob1Center));

                blob2Center = connCompPropsOrNull(blobInd2).Centroid;
                
                centrDist = norm(blob1Center - blob2Center);
                if centrDist > maxDistBetweenBlobs
                    continue;
                end
            end

            % skip distant blobs
            
            colorDist = utils.PixelClassifier.distanceTwoMixtureGaussiansEmd(mixGaussList{blobInd1}.Means, mixGaussList{blobInd1}.Weights,mixGaussList{blobInd2}.Means, mixGaussList{blobInd2}.Weights);
            
            if debug
                %fprintf('blobs[%d-%d] colorDist=%.2f\n', blobInd1,blobInd2,colorDist);
            end
            
            if colorDist > maxBlobColorMergeDist
                continue;
            end
            
            mergeBlobsRecipe(end+1,:) = [blobInd1, blobInd2];
        end
    end
end

% Merges blobs with similar color. Returns mask of merged blobs.
function mergedBlobsMask = mergeBlobsWithSimilarColor(this, connComp, connCompPropsOrNull, imageRgb, maxDistBetweenBlobs, maxBlobColorMergeDist, debug)
    mergeRecipe = this.mergeBlobsRecipeViaColorSimilarity(connComp, connCompPropsOrNull, imageRgb, maxDistBetweenBlobs, maxBlobColorMergeDist, debug);
    
    % merge masks
    mergedBlobsMask = false(size(imageRgb,1), size(imageRgb,2));
    for i=1:size(mergeRecipe,1)
        blobInd1 = mergeRecipe(i,1);
        blobInd2 = mergeRecipe(i,2);
        mask1 = utils.BlobsHelper.mergeBlobs(connComp, blobInd1, blobInd2);
        mergedBlobsMask = mergedBlobsMask | mask1;
    end
    
    mergedBlobsMask = im2uint8(mergedBlobsMask);
end

end

methods(Static)

function resultGray = removeIslands(connComps, imageGray, islandsToRemoveMask)
    resultGray = imageGray;
    
    if iscell(islandsToRemoveMask)
        islandsToRemoveMask = cell2mat(islandsToRemoveMask);
    end
    
    
    deadIslands=connComps.PixelIdxList(islandsToRemoveMask);
 
    for k=1:length(deadIslands)
        oneInds=deadIslands{k};
        
        islandMask = ind2sub(size(imageGray), oneInds);
        resultGray(islandMask)=0; % remove island's pixel
    end
end

end

end
