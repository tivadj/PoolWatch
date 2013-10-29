% Detects human's body shape in the image.
classdef HumanDetector < handle

properties
  skinDetector;
  skinClassifierFun;
  distanceCompensator;
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
end

function BodyDescr = GetHumanBodies(obj, image, debug)
    if debug
        imshow(image), title('Original image');
    end
    
    % isolate body shapes
    
    %imageBody = obj.IsolateBodyShapes(image, debug);
    
    waterMask = utils.PixelClassifier.applyToImage(image, obj.v.waterClassifierFun);
    imageBody = utils.applyMask(image, ~waterMask);
    %imageBody = utils.PixelClassifier.applyAndGetImage(image, obj.v.watClassifFun, debug);
    
    if debug
        imshow(imageBody);
    end    
    
%     sel=strel('disk',1,0);
%     bodyI2=imopen(imageBody, sel);
%     if debug
%         imshow(bodyI2);
%     end
%     
%     bodyGray=bodyI2(:,:,1);
%     if debug
%         imshow(bodyGray);
%     end
    
    imageInputGray = imageBody(:,:,1); % TODO: make gray?
    
    if debug
        imshow(imageInputGray);
    end
    
    % cut bridges between connected components
    % =1 dashed lines
    % =2 too match, some parts flying around
    narrowRad = 1;     
    sel=strel('disk',narrowRad,0);
    noNarrowBridges=imopen(imageInputGray, sel);
    
    if debug
        imshow(noNarrowBridges);
    end

    % find islands of pixels 
    connComps=bwconncomp(noNarrowBridges, 8);
    connCompProps=regionprops(connComps,'Area','MinorAxisLength','MajorAxisLength','Extent','Centroid');

    if debug
        noNarrowBridgesDebug1 = utils.drawRegionProps(noNarrowBridges, connComps, 120);
        imshow(noNarrowBridgesDebug1);
    end
    
    % remove too small or too big
    % expect swimmer shape island to be in range
    % 50 is too big; head of 40px is missed (16 frames avi, frame=11)
    % TODO: blob size should depend on 2D ortho position in the pool
    %swimmerShapeAreaMin = 50; % any patch below is noise
    
    swimmerShapeAreaMin = 6; % any patch below is noise
    swimmerShapeAreaMax = 5000; % anything greater can't be a swimmer
    
    %
    
    %noiseIslands = [bodyConnCompAreas(:).Area] < swimmerShapeAreaMin;
    blocksCount = length(connCompProps);
    noiseIslands = zeros(1, blocksCount,'uint8');
    for i=1:blocksCount
        centroid = connCompProps(i).Centroid;
        centroidWorld = CameraDistanceCompensator.cameraToWorld(obj.distanceCompensator, centroid);
        
        actualArea = connCompProps(i).Area;

        worldArea = 999; % unused
        expArea = CameraDistanceCompensator.worldAreaToCamera(obj.distanceCompensator, centroidWorld, worldArea);
        isNoise = actualArea < expArea/100;
        noiseIslands(i) = isNoise;
    end
    noiseIslands = cast(noiseIslands,'logical'); % indices must be logical
    
    imageNoNoise = HumanDetector.removeIslands(connComps, noNarrowBridges, noiseIslands);

    if debug
        imshow(imageNoNoise), title('small islands removed');
    end

    noiseIslands = [connCompProps(:).Area] > swimmerShapeAreaMax;
    imageNoBigIslands = HumanDetector.removeIslands(connComps, imageNoNoise, noiseIslands);

    if debug
        imshow(imageNoBigIslands), title('large islands removed');
    end
    
    
    % [MinorAxisLength MajorAxisLength]
    % [37.6252 144.0715]=0.261 = human
    % [10.7585 242.3401]=0.044 = light reflection
    % remove alongated stripes created by lane markers and water 'blique' 
    noiseIslands = [connCompProps(:).MinorAxisLength]./[connCompProps(:).MajorAxisLength]<0.1;
    imageNoSticks = HumanDetector.removeIslands(connComps, imageNoBigIslands, noiseIslands);

    if debug
        imshow(imageNoSticks), title('minor/major axis');
    end
    
    % remove 'outline' objects
    noiseIslands = [connCompProps(:).Extent] < 0.1;
    imageNoOutlines = HumanDetector.removeIslands(connComps, imageNoSticks, noiseIslands);

    if debug
        imshow(imageNoOutlines), title('outline islands removed');
    end
    
    % glue body parts
    % which can be disconnected by appearance of swimming clothes, lane markers etc.
    bodyApartMaxDist=20;
    sel=strel('disk',ceil(bodyApartMaxDist/2),0);
    imageGluedParts=imclose(imageNoOutlines, sel);
    imageGluedParts(imageGluedParts > 0) = 255; % make a BW mask

    if debug
        imshow(imageGluedParts), title('parts glued');
    end
    
    % gluing fix: gluing by closing may generate small islands; remove them
    connComps = bwconncomp(imageGluedParts, 8);
    connCompsProps = regionprops(connComps, 'Area','Centroid');
    
    smallIslands = [connCompsProps(:).Area] < swimmerShapeAreaMin;
    imageGluedPartsNoSmall = HumanDetector.removeIslands(connComps, imageGluedParts, smallIslands);

    if debug
        imshow(imageGluedPartsNoSmall), title('post glued parts: small islands removed');
    end
    
    % remove swimmer shapes of unfeasible area
    % depends: parts of shape must be glued
    imageFeasAreaShapes = imageGluedPartsNoSmall;

    centroidMat = reshape([connCompsProps(:).Centroid],2,[])';
    centroidCells = mat2cell(centroidMat,ones(size(centroidMat, 1),1),2);
    
    if ~isempty(centroidCells)
        %isFeas = @(centroid,area) CameraDistanceCompensator.isFeasibleArea(obj.distanceCompensator, centroid, area);
        %feasCmps = arrayfun(isFeas, centroidCells, [connCompsProps(:).Area]');
        
        blocksCount = length(centroidCells);
        feasCmps = zeros(1, blocksCount);
        for i=1:blocksCount
            actualArea = connCompsProps(i).Area;

            centroid = centroidMat(i,:);
            
            centroidWorld = CameraDistanceCompensator.cameraToWorld(obj.distanceCompensator, centroid);
            
            worldArea = 999; % unused
            expArea = CameraDistanceCompensator.worldAreaToCamera(obj.distanceCompensator, centroidWorld, worldArea);
            
            isFeasible = actualArea > expArea/4 && actualArea < expArea*4;
            feasCmps(i) = isFeasible;
        end

        noiseIslands = ~feasCmps;
        imageFeasAreaShapes = HumanDetector.removeIslands(connComps, imageGluedPartsNoSmall, noiseIslands);

        if debug
            imshow(imageFeasAreaShapes), title('unfeasible area islands removed');
        end
    end
    
    % remove swimmer shapes of unfeasible area: using orthogonal projection (Top) view
%     connComps = bwconncomp(imageFeasAreaShapes, 8);
%     connCompsProps = regionprops(connComps, 'Area','Centroid');
% 
%     desiredImageSize = [size(imageFeasAreaShapes,2), size(imageFeasAreaShapes,1)];
%     imageGluedPartsNoSmallTop = CameraDistanceCompensator.convertCameraImageToTopView(obj.distanceCompensator, imageFeasAreaShapes, desiredImageSize);
%     
%     if debug
%         imshow(imageGluedPartsNoSmallTop), title('top view');
%     end
%     
%     connCompsTop = bwconncomp(imageGluedPartsNoSmallTop, 8);
%     if connComps.NumObjects > 0 && connComps.NumObjects == connCompsTop.NumObjects
%         connCompsTopProps = regionprops(connCompsTop, 'Area','Centroid');
% 
%         % 
%         assert(connComps.NumObjects == connCompsTop.NumObjects, 'Original and Top images have different number of connected components');
%         topToSkewed = CameraDistanceCompensator.findComponentMap(obj.distanceCompensator,connComps,connCompsProps,connCompsTop,connCompsTopProps);
% 
%         %
%         swimmerShapeAreaMinM = 25*50/10000; % 25cm x 50cm
%         swimmerShapeAreaMaxM = 100*200/10000; % anything greater can't be a swimmer
% 
%         %
%         destImageSize = [size(imageGluedPartsNoSmallTop,1) size(imageGluedPartsNoSmallTop,2)];
%         areaM = CameraDistanceCompensator.scaleTopViewImageToWorldArea([connCompsTopProps(:).Area], destImageSize);
%         noiseIslands =  areaM < swimmerShapeAreaMinM;
%         noiseIslandsInds = find(noiseIslands);
%         noiseIslandsOriginInds = values(topToSkewed, num2cell(noiseIslandsInds));
% 
%         imageTopNoSmall = HumanDetector.removeIslands(connComps, imageFeasAreaShapes, noiseIslandsOriginInds);
% 
%         if debug
%             imshow(imageTopNoSmall), title('top-small islands removed');
%         end
% 
%         noiseIslands = areaM > swimmerShapeAreaMaxM;
%         noiseIslandsInds = find(noiseIslands);
%         noiseIslandsOriginInds = values(topToSkewed, num2cell(noiseIslandsInds));
% 
%         imageTopNoLarge = HumanDetector.removeIslands(connComps, imageTopNoSmall, noiseIslandsOriginInds);
% 
%         if debug
%             imshow(imageTopNoLarge), title('top-large islands removed');
%         end
%     else
%         imageTopNoLarge = imageFeasAreaShapes;
% 
%         if connComps.NumObjects > 0
%             fprintf('Warning: Number of connected components in skewed (%d) and topView (%d) does not match',connComps.NumObjects, connCompsTop.NumObjects);
%         end
%     end
% 
    resultImage = imageFeasAreaShapes;
        
    if debug
        % take corresponding RGB image
        bodyNoIslandsMask=im2uint8(resultImage);
        bodyNoIslandsRGB=utils.applyMask(image, bodyNoIslandsMask);
        imshow(bodyNoIslandsRGB);
    end

    % return info about swimmer's shapes
    connComps = bwconncomp(resultImage, 8); % TODO: 8 or 4?
    connCompsProps = regionprops(connComps, 'BoundingBox','Centroid','FilledImage');
    resultCells = struct('BoundingBox',[],'Centroid',[],'FilledImage',[]);
    resultCells(1) = []; % make 1x0 struct
    for i=1:length(connCompsProps)
        props = connCompsProps(i);
        imgFill = props.FilledImage;
        
        imgFillBoundaries = bwboundaries(imgFill,4,'noholes');
        imgOutlinePixelsLocal=imgFillBoundaries{1};
        
        bndBox = floor(props.BoundingBox);
        imgOutlinePixels = [imgOutlinePixelsLocal(:,1) + bndBox(2), imgOutlinePixelsLocal(:,2) + bndBox(1)];
        
        resultCells(i).BoundingBox = props.BoundingBox;
        resultCells(i).Centroid = props.Centroid;
        resultCells(i).OutlinePixels = imgOutlinePixels;
    end
    BodyDescr = resultCells;
end

function SetWeight(obj, weight)
   obj.Weight = weight;
end

end

methods (Access = private)
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
