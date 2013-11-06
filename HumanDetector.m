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
    
    % glue body parts
    % which can be disconnected by appearance of swimming clothes, lane markers etc.
    % TODO: gluing radius should depend on distance from camera (further objects
    % should be glued with smaller radius
    bodyApartMaxDist=20;
    sel=strel('disk',ceil(bodyApartMaxDist/2),0);
    imageGluedParts=imclose(imageNoOutlines, sel);
    imageGluedParts(imageGluedParts > 0) = 255; % make a BW mask

    if debug
        imshow(imageGluedParts), title('parts glued');
    end
    
    % gluing fix: gluing by closing may generate small islands; remove them
    connComp = bwconncomp(imageGluedParts, 8);
    connCompProps = regionprops(connComp, 'Area','Centroid');
    
    if debug
        imgBlobs = utils.drawRegionProps(imageGluedParts, connComp, 120);
        imshow(imgBlobs);
    end
    
    smallIslands = [connCompProps(:).Area] < shapeMinAreaPixels;
    imageGluedPartsNoSmall = HumanDetector.removeIslands(connComp, imageGluedParts, smallIslands);

    if debug
        imshow(imageGluedPartsNoSmall), title('post glued parts: small islands removed');
    end
    
    % remove swimmer shapes of unfeasible area
    % depends: parts of shape must be glued
    imageFeasAreaShapes = imageGluedPartsNoSmall;
    
    blocksCount = connComp.NumObjects;
    if blocksCount > 0
        % 0.3 is too large - far away head is undetected 
        % eg. head 20 cm x 20 cm = 0.04
        bodyAreaMin = 0.04;
        bodyAreaMax = 2; % man of size 1m x 2m

        noiseIslands = zeros(1, blocksCount,'uint8');
        for i=1:blocksCount
            centroid = connCompProps(i).Centroid;
            centroidWorld = CameraDistanceCompensator.cameraToWorld(obj.distanceCompensator, centroid);

            actualArea = connCompProps(i).Area;

            expectAreaMax = CameraDistanceCompensator.worldAreaToCamera(obj.distanceCompensator, centroidWorld, bodyAreaMax);
            isNoise = actualArea > expectAreaMax;

            if ~isNoise
                expectAreaMim = CameraDistanceCompensator.worldAreaToCamera(obj.distanceCompensator, centroidWorld, bodyAreaMin);
                isNoise = actualArea < expectAreaMim;
            end

            %isNoise = actualArea < expectAreaMax/100;
            noiseIslands(i) = isNoise;
        end
        noiseIslands = cast(noiseIslands,'logical'); % indices must be logical

        imageFeasAreaShapes = HumanDetector.removeIslands(connComp, imageGluedPartsNoSmall, noiseIslands);

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
    end

    % return info about swimmer's shapes
    connComp = bwconncomp(resultImage, 8); % TODO: 8 or 4?
    connCompProps = regionprops(connComp, 'BoundingBox','Centroid','FilledImage');
    resultCells = struct('BoundingBox',[],'Centroid',[],'FilledImage',[]);
    resultCells(1) = []; % make 1x0 struct
    for i=1:length(connCompProps)
        props = connCompProps(i);
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
