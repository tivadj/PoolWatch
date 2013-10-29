classdef CameraDistanceCompensator

properties
%v.MRectPool % matrix to transform image to world coordinates
end

methods(Static)

function obj = create()
    obj = utils.TypeErasedClass;
    
    % matrix to transform 'MVI_3177_0127_640x476.png' image coordinates
    % into rectangular [640x476] coordinate (origin is top left).

    srcInitDimW = [480 357];
    srcCoordScaled=[181 124; 387 117; 563 267; -417 196]; % on scaled image W480 x H357
    
    poolSize = CameraDistanceCompensator.poolSize;
    expectCameraSizeW=CameraDistanceCompensator.expectCameraSize;
    srcCoord=[srcCoordScaled(:,1)*expectCameraSizeW(1)/srcInitDimW(1) srcCoordScaled(:,2)*expectCameraSizeW(2)/srcInitDimW(2)]; % world coordinates
    %dstCoord=[0 0; 640 0; 640 476; 0 476];
    dstCoord=[0 0; poolSize(1) 0; poolSize(1) poolSize(2); 0 poolSize(2)];
    obj.v.MRectPool=cv.getPerspectiveTransform(srcCoord, dstCoord); % image to world transformation

    % fit line Area ~ Distance from camera
%     areaClose = 2674;
%     areaFar = 27;
%     obj.areaVsDist = polyfit([0 476], [areaClose areaFar], 1); % = -0.1092d+58
    %obj.areaVsDist = [0.2259 -163.6806 2.9617e+04]; % parabola
end

function poolSize = poolSize()
    poolSize = [10 25];
end

function size = expectCameraSize()
    size=[640 476];
end

function areaM = scaleTopViewImageToWorldCoord(topViewPos, topViewImageSize)
    poolSize = CameraDistanceCompensator.poolSize;
    areaM = topViewPos .* poolSize ./ topViewImageSize;
end

% for now we analyze only 5 lanes, each 2m => 5*2=10m    
function areaM = scaleTopViewImageToWorldArea(areaTopPix, destImageSize)
    % 640x476    -- 50m x 25m
    % areaTopPix -- ?
    poolSize = CameraDistanceCompensator.poolSize;
    areaM = areaTopPix * prod(poolSize) / prod(destImageSize);
end

% used when we draw TopView image.
function topViewImagePos = scaleWorldToTopViewImageCoord(worldPos, destImageSize)
    poolSize = CameraDistanceCompensator.poolSize;
    topViewImagePos = worldPos ./ poolSize .* destImageSize;
end

function worldPos = cameraToWorld(obj, imagePos)
    num = size(imagePos,1);
    rectPosHom = obj.v.MRectPool * [imagePos ones(num,1)]';
    
    worldPos = utils.normalizeHomog(rectPosHom');
end

function imagePos = worldToCamera(obj, worldPos)
    num = size(worldPos,1);
    %rectPosHomSlow = inv(obj.MRectPool) * [worldPos ones(num,1)]';
    rectPosHom = obj.v.MRectPool \ [worldPos ones(num,1)]';
    
    imagePos = utils.normalizeHomog(rectPosHom');
end

% finds Map which associates components from TopView image to components in original skewed image.
function topToSkewed = findComponentMap(distanceCompensator,connComps,connCompsProps,connCompsTop,connCompsTopProps)
    mappedCentroids = CameraDistanceCompensator.cameraToWorld(distanceCompensator, reshape([connCompsProps(:).Centroid],2,[])');

    cost=zeros(connComps.NumObjects, connCompsTop.NumObjects);
    for row=1:connComps.NumObjects
    for col=1:connCompsTop.NumObjects
        len = norm(mappedCentroids(row,:) - connCompsTopProps(col).Centroid);
        cost(row,col)=len;
    end
    end
    
    [ass,unass1, unass2] = assignDetectionsToTracks(cost, 9999);
    assert(isempty(unass1));
    assert(isempty(unass2));
    
    % map old elements to new
    topToSkewed = containers.Map(ass(:,2), ass(:,1));
end

function isFeas = isFeasibleArea(obj, swimmerImagePos, area)
    % swimmerImagePos may be 2x1 matrix or 1x1 cell with such matrix
    if iscell(swimmerImagePos)
        swimmerImagePos = swimmerImagePos{1};
    end
    
    rectPosHom = obj.v.MRectPool * [swimmerImagePos 1]';
    rectPos = rectPosHom/rectPosHom(3); % normalize homog coordinate
    
    yrecWorld = rectPos(2);
    poolSize = CameraDistanceCompensator.poolSize;
    expectCameraSize=CameraDistanceCompensator.expectCameraSize;
    yrec = yrecWorld / poolSize(2) * expectCameraSize(2); % TopView image coordinate
    

    % 3sig
%     x1dw = [0.4857 -186.0575];
%     x1up = [0.5917 -206.2034];
    
    % 4sig
    x1dw = [0.4680 -182.6998-5]; % +-5 make somewhat wider
    x1up = [0.6094 -209.5611+5];
    
    if yrec < 390 % limit far away objects to not diminish to zero
        limDw = 3;
        limUp = 24;
    else
        limDw = x1dw * [yrec 1]';
        limUp = x1up * [yrec 1]';
    end
    
    %
    equivDiam = 2*sqrt(area/pi);
    
    % above x1dw and below x1up
    f1 = limDw < equivDiam;
    f2 = equivDiam < limUp;
    
    isFeas = f1 && f2;
end

% finds what area would occupy the object in world position __worldPos__ (in m) if
% it occupy __worldArea__ (in m^2) in world space.
function areaCam = worldAreaToCamera(obj, worldPos, worldArea)
    expectCameraSize = CameraDistanceCompensator.expectCameraSize;
    topViewPos = CameraDistanceCompensator.scaleWorldToTopViewImageCoord(worldPos, expectCameraSize);
    
    % see AreaVsY_fit_parabola1.png
    y = topViewPos(2);
    
    if y < 390
        areaCam = 20. + 0.310144 * y;
    else
        areaCam = 0.2259*y^2 - 163.6806*y + 29617;
    end
    
    %areaCam = [0.2259 -163.6806 29617] * [ylim^2 ylim 1]';
end

function imageRec = convertCameraImageToTopView(obj, image, outputImageSize)
    if ~exist('outputImageSize', 'var')
        outputImageSize = [size(image,2) size(image,1)];
    end
    
    % 1. camera coord to world coordinate
    % 2. scale world coordinate into outputImageSize
    poolSize = CameraDistanceCompensator.poolSize;
    M = diag([outputImageSize(1)/poolSize(1) outputImageSize(2)/poolSize(2) 1]) * obj.v.MRectPool;
    
	imageRec = cv.warpPerspective(image, M, 'DSize', outputImageSize);
end

end
end
