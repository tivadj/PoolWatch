classdef CameraDistanceCompensator < handle

properties
v;
%obj.v.cameraMatrix
%obj.v.rvec
%obj.v.tvec
%obj.v.worldToCamera
end

methods
    
function this = CameraDistanceCompensator()
    this.initCameraPosition();
end

function poolSize = poolSize(this)
    poolSize = [10 25];
end

% TODO: remove
function size = expectCameraSize(this)
    size=[640 476];
end

function initCameraPosition(obj)
    % Cannon D20 640x480
    cx=323.07199373780122;
    cy=241.16033688735058;
    fx=526.96329424435044;
    fy=527.46802103114874;
    cameraMatrix = [fx 0 cx; 0 fy cy; 0 0 1];
    obj.v.cameraMatrix = cameraMatrix;
    
    distCoeffs = [0 0 0 0];
    obj.v.distCoeffs = distCoeffs;

    %
    imagePoints = zeros(0,2);
    worldPoints = zeros(0,3);

    zeroHeight = 0;
    
    % top, origin (0,0)
    imagePoints(end+1,:) = [242 166];
    worldPoints(end+1,:) = [0 0 zeroHeight];
%     % top, 3 marker
%     imagePoints(end+1,:) = [454 169];
%     worldPoints(end+1,:) = [0 8];
    % top, 4 marker
    imagePoints(end+1,:) = [516 156];
    worldPoints(end+1,:) = [0 10 zeroHeight];
    % bottom, 2 marker
    imagePoints(end+1,:) = [-71 304];
    worldPoints(end+1,:) = [25 6 zeroHeight];
%     % bottom, 3 marker
%     imagePoints(end+1,:) = [231 133];
%     worldPoints(end+1,:) = [25 8];
    % bottom, 4 marker
    imagePoints(end+1,:) = [730 365];
    worldPoints(end+1,:) = [25 10 zeroHeight];
    
    %
    [rvec,tvec] = cv.solvePnP(worldPoints, imagePoints, cameraMatrix, distCoeffs);
    obj.v.rvec = rvec;
    obj.v.tvec = tvec;
    
    rotMat = cv.Rodrigues(rvec);
    
    worldToCamera = [rotMat tvec; [0 0 0 1]];
    obj.v.worldToCamera = worldToCamera;
    
end

function worldPos = cameraToWorld(obj, imagePos)
    num = size(imagePos,1);
    
    % image to camera coordinates
    camPos = inv(obj.v.cameraMatrix)*[imagePos 1]'; % 3xN

    cameraToWorld = inv(obj.v.worldToCamera); % 4x4

    % find fourth homog component so that world z=0
    zeroHeight = 0;
    %homZ = (repmat(zeroHeight,1,num) - cameraToWorld(3,1:3)*camPos) ./ cameraToWorld(3,4); % 1xN
    homZ = (zeroHeight - cameraToWorld(3,1:3)*camPos) ./ cameraToWorld(3,4); % 1xN

    reW1 = cameraToWorld * [camPos; homZ];
    worldPos = utils.normalizeHomog(reW1');
end

function imagePos = worldToCamera(obj, worldPos)
    num = size(worldPos,1);
    
    if length(worldPos) == 2
        zeroHeight = 0;
        worldPos3 = [worldPos zeroHeight];
    else
        worldPos3 = worldPos;
        worldPos = worldPos(:,1:2);
    end
    
    imagePos = cv.projectPoints(worldPos3, obj.v.rvec, obj.v.tvec, obj.v.cameraMatrix, obj.v.distCoeffs);
    imagePos = reshape(imagePos, num, []);
end

% finds the area of shape in image (in pixels^2) of an object with world position __worldPos__ (in m)
function areaCam = worldAreaToCamera(obj, worldPos, worldArea)
    widthHf = sqrt(worldArea) / 2;
    
    % project body bounding box into camera
    worldBounds = [
        worldPos + [-widthHf -widthHf 0];
        worldPos + [-widthHf  widthHf 0];
        worldPos + [ widthHf  widthHf 0];
        worldPos + [ widthHf -widthHf 0]];
    
    camBounds = obj.worldToCamera(worldBounds);
    areaCam = polyarea(camBounds(:,1), camBounds(:,2));
end

function imageRec = convertCameraImageToTopView(obj, image, outputImageSize)
    error('not implemented');
    if ~exist('outputImageSize', 'var')
        outputImageSize = [size(image,2) size(image,1)];
    end
    
    % 1. camera coord to world coordinate
    % 2. scale world coordinate into outputImageSize
    poolSize = obj.poolSize;
    M = diag([outputImageSize(1)/poolSize(1) outputImageSize(2)/poolSize(2) 1]) * obj.v.MRectPool;
    
	imageRec = cv.warpPerspective(image, M, 'DSize', outputImageSize);
end

end
end
