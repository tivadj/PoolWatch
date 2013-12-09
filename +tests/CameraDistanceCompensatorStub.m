classdef CameraDistanceCompensatorStub

methods

function poolSize = poolSize(this)
    poolSize = [100 25];
end

function worldPos = cameraToWorld(obj, imagePos)
    worldPos = [imagePos 0]; % z = 0
end

function imagePos = worldToCamera(obj, worldPos)
    imagePos = worldPos;
end

end    
end

