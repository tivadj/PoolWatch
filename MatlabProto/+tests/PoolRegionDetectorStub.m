classdef PoolRegionDetectorStub
methods

function waterMask = getWaterMask(this, image)
    [w,h] = size(image);
    waterMask = true(h, w);
end

function poolMask = getPoolMask(this, image, varargin)
    [w,h] = size(image);
    poolMask = true(h, w);
end

function dividersMask = getLaneDividersMask(this, image, varargin)
    [w,h] = size(image);
    dividersMask = false(h, w);
end

end
end

