classdef PoolRegionDetector < handle
properties
    fleshClassifierFun;
    waterClassifierFun;
end
methods

function this = PoolRegionDetector(fleshClassifierFun, waterClassifierFun)
    this.fleshClassifierFun = fleshClassifierFun;
    this.waterClassifierFun = waterClassifierFun;
end
    
function waterMask = getWaterMask(this, image)
    waterMask = utils.PixelClassifier.applyToImage(image, this.waterClassifierFun);
end

function poolMask = getPoolMask(this, image, waterMask, forceSingleBlob, debug)
    poolMask = PoolBoundaryDetector.getPoolMask(image, waterMask, forceSingleBlob, debug);
end

function dividersMask = getLaneDividersMask(this, image, imagePoolBnd, waterMask, debug)
    dividersMask = PoolBoundaryDetector.getLaneDividersMask(image, imagePoolBnd, waterMask, this.fleshClassifierFun, debug);
end

end
end

