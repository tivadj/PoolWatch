classdef ConvexHullSkinClassifier < handle
properties
    ballPoints;
    ballConvexHullTriInds; % =convhulln(ballPoints)
end

methods
    
function obj = ConvexHullSkinClassifier(obj, ballPoints, ballConvexHullTriInds)
    obj.ballPoints = ballPoints;
    obj.ballConvexHullTriInds = ballConvexHullTriInds;
end

function classifRes = isSkinPixel(obj, pixelsByRow)
    classifRes = utils.inhull(pixelsByRow, obj.ballPoints, obj.ballConvexHullTriInds, 0.2);
end

end
end