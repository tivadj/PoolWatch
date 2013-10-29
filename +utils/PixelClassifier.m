classdef PixelClassifier
methods(Static)
    
% returns mask of pixels which positively classified.
% imageMask = MxNx1
function imageMask = applyToImage(image, pixelClassifFun, debug)
    if ~exist('debug', 'var')
        debug = false;
    end
    
    pixelTriples = reshape(image, [], 3);

    % apply skin classifier to input image
    classifRes=pixelClassifFun(double(pixelTriples));

    if debug
        hist(classifRes);
    end

    % construct mask
    classifThrMask=classifRes > 0.5;

    classifThr = classifRes;
    classifThr( classifThrMask)=1;
    classifThr(~classifThrMask)=0;
    classifThr=im2uint8(classifThr);

    imageMask = reshape(classifThr, size(image,1), size(image,2));
end

function classifFun = getConvexHullClassifier(ballPixelsByRow, convexHullTriInd)
    classifFun = @(pixelsByRow) utils.inhull(pixelsByRow, ballPixelsByRow, convexHullTriInd, 0.2);
end

function filteredImage = applyAndGetImage(image, pixelClassifFun, debug)
    if ~exist('debug', 'var')
        debug = false;
    end

    maskSuccess = utils.PixelClassifier.applyToImage(image, pixelClassifFun);
    filteredImage = utils.applyMask(image, maskSuccess);
end

end
end