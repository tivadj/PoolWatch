classdef ColorAppearanceController < handle

properties
end

methods

function onAssignDetectionToTrackedObject(this, track, detect, image)
    % retraining GMM on each frame is too costly
    % => retrain it during the initial time until enough pixels is accumulated
    % then we assume swimmer's shape doesn't change
    if track.canAcceptAppearancePixels
        pixs = this.getDetectionPixels(detect, image);
        track.pushAppearancePixels(pixs);
    end
end

% Calculate how similar track and detection are.
function avgProb = similarityScore(this, track, detect, image)
    pixs = this.getDetectionPixels(detect, image);

    probs = track.predict(pixs);
    avgProb = mean(probs);
end

% Min probability when blob is treated similar to track.
function prob = minAppearanceSimilarityScore(this)
    prob = 0.00005;
end

function forePixs = getDetectionPixels(obj, detect, image)
    bnd = ceil(detect.BoundingBox); % ceil to get positive index for pos=0.5
    shapeImage = image(bnd(2):bnd(2)+bnd(4)-1,bnd(1):bnd(1)+bnd(3)-1,:);
    forePixs = reshape(shapeImage, [], 3);

    shapeMask = detect.FilledImage;
    shapeMask = reshape(shapeMask, [], 1);
    
    assert(length(shapeMask) == size(forePixs,1));
    forePixs = forePixs(shapeMask,:);
end

end
    
end

