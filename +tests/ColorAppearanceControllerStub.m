classdef ColorAppearanceControllerStub < handle

properties
end

methods

function onAssignDetectionToTrackedObject(this, track, detect, image)
end

% Calculate how similar track and detection are.
function avgProb = similarityScore(this, track, detect, image)
    avgProb = 0.5;
end

% Min probability when blob is treated similar to track.
function prob = minAppearanceSimilarityScore(this)
    prob = 0.5;
end

end
    
end

