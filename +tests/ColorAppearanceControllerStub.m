classdef ColorAppearanceControllerStub < handle

properties
    scenario;
end

methods
    
function this = ColorAppearanceControllerStub(scenario)
    this.scenario = scenario;
end

function onAssignDetectionToTrackedObject(this, track, detect, image)
    testBlob = this.getFrameBlob(detect.Id);
    if isfield(testBlob, 'ColorAppearance')
        if isempty(track.v.AppearancePixels)
            track.v.AppearancePixels = testBlob.ColorAppearance;
        else
            track.v.AppearancePixels = [track.v.AppearancePixels testBlob.ColorAppearance];
        end
    end
end

% Calculate how similar track and detection are.
function avgProb = similarityScore(this, track, detect, image)
    testBlob = this.getFrameBlob(detect.Id);
    
    if isfield(testBlob, 'ColorAppearance')        
        blobAppear = testBlob.ColorAppearance;
        
        trackAppear = mean(track.v.AppearancePixels);
        diff = abs(trackAppear - blobAppear);
        avgProb = (255 - diff) / 255;
    else
        avgProb = 0.5;
    end
end

% Min probability when blob is treated similar to track.
function prob = minAppearanceSimilarityScore(this)
    prob = 0.5;
end

function curBlob = getFrameBlob(this, blobId)
    % assume blobId is unique among all frames
    singleBlob = [];
    for i=1:length(this.scenario.ScenarioFrameList)
        frame = this.scenario.ScenarioFrameList(i);
        for blobInd=1:length(frame.Blobs)
            curBlob = frame.Blobs(blobInd);
            if curBlob.Id == blobId
                if ~isempty(singleBlob)
                    error('blobId must be unique among all frames');
                end
                singleBlob = curBlob;
            end
        end
    end
    testBlob = singleBlob;
end

end
    
end

