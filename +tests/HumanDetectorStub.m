classdef HumanDetectorStub
properties
    testScenario;
end
methods
    
function this = HumanDetectorStub(testScenario)
    this.testScenario = testScenario;
end

function BodyDescr = GetHumanBodies(this, frameId, varargin)
    BodyDescr = struct(DetectedBlob);
    BodyDescr(1) = [];

    testFrame = this.testScenario.getFrame(frameId);
    if isempty(testFrame)
        return;
    end
    
    for i=1:length(testFrame.Blobs)
        testBlob = testFrame.Blobs(i);

        blob = DetectedBlob;
        blob.Id = testBlob.Id;
        blob.Centroid = testBlob.WorldPos(1:2);
        BodyDescr(end+1) = struct(blob);
    end
end

end
end

