classdef TrackScenario <handle
properties
    Fps;
    ScenarioFrameList;
end
methods
    
function this = TrackScenario()
    this.ScenarioFrameList = struct(tests.ScenarioFrame);
    this.ScenarioFrameList(1) = [];
end

function load(this, scenarioFilePath, debug)

end

function play(this, tracker, debug)
    frameInd = 1;
    
    image = zeros(1,1,1,'uint8');
    for frameSyncPointInd=1:length(this.ScenarioFrameList)
        frame = this.ScenarioFrameList(frameSyncPointInd);
        
        nextFrame = @(image) tracker.nextFrame(image, 1/this.Fps, this.Fps,  debug);
        
        % play all frames which have occured before sync point
        for ind=frameInd:frame.SeqNum-1
            fprintf('f=%d\n', ind);
            nextFrame(image);
        end
        
        % process sync frame
        fprintf('f=%d\n', frame.SeqNum);
        nextFrame(image);
        
        frameInd = frame.SeqNum + 1;
    end
end

function frame = getFrame(this, frameId)
    frame = [];
    for i=1:length(this.ScenarioFrameList)
        cur = this.ScenarioFrameList(i);
        if cur.SeqNum == frameId
            frame = cur;
            break;
        end
    end
end

end
end