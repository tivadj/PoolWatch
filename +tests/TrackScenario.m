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
        
        nextFrameFun = @(image) tracker.nextFrame(image, 1000/this.Fps, this.Fps,  debug);
        
        % play all frames which have occured before sync point
        for ind=frameInd:frame.SeqNum-1
            fprintf('f=%d\n', ind);
            nextFrameFun(image);
        end
        
        % process sync frame
        fprintf('f=%d\n', frame.SeqNum);
        nextFrameFun(image);
        
        % update speed for newly created track
        %tracker.getTrack(frame.SeqNum, 
        for blobInd=1:length(frame.Blobs)
            blob = frame.Blobs(blobInd);
            if isfield(blob,'WorldVelocity')
                track = tracker.getTrackByBlobId(frame.SeqNum, blob.Id);
                assert(~isempty(track));
                track.SetVelocity(blob.WorldVelocity);
            end
        end
            
        
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