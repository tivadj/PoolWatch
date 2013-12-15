classdef MotherTests
methods(Static)

function scenario = loadTemporalDisappear
    scenario = tests.TrackScenario;
    scenario.Fps = 1;
    
    frame = tests.ScenarioFrame;
    frame.SeqNum = 1;
    frame.Blobs(1).Id = 2;
    frame.Blobs(1).WorldPos = [8.2 10];
    frame.Blobs(1).WorldVelocity = [0.2 0];
    scenario.ScenarioFrameList(end+1) = struct(frame);
    
    % frame=2: temporal disappearance
    
    frame = tests.ScenarioFrame;
    frame.SeqNum = 3;
    frame.Blobs(1).Id = 4;
    frame.Blobs(1).WorldPos = [8.6 10];
    scenario.ScenarioFrameList(end+1) = struct(frame);
end

function scenario = loadLimitMaxSpeed
    scenario = tests.TrackScenario;
    scenario.Fps = 1;
    
    frame = tests.ScenarioFrame;
    frame.SeqNum = 1;
    frame.Blobs(1).Id = 1;
    frame.Blobs(1).WorldPos = [8.2 10];
    frame.Blobs(1).WorldVelocity = [0.2 0];
    scenario.ScenarioFrameList(end+1) = struct(frame);
    
    % D2 detection is too far => new track should be created
    
    frame = tests.ScenarioFrame;
    frame.SeqNum = 2;
    frame.Blobs(1).Id = 2;
    frame.Blobs(1).WorldPos = [9 10];
    scenario.ScenarioFrameList(end+1) = struct(frame);
end

% Color appearance test
function scenario = loadTemporalDisappearColorBoth
    scenario = tests.TrackScenario;
    scenario.Fps = 1;
    
    frame = tests.ScenarioFrame;
    frame.SeqNum = 1;
    frame.Blobs(1).Id = 1;
    frame.Blobs(1).WorldPos = [8 10];
    frame.Blobs(1).WorldVelocity = [-0.2 0];
    frame.Blobs(1).ColorAppearance = 10;
    frame.Blobs(2).Id = 2;
    frame.Blobs(2).WorldPos = [7.6 10.5];
    frame.Blobs(2).WorldVelocity = [0.2 0];
    frame.Blobs(2).ColorAppearance = 200;
    scenario.ScenarioFrameList(end+1) = struct(frame);
    
    % frame=2 no detections (both disappear)
    
    frame = tests.ScenarioFrame;
    frame.SeqNum = 3;
    frame.Blobs(1).Id = 3;
    frame.Blobs(1).WorldPos = [7.6 10];
    frame.Blobs(1).ColorAppearance = 10;
    frame.Blobs(2).Id = 4;
    frame.Blobs(2).WorldPos = [8 10.5];
    frame.Blobs(2).ColorAppearance = 200;
    scenario.ScenarioFrameList(end+1) = struct(frame);
end

% Color appearance test
function scenario = loadRecoverTrackByAppearance
    scenario = tests.TrackScenario;
    scenario.Fps = 1;
    
    frame = tests.ScenarioFrame;
    frame.SeqNum = 1;
    frame.Blobs(1).Id = 20;
    frame.Blobs(1).WorldPos = [8 10];
    frame.Blobs(1).WorldVelocity = [0.2 -0.2];
    frame.Blobs(1).ColorAppearance = 255;
    scenario.ScenarioFrameList(end+1) = struct(frame);
    
    frame = tests.ScenarioFrame;
    frame.SeqNum = 2;
    frame.Blobs(1).Id = 36;
    frame.Blobs(1).WorldPos = [8.14 9.86];
    frame.Blobs(1).ColorAppearance = 10; %
    scenario.ScenarioFrameList(end+1) = struct(frame);

    frame = tests.ScenarioFrame;
    frame.SeqNum = 3;
    frame.Blobs(1).Id = 38;
    frame.Blobs(1).WorldPos = [8.4 10];
    frame.Blobs(1).ColorAppearance = 255;
    scenario.ScenarioFrameList(end+1) = struct(frame);
end

end
end