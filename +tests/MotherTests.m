classdef MotherTests
methods(Static)

function scenario = loadTemporalDisappear
    scenario = tests.TrackScenario;
    scenario.Fps = 1;
    
    % temporal disappear
    frame = tests.ScenarioFrame;
    frame.SeqNum = 1;
    frame.Blobs = struct(tests.ScenarioBlob);
    frame.Blobs(1).Id = 1;
    frame.Blobs(1).WorldPos = [8 10];
    scenario.ScenarioFrameList(end+1) = struct(frame);

    frame = tests.ScenarioFrame;
    frame.SeqNum = 2;
    frame.Blobs(1).Id = 2;
    frame.Blobs(1).WorldPos = [8.2 10];
    scenario.ScenarioFrameList(end+1) = struct(frame);
    
    frame = tests.ScenarioFrame;
    frame.SeqNum = 4;
    frame.Blobs(1).Id = 4;
    frame.Blobs(1).WorldPos = [8.6 10];
    scenario.ScenarioFrameList(end+1) = struct(frame);
end

end
end