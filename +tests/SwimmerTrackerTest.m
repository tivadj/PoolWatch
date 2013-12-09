classdef SwimmerTrackerTest < matlab.unittest.TestCase

methods(Test)
   
function testTemporalDisappear(testCase)
    debug = 0;

    % scenario
    scenario = tests.MotherTests.loadTemporalDisappear();

    % tracker
    
    poolRegionDetector = tests.PoolRegionDetectorStub;
    distanceCompensator = tests.CameraDistanceCompensatorStub;
    humanDetector = tests.HumanDetectorStub(scenario);
    colorAppearance = tests.ColorAppearanceControllerStub;
    tracker = SwimmerTracker(poolRegionDetector, distanceCompensator, humanDetector, colorAppearance);
    
    scenario.play(tracker, debug);
    
    tracksInfo = tracker.getTracksInfo;
    testCase.verifyEqual(1, length(tracksInfo));
    testCase.verifyEqual([8.6 10], tracksInfo(1).ImagePos);
end

end
end
