classdef SwimmerTrackerTest < matlab.unittest.TestCase
    

methods(Test)
   
function testTemporalDisappear(testCase)
    debug = 0;

    scenario = tests.MotherTests.loadTemporalDisappear();

    % swimmer's speed is big enough to expect D3 to be associated with single track
    tracker = testCase.createDummyTracker(scenario);
    tracker.v.swimmerMaxSpeed = 2;
    
    scenario.play(tracker, debug);
    testCase.verifyEqual(tracker.tracksCount, 1);
    testCase.verifyEqual(tracker.trackInfoOne.WorldPos, [8.6 10 0]);
end

function testLimitMaxSpeed(testCase)
    debug = 0;

    scenario = tests.MotherTests.loadLimitMaxSpeed();
   
    % swimmer's speed is low so there should be two tracks
    tracker = testCase.createDummyTracker(scenario);
    tracker.v.swimmerMaxSpeed = 0.1; % low speed
    tracker.v.minDistToNewTrack = 0; % allow creating tracks anywhere
    
    scenario.play(tracker, debug);

    testCase.verifyEqual(tracker.tracksCount, 2);
    
    t1 = tracker.getTrackByBlobId(1, 1);
    t1Info = tracker.trackInfo(t1.idOrCandidateId);
    testCase.verifyEqual(t1Info.WorldPos, [8.4 10 0], 'RelTol', 0.001); % +0.2 from initial pos

    t2 = tracker.getTrackByBlobId(2, 2);
    t2Info = tracker.trackInfo(t2.idOrCandidateId);
    testCase.verifyEqual(t2Info.WorldPos, [9 10 0]);
end

% color appearance
function testTemporalDisappearBoth(testCase)
    debug = 0;

    scenario = tests.MotherTests.loadTemporalDisappearColorBoth();
   
    % swimmer's speed is low so there should be two tracks
    tracker = testCase.createDummyTracker(scenario);
    tracker.v.minDistToNewTrack = 0; % allow creating tracks anywhere
    tracker.v.swimmerMaxSpeed = 2;
    
    scenario.play(tracker, debug);

    t1 = tracker.getTrackByBlobId(1, 1);
    t1Info = tracker.trackInfo(t1.idOrCandidateId);
    testCase.verifyEqual(t1Info.WorldPos, [7.6 10 0]);

    t2 = tracker.getTrackByBlobId(1, 2);
    t2Info = tracker.trackInfo(t2.idOrCandidateId);
    testCase.verifyEqual(t2Info.WorldPos, [8 10.5 0]);
end

% color appearance
function testRecoverTrackByAppearance(testCase)
    debug = 0;

    scenario = tests.MotherTests.loadRecoverTrackByAppearance();
   
    % swimmer's speed is low so there should be two tracks
    tracker = testCase.createDummyTracker(scenario);
    tracker.v.minDistToNewTrack = 0; % allow creating tracks anywhere
    tracker.v.swimmerMaxSpeed = 2;
    
    scenario.play(tracker, debug);

    t1 = tracker.getTrackByBlobId(1, 20);
    t1Info = tracker.trackInfo(t1.idOrCandidateId);
    testCase.verifyEqual(t1Info.ImagePos, [8.4 10]);
    
    % tracker should detect that D36 was a false detection and reject assigned
    % detection for frame=2, for this tracker should keep multiple hypothesis
    %t1Info = tracker.trackInfo(t1.idOrCandidateId, 2);
    %testCase.verifyEqual(t1Info.ImagePos, []);
end


end

methods

function tracker = createDummyTracker(this, scenario)
    poolRegionDetector = tests.PoolRegionDetectorStub;
    distanceCompensator = tests.CameraDistanceCompensatorStub;
    humanDetector = tests.HumanDetectorStub(scenario);
    colorAppearance = tests.ColorAppearanceControllerStub(scenario);
    tracker = SwimmingPoolObserver(poolRegionDetector, distanceCompensator, humanDetector, colorAppearance);
end

end

end
