% Configuration
addpath('C:/devb/cplex/mexopencv'); % required for OpenCV cv.* functions

%% Track swimmers in video

%SwimmerTracker.batchTrackSwimmersInVideo(tr,true);

poolTracker=RunSwimmingPoolVideoFileTracker.create;
RunSwimmingPoolVideoFileTracker.run(poolTracker);

%%
%tr=SwimmerTracker.create;

%% Check skin classifier
skinClassifierRunner=RunSkinClassifier.create(true);
RunSkinClassifier.run(skinClassifierRunner);

%% Check human detector
humanDetectorRunner=RunHumanDetector.create;
RunHumanDetector.run(humanDetectorRunner);

%% Projection
distComp = RunCameraDistanceCompensator.create;
RunCameraDistanceCompensator.run(distComp);
%%
opticFlow1 = RunOpticalFlow1.create;
RunOpticalFlow1.run(opticFlow1)
%%

% TRACK TESTS
%{
% test
detectionsPerFrame=cell(2,1);
bd=struct();
bd(1,1).Centroid = [150 320]; % X=150 Y=320
bd(2,1).Centroid = [505 360];
bd(3,1).Centroid = [340 120];
detectionsPerFrame{1}=bd;
bd=struct();
bd(1,1).Centroid = [70 100];
bd(2,1).Centroid = [154 318];
detectionsPerFrame{2}=bd;
%}