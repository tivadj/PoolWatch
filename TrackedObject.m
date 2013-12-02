classdef TrackedObject <handle
properties
    Id;
    IsTrackCandidate; % true=TrackCandidate
    TrackCandidateId;
    FirstAppearanceFrameIdx;
    PromotionFramdInd; % the frame when candidate was promoted to track

    KalmanFilter; % used to predict position of track candidate
    Assignments;
    v;
    %v.AppearanceMixtureGaussians; % type: cv.EM, accumulated color signature up to the last frame
    %v.AppearancePixels;
end
methods(Static)
    function obj = NewTrackCandidate(trackCandidateId)
        obj = TrackedObject;
        obj.IsTrackCandidate = true;
        obj.TrackCandidateId = trackCandidateId;
        obj.Id = -1;
        obj.Assignments = cell(1,1);
        obj.v.AppearanceMixtureGaussians = cv.EM('Nclusters', 16, 'CovMatType', 'Spherical');
        obj.v.AppearancePixels = zeros(0,3,'uint8');
    end
end

methods
    function detectCount = getDetectionsCount(obj, upToFrame)
        detectCount = 0;
        for i=obj.FirstAppearanceFrameIdx:upToFrame
            ass = obj.Assignments{i};
            if ~isempty(ass) && ass.IsDetectionAssigned
                detectCount = detectCount + 1;
            end
        end
    end
end
end