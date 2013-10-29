classdef TrackedObject <handle
properties
    Id;
    %v.IsTrackCandidate; % true=TrackCandidate
    %v.TrackCandidateId;
    %v.FirstAppearanceFrameIdx;
    PromotionFramdInd; % the frame when candidate was promoted to track

    KalmanFilter; % used to predict position of track candidate
    Assignments;
    v;
end
methods(Static)
    function obj = NewTrackCandidate(trackCandidateId)
        obj = TrackedObject;
        obj.v.IsTrackCandidate = true;
        obj.v.TrackCandidateId = trackCandidateId;
        obj.Id = -1;
        obj.Assignments = cell(1,1);
    end
end

methods
    function detectCount = getDetectionsCount(obj, upToFrame)
        detectCount = 0;
        for i=obj.v.FirstAppearanceFrameIdx:upToFrame
            ass = obj.Assignments{i};
            if ~isempty(ass) && ass.IsDetectionAssigned
                detectCount = detectCount + 1;
            end
        end
    end
end
end