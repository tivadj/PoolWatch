classdef TrackInfoHistory
% Represents coordinates of a tracked object through its lifetime.

properties
    Id;
    IsTrackCandidate; % true=TrackCandidate
    TrackCandidateId;
    FirstAppearanceFrameIdx;
    PromotionFramdInd; % the frame when candidate was promoted to track

    Assignments;
end

methods
    
function this = TrackInfoHistory()
    this.Id = -1;
    this.IsTrackCandidate = true;
    this.PromotionFramdInd = -1;
end

function id = idOrCandidateId(this)
    if this.Id > 0
        id = this.Id;
    else
        % make id negative to distinguish from TrackId
        id = -this.TrackCandidateId;
    end
end

% Get the number of frames in which the observation is associated with this track.
function detectCount = getDetectionsCount(this, toFrame)
    detectCount = 0;

    upToFrame = min([toFrame length(this.Assignments)]);
    for i=this.FirstAppearanceFrameIdx:upToFrame
        ass = this.Assignments{i};
        if ~isempty(ass) && ass.IsDetectionAssigned
            detectCount = detectCount + 1;
        end
    end
end

end
end
