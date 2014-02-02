classdef TrackChangePerFrame
% Represents changes wich happen in the observed with scene with each track.
% The list of such change per each 
    
properties
    TrackCandidateId; % type: int32
    UpdateType; % int32 enum
    EstimatedPosWorld; % single[X,Y,Z] corrected by sensor position (in world coord)
    
    ObservationInd; % type:int32, 0=no observation; >0 observation index
    ObservationPosPixExactOrApprox; % single[X,Y]; required to avoid world->camera conversion on drawing
end

properties (Constant)
    New = int32(1);
    ObservationUpdate = int32(2);
    NoObservation = int32(3);
    %Finished = 9;
end


methods
end

end

