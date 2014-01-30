classdef TrackChangePerFrame
% Represents changes wich happen in the observed with scene with each track.
% The list of such change per each 
    
properties
    TrackCandidateId;
    UpdateType; % enum
    EstimatedPosWorld; % [X,Y,Z] corrected by sensor position (in world coord)
    
    ObservationInd; % type:int32, 0=no observation; >0 observation index
    ObservationPosPixExactOrApprox; % [X,Y]; required to avoid world->camera conversion on drawing
end

properties (Constant)
    New = 1;
    ObservationUpdate = 2;
    NoObservation = 3;
    %Finished = 9;
end


methods
end

end

