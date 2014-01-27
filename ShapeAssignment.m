classdef ShapeAssignment
    properties
        EstimatedPosWorld; % corrected by sensor position (in world coord)

        IsDetectionAssigned;
        DetectionInd;

        % corresponding centroid from associated blob or estimated world position,
        % converted to image coordinates, in case of no observation.
        ObservationPosPixExactOrApprox; % [X,Y]

        v;
        
        % used for checking if EstimatedPos is correctly back projected into image view
        %v.EstimatedPosImagePix; (in image coord) 
    end
end