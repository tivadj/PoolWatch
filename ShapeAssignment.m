classdef ShapeAssignment
    properties
        IsDetectionAssigned;
        PredictedPos; % position predicted by Kalman filter (in world coord)
        EstimatedPos; % corrected by sensor position (in world coord)
        v;

        % Specific for Image view 
        DetectionInd;
        
        % used for checking if EstimatedPos is correctly back projected into image view
        %v.EstimatedPosImagePix; (in image coord) 
    end
    
    methods
        % TODO: not visible outside
        function pos=DetectionOrPrediction(obj)
            if obj.IsDetectionAssigned
                %pos=obj.DetectionCentroid;
            else
                pos=obj.PredictedPos;
            end
        end
    end
end