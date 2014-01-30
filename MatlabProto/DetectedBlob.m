classdef DetectedBlob

properties
    Id;
    BoundingBox;
    Centroid;      % [X,Y] in pixels
    OutlinePixels;
    FilledImage;
    
    CentroidWorld; % [X,Y,Z] cenroid converted to world coordinates
end
    
end

