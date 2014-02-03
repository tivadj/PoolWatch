classdef ScenarioBlob
properties
Id;

% position (in meters)
WorldPos; % [x y]

% speed (in m/s)
WorldVelocity; % [vx vy]

% eg: color in 1D space: close values mean similar appearance
ColorAppearance; % type:int, 
end    
end
