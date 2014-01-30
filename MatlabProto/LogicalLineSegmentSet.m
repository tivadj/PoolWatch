classdef LogicalLineSegmentSet <handle
% Represents a reference to a list of virtual line segments.
properties
    Segments;
end

methods
function this = LogicalLineSegmentSet
    this.Segments = cell(1,0);
end
end

end