classdef LogicalLineSegment <handle
properties
    Id; % :int32
    Points; % points [Nx2]
    Next; % :typeof(this) pointer to next segment or []
    Prev; % :typeof(this) pointer to previous segment or []
    CanBeMerged; % :bool, true if this group can be combined with another segment
end

methods
    
% segment: Nx2
function this = LogicalLineSegment(points)
    assert(isa(points, 'double'));
    
    [N,dims] = size(points);
    assert(N >= 2);
    assert(dims == 2);
    
    %    
    this.CanBeMerged = true;
    this.Points = points;
end
    
function integrate(this, otherSeg)
    if this.isempty
        this.Points = otherSeg.Points;
        return;
    end

    [where,otherPntInd,~] = this.closestIntegration(otherSeg);
    
    otherPnts = otherSeg.terminalPoints;
    this.integrateSegment(where, otherPnts, otherPntInd);
end

% Find how segments can be integrated.
function [where, otherPntInd, dist] = closestIntegration(this, otherSeg)
    p1 = this.Points(1,:);
    p2 = this.Points(this.pointsCount,:);
    p3 = otherSeg.Points(1,:);
    p4 = otherSeg.Points(otherSeg.pointsCount,:);
    
    % group by gluing closest points
    
    allPoints = [p1; p2; p3; p4];
    checkInds = [1 3; 1 4; 2 3; 2 4];
    dists = arrayfun(@(k1,k2) norm(allPoints(k1,:)-allPoints(k2,:)), checkInds(:,1), checkInds(:,2));
    [dist, minInd] = min(dists);
    
    closestInd1 = checkInds(minInd,1);
    closestInd2 = checkInds(minInd,2);
    
    otherPntInd = closestInd2 - 2;
    
    if closestInd1 == 1        
        where = 'begin';
    else
        where = 'end';
    end
end

% Integrates segment (integratePnt, otherPnt) using integratePnt as a merging
% point.
% where : {'initial', 'begin', 'end'}. Initial is applied when the first segment
% is added to empty polyline.
function integrateSegment(this, where, otherPnts, otherSegPntInd)
    if strcmpi(where, 'initial')
        this.Points = [p3; p4];
        return;
    end
    
    integratePnt = otherPnts(otherSegPntInd,:);
    otherPnt = otherPnts(3 - otherSegPntInd,:);

    if strcmpi(where, 'begin')
        thisIntegratePnt = this.Points(1,:);
    elseif strcmpi(where, 'end')
        thisIntegratePnt = this.Points(this.pointsCount,:);
    else
        error('argument where must be "begin" or "end" strings');
    end
    
    % Segments are grouped using mid-point between integratePnt and end
    % of this polyline.
    midPoint = mean([thisIntegratePnt; integratePnt], 1);
    
    % update this polyline
    if strcmpi(where, 'begin')
        this.Points(1,:) = midPoint;
        this.Points = [otherPnt; this.Points]; % add new to begin
    else strcmpi(where, 'end')
        this.Points(this.pointsCount,:) = midPoint;
        this.Points = [this.Points; otherPnt]; % add new to begin
    end
end

function linkSegment(this, other)
    % integrate
    [where, otherSegPntInd, ~] = this.closestIntegration(other);
    
    % coerce two logical segments
    
    % get midpoint
    if strcmpi(where, 'begin')
        thisSegPntInd = 1;
    elseif strcmpi(where, 'end')
        thisSegPntInd = this.pointsCount;
    end
    
    closest1 = this.Points(thisSegPntInd,:);
    
    otherPnts = other.terminalPoints;
    closest2 = otherPnts(otherSegPntInd,:);
    
    midPoint = mean([closest1; closest2]);
    this.Points(thisSegPntInd,:) = midPoint;
    
    if otherSegPntInd == 1
        other.Points(1,:) = midPoint;
    else
        other.Points(other.pointsCount,:) = midPoint;
    end
    
    % link
    
    if strcmpi(where, 'begin')
        this.Prev = other;
        other.Next = this;
    elseif strcmpi(where, 'end')
        this.Next = other;
        other.Prev = this;
    end
end

function points = asPolyline(this)
    assert(isempty(this.Prev) || isempty(this.Next), 'Segment must be first or last in chain');

    neigh = this.Prev;
    if isempty(neigh)
        neigh = this.Next;
    end
    
    [where, ~, ~] = this.closestIntegration(neigh);
    
    if strcmp(where,'begin')
        thisPntInd = 1;
    else
        thisPntInd = 2;
    end        
    
    points = zeros(0,2);
    if thisPntInd == 1
        points(end+1,:) = this.Points(this.pointsCount,:);
    else
        points(end+1,:) = this.Points(1,:);
    end
    
    prevSeg = [];
    seg = this;
    segEndPntWhere = thisPntInd;
    while true
        if segEndPntWhere == 1
            points(end+1,:) = seg.Points(1,:);
        else
            points(end+1,:) = seg.Points(seg.pointsCount,:);
        end
        
        % get next segment in chain
        
        if ~isempty(prevSeg)
            if seg.Next == prevSeg
                nextSeg = seg.Prev;
            else
                nextSeg = seg.Next;
            end
        else
            nextSeg = seg.Next;
            if isempty(nextSeg)
                nextSeg = seg.Prev;
            end
        end
        
        if isempty(nextSeg)
            break;
        end

        [~, neighPntInd, ~] = seg.closestIntegration(nextSeg);
        
        prevSeg = seg;
        seg = nextSeg;
        segEndPntWhere = 3 - neighPntInd;
    end
end

function count = pointsCount(this)
    count = size(this.Points, 1);
end

function isEmpty = isempty(this)
    isEmpty = isempty(this.Points);
end

function segm = terminalPoints(this)
    segm = [this.Points(1,:); this.Points(this.pointsCount,:)];
end

function angle = slopeAngle(this)
    seg = this.terminalPoints;
    v = seg(2,:)-seg(1,:);
    angle = atan2(v(2), v(1));
end

end
end

