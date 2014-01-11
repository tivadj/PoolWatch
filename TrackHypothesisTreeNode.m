classdef TrackHypothesisTreeNode < handle
properties
    Id; % :int
    FamilyId; % :int
    Children; % :List<this>
    Parent; % :this
    DetectionInd; % :int
    FrameInd; % :int
    ObservationPos; % : [X,Y] in pixels
    ObservationWorldPos; % :[X,Y Z] in meters
    EstimatedWorldPos; % :[X,Y,Z]
    CreationReason; % enum
    KalmanFilterState;     % :[X,Y,vx,vy]
    KalmanFilterStateCovariance; % [4x4]
    KalmanFilterStatePrev; % :[X,Y,vx,vy], state which leads to this state
    KalmanFilterStateCovariancePrev;
    Score;       % :double, score which determines validity of the hypothesis (from root to this node)
    ScoreKalman;
    v;
end

properties (Constant)
    SequantialCorrespondence = 1;
    New = 2;
    NoObservation = 3;
end

methods
    
function this = TrackHypothesisTreeNode()
    this.Id = 0;
    this.Children = [];
    this.Parent = [];
end

function str = briefInfoStr(this)
    if this.DetectionInd > 0
        observSymb = sprintf('%d', this.DetectionInd);
    else
        observSymb = '0';
    end
    str = sprintf('(%d o%s)', this.Id, observSymb);
end

function addChild(this, child)
    assert(~isempty(child));

    % remove child from existent parent
    oldParent = child.Parent;
    if ~isempty(oldParent)
        ind = oldParent.childIndex(child);
        assert(ind ~= -1);
        
        oldParent.removeChildByIndex(ind);
    end
    
    this.Children{end+1} = child;
    child.Parent = this;
end

function clearChildren(this)
    for i=length(this.Children):-1:1
        this.removeChildByIndex(i);
    end
end

function removeChildByIndex(this, childInd)
    assert(childInd <= length(this.Children));
    
    this.Children{childInd}.Parent = [];
    this.Children(childInd) = [];
end

function childInd = childIndex(this, child)
    for childInd=1:length(this.Children)
        if this.Children{childInd} == child
            return;
        end
    end
    childInd = -1;
end

function leafSet = getLeafSet(this, includeThis)
    leafSet = cell(1,0);
    
    if isempty(this.Children)
        if includeThis
            leafSet{end+1} = this;
        end
    else
        childSet = this.Children;
        for childInd=1:length(childSet)
            leafSubSet1 = childSet{childInd}.getLeafSet(true);
            
            for i=1:length(leafSubSet1)
                leafSet{end+1} = leafSubSet1{i};
            end
        end
    end
end

function pathFromRoot = getPathFromRoot(this)
    seq = cell(1,0);
    
    cur = this;
    while ~isempty(cur)
        seq{end+1} = cur;
        cur = cur.Parent;
    end    
    
    count = length(seq);
    pathFromRoot = cell(1, count);
    for i=1:count
        pathFromRoot{i} = seq{count - i + 1};
    end
end

% ancestorIndex = 0 self
% ancestorIndex = 1 parent
% ancestorIndex = N N-th parent
function ancestor = getAncestor(this, ancestorIndex)
    cur = this;
    while ancestorIndex > 0
        if isempty(cur)
            break;
        end
        
        cur = cur.Parent;
        ancestorIndex = ancestorIndex - 1;
    end    
    
    ancestor = cur;
end

end % methods

methods (Static)

function ancestor = commonAncestor(child, other)
    assert(~isempty(child));
    assert(~isempty(other));
    
    ancestor = [];
    
    childAnc = child;
    while ~isempty(childAnc)
        otherAnc = other;
        
        while ~isempty(otherAnc)
            if childAnc.Id == otherAnc.Id
                ancestor = childAnc;
                return;
            end
            
            otherAnc = otherAnc.Parent;
        end
        childAnc = childAnc.Parent;
    end
end

end % methods

end

