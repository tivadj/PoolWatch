classdef MultiHypothesisTracker < handle
%MULTIHYPOTHESISTRACKER Summary of this class goes here
%   Detailed explanation goes here

properties
    distanceCompensator;
    
    trackHypothesisForestPseudoNode; % type: TrackHypothesisTreeNode
    
    maxObservationsCountPerFrame; % type: int32
    pruneDepth;
    
    encodedTreeString; % type: int32[] hypothesis tree as string cache
    encodedTreeStringNextIndex; % type: int actual length of the tree as string
    
    trackStatusList; % struct<TrackChangePerFrame> used locally in trackBlobs method only, but 
    frameIndWithTrackInfo; % type: int32
    v;
    %v.nextTrackCandidateId; % type: int32
    %v.nativeRun; % type: bool, whether to execute native code or Matlab code
end

methods
    
function this = MultiHypothesisTracker(distanceCompensator)
    assert(~isempty(distanceCompensator));
    this.distanceCompensator = distanceCompensator;
    
    this.pruneDepth = int32(5);
    
    % init track hypothesis pseudo root
    % values of (Id,FrameInd,DetectionInd) are used in unique observation Id generation
    this.trackHypothesisForestPseudoNode = TrackHypothesisTreeNode;
    this.trackHypothesisForestPseudoNode.Id = int32(0);
    this.trackHypothesisForestPseudoNode.FrameInd = int32(0);
    this.trackHypothesisForestPseudoNode.DetectionInd = int32(0);
    
    this.maxObservationsCountPerFrame = 1000;
    assert(this.trackHypothesisForestPseudoNode.DetectionInd < this.maxObservationsCountPerFrame);

    this.v.swimmerMaxSpeed = 2.3; % max speed for swimmers 2.3m/s

    % new tracks are allocated for detections further from any existent tracks by this distance
    this.v.minDistToNewTrack = 0.5;

    % cache of track results
    this.trackStatusList = struct(TrackChangePerFrame);

    this.purgeMemory();
end

function purgeMemory(this)
    this.v.nativeRun = true;
    this.v.nextTrackCandidateId=int32(1);

    this.trackHypothesisForestPseudoNode.clearChildren;
    
    this.encodedTreeString = zeros(1, 5000000, 'int32');
end

function [frameIndWithTrackInfo,trackStatusListResult] = trackBlobs(this, frameInd, elapsedTimeMs, fps, frameDetections, image, debug)
    assert(length(frameDetections) <= this.maxObservationsCountPerFrame, 'observationCount exceeds the limit, used in unique ObservationId generation');

    % clear result cache
    this.trackStatusList(:) = [];
    
    shapeCentroidNoise = 0.5;
    %swimmerMaxShiftPerFrameM = elapsedTimeMs * this.v.swimmerMaxSpeed / 1000 + this.humanDetector.shapeCentroidNoise;
    swimmerMaxShiftPerFrameM = elapsedTimeMs * this.v.swimmerMaxSpeed / 1000 + shapeCentroidNoise;
    this.processDetections(image, frameInd, elapsedTimeMs, fps, frameDetections, swimmerMaxShiftPerFrameM, debug);
    
    frameIndWithTrackInfo = this.frameIndWithTrackInfo;
    trackStatusListResult = this.trackStatusList;
end

function processDetections(this, image, frameInd, elapsedTimeMs, fps, frameDetections, swimmerMaxShiftPerFrameM, debug)
    this.growTrackHyposhesisTree(frameInd, frameDetections, elapsedTimeMs, fps, swimmerMaxShiftPerFrameM, debug);
    
    leafSetNew = this.trackHypothesisForestPseudoNode.getLeafSet(false);

    % construct track scores
    
    trackIdToScore = containers.Map('KeyType', 'int32', 'ValueType', 'double');
    trackIdToScoreKalman = containers.Map('KeyType', 'int32', 'ValueType', 'double');
    trackScores = this.calcTrackHypothesisScore(leafSetNew, trackIdToScore, trackIdToScoreKalman);
   
    if debug
        fprintf('grown tree, hypothesisCount=%d\n', length(leafSetNew));
        %this.printHypothesis(leafSetNew, trackIdToScore, trackIdToScoreKalman);
        this.printHypothesis(leafSetNew, [], []);
    end

    % TrackId - TrackObj map
    allTrackIdToObj = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    for i=1:length(leafSetNew)
        allTrackIdToObj(leafSetNew{i}.Id) = leafSetNew{i};
    end
    
    bestTrackLeafs = this.findBestTracks(leafSetNew, trackIdToScore, allTrackIdToObj, debug);

    if debug
        fprintf('best tracks (before pruning)\n');
        this.printHypothesis(bestTrackLeafs, trackIdToScore, trackIdToScoreKalman);
    end

    %
    pruneWindow = this.pruneDepth + 1;
    
    this.collectTrackChanges(frameInd, bestTrackLeafs, pruneWindow);
    
    this.pruneHypothesisTree(bestTrackLeafs, pruneWindow, debug);

    if debug
        fprintf('pruned tree\n');
        this.printHypothesis(bestTrackLeafs, trackIdToScore, trackIdToScoreKalman);
    end
end

function trackScores = calcTrackHypothesisScore(this, leafSetNew, trackIdToScore, trackIdToScoreKalman)
    trackScores = zeros(1, length(leafSetNew));
    for leafSetInd=1:length(leafSetNew)
        leaf = leafSetNew{leafSetInd};
        
        trackSeq = leaf.getPathFromRoot();
        trackSeq(1) = []; % exclude pseudo root from seq

%         [scoreMan,scoreKalman] = this.calcTrackSequenceScoreNew(trackSeq, kalmanFilter);
%         if debug
%             fprintf('root %s leaf %s score=%.4f score2=%.4f\n', trackSeq{1}.briefInfoStr, leaf.briefInfoStr, scoreMan, scoreKalman);
%         end
        
        % TODO: keep asserts
        %assert(scoreMan == leaf.Score);
        %assert(scoreKalman == leaf.ScoreKalman);
        
        scoreKalman = leaf.ScoreKalman;
        
        trackScores(leafSetInd) = scoreKalman;
        trackIdToScore(leaf.Id) = scoreKalman;
        trackIdToScoreKalman(leaf.Id) = scoreKalman;
    end
end

function growTrackHyposhesisTree(this, frameInd, frameDetections, elapsedTimeMs, fps, swimmerMaxShiftPerFrameM, debug)
    addedDueNoObservation = 0;
    addedDueCorrespondence = 0;
    addedNew = 0;

    kalmanFilter = this.createKalmanPredictor([0 0], fps);
    
    leafSet = this.trackHypothesisForestPseudoNode.getLeafSet(false);
    if debug
        fprintf('leafSet before=%d\n', length(leafSet));
    end

    % hypothesis2: track has correspondent observations in this frame
    for leafInd=1:length(leafSet)
        leaf = leafSet{leafInd};

        for blobInd=1:length(frameDetections)
            blob = frameDetections(blobInd);
            centrPix = blob.Centroid;

            % consider only blobs within specified gate
            centrWorld = blob.CentroidWorld;
            dist = norm(leaf.EstimatedWorldPos-centrWorld);
            if dist > swimmerMaxShiftPerFrameM
                continue;
            end
            
            kalmanFilter.State = leaf.KalmanFilterState;
            kalmanFilter.StateCovariance = leaf.KalmanFilterStateCovariance;
            posPredicted = kalmanFilter.predict();
            posEstimate2 = kalmanFilter.correct(centrWorld(1:2));
            posEstimate = [posEstimate2 0]; % z=0

            childHyp = TrackHypothesisTreeNode;
            childHyp.Id = this.v.nextTrackCandidateId;
            this.v.nextTrackCandidateId = this.v.nextTrackCandidateId + 1;
            childHyp.FamilyId = leaf.FamilyId;
            childHyp.DetectionInd = int32(blobInd);
            childHyp.FrameInd = frameInd;
            childHyp.ObservationPos = centrPix;
            childHyp.ObservationWorldPos = centrWorld;
            childHyp.EstimatedWorldPos = posEstimate;
            childHyp.CreationReason = TrackHypothesisTreeNode.SequantialCorrespondence;
            childHyp.KalmanFilterState = kalmanFilter.State;
            childHyp.KalmanFilterStateCovariance = kalmanFilter.StateCovariance;
            childHyp.KalmanFilterStatePrev = leaf.KalmanFilterState;
            childHyp.KalmanFilterStateCovariancePrev = leaf.KalmanFilterStateCovariance;
            
            % calculate score
            [scorePart, scoreKalmanPart] = this.calcTrackShiftScore(leaf, childHyp, kalmanFilter);
            childHyp.Score = leaf.Score + scorePart;
            childHyp.ScoreKalman = leaf.ScoreKalman + scoreKalmanPart;

            leaf.addChild(childHyp);
            addedDueCorrespondence = addedDueCorrespondence + 1;
            
            if debug
                %fprintf('correspondence leaf %s hyp%s\n', leaf.briefInfoStr, childHyp.briefInfoStr);
            end
        end
    end
    
    % hypothesis1: (no observation) track has no associated observation in this frame
    if true
        for leafInd=1:length(leafSet)
            leaf = leafSet{leafInd};

            kalmanFilter.State = leaf.KalmanFilterState;
            kalmanFilter.StateCovariance = leaf.KalmanFilterStateCovariance;
            predictedPos2 = kalmanFilter.predict();
            predictedPos = [predictedPos2 0]; % z=0
            
            % perf oversimplification
            % prohibit hypothesis with predicted position close to another observation;
            % it assumes that swimmers can't be too close to each other
            hasCloseObservations = false;
            for i=1:length(frameDetections)
                blob = frameDetections(blobInd);

                % consider only blobs within specified gate
                dist = norm(predictedPos-blob.CentroidWorld);
                if dist < this.v.minDistToNewTrack
                    hasCloseObservations = true;
                    break;
                end
            end
            
            if hasCloseObservations
                continue;
            end
            
            % create hypothesis node

            childHyp = TrackHypothesisTreeNode;
            childHyp.Id = this.v.nextTrackCandidateId;
            this.v.nextTrackCandidateId = this.v.nextTrackCandidateId + 1;
            childHyp.FamilyId = leaf.FamilyId;
            childHyp.DetectionInd = int32(-1);
            childHyp.FrameInd = frameInd;
            childHyp.ObservationPos = [];
            childHyp.ObservationWorldPos = [];
            childHyp.EstimatedWorldPos = predictedPos;
            childHyp.CreationReason = TrackHypothesisTreeNode.NoObservation;
            childHyp.KalmanFilterState = kalmanFilter.State;
            childHyp.KalmanFilterStateCovariance = kalmanFilter.StateCovariance;
            childHyp.KalmanFilterStatePrev = leaf.KalmanFilterState;
            childHyp.KalmanFilterStateCovariancePrev = leaf.KalmanFilterStateCovariance;

            % calculate score
            [scorePart, scoreKalmanPart] = this.calcTrackShiftScore(leaf, childHyp, kalmanFilter);
            childHyp.Score = leaf.Score + scorePart;
            childHyp.ScoreKalman = leaf.ScoreKalman + scoreKalmanPart;

            leaf.addChild(childHyp);
            addedDueNoObservation = addedDueNoObservation + 1;

            if debug
                %fprintf('noObs leaf %s hyp%s\n', leaf.briefInfoStr, childHyp.briefInfoStr);
            end
        end
    end
    
    % hypothesis3: (new track) track got the initial observation in this frame
    % perfomance oversimplification:
    % try to initiate new track sparingly (each N frames)
    
    initNewTrackDelay = 60;
    if mod(frameInd-1, initNewTrackDelay) == 0
        for blobInd=1:length(frameDetections)
            blob = frameDetections(blobInd);
            centrWorld = blob.CentroidWorld;

            % create hypothesis node
            
            childHyp = TrackHypothesisTreeNode;
            childHyp.Id = this.v.nextTrackCandidateId;
            this.v.nextTrackCandidateId = this.v.nextTrackCandidateId + 1;
            childHyp.FamilyId = childHyp.Id;
            childHyp.DetectionInd = int32(blobInd);
            childHyp.FrameInd = frameInd;
            childHyp.ObservationPos = blob.Centroid;
            childHyp.ObservationWorldPos = centrWorld;
            childHyp.EstimatedWorldPos = centrWorld;
            childHyp.CreationReason = TrackHypothesisTreeNode.New;
            childHyp.KalmanFilterState = [centrWorld(1:2) 0 0]; % vx=0 vy=0
            childHyp.KalmanFilterStateCovariance = eye(length(childHyp.KalmanFilterState));
            childHyp.KalmanFilterStatePrev = []; % no previous state
            childHyp.KalmanFilterStateCovariancePrev = [];

            % calculate score
            [scorePart, scoreKalmanPart] = this.calcTrackShiftScore([], childHyp, kalmanFilter);
            childHyp.Score = scorePart;
            childHyp.ScoreKalman = scoreKalmanPart;

            this.trackHypothesisForestPseudoNode.addChild(childHyp);
            addedNew = addedNew + 1;

            if debug
                %fprintf('new hyp%s\n', childHyp.briefInfoStr);
            end
        end
    end
    
    if debug
        fprintf('addedDueNoObservation=%d\n', addedDueNoObservation);
        fprintf('addedDueCorrespondence=%d\n', addedDueCorrespondence);
        fprintf('addedNew=%d\n', addedNew);
    end
end

function result = isPseudoRoot(this, trackTreeNode)
    result = (trackTreeNode.Id == this.trackHypothesisForestPseudoNode.Id);
end

% print optimal tracks in ascending rootId order
function printHypothesis(this, bestTrackLeafs, trackIdToScore, trackIdToScoreKalman)
    allTrackIdToObj = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    leafIdRootId = zeros(length(bestTrackLeafs),2, 'int32');
    
    for i=1:length(bestTrackLeafs)
        leaf = bestTrackLeafs{i};
        leafId = leaf.Id;
        allTrackIdToObj(leafId) = leaf;
        
        leafIdRootId(i,1) = leafId;
        leafIdRootId(i,2) = leaf.FamilyId;;
    end 
    leafIdRootId = sortrows(leafIdRootId, 2);

    for leafId = reshape(leafIdRootId(:,1), 1, [])
        leaf = allTrackIdToObj(leafId);

        path = leaf.getPathFromRoot;
        path(1) = [];
        
        root = path{1};
        fprintf('F%d Len=%d', root.FamilyId, length(path));
        
        leafParentStr = 'nil';
        if ~isempty(leaf.Parent) && ~this.isPseudoRoot(leaf.Parent)
            leafParentStr = leaf.Parent.briefInfoStr;
        end
        fprintf(' LeafParent=%s', leafParentStr);
        
        fprintf(' Leaf=%s', leaf.briefInfoStr);
        
        if ~isempty(trackIdToScore)
            score = trackIdToScore(leaf.Id);
            fprintf(' score=%.4f', score);
        end

        fprintf(' score*=%.4f', leaf.ScoreKalman);

        if ~isempty(trackIdToScore)
            scoreK = trackIdToScoreKalman(leaf.Id);
            fprintf(' scoreK=%.4f', scoreK);
            
            parentScoreK = 0;
            if ~isempty(leaf.Parent) && ~this.isPseudoRoot(leaf.Parent)
                % TODO: store score in tree node
                %parentScoreK = trackIdToScoreKalman(leaf.Parent.Id);
                parentScoreK = leaf.Parent.ScoreKalman;
            end
            deltaScoreK = scoreK - parentScoreK;
            fprintf(' ds=%.4f', deltaScoreK);
        end

        fprintf(' [');
        for i=1:length(path)
            fprintf('%s ', path{i}.briefInfoStr);
        end
        fprintf(']\n');
    end
end

function bestTrackLeaves = findBestTracks(this, leafSetNew, trackIdToScore, allTrackIdToObj, debug)
    allTrackIds = cellfun(@(c) c.Id, leafSetNew);

%     tic;
%     g1 = this.createTrackIncopatibilityGraph(leafSetNew, debug);
%     t1 =  toc;
%     if debug
%         fprintf('createTrackIncopatibilityGraph MatLab time=%d\n', t1);
%     end
    
    tic;
    g2 = this.createTrackIncopatibilityGraphDLang(debug);
    t2 =  toc;
    if debug
        fprintf('createTrackIncopatibilityGraph DLang time=%d\n', t2);
    end

%     g11=[min(g1,[],2) max(g1,[],2)];
%     g22=[min(g2,[],2) max(g2,[],2)];
%     
%     normIncompatMat1 = sortrows(g11, [1,2]);
%     normIncompatMat2 = sortrows(g22, [1,2]);
%     
%     assert(all(normIncompatMat1(:) == normIncompatMat2(:)));
    
    incompatibTrackEdgesMat = g2;
    
    % the graph may have isolated and connected vertices
    
    connectedEdgesCount = size(incompatibTrackEdgesMat,1);
    connectedVertices = int32(reshape(unique(sort(incompatibTrackEdgesMat(:))),1,[]));
    connectedVertexCount=length(connectedVertices);
    
    % select isolated vertices which automatically constitute the solution
    isolatedVertices = setdiff(allTrackIds, connectedVertices);
    
    % all isolated vertices are automatically part of the solution
    bestTrackLeaves = cell(1,0);
    
    for trackId=isolatedVertices
        bestTrackLeaves{end+1} = allTrackIdToObj(trackId);
    end

    if connectedEdgesCount == 0
        return;
    end

    if debug
        componentsCount = utils.PW.connectedComponentsCountNative(incompatibTrackEdgesMat);
        fprintf('V=%d E=%d ConnCompCount=%d\n', connectedVertexCount, connectedEdgesCount, componentsCount);
    end

    % vertexWeights
    
    vertexWeights = zeros(1, connectedVertexCount);
    for vertexInd=1:connectedVertexCount
        vertexId = connectedVertices(vertexInd);
        %score = trackIdToScore(vertexId);
        hypNode = allTrackIdToObj(vertexId);
        score = hypNode.ScoreKalman;
        vertexWeights(vertexInd) = score;
    end

    assert(~isempty(connectedVertices));
    
    % approximite solution for Independent Set Problem
    if debug
        fprintf('PWMaxWeightInependentSetMaxFirst...');
    end
    tic;
    indepVerticesMask = PWMaxWeightInependentSetMaxFirst(connectedVertices, incompatibTrackEdgesMat, vertexWeights);
    indepVerticesMask = double(indepVerticesMask); % required by bintprog
    approxMWISPTime=toc;

    if debug
        fprintf('IndepSet (approx) weights = %f time=%.2f\n', sum(vertexWeights .* indepVerticesMask), approxMWISPTime);
    end
    
    %
    allowExactMWISP = connectedEdgesCount  < 200; % try exact solution for small problems only
    if allowExactMWISP
        otimizationMaxTime = 5;
        initialSolution = indepVerticesMask;
        exactIndepVertices = this.maxWeightIndependentSetByLinearOptimization(connectedVertices, vertexWeights, incompatibTrackEdgesMat, initialSolution, otimizationMaxTime, debug);
        if ~isempty(exactIndepVertices)
            indepVerticesMask = exactIndepVertices;
        end
    end

    %
    optimTrackIdInds = find(indepVerticesMask);
    optimTrackIds = connectedVertices(optimTrackIdInds);
    
    if debug
        fprintf('MWIS cardinality=%d\n', length(optimTrackIdInds));
    end
    
    % result solution = isolated vertices + result of MWISP
    
    for trackId=optimTrackIds
        bestTrackLeaves{end+1} = allTrackIdToObj(trackId);
    end
end

% Searches for solution for MWISP using linear programming formulation. Returns mask of vertices 
% in the independent set or null if solution was not found in 'otimizationMaxTime' timeframe.
function indepVerticesMask = maxWeightIndependentSetByLinearOptimization(this, connectedVertices, vertexWeights, incompatibTrackEdgesMat, initialSolution, otimizationMaxTime, debug)
    % trackIdToVertexInd normalized trackId into zero-based indices

    connectedVertexCount=length(connectedVertices);
    connectedEdgesCount = size(incompatibTrackEdgesMat,1);
    trackIdToVertexInd = containers.Map('KeyType', 'int32', 'ValueType', 'int32');
    for vertexInd=1:connectedVertexCount
        leafId = connectedVertices(vertexInd);
        trackIdToVertexInd(leafId) = vertexInd;
    end

    % build constraint matrix A

    A = zeros(connectedEdgesCount,connectedVertexCount); % costraints on edges
    for edgeInd=1:connectedEdgesCount
        trackIdFrom = incompatibTrackEdgesMat(edgeInd, 1);
        trackIdTo   = incompatibTrackEdgesMat(edgeInd, 2);

        vertexIndFrom = trackIdToVertexInd(trackIdFrom);
        vertexIndTo   = trackIdToVertexInd(trackIdTo);

        A(edgeInd, vertexIndFrom) = 1;
        A(edgeInd, vertexIndTo  ) = 1;
    end
    b = ones(connectedEdgesCount,1);


    % find set of tracks with maximal weight
    % bintprog minimizes objective function but MWISP maximizes => put minus sign in obj fun
    opt = optimoptions('bintprog');
    opt.MaxTime = otimizationMaxTime;
    [x,fval,exitflag,output] = bintprog(-vertexWeights, A, b, [], [], initialSolution, opt);
    if exitflag==1
        % solution found
        indepVerticesMask = reshape(x, 1, []);
        if debug
            fprintf('IndepSet (exact) weight=%f bintprog time=%f\n', -fval, output.time);
        end
    else
        indepVerticesMask = [];
        fprintf('bintprog timeout, fallback to rough solution\n');
    end
end

function incompatibTrackEdgesMat = createTrackIncopatibilityGraph(this, leafSetNew, debug)
    oneHypObsIds=java.util.HashSet;
    othHypObsIds=java.util.HashSet;
    
    % construct track incompatibility lists
    leafCount = length(leafSetNew);
    edgesCountMax = leafCount*(leafCount-1)/2;
    incompatibTrackEdgesMat = zeros(edgesCountMax,2,'int32');
    edgeInd = 1;
    for leafInd=1:leafCount
        leaf = leafSetNew{leafInd};
        
        oneHypObsIds.clear;
        this.populateHypothesisObservationIds(leaf, oneHypObsIds);
        
        for othLeafInd=leafInd+1:leafCount
            othLeaf = leafSetNew{othLeafInd};

            %hasCommonObs = utils.PW.hasCommonObservation(leaf, othLeaf);
            
            %
            othHypObsIds.clear;
            this.populateHypothesisObservationIds(othLeaf, othHypObsIds);
            othHypObsIds.retainAll(oneHypObsIds); % get 'intersection'
            
            % two trajectories (hypotheses) are incompatible if they were assigned common observation
            hasCommonObsJavaImpl = ~othHypObsIds.isEmpty;
            %assert(hasCommonObs == hasCommonObsJavaImpl);            
            hasCommonObs = hasCommonObsJavaImpl;

            if hasCommonObs
                % two tracks are incompatible => there must be edge in the graph
                % hence, independent set (ISP P=problem) solution will select only compatible vertices
                
                assert(edgeInd <= edgesCountMax);
                incompatibTrackEdgesMat(edgeInd,:) = [leaf.Id othLeaf.Id]; % track ids
                edgeInd = edgeInd + 1;
                if debug
                    %fprintf('edge %s-%s\n', leaf.briefInfoStr, othLeaf.briefInfoStr);
                end
            end
        end
    end
    incompatibTrackEdgesMat(edgeInd:end,:) = [];
end

function incompatibTrackEdgesMat = createTrackIncopatibilityGraphDLang(this, debug)
    % encode hypothesis tree into string
    
    this.encodedTreeStringNextIndex = 1;
    this.hypothesisTreeToTreeStringRec(this.trackHypothesisForestPseudoNode, debug);
    treeStr = this.encodedTreeString(1,1:this.encodedTreeStringNextIndex-1);
    
    incompatibTrackEdgesMat = PWComputeTrackIncopatibilityGraph(treeStr);
    incompatibTrackEdgesMat = reshape(incompatibTrackEdgesMat, 2, [])';
end

function reserveEncodedTreeString(this, newSize)
    % resize cache
    cacheSize = length(this.encodedTreeString);
    if newSize > cacheSize
        newCacheSize = int32(cacheSize * 2);
        newCache = zeros(1,newCacheSize, 'int32');

        newCache(1,1:this.encodedTreeStringNextIndex-1) = this.encodedTreeString(1,1:this.encodedTreeStringNextIndex-1);
        this.encodedTreeString = newCache;
    end
end

function hypothesisTreeToTreeStringRec(this, startFromNode, debug)
    compoundId = compoundObservationId(this, startFromNode);

    % 4 items: start node id, obs id, open and close brackets
    this.reserveEncodedTreeString(this.encodedTreeStringNextIndex + 4);
    
    i = this.encodedTreeStringNextIndex;
    this.encodedTreeString(1,i+0) = startFromNode.Id;
    this.encodedTreeString(1,i+1) = compoundId;
    this.encodedTreeStringNextIndex = i + 2;
    
    openBracket = -1;
    closeBracket = -2;

    % construct edge list representation of hypothesis graph
    childrenCount = length(startFromNode.Children);
    if childrenCount > 0
        this.encodedTreeString(1,this.encodedTreeStringNextIndex) = openBracket;
        this.encodedTreeStringNextIndex = this.encodedTreeStringNextIndex + 1;
        
        for i=1:childrenCount
            child = startFromNode.Children{i};       
            this.hypothesisTreeToTreeStringRec(child, debug);
        end
        
        this.encodedTreeString(1,this.encodedTreeStringNextIndex) = closeBracket;
        this.encodedTreeStringNextIndex = this.encodedTreeStringNextIndex + 1;
    end
end

% hypObsIds (type java.util.HashSet) ids of all observations except 0 (which is ID for 'missed observation')
function populateHypothesisObservationIds(this, leaf, hypObsIds)

    cur = leaf;
    while (true)
        assert(~isempty(cur), 'Each node has the parent or pseudo root');
        
        if this.isPseudoRoot(cur)
            break;
        end
        if isempty(cur.DetectionInd) % skip 'missed observation' nodes
            continue;
        end
        
        compoundId = this.compoundObservationId(cur);        
        hypObsIds.add(compoundId);
        
        cur = cur.Parent;
    end
end

function compoundId = compoundObservationId(this, hypothesisNode)
    % put together FrameInd+DetectionInd as a unique id of observation
    % NOTE: assume observationsCount < 1000 for each frame
    compoundId = int32(hypothesisNode.FrameInd * this.maxObservationsCountPerFrame + hypothesisNode.DetectionInd);
end

% Fix track hypothesis (if any) into track records.
function collectTrackChanges(this, frameInd, bestTrackLeafs, pruneWindow)
    % gather new tracks or get track assignments
    readyFrameInd = frameInd - pruneWindow;
    if readyFrameInd < 1
        readyFrameInd = -1;
    end
    this.frameIndWithTrackInfo = readyFrameInd;
    
    for i=1:length(bestTrackLeafs)
        leaf = bestTrackLeafs{i};
        
        % find ancestor to remove from hypothesis tree
        
        fixedAncestor = leaf.getAncestor(pruneWindow);
        if isempty(fixedAncestor)
            continue;
        end
        if this.isPseudoRoot(fixedAncestor)
            continue;
        end
        
        % find corresponding track record

        change = TrackChangePerFrame();
        change.TrackCandidateId = fixedAncestor.FamilyId;

        if fixedAncestor.CreationReason == TrackHypothesisTreeNode.New
            change.UpdateType = TrackChangePerFrame.New;
        elseif fixedAncestor.CreationReason == TrackHypothesisTreeNode.SequantialCorrespondence
            change.UpdateType = TrackChangePerFrame.ObservationUpdate;
        elseif fixedAncestor.CreationReason == TrackHypothesisTreeNode.NoObservation
            change.UpdateType = TrackChangePerFrame.NoObservation;
        end
        
        estimatedPos = fixedAncestor.EstimatedWorldPos;
        change.EstimatedPosWorld = estimatedPos;
        
        change.ObservationInd = fixedAncestor.DetectionInd;
        obsPos = fixedAncestor.ObservationPos;
        if isempty(obsPos)
            obsPos = this.distanceCompensator.worldToCamera(estimatedPos);            
        end
        change.ObservationPosPixExactOrApprox = single(obsPos);
        this.trackStatusList(end+1) = struct(change);
    end
end

% Performs N-scan pruning.
function pruneHypothesisTree(this, leafSet, pruneWindow, debug)
    prunedTreeSet = cell(1,0);
    for leafInd = 1:length(leafSet)
        leaf = leafSet{leafInd};
        
        % find new family root
        newRoot = leaf;
        stepBack = 1;
        while true
            if stepBack == pruneWindow
                break;
            end
            
            % stop if parent is the pseudo root
            if isempty(newRoot.Parent) || newRoot.Parent.Id == this.trackHypothesisForestPseudoNode.Id
                break;
            end
            
            newRoot = newRoot.Parent;
            stepBack = stepBack + 1;
        end
        
        % if tree is shallow - no need to prune it
        if true
            prunedTreeSet{end+1} = newRoot;
        end
    end
    
    this.trackHypothesisForestPseudoNode.clearChildren;
    for i=1:length(prunedTreeSet)
        this.trackHypothesisForestPseudoNode.addChild(prunedTreeSet{i});
    end
end

function kalman = createKalmanPredictor(obj, initPos2D, fps)
    % init Kalman Filter
    stateModel = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1]; 
    measurementModel = [1 0 0 0; 0 1 0 0]; 
    kalman = vision.KalmanFilter(stateModel, measurementModel);
    
    % max shift per frame = 20cm
    %kalman.ProcessNoise = (0.2 / 3)^2;
    obj.setTrackKalmanProcessNoise(kalman, 'normal', fps);
    
    % max measurment error per frame = 1m (far away it can be 5m)
    kalman.MeasurementNoise = (5 / 3)^2;
    kalman.State = [initPos2D 0 0]; % v0=0
end

function setTrackKalmanProcessNoise(obj, kalman, swimmerLocactionStr, fps)
    % 2.3m/s is max speed for swimmers
    % let say 0.5m/s is an average speed
    % estimate sigma as one third of difference between max and mean shift per frame
    maxShiftM = 2.3 / fps;
    meanShiftM = 0.5 / fps;
    sigma = (maxShiftM - meanShiftM) / 3;

    if strcmp(swimmerLocactionStr, 'normal')
        kalman.ProcessNoise = sigma^2;
    elseif strcmp(swimmerLocactionStr, 'nearBorder')
        kalman.ProcessNoise = 999^2; % suppress process model
    end
end

function score = calcTrackSequenceScore(this, trackSeq)
    dist = 0;
    
    if isempty(trackSeq)
        return;
    end
    
    oldCenter = trackSeq{1}.EstimatedPos;
    trackLen = 1;
    
    for i=2:length(trackSeq)
        trackLen = trackLen + 1;
        
        pos = trackSeq{i}.EstimatedPos;
        distPart = norm(pos - oldCenter);
        
        dist = dist + distPart;
        oldCenter = pos;
    end
    
    if trackLen == 1
        score = 0;
    else
        maxScore = 9999;
        if dist < 0.0001
            score = maxScore
        else
            score = 1 / dist;
        end
    end
end

function [scorePart, kalmanScorePart] = calcTrackShiftScore(this, parentTrackNode, trackNode, kalmanFilter)
    % penalty for missed observation
    % prob 0.4 - penalty -0.9163
    % prob 0.6 - penalty -0.5108
    probDetection = 0.6;
    penalty = log(1 - probDetection);

    % initial track score
    nt = 5; % number of expected targets
    fa = 25; % number of expected FA (false alarms)
    precision = nt / (nt + fa);
    initialScore = -log(precision);

    % if initial score is large, then tracks with missed detections may
    % be evicted from hypothesis tree
    initialScore = abs(6*penalty);

    first = trackNode;
    if first.CreationReason == TrackHypothesisTreeNode.New
        assert(isempty(parentTrackNode));
        
        kalmanScorePart = initialScore;
        scorePart = initialScore;
    else
        assert(~isempty(parentTrackNode));
        kalmanFilter.State = parentTrackNode.KalmanFilterState;
        kalmanFilter.StateCovariance = parentTrackNode.KalmanFilterStateCovariance;
        
        predictedPos2 = kalmanFilter.predict(); % [X,Y]
        predictedPos = [predictedPos2 0]; % z=0
        
        worldPos = first.EstimatedWorldPos;

%         p1 = trackSeq{i}.ObservationWorldPos;
%         if isempty(p1)
%             p1 = trackSeq{i}.EstimatedWorldPos;
%         end

        p1 = first.ObservationWorldPos;
        if ~isempty(p1)
            % got observation
            
            kalmanScorePart = kalmanFilter.distance(p1(1:2)); 

            distPart = norm(worldPos - predictedPos);
            if distPart < 0.001
                scorePart = 10;
            else
                scorePart = 1 / distPart;
            end
            %sig = this.v.swimmerMaxSpeed / 3;
            %(1/(2*pi*sig))*exp(-0.5*distPart^2)
        else
            kalmanScorePart = penalty;
            scorePart = penalty;
        end        
    end
end

function discardTrackCore(this, trackCandidateId)
    % ignore clean up requests
end

function [score,kalmanScore] = calcTrackSequenceScoreNew(this, trackSeq, kalmanFilter)
    score = 0;
    kalmanScore = 0;

    % penalty for missed observation
    % prob 0.4 - penalty -0.9163
    % prob 0.6 - penalty -0.5108
    probDetection = 0.6;
    penalty = log(1 - probDetection);

    % initial track score
    nt = 5; % number of expected targets
    fa = 25; % number of expected FA (false alarms)
    precision = nt / (nt + fa);
    initialScore = -log(precision);

    % if initial score is large, then tracks with missed detections may
    % be evicted from hypothesis tree
    initialScore = abs(6*penalty);

    first = trackSeq{1};
    if first.CreationReason == TrackHypothesisTreeNode.New
        kalmanScore = initialScore;
        score = initialScore;

        startFrom = 2;
        % first element of track may not have prev Kalman Filter state
        kalmanFilter.State = first.KalmanFilterState;
        kalmanFilter.StateCovariance = first.KalmanFilterStateCovariance;
        
        %
        [a1,a2] = this.calcTrackShiftScore([], first, kalmanFilter);
        assert(a2 == kalmanScore);
    else
        startFrom = 1;

        assert(~isempty(first.KalmanFilterStatePrev), 'Subsequent track node must have Kalman Filter state');
        kalmanFilter.State = first.KalmanFilterStatePrev;
        kalmanFilter.StateCovariance = first.KalmanFilterStateCovariancePrev;
    end
    
    for i=startFrom:length(trackSeq)
        curNode = trackSeq{i};
        predictedPos2 = kalmanFilter.predict(); % [X,Y]
        predictedPos = [predictedPos2 0]; % z=0
        
        worldPos = curNode.EstimatedWorldPos;

%         p1 = trackSeq{i}.ObservationWorldPos;
%         if isempty(p1)
%             p1 = trackSeq{i}.EstimatedWorldPos;
%         end

        % check world pos is correct
%         if ~isempty(curNode.ObservationWorldPos)
%         leaf = curNode.Parent;
%         assert(~this.isPseudoRoot(leaf));
%         centrWorld = curNode.ObservationWorldPos;
%         kalmanFilter2 = this.createKalmanPredictor([0 0], 30);
%         kalmanFilter2.State = leaf.KalmanFilterState;
%         kalmanFilter2.StateCovariance = leaf.KalmanFilterStateCovariance;
%         posPredicted = kalmanFilter2.predict();
%         posEstimate2 = kalmanFilter2.correct(centrWorld(1:2));
%         posEstimate = [posEstimate2 0]; % z=0
%         assert(all(posEstimate == worldPos));
%         end


        observWorld = curNode.ObservationWorldPos;
        if ~isempty(observWorld)
            % got observation
            
            %kalmanScorePart = kalmanFilter.distance(observWorld(1:2)); 
            kalmanScorePart = kalmanDistance(kalmanFilter, observWorld(1:2));
            corrected = kalmanFilter.correct(observWorld(1:2));
            corrected = [corrected 0];
            %assert(all(corrected == worldPos)); % TODO:

            distPart = norm(worldPos - predictedPos);
            if distPart < 0.001
                scorePart = 10;
            else
                scorePart = 1 / distPart;
            end
            %sig = this.v.swimmerMaxSpeed / 3;
            %(1/(2*pi*sig))*exp(-0.5*distPart^2)
        else
            kalmanScorePart = penalty;
            scorePart = penalty;
        end        
        
%         curNodeParent = curNode.Parent;
%         if this.isPseudoRoot(curNodeParent)
%             curNodeParent = [];
%         end
%         [a1,a2] = this.calcTrackShiftScore(curNodeParent, curNode, kalmanFilter);
        %assert(a1 == scorePart); % TODO:
        %assert(a2 == kalmanScorePart);
        
        kalmanScore = kalmanScore + kalmanScorePart;
        score = score + scorePart;
    end
    
%     maxScore = 9999;
%     if dist < 0.0001
%         score = maxScore;
%     else
%         score = 1 / dist;
%     end
end

end
    
end

