classdef SwimmerTracker < handle
properties
    frameInd;                 % type:int, number of processed frames
    detectionsPerFrame; % type: List<ShapeAssignment>
    tracks; % type: List<TrackedObject>
    %v.nextTrackId;
    % v.nextTrackCandidateId;
    poolRegionDetector;
    distanceCompensator;
    humanDetector;
    colorAppearance;
    trackHypothesisForestPseudoNode; % type: TrackHypothesisTreeNode
    v;
    %v.swimmerMaxSpeed;

    %v.encodedTreeString; % type: int32[] hypothesis tree as string cache
    %v.encodedTreeStringSize; % type: int actual length of the tree as string
    maxObservationsCountPerFrame; % type: int32
end

methods
    
function this = SwimmerTracker(poolRegionDetector, distanceCompensator, humanDetector, colorAppearance)
    assert(~isempty(poolRegionDetector));
    assert(~isempty(distanceCompensator));
    assert(~isempty(humanDetector));
    assert(~isempty(colorAppearance));    
    
    this.poolRegionDetector = poolRegionDetector;
    this.distanceCompensator = distanceCompensator;
    this.humanDetector = humanDetector;
    this.colorAppearance = colorAppearance;
    
    % configure tracker
    this.v.swimmerMaxSpeed = 2.3; % max speed for swimmers 2.3m/s

    % new tracks are allocated for detections further from any existent tracks by this distance
    this.v.minDistToNewTrack = 0.5;
    
    this.v.pruneDepth = 5;
    
    % init track hypothesis pseudo root
    % values of (Id,FrameInd,DetectionInd) are used in unique observation Id generation
    this.trackHypothesisForestPseudoNode = TrackHypothesisTreeNode;
    this.trackHypothesisForestPseudoNode.Id = 0;
    this.trackHypothesisForestPseudoNode.FrameInd = 0;
    this.trackHypothesisForestPseudoNode.DetectionInd = 0;
    
    this.maxObservationsCountPerFrame = 1000;
    assert(this.trackHypothesisForestPseudoNode.DetectionInd < this.maxObservationsCountPerFrame);

    purgeMemory(this);
end


function purgeMemory(obj)
    obj.tracks=cell(0,0);
    obj.v.nextTrackId = 1;
    obj.frameInd = 0;
    obj.v.nextTrackCandidateId=1;
    obj.trackHypothesisForestPseudoNode.clearChildren;
    
    obj.v.encodedTreeString = zeros(1, 5000000, 'int32');
    obj.v.encodedTreeStringSize = 1;
end

% elapsedTimeMs - time in milliseconds since last frame
function nextFrame(this, image, elapsedTimeMs, fps, debug)
    this.frameInd = this.frameInd + 1;

    %
    waterMask = this.poolRegionDetector.getWaterMask(image);
    if debug
        imshow(utils.applyMask(image, waterMask));
    end

    if isfield(this.v, 'poolMask')
        poolMask = this.v.poolMask;
    else
        poolMask = this.poolRegionDetector.getPoolMask(image, waterMask, false, debug);
        if debug
            imshow(utils.applyMask(image, poolMask));
        end
        this.v.poolMask = poolMask;
    end
    
    if true || ~isfield(this.v, 'dividersMask')
        dividersMask = this.poolRegionDetector.getLaneDividersMask(image, poolMask, waterMask, debug);
        if debug
            imshow(utils.applyMask(image, dividersMask));
        end
    end
    
    % narrow observable area to pool boundary
    % remove lane dividers from observation
    imageSwimmers = utils.applyMask(image, poolMask & ~dividersMask);
    if debug
        imshow(imageSwimmers);
    end

    % find shapes
    
    bodyDescrs = this.humanDetector.GetHumanBodies(this.frameInd, imageSwimmers, waterMask, debug);
    this.detectionsPerFrame{this.frameInd} = bodyDescrs;
    
    if debug
        blobsCount = length(bodyDescrs);
        fprintf('blobsCount=%d\n', blobsCount);
        for i=1:blobsCount
            centr = bodyDescrs(i).Centroid;
            fprintf('Blob[%d] Centroid=[%.0f %.0f]\n', i, centr(1), centr(2));
        end
    end
    
    assert(length(bodyDescrs) <= this.maxObservationsCountPerFrame, 'observationCount exceeds the limit, used in unique ObservationId generation');
    
    % track shapes

    processDetections(this, this.frameInd, elapsedTimeMs, fps, bodyDescrs, imageSwimmers, debug);
end

function processDetections(this, frameInd, elapsedTimeMs, fps, frameDetections, image, debug)
    % remember tracks count at the beginning of current frame processing
    % because new tracks may be added
    %tracksCountPrevFrame = length(obj.tracks);
    
    swimmerMaxShiftPerFrameM = elapsedTimeMs * this.v.swimmerMaxSpeed / 1000 + this.humanDetector.shapeCentroidNoise;
    this.processDetections_Mht(image, frameInd, elapsedTimeMs, fps, frameDetections, swimmerMaxShiftPerFrameM, debug);
    return;

    %
    predictSwimmerPositions(this, frameInd, fps);

    trackDetectCost = this.calcTrackToDetectionAssignmentCostMatrix(image, frameInd, frameDetections, swimmerMaxShiftPerFrameM, debug);

    % make assignment using Hungarian algo
    % all unassigned detectoins and unprocessed tracks

    % shapes with lesser proximity should not be assigned
    minAppearPerPixSimilarity = this.colorAppearance.minAppearanceSimilarityScore;
    
    unassignedTrackCost = 1 / minAppearPerPixSimilarity;
    unassignedDetectionCost = unassignedTrackCost;
    
    [assignment, unassignedTracks, unassignedDetections] = assignDetectionsToTracks(trackDetectCost, unassignedTrackCost, unassignedDetectionCost);
    
    if debug && isempty(assignment)
        display(trackDetectCost);
    end

    for i=1:size(assignment,1)
        trackInd = assignment(i,1);
        detectInd = assignment(i,2);

        detect=frameDetections(detectInd);

        ass = this.tracks{trackInd}.Assignments{frameInd};
        ass.IsDetectionAssigned = true;
        ass.DetectionInd = detectInd;

        % estimate position
        imagePos = detect.Centroid;
        ass.v.EstimatedPosImagePix = imagePos;

        % project image position into TopView
        worldPos = this.distanceCompensator.cameraToWorld(imagePos);
        
        worldPos2 = worldPos(1:2);
        posEstimate2 = this.tracks{trackInd}.KalmanFilter.correct(worldPos2);
        posEstimate = [posEstimate2 worldPos(3)];
        
        ass.EstimatedPos = posEstimate;

        this.tracks{trackInd}.Assignments{frameInd} = ass;
        
        %
        this.onAssignDetectionToTrackedObject(this.tracks{trackInd}, detect, image);
   end

    assignTrackCandidateToUnassignedDetections(this, unassignedDetections, frameDetections, frameInd, image, fps);

    driftUnassignedTracks(this, unassignedTracks, frameInd);

    promoteMatureTrackCandidates(this, frameInd);

    % assert EstimatedPos is initialized
    for r=1:length(this.tracks)
        if frameInd > length(this.tracks{r}.Assignments)
            assert(false, 'IndexOutOfRange: frameInd=%d greater than length(assignments)=%d\n', frameInd, length(this.tracks{r}.Assignments));
        end
        
        a = this.tracks{r}.Assignments{frameInd};
        if isempty(a)
            assert(false);
        end

        assert(~isempty(a.EstimatedPos), 'EstimatedPos for track ind=%d ass ind=%d is not initalized\n', r, frameInd);
    end
end

function predictSwimmerPositions(obj, frameInd, fps)
    % predict pos for each tracked object
    for r=1:length(obj.tracks)
        track = obj.tracks{r};

        worldPos2 = track.KalmanFilter.predict();
        worldPos = [worldPos2 0]; % append zeroHeight = 0
        
        %
        ass = ShapeAssignment();
        ass.IsDetectionAssigned = false;
        ass.DetectionInd = -1;
        ass.PredictedPos = worldPos; % take pair (x,y)
        obj.tracks{r}.Assignments{frameInd} = ass;
    end
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

function trackDetectCost = calcTrackToDetectionAssignmentCostMatrix(this, image, frameInd, frameDetections, swimmerMaxShiftPerFrameM, debug)
    % calculate distance from predicted pos of each tracked object
    % to each detection

    trackObjCount = length(this.tracks);
    detectCount = length(frameDetections);
    
    trackDetectCost=zeros(trackObjCount, detectCount);
    for trackInd=1:trackObjCount
        track = this.tracks{trackInd};
        trackedObjPos = track.Assignments{frameInd}.PredictedPos;
        
        if debug
            % recover track pos
            trackImgPos = this.getPrevFrameDetectionOrPredictedImagePos(track, frameInd);
            fprintf('Track Id=%d ImgPos=[%.0f %.0f]\n', track.idOrCandidateId, trackImgPos(1), trackImgPos(2));
        end
        
        % calculate cost of assigning detection to track
        
        for blobInd=1:detectCount
            detect = frameDetections(blobInd);

            maxCost = 99999;
            
            centrPix = detect.Centroid;
            centrWorld = this.distanceCompensator.cameraToWorld(centrPix);
            
            dist=norm(centrWorld-trackedObjPos);
            
            % keep swimmers inside the area of max possible shift per frame
            if dist > swimmerMaxShiftPerFrameM
                dist = maxCost;
            else
                avgProb = this.colorAppearance.similarityScore(track, detect, image);

                dist = min([1/avgProb maxCost]);
                if debug
                    fprintf('calcTrackCost: Blob[%d] [%.0f %.0f] dist=%d avgProb=%d\n', blobInd, centrPix(1), centrPix(2), dist, avgProb);
                end
            end
            
            trackDetectCost(trackInd,blobInd) = dist;
        end
    end
end

function processDetections_Mht(this, image, frameInd, elapsedTimeMs, fps, frameDetections, swimmerMaxShiftPerFrameM, debug)
    this.growTrackHyposhesisTree(frameInd, frameDetections, elapsedTimeMs, fps, swimmerMaxShiftPerFrameM, debug);
    
    leafSetNew = this.trackHypothesisForestPseudoNode.getLeafSet(false);

    % construct track scores
    
    trackIdToScore = containers.Map('KeyType', 'int32', 'ValueType', 'double');
    trackIdToScoreKalman = containers.Map('KeyType', 'int32', 'ValueType', 'double');
    kalmanFilter = this.createKalmanPredictor([0 0], fps);
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
    pruneWindow = this.v.pruneDepth + 1;
    
    this.allocateTrackAssignments(bestTrackLeafs, pruneWindow);
    
    this.pruneHypothesisTree(bestTrackLeafs, pruneWindow, debug);

    if debug
        fprintf('pruned tree\n');
        this.printHypothesis(bestTrackLeafs, trackIdToScore, trackIdToScoreKalman);
    end
end

% returns -1 if there is not available frames.
function queryFrameInd = getFrameIndWithReadyTrackInfo(this)
    queryFrameInd = this.frameInd - this.v.pruneDepth - 1;
    
    if queryFrameInd < 1
        queryFrameInd = -1;
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
            % calculate blob centroid positions
            centrWorld = this.distanceCompensator.cameraToWorld(centrPix);
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
            childHyp.DetectionInd = blobInd;
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
                blobCentr = blob.Centroid;

                % consider only blobs within specified gate
                % calculate blob centroid positions
                blobWorld = this.distanceCompensator.cameraToWorld(blobCentr);
                dist = norm(predictedPos-blobWorld);
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
            childHyp.DetectionInd = -1;
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
            centrPix = blob.Centroid;
            centrWorld = this.distanceCompensator.cameraToWorld(centrPix);

            % create hypothesis node
            
            childHyp = TrackHypothesisTreeNode;
            childHyp.Id = this.v.nextTrackCandidateId;
            this.v.nextTrackCandidateId = this.v.nextTrackCandidateId + 1;
            childHyp.FamilyId = childHyp.Id;
            childHyp.DetectionInd = blobInd;
            childHyp.FrameInd = frameInd;
            childHyp.ObservationPos = centrPix;
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
    
    this.v.encodedTreeStringNextIndex = 1;
    this.hypothesisTreeToTreeStringRec(this.trackHypothesisForestPseudoNode, debug);
    treeStr = this.v.encodedTreeString(1,1:this.v.encodedTreeStringNextIndex-1);
    
    incompatibTrackEdgesMat = PWComputeTrackIncopatibilityGraph(treeStr);
    incompatibTrackEdgesMat = reshape(incompatibTrackEdgesMat, 2, [])';
end

function reserveEncodedTreeString(this, newSize)
    % resize cache
    cacheSize = length(this.v.encodedTreeString);
    if newSize > cacheSize
        newCacheSize = int32(cacheSize * 2);
        newCache = zeros(1,newCacheSize, 'int32');

        newCache(1,1:this.v.encodedTreeStringNextIndex-1) = this.v.encodedTreeString(1,1:this.v.encodedTreeStringNextIndex-1);
        this.v.encodedTreeString = newCache;
    end
end

function hypothesisTreeToTreeStringRec(this, startFromNode, debug)
    compoundId = compoundObservationId(this, startFromNode);

    % 4 items: start node id, obs id, open and close brackets
    this.reserveEncodedTreeString(this.v.encodedTreeStringNextIndex + 4);
    
    i = this.v.encodedTreeStringNextIndex;
    this.v.encodedTreeString(1,i+0) = startFromNode.Id;
    this.v.encodedTreeString(1,i+1) = compoundId;
    this.v.encodedTreeStringNextIndex = i + 2;
    
    openBracket = -1;
    closeBracket = -2;

    % construct edge list representation of hypothesis graph
    childrenCount = length(startFromNode.Children);
    if childrenCount > 0
        this.v.encodedTreeString(1,this.v.encodedTreeStringNextIndex) = openBracket;
        this.v.encodedTreeStringNextIndex = this.v.encodedTreeStringNextIndex + 1;
        
        for i=1:childrenCount
            child = startFromNode.Children{i};       
            this.hypothesisTreeToTreeStringRec(child, debug);
        end
        
        this.v.encodedTreeString(1,this.v.encodedTreeStringNextIndex) = closeBracket;
        this.v.encodedTreeStringNextIndex = this.v.encodedTreeStringNextIndex + 1;
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
function allocateTrackAssignments(this, bestTrackLeafs, pruneWindow)
    % gather new tracks or get track assignments
    
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
        
        track = [];
        if fixedAncestor.CreationReason == TrackHypothesisTreeNode.New
            % allocate new track
            
            track = TrackedObject.NewTrackCandidate(fixedAncestor.FamilyId);
            track.FirstAppearanceFrameIdx = fixedAncestor.FrameInd;
        else
            % find track with FimilyId
            for i1=1:length(this.tracks)
                if this.tracks{i1}.TrackCandidateId == fixedAncestor.FamilyId
                    track = this.tracks{i1};
                    break;
                end
            end
            assert(~isempty(track), 'Track with ancestor.FamilyId must exist');
        end

        %
        %detect = frameDetections(blobInd);

        % project image position into TopView
        %imagePos = detect.Centroid;
        %worldPos = obj.distanceCompensator.cameraToWorld(imagePos);

        %track.KalmanFilter = createKalmanPredictor(obj, worldPos(1:2), fps);

        ass = ShapeAssignment();
        ass.IsDetectionAssigned = fixedAncestor.CreationReason == TrackHypothesisTreeNode.New | fixedAncestor.CreationReason == TrackHypothesisTreeNode.SequantialCorrespondence;
        ass.DetectionInd = fixedAncestor.DetectionInd;
        ass.v.EstimatedPosImagePix = fixedAncestor.ObservationPos;
        ass.PredictedPos = fixedAncestor.EstimatedWorldPos;
        ass.EstimatedPos = fixedAncestor.EstimatedWorldPos;

        track.Assignments{fixedAncestor.FrameInd} = ass;

        %obj.onAssignDetectionToTrackedObject(track, detect, image);

        this.tracks{end+1} = track;
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

function trackImgPos = getPrevFrameDetectionOrPredictedImagePos(this, track, curFrameInd)
    assert(curFrameInd >= 2, 'There is no previous frame for the first frame');
    
    prevFrameAssign = track.Assignments{curFrameInd-1};

    if prevFrameAssign.IsDetectionAssigned
        prevFrameBlobs = this.detectionsPerFrame{curFrameInd-1};
        prevBlob = prevFrameBlobs(prevFrameAssign.DetectionInd);
        trackImgPos = prevBlob.Centroid;
    else
        trackImgPos = this.distanceCompensator.worldToCamera(prevFrameAssign.EstimatedPos);
    end
end

function promoteMatureTrackCandidates(obj, frameInd)
    % One period of breaststroke=90 frames for 30fps video
    % at least half of frames should have detected the human
    trackCandidateMaturingTime = 90; % number of frames to scrutinize if candidate is a human or noise
    trackCandidatePromotionRatio = 8/15; % candidate is promoted to track if is frequently detected
    
    trackCandidateInfancyTime = 5; % number of frames to scrutinize the candidate if it just noise and to be removed early
    trackCandidateInfancyRatio = 1/5+0.01; % candidate is discarded if detected infrequently during infancy time

    for candInd=1:length(obj.tracks)
        cand=obj.tracks{candInd};
        if ~cand.IsTrackCandidate
            continue;
        end

        lifeTime = frameInd - cand.FirstAppearanceFrameIdx + 1;
        detectCount = cand.getDetectionsCount(frameInd);
        detectRatio = detectCount / lifeTime;

        if lifeTime <= trackCandidateInfancyTime
            if detectRatio < trackCandidateInfancyRatio
                % candidate is a noise, discard it
                obj.tracks{candInd}=[]; % mark for deletion
            end
        elseif lifeTime >= trackCandidateMaturingTime
            if detectRatio < trackCandidatePromotionRatio
                % candidate is a noise, discard it
                obj.tracks{candInd}=[]; % mark for deletion
            else
                % candidate is a stable detection, promote to track

                newTrackId = obj.v.nextTrackId;
                obj.v.nextTrackId = obj.v.nextTrackId + 1;

                % promote TrackCandidiate into Track
                obj.tracks{candInd}.Id = newTrackId;
                obj.tracks{candInd}.IsTrackCandidate = false;
                obj.tracks{candInd}.PromotionFramdInd = frameInd;
            end 
        end        
    end

    % clean empty track candidate slots
    for i=length(obj.tracks):-1:1 % goes backward
        if isempty(obj.tracks{i})
            obj.tracks(i) = [];
        end
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

function resetKalmanState(obj, kalmanObj)
    kalmanObj.StateTransitionModel = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1];
end

% associate unassigned detection with track candidate
function assignTrackCandidateToUnassignedDetections(obj, unassignedDetectionsByRow, frameDetections, frameInd, image, fps)
    blobInds=unassignedDetectionsByRow';
    
    farAwayBlobsMask = true(1, length(blobInds));
    
    % avoid creating new tracks for blobs which lie too close to existent tracks
    if frameInd > 1
        for trackInd=1:length(obj.tracks)
            track = obj.tracks{trackInd};

            trackImgPos = obj.getPrevFrameDetectionOrPredictedImagePos(track, frameInd);
            
            
            for blobInd=blobInds(farAwayBlobsMask)
                blob = frameDetections(blobInd);
                dist = norm(trackImgPos - blob.Centroid);
                
                % new tracks are allocated for detections further from any existent tracks by this distance
                if dist < obj.v.minDistToNewTrack
                    farAwayBlobsMask(blobInd) = false;
                end
            end
        end
    end
    
    farAwayBlobs = blobInds(farAwayBlobsMask);
    
    for blobInd=farAwayBlobs
        % new track candidate
        
        cand = TrackedObject.NewTrackCandidate(obj.v.nextTrackCandidateId);
        obj.v.nextTrackCandidateId = obj.v.nextTrackCandidateId + 1;

        cand.FirstAppearanceFrameIdx = frameInd;
        
        %
        detect = frameDetections(blobInd);
        
        % project image position into TopView
        imagePos = detect.Centroid;
        worldPos = obj.distanceCompensator.cameraToWorld(imagePos);

        cand.KalmanFilter = createKalmanPredictor(obj, worldPos(1:2), fps);

        ass = ShapeAssignment();
        ass.IsDetectionAssigned = true;
        ass.DetectionInd = blobInd;
        ass.v.EstimatedPosImagePix = imagePos;
        ass.PredictedPos = worldPos;
        ass.EstimatedPos = worldPos;

        cand.Assignments{frameInd} = ass;
        
        obj.onAssignDetectionToTrackedObject(cand, detect, image);

        obj.tracks{end+1} = cand;
    end
end

function driftUnassignedTracks(this, unassignedTracksByRow, frameInd)
    % for unassigned tracks initialize EstimatedPos = PredictedPos by default
    for trackInd=unassignedTracksByRow'
        a = this.tracks{trackInd}.Assignments{frameInd};
        assert(~isempty(a.PredictedPos), 'Kalman Predictor must have initialized the PredictedPos (track=%d assignment=%d)', trackInd, frameInd);

        estPos = a.PredictedPos;
        a.EstimatedPos = estPos;
        this.tracks{trackInd}.Assignments{frameInd} = a;

        % TODO: finish track if detections lost for some time
        
        %
        x = estPos(1);
        poolSize = this.distanceCompensator.poolSize;
        if x < 1 || x > poolSize(2) - 1
            isNearBorder = true;
        else
            isNearBorder = false;
        end

        % we don't use Kalman filter near pool boundary as swimmers
        % usually reverse their direction here
        if isNearBorder
            kalmanObj = this.tracks{trackInd}.KalmanFilter;
            
            % after zeroing velocity, Kalman predictions would aim single position
            kalmanObj.State(3:end)=0; % zero vx and vy
        end
    end
end

function onAssignDetectionToTrackedObject(this, track, detect, image)
    this.colorAppearance.onAssignDetectionToTrackedObject(track, detect, image);
end

function imageWithTracks = adornImageWithTrackedBodies(obj, image, coordType, queryFrameInd)
    if ~exist('queryFrameInd', 'var')
        queryFrameInd = obj.frameInd;
    end
    
    pathStartFrame = max([1, queryFrameInd - 250]);
    
    detects = obj.detectionsPerFrame{queryFrameInd};
    %imageWithTracks = drawDetections(obj, image, detects);

    if strcmp('TopView', coordType)
        desiredImageSize = [size(image,2), size(image,1)];
        image = obj.distanceCompensator.convertCameraImageToTopView(image, desiredImageSize);
    end

    imageWithTracks = adornTracks(obj, image, pathStartFrame, queryFrameInd, obj.detectionsPerFrame, obj.tracks, coordType);
end

function videoUpd = generateVideoWithTrackedBodies(obj,mediaReader, framesToTake, detectionsPerFrame, tracks)
    fprintf(1,'composing video + tracking');
    tic;

    % show tracks across video
    framesCount = length(framesToTake);
    for timeIndInd=1:framesCount
        frameBegin = tic;
        frameIndOrig = framesToTake(timeIndInd);
        fprintf(1, 'processing frame %d', frameIndOrig);

        image = read(mediaReader, frameIndOrig);
        pathStartFrame = max([1, timeIndInd-100]);
        imgTracks = adornTracks(image, pathStartFrame, timeIndInd, detectionsPerFrame, tracks);

        if (isempty(videoUpd))
            videoUpd = zeros(size(image,1), size(image,2), size(image,3), framesCount, 'uint8');
        end
        fprintf(1,  ' took %f sec\n', toc(frameBegin));
        videoUpd(:,:,:,timeIndInd) = imgTracks;
    end
    fprintf(1, ' took time=%d\n', toc);
end

function imageAdorned = adornTracks(obj, image, fromTime, toTimeInc, detectionsPerFrame, tracks, coordType)
    % show each track
    for r=1:length(tracks)
        track=tracks{r};
        
        % pick track color
        % color should be the same for track candidate and later for the track
        candColor = obj.getTrackColor(track);
        
        % construct track path as polyline to draw by single command
        [candPolyline,initialCandPos] = buildTrackPath(obj, track, fromTime, toTimeInc, coordType);
        
        % draw track path
        if ~isempty(candPolyline)
            image = cv.polylines(image, candPolyline, 'Closed', false, 'Color', candColor);
        end
        
        % draw initial position
        if ~isempty(initialCandPos)
            image=cv.circle(image, initialCandPos, 3, 'Color', candColor);
        end
        
        % process last frame

        lastAss = [];
        % TODO: implement track termination
        if toTimeInc <= length(track.Assignments)
            lastAss = track.Assignments{toTimeInc};
        end
        
        if ~isempty(lastAss)
            if lastAss.IsDetectionAssigned
                % draw shape contour
                frameDetects = detectionsPerFrame{toTimeInc};
                shapeInfo = frameDetects(lastAss.DetectionInd);
                outlinePixels = shapeInfo.OutlinePixels;

                % convert (Row,Col) into (X,Y)
                outlinePixels = circshift(outlinePixels, [0 1]);

                % packs (X,Y) into cell array for cv.polyline
                outlinePixelsCell = mat2cell(outlinePixels, ones(length(outlinePixels),1),2);
                if strcmp('TopView', coordType)
                    outlinePixelsCellTop = obj.cameraToTopView(outlinePixelsCell);
                    outlinePixelsCell = outlinePixelsCellTop;
                end

                image = cv.polylines(image, outlinePixelsCell, 'Closed', true, 'Color', candColor);

                % draw box in the last frame
                bnd=shapeInfo.BoundingBox;
                               
                box = cell(1,4);
                box{1} = [bnd(1) bnd(2)];
                box{2} = [bnd(1)+bnd(3) bnd(2)];
                box{3} = [bnd(1)+bnd(3) bnd(2)+bnd(4)];
                box{4} = [bnd(1) bnd(2)+bnd(4)];
                
                % for 'camera' we do not change box
                % for 'TopView' we convert box to TopView
                
                if strcmp('TopView', coordType)
                    box = obj.cameraToTopView(box);
                end

                %image=cv.rectangle(image, box, 'Color', candColor);
                image=cv.polylines(image, box, 'Closed', true, 'Color', candColor);
            end
            
            %
            % put text for the last frame
            labelTrackCandidates = false;
            if labelTrackCandidates || ~track.IsTrackCandidate
                estPos = lastAss.EstimatedPos;
                estPosImage = obj.getViewCoord(estPos, coordType, lastAss);
                textPos = estPosImage;

                if lastAss.IsDetectionAssigned && ~isempty(box)
                    %textPos = [max(box(1),box(1) + box(3) - 26), box(2) - 13];
                    boxMat = reshape([box{:}],2,[])';
                    textPos = [max(boxMat(:,1)) min(boxMat(:,2))];
                end

                text1 = int2str(track.idOrCandidateId);
                image = cv.putText(image, text1, textPos, 'Color', candColor);            
            end
        end
    end
    
    imageAdorned = image;
end

% coordType='camera' convert position to camera coordinate
% coordType='TopView' convert position to TopView coordinate
function [trackPolyline,initialTrackPos] = buildTrackPath(obj, track, fromTime, toTimeInc, coordType)
    curPolyline = cell(0,0);
    trackPolyline = cell(0,0);
    initialTrackPos = [];
    for timeInd=fromTime:toTimeInc
        trackBreaks = false;
        if timeInd > length(track.Assignments)
            % TODO: implement track termination
            trackBreaks = true;
        else
            ass = track.Assignments{timeInd};
            if isempty(ass)
                trackBreaks = true;
            end
        end
        
        if trackBreaks
            % push next polyline
            if ~isempty(curPolyline)
                trackPolyline{end+1} =  curPolyline;
                curPolyline = cell(0,0);
            end
            continue;
        end

        %
        worldPos = ass.EstimatedPos;
        estPosImage = obj.getViewCoord(worldPos, coordType, ass);
            
        curPolyline{end+1} = estPosImage;

        if timeInd == 1
            initialTrackPos = estPosImage;
        end
    end

    % push last polyline
    if ~isempty(curPolyline)
        trackPolyline{end+1} =  curPolyline;
    end
end

function pos = getViewCoord(obj, worldPos, coordType, assignment)
    if strcmp('camera', coordType)
        pos = obj.distanceCompensator.worldToCamera(worldPos);

        if assignment.IsDetectionAssigned && norm(assignment.v.EstimatedPosImagePix - pos) > 10
            %warning('image and back projected coord diverge too much Expect=%d Actual=%d',assignment.v.EstimatedPosImagePix ,pos);
        end
    elseif strcmp('TopView', coordType)
        expectCameraSize = CameraDistanceCompensator.expectCameraSize;
        pos = CameraDistanceCompensator.scaleWorldToTopViewImageCoord(worldPos, expectCameraSize);
    else
        error('invalid argument coordType %s', coordType);
    end
end

function posTopView = cameraToTopView(obj, XByRowCell)
    posTopView = cell(1,length(XByRowCell));
    
    for i=1:length(XByRowCell)
        worldPos = CameraDistanceCompensator.cameraToWorld(obj.distanceCompensator, XByRowCell{i});

        expectCameraSize = CameraDistanceCompensator.expectCameraSize;
        pos = CameraDistanceCompensator.scaleWorldToTopViewImageCoord(worldPos, expectCameraSize);

        posTopView{i} = pos;
    end
end

function imageTopView = adornImageWithTrackedBodiesTopView(obj, image)
    desiredImageSize = [size(image,2), size(image,1)];
    imageTopView = CameraDistanceCompensator.convertCameraImageToTopView(obj.distanceCompensator, image, desiredImageSize);
    
    pathStartFrame = max([1, obj.frameInd - 100]);
    
    for track=[obj.tracks{:}]
        trackColor = obj.getTrackColor(track);
        
        % construct track path as polyline to draw by single command
        [trackPolyline,initialCandPos] = buildTrackPath(obj, track, pathStartFrame, obj.frameInd, 'TopView');
        
        % draw track path
        if ~isempty(trackPolyline)
            imageTopView = cv.polylines(imageTopView, trackPolyline, 'Closed', false, 'Color', trackColor);
        end
        
        % draw initial position
        if ~isempty(initialCandPos)
            imageTopView=cv.circle(imageTopView, initialCandPos, 3, 'Color', trackColor);
        end
    end
end

function rewindToFrameDebug(obj, toFrame, debug)
    obj.frameInd = toFrame;
end

function color = getTrackColor(obj, track)
    c_list = ['g' 'r' 'b' 'c' 'm' 'y'];
    c_list = utils.convert_color(c_list)*255;

    % pick track color
    % color should be the same for track candidate and later for the track
    color = c_list(1+mod(track.TrackCandidateId, length(c_list)),:);
end

function imageWithDetects = drawDetections(obj, image, detects)
    detectColor = [255 255 255];
    
    for i=length(detects)
        detect=detects(i);
        outlinePixels = detect.OutlinePixels;

        % convert (Row,Col) into (X,Y)
        outlinePixels = circshift(outlinePixels, [0 1]);

        % packs (X,Y) into cell array for cv.polyline
        outlinePixelsCell = mat2cell(outlinePixels, ones(length(outlinePixels),1), 2);

        image = cv.polylines(image, outlinePixelsCell, 'Closed', true, 'Color', detectColor);
        
        %

        box = detect.BoundingBox;
        image = cv.rectangle(image, box, 'Color', detectColor);
    end
    
    imageWithDetects = image;
end

% Internal method (for testing). Dumps position information for all tracks.
function result = getTrackById(this, trackId)
    result = [];
    for trackInd=1:length(this.tracks)
        curTrack = this.tracks{trackInd};
        if curTrack.idOrCandidateId == trackId
            result = curTrack;
            break;
        end
    end
end

% Internal method (for testing). Dumps position information for all tracks.
function trackPosInfo = trackInfo(this, trackId, queryFrameInd)
    if ~exist('queryFrameInd', 'var')
        queryFrameInd = this.frameInd; % take last frame
    end
    
    trackPosInfo = struct('TrackId', [], 'WorldPos', [], 'ImagePos', []);
    trackPosInfo(1) = [];
    
    if this.frameInd < 1
        return;
    end
    
    track = this.getTrackById(trackId);
    if isempty(track)
        return;
    end

    %

    trackInfo = struct;
    trackInfo.TrackId = track.idOrCandidateId;

    ass = track.Assignments{queryFrameInd};
    trackInfo.WorldPos = ass.EstimatedPos;
    trackInfo.ImagePos = [];
    if ass.IsDetectionAssigned
        blobs = this.detectionsPerFrame{queryFrameInd};
        trackInfo.ImagePos = blobs(ass.DetectionInd).Centroid;
    end

    trackPosInfo(end+1) = trackInfo;
end

function trackPosInfo = trackInfoOne(this)
    assert(length(this.tracks) == 1, 'Expected single track but were %d tracks', length(this.tracks));
    track = this.tracks{1};
    trackPosInfo = this.trackInfo(track.Id);
end

function result = tracksCount(this)
    result = length(this.tracks);
end

function blob = getBlobInFrame(this, frameId, blobId)
    blob = [];
    
    blobs = this.detectionsPerFrame{frameId};
    for blobInd=1:length(blobs)
        curBlob = blobs(blobInd);
        
        if curBlob.Id == blobId
            blob = cubBlob;
            break;
        end
    end
end

% Gets track which was associated with detection blobId in frame frameId.
function track = getTrackByBlobId(this, frameId, blobId)
    track = [];
    blobs = this.detectionsPerFrame{frameId};
    
    for trackInd=1:length(this.tracks)
        curTrack = this.tracks{trackInd};
        
        ass = curTrack.Assignments{frameId};
        if ass.IsDetectionAssigned
            blob = blobs(ass.DetectionInd);
            if blob.Id == blobId
                track = curTrack;
                return;
            end
        end
    end
end

end
end
