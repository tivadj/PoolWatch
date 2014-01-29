classdef SwimmerTracker < handle
properties
    frameInd;                 % type:int, number of processed frames
    detectionsPerFrame; % type: List<ShapeAssignment>
    tracks; % type: List<TrackedObject>
    tracksHistory; % type: List<TrackedObject>
    %v.nextTrackId;
    % v.nextTrackCandidateId;
    poolRegionDetector;
    distanceCompensator;
    humanDetector;
    colorAppearance;
    v;
    %v.swimmerMaxSpeed;
    trackStatusList; % struct<TrackChangePerFrame> used locally in trackBlobs method only, but 
    blobTracker;     % MultiHypothesisTracker
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
    this.blobTracker = MultiHypothesisTracker(distanceCompensator);
    
    % configure tracker
    this.v.swimmerMaxSpeed = 2.3; % max speed for swimmers 2.3m/s

    % new tracks are allocated for detections further from any existent tracks by this distance
    this.v.minDistToNewTrack = 0.5;
    
    % cache of track results
    this.trackStatusList = struct(TrackChangePerFrame);
    
    purgeMemory(this);
end

function purgeMemory(obj)
    obj.tracks=cell(0,0);
    obj.tracksHistory=cell(0,0);
    obj.v.nextTrackId = 1;
    obj.frameInd = 0;
    obj.v.nextTrackCandidateId=1;
    obj.v.queryFrameInd = -1;
    
    if ~isempty(obj.blobTracker)
        obj.blobTracker.purgeMemory();
    end
end

% Returns frame number for which track info (position, velocity vector etc) is available.
% returns -1 if there is not available frames.
function queryFrameInd = getFrameIndWithReadyTrackInfo(this)
    % This tracker has no history, return last processed frame
    %queryFrameInd = this.frameInd;
    queryFrameInd = this.v.queryFrameInd;
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
    
    % associate corresponding world coordinates of blob
    for i=1:length(bodyDescrs)
        centr = bodyDescrs(i).Centroid;
        centrWorld = this.distanceCompensator.cameraToWorld(centr);
        bodyDescrs(i).CentroidWorld = centrWorld;
    end
    
    this.detectionsPerFrame{this.frameInd} = bodyDescrs;
    
    if debug
        blobsCount = length(bodyDescrs);
        fprintf('blobsCount=%d\n', blobsCount);
        for i=1:blobsCount
            centr = bodyDescrs(i).Centroid;
            fprintf('Blob[%d] Centroid=[%.0f %.0f]\n', i, centr(1), centr(2));
        end
    end
    
    %
    if ~isempty(this.blobTracker)
        [frameIndWithTrackInfo,trackStatusList] = this.blobTracker.trackBlobs(this.frameInd, elapsedTimeMs, fps, bodyDescrs, imageSwimmers, debug);
    else
        [frameIndWithTrackInfo,trackStatusList] = this.trackBlobs(this.frameInd, elapsedTimeMs, fps, bodyDescrs, imageSwimmers, debug);
    end
    
    this.v.queryFrameInd = frameIndWithTrackInfo;
    
    if true % assert
        for i=1:length(trackStatusList)
            change = trackStatusList(i);
            assert(~isempty(change.EstimatedPosWorld));
            assert(~isempty(change.ObservationPosPixExactOrApprox));
        end
    end
    
    if debug
        fprintf('trackStatusList.length=%d\n', length(trackStatusList));
        for i=1:length(trackStatusList)
            change = trackStatusList(i);
            pix = change.ObservationPosPixExactOrApprox;
            fprintf('change %d: UpdType=%d ObsId=%d [%f %f]\n', i, change.UpdateType, change.ObservationInd, pix(1), pix(2));
        end
    end

	% update positions history for each track
    this.recordTrackStatus(frameIndWithTrackInfo,trackStatusList);

    this.promoteMatureTrackCandidates(this.frameInd);
end

function recordTrackStatus(this, frameIndWithTrackInfo, trackStatusList)
    for i=1:length(trackStatusList)
        trackStatus = trackStatusList(i);

        % find corresponding track by ID
        trackHistInd = -1;
        if trackStatus.UpdateType == TrackChangePerFrame.New
            trackHistInd = -1;
        elseif trackStatus.UpdateType == TrackChangePerFrame.ObservationUpdate ||...
               trackStatus.UpdateType == TrackChangePerFrame.NoObservation
               %trackStatus.UpdateType == TrackChangePerFrame.Finished
            trackByIdPred = @(t) t.TrackCandidateId == trackStatus.TrackCandidateId;
            trackHistInd = this.getTrackIndFromHistory(trackByIdPred);
        end
        
        if trackHistInd == -1 && trackStatus.UpdateType == TrackChangePerFrame.NoObservation
            % track history was removed as false detection
            continue;
        end
        % remove track record
%         if trackStatus.UpdateType == TrackChangePerFrame.Finished
%             this.tracksHistory(trackHistInd) = [];
%         end

        %
        ass = ShapeAssignment;
        ass.EstimatedPosWorld = trackStatus.EstimatedPosWorld;
        ass.ObservationPosPixExactOrApprox = trackStatus.ObservationPosPixExactOrApprox;

        % copy blob coordinates
        if trackStatus.UpdateType == TrackChangePerFrame.ObservationUpdate || trackStatus.UpdateType == TrackChangePerFrame.New
            ass.IsDetectionAssigned = true;
            ass.DetectionInd = trackStatus.ObservationInd;
        elseif trackStatus.UpdateType == TrackChangePerFrame.NoObservation
            ass.IsDetectionAssigned = false;
            ass.DetectionInd = 0;
        end
        
        % allocate new track
        if trackHistInd ==-1
            % For one-step tracker, the UpdateType is always 'New'
            % For MHT, family root may be lost before it is collected by pruning mechanism
            % hence trackStatus.UpdateType potentially may be anything;
            % in this case, if there is no track for given change, we will create new track
            
            trackRecord = TrackInfoHistory();
            trackRecord.TrackCandidateId = trackStatus.TrackCandidateId;
            trackRecord.FirstAppearanceFrameIdx = frameIndWithTrackInfo;

            this.tracksHistory{end+1} = trackRecord;
            trackHistInd = length(this.tracksHistory);
        end
        
        assert(trackHistInd <= length(this.tracksHistory));
        this.tracksHistory{trackHistInd}.Assignments{frameIndWithTrackInfo} = ass;
    end
end

function [frameIndWithTrackInfo,trackStatusListResult] = trackBlobs(this, frameInd, elapsedTimeMs, fps, frameDetections, image, debug)
    % remember tracks count at the beginning of current frame processing
    % because new tracks may be added
    %tracksCountPrevFrame = length(obj.tracks);
    
    %this.trackStatusList(:) = [];
    % cache of tracking results for each frame
    %this.trackStatusList = struct(TrackChangePerFrame);
    %this.trackStatusList(1)=[];
    this.trackStatusList(:) = [];

    %
    predictSwimmerPositions(this, frameInd, fps);

    swimmerMaxShiftPerFrameM = elapsedTimeMs * this.v.swimmerMaxSpeed / 1000 + this.humanDetector.shapeCentroidNoise;
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

        % estimate position

        track = this.tracks{trackInd};
        detect=frameDetections(detectInd);

        worldPos = detect.CentroidWorld;
        posEstimate2 =track.KalmanFilter.correct(worldPos(1:2));
        posEstimate = [posEstimate2 worldPos(3)];
        
        track.LastEstimatedPosWorld = posEstimate;
        track.LastObservationPosExactOrApprox = detect.Centroid;

        %
        this.onAssignDetectionToTrackedObject(track, detect, image);
        
        %
        status = TrackChangePerFrame;
        status.TrackCandidateId = track.TrackCandidateId;
        status.UpdateType = TrackChangePerFrame.ObservationUpdate;
        status.EstimatedPosWorld = posEstimate;
        status.ObservationInd = detectInd;
        status.ObservationPosPixExactOrApprox = detect.Centroid;
        this.trackStatusList(end+1) = struct(status);
   end

    assignTrackCandidateToUnassignedDetections(this, unassignedDetections, frameDetections, frameInd, image, fps);

    driftUnassignedTracks(this, unassignedTracks, frameInd);

    % assert EstimatedPos is initialized
    for r=1:length(this.trackStatusList)
        status = this.trackStatusList(r);

        if isempty(status.EstimatedPosWorld)
            assert(false);
        end
        if isempty(status.ObservationPosPixExactOrApprox)
            assert(false);
        end
    end
    
    frameIndWithTrackInfo = frameInd;
    trackStatusListResult = this.trackStatusList; % take data from cache
end

function predictSwimmerPositions(obj, frameInd, fps)
    % predict pos for each tracked object
    for r=1:length(obj.tracks)
        track = obj.tracks{r};

        worldPos2 = track.KalmanFilter.predict();
        worldPos = [worldPos2 0]; % append zeroHeight = 0

        %%%%%
        track.PredictedPosWorld = worldPos;
        
        %
%         ass = ShapeAssignment();
%         ass.IsDetectionAssigned = false;
%         ass.DetectionInd = -1;
%         ass.PredictedPos = worldPos; % take pair (x,y)
%         obj.tracks{r}.Assignments{frameInd} = ass;
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
        
        if debug
            % recover track pos
            trackImgPos = track.LastObservationPosExactOrApprox;
            fprintf('Track Id=%d ImgPos=[%.0f %.0f]\n', track.idOrCandidateId, trackImgPos(1), trackImgPos(2));
        end
        
        % calculate cost of assigning detection to track
        
        for blobInd=1:detectCount
            detect = frameDetections(blobInd);

            maxCost = 99999;
            
            centrWorld = detect.CentroidWorld;
            dist=norm(centrWorld-track.PredictedPosWorld);
            
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

function discardTrackCore(this, trackCandidateId)
    trackIndToRemove = -1;
    for trackInd=1:length(this.tracks)
        track = this.tracks{trackInd};
        if track.TrackCandidateId == trackCandidateId
            trackIndToRemove = trackInd;
            break;
        end
    end

    if trackIndToRemove ~= -1
        this.tracks(trackIndToRemove) = [];
    end
end

function promoteMatureTrackCandidates(obj, frameInd)
    % One period of breaststroke=90 frames for 30fps video
    % at least half of frames should have detected the human
    trackCandidateMaturingTime = 10; % number of frames to scrutinize if candidate is a human or noise
    trackCandidatePromotionRatio = 8/15; % candidate is promoted to track if is frequently detected
    
    trackCandidateInfancyTime = 5; % number of frames to scrutinize the candidate if it just noise and to be removed early
    trackCandidateInfancyRatio = 1/5+0.01; % candidate is discarded if detected infrequently during infancy time

    for candInd=1:length(obj.tracksHistory)
        cand=obj.tracksHistory{candInd};
        if ~cand.IsTrackCandidate
            continue;
        end

        lifeTime = frameInd - cand.FirstAppearanceFrameIdx + 1;
        detectCount = cand.getDetectionsCount(frameInd);
        detectRatio = detectCount / lifeTime;

        if lifeTime <= trackCandidateInfancyTime
            if detectRatio < trackCandidateInfancyRatio
                % candidate is a noise, discard it
                obj.discardTrackCore(cand.TrackCandidateId);
                obj.tracksHistory{candInd}=[]; % mark for deletion
            end
        elseif lifeTime >= trackCandidateMaturingTime
            if detectRatio < trackCandidatePromotionRatio
                % candidate is a noise, discard it
                obj.discardTrackCore(cand.TrackCandidateId);
                obj.tracksHistory{candInd}=[]; % mark for deletion
            else
                % candidate is a stable detection, promote to track

                newTrackId = obj.v.nextTrackId;
                obj.v.nextTrackId = obj.v.nextTrackId + 1;

                % promote TrackCandidiate into Track
                obj.tracksHistory{candInd}.Id = newTrackId;
                obj.tracksHistory{candInd}.IsTrackCandidate = false;
                obj.tracksHistory{candInd}.PromotionFramdInd = frameInd;
            end 
        end        
    end

    % clean empty track candidate slots
    for i=length(obj.tracksHistory):-1:1 % goes backward
        if isempty(obj.tracksHistory{i})
            obj.tracksHistory(i) = [];
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

            for blobInd=blobInds(farAwayBlobsMask)
                blob = frameDetections(blobInd);
                dist = norm(track.LastEstimatedPosWorld - blob.CentroidWorld);
                
                % new tracks are allocated for detections further from any existent tracks by this distance
                if dist < obj.v.minDistToNewTrack
                    farAwayBlobsMask(blobInd) = false;
                end
            end
        end
    end
    
    farAwayBlobs = blobInds(farAwayBlobsMask);
    
    for blobInd=farAwayBlobs
        detect = frameDetections(blobInd);

        % new track candidate
        
        cand = TrackedObject.NewTrackCandidate(obj.v.nextTrackCandidateId);
        obj.v.nextTrackCandidateId = obj.v.nextTrackCandidateId + 1;

        % project image position into TopView
        worldPos = detect.CentroidWorld;
        cand.KalmanFilter = createKalmanPredictor(obj, worldPos(1:2), fps);
        cand.LastEstimatedPosWorld = worldPos; % world pos is the initial estimate
        cand.LastObservationPosExactOrApprox = detect.Centroid;
        
        obj.onAssignDetectionToTrackedObject(cand, detect, image);
        obj.tracks{end+1} = cand;
        
        %
        
        status = TrackChangePerFrame;
        status.TrackCandidateId = cand.TrackCandidateId;
        status.UpdateType = TrackChangePerFrame.New;
        status.EstimatedPosWorld = worldPos;
        status.ObservationInd = blobInd;
        status.ObservationPosPixExactOrApprox = detect.Centroid;
        obj.trackStatusList(end+1) = struct(status);
    end
end

function driftUnassignedTracks(this, unassignedTracksByRow, frameInd)
    % for unassigned tracks initialize EstimatedPos = PredictedPos by default
    for trackInd=unassignedTracksByRow'
        track = this.tracks{trackInd};
        assert(~isempty(track.PredictedPosWorld), 'PredictedPos must have been initialized (trackInd=%d frameInd=%d)', trackInd, frameInd);

        % predicted pos is the best estimate
        estimatedPos = track.PredictedPosWorld;

        % TODO: finish track if detections lost for some time
        
        %
        x = estimatedPos(1);
        poolSize = this.distanceCompensator.poolSize;
        if x < 1 || x > poolSize(2) - 1
            isNearBorder = true;
        else
            isNearBorder = false;
        end

        % we don't use Kalman filter near pool boundary as swimmers
        % usually reverse their direction here
        if isNearBorder
            kalmanObj = track.KalmanFilter;
            
            % after zeroing velocity, Kalman predictions would aim single position
            kalmanObj.State(3:end)=0; % zero vx and vy
        end
        
        %
        track.LastEstimatedPosWorld = estimatedPos;
        approxBlobCentroid = this.distanceCompensator.worldToCamera(estimatedPos);
        track.LastObservationPosExactOrApprox = approxBlobCentroid;
        
        %
        status = TrackChangePerFrame;
        status.TrackCandidateId = track.TrackCandidateId;
        status.UpdateType = TrackChangePerFrame.NoObservation;
        status.EstimatedPosWorld = estimatedPos;
        status.ObservationInd = 0;
        status.ObservationPosPixExactOrApprox = approxBlobCentroid;
        this.trackStatusList(end+1) = struct(status);
    end
end

function onAssignDetectionToTrackedObject(this, track, detect, image)
    this.colorAppearance.onAssignDetectionToTrackedObject(track, detect, image);
end

function rewindToFrameDebug(obj, toFrame, debug)
    obj.frameInd = toFrame;
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

function trackInd = getTrackIndFromHistory(this, pred)
    trackInd = -1;
    
    for i=1:length(this.tracksHistory)
        curTrack = this.tracksHistory{i};
        
        if pred(curTrack)
            trackInd = i;
            break;
        end
    end
end

end
end
