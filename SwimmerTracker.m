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
    v;
    %v.swimmerMaxSpeed;
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
    
    purgeMemory(this);
end

function purgeMemory(obj)
    obj.tracks=cell(0,0);
    obj.v.nextTrackId = 1;
    obj.frameInd = 0;
    obj.v.nextTrackCandidateId=1;
end

% Returns frame number for which track info (position, velocity vector etc) is available.
% returns -1 if there is not available frames.
function queryFrameInd = getFrameIndWithReadyTrackInfo(this)
    % This tracker has no history, return last processed frame
    queryFrameInd = this.frameInd;
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
    
    % track shapes

    processDetections(this, this.frameInd, elapsedTimeMs, fps, bodyDescrs, imageSwimmers, debug);
end

function processDetections(this, frameInd, elapsedTimeMs, fps, frameDetections, image, debug)
    % remember tracks count at the beginning of current frame processing
    % because new tracks may be added
    %tracksCountPrevFrame = length(obj.tracks);
    
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

end
end
