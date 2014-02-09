classdef SwimmingPoolObserver < handle
properties
    frameInd;                 % type:int, number of processed frames
    detectionsPerFrame; % type: List<ShapeAssignment>
    tracksHistory; % type: List<TrackInfoHistory>
    %v.nextTrackId;
    poolRegionDetector;
    distanceCompensator;
    humanDetector;
    blobTracker;     % MultiHypothesisBlobTracker
    trackPainterHandle; % id of the native track painter object
    v;
end

methods
    
function this = SwimmingPoolObserver(poolRegionDetector, distanceCompensator, humanDetector, blobTracker)
    assert(~isempty(poolRegionDetector));
    assert(~isempty(distanceCompensator));
    assert(~isempty(humanDetector));
    assert(~isempty(blobTracker));
    
    this.poolRegionDetector = poolRegionDetector;
    this.distanceCompensator = distanceCompensator;
    this.humanDetector = humanDetector;
    this.blobTracker = blobTracker;
    
    purgeMemory(this);
end

function purgeMemory(obj)
    obj.tracksHistory=cell(0,0);
    obj.v.nextTrackId = int32(1);
    obj.frameInd = int32(0);
    obj.v.queryFrameInd = int32(-1);
    obj.blobTracker.purgeMemory();
    
    pruneWindow = obj.blobTracker.pruneDepth + 1;
    fps = single(30); % TODO: init from client
    obj.trackPainterHandle = PWSwimmingPoolObserver(int32(0), 'new', pruneWindow, fps);
end

% Returns frame number for which track info (position, velocity vector etc) is available.
% returns -1 if there is not available frames.
function queryFrameInd = getFrameIndWithReadyTrackInfo(this)
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
        bodyDescrs(i).CentroidWorld = single(centrWorld);
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
    
    % update native painter with new blobs
    if ~this.blobTracker.v.nativeRun
        PWSwimmingPoolObserver(this.trackPainterHandle, 'setBlobs', int32(this.frameInd), bodyDescrs);
    else
        PWSwimmingPoolObserver(this.trackPainterHandle, 'processBlobs', int32(this.frameInd), imageSwimmers, bodyDescrs);
        
        pruneWindow=6;
        readyFrameInd = this.frameInd - pruneWindow;
        if readyFrameInd < 1
            readyFrameInd = -1;
        end
        this.v.queryFrameInd=readyFrameInd;

        return;
    end
    
    
    %
    [frameIndWithTrackInfo,trackStatusList] = this.blobTracker.trackBlobs(this.frameInd, elapsedTimeMs, fps, bodyDescrs, imageSwimmers, debug);
    
    % remember index of the last ready frame 
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
    
    % update native painter with track changes
    if frameIndWithTrackInfo ~= -1
        if ~this.blobTracker.v.nativeRun
            PWSwimmingPoolObserver(this.trackPainterHandle, 'setTrackChangesPerFrame', int32(frameIndWithTrackInfo), trackStatusList);
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
                obj.blobTracker.discardTrackCore(cand.TrackCandidateId);
                obj.tracksHistory{candInd}=[]; % mark for deletion
            end
        elseif lifeTime >= trackCandidateMaturingTime
            if detectRatio < trackCandidatePromotionRatio
                % candidate is a noise, discard it
                obj.blobTracker.discardTrackCore(cand.TrackCandidateId);
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

function imageWithTracks = adornImageWithTrackedBodies(this, queryImage, coordType, queryFrameInd)
    trailLength=int32(250);

    if ~this.blobTracker.v.nativeRun
        % Matlab drawing
        imageWithTracks = TrackPainter.adornImageWithTrackedBodies(queryImage, coordType, queryFrameInd, trailLength, this.detectionsPerFrame, this.tracksHistory, this.distanceCompensator);
    else
        % native drawing
        imageWithTracks=PWSwimmingPoolObserver(this.trackPainterHandle, 'adornImage', queryImage, int32(queryFrameInd), trailLength);
    end
end

% TODO: what is it?
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
