classdef OneStepTracker < handle
% This tracker use a distance matrix of track-observations and 
% assignment algorithm for association.
    
properties
    tracks; % type: List<OneStepTrackedObject>
    % v.nextTrackCandidateId;
    distanceCompensator;
    colorAppearance;
    v;
    %v.swimmerMaxSpeed;
    trackStatusList; % struct<TrackChangePerFrame> used locally in trackBlobs method only, but 
end

methods

function this = OneStepTracker(distanceCompensator, colorAppearance)
    assert(~isempty(distanceCompensator));
    assert(~isempty(colorAppearance));    

    this.distanceCompensator = distanceCompensator;
    this.colorAppearance = colorAppearance;

    % configure tracker
    this.v.swimmerMaxSpeed = 2.3; % max speed for swimmers 2.3m/s

    % new tracks are allocated for detections further from any existent tracks by this distance
    this.v.minDistToNewTrack = 0.5;
    
    % cache of track results
    this.trackStatusList = struct(TrackChangePerFrame);
end

function purgeMemory(obj)
    obj.v.nextTrackCandidateId=int32(1);
    obj.tracks=cell(0,0);
end

function [frameIndWithTrackInfo,trackStatusListResult] = trackBlobs(this, frameInd, elapsedTimeMs, fps, frameDetections, image, debug)
    % remember tracks count at the beginning of current frame processing
    % because new tracks may be added
    %tracksCountPrevFrame = length(obj.tracks);
    
    % purge cache
    this.trackStatusList(:) = [];

    %
    predictSwimmerPositions(this, frameInd, fps);

    %swimmerMaxShiftPerFrameM = elapsedTimeMs * this.v.swimmerMaxSpeed / 1000 + this.humanDetector.shapeCentroidNoise;
    shapeCentroidNoise = 0.5;
    swimmerMaxShiftPerFrameM = elapsedTimeMs * this.v.swimmerMaxSpeed / 1000 + shapeCentroidNoise;
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
        status.EstimatedPosWorld = single(posEstimate);
        status.ObservationInd = int32(detectInd);
        status.ObservationPosPixExactOrApprox = single(detect.Centroid);
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
        
        cand = OneStepTrackedObject.NewTrackCandidate(obj.v.nextTrackCandidateId);
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
        status.EstimatedPosWorld = single(worldPos);
        status.ObservationInd = int32(blobInd);
        status.ObservationPosPixExactOrApprox = single(detect.Centroid);
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
        status.EstimatedPosWorld = single(estimatedPos);
        status.ObservationInd = int32(0);
        status.ObservationPosPixExactOrApprox = single(approxBlobCentroid);
        this.trackStatusList(end+1) = struct(status);
    end
end

function onAssignDetectionToTrackedObject(this, track, detect, image)
    this.colorAppearance.onAssignDetectionToTrackedObject(track, detect, image);
end

end % methods
end

