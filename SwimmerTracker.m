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
    v;
end

methods
    
function this = SwimmerTracker(poolRegionDetector, distanceCompensator, humanDetector)
    assert(~isempty(poolRegionDetector));
    assert(~isempty(distanceCompensator));
    assert(~isempty(humanDetector));    
    
    this.poolRegionDetector = poolRegionDetector;
    this.distanceCompensator = distanceCompensator;
    this.humanDetector = humanDetector;
    
    purgeMemory(this);
end


function purgeMemory(obj)
    obj.tracks=cell(0,0);
    obj.v.nextTrackId = 1;
    obj.frameInd = 0;
    obj.v.nextTrackCandidateId=1;
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
    
    bodyDescrs = this.humanDetector.GetHumanBodies(imageSwimmers, waterMask, debug);
    this.detectionsPerFrame{this.frameInd} = bodyDescrs;
    
    if debug
        for i=1:length(bodyDescrs)
            centr = bodyDescrs(i).Centroid;
            fprintf('Blob[%d] Centroid=[%.0f %.0f]\n', i, centr(1), centr(2));
        end
    end
    
    % track shapes

    processDetections(this, this.frameInd, elapsedTimeMs, fps, bodyDescrs, imageSwimmers, debug);
end

function processDetections(obj, frameInd, elapsedTimeMs, fps, frameDetections, image, debug)
    % remember tracks count at the beginning of current frame processing
    % because new tracks may be added
    %tracksCountPrevFrame = length(obj.tracks);

    %
    predictSwimmerPositions(obj, frameInd, fps);

    trackDetectCost = calcTrackToDetectionAssignmentCostMatrix(obj, image, frameInd, elapsedTimeMs, frameDetections, debug);

    % make assignment using Hungarian algo
    % all unassigned detectoins and unprocessed tracks

    minAppearPerPixSimilarity = 0.00005; % shapes with lesser proximity should not be assigned
    
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

        ass = obj.tracks{trackInd}.Assignments{frameInd};
        ass.IsDetectionAssigned = true;
        ass.DetectionInd = detectInd;

        % estimate position
        imagePos = detect.Centroid;
        ass.v.EstimatedPosImagePix = imagePos;

        % project image position into TopView
        worldPos = CameraDistanceCompensator.cameraToWorld(obj.distanceCompensator, imagePos);
        
        worldPos2 = worldPos(1:2);
        posEstimate2 = obj.tracks{trackInd}.KalmanFilter.correct(worldPos2);
        posEstimate = [posEstimate2 worldPos(3)];
        
        ass.EstimatedPos = posEstimate;

        obj.tracks{trackInd}.Assignments{frameInd} = ass;
        
        %
        obj.onAssignDetectionToTrackedObject(obj.tracks{trackInd}, detect, image);
   end

    assignTrackCandidateToUnassignedDetections(obj, unassignedDetections, frameDetections, frameInd, image, fps);

    driftUnassignedTracks(obj, unassignedTracks, frameInd);

    promoteMatureTrackCandidates(obj, frameInd);

    % assert EstimatedPos is initialized
    for r=1:length(obj.tracks)
        if frameInd > length(obj.tracks{r}.Assignments)
            assert(false, 'IndexOutOfRange: frameInd=%d greater than length(assignments)=%d\n', frameInd, length(obj.tracks{r}.Assignments));
        end
        
        a = obj.tracks{r}.Assignments{frameInd};
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

function trackDetectCost = calcTrackToDetectionAssignmentCostMatrix(obj, image, frameInd, elapsedTimeMs, frameDetections, debug)
    % 2.3m/s is max speed for swimmers
    shapeCentroidNoise = 0.5; % shape may change significantly
    swimmerMaxShiftPerFrameM = elapsedTimeMs * 2.3 / 1000 + shapeCentroidNoise;
    
    % calculate distance from predicted pos of each tracked object
    % to each detection

    trackObjCount = length(obj.tracks);
    detectCount = length(frameDetections);
    
    trackDetectCost=zeros(trackObjCount, detectCount);
    for trackInd=1:trackObjCount
        track = obj.tracks{trackInd};
        trackedObjPos = track.Assignments{frameInd}.PredictedPos;
        
        if debug
            % recover track pos
            trackImgPos = obj.getPrevFrameDetectionOrPredictedImagePos(track, frameInd);
            fprintf('Track Id=%d ImgPos=[%.0f %.0f]\n', track.idOrCandidateId, trackImgPos(1), trackImgPos(2));
        end
        
        % calculate cost of assigning detection to track
        
        for blobInd=1:detectCount
            detect = frameDetections(blobInd);

            maxCost = 99999;
            
            centrPix = detect.Centroid;
            centrWorld = CameraDistanceCompensator.cameraToWorld(obj.distanceCompensator, centrPix);
            
            dist=norm(centrWorld-trackedObjPos);
            
            % keep swimmers inside the area of max possible shift per frame
            if dist > swimmerMaxShiftPerFrameM
                dist = maxCost;
            else
                pixs = obj.getDetectionPixels(detect, image);
                
                probs = track.predict(pixs);
                avgProb = mean(probs);
                dist = min([1/avgProb maxCost]);
                if debug
                    fprintf('calcTrackCost: Blob[%d] [%.0f %.0f] dist=%d avgProb=%d\n', blobInd, centrPix(1), centrPix(2), dist, avgProb);
                end
            end
            
            trackDetectCost(trackInd,blobInd) = dist;
        end
    end
end

function forePixs = getDetectionPixels(obj, detect, image)
    bnd = ceil(detect.BoundingBox); % ceil to get positive index for pos=0.5
    shapeImage = image(bnd(2):bnd(2)+bnd(4)-1,bnd(1):bnd(1)+bnd(3)-1,:);
    forePixs = reshape(shapeImage, [], 3);

    shapeMask = detect.FilledImage;
    shapeMask = reshape(shapeMask, [], 1);
    
    assert(length(shapeMask) == size(forePixs,1));
    forePixs = forePixs(shapeMask,:);
end

function trackImgPos = getPrevFrameDetectionOrPredictedImagePos(this, track, curFrameInd)
    assert(curFrameInd >= 2, 'There is no previous frame for the first frame');
    
    prevFrameAssign = track.Assignments{curFrameInd-1};

    if prevFrameAssign.IsDetectionAssigned
        prevFrameBlobs = this.detectionsPerFrame{curFrameInd-1};
        prevBlob = prevFrameBlobs(prevFrameAssign.DetectionInd);
        trackImgPos = prevBlob.Centroid;
    else
        trackImgPos = CameraDistanceCompensator.worldToCamera(this.distanceCompensator, prevFrameAssign.EstimatedPos);
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
                minDistToNewTrack = 0.5;
                if dist < minDistToNewTrack
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
        worldPos = CameraDistanceCompensator.cameraToWorld(obj.distanceCompensator, imagePos);

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

function driftUnassignedTracks(obj, unassignedTracksByRow, frameInd)
    % for unassigned tracks initialize EstimatedPos = PredictedPos by default
    for trackInd=unassignedTracksByRow'
        a = obj.tracks{trackInd}.Assignments{frameInd};
        assert(~isempty(a.PredictedPos), 'Kalman Predictor must have initialized the PredictedPos (track=%d assignment=%d)', trackInd, frameInd);

        estPos = a.PredictedPos;
        a.EstimatedPos = estPos;
        obj.tracks{trackInd}.Assignments{frameInd} = a;

        % TODO: finish track if detections lost for some time
        
        %
        x = estPos(1);
        poolSize = CameraDistanceCompensator.poolSize;
        if x < 1 || x > poolSize(2) - 1
            isNearBorder = true;
        else
            isNearBorder = false;
        end

        % we don't use Kalman filter near pool boundary as swimmers
        % usually reverse their direction here
        if isNearBorder
            kalmanObj = obj.tracks{trackInd}.KalmanFilter;
            
            % after zeroing velocity, Kalman predictions would aim single position
            kalmanObj.State(3:end)=0; % zero vx and vy
        end
    end
end

function onAssignDetectionToTrackedObject(obj, track, detect, image)
    % retraining GMM on each frame is too costly
    % => retrain it during the initial time until enough pixels is accumulated
    % then we assume swimmer's shape doesn't change
    if track.canAcceptAppearancePixels
        pixs = obj.getDetectionPixels(detect, image);
        track.pushAppearancePixels(pixs);
    end
end

function imageWithTracks = adornImageWithTrackedBodies(obj, image, coordType)
    pathStartFrame = max([1, obj.frameInd - 250]);
    
    detects = obj.detectionsPerFrame{obj.frameInd};
    %imageWithTracks = drawDetections(obj, image, detects);

    if strcmp('TopView', coordType)
        desiredImageSize = [size(image,2), size(image,1)];
        image = CameraDistanceCompensator.convertCameraImageToTopView(obj.distanceCompensator, image, desiredImageSize);
    end

    imageWithTracks = adornTracks(obj, image, pathStartFrame, obj.frameInd, obj.detectionsPerFrame, obj.tracks, coordType);
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
        lastAss = track.Assignments{toTimeInc};
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
        ass = track.Assignments{timeInd};
        if isempty(ass)
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
        pos = CameraDistanceCompensator.worldToCamera(obj.distanceCompensator, worldPos);

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

end
end
