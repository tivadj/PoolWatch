classdef SwimmerTracker < handle
properties
    detectionsPerFrame;
    tracks;
    %v.nextTrackId;
    %trackCandidates;
    frameInd;                 % number of processed frames
    v;
    % v.nextTrackCandidateId;
    % v.distanceCompensator;
end

methods
    
function obj = SwimmerTracker()
    purgeMemory(obj);
end

function ensureInitialized(obj, debug)
    if ~isfield(obj.v, 'distanceCompensator')
        obj.v.distanceCompensator = CameraDistanceCompensator.create;
    end
    
    % initialize classifier
    if ~isfield(obj.v, 'cl2')
         obj.v.cl2=SkinClassifierStatics.create;
         SkinClassifierStatics.populateSurfPixels(obj.v.cl2);
        SkinClassifierStatics.prepareTrainingDataMakeNonoverlappingHulls(obj.v.cl2, debug);
        SkinClassifierStatics.findSkinPixelsConvexHullWithMinError(obj.v.cl2, debug);
    end
    
    % human detector
    if ~isfield(obj.v, 'det')
        clear det;
        %svmClassifierFun=@(XByRow) utils.SvmClassifyHelper(obj.v.skinClassif, XByRow, 1000);
        skinHullClassifierFun=@(XByRow) utils.inhull(XByRow, obj.v.cl2.v.skinHullClassifHullPoints, obj.v.cl2.v.skinHullClassifHullTriInds, 0.2);
        skinClassifier=skinHullClassifierFun;
        
        % init water classifer
        humanDetectorRunner = RunHumanDetector.create;
        %waterClassifierFun = RunHumanDetector.getWaterClassifierAsConvHull(humanDetectorRunner, debug);
        waterClassifierFun = RunHumanDetector.getWaterClassifierAsMixtureOfGaussians(humanDetectorRunner,6,debug);
        obj.v.waterClassifierFun = waterClassifierFun;
        
        obj.v.det = HumanDetector(skinClassifier, waterClassifierFun, obj.v.distanceCompensator);
    end
end

function purgeMemory(obj)
    obj.tracks=cell(0,0);
    obj.v.nextTrackId = 1;
    obj.frameInd = 0;
    obj.v.nextTrackCandidateId=1;
end

% elapsedTimeMs - time in milliseconds since last frame
function nextFrame(obj, image, elapsedTimeMs, fps, debug)
    ensureInitialized(obj, debug);

    obj.frameInd = obj.frameInd + 1;

    if debug
    end

    fprintf(1, 'find shapes\n');
    bodyDescrs = obj.v.det.GetHumanBodies(image, debug);
    obj.detectionsPerFrame{obj.frameInd} = bodyDescrs;
    
    fprintf(1, 'track shapes\n');

    processDetections(obj, obj.frameInd, elapsedTimeMs, fps, bodyDescrs);
end

function processDetections(obj, frameInd, elapsedTimeMs, fps, frameDetections)
    % remember tracks count at the beginning of current frame processing
    % because new tracks may be added
    %tracksCountPrevFrame = length(obj.tracks);

    %
    predictSwimmerPositions(obj, frameInd, fps);

    trackDetectCost = calcTrackToDetectionAssignmentCostMatrix(obj, frameInd, elapsedTimeMs, frameDetections);

    % make assignment using Hungarian algo
    % all unassigned detectoins and unprocessed tracks

    % TODO: how to estimate these parameters?
    unassignedTrackCost = 50;
    unassignedDetectionCost = 50; % detections are noisy => cost is small
    [assignment, unassignedTracks, unassignedDetections] = assignDetectionsToTracks(trackDetectCost, unassignedTrackCost, unassignedDetectionCost);

    for i=1:size(assignment,1)
        trackInd = assignment(i,1);
        detectInd = assignment(i,2);

        shape=frameDetections(detectInd);

%        if trackInd <= tracksCountPrevFrame % track
            % make an assignment
            ass = obj.tracks{trackInd}.Assignments{frameInd};
            ass.IsDetectionAssigned = true;
            ass.DetectionInd = detectInd;

            %obj.tracks{trackInd}.Assignments{frameInd}.DetectionCentroid = shape.Centroid;
%             if isfield(shape,'BoundingBox')
%                 obj.tracks{trackInd}.Assignments{frameInd}.DetectionBoundingBox = shape.BoundingBox;
%             end

            % estimate position
            imagePos = shape.Centroid;
            ass.v.EstimatedPosImagePix = imagePos;
            
            % project image position into TopView
            worldPos = CameraDistanceCompensator.cameraToWorld(obj.v.distanceCompensator, imagePos);
            worldPos2 = worldPos(1:2);
            
            posEstimate = obj.tracks{trackInd}.KalmanFilter.correct(worldPos2);
            posEstimate = [posEstimate 0]; % append zeroHeight = 0
            ass.EstimatedPos = posEstimate;
            
            obj.tracks{trackInd}.Assignments{frameInd} = ass;
%         else % track candidate
%             candInd = trackInd - tracksCountPrevFrame;
%             cand=obj.trackCandidates{candInd};
% 
%             obj.trackCandidates{candInd}.DetectionIdList{end+1} = detectInd;
% 
%             % estimate position
%             posEstimate = cand.KalmanFilter.correct(shape.Centroid);
%             obj.trackCandidates{candInd}.EstimatedPosList{end+1} = posEstimate;
%        end
   end

    assignTrackCandidateToUnassignedDetections(obj, unassignedDetections, frameDetections, frameInd, fps);

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

        worldPos = track.KalmanFilter.predict();
        worldPos = [worldPos 0]; % append zeroHeight = 0

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

function trackDetectCost = calcTrackToDetectionAssignmentCostMatrix(obj, frameInd, elapsedTimeMs, frameDetections)
    % 2.3m/s is max speed for swimmers
    shapeCentroidNoise = 0.5; % shape may change significantly
    swimmerMaxShiftPerFrameM = elapsedTimeMs * 2.3 / 1000 + shapeCentroidNoise;
    
    % calculate distance from predicted pos of each tracked object
    % to each detection

    trackObjCount = length(obj.tracks);
    detectCount = length(frameDetections);
    
    trackDetectCost=zeros(trackObjCount, detectCount);
    for r=1:trackObjCount
        trackedObjPos = obj.tracks{r}.Assignments{frameInd}.PredictedPos;
%         if r <= tracksCountBefore % track
%             trackedObjPos = obj.tracks{r}.Assignments{frameInd}.PredictedPos;
%         else % track candidate
%             candInd = r - tracksCountBefore;
%             cand = obj.trackCandidates{candInd};            
%             
%             posInd = frameInd - cand.FirstAppearanceFrameIdx + 1;            
%             trackedObjPos = cand.PredictedPosList{posInd};
%         end
        
        for d=1:detectCount
            centrPix = frameDetections(d).Centroid;
            centrWorld = CameraDistanceCompensator.cameraToWorld(obj.v.distanceCompensator, centrPix);
            
            dist=norm(centrWorld-trackedObjPos);
            
            if dist > swimmerMaxShiftPerFrameM
                dist = 9999;
            end
            trackDetectCost(r,d) = dist;
        end
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
        if ~cand.v.IsTrackCandidate
            continue;
        end

        lifeTime = frameInd - cand.v.FirstAppearanceFrameIdx + 1;
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
                obj.tracks{candInd}.v.IsTrackCandidate = false;
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

function kalman = createCalmanPredictor(obj, initPos, fps)
    % init Kalman Filter
    stateModel = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1]; 
    measurementModel = [1 0 0 0; 0 1 0 0]; 
    kalman = vision.KalmanFilter(stateModel, measurementModel);
    
    % max shift per frame = 20cm
    %kalman.ProcessNoise = (0.2 / 3)^2;
    obj.setTrackKalmanProcessNoise(kalman, 'normal', fps);
    
    % max measurment error per frame = 1m (far away it can be 5m)
    kalman.MeasurementNoise = (5 / 3)^2;
    kalman.State = [initPos 0 0]; % v0=0
end

% associate unassigned detection with track candidate
function assignTrackCandidateToUnassignedDetections(obj, unassignedDetectionsByRow, frameDetections, frameInd, fps)
    for d=unassignedDetectionsByRow'
        % new track candidate
        
        cand = TrackedObject.NewTrackCandidate(obj.v.nextTrackCandidateId);
        obj.v.nextTrackCandidateId = obj.v.nextTrackCandidateId + 1;

        cand.v.FirstAppearanceFrameIdx = frameInd;
        
        %
        shape = frameDetections(d);
        
        % project image position into TopView
        imagePos = shape.Centroid;
        worldPos = CameraDistanceCompensator.cameraToWorld(obj.v.distanceCompensator, imagePos);

        cand.KalmanFilter = createCalmanPredictor(obj, worldPos(1:2), fps);

        ass = ShapeAssignment();
        ass.IsDetectionAssigned = true;
        ass.DetectionInd = d;
        ass.v.EstimatedPosImagePix = imagePos;
        ass.PredictedPos = worldPos;
        ass.EstimatedPos = worldPos;

        cand.Assignments{frameInd} = ass;

        obj.tracks{end+1} = cand;
    end
end

function driftUnassignedTracks(obj, unassignedTracksByRow, frameInd)
    % for unassigned tracks initialize EstimatedPos = PredictedPos by default
    for trackInd=unassignedTracksByRow'
       % if trackInd <= tracksCountBefore % track
            a = obj.tracks{trackInd}.Assignments{frameInd};
            assert(~isempty(a.PredictedPos), 'Kalman Predictor must have initialized the PredictedPos (track=%d assignment=%d)', trackInd, frameInd);

            a.EstimatedPos = a.PredictedPos;
            obj.tracks{trackInd}.Assignments{frameInd} = a;

            % TODO: finish track if detections lost for some time
%         else
%             candInd = trackInd - tracksCountBefore;
%             cand = obj.trackCandidates{candInd};
%             
%             obj.trackCandidates{candInd}.DetectionIdList{end+1} = 0;
%             
%             posInd = frameInd - cand.FirstAppearanceFrameIdx + 1;
%             predPos = cand.PredictedPosList{posInd};
%             assert(~isempty(predPos), 'candidate.PredictedPos is not initialized by Kalman predictor(candidate=%d assignment=%d)', candInd, frameInd);
% 
%             obj.trackCandidates{candInd}.EstimatedPosList{end+1} = predPos;
%        end
    end
end

function batchTrackSwimmersInVideo(obj, debug)
    
    ensureInitialized(obj, debug);
    
    videoFilePath = fullfile('output/mvi3177_blueWomanLane3.avi');
    obj.v.detectionsPerFrame = detectHumanBodies(obj,videoFilePath,Inf,debug);
    obj.v.tracks = trackThroughFrames(obj, obj.v.detectionsPerFrame, debug);
    obj.v.videoTracks = generateVideoWithTrackedBodies(obj, obj.v.mmReader, obj.v.framesToTake, obj.v.detectionsPerFrame, obj.v.tracks);

     % status
     fprintf(1, 'tracks count=%d\n', length(obj.v.tracks));
    
    % write output
%     writerObj = VideoWriter('output/mvi3177_741t_kalman1.avi');
%     writerObj.open();
%     writerObj.writeVideo(obj.v.videoTracks);
%     writerObj.close();
    

    % play movie
     new_movie = immovie(obj.v.videoTracks);
     implay(new_movie);
end

function imageWithTracks = adornImageWithTrackedBodies(obj, image, coordType)
    pathStartFrame = max([1, obj.frameInd - 250]);
    
    detects = obj.detectionsPerFrame{obj.frameInd};
    %imageWithTracks = drawDetections(obj, image, detects);

    if strcmp('TopView', coordType)
        desiredImageSize = [size(image,2), size(image,1)];
        image = CameraDistanceCompensator.convertCameraImageToTopView(obj.v.distanceCompensator, image, desiredImageSize);
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
            labelTrackCandidates = true;
            if labelTrackCandidates || ~track.v.IsTrackCandidate
                estPos = lastAss.EstimatedPos;
                estPosImage = obj.getViewCoord(estPos, coordType, lastAss);
                textPos = estPosImage;

                if lastAss.IsDetectionAssigned && ~isempty(box)
                    %textPos = [max(box(1),box(1) + box(3) - 26), box(2) - 13];
                    boxMat = reshape([box{:}],2,[])';
                    textPos = [max(boxMat(:,1)) min(boxMat(:,2))];
                end

                if track.v.IsTrackCandidate
                    text1 = sprintf('-%d', track.v.TrackCandidateId);
                else
                    text1 = int2str(track.Id);
                end

                image = cv.putText(image, text1, textPos, 'Color', candColor);            
            end
        end
    end
    
%     % show each track candidate
%     for r=1:length(obj.trackCandidates)
%         cand=obj.trackCandidates{r};
%         
%         % construct track path as polyline to draw by single command
%         localStart = max([1, fromTime - cand.FirstAppearanceFrameIdx + 1]);
%         localEnd = min([length(cand.EstimatedPosList), toTimeInc - cand.FirstAppearanceFrameIdx + 1]);
% 
%         candPolyline = cell(0,0);
%         for timeInd=localStart:localEnd
%             pos = cand.EstimatedPosList{timeInd};
%             candPolyline{end+1} = pos;
%         end
% 
%         % pick track color
%         %colInd = length(obj.tracks)+r-1;
%         candColor = c_list(1+mod(cand.TrackCandidateId, length(c_list)),:);
% 
%         % draw candidate path
%         if ~isempty(candPolyline)
%             image = cv.polylines(image, candPolyline, 'Closed', false, 'Color', candColor);
%         end
%         
%         % draw initial position
%         if localStart == 1
%             initialCandPos = cand.EstimatedPosList{1};
%             image=cv.circle(image, initialCandPos, 3, 'Color', candColor);
%         end
%         
%         % process last frame
%         det = cand.DetectionIdList{localEnd};
%         box = [];
%         if det > 0
%             % draw shape contour
%             frameDetects = detectionsPerFrame{toTimeInc};
%             detect = frameDetects(det);
%             outlinePixels = detect.OutlinePixels;
% 
%             % convert (Row,Col) into (X,Y)
%             outlinePixels = circshift(outlinePixels, [0 1]);
% 
%             % packs (X,Y) into cell array for cv.polyline
%             outlinePixelsCell = mat2cell(outlinePixels, ones(length(outlinePixels),1),2);
% 
%             image = cv.polylines(image, outlinePixelsCell, 'Closed', true, 'Color', candColor);
% 
%             % draw box in the last frame
%             box=detect.BoundingBox;
%             image=cv.rectangle(image, box, 'Color', candColor);
%         end
% 
%         %
%         % put text for the last frame
%         textPos = cand.EstimatedPosList{localEnd};
%         if ~isempty(box)
%             textPos = [max(box(1),box(1) + box(3) - 26), box(2) - 13];
%         end
%         candText = sprintf('-%d', cand.TrackCandidateId);
%         image = cv.putText(image, candText, textPos, 'Color', candColor);
%     end
    
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
        pos = CameraDistanceCompensator.worldToCamera(obj.v.distanceCompensator, worldPos);

        if assignment.IsDetectionAssigned && norm(assignment.v.EstimatedPosImagePix - pos) > 10
            warning('image and back projected coord diverge too much Expect=%d Actual=%d',assignment.v.EstimatedPosImagePix ,pos);
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
        worldPos = CameraDistanceCompensator.cameraToWorld(obj.v.distanceCompensator, XByRowCell{i});

        expectCameraSize = CameraDistanceCompensator.expectCameraSize;
        pos = CameraDistanceCompensator.scaleWorldToTopViewImageCoord(worldPos, expectCameraSize);

        posTopView{i} = pos;
    end
end

function imageTopView = adornImageWithTrackedBodiesTopView(obj, image)
    desiredImageSize = [size(image,2), size(image,1)];
    imageTopView = CameraDistanceCompensator.convertCameraImageToTopView(obj.v.distanceCompensator, image, desiredImageSize);
    
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

        
%         worldPos = track.Assignments{obj.frameInd}.EstimatedPos;
%         estPosTopView = CameraDistanceCompensator.scaleWorldToTopViewImageCoord(worldPos, desiredImageSize);
%         %imageTopView = cv.rectangle(imageTopView, box, 'Color', candColor);
%         imageTopView=cv.circle(imageTopView, estPosTopView, 3, 'Color', trackColor);
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
    color = c_list(1+mod(track.v.TrackCandidateId, length(c_list)),:);
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

function tracks = testHumanBodyDetectorOnImage(obj, debug)
    I = read(obj.v.mmReader, 282);
    imshow(I);
    
    % temporary mask to highlight lane3
    load(fullfile('data/Mask_lane3Mask.mat'), 'lane3Mask')
    I = utils.applyMask(I, lane3Mask);

    %
    bodyDescrs = obj.v.det.GetHumanBodies(I, debug);

    imshow(I);
    hold on
    k=1;
    for shapeInfo=bodyDescrs
        box=shapeInfo.BoundingBox;
        boxXs = [box(1), box(1) + box(3), box(1) + box(3), box(1), box(1)];
        boxYs = [box(2), box(2), box(2) + box(4), box(2) + box(4), box(2)];
        plot(boxXs, boxYs, 'g');

        % draw outline
        plot(shapeInfo.OutlinePixels(:,2), shapeInfo.OutlinePixels(:,1), 'g');

        % draw dot in the center of bounding box
        plot(shapeInfo.Centroid(1), shapeInfo.Centroid(2), 'g.');
        text(max(box(1),box(1) + box(3) - 26), box(2) - 13, int2str(k), 'Color', 'g');
        k=k+1;
    end
    hold off
end

end % methods

end
