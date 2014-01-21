classdef TrackPainter
%TRACKPAINTER Summary of this class goes here
%   Detailed explanation goes here
    
methods(Static)

function imageWithTracks = adornImageWithTrackedBodies(image, coordType, queryFrameInd, detectionsPerFrame, tracks, distanceCompensator)
    pathStartFrame = max([1, queryFrameInd - 250]);
    
    %detects = detectionsPerFrame{queryFrameInd};
    %imageWithTracks = drawDetections(obj, image, detects);

    if strcmp('TopView', coordType)
        desiredImageSize = [size(image,2), size(image,1)];
        image = distanceCompensator.convertCameraImageToTopView(image, desiredImageSize);
    end

    imageWithTracks = TrackPainter.adornTracks(image, pathStartFrame, queryFrameInd, detectionsPerFrame, tracks, distanceCompensator, coordType);
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

function imageAdorned = adornTracks(image, fromTime, toTimeInc, detectionsPerFrame, tracks, distanceCompensator, coordType)
    % show each track
    for r=1:length(tracks)
        track=tracks{r};
        
        % pick track color
        % color should be the same for track candidate and later for the track
        candColor = TrackPainter.getTrackColor(track);
        
        % construct track path as polyline to draw by single command
        [candPolyline,initialCandPos] = TrackPainter.buildTrackPath(track, fromTime, toTimeInc, coordType, distanceCompensator);
        
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
                    outlinePixelsCellTop = TrackPainter.cameraToTopView(outlinePixelsCell, distanceCompensator);
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
                    box = TrackPainter.cameraToTopView(box, distanceCompensator);
                end

                %image=cv.rectangle(image, box, 'Color', candColor);
                image=cv.polylines(image, box, 'Closed', true, 'Color', candColor);
            end
            
            %
            % put text for the last frame
            labelTrackCandidates = false;
            if labelTrackCandidates || ~track.IsTrackCandidate
                estPos = lastAss.EstimatedPos;
                estPosImage = TrackPainter.getViewCoord(estPos, coordType, lastAss, distanceCompensator);
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
function [trackPolyline,initialTrackPos] = buildTrackPath(track, fromTime, toTimeInc, coordType, distanceCompensator)
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
        estPosImage = TrackPainter.getViewCoord(worldPos, coordType, ass, distanceCompensator);
            
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

function pos = getViewCoord(worldPos, coordType, assignment, distanceCompensator)
    if strcmp('camera', coordType)
        pos = distanceCompensator.worldToCamera(worldPos);

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

function posTopView = cameraToTopView(XByRowCell, distanceCompensator)
    posTopView = cell(1,length(XByRowCell));
    
    for i=1:length(XByRowCell)
        worldPos = CameraDistanceCompensator.cameraToWorld(distanceCompensator, XByRowCell{i});

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

function color = getTrackColor(track)
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

end % methods    
end

