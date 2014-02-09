classdef RunSwimmingPoolVideoFileTracker
methods(Static)

function obj = create()
    obj = utils.TypeErasedClass;   
end

function run(obj)
    debug = 0;
    renderTopView = false;
    % for debugging
    obj.v.keepTrackHistory = false;
    
    if ~isfield(obj.v, 'tracker')
        obj.v.tracker=utils.PW.createSwimmerTracker(debug);
    end
    
    RunSwimmingPoolVideoFileTracker.init(obj,renderTopView);
    RunSwimmingPoolVideoFileTracker.processFrames(obj, renderTopView, debug);
    %RunSwimmingPoolVideoFileTracker.testSwimmerShapeAreaVsDistance(obj, debug);
end

function init(obj, renderTopView)
    % open video
    
    %videoFilePath = fullfile('data/mvi3177_blueWomanLane3_16frames.avi');
    videoFilePath = fullfile('../../output/mvi3177_blueWomanLane3.avi');
    %videoFilePath = fullfile('../rawdata/MVI_3177.mov');
    %videoFilePath = fullfile('../dinosaur/mvi3177_1461_2041_kid1_lane4.avi');
    
    videoReader = VideoReader(videoFilePath);
    obj.v.videoReader = videoReader;

    frameRate = get(videoReader,'FrameRate');
    procFrames=videoReader.NumberOfFrames;
    fprintf('video FrameRate=%d NumberOfFrames=%d\n', frameRate, procFrames);
    

    %
    toFrame=Inf;
    framesToTakeLast = min([procFrames toFrame]);
    fprintf('analysis of first %d frames\n', framesToTakeLast);
    
    takeEachNthFrame = 1;
    framesToTake=1:takeEachNthFrame:framesToTakeLast;
    %framesToTake=floor(10/1100*procFrames):takeEachNthFrame:framesToTakeLast;
    %framesToTake=2476;
    %framesToTake=16;
    obj.v.framesToTake = framesToTake;

    % fps
    fps = videoReader.FrameRate / double(takeEachNthFrame);
    obj.v.fps = fps;
    obj.v.elapsedTimePerFrameMs = 1000 / fps;
    %obj.v.elapsedTimePerFrameMs = 1000 / videoReader.FrameRate * 29; % for /mvi3177_blueWomanLane3_16frames.avi

    %
    framesCount=length(framesToTake);
    obj.v.framesCount = framesCount;
    fprintf('framesCount=%d\n', framesCount);

    % temporary mask to highlight lane3
    load(fullfile('data/Mask_lane3Mask.mat'), 'lane3Mask'); obj.v.laneMask = lane3Mask;
    %load(fullfile('../dinosaur/Mask_lane3_blackMan1.mat'), 'lane3_blackMan1'); obj.v.laneMask = lane3_blackMan1;
    %load(fullfile('../dinosaur/Mask_lane4_1.mat'), 'lane4_1'); obj.v.laneMask = lane4_1;
    %load(fullfile('../dinosaur/Mask_lane2_2.mat'), 'lane2Mask'); obj.v.laneMask = lane2Mask;
    

    %
    outWidth = videoReader.Width;
    
    if renderTopView
        outWidth = outWidth * 2; % allocate space to the rightfor TopView
    end
    
    obj.v.videoWithTracksDual = zeros(videoReader.Height,outWidth,videoReader.BitsPerPixel/8,framesCount, 'uint8');
end

function processFrames(obj, renderTopView, debug)
    
    if ~obj.v.keepTrackHistory
        obj.v.tracker.purgeMemory();
    end    
    
    if ~renderTopView
        % reset subplotting
        if ~isempty(get(0, 'Children'))
            subplot(1,1,1);
        end
    end
    
    % track video
    frameOrder=int32(1);
    
    for frameInd = obj.v.framesToTake
        fprintf(1, 'SwimmerTracker: processing frame %d (#%d of %d)\n', frameInd, frameOrder, length(obj.v.framesToTake));
        
        userBreakFile = dir('_UserBreak.txt');
        if ~isempty(userBreakFile)
            warning('Interrupted by user');
            break;
        end
        
        if obj.v.keepTrackHistory
            rewindToFrame = frameInd - 1; % is incremented when started processing frame
            %obj.v.tracker.rewindToFrameDebug(rewindToFrame, debug);
            obj.v.tracker.frameInd = rewindToFrame;
        end

        %image=video(:,:,:,num);
        image = read(obj.v.videoReader, frameInd);
        %imshow(image)
        
        %
        %image = utils.applyMask(image, obj.v.laneMask);
        %imshow(image)

        % do frame analysis
        
        obj.v.tracker.nextFrame(image, obj.v.elapsedTimePerFrameMs, obj.v.fps, debug);

        % get debug image

        queryFrameInd = obj.v.tracker.getFrameIndWithReadyTrackInfo();
        if true && queryFrameInd ~= -1
            queryImage = read(obj.v.videoReader, queryFrameInd);
            
            imageWithTracks  = obj.v.tracker.adornImageWithTrackedBodies(queryImage, 'camera', queryFrameInd);

            if renderTopView
                subplot(2,1,1);
            end
            if true || debug
                imshow(imageWithTracks)
                pause(0.5);
            end

            if renderTopView
                %imageWithTracksTopView = obj.v.tracker.adornImageWithTrackedBodiesTopView(image);
                imageWithTracksTopView =obj.v.tracker.adornImageWithTrackedBodies(queryImage, 'TopView');
                subplot(2,1,2);
                imshow(imageWithTracksTopView)
            end

            % store frame

            obj.v.videoWithTracksDual(:,1:size(queryImage,2),:,frameOrder) = imageWithTracks;
        end
        
        if renderTopView
            obj.v.videoWithTracksDual(:,size(image,2)+1:end,:,frameOrder) = imageWithTracksTopView;
        end

        drawnow;
        frameOrder=frameOrder+1;
    end
    
    % status
    fprintf(1, 'tracks count=%d\n', length(obj.v.tracker.tracksHistory));

    %{
    % write output
        framesCount = size(poolTracker.v.videoWithTracksDual,4);
        [~,videoFileName,~] = fileparts(poolTracker.v.videoReader.Name);
        outFile = sprintf('../../output/%s_%s_n%d.avi',videoFileName, utils.PW.timeStampNow, framesCount)
        writerObj = VideoWriter(outFile);
        writerObj.FrameRate = 30;
        writerObj.open();
        writerObj.writeVideo(poolTracker.v.videoWithTracksDual(:,:,:,1:framesCount));
        writerObj.close();
        implay(immovie(poolTracker.v.videoWithTracksDual(:,:,:,1:framesCount)), poolTracker.v.videoReader.FrameRate);
    %}

    % play movie
    videoWithTracksMovie = immovie(obj.v.videoWithTracksDual(:,:,:,1:length(obj.v.framesToTake)));
    implay(videoWithTracksMovie, obj.v.videoReader.FrameRate);
end

% Check how shape area changes when it moves far away from camera.
function testSwimmerShapeAreaVsDistance(obj, debug)
    %obj.v.tracker.detectionsPerFrame{1:
    theTrack = [];
    for trackInd=1:length(obj.v.tracker.tracks)
        track = obj.v.tracker.tracks{trackInd};
        if track.TrackCandidateId == 1
            theTrack = track;
            break;
        end            
    end
    assert(~isempty(theTrack));
    
    shapeHist = [];
    
    for frameInd=1:length(track.Assignments)
        ass=track.Assignments{frameInd};
        
        if ~ass.IsDetectionAssigned
            continue;
        end
        
        %
        
        detect = obj.v.tracker.detectionsPerFrame{frameInd}(ass.DetectionInd);

        im1 = zeros(obj.v.mmReader.Height, obj.v.mmReader.Width, 'uint8');
        %im1(detect.OutlinePixels(:,1), detect.OutlinePixels(:,2)) = 255;
        inds = sub2ind([obj.v.mmReader.Height, obj.v.mmReader.Width], detect.OutlinePixels(:,1), detect.OutlinePixels(:,2));
        im1(inds) = 255;
        %imshow(im1);
        im2=imfill(im1);
        %imshow(im2);
        
        area = sum(im2(:) > 0);
        
        % Y in rectangular coordinates
        T=[ -0.4503   -3.7396  726.9509;
            -0.1145   -3.3700  584.8146;
            -0.0003   -0.0063    1.0000]
        yrec = T * [detect.Centroid 1]';
        yrec = yrec / yrec(3);

        shapeHist = [shapeHist; frameInd yrec(2) area];
    end
    
    obj.v.shapeHist = shapeHist;
    plot(obj.v.shapeHist(:,2), obj.v.shapeHist(:,3))
    
%     plot(demo1.v.shapeHist(:,2), demo1.v.shapeHist(:,3),'.')
%     
%     plot(demo1.v.shapeHist(:,2), demo1.v.shapeHist(:,3),'.')
%     x1=polyfit(demo1.v.shapeHist(:,2), demo1.v.shapeHist(:,3),1)
%     x2=polyfit(demo1.v.shapeHist(:,2), demo1.v.shapeHist(:,3),2)
%     xs=390:480;
%     hold on
%     %plot(xs, x1(1)*xs+x1(2),'r')
%     plot(xs, x2(1)*xs.^2+x2(2)*xs+x2(3),'b')
%     hold off
    
end

end
end