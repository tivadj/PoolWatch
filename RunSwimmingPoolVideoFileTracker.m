classdef RunSwimmingPoolVideoFileTracker
methods(Static)

function obj = create()
    obj = utils.TypeErasedClass;
    obj.v.debug = false;
    obj.v.tr=SwimmerTracker;
    
    % for debugging
    obj.v.keepTrackHistory = false;
end

function run(obj)
    debug = 1;
    renderTopView = false;
    obj.v.keepTrackHistory = false;
    
    RunSwimmingPoolVideoFileTracker.init(obj,renderTopView);
    RunSwimmingPoolVideoFileTracker.processFrames(obj, renderTopView, debug);
end

function init(obj, renderTopView)
    % open video
    
    %videoFilePath = fullfile('data/mvi3177_blueWomanLane3_16frames.avi');
    videoFilePath = fullfile('../output/mvi3177_blueWomanLane3.avi');
    %videoFilePath = fullfile('../rawdata/MVI_3177.mov');
    
    videoReader = VideoReader(videoFilePath);
    obj.v.videoReader = videoReader;

    frameRate = get(videoReader,'FrameRate');
    procFrames=videoReader.NumberOfFrames;
    fprintf('video FrameRate=%d NumberOfFrames=%d\n', frameRate, procFrames);
    
    %obj.v.elapsedTimePerFrameMs = 1000 / videoReader.FrameRate;
    obj.v.elapsedTimePerFrameMs = 1000 / videoReader.FrameRate * 29; % for /mvi3177_blueWomanLane3.avi


    toFrame=Inf;
    framesToTakeLast = min([procFrames toFrame]);
    fprintf('analysis of first %d frames\n', framesToTakeLast);
    
    %framesToTake=1:20:procFrames;
    framesToTake=285:framesToTakeLast;
    %framesToTake=2476;
    %framesToTake=16;
    obj.v.framesToTake = framesToTake;

    %
    framesCount=length(framesToTake);
    obj.v.framesCount = framesCount;
    fprintf('framesCount=%d\n', framesCount);

    % temporary mask to highlight lane3
    lane3Mask = [];
    load(fullfile('data/Mask_lane3Mask.mat'), 'lane3Mask')
    obj.v.lane3Mask = lane3Mask;

    %
    outWidth = videoReader.Width;
    
    if renderTopView
        outWidth = outWidth * 2; % allocate space to the rightfor TopView
    end
    
    obj.v.videoWithTracksDual = zeros(videoReader.Height,outWidth,videoReader.BitsPerPixel/8,framesCount, 'uint8');
end

function processFrames(obj, renderTopView, debug)
    
    if ~obj.v.keepTrackHistory
        obj.v.tr.purgeMemory();
    end    
    
    if ~renderTopView
        subplot(1,1,1); % reset subplotting
    end
    
    % track video
    i=1;
    
    for frameInd = obj.v.framesToTake
        fprintf(1, 'SwimmerTracker: processing frame %d (#%d of %d)\n', frameInd, i, length(obj.v.framesToTake));
        
        userBreakFile = dir('_UserBreak.txt');
        if ~isempty(userBreakFile)
            warning('Interrupted by user');
            break;
        end
        
        if obj.v.keepTrackHistory
            rewindToFrame = frameInd - 1; % is incremented when started processing frame
            %obj.v.tr.rewindToFrameDebug(rewindToFrame, debug);
            obj.v.tr.frameInd = rewindToFrame;
        end

        %image=video(:,:,:,num);
        image = read(obj.v.videoReader, frameInd);
        %imshow(image)

        image = utils.applyMask(image, obj.v.lane3Mask);

        % do frame analysis
        
        obj.v.tr.nextFrame(image, obj.v.elapsedTimePerFrameMs, debug);

        % get debug image

        imageWithTracks = obj.v.tr.adornImageWithTrackedBodies(image, 'camera');
        if renderTopView
            subplot(2,1,1);
        end
        imshow(imageWithTracks)
        
        if renderTopView
            %imageWithTracksTopView = obj.v.tr.adornImageWithTrackedBodiesTopView(image);
            imageWithTracksTopView =obj.v.tr.adornImageWithTrackedBodies(image, 'TopView');
            subplot(2,1,2);
            imshow(imageWithTracksTopView)
        end
    
        % store frame
        
        obj.v.videoWithTracksDual(:,1:size(image,2),:,i) = imageWithTracks;
        
        if renderTopView
            obj.v.videoWithTracksDual(:,size(image,2)+1:end,:,i) = imageWithTracksTopView;
        end

        drawnow;
        i=i+1;
    end
    
    % status
    fprintf(1, 'tracks count=%d\n', length(obj.v.tr.tracks));

    %{
    % write output
        framesCount = size(poolTracker.v.videoWithTracksDual,4);
        [~,videoFileName,~] = fileparts(poolTracker.v.videoReader.Name);
        outFile = sprintf('../output/%s_n%d_%s.avi',videoFileName, framesCount, utils.PW.timeStampNow)
        writerObj = VideoWriter(outFile);
        writerObj.FrameRate = 30;
        writerObj.open();
        writerObj.writeVideo(poolTracker.v.videoWithTracksDual);
        writerObj.close();
    %}

    % play movie
    videoWithTracksMovie = immovie(obj.v.videoWithTracksDual);
    implay(videoWithTracksMovie, obj.v.videoReader.FrameRate);
end
    
% Check how shape area changes when it moves far away from camera.
function testSwimmerShapeAreaVsDistance(obj)
    %obj.v.tr.detectionsPerFrame{1:
    theTrackInd = -1;
    for trackInd=1:length(obj.v.tr.tracks)
        if obj.v.tr.tracks{trackInd}.Id == 1
            theTrackInd = trackInd;
            break;
        end            
    end
    assert(theTrackInd > 0);
    
    track=obj.v.tr.tracks{theTrackInd};
    shapeHist = [];
    %track.Assignments
    
    for frameInd=1:length(track.Assignments)
        ass=track.Assignments{frameInd};
        
        if ~ass.IsDetectionAssigned
            continue;
        end
        
        %
        
        detect = obj.v.tr.detectionsPerFrame{frameInd}(ass.DetectionInd);

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