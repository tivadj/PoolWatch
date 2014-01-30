classdef VideoHelper
methods(Static)

function frames = readFrames(filePath, frameInds)
    videoReader = VideoReader(filePath);
    
    if ~exist('frameInds', 'var')
        frameInds = 1:videoReader.NumberOfFrames;
    end
    
    outFramesCount = length(frameInds);

    % assume RGB images
    frames  = zeros(videoReader.Height, videoReader.Width, videoReader.BitsPerPixel/8, outFramesCount, 'uint8');
    
    ord = 1;
    for frameInd=frameInds
        image = read(videoReader, frameInd);
        frames(:,:,:, ord) = image;
        
        ord = ord + 1;
    end
end

function frame = readFrameSingle(filePath, frameInd)
    assert(length(frameInd) == 1, 'Expected single frame index for reading');
    frame = squeeze(utils.VideoHelper.readFrames(filePath, frameInd));
end

%{
% write output
    writerObj = VideoWriter('../output/mvi3177_blueWomanLane3_3_lane_24Oct1107.avi');
    writerObj.open();
    writerObj.writeVideo(poolTracker.v.videoWithTracksDual);
    writerObj.close();
%}

end
end