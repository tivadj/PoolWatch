debug=1;
history=500; %default=500
varThreshold=16; % default=16
bShadowDetection=true; %default=unknown?
bg = cv.BackgroundSubtractorMOG2(history,varThreshold,'BShadowDetection',bShadowDetection);

mmReader = demo1.v.mmReader;

videoFore = zeros(mmReader.Height,mmReader.Width,mmReader.BitsPerPixel/8,mmReader.NumberOfFrames, 'uint8');

for frameInd=1:mmReader.NumberOfFrames
    fprintf('frameInd=%d\n', frameInd);
    
    image = read(mmReader, frameInd);
    if debug
        figure(6), imshow(image);
    end
    
    learningRate = 0; % default=0
    maskFore = bg.apply(image,'LearningRate', learningRate);

    imgBg = bg.getBackgroundImage();
    
    imgFore = utils.applyMask(image, maskFore);
    if debug
        figure(14);
        imshow(imgFore);
    end
    
    videoFore(:,:,:,frameInd)=imgFore;
end

implay(immovie(videoFore), mmReader.FrameRate);

%{
% write output
writerObj = VideoWriter('output/mvi3177_blueWomanLane3_foreground_only_BackgroundSubtractorMOG.avi');
writerObj.open();
writerObj.writeVideo(videoFore);
writerObj.close();
%}
    