% process video
% obj = mmreader('data/MVI_3177.MOV'); % error: not in the path
timeProgramStart = tic;
obj = mmreader(fullfile('output/mvi3177_blueWomanLane3.avi'));
frameRate = get(obj,'FrameRate')
%video = read(obj, [1 100]);
%video = read(obj);
%implay(video, frameRate)

% I2=cv.rectangle(I, [10 10 100 300]);
% imshow(I2);

procFrames=obj.NumberOfFrames;
%framesToTake=1:30:procFrames;
framesToTake=1:procFrames;
%framesToTake=1:50;
framesCount=length(framesToTake);
videoUpd=zeros(obj.Height, obj.Width, floor(obj.BitsPerPixel/8), framesCount, 'uint8');
detectionsPerFrame=cell(framesCount,1);

% find bodies in video
for i=1:framesCount
    num = framesToTake(i);
    fprintf(1, 'processing frame %d', num);
    
    %image=video(:,:,:,num);
    image = read(obj, num);
    %imshow(image)
    
    image = utils.applyMask(image, lane3Mask);
    
    tic
    bodyDescrs = det.GetHumanBodies(image, false);
    t1=toc;
    fprintf(1, ' took time=%f\n', t1);
    
    detectionsPerFrame{i,1} = bodyDescrs;
end


%save('MVI_3177_detections.mat', 'detectionsPerFrame');
fileName=sprintf('output/MVI_3177_detections_%s.mat',datestr(now, 'yyyymmdd-HHMMSS'));
save(fileName, 'detectionsPerFrame');
%load('MVI_3177_143t_detections_20130911-110959.mat', 'detectionsPerFrame');

timeRun=toc(timeProgramStart);
fprintf(1,'program run time=%d\n',timeRun);

% extract convenient parts of video
%{
obj = mmreader(fullfile('rawdata/MVI_3177.MOV'));
blueWomanLane3 = read(obj, [` 2951]);
implay(blueWomanLane3)

% Lane3 - Woman Blue
waterImage = utils.applyMask(I,waterMask);
imshow(waterImage);
% TODO: work separately with lane 3
lane3Mask=roipoly(blueWomanLane3(:,:,:,186));
%save(fullfile('data/Mask_Water1.mat'), 'lane3Mask')
%load(fullfile('data/Mask_lane3Mask.mat'), 'lane3Mask')
imshow(utils.applyMask(blueWomanLane3(:,:,:,186), lane3Mask));
%}