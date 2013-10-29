classdef RunOpticalFlow1
methods(Static)
function obj = create
    obj = utils.TypeErasedClass;
end

function run(obj)
    debug = 1;
    
    %RunOpticalFlow1.testOpticFlowFarneback(obj, debug);
    RunOpticalFlow1.testOpticFlowPyramidalLucasKanade(obj, debug);
end

function testOpticFlowFarneback(obj, debug)
    
    videoReader = VideoReader('../output/mvi3177_blueWomanLane3.avi');
    prevImGray = [];
    prevImRgb = [];
    minFlowOffset = [];
    points = [];
    for i=1:videoReader.NumberOfFrames
        imgRgb = read(videoReader, i);
        imgGray = rgb2gray(imgRgb);
        
        imshow(imgRgb);

%         initFlow = zeros([size(imgGray) 2], 'single');
%         initFlow(:,:,1) = 26;
%         initFlow(:,:,1) = -19;
        if ~isempty(prevImGray)
            %, 'InitialFlow', initFlow
            flow = cv.calcOpticalFlowFarneback(prevImGray, imgGray);
            
            if isempty(minFlowOffset)
                %minFlowOffset=-1;
                %minFlowOffset = RunOpticalFlow1.flowOffsetByPercentile(flow, 99.9);
                %fprintf('optical flow minOffset=%d\n', minFlowOffset);
            end
            
            if isempty(points)
                step = 4;
                %roiRecXY = [209 267 23 22]; % from 16 frames
                roiRecXY = [209 266 20 21]; % from full video
                points = RunOpticalFlow1.chooseUniformPoints(step, roiRecXY, size(flow));
            end
            
            shiftedPoints = RunOpticalFlow1.shiftPointsByFlow(points, flow);
            
            %imgFlow = RunOpticalFlow1.drawFlowNoLoops(imgRgb, flow, 6 , minFlowOffset);
            imgFlow = RunOpticalFlow1.drawFlow(prevImRgb, points, shiftedPoints);
            latter = imshow(imgRgb);
            alpha(latter, 0.4);
            hold on
            former = imshow(imgFlow);
            alpha(former, 0.6);
            hold off
    
            points = shiftedPoints;
        end
        prevImGray = imgGray;
        prevImRgb = imgRgb;
    end
    
end

function minOffset = flowOffsetByPercentile(flow, valuePercent)
    % find min offset threshold
    flowAsRows = reshape(flow, [], 2);
    offsets=cellfun(@norm, num2cell(flowAsRows,2));

    % find threshold value
    minOffset = prctile(offsets, valuePercent);
end

function imgFlowRgb = drawFlowNoLoops(imgRgb, flow, step, minOffset)
    if ~exist('step', 'var')
        step = 16;
    end

    stepX = step; stepY = step;
    %stepX = 17; stepY = 117;
    
    % find positions where to draw flow
    takeYs = step/2:stepY:size(flow,1);
    takeXs = step/2:stepX:size(flow,2);
    [Xsq,Ysq] = meshgrid(takeXs, takeYs);

    x = reshape(Xsq, [], 1);
    y = reshape(Ysq, [], 1);
    
    flowValues = flow(takeYs, takeXs, :);
    flowValues = reshape(flowValues, [], 2);
    
    % take only significant flow
    bigFlowMask = sum(flowValues.^2,2) > minOffset;
    
    
    % construct line begin/end points
    begPoints = [x y];
    begPoints = begPoints(bigFlowMask,:);
    
    offsetRows = flowValues(bigFlowMask,:);
    
    endPoints = begPoints + offsetRows;
    
    % prepare data for drawing
    polylines = [num2cell(begPoints,2) num2cell(endPoints,2)];
    polylines =  num2cell(polylines, 2);

    col = [0 255 0];
    imgFlowRgb = cv.polylines(imgRgb, polylines, 'Color', col);
%     for i=1:size(begPoints,1)
%         imgFlowRgb = cv.circle(imgFlowRgb, begPoints(i,:), 1, 'Color', col);
%     end
end

function imgFlowRgb = drawFlow(imgRgb, points, shiftedPoints)
    imgFlowRgb = imgRgb;

    if iscell(points)
        len = length(points);
    else
        len = size(points,1);
    end
    
    for i=1:len
        if iscell(points)
            p1 = points{i};
        else
            p1 = points(i,:);
        end
        if iscell(shiftedPoints)
            p2 = shiftedPoints{i};
        else
            p2 = shiftedPoints(i,:);
        end
        imgFlowRgb = cv.line(imgFlowRgb, p1, p2, 'Color', [0 255 0], 'LineType', 8);
    end
end

function testOpticFlowPyramidalLucasKanade(obj, debug)
    
    videoReader = VideoReader('../output/mvi3177_blueWomanLane3.avi');
    prevImGray = [];
    prevImRgb = [];
    goodTrackPoints = [];
    for i=1:videoReader.NumberOfFrames
        imgRgb = read(videoReader, i);
        imgGray = rgb2gray(imgRgb);
        
        imshow(imgRgb);

        if isempty(goodTrackPoints)
            step = 4;
            %roiRecXY = [209 267 23 22]; % from 16 frames
            roiRecXY = [209 266 20 21]; % from full video           
            goodTrackPoints = RunOpticalFlow1.chooseUniformPoints(step, roiRecXY, size(imgGray));
            goodTrackPoints = num2cell(goodTrackPoints, 2);
            %goodTrackPoints = cv.goodFeaturesToTrack(imgGray);
        end
        
        if ~isempty(prevImGray)
            [newTrackPoints,status,errs] = cv.calcOpticalFlowPyrLK(prevImGray, imgGray, goodTrackPoints);
            
            imgFlow = RunOpticalFlow1.drawFlow(prevImRgb, goodTrackPoints, newTrackPoints);

            latter = imshow(imgRgb);
            alpha(latter, 0.4);
            hold on
            former = imshow(imgFlow);
            alpha(former, 0.6);
            hold off
            
            % draw lines
%             lines = num2cell([goodTrackPoints' newTrackPoints'], 2);
%             imgRgb = cv.polylines(imgRgb, lines, 'Color', [0 255 0]);
%             imshow(imgRgb);

            goodTrackPoints = newTrackPoints;
        end

        prevImGray = imgGray;
        prevImRgb = imgRgb;
    end
    
end

function posList = chooseUniformPoints(step, roiRecXY, flowSize)
    colLeft  = floor(roiRecXY(1) + step/2);
    colRight = min([floor(roiRecXY(1) + roiRecXY(3) - 1) flowSize(2)]);
    rowTop = floor(roiRecXY(2) + step/2);
    rowBot = min([floor(roiRecXY(2) + roiRecXY(4) - 1) flowSize(1)]);

    posList = [];

    for col=colLeft:step:colRight
    for row=rowTop:step:rowBot
        posList = [posList; [col row]];
    end
    end
end

function shiftedPoints = shiftPointsByFlow(points, flow)
    shiftedPoints = zeros(size(points),class(points));
    
    for i=1:size(points,1)
        col = floor(points(i,1));
        row = floor(points(i,2));
        
        flowVec = squeeze(flow(row,col,:));
        fprintf('pos=(%d; %d) flow=(%f; %f)\n', col, row, flowVec(1), flowVec(2));
        
        shiftedPoints(i,:) = [col+flowVec(1) row+flowVec(2)];
    end
end

end
end