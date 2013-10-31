classdef RunEarthMoverDistance1
    
methods(Static)
    
function obj = create
    obj = utils.TypeErasedClass;
end

function run(obj, debug)
    %RunEarthMoverDistance1.simple1(obj, debug);
    %RunEarthMoverDistance1.testRgbConversion(obj, debug);    
    
    RunEarthMoverDistance1.testEmd1(obj, debug);
    %RunEarthMoverDistance1.analyzeHistogramDistributions(obj,debug);
end

function simple1(obj, debug)
    % s1 = single([1 0; 0 1]);
    % s2 = single([0 0; 1 1]);
    % 
    % d=cv.EMD(s1, s2)
    % 
    % implay(fullfile(cd, 'rawdata/MVI_3178.mov'))
end

function testRgbConversion(obj, debug)
    pixs=[170 187 204; 171 188 205];
    
    pixs32 = RunEarthMoverDistance1.rgb2uint32(pixs);
    
    r=0;
    g=0;
    b=0;
    [r g b]=RunEarthMoverDistance1.uint322rgb(pixs32)
end

function testEmd1(obj, debug)
    appearDirPath = '../dinosaur/appear1';
    bgPixel = [0 0 0];
    cellSizePixels = 3; % power of 2
    histInclMin = 1; % pixels with lesser occurence number are not incuded in histogram
    fprintf('qunatization cellSizePixels=%d\n', cellSizePixels);

    if ~isfield(obj.v, 'appearModel')
        obj.v.appearModel = RunEarthMoverDistance1.learnAppearModelsFromDirectoryStructure(obj, appearDirPath, bgPixel, cellSizePixels, histInclMin, debug);
    end

    if ~isfield(obj.v, 'testHistMats')
        
        testFiles = {...
            {'..\dinosaur\appear1\lane3_blackMan1\mvi3177_3215_4045_blackMan1_t20131030104547.png'},...
            {'..\dinosaur\appear1\lane3_blueWoman1\mvi3177_blueWomanLane3_16frames_t20131030095143.png'},...
            {'..\dinosaur\appear1\lane4_kid1\mvi3177_1461_2041_kid1_lane4_t20131030113417.png'},...
            {'..\dinosaur\appear1\lane4_magentaWoman1\mvi3177_2011_3881_magentaWom_lane4_t20131030111459.png'},...
            };

        testHistMats = {};
        for i=1:length(testFiles)
            groupRgbHist = containers.Map('KeyType','uint32','ValueType','uint32');

            filePath = testFiles{i}{1};
            RunEarthMoverDistance1.accumulateRgbHistogramFromImageFile(groupRgbHist, filePath, bgPixel, cellSizePixels);

            %RunEarthMoverDistance1.trimHistogram(groupRgbHist, histInclMin);
            groupRgbHistMat = RunEarthMoverDistance1.rgbHistogramToArray(groupRgbHist);
            testHistMats{end+1} = groupRgbHistMat;
        end
        obj.v.testHistMats = testHistMats;
    end
    
    RunEarthMoverDistance1.testAppearModelOnOtherShapes(obj.v.appearModel, obj.v.testHistMats);
end

function testEmd1Clear(obj)
    if isfield(obj.v, 'appearModel')
        obj.v = rmfield(obj.v, 'appearModel');
    end
    
    if isfield(obj.v, 'testHistMats')
        obj.v = rmfield(obj.v, 'testHistMats');
    end
end

% groupsDir = directory which contains folder per each target
% Each such folder contains images of shapes of single target.
function appearModel = learnAppearModelsFromDirectoryStructure(obj, appearDir, bgPixel, cellSizePixels, histInclMin, debug)
    appearDirItems = dir(appearDir);
    
    % find name of targets; they are encoded as directory name
    targetsName = {};
    for i=1:length(appearDirItems)
        d = appearDirItems(i);
        
        if d.isdir && (strcmp(d.name,'.') ~= 1 && strcmp(d.name,'..') ~= 1)
            targetsName{end+1} = d.name;
        end
    end
    
    % map between shape name and its histogram (Nx4), N=number of pixels
    % 4elements are = [occur R G B]
    appearModel = containers.Map('KeyType','char','ValueType','any');
    for groupNameInd = 1:length(targetsName)
        groupName = targetsName{groupNameInd};
        fprintf('processing group=%s\n', groupName);
        
        groupRgbHist = containers.Map('KeyType','uint32','ValueType','uint32');
        
        files = dir(fullfile(appearDir, groupName, '*.png'));
        for fileInd=1:length(files)
            file = files(fileInd);
            filePath = fullfile(appearDir, groupName, file.name);
            fprintf('shape pixels=%s ', filePath);
            
            tic;
            RunEarthMoverDistance1.accumulateRgbHistogramFromImageFile(groupRgbHist, filePath, bgPixel, cellSizePixels);
            fprintf('took %.2f sec\n', toc);
        end
        
        RunEarthMoverDistance1.trimHistogram(groupRgbHist, histInclMin);
        
        groupRgbHistMat = RunEarthMoverDistance1.rgbHistogramToArray(groupRgbHist);
        
        appearModel(groupName) = groupRgbHistMat;
    end
end

% removes from histogram map all values less than histInclMin.
% Histogram is passed by reference.
function trimHistogram(groupRgbHist, histInclMin)
    groupRgbHistValues = groupRgbHist.values;
    exclPixelMask = cell2mat(groupRgbHistValues) < histInclMin;

    groupRgbHistKeys = cell2mat(groupRgbHist.keys);
    exclPixelKeys = groupRgbHistKeys(exclPixelMask);
    groupRgbHist.remove(num2cell(exclPixelKeys));
end

% convert hashtable to histogram bar record
function histBarRecordMat = rgbHistogramToArray(groupRgbHist)
    rgb32 = cell2mat(groupRgbHist.keys)';
    
    [r1 g1 b1] = RunEarthMoverDistance1.uint322rgb(rgb32);

    occur = cell2mat(groupRgbHist.values)';
    
    % Occur R G B
    histBarRecordMat = single([occur r1 g1 b1]);
end

function accumulateRgbHistogramFromImageFile(groupRgbHist, imageFilePath, transpPixel, cellSizePixels)
    img = imread(imageFilePath);
    pixels = reshape(img, [], 3);

    % remove black pixels
    %pixels = utils.PW.matRemoveIf(pixels, @(rgb) all(rgb == transpPixel));
    
    transpPixelMat = repmat(transpPixel, size(pixels,1), 1);
    rowIndsToRem = sum(pixels == transpPixelMat, 2) == 3;
    pixels(rowIndsToRem, :) = [];

    RunEarthMoverDistance1.accumulateRgbHistogramFromPixels(groupRgbHist, pixels, cellSizePixels);
end

function accumulateRgbHistogramFromPixels(groupRgbHist, pixels, cellSizePixels)
    % quntize - put pixels in cells of specified size
    % the information is lost but processing perf increases
    cells = fix(pixels ./ cellSizePixels) + 1;
    
    cells32 = RunEarthMoverDistance1.rgb2uint32(cells);
    
    one32 = cast(1, 'uint32');
    arrayfun(@(cell32) RunEarthMoverDistance1.containersMapIncValue(groupRgbHist, cell32, one32), cells32);
end

% Helper for map(key) += incValue. Map is processed by reference.
function containersMapIncValue(map, key, incValue)
    if map.isKey(key)
        map(key) = map(key) + incValue;
    else
        map(key) = incValue;
    end
end

function accumulateRgbHistogramFromPixelsOld(groupRgbHist, pixels, cellSizePixels)
    one32 = cast(1, 'uint32');
    
    for pixInd=1:length(pixels)
        pix = pixels(pixInd,:); % zero based RGB coordinates
        
        % quntize - put pixels in cells of specified size
        % the information is lost but processing perf increases
        cell = fix(pix ./ cellSizePixels) + 1;
        
        cell32 = RunEarthMoverDistance1.rgb2uint32(cell(1),cell(2),cell(3));
        if groupRgbHist.isKey(cell32)
            groupRgbHist(cell32) = groupRgbHist(cell32) + one32;
        else
            groupRgbHist(cell32) = one32;
        end
    end
end

% testHistMats = cell of histograms (each in a form of matrix Nx4)
function testAppearModelOnOtherShapes(appearModel, testHistMats)
    groupsCount = length(appearModel);
    testsCount = length(testHistMats);
    
    distMat = zeros(groupsCount,testsCount);
    
    appearModelKeys = keys(appearModel);
    
    % process matrix upper triangle
    for row=1:groupsCount
        groupName = appearModelKeys{row}
        groupHist = appearModel(groupName);
        
        for col=1:testsCount
            testHist = testHistMats{col};

            tic;
            dist=cv.EMD(groupHist, testHist);
            fprintf('dist(%d,%d)=%.4f found in %.2f sec. S1=%d S2=%d\n', row, col, dist, toc, length(groupHist), length(testHist));

            distMat(row,col) = dist;
        end
    end
    display(distMat);
end

function analyzeHistogramDistributions(obj, debug)
    appearModelMap = obj.v.appearModel.values;
    hist1 = appearModelMap{1};
    hist(hist1(:,1),1:10)
    prctile(hist1(:,1), 90)
    sum(hist1(:,1)<=3)/sum(hist1(:,1))
end

function rgbInt = rgb2uint32(rgbByRow)
    rgbByRow32 = cast(rgbByRow, 'uint32');

    rgbInt = bitor(bitor(bitshift(rgbByRow32(:,1), 24), bitshift(rgbByRow32(:,2), 16)), bitshift(rgbByRow32(:,3), 8));
    
    assert(isa(rgbInt,'uint32'));
end

% [r g b]=uint322rgb(cast(2864434176,'uint32'))
function [r,g,b] = uint322rgb(rgb32)
    assert(isa(rgb32,'uint32'));
    
    r = bitsra(rgb32, 24);
%     g = bitshift(rgb32,8);
%     g = bitsra(g,24);
%     b = bitshift(rgb32,16);
%     b = bitsra(b,24);
    
    % 4278190080 % FF000000
    % 16711680   % 00FF0000
    % 65280      % 0000FF00
    
    g = bitsra(bitand(rgb32, 16711680), 16);
    b = bitsra(bitand(rgb32, 65280), 8);
    
    %assert(isa(r,'uint8'));
    %assert(isa(g,'uint8'));
    %assert(isa(b,'uint8'));
end

function [r,g,b] = uint322rgbOld(rgb32)
    assert(isa(rgb32,'uint32'));
    
    r = bitsra(rgb32, 24);
%     g = bitshift(rgb32,8);
%     g = bitsra(g,24);
%     b = bitshift(rgb32,16);
%     b = bitsra(b,24);
    
    % 4278190080 % FF000000
    % 16711680   % 00FF0000
    % 65280      % 0000FF00
    
    g = bitsra(bitand(rgb32, 16711680), 16);
    b = bitsra(bitand(rgb32, 65280), 8);
    
    %assert(isa(r,'uint8'));
    %assert(isa(g,'uint8'));
    %assert(isa(b,'uint8'));
end

end
end