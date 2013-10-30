classdef RunEarthMoverDistance1
    
methods(Static)
    
function obj = create
    obj = utils.TypeErasedClass;
end

function run(obj, debug)
    %RunEarthMoverDistance1.simple1(obj, debug);
    %RunEarthMoverDistance1.testEmdOnCoupleOfFiles(obj, debug);
    
    RunEarthMoverDistance1.testEmd1(obj, debug);
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
    r=0;
    g=0;
    b=0;
    %
    rgb32 = rgb2uint32(170,187,204)
    [r g b]=uint322rgb(cast(2864434176,'uint32'))
end

function testEmd1(obj, debug)
    appearDirPath = '../dinosaur/appear1';
    bgPixel = [0 0 0];

    if ~isfield(obj.v, 'appearModel')
        obj.v.appearModel = RunEarthMoverDistance1.learnAppearModelsFromDirectoryStructure(obj, appearDirPath, bgPixel, debug);
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
            RunEarthMoverDistance1.accumulateRgbHistogramFromImageFile(groupRgbHist, filePath, bgPixel);

            groupRgbHistMat = RunEarthMoverDistance1.rgbHistogramToArray(groupRgbHist);
            testHistMats{end+1} = groupRgbHistMat;
        end
        obj.v.testHistMats = testHistMats;
    end
    
    RunEarthMoverDistance1.testAppearModelOnCandidates(obj.v.appearModel, obj.v.testHistMats);
end

% groupsDir = directory which contains folder per each target
% Each such folder contains images of shapes of single target.
function appearModel = learnAppearModelsFromDirectoryStructure(obj, appearDir, bgPixel, debug)
    appearDirItems = dir(appearDir);
%     names = {appearDir.name};
%     names = names([appearDir.isdir]);
    
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
        
        groupRgbHist = containers.Map('KeyType','uint32','ValueType','uint32');
        
        files = dir(fullfile(appearDir, groupName, '*.png'));
        for fileInd=1:length(files)
            file = files(fileInd);
            filePath = fullfile(appearDir, groupName, file.name);
            
            RunEarthMoverDistance1.accumulateRgbHistogramFromImageFile(groupRgbHist, filePath, bgPixel);
        end
        
        groupRgbHistMat = RunEarthMoverDistance1.rgbHistogramToArray(groupRgbHist);
        
        appearModel(groupName) = groupRgbHistMat;
    end
end

function testEmdOnCoupleOfFiles(obj, debug)
%     fileGroups={{'MVI_3177_175_woman_black','MVI_3177_2047_woman_black'},
%     {'MVI_3178_602_black1','MVI_3178_1913_black1'},
%     {'MVI_3178_2297_kid1','MVI_3178_2453_kid1'}};

    if (~isfield(obj.v, 'appearModel'))
        RunEarthMoverDistance1.learnAppearModelsOnCoupleOfFiles(obj, fileGroups, debug);
    end

    fprintf('finding distances between histograms\n');

    %!!!
    groupsCount = length(obj.v.appearModel);
    distMat = zeros(groupsCount,groupsCount);
    % process matrix upper triangle
    for row=1:(groupsCount-1)
    for col=(row+1):groupsCount
        imageBarRecordA = obj.v.appearModel(int2str(row));
        imageBarRecordB = obj.v.appearModel(int2str(col));

        tic;
        dist=cv.EMD(imageBarRecordA, imageBarRecordB);
        fprintf('dist(%d,%d)=%.4f found in %.2f sec. S1=%d S2=%d\n', row, col, dist, toc, length(imageBarRecordA), length(imageBarRecordB));

        distMat(row,col) = dist;
    end
    end
    display(distMat);

    % MVI_3177_175_woman_black.svg
    % MVI_3177_2047_woman_black.svg
    % MVI_3178_1913_black1.svg
    % MVI_3178_2297_kid1.svg
    % MVI_3178_2453_kid1.svg
    % MVI_3178_602_black1.svg
    %          0    4.6321   43.0203   25.5680   36.3205   50.2176
    %          0         0   12.7576    6.4162    7.8527   10.6642
    %          0         0         0   10.8882   19.5881   23.1406
    %          0         0         0         0   13.3240   11.7520
    %          0         0         0         0         0   16.8781
    %          0         0         0         0         0         0
end


function learnAppearModelsOnCoupleOfFiles(obj, fileGroups, debug)
    appearDir = fullfile('data/appearance');
    %appearFilesPattern = fullfile(appearDir, '*.svg');
    %svgFiles = dir(appearFilesPattern);

    % construct histograms for each image
    groupsCount = length(fileGroups);

    % map between shape name and its histogram (Nx4), N=number of pixels
    % 4elements are = [occur R G B]
    appearModel = containers.Map('KeyType','char','ValueType','any');

    for groupInd=1:groupsCount
        fprintf('processing image group=%d\n', groupInd);
        groupName = int2str(groupInd);

        groupRgbHist = containers.Map('KeyType','uint32','ValueType','uint32');

        group = fileGroups{groupInd};
        for fileNameInd=1:length(group)
            fileName = group{fileNameInd};
            filePath = fullfile(appearDir, strcat(fileName, '.svg'));
            pixels = utils.getPixelsFromLabelledImage(filePath, false, '#00FFFF');

            RunEarthMoverDistance1.accumulatePixelsInRgbHistogramMap(groupRgbHist, pixels);
        end

        fprintf(1, 'rgbHist constructed in % sec\n', toc);

        histBarRecordMat = RunEarthMoverDistance1.rgbHistogramToArray(groupRgbHist);

        appearModel(groupName) = histBarRecordMat;
    end
    
    obj.v.appearModel = appearModel;
end

% convert hashtable to histogram bar record
function histBarRecordMat = rgbHistogramToArray(groupRgbHist)
    histBarRecordMat = zeros(length(groupRgbHist),4,'single'); % Occur R G B

    rgbHistKeys = keys(groupRgbHist);
    i=1;
    for rgb32=[rgbHistKeys{:}]
        occur = groupRgbHist(rgb32);
        [r1 g1 b1] = RunEarthMoverDistance1.uint322rgb(rgb32);

        histBarRecordMat(i,:) = single([occur r1 g1 b1]);
        i=i+1;
    end
end

function accumulateRgbHistogramFromImageFile(groupRgbHist, imageFilePath, transpPixel)
    img = imread(imageFilePath);
    pixels = reshape(img, [], 3);

    % remove black pixels
    utils.PW.matRemoveIf(pixels, @(rgb) rgb(1) == transpPixel(1) && rgb(2) == transpPixel(2) && rgb(3) == transpPixel(3));

    RunEarthMoverDistance1.accumulateRgbHistogramFromPixels(groupRgbHist, pixels);
end

function accumulateRgbHistogramFromPixels(groupRgbHist, pixels)
    for pixInd=1:length(pixels)
        pix = pixels(pixInd,:) + 1; % +1 to Matlab index
        rgb32 = RunEarthMoverDistance1.rgb2uint32(pix(1),pix(2),pix(3));
        if groupRgbHist.isKey(rgb32)
            groupRgbHist(rgb32) = groupRgbHist(rgb32) + cast(1, 'uint32');
        else
            groupRgbHist(rgb32) = cast(1, 'uint32');
        end
    end
end

% testHistMats = cell of histograms (each in a form of matrix Nx4)
function testAppearModelOnCandidates(appearModel, testHistMats)
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

% rgb2uint32(170,187,204)
function rgbInt = rgb2uint32(r, g, b)
    r32 = cast(r, 'uint32');
    g32 = cast(g, 'uint32');
    b32 = cast(b, 'uint32');
    rgbInt = bitor(bitor(bitshift(r32, 24), bitshift(g32, 16)), bitshift(b32, 8));
    
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

end
end