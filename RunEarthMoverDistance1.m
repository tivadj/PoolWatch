classdef RunEarthMoverDistance1
    
methods(Static)
    
function obj = create
    obj = utils.TypeErasedClass;
end

function run(obj)
    debug = 1;
    
    %RunEarthMoverDistance1.simple1(obj, debug);
    %RunEarthMoverDistance1.testShowImageUsingColorSignatureColors(obj, debug);
    %RunEarthMoverDistance1.testRgbConversion(obj, debug);    
    
    %RunEarthMoverDistance1.analyzeHistogramDistributions(obj,debug);
    %RunEarthMoverDistance1.testColorAppearModelHistEarthMoverDistance1(obj, debug);
    %RunEarthMoverDistance1.testColorAppearModelHistBackProjection(obj, debug);
    %RunEarthMoverDistance1.testColorAppearModelMixtureOfGaussians(obj, debug);
    RunEarthMoverDistance1.testAppearModelRecognitionVersusNumberOfPixelsForTraining(obj, debug);
end

function simple1(obj, debug)
    % s1 = single([1 0; 0 1]);
    % s2 = single([0 0; 1 1]);
    % 
    % d=cv.EMD(s1, s2)
    % 
    % implay(fullfile(cd, 'rawdata/MVI_3178.mov'))
end

function testShowImageUsingColorSignatureColors(obj, debug)
    image = imread('../dinosaur/percepMetrImageDbFig31.png');
    figure(1), imshow(image);
    
    for nclust=2:16
    
    pixs = reshape(image, [], 3);
    mixGauss = cv.EM('Nclusters', nclust, 'CovMatType', 'Spherical');
    mixGauss.train(pixs);
    
    %utils.PixelClassifier.drawMixtureGaussiansEach(mixGauss.Means,mixGauss.Covs,mixGauss.Weights);
    
%     hold on
%     scatter3(pixs(:,1),pixs(:,2),pixs(:,3),1,pixs/255);
%     hold off
    
    rgb2FeaturesFun = @(pixsRgb) pixsRgb;
    [imageSig,uniqueColorsCount]=RunEarthMoverDistance1.imageUsingColorSignature(image,mixGauss.Means,mixGauss.Covs,mixGauss.Weights,rgb2FeaturesFun);
    uniqueColorsCount
    figure(2), imshow(imageSig);
    
    imwrite(imageSig,sprintf('../output/flower_nclust%2d_uniqcolor%d.png', nclust, uniqueColorsCount));
    
    end
end

function testRgbConversion(obj, debug)
    pixs=[170 187 204; 171 188 205];
    
    pixs32 = RunEarthMoverDistance1.rgb2uint32(pixs);
    
    r=0;
    g=0;
    b=0;
    [r g b]=RunEarthMoverDistance1.uint322rgb(pixs32)
end

function prepareTrainTestData(obj, debug)
    if isfield(obj.v, 'data')
        return;
    end
        
    [data, labelNames] = RunEarthMoverDistance1.loadColorAppearanceData(obj, debug);
    obj.v.labelNames = labelNames;
    obj.v.data = data;
    
    trainRatio = 0.5;
    %trainRatio = 0.5+0.3*rand;
    obj.v.trainRatio = trainRatio;
    trainMask = RunEarthMoverDistance1.splitTrainTestCases([data.Label], trainRatio);        
    obj.v.trainMask = trainMask;
end

function testColorAppearModelHistEarthMoverDistance1(obj, debug)
    bgPixel = [0 0 0];
    cellSizePixels = 3; % power of 2 NOTE: not used
    channels = [1 3];
    histSize = [32 32];
    histInclMin = 1; % pixels with lesser occurence number are not incuded in histogram
    fprintf('qunatization cellSizePixels=%d\n', cellSizePixels);
    trainRatio = 0.5;
    %trainRatio = 0.5+0.3*rand;

    RunEarthMoverDistance1.prepareTrainTestData(obj, debug);

    if isfield(obj.v, 'appearModel')
        appearModel = obj.v.appearModel;
    else
        appearModel = RunEarthMoverDistance1.learnColorHistFromData(obj, obj.v.data, obj.v.labelNames, obj.v.trainMask, bgPixel, cellSizePixels, histInclMin, channels, histSize, debug);
        obj.v.appearModel = appearModel;
    end

    % test
    
    testMask = ~obj.v.trainMask;
    classCount = length(obj.v.labelNames);
    
    labelClassifFun = @(pixs) RunEarthMoverDistance1.classifyColorHistEarthMoverDistance(appearModel, pixs, channels, histSize);
    
    confMat = RunEarthMoverDistance1.testClassifier(classCount, obj.v.data, testMask, labelClassifFun);

    RunEarthMoverDistance1.printConfusionMatrix(confMat);
    
    recogRate = utils.PixelClassifier.recognitionRate(confMat);
    fprintf('recognition rate=%.4f\n', recogRate);
    
    RunEarthMoverDistance1.showAppearModelHistograms(obj, appearModel,[],histSize);
%     RunEarthMoverDistance1.showAppearModelHistograms(obj.v.appearModel, obj.v.testHistMats, histSize);
%     RunEarthMoverDistance1.testAppearModelOnOtherShapes(obj.v.appearModel, obj.v.testHistMats);
end

function testColorAppearModelHistBackProjection(obj, debug)
    appearDir = '../dinosaur/appear1';

    targetNames = RunEarthMoverDistance1.getAppearModelTargetNames(appearDir);
    
    for groupNameInd = 1:length(targetNames)
        groupName = targetNames{groupNameInd};
        fprintf('processing group=%s\n', groupName);

        files = dir(fullfile(appearDir, groupName, '*.png'));
        for fileInd=1:length(files)
            file = files(fileInd);
            filePath = fullfile(appearDir, groupName, file.name);
            fprintf('shape pixels=%s ', filePath);
            
            image = imread(filePath);
            imshow(image);
            
            hsv1 = cv.cvtColor(image, cv.COLOR_RGB2HSV);
            imshow(hsv1(:,:,1))
            imshow(hsv1(:,:,2))
            imshow(hsv1(:,:,3))
            imshow(hsv1(:,:,4))
            cv.absdiff(1,-5)
            cv.absdiff([1 2], [-5 -6])
            
            pixs = reshape(image, [], 3);

            %hist = {linspace(0,180,30+1),linspace(0,256,32+1)};
            rEdges = 0:255;
            %edges = {rEdges,rEdges,rEdges};
            channels=int32([0 1]);
            histSize=int32([32 32]);
            ranges=single([0 255; 0 255]);
            
            im = imread('cameraman.tif');
            imshow(im)
            im_noise = imnoise(im, 'gaussian', 0, 0.005);
            imshow(im_noise)
            im_denoise = cv.fastNlMeansDenoising(im, single(18)); % TODO: doesn't work from Matlab
            imshow(im_denoise)
            
            %
            
            image = imread('coloredChips.png');
            cv.imshow('main', image)
            hist1 = cv.calcHist({image}, [0 1], [], [180, 256], [0 180 0 256]);
            hist1 = cv.calcHist( image , [0 1], [], [180, 256], [0 180 0 256]);
            hist1 = cv.calcHist([image], channels, [], histSize, ranges);
            hist1 = cv.calcHist({image}, [1], [], [32], [1 255]);
            hist1 = cv.calcHist({image}, {1}, [], {32}, {1 255});
            hist1 = cv.calcHist({image}, {channels}, [], {histSize}, {ranges});
            
            hist1Int32=int32(hist1);
            bp1=cv.calcBackProject(image, hist1, edges);
        end
    end
end

function testColorAppearModelMixtureOfGaussians(obj, debug)
    RunEarthMoverDistance1.prepareTrainTestData(obj, debug);
    
    labelNames = obj.v.labelNames;
    classCount = length(labelNames);
    
    for mixCnt=16
    % header of log output
    fprintf('%d', mixCnt); % no newline
        
    mixCount = mixCnt;
    covMatType= 'Spherical';
    %covMatType= 'Diagonal';
    %fprintf('mixCount=%d covMatType=%s\n', mixCount, covMatType);
    
    % RGB
    rgb2FeaturesFun = @(pixsRgb) pixsRgb;
    % Luv
    %rgb2FeaturesFun = @(pixsRgb) RunEarthMoverDistance1.pixelsRgb2Luv(pixsRgb);
    %features2ColorFun = @(feats) % TODO: convert Luv to Rgb?
    
    % train mixture of Gaussians
    if isfield(obj.v, 'groupMixGaussList')
        groupMixGaussList = obj.v.groupMixGaussList;
    else
        tic;
        %groupMixGaussList = RunEarthMoverDistance1.trainMixtureGaussians(classCount, obj.v.data, obj.v.trainMask, mixCount, covMatType, rgb2FeaturesFun, debug);
        groupMixGaussList = RunEarthMoverDistance1.trainMixtureGaussiansBatchMode(classCount, obj.v.data, obj.v.trainMask, mixCount, covMatType, rgb2FeaturesFun, debug);        
        learnTime = toc;

        if debug
            fprintf('mixture of Gaussians trained in %.2f sec\n', learnTime);
        end
        obj.v.groupMixGaussList = groupMixGaussList;
        
        if debug
            for classInd=1:classCount
                figure(classInd);
                clf;
                mix = groupMixGaussList{classInd};
                utils.PixelClassifier.drawMixtureGaussiansEach(mix.Means,mix.Covs, mix.Weights);
                title(num2str(classInd));
            end
        end
    end

    % test classifier
    if true
        mixGaussFun = @(pixsFore) RunEarthMoverDistance1.classifyMixGaussian(groupMixGaussList, pixsFore, 1:classCount);
        %mixGaussFakeFun = @(pixsFore, clusts) RunEarthMoverDistance1.classifyMixGaussian(groupMixGaussList, pixsFore, clusts);
        tic;
        confusMat = RunEarthMoverDistance1.simulateClassifier(classCount, obj.v.data, ~obj.v.trainMask, mixGaussFun, rgb2FeaturesFun, debug);
        %confusMat = RunEarthMoverDistance1.simulateClassifierMixGaussisansClassifier(classCount, obj.v.data, ~obj.v.trainMask, obj.v.groupMixGaussList, mixGaussFakeFun, debug);
        testTime = toc;
        %display(confusMat);
        obj.v.confusMat = confusMat;

        if debug
            RunEarthMoverDistance1.printConfusionMatrix(obj.v.confusMat);
        end

        recogRate = utils.PixelClassifier.recognitionRate(confusMat);
        %fprintf(' %.2f %.4f %.2f %.2f;\n', obj.v.trainRatio, recogRate, learnTime, testTime);
        fprintf(' %.4f\n', recogRate);
    end
    
    %RunEarthMoverDistance1.showImageUsingSignatureColors(obj.v.data, ~obj.v.trainMask, obj.v.groupMixGaussList, rgb2FeaturesFun);
    
%     utils.PixelClassifier.drawMixtureGaussiansEach(obj.v.groupMixGaussList{6}.Means,obj.v.groupMixGaussList{6}.Covs, obj.v.groupMixGaussList{6}.Weights);
%     hold on;
%     utils.PixelClassifier.drawMixtureGaussiansEach(obj.v.groupMixGaussList{7}.Means,obj.v.groupMixGaussList{7}.Covs, obj.v.groupMixGaussList{7}.Weights);
%     hold off;
    end
end

% groupsDir = directory which contains folder per each target
% Each such folder contains images of shapes of single target.
function appearModel = learnColorHistFromData(obj, data, labelNames, trainMask, bgPixel, cellSizePixels, histInclMin, channels, histSize, debug)
    classCount = length(labelNames);
    
    % map between shape name and its histogram (Nx4), N=number of pixels
    % 4elements are = [occur R G B]
    classHistList = cell(1, classCount);
    for itemInd=find(trainMask)
        item = data(itemInd);
        
        if isempty(classHistList{item.Label})
            groupRgbHist = containers.Map('KeyType','uint32','ValueType','uint32');
            classHistList{item.Label} = groupRgbHist;
        else
            groupRgbHist = classHistList{item.Label};
        end
        
        tic;
        RunEarthMoverDistance1.accumulateColorHistogramFromImageFile(groupRgbHist, item.FilePath, bgPixel, cellSizePixels, channels, histSize);
        fprintf('learnColorHistFromData: train case %d took %.2f sec\n', itemInd, toc);
    end
    
    % populate result
    appearModel = containers.Map('KeyType','char','ValueType','any');
    for i=1:classCount
        groupRgbHist = classHistList{i};
        
        RunEarthMoverDistance1.trimHistogram(groupRgbHist, histInclMin);
        
        %groupRgbHistMat = RunEarthMoverDistance1.rgbHistogramToArray(groupRgbHist);
        groupRgbHistMat = RunEarthMoverDistance1.twoComponentsHistogramToArray(groupRgbHist);

        RunEarthMoverDistance1.showHistInRows(groupRgbHistMat, histSize);
        h1=RunEarthMoverDistance1.histInRowsTo2D(groupRgbHistMat, histSize);
        contour(h1);
        
        groupName = labelNames{i};
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

function accumulateColorHistogramFromImageFile(groupRgbHist, imageFilePath, transpPixel, cellSizePixels, channels, histSize)
    img = imread(imageFilePath);
    
    %convert to lab
%     labTransformation = makecform('srgb2lab');
%     labI = applycform(img,labTransformation);
%     figure(1), imshow(labI(:,:,1))
%     figure(2), imshow(labI(:,:,2))
%     figure(3), imshow(labI(:,:,3))

    % luv
%     imgLuv2 = rgbConvert(img, 'luv');
%     figure(1), imshow(imgLuv2(:,:,1))
%     figure(2), imshow(imgLuv2(:,:,2))
%     figure(3), imshow(imgLuv2(:,:,3))


    % hsv
    %imgHsv = rgb2hsv(img);
    pixels = reshape(img, [], 3);

    % remove black pixels
    
    %pixels = utils.PW.matRemoveIf(pixels, @(rgb) all(rgb == transpPixel));
    transpPixelMat = repmat(transpPixel, size(pixels,1), 1);
    rowIndsToRem = sum(pixels == transpPixelMat, 2) == 3;
    pixels(rowIndsToRem, :) = [];
    
    RunEarthMoverDistance1.accumulateColorHistogramFromPixels(groupRgbHist, pixels, channels, histSize);
end
    
function accumulateColorHistogramFromPixels(groupRgbHist, pixels, channels, histSize)
    % luv1
    pixellsStack = reshape(pixels, [], 1, 3); % each pixels is in OZ axis
    luvStack = rgbConvert(pixellsStack, 'luv'); % requires piotr_toolbox
    pixels = reshape(luvStack, [], 3);
    
    % select pixels in range of interest
    %ranges = [0 0.37 0 1 0 0.89]; % Luv
    ranges =  [0 0.37 0.13 0.6 0.3 0.78]; % Luv, truncated
   
    limitForChan = @(chan, offset) ranges(2*(chan-1)+1+offset);
    chanX = channels(1);
    chanY = channels(2);
    limX1 = limitForChan(chanX,0);
    limX2 = limitForChan(chanX,1);
    limY1 = limitForChan(chanY,0);
    limY2 = limitForChan(chanY,1);
    histCellSize = [limX2-limX1, limY2-limY1] ./ histSize;

    remMask = pixels(:,chanX) < limX1 | pixels(:,chanX) > limX2 |...
              pixels(:,chanY) < limY1 | pixels(:,chanY) > limY2;
    toRemove = sum(remMask);
    pixels(remMask,:) = [];
    %fprintf('toRemove=%d remains=%d\n', toRemove, length(pixels));
        
    % L*u*v*
    cell1 = int32(floor((pixels(:,chanX) - limX1) / histCellSize(1)) + 1);
    cell2 = int32(floor((pixels(:,chanY) - limY1) / histCellSize(2)) + 1);
    

    % hsv
%     cell1 = floor(pixels(:,1) * histCellSize(1)) + 1; % H
%     cell2 = floor(pixels(:,2) * histCellSize(2)) + 1; % S
    
    % uint32 contains: cell1 and cell2 hist coords
    cells32 = RunEarthMoverDistance1.packTwo16Bit(cell1, cell2);
    [cell1Re, cell2Re] = RunEarthMoverDistance1.unpackTwo16Bit(cells32);
    assert(sum(cell1Re~=cell1) == 0);
    assert(sum(cell2Re~=cell2) == 0);

    one32 = cast(1, 'uint32');
    arrayfun(@(cell32) RunEarthMoverDistance1.containersMapIncValue(groupRgbHist, cell32, one32), cells32);
end

%function [histCellX, histCellY] = pixelsTo

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

function [data, labelNames] = loadColorAppearanceData(obj, debug)
    appearDir = '../dinosaur/appear1';
    labelNames = RunEarthMoverDistance1.getAppearModelTargetNames(appearDir);

    data = struct('FilePath', [], 'Label', []);
    data(1) = [];
    
    i = 1;
    for labelNameInd = 1:length(labelNames)
        labelName = labelNames{labelNameInd};
        fprintf('processing group=%s\n', labelName);

        files = dir(fullfile(appearDir, labelName, '*.png'));
        for fileInd=1:length(files)
            file = files(fileInd);
            filePath = fullfile(appearDir, labelName, file.name);
            fprintf('shape pixels=%s\n', filePath);
            
            data(i).FilePath = filePath;
            data(i).Label = labelNameInd;
            i = i + 1;
        end
    end
end

function targetNames = getAppearModelTargetNames(appearDir)
    appearDirItems = dir(appearDir);
    
    % find name of targets; they are encoded as directory name
    targetNames = {};
    for i=1:length(appearDirItems)
        d = appearDirItems(i);
        
        if d.isdir && (strcmp(d.name,'.') ~= 1 && strcmp(d.name,'..') ~= 1)
            targetNames{end+1} = d.name;
        end
    end
end

% split data into train/test parts
function trainMask = splitTrainTestCases(labels, trainProportion)
    uniqueLabels = unique(labels);
    
    trainMask = false(1,length(labels));
    for lab=uniqueLabels
        sampleInds = find(labels == lab);
        
        sampleSize = length(sampleInds);        
        trainSize = ceil(sampleSize * trainProportion);
        trainIndInds = randperm(sampleSize, trainSize);
        
        trainInds = sampleInds(trainIndInds);
        
        trainMask(trainInds) = true;
    end
end

function postLabel = classifyColorHistEarthMoverDistance(appearModel, forePixs, channels, histSize)
    testHist = containers.Map('KeyType','uint32','ValueType','uint32');
    RunEarthMoverDistance1.accumulateColorHistogramFromPixels(testHist, forePixs, channels, histSize);

    %RunEarthMoverDistance1.trimHistogram(groupRgbHist, histInclMin);
    %groupRgbHistMat = RunEarthMoverDistance1.rgbHistogramToArray(groupRgbHist);
    testHistMat = RunEarthMoverDistance1.twoComponentsHistogramToArray(testHist); % NORM hist
    
    testHistMat=RunEarthMoverDistance1.normailizeRowHist(testHistMat, 1);
    
    classCount = length(appearModel);
    appearModelKeys = keys(appearModel);
    
    dists = zeros(1, classCount);
    for classInd=1:classCount
        groupName = appearModelKeys{classInd};
        groupHistMat = appearModel(groupName);

        % normalize histograms
        groupHistMat=RunEarthMoverDistance1.normailizeRowHist(groupHistMat, 1);
        
        tic;
        dist=cv.EMD(testHistMat, groupHistMat);

        dists(classInd) = dist;
    end
    
    % choose class with minimal distance between histograms
    [val, postLabel] = min(dists);
end

function groupMixGaussList = trainMixtureGaussians(classCount, data, trainMask, nClusters, covMatType, rgb2FeaturesFun, debug)
    groupMixGaussList = cell(1,classCount);
    for itemInd=find(trainMask)
        item = data(itemInd);
        itemLabel = item.Label;
        if debug
            fprintf('C=%d file=%s\n', itemLabel, item.FilePath);
        end

        maxIters=1000;
        if isempty(groupMixGaussList{itemLabel})
            groupMixGauss = cv.EM('Nclusters', nClusters, 'CovMatType', covMatType, 'MaxIters', maxIters); % mexopencv
            groupMixGaussList{itemLabel} = groupMixGauss;
        else
            groupMixGauss = groupMixGaussList{itemLabel};
        end

        %
        image = imread(item.FilePath);
        imshow(image);

        pixs = reshape(image, [], 3);
        pixs(sum(pixs,2) == 0, :) = []; % remove black pixels

        % convert RGB to Luv or other domain
        features = rgb2FeaturesFun(pixs);

        groupMixGauss.train(features);
    end
end

% Train in a batch mode. Accumulate all available pixels for each cluster and 
% only train after this step.
function [groupMixGaussList, trainPixsCount] = trainMixtureGaussiansBatchMode(classCount, data, trainMask, nClusters, covMatType, rgb2FeaturesFun, debug)
    % accumulate features
    featsPerCluster = cell(1,classCount);
    for itemInd=find(trainMask)
        item = data(itemInd);
        itemLabel = item.Label;
        if debug
            fprintf('C=%d file=%s\n', itemLabel, item.FilePath);
        end

        %
        image = imread(item.FilePath);
        if debug
            imshow(image);
        end

        pixs = reshape(image, [], 3);
        pixs(sum(pixs,2) == 0, :) = []; % remove black pixels

        % convert RGB to Luv or other domain
        features = rgb2FeaturesFun(pixs);
        
        if isempty(featsPerCluster{itemLabel})
            numFeat = size(features, 2);
            accumFeats = zeros(0, numFeat);
        else
            accumFeats = featsPerCluster{itemLabel};
        end
        
        featsPerCluster{itemLabel} = [accumFeats; features];
    end
    
    % train
    groupMixGaussList = cell(1,classCount);
    trainPixsCount = zeros(1,classCount);
    for clustInd=1:classCount
        maxIters=1000;
        groupMixGauss = cv.EM('Nclusters', nClusters, 'CovMatType', covMatType, 'MaxIters', maxIters); % mexopencv
        
        features = featsPerCluster{clustInd};
        if ~isempty(features) % allow non trained GMM
            groupMixGauss.train(features);
            groupMixGaussList{clustInd} = groupMixGauss;
            trainPixsCount(clustInd) = size(features, 1);
        end
    end
end

function postLabel = classifyMixGaussian(groupMixGaussList, pixsFore, clustInds)
    classCount = length(groupMixGaussList);
    dists = zeros(1, classCount);
    %for classInd=1:classCount
    for classInd=clustInds
        groupMixGauss = groupMixGaussList{classInd};
        logProbs=groupMixGauss.predict(pixsFore);
        probs = exp(logProbs);

        %probs2=utils.PixelClassifier.evalMixtureGaussians(pixsFore, groupMixGauss.Means, groupMixGauss.Covs, groupMixGauss.Weights);
        %logProbs3=utils.PixelClassifier.logMixtureGaussians(pixsFore, groupMixGauss.Means, groupMixGauss.Covs, groupMixGauss.Weights);
        
        dist1 = mean(probs);
        %dist1 = sum(probs);
        dists(classInd) = dist1;

        % visualize
%             imgGrayCol = zeros(size(pixs,1), 1);
%             imgGrayCol(foreInds,1) = exp(out1);
%             imgGray = reshape(imgGrayCol, size(image,1), []);
%             subplot(testsCount, groupsCount, (testInd-1)*groupsCount+classInd);
%             imshow(imgGray, []);
    end

    [val,ind] = max(dists);
    postLabel = ind;
end

function confusMat = simulateClassifier(classCount, data, testMask, classifLabelFun, rgb2FeaturesFun, debug)
    confusMat = zeros(classCount,classCount, 'int32');
    for itemInd=find(testMask)
        item = data(itemInd);
        if debug
            fprintf('C=%d file=%s\n', item.Label, item.FilePath);
        end
        
        image = imread(item.FilePath);

        pixs = reshape(image, [], 3);
        foreInds = sum(pixs,2) > 0;
        pixsFore = pixs(foreInds, :); % remove black pixels
        
        % RGB to Luv or other domain
        features = rgb2FeaturesFun(pixsFore);
        
        % classify pixels in one of the classes
        postLabel = classifLabelFun(features);

        confusMat(item.Label, postLabel) = confusMat(item.Label, postLabel) + 1;
    end
end

% attempt to change how Mixture of Gaussians works
function confusMat = simulateClassifierMixGaussisansClassifier(classCount, data, testMask, groupMixGaussList, classifLabelFakeFun, debug)
    confusMat = zeros(classCount,classCount, 'int32');
    for itemInd=find(testMask)
        item = data(itemInd);
        if debug
            fprintf('C=%d file=%s\n', item.Label, item.FilePath);
        end
        
        image = imread(item.FilePath);

        pixs = reshape(image, [], 3);
        foreInds = sum(pixs,2) > 0;
        pixsFore = pixs(foreInds, :); % remove black pixels
        
        % classify pixels in one of the classes
        maxNeighboursCount=3;
        clustInds = setdiff((1:classCount)', item.Label);
        
        inds=randperm(length(clustInds),maxNeighboursCount-1);
        clustInds = clustInds(inds);
        clustInds = [clustInds; item.Label];
        clustInds = sortrows(clustInds);
        postLabel = classifLabelFakeFun(pixsFore, clustInds');
        
        if false && item.Label==6 && item.Label ~= postLabel
            probFore6 = groupMixGaussList{6}.predict(pixsFore);
            probFore6 = exp(probFore6);
            probs6 = zeros(1, size(image,1) * size(image,2));
            probs6(foreInds) = probFore6;
            imgResp6 = reshape(probs6, size(image,1), size(image,2));
            imgResp6max = ordfilt2(imgResp6, 16, ones(4,4));

            probFore8 = groupMixGaussList{8}.predict(pixsFore);
            probFore8 = exp(probFore8);
            probs8 = zeros(1, size(image,1) * size(image,2));
            probs8(foreInds) = probFore8;
            imgResp8 = reshape(probs8, size(image,1), size(image,2));
            imgResp8max = ordfilt2(imgResp8, 16, ones(4,4));
            
            figure(16), imshow(imgResp6,[]);
            figure(26), imshow(imgResp6max,[]);
            figure(18), imshow(imgResp8,[]);
            figure(28), imshow(imgResp8max,[]);
        end

        confusMat(item.Label, postLabel) = confusMat(item.Label, postLabel) + 1;
    end
end

function testAppearModelRecognitionVersusNumberOfPixelsForTraining(obj, debug)
    RunEarthMoverDistance1.prepareTrainTestData(obj, debug);
    
    labelNames = obj.v.labelNames;
    classCount = length(labelNames);
    
    histList = {};
    mixCount = 1;
    covMatType = 'Spherical';
    %figure
    clf;
    for classToScrutinize = 1:classCount
    fprintf('classToScrutinize=%d\n', classToScrutinize);
    
    data = obj.v.data;
    inds = find([data.Label] == classToScrutinize);
    %rng(100); % sync generator
    inds = inds(randperm(length(inds)));
    
    % select first item as a target to compare
    targetItem = data(inds(1));
    targetImg = imread(targetItem.FilePath);
    targetPixs = reshape(targetImg, [], 3);
    targetPixs(sum(targetPixs,2) == 0,:) = []; % remove blacks
    
    % learn GMM appearance model on increasing subset of data and look how recognition improves
    hist = [];
    for lastItemToTrainInd = 2:length(inds)-1
        trainItemsCount = lastItemToTrainInd - 1;
        fprintf('trainItemsCount = %d\n', trainItemsCount);
        
        subData = data(inds(2:lastItemToTrainInd));
        
        rgb2FeaturesFun = @(rgb) rgb;
        [groupMixGaussList,trainPixsCountList] = RunEarthMoverDistance1.trainMixtureGaussiansBatchMode(classCount, subData, true(1,trainItemsCount), mixCount, covMatType, rgb2FeaturesFun, false);
        mixGauss = groupMixGaussList{classToScrutinize};
        trainPixsCount = trainPixsCountList(classToScrutinize);
        
        logProbs = mixGauss.predict(targetPixs);
        probs = exp(logProbs);
        avgProb = mean(probs);
        probsMy = utils.PixelClassifier.evalMixtureGaussians(targetPixs, mixGauss.Means, mixGauss.Covs, mixGauss.Weights);
        logProbsMy = utils.PixelClassifier.logMixtureGaussians(targetPixs, mixGauss.Means, mixGauss.Covs, mixGauss.Weights);
        hist = [hist; trainItemsCount trainPixsCount avgProb];
    end
    
    histList{end+1} = hist;
    
    for i=1:length(histList)
        hist = histList{i};
        hold on;
        plot(hist(:,2), hist(:,3));
        hold off;
    end
    drawnow;
    
    end
    

    obj.v.histList = histList;
    title('recog VS train items count');
end

function analyzeHistogramDistributions(obj, debug)
    appearModelMap = obj.v.appearModel.values;
    hist1 = appearModelMap{1};
    hist(hist1(:,1),1:10)
    prctile(hist1(:,1), 90)
    sum(hist1(:,1)<=3)/sum(hist1(:,1))
end

function showHistInRows(histMat, histSize)
    h1 = RunEarthMoverDistance1.histInRowsTo2D(histMat, histSize);
    bar3(h1)
end

function showAppearModelHistograms(obj, appearModel, testHistMats, histSize)
    groupsCount = length(appearModel);
    testsCount = length(testHistMats);
    
    appearModelKeys = keys(appearModel);
    
    % process matrix upper triangle
    hists = cell(1, groupsCount);
    for row=1:groupsCount
        groupName = appearModelKeys{row}
        groupHist = appearModel(groupName);
        
        max(groupHist)
        
        groupHist = RunEarthMoverDistance1.normailizeRowHist(groupHist, 1);
        
        groupHist2D = RunEarthMoverDistance1.histInRowsTo2D(groupHist, histSize);
        %figure(row)
        hists{row} = groupHist2D;
        
%         subplot(1,2,1);
%         contour(groupHist2D);
%         xlabel('H'); ylabel('S');
        
%         subplot(1,2,2);
%         bar3(groupHist2D);
%         xlabel('H'); ylabel('S');
    end
    return;
    figure;
    for col=1:testsCount
        testHist = testHistMats{col};
        
        max(testHist)
        
        testHist = RunEarthMoverDistance1.normailizeRowHist(testHist, 1);

        hist2D = RunEarthMoverDistance1.histInRowsTo2D(testHist, histSize);
        
        
        %subplot(1,testsCount*2,(col-1)*2+1);
        subplot(1,testsCount,col);
        contour(hist2D);
        xlabel('H'); ylabel('S');
        
%         subplot(1,testsCount*2,(col-1)*2+2);
%         bar3(hist2D);
%         xlabel('H'); ylabel('S');
        
        title(sprintf('test %d', col));
    end
end

% For each pixel find most probable cluster and draw pixel with the color
% of cluster mean.
function [imgColorSig, uniqueColorsCount] = imageUsingColorSignature(image, means, covMatList, weights, rgb2FeaturesFun)
    pixs = reshape(image, [], 3);
    
    foreMask = sum(pixs,2)>0;
    forePixs = pixs(foreMask,:); % remove black color

    feats = rgb2FeaturesFun(forePixs);
    featsDbl = double(feats);
    featCount = size(feats, 1);

    classCount = length(weights);
    mostProb = -Inf(featCount,1);
    classLabel = ones(featCount,1); % every pixel in 1st class
    
    for classInd=1:classCount
        mu = means(classInd,:);
        S = covMatList{classInd};
        prob = mvnpdf(featsDbl, mu, S);
        prob(:) = prob(:) .* weights(classInd);
        
        closerInds = prob > mostProb;
        mostProb(closerInds) = prob(closerInds);
        classLabel(closerInds) = classInd;
    end
    
    uniqueColorsCount = length(unique(classLabel));
    
    % replace class label with mean of the corresponding cluster
    sigPixs = means(classLabel,:);
    
    imgColorSig = pixs;
    imgColorSig(foreMask,:) = sigPixs;
    
    imgColorSig = reshape(imgColorSig, size(image,1), size(image,2), 3);
    imgColorSig = uint8(imgColorSig);
end

function showImageUsingSignatureColors(data, imagesToShowMask, mixGaussList, rgb2FeaturesFun)
    classCount = length(mixGaussList);
    for i=find(imagesToShowMask)
        item = data(i);

        % predict label
        pixsFore = imread(item.FilePath);
        pixsFore = reshape(pixsFore, [], 3);
        pixsFore(sum(pixsFore,2)==0, :) = [];
        
        postLabel = RunEarthMoverDistance1.classifyMixGaussian(mixGaussList, pixsFore);
        
        % draw how each component of the mixture of Gaussians cluster image colors

        for classInd=1:classCount
            image = imread(item.FilePath);
            mix = mixGaussList{classInd};
            [imageSig,unqCol] = RunEarthMoverDistance1.imageUsingColorSignature(image, mix.Means, mix.Covs, mix.Weights, rgb2FeaturesFun);
            fprintf('unqCol=%d\n',unqCol);

            subplot(3,4,classInd);
            imshow(imageSig);
            title(num2str(classInd));
        end
        
        subplot(3,4,10);
        imshow(image);
        title(sprintf('original %d', i));
        
        if postLabel ~= item.Label
            fprintf('misclassification\n');
        end
    end
end

function printConfusionMatrix(confusMat)
    classCount = size(confusMat, 1);
    
    [~,inds] = max(confusMat, [], 2);
    succ = (inds == (1:classCount)');
    display([(1:classCount)' succ confusMat]);
end

function rowHistN = normailizeRowHist(rowHist, scalar)
    rowHistN = rowHist;
    rowHistN(:,1) = rowHistN(:,1) / sum(rowHistN(:,1)) * scalar;
end

function histMat2D = histInRowsTo2D(histRows, histSize)
    histMat2D=zeros(histSize(1), histSize(2));
    for i=1:size(histRows,1)
        histMat2D(histRows(i,2), histRows(i,3)) = histRows(i,1);
    end
end

function histBarRecordMat = twoComponentsHistogramToArray(groupHist)
    [cell1,cell2] = RunEarthMoverDistance1.unpackTwo16Bit(cell2mat(groupHist.keys)');

    occur = cell2mat(groupHist.values)';
    
    % Occur R G B
    histBarRecordMat = single([occur cell1 cell2]);
end

function int32 = packTwo16Bit(int16Hi, int16Lo)
    cells32 = cast(int16Hi, 'uint32');
    cells32 = bitor(bitshift(cells32, 16), cast(int16Lo, 'uint32'));

    int32 = cells32;
end

function pixsLuv = pixelsRgb2Luv(pixsRgb)
    pixsRgbPre = reshape(pixsRgb, [], 1, 3);
    pixsLuv = rgbConvert(pixsRgbPre,'luv');
    pixsLuv = reshape(pixsLuv, [], 3);
    
    % TODO: why normalizing decreases recognition performance (0.5 -> 0.3)
    % normalize each channel to [0..1] range
    % ranges = [0 0.37 0 1 0 0.89]; % Luv
%     pixsLuv(:,1) = pixsLuv(:,1) * (1/0.37);
%     pixsLuv(:,3) = pixsLuv(:,3) * (1/0.89);
end

function [int16Hi,int16Lo] = unpackTwo16Bit(int32)
    % 65535        % FFFF
    int16Lo = bitand(int32, 65535);

    int16Hi = bitsra(int32, 16);
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