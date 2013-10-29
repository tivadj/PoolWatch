classdef RunSkinClassifier < handle
methods(Static)

function obj = create(debug)
    obj = utils.TypeErasedClass;
end

function run(obj)
    debug = 0;
    
    RunSkinClassifier.testSkinClassifier(obj, debug);
end

function testSkinClassifier(obj, debug)
    if ~isfield(obj.v, 'skinHullClassifierFun')
        obj.v.skinHullClassifierFun = RunSkinClassifier.createSkinClassifier(debug);
    end
    
    videoFilePath = fullfile('output/mvi3177_blueWomanLane3.avi');
    mmReader = mmreader(videoFilePath);
    obj.v.mmReader = mmReader;
    
    I = read(mmReader, 100);
    imshow(I);

    i1 = utils.PixelClassifier.applyAndGetImage(I, obj.v.skinHullClassifierFun, debug);
    imshow(i1);
end

function skinHullClassifierFun = createSkinClassifier(debug)
    % build skin pixels convex hull
    cl2=SkinClassifierStatics.create;
    SkinClassifierStatics.populateSurfPixels(cl2);
    SkinClassifierStatics.prepareTrainingDataMakeNonoverlappingHulls(cl2, debug);
    SkinClassifierStatics.findSkinPixelsConvexHullWithMinError(cl2, debug);
    
    %
    skinHullClassifierFun=utils.PixelClassifier.getConvexHullClassifier(cl2.v.skinHullClassifHullPoints, cl2.v.skinHullClassifHullTriInds);
end

function tinkerRegions(obj)
    I = imread('data/MVI_3177_0127.png');
    imshow(I);

    % skin

    % s1=roipoly(I); % p1 leg
    % s2=roipoly(I); % p1 hand up
    % s3=roipoly(I); % p2
    % s4=roipoly(I); % p2
    % s5=roipoly(I); % p2
    % s6=roipoly(I); % p2
    % s7=roipoly(I); % p3
    % 
    % skinMask=s1 | s2 | s3 | s4 | s5 | s6 | s7;
    %save('data/Mask_Skin1.mat', 'skinMask')
    load('data/Mask_Skin1.mat','skinMask');

    % skinMask=im2uint8(skinMask); % NOTE: not just casting uint8(x)
    % skinMask3=cat(3, skinMask, skinMask, skinMask);
    % skinIm = bitand(I, skinMask3);
    % imshow(skinIm);

    % find average skin color
    % skinImR = skinIm(:,:,1);
    % skinImG = skinIm(:,:,2);
    % skinImB = skinIm(:,:,3);
    % median(skinImR(skinImR > 0)) % skin red
    % median(skinImG(skinImG > 0)) % skin green
    % median(skinImB(skinImB > 0)) % skin blue

    % marker lanes
    % marker1=roipoly(I);
    % markerMask=marker1;
    %save('Mask_Marker1.mat', 'markerMask')
    % load('data/Mask_Marker1.mat','markerMask');

    % sign plate
    % sign1=roipoly(I);
    % signMask=sign1;
    %save('Mask_Sign1.mat', 'signMask')
    % load('Mask_Sign1.mat','signMask');

    % pool boundary
    % pool1=roipoly(I);
    % poolMask=pool1;
    %save('data/Mask_Pool1.mat', 'poolMask')
    load('data/Mask_Pool1.mat','poolMask');

    mint=im2uint8(waterMask);
    mintRgb=cat(3,mint,mint,mint);
    waterMask=bitand(I,mintRgb);
    imshow(waterMask)
    imhist(waterMask(waterMask > 0))
    waterMask1=waterMask(:,:,1);
    waterMask2=waterMask(:,:,2);
    waterMask3=waterMask(:,:,3);
    figure, imhist(waterMask1(waterMask1>0)) %red
    figure, imhist(waterMask2(waterMask2>0)) %green
    figure, imhist(waterMask3(waterMask3>0)) %blue

    % find average water color
    waterRgb=zeros(1,3);
    waterRgb(1)=median(waterMask1(waterMask1>0)); % 83
    waterRgb(2)=median(waterMask2(waterMask2>0)); % 129
    waterRgb(3)=median(waterMask3(waterMask3>0)); % 138

    % image of water color
    av2=reshape(waterRgb, [1,1,3]);
    av2=cast(av2,'uint8');
    avWat=repmat(av2,[size(I,1) size(I,2)]);
    avWat=cast(avWat,'uint8');
    imshow(avWat) % 'average' water color

    % find distances from pixels to 'water' cluster
    Icol=reshape(I,size(I,1)*size(I,2),3);
    avWater = repmat(waterRgb, size(I,1)*size(I,2), 1);
    avWater = cast(avWater, 'uint8');
    distFromWat=sum((Icol - avWater ).^2, 2);
    hist(distFromWat)

    % =no separation Water-Other
    distFromWat2d=sum((I-avWat).^2, 3);
    imshow(distFromWat2d,[]);

    skinRgb=uint8([191 118 84]);
    %sum(([208 114 124]-double(skinRgb)).^2)
    skinIm=repmat(reshape(skinRgb, [1 1 3]), [size(I,1) size(I,2)]);
    distFromSkin = sum((double(I)-double(skinIm)).^2, 3);
    %distFromSkin = prod((double(I)-double(skinIm)).^2, 3).^(1/3);
    imshow(distFromSkin, [])
    %imshow(imcomplement(distFromSkin), [])
    min(distFromSkin(:))
    max(distFromSkin(:))

    % Begin classifier
    % skin-other classifier (perceptron)
    % observation per column (shape for neural network training)
    trainInput = reshape(I, size(I,1)*size(I,2), 3)';
    trainOutput = reshape(skinMask, 1, size(I,1)*size(I,2));

    % take only pixels in the pool
    poolBoundaryVector=reshape(poolMask, 1, size(I,1)*size(I,2));
    trainInput = trainInput(:,poolBoundaryVector);
    trainOutput = trainOutput(poolBoundaryVector);

    % choose two balanced by size sets of skin-other pixels
    % choose unique pixels
    [trainInput,ia,ic] = unique(trainInput','rows');
    trainInput=trainInput';
    trainOutput = trainOutput(ia');
    skinPixelsCount=sum(trainOutput>0);
    nonSkinPixels=trainInput(:,trainOutput==0);
    nonSkinPixelsSampleInds=randperm(length(nonSkinPixels), skinPixelsCount); % sample some non skin pixels
    nonSkinPixelsSample=nonSkinPixels(:,nonSkinPixelsSampleInds);
    trainInputBal = [trainInput(:,trainOutput>0) nonSkinPixelsSample];
    trainInputBal = double(trainInputBal);
    trainOutputBal = [ones(skinPixelsCount,1); zeros(length(nonSkinPixelsSample),1)]';

    % trainInput = [];
    % trainOutput = [];
    % for i=1:size(I,1)
    %     for j=1:size(I,2)
    %         out1 = 0;
    %         if skinMask(i,j)
    %             out1 = 1;
    %         elseif (markerMask(i,j) || cleanWatMask(i,j) || signMask(i,j))
    %             out1 = 0;
    %         end
    %         
    %         if (~isnan(out1))
    %             trainInput = cat(2, trainInput, squeeze(I(i,j,:)));
    %             trainOutput = [trainOutput out1];
    %         end
    %     end
    % end
    % required for perceptron training
    % and trainOutput may be int
    trainInput = double(trainInput);
    %trainOutput = double(trainOutput); % perceptron requires double too

    % visualize training points
    % TODO: how to legend axes?
    pointsSkinMask=trainOutputBal > 0;
    pointsSkin = trainInputBal(:,pointsSkinMask);
    plot3(pointsSkin(1,:), pointsSkin(2,:), pointsSkin(3,:), 'r.')
    pointsOther = trainInputBal(:,~pointsSkinMask);
    hold on
    plot3(pointsOther(1,:), pointsOther(2,:), pointsOther(3,:), 'k.')
    % [x1,x2]=meshgrid(0:255,0:255);
    % Z=(-303*x1+4871*x2+922)/(4673);
    % plot3(x1,x2,Z,'b');
    hold off

    % train perceptron
    net = perceptron;
    net = train(net,trainInputBal,trainOutputBal); % perceptron <- samples in columns
    net([191 118 84]') % skin
    net([208 114 124]') % red marker
    net([187 139 111]') % skin2
    [cell2mat(net.IW) cell2mat(net.b)] %  [247 242 225]*[ -303 4871 -4673]'+922
    %perf = perform(net, trainInput, trainOutput);

    % 2-layer NN
    net=feedforwardnet([3 1]); % trainlm
    %net=feedforwardnet([5],'traingda');
    %n1.trainParam.max_perf_inc = 1.004; % if gradient ratio is greater then learning rate is dicreased
    %net.trainParam.lr_dec = 0.9; % decrease factor
    %view(net);
    [net,tr]=train(net, trainInputBal, trainOutputBal);
    %[net,tr]=train(net, trainInput, trainOutput, 'useGPU','yes');
    %[net,tr]=train(net, trainInput, trainOutput, 'useParallel','yes','showResources','yes');

    % train svm
    opts=statset('Display','iter');
    % observation per row
    %'kernel_function','rbf',
    %'rbf_sigma', 50,
    %'kernel_function','mlp',...
    cl=svmtrain(trainInputBal, trainOutputBal, 'kernel_function','rbf','rbf_sigma', 1,'options', opts); 
    pixelTriples=reshape(I, size(I,1)*size(I,2), 3);
    classifRes = utils.SvmClassifyHelper(cl, double(pixelTriples), 50000); % batchSize
    %classifRes = svmclassify(cl,double(pixelTriples));
    classifResOne = im2uint8(classifRes);
    pixelTriples = bitand(pixelTriples, cat(2, classifResOne, classifResOne, classifResOne)); % clear background
    Iskin2 = reshape(pixelTriples, size(I,1), size(I,2), 3);
    imshow(Iskin2)

    % apply perceptron classification result to input image
    Iskin2=I;
    pixelTriples=reshape(I, size(I,1)*size(I,2), 3)';
    classifRes=net(pixelTriples);
    hist(classifRes);
    classifResOne=classifRes > 0.9; % 0.03
    classifRes( classifResOne)=1;
    classifRes(~classifResOne)=0;
    classifRes=im2uint8(classifRes);
    pixelTriples = bitand(pixelTriples, cat(1, classifRes, classifRes, classifRes)); % clear background
    Iskin2 = reshape(pixelTriples', size(I,1), size(I,2), 3);
    imshow(Iskin2)

    %save('SkinClassifier_nn31.mat', 'net');
    %load('SkinClassifier_nn31.mat', 'net');


    %
    cl2=SkinClassifierStatics.create;
    SkinClassifierStatics.populateSurfPixels(cl2, 2);
    SkinClassifierStatics.visualizePoints(cl2);
    SkinClassifierStatics.prepareTrainingData(cl2);
    %SkinClassifierStatics.trainSurfSkinSvm(cl2);
    SkinClassifierStatics.trainSurfSkinNet(cl2);
    SkinClassifierStatics.visualizePoints(cl2);
    hold on
    SkinClassifierStatics.visualizeSkinConvexHull(cl2,200000)
    hold off
    SkinClassifierStatics.analyzeSurfSkinPoints(cl2);

    SkinClassifierStatics.projectOn2D(cl2);
    SkinClassifierStatics.testOnImage(cl2,fullfile('data/MVI_3177_0127_640x480.png'), 0.1);
    SkinClassifierStatics.testOnImage(cl2,fullfile('data/MVI_3177_0127.png'), 0.1);

    % run skinHullClassifier on an image
    skinHullClassifFun = @(pixByRow) cl2.v.skinHullClassif.isSkinPixel(pixByRow);
    skinImg = SkinClassifierStatics.applySkinClassifierToImage(I, skinHullClassifFun);
    imshow(skinImg)

    videoUpd = SkinClassifierStatics.applySkinClassifierToVideo(fullfile('rawdata/MVI_3177.MOV'), skinHullClassifFun);
    movie1=immovie(videoUpd);
    implay(movie1);
end

end
end