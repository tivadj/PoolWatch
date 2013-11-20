classdef PixelClassifier
methods(Static)
    
% returns mask of pixels which positively classified.
% imageMask = MxNx1
function imageMask = applyToImage(image, pixelClassifFun, debug)
    if ~exist('debug', 'var')
        debug = false;
    end
    
    pixelTriples = reshape(image, [], 3);

    % apply skin classifier to input image
    classifRes=pixelClassifFun(double(pixelTriples));

    if debug
        hist(classifRes);
    end

    % construct mask
    classifThrMask=classifRes > 0.5;

    classifThr = classifRes;
    classifThr( classifThrMask)=1;
    classifThr(~classifThrMask)=0;
    classifThr=im2uint8(classifThr);

    imageMask = reshape(classifThr, size(image,1), size(image,2));
end

function classifFun = getConvexHullClassifier(ballPixelsByRow, convexHullTriInd)
    classifFun = @(pixelsByRow) utils.inhull(pixelsByRow, ballPixelsByRow, convexHullTriInd, 0.2);
end

function filteredImage = applyAndGetImage(image, pixelClassifFun, debug)
    if ~exist('debug', 'var')
        debug = false;
    end

    maskSuccess = utils.PixelClassifier.applyToImage(image, pixelClassifFun);
    filteredImage = utils.applyMask(image, maskSuccess);
end

% evaluate Mixture of Gaussians for given points X.
% X=[Nxl]
% m=[mixCount,l]
% S=[l,l,mixCount]
% weights=[1xmixCount] weights of each gaussian in the mixture
% mixCount=number of mixtures in the pdf
% l=dimension space
% result=value of a function in each point X
function result = evalMixtureGaussians(X, m, S, weights)
    [N,l]=size(X);
    [mixCount,l2]=size(m);
    [l3,l4,c2]=size(S);
    c3 = length(weights);
    
    assert(mixCount==c2 && mixCount==c3);
    assert(l==l2 && l==l3 && l==l4); 

    result = zeros(N,1);
    for gaussInd=1:mixCount
        result = result + weights(gaussInd) * mvnpdf(X, m(gaussInd,:), S(:,:,gaussInd));
    end
end

function result = evalMixtureGaussiansNoLoops(X, m, S, weights)
    [~,l]=size(X);
    [mixCount,l2]=size(m);
    [l3,l4,c2]=size(S);
    c3 = length(weights);
    
    assert(mixCount==c2 && mixCount==c3);
    assert(l==l2 && l==l3 && l==l4); 

    valueCells=arrayfun(@(gaussInd) weights(gaussInd) * mvnpdf(X, m(gaussInd,:), S(:,:,gaussInd)), 1:mixCount, 'UniformOutput', false);
    value=cell2mat(valueCells);
    result=sum(value,2);
end

% mList, SList, weightsList are lists of (m,S and weights) parameters for each mixtures of gaussians.
% priors=Bayes priors for each class
function clazz = bayesClassifierMixtureGaussians(X, mList, SList, weightsList, priors)
    classCount = length(mList);
    [N,~]=size(X);
    
    clazz = ones(N,1,'int32'); % initially each put 1st class
    classStrength = -ones(N,1);
    for classInd=1:classCount
        curStrength = priors(classInd) * utils.PixelClassifier.evalMixtureGaussians(X, mList{classInd}', SList{classInd}, weightsList{classInd});

        % put points in the strongest class
        strongerMask = classStrength < curStrength;
        clazz(strongerMask) = classInd;
        classStrength(strongerMask) = curStrength(strongerMask);
    end
end

% hitProb=min prob of whether a point belongs to a gaussian.
function surf = drawMixtureGaussians(m, S, weights, colorChar)
    [mixCount,l]=size(m);
    assert(l==3,'trisurf requires 3 dimensions');
    
    pointsCountMax=1000;
    pointsCount=0;
    X = zeros(pointsCount,l);
    for i=1:mixCount
        pointsCountPerGaussian  = round(pointsCountMax * weights(i));
        
        if pointsCountPerGaussian > 0
            S1 = S(i)*eye(l);
            X(pointsCount+1:pointsCount+pointsCountPerGaussian,:) = mvnrnd(m(i,:), S1, pointsCountPerGaussian);
            pointsCount = pointsCount + pointsCountPerGaussian;
        end
    end
    
    X(pointsCount+1:end,:)=[];
    
    % find convex hull
    triInd = convhulln(X);
    surf = trisurf(triInd, X(:,1), X(:,2), X(:,3), 'FaceColor', colorChar);
end

end
end