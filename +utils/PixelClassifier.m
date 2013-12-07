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
    assert(l==l2); 
    
    isSphericalCov = iscell(S);
    if isSphericalCov
        c2 = length(S);
    else
        % diagonal cov
        assert(isnumeric(S));
        [l3,l4,c2]=size(S);
        assert(l==l3 && l==l4); 
    end
    assert(mixCount == c2);

    c3 = length(weights);
    assert(mixCount==c3);

    %
    XDbl = double(X);
    result = zeros(N,1);
    for gaussInd=1:mixCount
        if isSphericalCov
            cov = S{gaussInd};
        else
            cov = S(:,:,gaussInd);
        end
        result = result + weights(gaussInd) * mvnpdf(XDbl, m(gaussInd,:), cov);
    end
end

% Computes logarithm of probability of picking X from GMM with mean m, covariance S and mixture components weights.
function result = logMixtureGaussians(X, m, S, weights)
    [N,l]=size(X);
    
    [mixCount,l2]=size(m);
    assert(l==l2); 
    
    isSphericalCov = iscell(S);
    if isSphericalCov
        c2 = length(S);
    else
        % diagonal cov
        assert(isnumeric(S));
        [l3,l4,c2]=size(S);
        assert(l==l3 && l==l4); 
    end
    assert(mixCount == c2);

    c3 = length(weights);
    assert(mixCount==c3);

    %
    XDbl = double(X);
    
    % compute log-probabilities for each mixture component

    mixLogProbs = zeros(N,mixCount);
    for gaussInd=1:mixCount
        if isSphericalCov
            cov = S{gaussInd};
        else
            cov = S(:,:,gaussInd);
        end
        mixLogProbs(:,gaussInd) = log(weights(gaussInd)) + utils.PixelClassifier.logMvnPdf(XDbl, m(gaussInd,:), cov);
    end
    
    % compute sum of weighted probabilities of all mixture components
    result = utils.PixelClassifier.logSumLogProbs(mixLogProbs);
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

% Classify pixels according to two given mixture of gaussians.
function labels = expectMax(em1, em2, pix)
    labels = utils.PixelClassifier.expectMaxByMatlab(em1, em2, pix);
end

function labels = expectMaxByOpenCV(em1, em2, pix)
    [~,minInd] = max([em1.predict(pix), em2.predict(pix)], [], 2);
    labels = minInd;
end

function labels = expectMaxByMatlab(em1, em2, pix)
    covs1 = cell2mat(em1.Covs);
    covs1 = reshape(covs1,3,3,[]);

    covs2 = cell2mat(em2.Covs);
    covs2 = reshape(covs2,3,3,[]);
    
    lab1 = utils.PixelClassifier.evalMixtureGaussians(pix, em1.Means, covs1, em1.Weights);
    lab2 = utils.PixelClassifier.evalMixtureGaussians(pix, em2.Means, covs2, em2.Weights);

    [~,minInd] = max([lab1 lab2], [], 2);
    labels = minInd;
end

% Computes logarithm of sum of probabilities (given as log-probabilities) in a numerically stable fashion.
% Note, direct computation is:
% result = log(sum(exp(logProbs)))
function result = logSumLogProbs(logProbs)
    assert(all(all(logProbs <= 0)));
    maxLProbCol = max(logProbs, [], 2);
    
    probsCount = size(logProbs, 2);
    
    % repeat mean (in column) for each log prob column
    maxLProbMat = repmat(maxLProbCol, 1, probsCount);
    
    result = maxLProbCol + log(sum(exp(logProbs - maxLProbMat), 2));
end

% evaluates logarithm of multinormal distribution = log(mvnpdf(X,m,S))
% N=number of points
% X[NxDim] = data points
% m[1xDim] = mean
% S[DimxDim] = sigma
function result = logMvnPdf(X, m, S)
    [n,dim] = size(X);

    centrX = X - repmat(m, n, 1);
    detS = det(S);
    invS = inv(S);
    
    gauss = @(x) -0.5 * (dim*log(2*pi) + log(detS) + (x * invS * x'));
    
    result = arrayfun(@(i) gauss(centrX(i,:)), (1:n)');
end

% hitProb=min prob of whether a point belongs to a gaussian.
function surf = drawMixtureGaussians(m, S, weights, colorChar)
    [mixCount,l]=size(m);
    assert(l==3,'trisurf requires 3 dimensions');
    
    % generate set of points according to distribution
    pointsCountMax=1000;
    pointsCount=0;
    X = zeros(pointsCount,l);
    for i=1:mixCount
        pointsCountPerGaussian  = round(pointsCountMax * weights(i));
        
        if pointsCountPerGaussian > 0
            % determine covariance matrix
            if iscell(S) % each cell contains covariance matrix
                S1 = S{i};
            else % cov mat = diagonal with same element
                S1 = S1*eye(l);
            end
            
            X(pointsCount+1:pointsCount+pointsCountPerGaussian,:) = mvnrnd(m(i,:), S1, pointsCountPerGaussian);
            pointsCount = pointsCount + pointsCountPerGaussian;
        end
    end
    
    X(pointsCount+1:end,:)=[];
    
    % find convex hull
    triInd = convhulln(X);
    surf = trisurf(triInd, X(:,1), X(:,2), X(:,3), 'FaceColor', colorChar);
    xlabel('R');
    ylabel('G');
    zlabel('B');
end

function surf = drawMixtureGaussiansEach(m, S, weights)
    [mixCount,l]=size(m);
    assert(l==3,'trisurf requires 3 dimensions');
    
    % generate set of points according to distribution
    pointsCountMax=1000;
    hold on;
    for i=1:mixCount
        pointsCountPerGaussian  = round(pointsCountMax * weights(i));
        
        if pointsCountPerGaussian > 0
            % determine covariance matrix
            if iscell(S) % each cell contains covariance matrix
                S1 = S{i};
            else % cov mat = diagonal with same element
                S1 = S1*eye(l);
            end
            
            X = mvnrnd(m(i,:), S1, pointsCountPerGaussian);
            
            % find convex hull
            triInd = convhulln(X);
            surf = trisurf(triInd, X(:,1), X(:,2), X(:,3), 'FaceColor', m(i,:)/255);
        end
    end
    hold off;
    
    xlabel('R');
    ylabel('G');
    zlabel('B');
end

function recogRate = recognitionRate(confusMat)
    diagSum = sum(diag(confusMat));
    recogRate = diagSum / sum(confusMat(:));
end

end
end