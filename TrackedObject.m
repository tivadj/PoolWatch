classdef TrackedObject <handle
properties
    TrackCandidateId;
    
    % used to cache predicted pos associated with track
    PredictedPosWorld; % [X,Y,Z]
    
    % represents the most recent position of this target (tracked object)
    % used for logging of each track image positions
    LastEstimatedPosWorld; % [X,Y,Z]
    LastObservationPosExactOrApprox; % [X,Y] in pixels

    KalmanFilter; % used to predict position of track candidate
    v;
    %v.AppearanceGmm; % type: cv.EM, accumulated color signature up to the last frame
    %v.AppearanceGmmTrained; % type: bool, whether appearamance GMM is trained
    %v.AppearancePixels;
end
methods(Static)
    function obj = NewTrackCandidate(trackCandidateId)
        obj = TrackedObject;
        obj.TrackCandidateId = trackCandidateId;
        obj.v.AppearanceGmm = cv.EM('Nclusters', 16, 'CovMatType', 'Spherical');
        obj.v.AppearanceGmmTrained = false;
        obj.v.AppearancePixels = zeros(0,3,'uint8');
    end
end

methods
    
function id = idOrCandidateId(this)
    % make id negative to distinguish from TrackId
    id = -this.TrackCandidateId;
end

function result = appearPixMax(this)
    % number of pixels to keep in buffer
    % param = each shape image is approx 40x40; take 20 frames => 40x40x20=32000
    result = 15000;
end

function result = canAcceptAppearancePixels(this)
    numPixs = size(this.v.AppearancePixels, 1);
    result = numPixs < this.appearPixMax;
end

function pushAppearancePixels(this, pixs)
    % avoid pixels buffer unbounded growth
    if ~this.canAcceptAppearancePixels
        return;
    end

    allPixs = [this.v.AppearancePixels; pixs];
    this.v.AppearancePixels = allPixs;
    this.v.AppearanceGmmTrained = false;
end

function probs = predict(this, testPixs)
    % lazy training because it is an expensive operation
    if ~this.v.AppearanceGmmTrained
        trainPixs = this.v.AppearancePixels;
        numPixs = size(trainPixs, 1);
        
        % there must be enough pixels for training
        nClust = min([16 numPixs]);
        this.v.AppearanceGmm.Nclusters = nClust;
        this.v.AppearanceGmm.train(trainPixs);
        this.v.AppearanceGmmTrained = true;
    end
    logProbs = this.v.AppearanceGmm.predict(testPixs);
    % BUG: OpenCV, sometimes predict method returns positive log probabilities (may be due to overflows)
    probs = zeros(size(logProbs));
    probs(logProbs < 0) = exp(logProbs(logProbs < 0)); % use only logProb < 0

    %probs = utils.PixelClassifier.evalMixtureGaussians(testPixs, this.v.AppearanceGmm.Means, this.v.AppearanceGmm.Covs, this.v.AppearanceGmm.Weights);
    %logProbs2 = log(probs);
    %fprintf('logProb error %d\n', sum((logProbs-logProbs2).^2));
end

function SetVelocity(this, worldVelocity)
    this.KalmanFilter.State(3:4) = worldVelocity;
end
end
end