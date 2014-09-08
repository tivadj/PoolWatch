#include "KalmanFilterMovementPredictor.h"
#include "algos1.h"

bool normalizedDistance(const cv::Matx21f& pos, const cv::Matx21f& mu, const cv::Matx22f& sigma, float& dist)
{
	const int InvMethod = cv::DECOMP_LU;

	auto zd = pos - mu;
	
	bool invOp = false;
	auto sigmaInv = sigma.inv(InvMethod, &invOp);
	if (!invOp)
		return false;

	cv::Matx<float,1,1> mahalanobisDistance = zd.t() * sigmaInv * zd;
	auto md = mahalanobisDistance.val[0];

	double determinant = cv::determinant(sigma);

	dist = static_cast<float>(md + log(determinant));
	return true;
}

/// Computes distance between predicted position of the Kalman Filter and given observed position.
/// It corresponds to Matlab's vision.KalmanFilter.distance() function.
/// Parameter observedPos is a column vector.
// http://www.mathworks.com/help/vision/ref/vision.kalmanfilter.distance.html
bool kalmanFilterDistance(const cv::KalmanFilter& kalmanFilter, const cv::Matx21f& observedPos, float& dist)
{
	cv::Mat residualCovariance = kalmanFilter.measurementMatrix * kalmanFilter.errorCovPost * kalmanFilter.measurementMatrix.t() + kalmanFilter.measurementNoiseCov;
	//auto rc = residualCovariance.at<float>(0, 0);

	cv::Mat mu = kalmanFilter.measurementMatrix * kalmanFilter.statePre;
	//auto mm = mu.at<float>(0, 0);

	bool distOp = normalizedDistance(observedPos, mu, residualCovariance, dist);
	if (!distOp)
		return false;

	return true;
}

// shift = distance from predicted to observed points
float estimateProbOfShift(float shift, float sigma)
{
	assert(shift >= 0 && "Distance must be non negative");
	
	float result = PoolWatch::normalProb(shift, 0, sigma);
	return  result * 2; // since shift is positive we consider only the positive part of a gauss curve
}


const float probDetection = 0.9f;
const float spatDensClutter = 0.0026f;
const float spatDensNew = 4e-6;

KalmanFilterMovementPredictor::KalmanFilterMovementPredictor(float maxShiftPerFrame)
{
	// 2.3m / s is max speed for swimmers
	// let say 0.5m / s is an average speed
	// estimate sigma as one third of difference between max and mean shift per frame
	maxShiftPerFrameM_ = maxShiftPerFrame;

	initKalmanFilter(kalmanFilter_);

	// penalty for missed observation
	// prob 0.4 - penalty - 0.9163
	// prob 0.6 - penalty - 0.5108
	const float penalty = log(1 - probDetection);
	penalty_ = penalty;
	//penalty_ = -1.79 / 200;
}

void KalmanFilterMovementPredictor::initKalmanFilter(cv::KalmanFilter& kalmanFilter)
{
	kalmanFilter.init(KalmanFilterDynamicParamsCount, 2, 0);

	kalmanFilter.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);

	//float meanShiftM = 0.5f / fps;
	//float meanShiftM = 23.0f / fps;
	//float sigma = (maxShiftM - meanShiftM) / 3;
	//float sigma = (maxShiftM - meanShiftM);
	float sigma = maxShiftPerFrameM_ / 3;
	setIdentity(kalmanFilter.processNoiseCov, cv::Scalar::all(sigma*sigma));

	kalmanFilter.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
	
	// in one frame the hands may be detected, in the next frame - the legs
	const float humanHeight = 2;
	float measS = (humanHeight + maxShiftPerFrameM_) / 3;
	setIdentity(kalmanFilter.measurementNoiseCov, cv::Scalar::all(measS*measS));
}

void KalmanFilterMovementPredictor::initScoreAndState(int frameInd, int observationInd, cv::Point3f const& blobCentrWorld, float& score, TrackHypothesisTreeNode& saveNode)
{
	TrackHypothesisTreeNode& childHyp = saveNode;

	// initial track score
	//int nt = 5; // number of expected targets
	//int fa = 25; // number of expected FA(false alarms)
	//float precision = nt / (float)(nt + fa);
	//float initialScore = -log(precision);

	// TODO: ?
	// if initial score is large, then tracks with missed detections may
	// be evicted from hypothesis tree

	//int timeToLive = 6; // life time of FA till it dies
	//initialScore = abs(timeToLive * penalty_);

	float initialScore = log(spatDensNew / spatDensClutter);

	score = initialScore;

	// save Kalman Filter state

	cv::Matx41f state(blobCentrWorld.x, blobCentrWorld.y, 0, 0); // [X,Y, vx=0, vy=0]'
	childHyp.KalmanFilterState = state;
	childHyp.KalmanFilterStateCovariance = cv::Matx44f::zeros();
	CV_DbgAssert(KalmanFilterDynamicParamsCount == decltype(state)::rows);
}

void KalmanFilterMovementPredictor::estimateAndSave(const TrackHypothesisTreeNode& curNode, const boost::optional<cv::Point3f>& blobCentrWorld, cv::Point3f& estPos, float& deltaMovementScore, TrackHypothesisTreeNode& saveNode, float* pShiftDistOrNull)
{
	const TrackHypothesisTreeNode* pLeaf = &curNode;
	TrackHypothesisTreeNode& childHyp = saveNode;

	// prepare Kalman Filter state
	cv::KalmanFilter& kalmanFilter = kalmanFilter_; // use cached Kalman Filter object

	// each hypothesis node has its own copy of Kalman Filter state, hence copy=true
	kalmanFilter.statePost = cv::Mat(pLeaf->KalmanFilterState, true);
	kalmanFilter.errorCovPost = cv::Mat(pLeaf->KalmanFilterStateCovariance, true);

	//
	cv::Mat predictedPosMat = kalmanFilter.predict();
	cv::Point3f predictedPos = cv::Point3f(predictedPosMat.at<float>(0, 0), predictedPosMat.at<float>(1, 0), CameraProjector::zeroHeight());

	// correct position if blob was detected
	if (blobCentrWorld != nullptr)
	{
		const auto& blobPosW = blobCentrWorld.get();

		cv::Matx21f obsPosWorld2(blobPosW.x, blobPosW.y); // ignore Z
		
		auto obsPosWorld2Mat = cv::Mat_<float>(obsPosWorld2, false);
		auto estPosMat = kalmanFilter.correct(obsPosWorld2Mat);

		estPos = cv::Point3f(estPosMat.at<float>(0, 0), estPosMat.at<float>(1, 0), CameraProjector::zeroHeight());
		//float dist = cv::norm(estPos - predictedPos);
		float dist = cv::norm(predictedPos - blobPosW);

		//
		//float shiftScoreOld = -1;
		//bool distOp = kalmanFilterDistance(kalmanFilter, obsPosWorld2, shiftScoreOld);
		//CV_Assert(distOp);
		
		//float sigma = maxShiftPerFrameM_ / 2; // 2 means that 2sig=95% of values will be at max shift
		float sigma = maxShiftPerFrameM_ / 3; // may not find the distant part of the blob
		float probObs = estimateProbOfShift(dist, sigma);
		auto shiftScore = std::log(probObs * probDetection / (spatDensClutter + spatDensNew));
		//const float minShiftScore = 0.001;
		//if (shiftScore < minShiftScore)
		//	shiftScore = minShiftScore;
		//CV_Assert(shiftScore >= 0);

		// if we limit score from the bottom, there may be jumps across the whole screen
		//float minScore = penalty_ * 1.1; // bound the score from below, otherwise track may be split near noisy blobs and new track continue
		//if (shiftScore < minScore)
		//	shiftScore = minScore;

		deltaMovementScore = shiftScore;

		if (pShiftDistOrNull != nullptr)
			*pShiftDistOrNull = dist;
	}
	else
	{
		// as there is no observation, the predicted position is the best estimate
		estPos = predictedPos;

		deltaMovementScore = penalty_; // punish for no observation
		
		if (pShiftDistOrNull != nullptr)
			*pShiftDistOrNull = -1;
	}

	// save Kalman Filter state
	childHyp.KalmanFilterState = kalmanFilter.statePost.clone();
	childHyp.KalmanFilterStateCovariance = kalmanFilter.errorCovPost.clone();
}

float KalmanFilterMovementPredictor::maxShiftPerFrame() const
{
	return maxShiftPerFrameM_;
}