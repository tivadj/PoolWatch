#include "KalmanFilterMovementPredictor.h"

float normalizedDistance(const cv::Mat& pos, const cv::Mat& mu, const cv::Mat& sigma)
{
	cv::Mat zd = pos - mu;
	cv::Mat mahalanobisDistance = zd.t() * sigma.inv() * zd;
	//auto md = mahalanobisDistance.at<float>(0, 0);
	double determinant = cv::determinant(sigma);
	float dist = mahalanobisDistance.at<float>(0, 0) + static_cast<float>(log(determinant));
	return dist;
}

float normalizedDistance(const cv::Point3f& pos, const cv::Point3f& mu, float sigma)
{
	cv::Point3f zd = pos - mu;
	float mahalanobisDistance = zd.dot(zd) * (1/sigma);
	
	double determinant = cv::determinant(sigma);
	float dist = mahalanobisDistance + static_cast<float>(log(determinant));
	return -dist;
}

/// Computes distance between predicted position of the Kalman Filter and given observed position.
/// It corresponds to Matlab's vision.KalmanFilter.distance() function.
/// Parameter observedPos is a column vector.
// http://www.mathworks.com/help/vision/ref/vision.kalmanfilter.distance.html
float kalmanFilterDistance(const cv::KalmanFilter& kalmanFilter, const cv::Mat& observedPos)
{
	cv::Mat residualCovariance = kalmanFilter.measurementMatrix * kalmanFilter.errorCovPost * kalmanFilter.measurementMatrix.t() + kalmanFilter.measurementNoiseCov;
	//auto rc = residualCovariance.at<float>(0, 0);

	cv::Mat mu = kalmanFilter.measurementMatrix * kalmanFilter.statePre;
	//auto mm = mu.at<float>(0, 0);

	float result = normalizedDistance(observedPos, mu, residualCovariance);
	return result;
}


KalmanFilterMovementPredictor::KalmanFilterMovementPredictor(float fps)
{
	initKalmanFilter(kalmanFilter_, fps);

	// penalty for missed observation
	// prob 0.4 - penalty - 0.9163
	// prob 0.6 - penalty - 0.5108
	const float probDetection = 0.95f;
	const float penalty = log(1 - probDetection);
	//penalty_ = penalty;
	penalty_ = -1.79 / 200;
}

void KalmanFilterMovementPredictor::initKalmanFilter(cv::KalmanFilter& kalmanFilter, float fps)
{
	kalmanFilter.init(KalmanFilterDynamicParamsCount, 2, 0);

	kalmanFilter.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);

	// 2.3m / s is max speed for swimmers
	// let say 0.5m / s is an average speed
	// estimate sigma as one third of difference between max and mean shift per frame
	float maxShiftM = 2.3f / fps;
	float meanShiftM = 0.5f / fps;
	float sigma = (maxShiftM - meanShiftM) / 3;
	setIdentity(kalmanFilter.processNoiseCov, cv::Scalar::all(sigma*sigma));

	kalmanFilter.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
	const float measS = 5.0f / 3;
	setIdentity(kalmanFilter.measurementNoiseCov, cv::Scalar::all(measS*measS));
}

KalmanFilterMovementPredictor::~KalmanFilterMovementPredictor()
{
}

void KalmanFilterMovementPredictor::initScoreAndState(int frameInd, int observationInd, cv::Point3f const& blobCentrWorld, float& score, TrackHypothesisTreeNode& saveNode)
{
	TrackHypothesisTreeNode& childHyp = saveNode;

	// initial track score
	int nt = 5; // number of expected targets
	int fa = 25; // number of expected FA(false alarms)
	float precision = nt / (float)(nt + fa);
	float initialScore = -log(precision);

	// if initial score is large, then tracks with missed detections may
	// be evicted from hypothesis tree
	initialScore = abs(6 * penalty_);

	score = initialScore;

	// save Kalman Filter state

	cv::Mat_<float> state(KalmanFilterDynamicParamsCount, 1, 0.0f); // [X,Y, vx=0, vy=0]'
	state(0) = blobCentrWorld.x;
	state(1) = blobCentrWorld.y;
	childHyp.KalmanFilterState = state;
	childHyp.KalmanFilterStateCovariance = cv::Mat_<float>::eye(KalmanFilterDynamicParamsCount, KalmanFilterDynamicParamsCount);
}

void KalmanFilterMovementPredictor::estimateAndSave(const TrackHypothesisTreeNode& curNode, const boost::optional<cv::Point3f>& blobCentrWorld, cv::Point3f& estPos, float& score, TrackHypothesisTreeNode& saveNode)
{
	const TrackHypothesisTreeNode* pLeaf = &curNode;
	TrackHypothesisTreeNode& childHyp = saveNode;

	// prepare Kalman Filter state
	cv::KalmanFilter& kalmanFilter = kalmanFilter_; // use cached Kalman Filter object
	kalmanFilter.statePost = pLeaf->KalmanFilterState.clone();
	kalmanFilter.errorCovPost = pLeaf->KalmanFilterStateCovariance.clone();

	//
	cv::Mat predictedPos2 = kalmanFilter.predict();
	cv::Point3f predictedPos = cv::Point3f(predictedPos2.at<float>(0, 0), predictedPos2.at<float>(1, 0), CameraProjector::zeroHeight());

	// correct position if blob was detected

	if (blobCentrWorld != nullptr)
	{
		const auto& blobPosW = blobCentrWorld.get();

		auto obsPosWorld2 = cv::Mat_<float>(2, 1); // ignore Z
		obsPosWorld2(0) = blobPosW.x;
		obsPosWorld2(1) = blobPosW.y;

		auto estPosMat = kalmanFilter.correct(obsPosWorld2);
		estPos = cv::Point3f(estPosMat.at<float>(0, 0), estPosMat.at<float>(1, 0), CameraProjector::zeroHeight());

		//
		auto shiftScore = kalmanFilterDistance(kalmanFilter, obsPosWorld2);
		score = pLeaf->Score + shiftScore;
	}
	else
	{
		// as there is no observation, the predicted position is the best estimate
		estPos = predictedPos;

		score = pLeaf->Score + penalty_; // punish for no observation
	}
	//childHyp.EstimatedPosWorld = estPos;
	//childHyp.Score = score;

	// save Kalman Filter state
	childHyp.KalmanFilterState = kalmanFilter.statePost.clone();
	childHyp.KalmanFilterStateCovariance = kalmanFilter.errorCovPost.clone();
}
