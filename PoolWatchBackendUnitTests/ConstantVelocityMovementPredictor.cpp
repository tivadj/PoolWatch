#include "stdafx.h"
#include "TestingUtils.h"
#include "SwimmingPoolObserver.h"

ConstantVelocityMovementPredictor::ConstantVelocityMovementPredictor(const cv::Point3f& defaultVelocity) : defaultVelocity_(defaultVelocity)
{	
	float maxShift = std::max(defaultVelocity_.x, defaultVelocity_.y);

	// treat part of the maxShift as expected max error
	float maxError = maxShift / 2;

	// select sigma so that 3*sgma = maxError
	sigma_ = maxError / 3;
}

ConstantVelocityMovementPredictor::~ConstantVelocityMovementPredictor()
{
}

void ConstantVelocityMovementPredictor::initScoreAndState(int frameInd, int observationInd, const cv::Point3f& blobCentrWorld, float& score, TrackHypothesisTreeNode& saveNode)
{
	const float initialTrackScore = 5;

	cv::Matx21f blobCentrWorldMat(blobCentrWorld.x, blobCentrWorld.y);
	cv::Matx22f covMat(sigma_*sigma_, 0, 0, sigma_*sigma_);
	bool distOp = normalizedDistance(blobCentrWorldMat, blobCentrWorldMat, covMat, score);
	CV_Assert(distOp);

	auto& saveState = nodeState(saveNode);
	saveState.x = blobCentrWorld.x;
	saveState.y = blobCentrWorld.y;
	saveState.vx = 0;
	saveState.vy = 0;

	auto specVelocity = getSwimmerVelocity(frameInd, observationInd);
	cv::Point3f swimmerVelocity = specVelocity.get_value_or(defaultVelocity_);
	saveState.vx = swimmerVelocity.x;
	saveState.vy = swimmerVelocity.y;
}

void ConstantVelocityMovementPredictor::estimateAndSave(const TrackHypothesisTreeNode& curNode, const boost::optional<cv::Point3f>& blobCentrWorld, cv::Point3_<float>& estPos, float& deltaMovementScore, TrackHypothesisTreeNode& saveNode)
{
	const auto& curState = nodeState(curNode);

	float x = curState.x;
	float y = curState.y;
	cv::Point3f curPos = cv::Point3f(x, y, 0);

	//
	cv::Point3f swimmerVelocity = cv::Point3f(curState.vx,curState.vy, CameraProjector::zeroHeight());

	// model constant speed movement
	estPos = curPos + swimmerVelocity;

	if (blobCentrWorld != nullptr)
	{
		cv::Matx21f blobCentrWorldMat(blobCentrWorld.get().x, blobCentrWorld.get().y);
		cv::Matx21f estPosMat(estPos.x, estPos.y);
		cv::Matx22f covMat(sigma_*sigma_, 0, 0, sigma_*sigma_);
		bool distOp= normalizedDistance(blobCentrWorldMat, estPosMat, covMat, deltaMovementScore);
		CV_Assert(distOp);
	}
	else
	{
		// penalty for missed observation
		// prob 0.4 - penalty - 0.9163
		// prob 0.6 - penalty - 0.5108
		const float probDetection = 0.6f;
		//const float penalty = log(1 - probDetection);
		const float penalty = 1.79/10;
		deltaMovementScore = penalty;
	}

	// save
	auto& saveState = nodeState(saveNode);
	saveState.x = estPos.x;
	saveState.y = estPos.y;
	saveState.vx = swimmerVelocity.x;
	saveState.vy = swimmerVelocity.y;
}

boost::optional<cv::Point3f> ConstantVelocityMovementPredictor::getSwimmerVelocity(int frameInd, int observationInd) const
{
	for (const TrackVelocity& trackVel : trackVelocityList_)
	{
		if (trackVel.FamilyIdHint.FrameInd == frameInd && trackVel.FamilyIdHint.ObservationInd == observationInd)
			return trackVel.Velocity;
	}
	return boost::optional<cv::Point3f>();
}

void ConstantVelocityMovementPredictor::setSwimmerVelocity(FamilyIdHint familyIdHint, const cv::Point3f& velocity)
{
	TrackVelocity vel(familyIdHint, velocity);
	trackVelocityList_.push_back(vel);
}

const ConstantVelocityMovementPredictor::SwimmerState& ConstantVelocityMovementPredictor::nodeState(TrackHypothesisTreeNode const& node) const
{
	// use Kalman Filter state matrix just to store position of a blob
	return reinterpret_cast<const SwimmerState&>(node.KalmanFilterState.val);
}

ConstantVelocityMovementPredictor::SwimmerState& ConstantVelocityMovementPredictor::nodeState(TrackHypothesisTreeNode& node) const
{
	return reinterpret_cast<SwimmerState&>(node.KalmanFilterState.val);
}