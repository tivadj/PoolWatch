#include "stdafx.h"
#include "TestingUtils.h"

ConstantVelocityMovementPredictor::ConstantVelocityMovementPredictor(const cv::Point3f& velocity) : velocity_(velocity)
{	
	float maxShift = std::max(velocity_.x, velocity_.y);

	// treat part of the maxShift as expected max error
	float maxError = maxShift / 2;

	// select sigma so that 3*sgma = maxError
	sigma_ = maxError / 3;
}

ConstantVelocityMovementPredictor::~ConstantVelocityMovementPredictor()
{
}

void ConstantVelocityMovementPredictor::initScoreAndState(const cv::Point3f& blobCentrWorld, float& score, TrackHypothesisTreeNode& saveNode)
{
	const float initialTrackScore = 5;
	score = normalizedDistance(blobCentrWorld, blobCentrWorld, sigma_);

	auto& saveState = nodeState(saveNode);
	saveState.at<float>(0, 0) = blobCentrWorld.x;
	saveState.at<float>(1, 0) = blobCentrWorld.y;
}

void ConstantVelocityMovementPredictor::estimateAndSave(const TrackHypothesisTreeNode& curNode, const cv::Point3_<float>& blobCentrWorld, cv::Point3_<float>& estPos, float& score, TrackHypothesisTreeNode& saveNode)
{
	const auto& curState = nodeState(curNode);

	float x = curState.at<float>(0, 0);
	float y = curState.at<float>(1, 0);
	cv::Point3f curPos = cv::Point3f(x, y, 0);

	// model constant speed movement
	estPos = curPos + velocity_;

	score = curNode.Score + normalizedDistance(blobCentrWorld, estPos, sigma_);

	// save
	auto& saveState = nodeState(saveNode);
	saveState.at<float>(0, 0) = estPos.x;
	saveState.at<float>(1, 0) = estPos.y;
}

cv::Mat const& ConstantVelocityMovementPredictor::nodeState(TrackHypothesisTreeNode const& node) const
{
	return node.KalmanFilterState;
}

cv::Mat& ConstantVelocityMovementPredictor::nodeState(TrackHypothesisTreeNode& node) const
{
	return node.KalmanFilterState;
}