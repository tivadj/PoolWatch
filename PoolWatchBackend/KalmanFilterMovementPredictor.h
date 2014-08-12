#pragma once
#include <opencv2\video\tracking.hpp> // KalmanFilter
#include "TrackHypothesisTreeNode.h"

float kalmanFilterDistance(const cv::KalmanFilter& kalmanFilter, const cv::Mat& observedPos);
_declspec(dllimport) float normalizedDistance(const cv::Point3f& pos, const cv::Point3f& mu, float sigma);

class _declspec(dllexport) KalmanFilterMovementPredictor : public SwimmerMovementPredictor
{
	const int KalmanFilterDynamicParamsCount = 4;
	const float NullPosX = -1;
	cv::KalmanFilter kalmanFilter_;
	float penalty_;
	float blobMaxShift_;
public:
	KalmanFilterMovementPredictor(float fps, float swimmerMaxSpeed);
	KalmanFilterMovementPredictor(const KalmanFilterMovementPredictor&) = delete;

private:
	void initKalmanFilter(cv::KalmanFilter& kalmanFilter, float fps, float swimmerMaxSpeed);

public:
	virtual ~KalmanFilterMovementPredictor();

	void initScoreAndState(int frameInd, int observationInd, const cv::Point3f& blobCentrWorld, float& score, TrackHypothesisTreeNode& saveNode) override;

	// Estimates position and score of movement from position of curNode to blobCentrWorld. Saves state into saveNode.
	void estimateAndSave(const TrackHypothesisTreeNode& curNode, const boost::optional<cv::Point3f>& blobCentrWorld, cv::Point3f& estPos, float& score, TrackHypothesisTreeNode& saveNode) override;
};
