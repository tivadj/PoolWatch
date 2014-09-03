#pragma once
#include <opencv2\video\tracking.hpp> // KalmanFilter
#include "PoolWatchFacade.h"
#include "MultiHypothesisBlobTracker.h"
#include "SwimmerMovementModel.h"

float kalmanFilterDistance(const cv::KalmanFilter& kalmanFilter, const cv::Mat& observedPos);
PW_EXPORTS bool normalizedDistance(const cv::Matx21f& pos, const cv::Matx21f& mu, const cv::Matx22f& sigma, float& dist);

class PW_EXPORTS KalmanFilterMovementPredictor : public SwimmerMovementPredictor
{
	const int KalmanFilterDynamicParamsCount = 4;
	const float NullPosX = -1;
	cv::KalmanFilter kalmanFilter_;
	float penalty_;
	float maxShiftPerFrameM_; // (in meters) maximum shift of a tracked object from frame to frame
public:
	KalmanFilterMovementPredictor(float maxShiftPerFrame);
	KalmanFilterMovementPredictor(const KalmanFilterMovementPredictor&) = delete;

private:
	void initKalmanFilter(cv::KalmanFilter& kalmanFilter);

public:
	void initScoreAndState(int frameInd, int observationInd, const cv::Point3f& blobCentrWorld, float& score, TrackHypothesisTreeNode& saveNode) override;

	void estimateAndSave(const TrackHypothesisTreeNode& curNode, const boost::optional<cv::Point3f>& blobCentrWorld, cv::Point3f& estPos, float& deltaMovementScore, TrackHypothesisTreeNode& saveNode) override;

	float maxShiftPerFrame() const override;
};
