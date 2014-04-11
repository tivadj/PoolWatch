#pragma once
#include <opencv2/core.hpp>

#include <QDir>

#include "PoolWatchFacade.h"
#include "KalmanFilterMovementPredictor.h"

class LinearCameraProjector : public CameraProjectorBase
{
public:
	LinearCameraProjector();
	virtual ~LinearCameraProjector();

	cv::Point2f worldToCamera(const cv::Point3f& world) const override;
	cv::Point3f cameraToWorld(const cv::Point2f& imagePos) const override;
};

class ConstantVelocityMovementPredictor : public SwimmerMovementPredictor
{
	cv::Point3f velocity_;
	float sigma_; // in N(mu,sigma^2) shows how actual and estimated position agree
public:
	ConstantVelocityMovementPredictor(const cv::Point3f& velocity);
	ConstantVelocityMovementPredictor(const ConstantVelocityMovementPredictor&) = delete;
	virtual ~ConstantVelocityMovementPredictor();

	void initScoreAndState(const cv::Point3f& blobCentrWorld, float& score, TrackHypothesisTreeNode& saveNode) override;

	void estimateAndSave(const TrackHypothesisTreeNode& curNode, const cv::Point3f& blobCentrWorld, cv::Point3f& estPos, float& score, TrackHypothesisTreeNode& saveNode) override;

private:
	// use Kalman Filter state matrix just to store position of a blob
	// state = [x y vx vy]' and (vx,vy) components are ignored
	const cv::Mat& nodeState(const TrackHypothesisTreeNode& node) const;
	      cv::Mat& nodeState(      TrackHypothesisTreeNode& node) const;
};


void configureLogToFileAppender(const QDir& logFolder, const QString& logFileName);