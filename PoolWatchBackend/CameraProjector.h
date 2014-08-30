#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>

#include "PoolWatchFacade.h"

PW_EXPORTS void approxCameraMatrix(int imageWidth, int imageHeight, float fovX, float fovY, float& cx, float& cy, float& fx, float& fy);
PW_EXPORTS void fillCameraMatrix(float cx, float cy, float fx, float fy, cv::Matx33f& cameraMatrix);

class PW_EXPORTS CameraProjectorBase
{
public:
	virtual ~CameraProjectorBase() {}
	virtual cv::Point2f worldToCamera(const cv::Point3f& world) const = 0;
	virtual cv::Point3f cameraToWorld(const cv::Point2f& imagePos) const = 0;

	// Finds the area of shape in image(in pixels ^ 2) of an object with world position __worldPos__(in m)
	float worldAreaToCamera(const cv::Point3f& worldPos, float worldArea) const;

	// Calculates distance which has a segment of length worldDist at position worldPos when translating to camera coordinates.
	float distanceWorldToCamera(const cv::Point3f& worldPos, float worldDist) const;
};

/// Class to map camera's image coordinates (X,Y in pixels) and swimming 
/// pool world coordinates ([x y z] in meters).
class PW_EXPORTS CameraProjector : public CameraProjectorBase
{
private:
	cv::Mat_<float> cameraMatrix33_;
	cv::Mat_<float> cameraMatrix33Inv_;
	cv::Mat_<float> distCoeffs_;
	cv::Mat_<float> rvec_;
	cv::Mat_<float> tvec_;
	cv::Mat_<float> worldToCamera44_;
	cv::Mat_<float> cameraToWorld44_;
public:
	CameraProjector();
	virtual ~CameraProjector();
public:
	bool orientCamera(const cv::Matx33f& cameraMat, const std::vector<cv::Point3f>& worldPoints, const std::vector<cv::Point2f>& imagePoints);
	cv::Point2f worldToCamera(const cv::Point3f& world) const override;
	cv::Point3f cameraToWorld(const cv::Point2f& imagePos) const override;
	cv::Point3f cameraPosition() const;

	static float zeroHeight() { return 0; }
};