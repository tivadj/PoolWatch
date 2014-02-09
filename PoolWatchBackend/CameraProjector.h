#pragma once

/// Class to map camera's image coordinates (X,Y in pixels) and swimming 
/// pool world coordinates ([x y z] in meters).
class CameraProjector
{
	const float zeroHeight = 0;

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
private:
	void init();
public:
	cv::Point2f worldToCamera(const cv::Point3f& world);
	cv::Point3f cameraToWorld(const cv::Point2f& imagePos);
};

