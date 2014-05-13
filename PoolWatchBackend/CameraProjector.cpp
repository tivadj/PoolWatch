#include <vector>
#include <iostream>
#include <array>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "algos1.h"
#include "PoolWatchFacade.h"
using namespace std;

// Approximates camera matrix parameters (fx, fy, cx, cy) using camera resoulution and expected
// horizontal and vertical field of view.
// fov=field of view
void approxCameraMatrix(int imageWidth, int imageHeight, float fovX, float fovY, float& cx, float& cy, float& fx, float& fy)
{
	cx = imageWidth / 2;
	cy = imageHeight / 2;

	// tan(fovx/2) = cx / fx
	fx = cx / std::tanf(fovX / 2);
	fy = cy / std::tanf(fovY / 2);
}

void fillCameraMatrix(float cx, float cy, float fx, float fy, cv::Matx33f& cameraMatrix)
{
	cameraMatrix << fx, 0, cx, 0, fy, cy, 0, 0, 1;
}

float CameraProjectorBase::worldAreaToCamera(cv::Point3f const& worldPos, float worldArea) const
{
	float widthHf = std::sqrt(worldArea) / 2;

	std::array<cv::Point3f, 4> ps;
	ps[0] = worldPos + cv::Point3f(-widthHf, -widthHf, 0);
	ps[1] = worldPos + cv::Point3f(-widthHf,  widthHf, 0);
	ps[2] = worldPos + cv::Point3f( widthHf,  widthHf, 0);
	ps[3] = worldPos + cv::Point3f( widthHf, -widthHf, 0);

	// project body bounding box into camera
	std::vector<cv::Point2f> cs(4);
	cs[0] = worldToCamera(ps[0]);
	cs[1] = worldToCamera(ps[1]);
	cs[2] = worldToCamera(ps[2]);
	cs[3] = worldToCamera(ps[3]);
	float area = (float)cv::contourArea(cs, false);
	return area;
}

float CameraProjectorBase::distanceWorldToCamera(const cv::Point3f& worldPos, float worldDist) const
{
	float worldArea = PoolWatch::sqr(worldDist / 2)*M_PI;

	float camArea = worldAreaToCamera(worldPos, worldArea);

	float camDist = std::sqrt(camArea / M_PI);
	return camDist;
}

CameraProjector::CameraProjector()
{
	distCoeffs_ = cv::Mat::zeros(5, 1, CV_32F); // do not use distortion
}


CameraProjector::~CameraProjector()
{
}

bool CameraProjector::orientCamera(const cv::Matx33f& cameraMat, const std::vector<cv::Point3f>& worldPoints, const std::vector<cv::Point2f>& imagePoints)
{
	// construct world-image points correspondence

	cv::Mat_<double> rvecDbl(3, 1);
	cv::Mat_<double> tvecDbl(3, 1);
	bool isPosEstimated = cv::solvePnP(worldPoints, imagePoints, cameraMat, distCoeffs_, rvecDbl, tvecDbl);
	if (!isPosEstimated)
		return false;

	// copy camera matrix

	cv::Mat(3, 3, CV_32FC1, const_cast<float*>(&cameraMat(0, 0))).copyTo(cameraMatrix33_);

	cameraMatrix33Inv_ = cameraMatrix33_.inv();

	//

	rvecDbl.convertTo(rvec_, CV_32F);
	tvecDbl.convertTo(tvec_, CV_32F);

	cv::Mat_<double> rotMat;
	cv::Rodrigues(rvecDbl, rotMat);

	//
	worldToCamera44_ = cv::Mat_<float>::eye(4, 4);
	for (int row = 0; row < 3; ++row)
	{
		for (int col = 0; col < 3; ++col)
		{
			worldToCamera44_(row, col) = (float)rotMat(row, col);
		}
		worldToCamera44_(row, 3) = (float)tvecDbl.at<double>(row, 0);
	}

	cameraToWorld44_ = worldToCamera44_.inv();

	return true;
}

cv::Point2f CameraProjector::worldToCamera(const cv::Point3f& world) const
{ 
	vector<cv::Point3f> worldPoints;
	worldPoints.push_back(world);

	std::vector<cv::Point2f> imagePointsProj;
	cv::projectPoints(worldPoints, rvec_, tvec_, cameraMatrix33_, distCoeffs_, imagePointsProj);
	return imagePointsProj[0];
}

cv::Point3f CameraProjector::cameraToWorld(const cv::Point2f& imagePos) const
{
	// image to camera coordinates
	cv::Mat_<float> imagePosMat(3, 1);
	imagePosMat(0, 0) = imagePos.x;
	imagePosMat(1, 0) = imagePos.y;
	imagePosMat(2, 0) = 1;
	
	cv::Mat camPos = cameraMatrix33Inv_ * imagePosMat;
	//cout << camPos << endl;

	cv::Mat_<float> row(1, 3);
	row(0, 0) = cameraToWorld44_(2, 0);
	row(0, 1) = cameraToWorld44_(2, 1);
	row(0, 2) = cameraToWorld44_(2, 2);
	cv::Mat tmp = row * camPos;
	//cout << "tmp=" << tmp << endl;

	float homZ = (zeroHeight() - tmp.at<float>(0, 0)) / cameraToWorld44_(2, 3);

	//
	cv::Mat_<float> camPos4(4,1);
	camPos4(0, 0) = camPos.at<float>(0, 0);
	camPos4(1, 0) = camPos.at<float>(1, 0);
	camPos4(2, 0) = camPos.at<float>(2, 0);
	camPos4(3, 0) = homZ;
	
	cv::Mat_<float> reW1 = cameraToWorld44_ * camPos4;
	
	// normalize homogeneous
	float x = reW1(0) / reW1(3);
	float y = reW1(1) / reW1(3);
	float z = reW1(2) / reW1(3);
	return cv::Point3f(x, y, z);
}

cv::Point3f CameraProjector::cameraPosition() const
{
	float x = cameraToWorld44_(0, 3);
	float y = cameraToWorld44_(1, 3);
	float z = cameraToWorld44_(2, 3);
	return cv::Point3f(x, y, z);
}