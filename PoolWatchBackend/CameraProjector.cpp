#include <vector>
#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"

#include "CameraProjector.h"
using namespace std;

CameraProjector::CameraProjector()
{
	init();
}


CameraProjector::~CameraProjector()
{
}

void CameraProjector::init()
{
	// construct camera matrix

	// Cannon D20 640x480
	float cx = 323.07199373780122f;
	float cy = 241.16033688735058f;
	float fx = 526.96329424435044f;
	float fy = 527.46802103114874f;

	float cameraMatrixArray[] = { fx, 0, cx, 0, fy, cy, 0, 0, 1 };
	cv::Mat_<float> cameraMatrix(3, 3, cameraMatrixArray);
	cameraMatrix.copyTo(cameraMatrix33_);
	//cout << "cameraMatrix33_=" << cameraMatrix33_ << endl;

	cameraMatrix33Inv_ = cameraMatrix33_.inv();
	//cout << "cameraMatrix33Inv_=" << cameraMatrix33Inv_ << endl;

	distCoeffs_ = cv::Mat::zeros(5, 1, CV_32F); // do not use distortion

	// construct world-image points correspondence

	std::vector<cv::Point3f> worldPoints;
	std::vector<cv::Point2f> imagePoints;

	// top, origin(0, 0)
	imagePoints.push_back(cv::Point2f(242, 166));
	worldPoints.push_back(cv::Point3f(0, 0, zeroHeight));
	
	//top, 4 marker
	imagePoints.push_back(cv::Point2f(516, 156));
	worldPoints.push_back(cv::Point3f(0, 10, zeroHeight));

	//bottom, 2 marker
	imagePoints.push_back(cv::Point2f(-71, 304));
	worldPoints.push_back(cv::Point3f(25, 6, zeroHeight));

	// bottom, 4 marker
	imagePoints.push_back(cv::Point2f(730, 365));
	worldPoints.push_back(cv::Point3f(25, 10, zeroHeight));

	//
	cv::Mat_<double> rvecDbl(3, 1);
	cv::Mat_<double> tvecDbl(3, 1);
	bool isPosEstimated = cv::solvePnP(worldPoints, imagePoints, cameraMatrix33_, distCoeffs_, rvecDbl, tvecDbl);
	rvecDbl.convertTo(rvec_, CV_32F);
	tvecDbl.convertTo(tvec_, CV_32F);
	//cout << "tvec_=" << tvec_ << endl;

	cv::Mat_<double> rotMat;
	cv::Rodrigues(rvecDbl, rotMat);
	//cout << "rotMat=" << rotMat << endl;

	//
	worldToCamera44_ = cv::Mat_<float>::eye(4,4);
	for (int row=0; row < 3; ++row)
	{
	    for (int col=0; col < 3; ++col)
	    {
			worldToCamera44_(row, col) = (float)rotMat(row, col);
	    }
		worldToCamera44_(row, 3) = (float)tvecDbl.at<double>(row, 0);
	}
	
	//cout << "worldToCamera44_=" << worldToCamera44_ << endl;
	cameraToWorld44_ = worldToCamera44_.inv();
	//cout << "cameraToWorld44_=" <<cameraToWorld44_ << endl;
}

cv::Point2f CameraProjector::worldToCamera(const cv::Point3f& world)
{ 
	vector<cv::Point3f> worldPoints;
	worldPoints.push_back(world);

	std::vector<cv::Point2f> imagePointsProj;
	cv::projectPoints(worldPoints, rvec_, tvec_, cameraMatrix33_, distCoeffs_, imagePointsProj);
	return imagePointsProj[0];
}

cv::Point3f CameraProjector::cameraToWorld(const cv::Point2f& imagePos)
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

	float homZ = (zeroHeight - tmp.at<float>(0, 0)) / cameraToWorld44_(2, 3);

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
