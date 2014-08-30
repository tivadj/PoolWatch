#include <vector>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include "VisualObservation.h"

using namespace std;

void getPoolMask(const cv::Mat& image, const cv::Mat_<uchar>& waterMask, cv::Mat_<uchar>& poolMask)
{
	cv::Mat imageWater;
	image.copyTo(imageWater, waterMask);

	cv::Mat imageWaterGray;
	cv::cvtColor(imageWater, imageWaterGray, CV_BGR2GRAY);

	//
	vector<vector<cv::Point> > contours;
	cv::findContours(imageWaterGray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	cv::Mat imageCntrs = imageWater.clone();
	
	vector<cv::Scalar> colors = {
		CV_RGB(0, 255, 0),
		CV_RGB(255, 0, 0),
		CV_RGB(0, 0, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(255, 0, 255),
		CV_RGB(255, 255, 0)
	};
	
	//
	poolMask = cv::Mat_<uchar>::zeros(image.rows, image.cols);

	// remove components with small area (assume pool is big)

	int contourIndex = 0;
	for (const auto& contour : contours)
	{
		auto area = cv::contourArea(contour);

		const int poolAreaMinPix = 5000;
		if (area >= poolAreaMinPix)
			cv::drawContours(poolMask, contours, contourIndex, cv::Scalar::all(255), CV_FILLED);
			

		auto col = colors[contourIndex % colors.size()];
		cv::drawContours(imageCntrs, contours, contourIndex, col, 1);

		contourIndex++;
	}

	// glue sparse blobs
	// size 13 has holes
	auto sel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(17, 17));
	cv::morphologyEx(poolMask, poolMask, cv::MORPH_CLOSE, sel);

	// fill holes
	// TODO: cv::floodFill()

	// leave some padding between water and pool tiles to avoid stripes of
	// tiles to be associated with a swimmer
	const int dividerPadding = 5;
	auto selPad = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dividerPadding, dividerPadding));
	cv::erode(poolMask, poolMask, selPad);

	// some objects may obstruct pool from camera
	// hence here we glue all islands of pixels
	// Also people in a pool are not detected as water. This leads to incorrect (smaller) pool boundary due to cavities and
	// real swimmers may be cut off in frame processing. Hence all blob parts of the pool are made convex.
	contours.clear();
	cv::findContours(poolMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	// merge points of all blobs

	// TODO: resolve poolMask was not found use case
	// if there is no blob found, then just return flag; then eg client may use previous pool mask
	std::vector<cv::Point> allBlobPoints;
	if (!contours.empty())
		allBlobPoints.swap(std::move(contours[0]));
	for (size_t i = 1; i < contours.size(); ++i)
		std::copy(begin(contours[i]), end(contours[i]), std::back_inserter(allBlobPoints));

	std::vector<cv::Point> poolConvexHullPoints;
	cv::convexHull(allBlobPoints, poolConvexHullPoints);

	// aquire pool mask
	std::vector<std::vector<cv::Point>> entireOutline(1);
	entireOutline[0] = std::move(poolConvexHullPoints);
	poolMask.setTo(0); // reset pool mask, because it was corrupted by cv::findCountours
	cv::drawContours(poolMask, entireOutline, 0, cv::Scalar::all(255), CV_FILLED);
}