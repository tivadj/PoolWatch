#include <vector>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include "PoolWatchFacade.h"

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
}