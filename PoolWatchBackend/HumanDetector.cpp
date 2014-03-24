#include "SwimmingPoolObserver.h"
#include <vector>
#include <cassert>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;

void getHumanBodies(const cv::Mat& image, const cv::Mat_<uchar>& waterMask, std::vector<DetectedBlob>& blobs)
{
	cv::Mat_<uchar> nonWaterMask;
	cv::subtract(255, waterMask, nonWaterMask);

	// get image with bodies

	cv::Mat imageBody;
	image.copyTo(imageBody, nonWaterMask);

	cv::Mat imageBodyBin;
	cvtColor(imageBody, imageBodyBin, CV_BGR2GRAY);
	cv::threshold(imageBodyBin, imageBodyBin, 1, 255, cv::THRESH_BINARY);
	
	// cut tenuous bridges between connected components
	// =3 may be too small
	// =5 good
	const int narrowSize = 5;
	auto sel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(narrowSize, narrowSize));
	cv::Mat noTenuousBridges = cv::Mat_<uchar>::zeros(image.rows, image.cols);
	cv::morphologyEx(imageBodyBin, noTenuousBridges, cv::MORPH_OPEN, sel);

	// remove noise on a pixel level(too small / large components)

	CV_Assert(noTenuousBridges.channels() == 1);

	struct ContourInfo
	{
		std::vector<cv::Point> outlinePixels;
		float area;
		bool markDeleted;
	};
	vector<ContourInfo> contourInfos;
	vector<vector<cv::Point>> contours;
	{
		cv::findContours(noTenuousBridges.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		contourInfos.resize(contours.size());

		std::transform(begin(contours), end(contours), begin(contourInfos), [](vector<cv::Point>& outlinePixels)
		{
			ContourInfo c;
			//c.outlinePixels = std::move(outlinePixels);
			c.outlinePixels = outlinePixels;
			c.markDeleted = false;

			auto area = cv::contourArea(c.outlinePixels);
			c.area = area;

			return c;
		});
	}

#if _DEBUG
	cv::Mat imageNoExtremeBlobs = cv::Mat_<uchar>::zeros(image.rows, image.cols);
#endif

	for (size_t i = 0; i < contourInfos.size(); ++i)
	{
		auto& contour = contourInfos[i];

		// expect swimmer shape island to be in range
		const int shapeMinAreaPixels = 6; // (in pixels) any patch below is noise
		const int swimmerShapeAreaMax = 5000; // (in pixels) anything greater can't be a swimmer

		if (contour.area >= shapeMinAreaPixels && contour.area < swimmerShapeAreaMax)
		{
#if _DEBUG
			cv::drawContours(imageNoExtremeBlobs, contours, i, cv::Scalar::all(255), CV_FILLED);
#endif
		}
		else
			contour.markDeleted = true;
	}

	// remove elongated stripes created by lane markers and water 'blique'

#if _DEBUG
	cv::Mat imageNoSticks = cv::Mat_<uchar>::zeros(image.rows, image.cols);
#endif

	for (size_t i = 0; i < contourInfos.size(); ++i)
	{
		auto& contour = contourInfos[i];
		if (contour.markDeleted)
			continue;

		cv::RotatedRect rotRec = cv::fitEllipse(contour.outlinePixels);
		
		auto minorAxis = std::min(rotRec.size.width, rotRec.size.height);
		auto majorAxis = std::max(rotRec.size.width, rotRec.size.height);
		float axisRatio = minorAxis / majorAxis;

		const float minBlobAxisRatio = 0.1f; // blobs with lesser ratio are noise
		if (axisRatio < minBlobAxisRatio)
			contour.markDeleted = true;
		else
		{
#if _DEBUG
			cv::drawContours(imageNoSticks, contours, i, cv::Scalar::all(255), CV_FILLED);
#endif
		}
	}

	//
	int blobId = 1;
	for (size_t i = 0; i < contourInfos.size(); ++i)
	{
		const auto& contour = contourInfos[i];
		if (contour.markDeleted)
			continue;

		DetectedBlob blob;
		blob.Id = blobId++;

		cv::Rect bnd = cv::boundingRect(contour.outlinePixels);
		blob.BoundingBox = cv::Rect2f(bnd.x, bnd.y, bnd.width, bnd.height);

		cv::Mat localImg = noTenuousBridges(Range(bnd.y, bnd.y + bnd.height), Range(bnd.x, bnd.x + bnd.width));
		blob.FilledImage = localImg;

		cv::Moments ms = cv::moments(localImg, true);
		float cx = ms.m10 / ms.m00;
		float cy = ms.m01 / ms.m00;
		blob.Centroid = cv::Point2f(bnd.x + cx, bnd.y + cy);

		// [N,2] of (Y,X) pairs, N=number of points
		auto outlPix = cv::Mat_<int32_t>(contour.outlinePixels.size(), 2);
		for (size_t i = 0; i < contour.outlinePixels.size(); ++i)
		{
			auto point = contour.outlinePixels[i];
			outlPix(i, 0) = point.y;
			outlPix(i, 1) = point.x;
		}
		blob.OutlinePixels = outlPix;
		
		//
		blobs.push_back(blob);
	}
}