#include "SwimmingPoolObserver.h"
#include <vector>
#include <cassert>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;

// Accumulates all given blobs into the single mask.
void buildBlobsMask(const std::vector<DetectedBlob>& blobs, cv::Mat& outMask)
{
	assert(!outMask.empty());

	cv::Mat curBlobExtended = cv::Mat::zeros(outMask.rows, outMask.cols, CV_8UC1); // minimal image extended to the size of the original image
	for (const auto& blob : blobs)
	{
		// extend blob to the

		curBlobExtended.setTo(0);
		auto bnds = blob.BoundingBox;

		auto dstMat = curBlobExtended(bnds);
		blob.FilledImage.copyTo(dstMat);

		cv::bitwise_or(outMask, curBlobExtended, outMask);
	}
}

// Split large blobs into parts taking into account the expected layout of blobs.
void splitBlobsAccordingToPreviousFrameBlobs(const cv::Mat& imageBodyBin, const std::vector<DetectedBlob>& expectedBlobs, cv::Mat& imageBodySplit)
{
	if (!expectedBlobs.empty())
	{
		// construct expected mask
		cv::Mat expectedMask = cv::Mat::zeros(imageBodyBin.rows, imageBodyBin.cols, CV_8UC1);
		buildBlobsMask(expectedBlobs, expectedMask);

		cv::Mat intersect;
		cv::bitwise_and(expectedMask, imageBodyBin, intersect);

		// find small blobs and add them to the intersection
		cv::Mat smallDebris;
		//cv::subtract(expectedMask, intersect, smallDebris);
		cv::subtract(imageBodyBin, intersect, smallDebris);

		vector<vector<cv::Point>> smallContours;
		cv::findContours(smallDebris, smallContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		// treat small debris around intersection area as a part of current frame blobs
		// append small debris to intersection blobs
		for (size_t i = 0; i < smallContours.size(); ++i)
		{
			const vector<cv::Point>& c = smallContours[i];
			float area = cv::contourArea(c, false);
			float maxSmallDebrisSize = 256;
			if (area < maxSmallDebrisSize)
				cv::drawContours(intersect, smallContours, i, cv::Scalar::all(255), CV_FILLED);
		}

		// slightly enlarge common regions to avoid diminishing shape in time
		const int blobsGap = 3;
		auto sel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(blobsGap, blobsGap));
		cv::Mat intersectPad;
		cv::morphologyEx(intersect, intersectPad, cv::MORPH_DILATE, sel);

		cv::subtract(imageBodyBin, intersectPad, imageBodySplit); // current without extended intersection
		cv::bitwise_or(imageBodySplit, intersect, imageBodySplit); // current as union of expected blobs and the remainder
	}
	else
		imageBodySplit = imageBodyBin;
}

void getHumanBodies(const cv::Mat& image, const cv::Mat_<uchar>& waterMask, const std::vector<DetectedBlob>& expectedBlobs, std::vector<DetectedBlob>& blobs)
{
	cv::Mat_<uchar> nonWaterMask;
	cv::subtract(255, waterMask, nonWaterMask);

	// get image with bodies

	cv::Mat imageBody;
	image.copyTo(imageBody, nonWaterMask);

	cv::Mat imageBodyBin;
	cvtColor(imageBody, imageBodyBin, CV_BGR2GRAY);
	cv::threshold(imageBodyBin, imageBodyBin, 1, 255, cv::THRESH_BINARY);
	
	CV_Assert(imageBodyBin.channels() == 1);

	cv::Mat imageBodySplit = cv::Mat::zeros(imageBodyBin.rows, imageBodyBin.cols, CV_8UC1);
	splitBlobsAccordingToPreviousFrameBlobs(imageBodyBin, expectedBlobs, imageBodySplit);

	// =3 may be too small
	// =5 good
	const int narrowSize = 5;
	auto sel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(narrowSize, narrowSize));

	// cut tenuous bridges between connected components
	cv::Mat noTenuousBridges = cv::Mat_<uchar>::zeros(image.rows, image.cols);
	cv::morphologyEx(imageBodySplit, noTenuousBridges, cv::MORPH_OPEN, sel);

	// remove noise on a pixel level(too small / large components)

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

#if PW_DEBUG
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
#if PW_DEBUG
			cv::drawContours(imageNoExtremeBlobs, contours, i, cv::Scalar::all(255), CV_FILLED);
#endif
		}
		else
			contour.markDeleted = true;
	}

	// remove elongated stripes created by lane markers and water 'blique'

#if PW_DEBUG
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
#if PW_DEBUG
			cv::drawContours(imageNoSticks, contours, i, cv::Scalar::all(255), CV_FILLED);
#endif
		}
	}

	//
	cv::Mat curOutlineMat = cv::Mat::zeros(image.rows, image.cols, CV_8UC1); // minimal image extended to the size of the original image
	std::vector<std::vector<cv::Point>> contoursListTmp(1);
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

		// draw shape without any noise that hit the bounding box
		curOutlineMat.setTo(0);
		contoursListTmp[0] = contour.outlinePixels;
		cv::drawContours(curOutlineMat, contoursListTmp, 0, cv::Scalar::all(255), CV_FILLED);
		//cv::Mat localImgOld = noTenuousBridges(Range(bnd.y, bnd.y + bnd.height), Range(bnd.x, bnd.x + bnd.width));
		cv::Mat localImg = curOutlineMat(bnd).clone();
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

		blob.AreaPix = contour.area;
		
		//
		blobs.push_back(blob);
	}
}