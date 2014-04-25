#include <vector>
#include <cassert>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include <boost/filesystem.hpp>

#include "HumanDetector.h"
#include "SwimmingPoolObserver.h"
#include "algos1.h"

using namespace std;
using namespace cv;

std::tuple<float,int,int> twoBlobsClosestDistanceSqr(vector<cv::Point> const& blobPoints1, vector<cv::Point> const& blobPoints2)
{
	CV_Assert(!blobPoints1.empty());
	CV_Assert(!blobPoints2.empty());

	float minDist = std::numeric_limits<float>::max();

	size_t minInd1 = -1;
	size_t minInd2 = -1;

	for (size_t i1 = 0; i1 < blobPoints1.size(); ++i1)
	{
		const auto& p1 = blobPoints1[i1];

		for (size_t i2 = 0; i2 < blobPoints2.size(); ++i2)
		{
			const auto& p2 = blobPoints2[i2];

			float dist2 = (float)(PoolWatch::sqr(p1.x - p2.x) + PoolWatch::sqr(p1.y - p2.y));
			if (dist2 < minDist)
			{
				minDist = dist2;
				minInd1 = i1;
				minInd2 = i2;
			}
		}
	}

	assert(minDist != std::numeric_limits<float>::max());
	assert(minInd1 != -1);
	assert(minInd2 != -1);

	return make_tuple(minDist, (int)minInd1, (int)minInd2);
}

void mergeBlobs(vector<cv::Point> const& blobPoints1, vector<cv::Point> const& blobPoints2, Mat& paintBuffer, vector<cv::Point>& mergedBlob, int drawBridgeCount)
{
	CV_Assert(paintBuffer.type() == CV_8UC1);
	CV_Assert(drawBridgeCount >= 1);

	struct BridgeRef
	{
		size_t i1;
		size_t i2;
		float dist;
	};

	vector<BridgeRef> twoBlobDistances;
	twoBlobDistances.reserve(blobPoints1.size() * blobPoints2.size());

	for (size_t i1 = 0; i1 < blobPoints1.size(); ++i1)
		for (size_t i2 = 0; i2 < blobPoints2.size(); ++i2)
		{
			const auto& p1 = blobPoints1[i1];
			const auto& p2 = blobPoints2[i2];

			float dist2 = (float)(PoolWatch::sqr(p1.x - p2.x) + PoolWatch::sqr(p1.y - p2.y));
			BridgeRef bridge;
			bridge.i1 = i1;
			bridge.i2 = i2;
			bridge.dist = dist2;
			twoBlobDistances.push_back(bridge);
		}

	// limit number of bridges to available number of connections between two blobs
	int takeBridges = std::min(drawBridgeCount, (int)(blobPoints1.size() * blobPoints2.size()));

	//
	vector<BridgeRef>::iterator sortTo = begin(twoBlobDistances);
	std::advance(sortTo, takeBridges);

	std::partial_sort(begin(twoBlobDistances), sortTo, end(twoBlobDistances), [](const BridgeRef& b1, const BridgeRef& b2) { return std::less<float>()(b1.dist, b2.dist); });

	for (int i = 0; i < takeBridges; ++i)
	{
		const auto& bridge = twoBlobDistances[i];
		const auto& p1 = blobPoints1[bridge.i1];
		const auto& p2 = blobPoints2[bridge.i2];
		cv::line(paintBuffer, p1, p2, cv::Scalar::all(255), 1);
	}

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(paintBuffer, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	CV_Assert(contours.size() == 1 && "Must be the single contour after merging two blobs");

	cv::drawContours(paintBuffer, contours, 0, cv::Scalar::all(255), CV_FILLED);
	mergedBlob.swap(contours[0]);
}

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

void fixContourInfo(ContourInfo& c, CameraProjectorBase* pCameraProjector)
{
	const auto& outlinePixels = c.outlinePixels;

	auto area = cv::contourArea(outlinePixels);
	c.area = area;

	cv::Point sum = std::accumulate(begin(outlinePixels), end(outlinePixels), cv::Point(), [](const cv::Point& a, const cv::Point& x) { return a + x; });

	float countF = (float)outlinePixels.size();
	c.contourCentroid = cv::Point2f(sum.x / countF, sum.y / countF);

	if (pCameraProjector != nullptr)
		c.contourCentroidWorld = pCameraProjector->cameraToWorld(c.contourCentroid);
	else
		c.contourCentroidWorld = cv::Point3f(-1, -1, -1);
}


SwimmerDetector::SwimmerDetector(const std::shared_ptr<CameraProjectorBase> cameraProjector)
: cameraProjector_(cameraProjector)
{
	// 0.3 is too large - far away head is undetected
	// eg.head 20 cm x 20 cm = 0.04
	// man of size 1m x 2m
	//float bodyAreaMin = 0.04;
	// 1.5 = blobs close to camera are merged with information sign
	bodyHalfLenth_ = 0.5; // parameter to specify how much close blobs should be merged

	cv::FileStorage fs;
	//
	auto FleshClassifierFileName = "skin_clasifier.yml";

	if (!fs.open(FleshClassifierFileName, cv::FileStorage::READ))
	{
		fleshClassifier_ = make_unique<WaterClassifier>(6, cv::EM::COV_MAT_SPHERICAL);
		trainSkinClassifier(*fleshClassifier_);

		if (!fs.open(FleshClassifierFileName, cv::FileStorage::WRITE))
			return;
		fleshClassifier_->write(fs);
		fs.release();
	}
	else
	{
		fleshClassifier_ = WaterClassifier::read(fs);
	}

	// TODO: why "const char*" can not be found
	//const char* LaneSeparatorFileName = "laneSeparator_clasifier.yml";
	auto LaneSeparatorFileName = "laneSeparator_clasifier.yml";

	if (!fs.open(LaneSeparatorFileName, cv::FileStorage::READ))
	{
		laneSeparatorClassifier_ = make_unique<WaterClassifier>(6, cv::EM::COV_MAT_SPHERICAL);

		trainLaneSeparatorClassifier(*laneSeparatorClassifier_);
		if (!fs.open(LaneSeparatorFileName, cv::FileStorage::WRITE))
			return;

		laneSeparatorClassifier_->write(fs);
		fs.release();
	}
	else
	{
		laneSeparatorClassifier_ = WaterClassifier::read(fs);
	}
}

SwimmerDetector::SwimmerDetector(int bodyHalfLenthPix) : SwimmerDetector(nullptr)
{
	bodyHalfLenthPix_ = bodyHalfLenthPix;
}

void SwimmerDetector::getBlobs(const cv::Mat& image, const std::vector<DetectedBlob>& expectedBlobs, std::vector<DetectedBlob>& blobs)
{
	auto skinClassifAbsFun = [this](const cv::Vec3d& pix) -> bool
	{
		double val = fleshClassifier_->computeOne(pix, true);
		return val > -10;
	};

	cv::Mat_<uchar> fleshMask;
	classifyAndGetMask(image, skinClassifAbsFun, fleshMask);

	//
	auto laneSepClassifFun = [this](const cv::Vec3d& pix) -> bool
	{
		return laneSeparatorClassifier_->predict(pix);
	};

	cv::Mat_<uchar> laneSepMask;
	classifyAndGetMask(image, laneSepClassifFun, laneSepMask);

	//
	cv::Mat fleshNoLaneSepsMat;
	cv::subtract(fleshMask, laneSepMask, fleshNoLaneSepsMat, noArray(), CV_16SC1);
	fleshNoLaneSepsMat.convertTo(fleshNoLaneSepsMat, CV_8UC1);

	// 1 = 1-pixel dots remain
	const int blobsGap = 3;
	auto sel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(blobsGap, blobsGap));
	cv::Mat fleshNoTenious;
	cv::morphologyEx(fleshNoLaneSepsMat, fleshNoTenious, cv::MORPH_OPEN, sel);

	// TODO: no water mask
	cv::Mat waterMask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	getHumanBodies(fleshNoTenious, waterMask, expectedBlobs, blobs);
}

void SwimmerDetector::getHumanBodies(const cv::Mat& image, const cv::Mat_<uchar>& waterMask, const std::vector<DetectedBlob>& expectedBlobs, std::vector<DetectedBlob>& blobs)
{
	int const& rows1 = image.rows;
	int const& width = image.cols;

	//cv::Mat_<uchar> nonWaterMask;
	//cv::subtract(255, waterMask, nonWaterMask);

	//// get image with bodies

	//cv::Mat imageBody;
	//image.copyTo(imageBody, nonWaterMask);

	//cv::Mat imageBodyBin;
	//cvtColor(imageBody, imageBodyBin, CV_BGR2GRAY);
	//cv::threshold(imageBodyBin, imageBodyBin, 1, 255, cv::THRESH_BINARY);
	//
	//CV_Assert(imageBodyBin.channels() == 1);

	//cv::Mat imageBodySplit = cv::Mat::zeros(imageBodyBin.rows, imageBodyBin.cols, CV_8UC1);
	//splitBlobsAccordingToPreviousFrameBlobs(imageBodyBin, expectedBlobs, imageBodySplit);

	//// =3 may be too small
	//// =5 good
	//const int narrowSize = 5;
	//auto sel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(narrowSize, narrowSize));

	//// cut tenuous bridges between connected components
	//cv::Mat noTenuousBridges = cv::Mat_<uchar>::zeros(rows1, width);
	//cv::morphologyEx(imageBodySplit, noTenuousBridges, cv::MORPH_OPEN, sel);

	cv::Mat noTenuousBridges = image;
	// remove noise on a pixel level(too small / large components)

	vector<ContourInfo> contourInfos;
	vector<vector<cv::Point>> contours;
	{
		cv::findContours(noTenuousBridges.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		contourInfos.resize(contours.size());

		std::transform(begin(contours), end(contours), begin(contourInfos), [this](vector<cv::Point>& outlinePixels)
		{
			ContourInfo c;
			//c.outlinePixels = std::move(outlinePixels);
			c.outlinePixels = outlinePixels;
			c.markDeleted = false;
			fixContourInfo(c, cameraProjector_.get());

			return c;
		});
	}

	// remove very small or large blobs

#if PW_DEBUG
	cv::Mat imageNoExtremeBlobs = cv::Mat_<uchar>::zeros(rows1, width);
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

	// merge small blobs around one large blob

	// process from largest blob to smallest
	std::sort(begin(contourInfos), end(contourInfos), [](const ContourInfo& c1, const ContourInfo& c2) { return std::greater<float>()(c1.area, c2.area); });

	cv::Mat mergeBuffer = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	vector<cv::Point> mergedBlobsContour;
	vector<uchar> processedMerging(contourInfos.size());
	vector<vector<cv::Point>> contoursList(1);
	for (size_t i = 0; i < contourInfos.size(); ++i)
	{
		auto& contour = contourInfos[i];
		if (processedMerging[i])
			continue;
		if (contour.markDeleted)
		{
			processedMerging[i] = true;
			continue;
		}

		auto bodyHalfLenthPix = bodyHalfLenthPix_; // default, used in testing scenarios
		if (cameraProjector_ != nullptr)
			cameraProjector_->distanceWorldToCamera(contour.contourCentroidWorld, bodyHalfLenth_);

		for (size_t neighInd = 0; neighInd < contourInfos.size(); ++neighInd)
		{
			auto& neigh = contourInfos[neighInd];
			if (neighInd == i) // skip current contour itself
				continue;
			if (neigh.markDeleted)
			{
				processedMerging[neighInd] = true;
				continue;
			}
			if (processedMerging[neighInd])
				continue;

			auto vec = neigh.contourCentroid - contour.contourCentroid;
			float dist = cv::norm(vec);

			if (dist < bodyHalfLenthPix)
			{
				auto color = cv::Scalar::all(255);

				mergeBuffer.setTo(0);

				contoursList[0] = contour.outlinePixels;
				cv::drawContours(mergeBuffer, contoursList, 0, color, CV_FILLED);

				contoursList[0] = neigh.outlinePixels;
				cv::drawContours(mergeBuffer, contoursList, 0, color, CV_FILLED);

				mergedBlobsContour.clear();
				mergeBlobs(contour.outlinePixels, neigh.outlinePixels, mergeBuffer, mergedBlobsContour, 5);

				// update two blobs infos
				contour.outlinePixels = mergedBlobsContour;
				fixContourInfo(contour, cameraProjector_.get());
				neigh.markDeleted = true;
				processedMerging[neighInd] = true;
			}
		}

		processedMerging[i] = true;
	}

#if PW_DEBUG
	cv::Mat mergedBlobs = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	for (size_t i = 0; i < contourInfos.size(); ++i)
	{
		CV_Assert(processedMerging[i]);
		auto& contour = contourInfos[i];
		if (contour.markDeleted)
			continue;
		contoursList[0] = contour.outlinePixels;
		cv::drawContours(mergedBlobs, contoursList, 0, cv::Scalar::all(255), CV_FILLED);
	}
#endif

	// remove elongated stripes created by lane markers and water 'blique'

#if PW_DEBUG
	cv::Mat imageNoSticks = cv::Mat_<uchar>::zeros(rows1, width);
#endif

	for (size_t i = 0; i < contourInfos.size(); ++i)
	{
		auto& contour = contourInfos[i];
		if (contour.markDeleted)
			continue;

		// TODO: number of outline pixels must be large
		if (contour.outlinePixels.size() < 6)
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

	// remove closely positioned blobs
	// meaning two swimmers can't be too close
	if (cameraProjector_ != nullptr)
	{
		for (size_t i = 0; i < contourInfos.size(); ++i)
		{
			auto& contour = contourInfos[i];
			if (contour.markDeleted)
				continue;

			for (size_t neighInd = i + 1; neighInd < contourInfos.size(); ++neighInd)
			{
				auto& neigh = contourInfos[neighInd];
				if (neigh.markDeleted)
					continue;
				tuple<float, int, int> minDist = twoBlobsClosestDistanceSqr(contour.outlinePixels, neigh.outlinePixels);
				float distSqr = get<0>(minDist);
				int minInd1 = get<1>(minDist);
				int minInd2 = get<2>(minDist);

				cv::Point posPix1 = contour.outlinePixels[minInd1];
				cv::Point posPix2 = neigh.outlinePixels[minInd2];

				cv::Point3f posWorld1 = cameraProjector_->cameraToWorld(posPix1);
				cv::Point3f posWorld2 = cameraProjector_->cameraToWorld(posPix2);

				auto vec = posWorld1 - posWorld2;
				auto distWorld = cv::norm(vec);

				const float minTwoSwimmersDist = 0.5; // min distance between two swimmers in meters
				if (distWorld < minTwoSwimmersDist)
				{
					neigh.markDeleted = true;
				}
			}
		}
	}

#if PW_DEBUG
	cv::Mat noCloseBlobs = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	for (size_t i = 0; i < contourInfos.size(); ++i)
	{
		CV_Assert(processedMerging[i]);
		auto& contour = contourInfos[i];
		if (contour.markDeleted)
			continue;
		contoursList[0] = contour.outlinePixels;
		cv::drawContours(noCloseBlobs, contoursList, 0, cv::Scalar::all(255), CV_FILLED);
	}
#endif

	// pupulate results

	cv::Mat curOutlineMat = cv::Mat::zeros(rows1, width, CV_8UC1); // minimal image extended to the size of the original image
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
		if (cameraProjector_ != nullptr)
			blob.CentroidWorld = cameraProjector_->cameraToWorld(blob.Centroid);
		else
			blob.CentroidWorld = cv::Point3f(-1, -1, -1);

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

void SwimmerDetector::trainLaneSeparatorClassifier(WaterClassifier& wc)
{
	auto svgFilter = "*.svg";
	auto skinMarkupColor = "#00FF00"; // green

	auto skinMarkupFiles = boost::filesystem::absolute("./data/LaneSeparator/").normalize();

	std::vector<cv::Vec3d> separatorPixels;
	std::vector<cv::Vec3d> nonSeparatorPixels;

	loadWaterPixels(skinMarkupFiles.string(), svgFilter, skinMarkupColor, separatorPixels, false);
	loadWaterPixels(skinMarkupFiles.string(), svgFilter, skinMarkupColor, nonSeparatorPixels, true, 5);

	// remove intersection of these two sets

	//auto colorCmpFun = [](const cv::Vec3d& x, const cv::Vec3d& y) -> bool {
	//	if (x(0) != y(0))
	//		return std::less<double>()(x(0), y(0));
	//	if (x(1) != y(1))
	//		return std::less<double>()(x(1), y(1));
	//	return std::less<double>()(x(2), y(2));
	//};
	//std::sort(begin(separatorPixels), end(separatorPixels), colorCmpFun);
	//std::sort(begin(nonSeparatorPixels), end(nonSeparatorPixels), colorCmpFun);

	//std::vector<cv::Vec3d> separatorPixelsNoCommon;
	//std::vector<cv::Vec3d> nonSeparatorPixelsNoCommon;
	//std::set_difference(begin(separatorPixels), end(separatorPixels), begin(nonSeparatorPixels), end(nonSeparatorPixels), back_inserter(separatorPixelsNoCommon), colorCmpFun);
	//std::set_difference(begin(nonSeparatorPixels), end(nonSeparatorPixels), begin(separatorPixels), end(separatorPixels), back_inserter(nonSeparatorPixelsNoCommon), colorCmpFun);

	//std::random_device rd;
	//std::mt19937 g(rd());
	//std::shuffle(begin(separatorPixelsNoCommon), end(separatorPixelsNoCommon), g);
	//std::shuffle(begin(nonSeparatorPixelsNoCommon), end(nonSeparatorPixelsNoCommon), g);

	//// normalize size of both sets
	//auto commonSize = std::min(separatorPixelsNoCommon.size(), nonSeparatorPixelsNoCommon.size());
	//separatorPixelsNoCommon.resize(commonSize);
	//nonSeparatorPixelsNoCommon.resize(commonSize);

	//
	cv::Mat separatorPixelsMat(separatorPixels);
	cv::Mat nonSeparatorPixelsMat(nonSeparatorPixels);

	wc.trainWater(separatorPixelsMat, nonSeparatorPixelsMat);
	wc.initCache();
}

void SwimmerDetector::trainSkinClassifier(WaterClassifier& wc)
{
	auto svgFilter = "*.svg";
	auto skinMarkupColor = "#00FF00"; // green
	auto skinMarkupColorEx = "#FFFF00"; // yellow

	auto skinMarkupFiles = boost::filesystem::absolute("./data/SkinMarkup/SkinPatch/").normalize();

	std::vector<cv::Vec3d> skinPixels;
	std::vector<cv::Vec3d> nonSkinPixels;

	loadWaterPixels(skinMarkupFiles.string(), svgFilter, skinMarkupColor, skinPixels, false);
	loadWaterPixels(skinMarkupFiles.string(), svgFilter, skinMarkupColorEx, nonSkinPixels, true);

	cv::Mat skinColorsMat(skinPixels);
	cv::Mat nonSkinColorsMat(nonSkinPixels);

	wc.trainWater(skinColorsMat, nonSkinColorsMat);
	wc.initCache();
}
