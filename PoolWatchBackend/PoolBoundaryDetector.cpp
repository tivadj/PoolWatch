#include <vector>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include "VisualObservation.h"
#include "algos1.h"
#include <numeric>

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

struct SegmentInfo
{
	cv::Point2f x1;
	cv::Point2f x2;
	float theta;
	//float distanceLineToVanishingPoint = -1;
	std::vector<int>* pLaneGroup = nullptr;
};

void twoPointsToLineInfos(const std::vector<cv::Vec4i>& lines, std::vector<SegmentInfo>& result)
{
	result.reserve(lines.size());
	for (auto& twoPoints : lines)
	{
		auto p1 = cv::Point2f(twoPoints[0], twoPoints[1]);
		auto p2 = cv::Point2f(twoPoints[2], twoPoints[3]);

		float dy = p2.y - p1.y;
		float dx = p2.x - p1.x;

		// take theta in 1 and 2nd quadrant
		if (dy < 0)
		{
			dx = -dx;
			dy = -dy;
		}

		SegmentInfo seg;
		seg.x1 = p1;
		seg.x2 = p2;
		seg.theta = std::atan2f(dy, dx);
		result.push_back(seg);
	}
}

// maxDistToVanishPoint=line is valid if the distance from the vanishing point to the line is less than this value.
void filterOutInvalidDirectionLines(const std::vector<SegmentInfo>& lines, const cv::Point2f& vanishPoint, float thetaStep, float maxDistToVanishPoint, std::vector<SegmentInfo>& outValidLines)
{
	outValidLines.reserve(lines.size());
	float maxDist = 0;
	float minDist = 9999;

	std::vector<float> distVanishToLine;
	distVanishToLine.reserve(lines.size());
	for (size_t i = 0; i < lines.size(); ++i)
	{
		auto line = lines[i];
		float dist = -1;
		bool distOp = PoolWatch::distLinePoint(line.x1, line.x2, vanishPoint, dist);
		CV_Assert(distOp && "Two equal points can not specify the line");

		maxDist = std::max(maxDist, dist);
		minDist = std::min(minDist, dist);

		distVanishToLine.push_back(dist);
	}

	std::sort(std::begin(distVanishToLine), std::end(distVanishToLine));

	//float maxDistPercToVanishPointNew = 0.75;
	//float ind = (size_t)(lines.size() * maxDistPercToVanishPointNew);
	//float maxDistToVanishPointNew = distVanishToLine[ind];

	for (size_t i = 0; i < lines.size(); ++i)
	{
		auto line = lines[i];
		float dist = -1;
		bool distOp = PoolWatch::distLinePoint(line.x1, line.x2, vanishPoint, dist);
		CV_Assert(distOp && "Two equal points can not specify the line");

		// line is valid if it passes the vanishing point
		bool isValid = dist < maxDistToVanishPoint;
		if (isValid)
			outValidLines.push_back(line);
	}
}

bool findVanishingPoint(const std::vector<SegmentInfo>& lines, cv::Point2f& outVanishPoint, bool medianOrAverage, float skipMarginPerc)
{
	CV_DbgAssert(lines.size() >= 2 && "There must be at least two lines to detect the vanishing point");

	size_t pairsCount = lines.size() * (lines.size() - 1) / 2;

	struct VanishPointCandidate
	{
		bool IsValid = true;
		cv::Point2f Point;
	};

	std::vector<VanishPointCandidate> vanishPointCandidates;
	vanishPointCandidates.reserve(pairsCount);

	// find the set of intersections
	for (size_t i1 = 0; i1 < lines.size(); ++i1)
	{
		const SegmentInfo& line1 = lines[i1];

		float dist = -1;
		bool distOp = PoolWatch::distLinePoint(line1.x1, line1.x2, cv::Point2f(-896, 95), dist);
		CV_Assert(distOp && "Two equal points can not specify the line");

		for (size_t i2 = i1 + 1; i2 < lines.size(); ++i2)
		{
			const SegmentInfo& line2 = lines[i2];

			cv::Point2f cross;
			bool crossOp = PoolWatch::intersectLines(line1.x1, line1.x2, line2.x1, line2.x2, cross);
			if (crossOp)
			{
				VanishPointCandidate pnt;
				pnt.Point = cross;
				vanishPointCandidates.push_back(pnt);
			}
		}
	}

	CV_DbgAssert(!vanishPointCandidates.empty());

	auto cmpByXFun = [](VanishPointCandidate& p1, VanishPointCandidate& p2) { return p1.Point.x < p2.Point.x; };

	if (medianOrAverage)
	{
		std::sort(std::begin(vanishPointCandidates), std::end(vanishPointCandidates), cmpByXFun);

		// take median point in OX direction
		int midInd = vanishPointCandidates.size() / 2;
		nth_element(std::begin(vanishPointCandidates), std::begin(vanishPointCandidates) + midInd, std::end(vanishPointCandidates), cmpByXFun);

		//
		auto cmpByYFun = [](VanishPointCandidate& p1, VanishPointCandidate& p2) { return p1.Point.y < p2.Point.y; };
		auto vanishPointCandidatesY = vanishPointCandidates;
		std::sort(std::begin(vanishPointCandidatesY), std::end(vanishPointCandidatesY), cmpByYFun);

		outVanishPoint = vanishPointCandidates[midInd].Point;
	}
	else
	{
		size_t margin = static_cast<size_t>(skipMarginPerc * vanishPointCandidates.size());

		// exclude outliers in OX direction
		std::sort(std::begin(vanishPointCandidates), std::end(vanishPointCandidates), cmpByXFun);
		
		for (size_t i = 0; i < margin; ++i)
			vanishPointCandidates[i].IsValid = false;
		for (size_t i = vanishPointCandidates.size() - margin; i < vanishPointCandidates.size(); ++i)
			vanishPointCandidates[i].IsValid = false;

		// exclude outliers in OY direction
		auto cmpByYFun = [](VanishPointCandidate& p1, VanishPointCandidate& p2) { return p1.Point.y < p2.Point.y; };
		std::sort(std::begin(vanishPointCandidates), std::end(vanishPointCandidates), cmpByYFun);
		for (size_t i = 0; i < margin; ++i)
			vanishPointCandidates[i].IsValid = false;
		for (size_t i = vanishPointCandidates.size() - margin; i < vanishPointCandidates.size(); ++i)
			vanishPointCandidates[i].IsValid = false;

		float sumX = 0;
		float sumY = 0;
		int sumCount = 0;		
		
		for (size_t i = 0; i < vanishPointCandidates.size(); ++i)
		{
			const auto& p = vanishPointCandidates[i];
			if (!p.IsValid)
				continue;
			sumX += p.Point.x;
			sumY += p.Point.y;
			sumCount++;
		}
		outVanishPoint = cv::Point2f(sumX / sumCount, sumY / sumCount);
	}
	return true;
}

// Finds the maximal distance between two segments.
float twoSegmentsDist(const SegmentInfo& seg1, const SegmentInfo& seg2)
{
	float dist1 = -1;
	bool distOp = PoolWatch::distLinePoint(seg1.x1, seg1.x2, seg2.x1, dist1);
	CV_Assert(distOp && "Two equal points can not specify the line");

	float dist2 = -1;
	distOp = PoolWatch::distLinePoint(seg1.x1, seg1.x2, seg2.x2, dist2);
	CV_Assert(distOp && "Two equal points can not specify the line");

	float dist3 = -1;
	distOp = PoolWatch::distLinePoint(seg2.x1, seg2.x2, seg1.x1, dist3);
	CV_Assert(distOp && "Two equal points can not specify the line");

	float dist4 = -1;
	distOp = PoolWatch::distLinePoint(seg2.x1, seg2.x2, seg1.x2, dist4);
	CV_Assert(distOp && "Two equal points can not specify the line");

	float maxDist = std::max({ dist1, dist2, dist3, dist4 });
	return maxDist;
}

void groupCloseLineSegments(std::vector<SegmentInfo>& lines, float maxSwimLaneWidth, std::vector<std::unique_ptr<std::vector<int>>>& swimLaneLineGroups)
{
	struct TwoSegmentDist
	{
		int SegInd1;
		int SegInd2;
		float MaxDist;
	};

	// calculate distance between each pair of lines

	std::vector<TwoSegmentDist> segmentsCrossDist;
	for (size_t i1 = 0; i1 < lines.size(); ++i1)
	{
		auto& seg1 = lines[i1];
		for (size_t i2 = i1+1; i2 < lines.size(); ++i2)
		{
			auto& seg2 = lines[i2];
			float maxDist = twoSegmentsDist(seg1, seg2);

			TwoSegmentDist distItem;
			distItem.SegInd1 = i1;
			distItem.SegInd2 = i2;
			distItem.MaxDist = maxDist;
			segmentsCrossDist.push_back(distItem);
		}
	}

	std::sort(std::begin(segmentsCrossDist), std::end(segmentsCrossDist), [](TwoSegmentDist& d1, TwoSegmentDist& d2) { return d1.MaxDist < d2.MaxDist; });

	// join close lines into groups

	for (size_t i = 0; i < segmentsCrossDist.size(); ++i)
	{
		TwoSegmentDist& d1 = segmentsCrossDist[i];
		if (d1.MaxDist > maxSwimLaneWidth)
			break;

		SegmentInfo& seg1 = lines[d1.SegInd1];
		SegmentInfo& seg2 = lines[d1.SegInd2];
		if (seg1.pLaneGroup == nullptr && seg2.pLaneGroup == nullptr)
		{
			// both segments have no owning group

			auto segmentsGroup = std::make_unique<std::vector<int>>();

			segmentsGroup->push_back(d1.SegInd1);
			seg1.pLaneGroup = segmentsGroup.get();

			segmentsGroup->push_back(d1.SegInd2);
			seg2.pLaneGroup = segmentsGroup.get();
			swimLaneLineGroups.push_back(std::move(segmentsGroup));
		}
		else if ((seg1.pLaneGroup != nullptr && seg2.pLaneGroup == nullptr) ||
			(seg1.pLaneGroup == nullptr && seg2.pLaneGroup != nullptr))
		{
			int segInd = -1;
			SegmentInfo* pSeg = nullptr;
			std::vector<int>* pGroup = nullptr;
			if (seg1.pLaneGroup != nullptr)
			{
				pGroup = seg1.pLaneGroup;
				segInd = d1.SegInd2;
				pSeg = &seg2;
			}
			else
			{
				pGroup = seg2.pLaneGroup;
				segInd = d1.SegInd1;
				pSeg = &seg1;
			}
			CV_DbgAssert(segInd != -1);
			CV_DbgAssert(pSeg != nullptr);
			CV_DbgAssert(pGroup != nullptr);

			pGroup->push_back(segInd);
			pSeg->pLaneGroup = pGroup;
		}
		else
		{
			// bot segments has the owning group

			auto pGroupToKeep = seg1.pLaneGroup;
			CV_DbgAssert(pGroupToKeep != nullptr);
			
			auto pGroupToVanish = seg2.pLaneGroup;
			CV_DbgAssert(pGroupToVanish != nullptr);

			if (pGroupToKeep != pGroupToVanish)
			{
				// put group2 into group1
				for (int segInd : *pGroupToVanish)
				{
					pGroupToKeep->push_back(segInd);

					SegmentInfo& groupItem = lines[segInd];
					groupItem.pLaneGroup = pGroupToKeep;
				}

				// note, empty group group2 will be removed later
				pGroupToVanish->clear();
			}
		}
	}

	// remove empty groups
	auto it = std::remove_if(std::begin(swimLaneLineGroups), std::end(swimLaneLineGroups), [](std::unique_ptr<std::vector<int>>& group) { return group->empty(); });
	swimLaneLineGroups.erase(it, std::end(swimLaneLineGroups));
}

void buildSwimLaneContours(const std::vector<SegmentInfo>& lines, const std::vector<std::unique_ptr<std::vector<int>>>& swimLaneLineGroups, std::vector<std::vector<cv::Point2f>>& outSwimLanes)
{
	std::vector<cv::Point2f> lanePointsAll;
	std::vector<cv::Point2f> laneHullPoint;

	// if the line has no other close lines, then it is not associated with the group
	// we may reconstruct a possible lane on top of it
	// for now, just ignore such lines

	for (const std::unique_ptr<std::vector<int>>& linesGroup : swimLaneLineGroups)
	{
		lanePointsAll.clear();
		for (int segInd : *linesGroup)
		{
			const SegmentInfo& seg = lines[segInd];
			lanePointsAll.push_back(seg.x1);
			lanePointsAll.push_back(seg.x2);
		}

		cv::convexHull(lanePointsAll, laneHullPoint, false, true);

		outSwimLanes.push_back(laneHullPoint);
	}
}

std::tuple<bool, std::string> getSwimLanes(const cv::Mat& image, std::vector<std::vector<cv::Point2f>>& swimLanes)
{
	//// find water mask

	//cv::FileStorage fs;
	//if (!fs.open("cl_water.yml", cv::FileStorage::READ))
	//{
	//	return make_tuple(false, "Can't find saved water classifier (try to change the working directory)");
	//}
	//auto waterClassifier = WaterClassifier::read(fs);

	//cv::Mat_<uchar> waterMask;
	//classifyAndGetMask(image, [&](const cv::Vec3d& pix) -> bool
	//{
	//	//bool b1 = wc.predict(pix);
	//	bool b2 = waterClassifier->predictFloat(cv::Vec3f(pix[0], pix[1], pix[2]));
	//	//assert(b1 == b2);
	//	return b2;
	//}, waterMask);

	//cv::bitwise_not(waterMask, waterMask);
	//cv::Mat_<uchar>& lanesMask = waterMask;

	cv::Mat_<uchar> imageGray;
	cv::cvtColor(image, imageGray, CV_BGR2GRAY);
	cv::Mat_<uchar> lanesMask;
	cv::threshold(imageGray, lanesMask, 1, 255, cv::THRESH_BINARY);

	cv::Mat_<uchar> edgesMask(image.rows, image.cols);
	float threshold1 = 50;
	float threshold2 = 200;
	int apertureSize = 3;
	bool L2gradient = false;
	cv::Canny(lanesMask, edgesMask, threshold1, threshold2, apertureSize, L2gradient);

	//
	vector<cv::Vec4i> lines;
	int thresh = 100;
	float rhoStep = 1;
	float thetaStep = CV_PI / 180;
	float minLineLength = 120;
	float maxLineGapReliable = 10; // allows to find reliable lines
	cv::HoughLinesP(edgesMask, lines, rhoStep, thetaStep, thresh, minLineLength, maxLineGapReliable);

	cv::Mat imageWithLinesRgbMatRelLines(image.rows, image.cols, CV_8UC3);
	image.copyTo(imageWithLinesRgbMatRelLines);
	std::array<char, 10> strBuf;
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
		line(imageWithLinesRgbMatRelLines, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 1, CV_AA);

		float dx = l[2] - l[0];
		float dy = l[3] - l[1];
		float perc = (rand() % 100) / 100.0f;
		auto mid = cv::Point(l[0], l[1]);
		mid.x += dx * perc;
		mid.y += dy * perc;
		_itoa_s(i, strBuf.data(), strBuf.size(), 10);
		cv::putText(imageWithLinesRgbMatRelLines, strBuf.data(), mid, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
	}

	vector<SegmentInfo> lineInfosRel;
	twoPointsToLineInfos(lines, lineInfosRel);

	// note, we need the vanishing point just to filter out false direction lines
	float skipMarginPerc = 0.25f; // percent of outliers from each margin side
	cv::Point2f vanishPoint;
	bool vanishPointOp = findVanishingPoint(lineInfosRel, vanishPoint, true, skipMarginPerc);
	
	//
	float maxLineGap = 100; // keep it big so that segments of different color of the same lane are connected
	cv::HoughLinesP(edgesMask, lines, rhoStep, thetaStep, thresh, minLineLength, maxLineGap);

	cv::Mat imageWithLinesRgbMatAllLines(image.rows, image.cols, CV_8UC3);
	image.copyTo(imageWithLinesRgbMatAllLines);
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
		line(imageWithLinesRgbMatAllLines, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 1, CV_AA);

		float dx = l[2] - l[0];
		float dy = l[3] - l[1];
		float perc = (rand() % 100) / 100.0f;
		auto mid = cv::Point(l[0], l[1]);
		mid.x += dx * perc;
		mid.y += dy * perc;
		_itoa_s(i, strBuf.data(), strBuf.size(), 10);
		cv::putText(imageWithLinesRgbMatAllLines, strBuf.data(), mid, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
	}

	vector<SegmentInfo> lineInfos;
	twoPointsToLineInfos(lines, lineInfos);

	//// note, we need the vanishing point just to filter out false direction lines
	//float skipMarginPerc = 0.25f; // percent of outliers from each margin side
	//cv::Point2f vanishPoint;
	//bool vanishPointOp = findVanishingPoint(lineInfos, vanishPoint, true, skipMarginPerc);

	// find max distance from vanishing point to any end of each direction segment
	float workingRad = 0;
	for (const auto& seg : lineInfos)
	{
		float dist = cv::norm(seg.x1 - vanishPoint);
		if (dist > workingRad)
			workingRad = dist;
		dist = cv::norm(seg.x2 - vanishPoint);
		if (dist > workingRad)
			workingRad = dist;
	}

	float maxLaneWidth = image.cols * std::sinf(thetaStep) * 2;
	float maxDistToVanishPoint = workingRad * std::sinf(thetaStep) * 2;

	std::vector<SegmentInfo> validLines;
	filterOutInvalidDirectionLines(lineInfos, vanishPoint, thetaStep, maxDistToVanishPoint, validLines);

	cv::Mat imageWithLinesRgbMatValidLines(image.rows, image.cols, CV_8UC3);
	image.copyTo(imageWithLinesRgbMatValidLines);
	for (size_t i = 0; i < validLines.size(); i++)
	{
		const SegmentInfo& seg = validLines[i];
		line(imageWithLinesRgbMatValidLines, seg.x1, seg.x2, cv::Scalar(0, 0, 255), 1, CV_AA);

		float dx = seg.x2.x - seg.x1.x;
		float dy = seg.x2.y - seg.x1.y;
		float perc = (rand() % 100) / 100.0f;
		auto mid = seg.x1;
		mid.x += dx * perc;
		mid.y += dy * perc;
		_itoa_s(i, strBuf.data(), strBuf.size(), 10);
		cv::putText(imageWithLinesRgbMatValidLines, strBuf.data(), mid, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
	}

	//

	cv::Point2f vanishPointAvg;
	vanishPointOp = findVanishingPoint(validLines, vanishPointAvg, false, skipMarginPerc);

	std::vector<SegmentInfo> validLines2;
	filterOutInvalidDirectionLines(lineInfos, vanishPointAvg, thetaStep, maxDistToVanishPoint, validLines2);

	cv::Mat imageWithLinesRgbMatValidLines2(image.rows, image.cols, CV_8UC3);
	image.copyTo(imageWithLinesRgbMatValidLines2);
	for (size_t i = 0; i < validLines2.size(); i++)
	{
		const SegmentInfo& seg = validLines2[i];
		line(imageWithLinesRgbMatValidLines2, seg.x1, seg.x2, cv::Scalar(0, 0, 255), 1, CV_AA);

		float dx = seg.x2.x - seg.x1.x;
		float dy = seg.x2.y - seg.x1.y;
		float perc = (rand() % 100) / 100.0f;
		auto mid = seg.x1;
		mid.x += dx * perc;
		mid.y += dy * perc;
		_itoa_s(i, strBuf.data(), strBuf.size(), 10);
		cv::putText(imageWithLinesRgbMatValidLines2, strBuf.data(), mid, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
	}

	//

	const float fixFactor2 = 0.7;
	float maxSwimLaneWidth = maxLaneWidth * fixFactor2;
	std::vector<std::unique_ptr<std::vector<int>>> swimLaneLineGroups;
	
	for (auto& line : validLines)
		line.pLaneGroup = nullptr;

	groupCloseLineSegments(validLines, maxSwimLaneWidth, swimLaneLineGroups);

	//
	buildSwimLaneContours(validLines, swimLaneLineGroups, swimLanes);

	//
	cv::Mat imageWithLinesRgbSwimLanes(image.rows, image.cols, CV_8UC3);
	image.copyTo(imageWithLinesRgbSwimLanes);
	
	std::vector<std::vector<cv::Point2i>> swimLanesInt(1);
	for (size_t i = 0; i < swimLanes.size(); ++i)
	{
		std::vector<cv::Point2i> contour(swimLanes[i].size());
		for (size_t j = 0; j < swimLanes[i].size(); ++j)
			contour[j] = swimLanes[i][j];

		swimLanesInt[0] = contour;
		cv::drawContours(imageWithLinesRgbSwimLanes, swimLanesInt, 0, cv::Scalar(0, 255, 255), 1);
	}

	return make_tuple(true, std::string());
}