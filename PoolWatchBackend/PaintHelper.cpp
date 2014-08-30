#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>

#include <boost/lexical_cast.hpp> // CV_RGB

#include "PaintHelper.h"
#include "MultiHypothesisBlobTracker.h"

namespace PoolWatch
{
	PaintHelper::PaintHelper()
	{
		getWellKnownColors(trackColors_);
	}

	void PaintHelper::getWellKnownColors(std::vector<cv::Scalar>& colors)
	{
		colors.push_back(CV_RGB(0, 255, 0));
		colors.push_back(CV_RGB(0, 0, 255));
		colors.push_back(CV_RGB(255, 0, 0));
		colors.push_back(CV_RGB(0, 255, 255)); // cyan
		colors.push_back(CV_RGB(255, 0, 255)); // magenta
		colors.push_back(CV_RGB(255, 255, 0)); // yellow
	}

	void PaintHelper::paintBlob(const DetectedBlob& blob, cv::Mat& image)
	{
		std::vector<cv::Point> pts(blob.OutlinePixels.rows);
		for (size_t i = 0; i < pts.size(); ++i)
			pts[i] = cv::Point(blob.OutlinePixels(i, 1), blob.OutlinePixels(i, 0));
		std::vector<std::vector<cv::Point>> ptsCln(1);
		ptsCln[0] = std::move(pts);

		auto blobColor = trackColors_[blob.Id % trackColors_.size()];
		cv::drawContours(image, ptsCln, 0, blobColor);

		auto idStr = boost::lexical_cast<std::string>(blob.Id);
		cv::putText(image, idStr, blob.Centroid, cv::FONT_HERSHEY_PLAIN, 0.7, blobColor);
	}

	void PaintHelper::paintTriangleHBase(cv::Mat& image, cv::Point center, float side, cv::Scalar color)
	{
		const float d = side/2;

		auto left = cv::Point(center.x - d, center.y + 2);
		auto right = cv::Point(center.x + d, center.y + 2);
		auto top = cv::Point(center.x, center.y - d);

		cv::line(image, left, right, color);
		cv::line(image, left, top, color);
		cv::line(image, right, top, color);
	}

	void PaintHelper::paintTrack(const TrackInfoHistory& track, int fromFrameOrd, int toFrameOrd, const cv::Scalar& color, const std::vector<std::vector<DetectedBlob>>& blobsPerFrame, cv::Mat& resultImage)
	{
		// limit to available observations

		int maxUpper = track.isFinished() ? track.LastAppearanceFrameIdx : (track.FirstAppearanceFrameIdx + track.Assignments.size() - 1);

		int localFromFrameOrd = fromFrameOrd;
		if (localFromFrameOrd > maxUpper)
			return;
		else if (localFromFrameOrd < track.FirstAppearanceFrameIdx)
			localFromFrameOrd = track.FirstAppearanceFrameIdx;

		int localToFrameOrd = toFrameOrd;
		if (localToFrameOrd < track.FirstAppearanceFrameIdx)
			return;
		if (localToFrameOrd < maxUpper) // do not show tracks, finished some time ago
			return;
		else if (localToFrameOrd > maxUpper)
			localToFrameOrd = maxUpper;

		assert(localFromFrameOrd <= localToFrameOrd);

		//
		CV_Assert(localFromFrameOrd >= track.FirstAppearanceFrameIdx);
		CV_Assert(localToFrameOrd >= track.FirstAppearanceFrameIdx);
		if (track.isFinished())
		{
			CV_Assert(localFromFrameOrd <= track.LastAppearanceFrameIdx);
			CV_Assert(localToFrameOrd <= track.LastAppearanceFrameIdx);
		}

		// mark the initial track observation with triangle
		{
			auto pChange = track.getTrackChangeForFrame(localFromFrameOrd);
			assert(pChange != nullptr);

			auto& cent = pChange->ObservationPosPixExactOrApprox;
			PoolWatch::PaintHelper::paintTriangleHBase(resultImage, cent, 6, color);
		}

		// mark the last track observation
		{
			auto pChange = track.getTrackChangeForFrame(localToFrameOrd);
			assert(pChange != nullptr);

			auto& cent = pChange->ObservationPosPixExactOrApprox;
			PoolWatch::PaintHelper::paintTriangleHBase(resultImage, cent, 6, color);
		}

		// draw track path in frames range of interest

		const int PointXNull = -1;
		cv::Point2f prevPoint(PointXNull, PointXNull);;
		for (int frameInd = localFromFrameOrd; frameInd <= localToFrameOrd; ++frameInd)
		{
			auto pChange = track.getTrackChangeForFrame(frameInd);
			assert(pChange != nullptr);

			auto cent = pChange->ObservationPosPixExactOrApprox;

			bool hasPrevPoint = prevPoint.x != PointXNull;
			if (hasPrevPoint)
				cv::line(resultImage, prevPoint, cent, color);

			prevPoint = cent;
		}

		// draw shape outline for the last frame

		auto pLastChange = track.getTrackChangeForFrame(toFrameOrd);
		if (pLastChange != nullptr)
		{
			const cv::Point2f& cent = pLastChange->ObservationPosPixExactOrApprox;

			if (pLastChange->ObservationInd >= 0)
			{
				const std::vector<DetectedBlob>& blobs = blobsPerFrame[localToFrameOrd];
				const DetectedBlob& obs = blobs[pLastChange->ObservationInd];

				//PoolWatch::PaintHelper pantHelper;
				//pantHelper.paintBlob(obs, resultImage);
				const cv::Mat_<int32_t>& outlinePixMat = obs.OutlinePixels; // [Nx2], N=number of points; (Y,X) per row

				// populate points array
				std::vector<cv::Point> outlinePoints(outlinePixMat.rows);
				for (int i = 0; i < outlinePixMat.rows; ++i)
					outlinePoints[i] = cv::Point(outlinePixMat(i, 1), outlinePixMat(i, 0));

				cv::polylines(resultImage, outlinePoints, true, color);

				// draw track id
				auto idStr = boost::lexical_cast<std::string>(pLastChange->FamilyId);
				cv::putText(resultImage, idStr, cent, cv::FONT_HERSHEY_PLAIN, 0.7, color);

				// mark the latest track observation with cross
				//cv::circle(resultImage, cent, 3, color);
				const int d = 3;
				cv::line(resultImage, cv::Point(cent.x - d, cent.y), cv::Point(cent.x + d, cent.y), color);
				cv::line(resultImage, cv::Point(cent.x, cent.y - d), cv::Point(cent.x, cent.y + d), color);
			}
			else
			{
				// mark estimated track position with circle
				cv::circle(resultImage, cent, 3, color);
			}
		}
	}

}