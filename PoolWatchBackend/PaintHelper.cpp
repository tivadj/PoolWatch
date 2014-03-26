#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>

#include <boost/lexical_cast.hpp> // CV_RGB

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

}