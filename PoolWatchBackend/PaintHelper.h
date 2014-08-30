#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>

#include "SwimmingPoolObserver.h"

namespace PoolWatch
{
	struct PW_EXPORTS PaintHelper
	{
	private:
		std::vector<cv::Scalar> trackColors_;

	public:
		static void getWellKnownColors(std::vector<cv::Scalar>& trackColors);

		PaintHelper();

		void paintBlob(const DetectedBlob& blob, cv::Mat& image);
		static void paintTriangleHBase(cv::Mat& image, cv::Point center, float side, cv::Scalar color);
		static void paintTrack(const TrackInfoHistory& track, int fromFrameOrd, int toFrameOrd, const cv::Scalar& color, const std::vector<std::vector<DetectedBlob>>& blobsPerFrame, cv::Mat& resultImage);
	};
}