#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp> // imread

#include "VisualObservation.h"
#include "algos1.h"

namespace PoolBoundaryDetectorTestsNS
{
	using namespace cv;

	void testPoolBoundary()
	{
		cv::Mat i1 = cv::imread("data/MVI_3177_0127_640x476.png");
		
		// find water mask
		cv::FileStorage fs;
		if (!fs.open("1.yml", cv::FileStorage::READ))
			return;
		auto pWc = WaterClassifier::read(fs);
		WaterClassifier& wc = *pWc;

		cv::Mat_<uchar> i1WaterMask;
		classifyAndGetMask(i1, [&wc](const cv::Vec3d& pix) -> bool
		{
			return wc.predict(pix);
		}, i1WaterMask);

		cv::Mat_<uchar> poolMask;
		getPoolMask(i1, i1WaterMask, poolMask);

		cv::Mat i2;
		i1.copyTo(i2, poolMask);
	}

	void run()
	{
		testPoolBoundary();
	}
}