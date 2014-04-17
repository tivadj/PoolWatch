
#include <opencv2\highgui.hpp>

#include "SwimmingPoolObserver.h"
#include "WaterClassifier.h"
#include "algos1.h"

namespace HumanBodiesTestsNS
{
	using namespace std;
	using namespace cv;

	void testHumanBodiesDetection()
	{
		cv::FileStorage fs;
		if (!fs.open("1.yml", cv::FileStorage::READ))
			return;
		auto pWc = WaterClassifier::read(fs);
		WaterClassifier& wc = *pWc;

		cv::Mat i1 = cv::imread("data/MVI_3177_0127_640x476.png");

		cv::Mat_<uchar> i1WaterMask;
		classifyAndGetMask(i1, [&wc](const cv::Vec3d& pix) -> bool
		{
			return wc.predict(pix);
		}, i1WaterMask);

		//
		cv::Mat_<uchar> poolMask;
		getPoolMask(i1, i1WaterMask, poolMask);

		cv::Mat imageSimple;
		i1.copyTo(imageSimple, poolMask);

		vector<DetectedBlob> blobs;
		getHumanBodies(imageSimple, i1WaterMask, blobs);
	}

	void run()
	{
		testHumanBodiesDetection();
	}
}