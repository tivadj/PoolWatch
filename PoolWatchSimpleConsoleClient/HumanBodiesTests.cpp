
#include <opencv2\highgui.hpp>
#include <boost/filesystem.hpp>

#include "SwimmingPoolObserver.h"
#include "WaterClassifier.h"
#include "algos1.h"
#include "HumanDetector.h"

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
		vector<DetectedBlob> expectedBlobs;
		getHumanBodies(imageSimple, i1WaterMask, expectedBlobs, blobs);
	}

	void testHumanDetectorSimple()
	{
		auto p1 = boost::filesystem::current_path();
		//cv::Mat image = cv::imread("data/MVI_3177_0127_640x476.png");
		cv::Mat image = cv::imread("../../output/mvi3177_blueWomanLane3_Frame8.png");

		SwimmerDetector sd;

		vector<DetectedBlob> blobs;
		vector<DetectedBlob> expectedBlobs;
		sd.getBlobs(image, expectedBlobs, blobs);
	}

	void run()
	{
		//testHumanBodiesDetection();
		testHumanDetectorSimple();
	}
}