#include "stdafx.h"
#include <iostream>
#include <numeric>
#include <random>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> // imread
#include <opencv2/highgui/highgui_c.h> // CV_FOURCC

#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>

#include <QDir>

#include <CppUnitTest.h>

#include "PoolWatchFacade.h"
#include "algos1.h"
#include "SwimmingPoolObserver.h"
#include "TestingUtils.h"

namespace PoolWatchBackendUnitTests
{
	using namespace cv;
	using namespace std;
	using namespace PoolWatch;
	using namespace Microsoft::VisualStudio::CppUnitTestFramework;

	extern log4cxx::LoggerPtr log_;

	TEST_CLASS(SwimmerDetectorTests)
	{
		const char* ClassName = "SwimmerDetectorTests";
		TEST_METHOD_INITIALIZE(MethodInitialize)
		{
			PoolWatchBackendUnitTests_MethodInitilize();
		}
	public:
		void testDetectBlobs(const std::string& imagePath, int expectedBlobsCount)
		{
			auto outDir = initTestMethodLogFolder(ClassName, "parallelMovementTest");
			auto logFileScope = scopeLogFileAppenderNew(outDir);

			int bodyHalfLenthPix = 30; // two blobs with distance smaller than this are merged
			SwimmerDetector sd(bodyHalfLenthPix);

			cv::Mat image = cv::imread(imagePath);

			std::vector<DetectedBlob> blobs;
			std::vector<DetectedBlob> expectedBlobs;
			sd.getBlobs(image, expectedBlobs, blobs);

			Assert::AreEqual(1, expectedBlobsCount);
		}
		TEST_METHOD(mvi_3177_i2287_blueWomanLane3)
		{
			auto path = "data/SwimmerDetection/mvi_3177_i2287_blueWomanLane3.png";
			testDetectBlobs(path, 1);
		}
		TEST_METHOD(mvi_3177_i2267_magentaWomanLane4)
		{
			auto path = "data/SwimmerDetection/mvi_3177_i2267_magentaWomanLane4.png";
			testDetectBlobs(path, 1);
		}
		TEST_METHOD(mvi_3177_i2267_whaleLane2)
		{
			auto path = "data/SwimmerDetection/mvi_3177_i2267_whaleLane2.png";
			testDetectBlobs(path, 1);
		}
	};
}