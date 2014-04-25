#include "stdafx.h"
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include "CppUnitTest.h"
#include "HumanDetector.h"
#include "TestingUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace PoolWatchBackendUnitTests
{
	TEST_CLASS(BlobUtilsTests)
	{
		TEST_METHOD_INITIALIZE(MethodInitialize)
		{
			PoolWatchBackendUnitTests_MethodInitilize();
		}

		void testMergeTwoBlobs(const std::string& blobsFileSvg)
		{
			cv::Mat black;
			cv::Mat_<bool> imageBlobs;
			loadImageAndMask(blobsFileSvg, "#FFFFFF", black, imageBlobs);

			cv::Mat imageBlobsTmp = imageBlobs.clone();
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(imageBlobsTmp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

			Assert::AreEqual<int>(2, contours.size());

			std::vector<cv::Point> mergedBlob;
			mergeBlobs(contours[0], contours[1], imageBlobs, mergedBlob, 5);

			Assert::IsTrue(!mergedBlob.empty());
		}
	public:
		
		TEST_METHOD(MergeBlobs_TwoPrisms)
		{
			auto path = "data/tests/MergeBlobs/MergeBlobs_TwoPrisms.svg";
			testMergeTwoBlobs(path);
		}
		TEST_METHOD(MergeBlobs_RussianDoll)
		{
			auto path = "data/tests/MergeBlobs/MergeBlobs_RussianDoll.svg";
			testMergeTwoBlobs(path);
		}
		TEST_METHOD(MergeBlobs_TwoCrescents)
		{
			auto path = "data/tests/MergeBlobs/MergeBlobs_TwoCrescents.svg";
			testMergeTwoBlobs(path);
		}
		TEST_METHOD(MergeBlobs_PrismTipAndWall)
		{
			auto path = "data/tests/MergeBlobs/MergeBlobs_PrismTipAndWall.svg";
			testMergeTwoBlobs(path);
		}

	};
}