#include <vector>
#include <algorithm>
#include <random>
#include <opencv2/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>

#include "VisualObservation.h"
#include "WaterClassifier.h"
#include "algos1.h"
#include "SvgImageMaskSerializer.h"

namespace SkinClassifierTestsNS
{
	using namespace std;
	using namespace cv;

	const char* SkinWnd = "skin";
	float skinThreshValue = -9;

	void trainAndWriteSkinClassifier(WaterClassifier& wc)
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

	void testSkinClassifier()
	{
		std::unique_ptr<WaterClassifier> pc;
		cv::FileStorage fs;
		if (!fs.open("skin_clasifier.yml", cv::FileStorage::READ))
		{
			pc = make_unique<WaterClassifier>(10, cv::EM::COV_MAT_SPHERICAL);
			trainAndWriteSkinClassifier(*pc);

			if (!fs.open("skin_clasifier.yml", cv::FileStorage::WRITE))
				return;
			pc->write(fs);
			fs.release();
		}
		else
		{
			pc = WaterClassifier::read(fs);
		}

		WaterClassifier& wc = *pc;

		cv::Mat image = cv::imread("../../output/mvi3177_blueWomanLane3_Frame8.png");
		if (image.empty())
			return;

		auto skinClassifFun = [&wc](const cv::Vec3d& pix) -> bool
		{
			return wc.predict(pix);
		};

		cv::Mat_<uchar> i1WaterMask;
		classifyAndGetMask(image, skinClassifFun, i1WaterMask);

		cv::Mat imageClassified;
		image.copyTo(imageClassified, i1WaterMask);
		
		//
		auto estimFun = [&wc](const cv::Vec3d& pix) -> double
		{
			return wc.computeOne(pix, true);
		};
		cv::Mat_<double> estimationMask;
		estimateClassifier(image, estimFun, estimationMask);

		auto skinClassifAbsFun = [&wc](const cv::Vec3d& pix) -> bool
		{
			double val = wc.computeOne(pix, true);
			float thr = skinThreshValue;
			return val > thr;
		};

		cv::Mat_<uchar> maskAbs;
		classifyAndGetMask(image, skinClassifAbsFun, maskAbs);
		cv::Mat imageClassifiedAbs;
		image.copyTo(imageClassifiedAbs, maskAbs);


		cv::imshow(SkinWnd, imageClassifiedAbs);
	}

	void trainAndWriteLaneSeparatorClassifier(WaterClassifier& wc)
	{
		auto svgFilter = "*.svg";
		auto skinMarkupColor = "#00FF00"; // green

		auto skinMarkupFiles = boost::filesystem::absolute("./data/LaneSeparator/").normalize();

		std::vector<cv::Vec3d> separatorPixels;
		std::vector<cv::Vec3d> nonSeparatorPixels;

		loadWaterPixels(skinMarkupFiles.string(), svgFilter, skinMarkupColor, separatorPixels, false);
		loadWaterPixels(skinMarkupFiles.string(), svgFilter, skinMarkupColor, nonSeparatorPixels, true, 5);

		// remove intersection of these two sets

		auto colorCmpFun = [](const cv::Vec3d& x, const cv::Vec3d& y) -> bool {
			if (x(0) != y(0))
				return std::less<double>()(x(0), y(0));
			if (x(1) != y(1))
				return std::less<double>()(x(1), y(1));
			return std::less<double>()(x(2), y(2));
		};
		std::sort(begin(separatorPixels), end(separatorPixels), colorCmpFun);
		std::sort(begin(nonSeparatorPixels), end(nonSeparatorPixels), colorCmpFun);

		std::vector<cv::Vec3d> separatorPixelsNoCommon;
		std::vector<cv::Vec3d> nonSeparatorPixelsNoCommon;
		std::set_difference(begin(separatorPixels), end(separatorPixels), begin(nonSeparatorPixels), end(nonSeparatorPixels), back_inserter(separatorPixelsNoCommon), colorCmpFun);
		std::set_difference(begin(nonSeparatorPixels), end(nonSeparatorPixels), begin(separatorPixels), end(separatorPixels), back_inserter(nonSeparatorPixelsNoCommon), colorCmpFun);

		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(begin(separatorPixelsNoCommon), end(separatorPixelsNoCommon), g);
		std::shuffle(begin(nonSeparatorPixelsNoCommon), end(nonSeparatorPixelsNoCommon), g);

		//// normalize size of both sets
		//auto commonSize = std::min(separatorPixelsNoCommon.size(), nonSeparatorPixelsNoCommon.size());
		//separatorPixelsNoCommon.resize(commonSize);
		//nonSeparatorPixelsNoCommon.resize(commonSize);

		//
		cv::Mat separatorPixelsMat(separatorPixelsNoCommon);
		cv::Mat nonSeparatorPixelsMat(nonSeparatorPixelsNoCommon);

		wc.trainWater(separatorPixelsMat, nonSeparatorPixelsMat);
		wc.initCache();
	}

	void testLaneSeparatorClassifier()
	{
		std::unique_ptr<WaterClassifier> pc;

		cv::FileStorage fs;
		if (!fs.open("laneSeparator_clasifier.yml", cv::FileStorage::READ))
		{
			pc = make_unique<WaterClassifier>(6, cv::EM::COV_MAT_SPHERICAL);

			trainAndWriteLaneSeparatorClassifier(*pc);
			if (!fs.open("laneSeparator_clasifier.yml", cv::FileStorage::WRITE))
				return;
			pc->write(fs);
			fs.release();
		}
		else
		{
			pc.swap(WaterClassifier::read(fs));
		}

		WaterClassifier& wc = *pc;

		cv::Mat image = cv::imread("../../output/mvi3177_blueWomanLane3_Frame8.png");
		if (image.empty())
			return;

		auto sepClassifFun = [&wc](const cv::Vec3d& pix) -> bool
		{
			return wc.predict(pix);
		};

		cv::Mat_<uchar> i1WaterMask;

		classifyAndGetMask(image, sepClassifFun, i1WaterMask);

		cv::Mat imageClassified;
		image.copyTo(imageClassified, i1WaterMask);

		cv::imshow(SkinWnd, imageClassified);
	}
	void run()
	{
		cv::namedWindow(SkinWnd);
		int value = 9;
		cv::createTrackbar("tb1", SkinWnd, &value, 150, [](int pos, void* userdata)
		{
			skinThreshValue = - pos / 10.0;
			cout << skinThreshValue << endl;
			//testSkinClassifier();
			testLaneSeparatorClassifier();
		});

		//testSkinClassifier();

		testLaneSeparatorClassifier();
		for (;;)
		{
			if (cv::waitKey(5) == 27)
				return;
		}
	}
}