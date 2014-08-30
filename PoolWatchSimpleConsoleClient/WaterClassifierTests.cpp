#include <functional>
#include <iostream>
#include <chrono> // std::chrono::system_clock
#include <array>

#include <opencv2\core.hpp>
#include <opencv2\ml.hpp>
#include <opencv2\highgui.hpp>

#include <ppl.h>
#include <amp.h>
#include <amp_graphics.h>

#include <boost/thread/tss.hpp>

#include "WaterClassifier.h"
#include "../PoolWatchBackend/algos_amp.cpp"
#include "algos1.h"
#include "SvgImageMaskSerializer.h"

namespace WaterClassifierTestsNS
{
	using namespace concurrency; // amp
	using namespace concurrency::graphics; // float_3
	using namespace Concurrency;
	using namespace cv;
	using namespace std;

	void sortAndDistinct(std::vector<cv::Vec3d>& pixels)
	{
		auto opLess = [](cv::Vec3d x, cv::Vec3d y)
		{
			std::less<double> ld;

			if (x[0] != y[0])
				return ld(x[0], y[0]);
			if (x[1] != y[1])
				return ld(x[1], y[1]);

			return ld(x[2], y[2]);
		};
		std::sort(begin(pixels), end(pixels), opLess);
		auto newEnd = std::unique(begin(pixels), end(pixels));

		pixels.resize(std::distance(begin(pixels), newEnd));
	}

	void trainWaterClassifier(WaterClassifier& wc)
	{
		auto waterMarkupFiles = "../../dinosaur/waterMarkup/";
		auto svgFilter = "*.svg";

		auto waterMarkupColor = "#0000FF";
		std::vector<cv::Vec3d> waterPixels;
		loadWaterPixels(waterMarkupFiles, svgFilter, waterMarkupColor, waterPixels);

		auto nonWaterMarkupColor = "#FFFF00";
		std::vector<cv::Vec3d> nonWaterPixels;
		loadWaterPixels(waterMarkupFiles, svgFilter, nonWaterMarkupColor, nonWaterPixels);

		//

		sortAndDistinct(waterPixels);
		sortAndDistinct(nonWaterPixels);

		cv::Mat waterColorsMat(waterPixels);
		cv::Mat nonWaterColorsMat(nonWaterPixels);
		wc.trainWater(waterColorsMat, nonWaterColorsMat);
	}

	void trainAndWriteWaterClassifier()
	{
		WaterClassifier wc(6, cv::EM::COV_MAT_SPHERICAL);
		trainWaterClassifier(wc);

		cv::FileStorage fs;
		if (!fs.open("1.yml", cv::FileStorage::WRITE))
			return;
		wc.write(fs);
	}

	void readAndTestWaterClassifier()
	{
		cv::FileStorage fs;
		if (!fs.open("1.yml", cv::FileStorage::READ))
			return;
		auto pWc = WaterClassifier::read(fs);
		WaterClassifier& wc = *pWc;

		//
		cv::Mat i1 = cv::imread("data/MVI_3177_0127_640x476.png");

		auto waterClassifFun = [&wc](const cv::Vec3d& pix) -> bool
		{
			return wc.predict(pix);
		};

		cv::Mat_<uchar> i1WaterMask;

		typedef std::chrono::system_clock Clock;
		std::chrono::time_point<Clock> now1 = Clock::now();
		classifyAndGetMask(i1, waterClassifFun, i1WaterMask);
		//classifyAndGetMaskParWorkPerPixel(i1, waterClassifFunPure, i1WaterMask);
		//classifyAndGetMaskParWorkLine(i1, waterClassifFunPure, i1WaterMask);
		//classifyAndGetMaskAmpFloat(i1, wc, i1WaterMask);

		std::chrono::time_point<Clock> now2 = Clock::now();

		auto elapsedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now2 - now1).count();
		std::cout << "elapsedMilliseconds=" << elapsedMilliseconds << endl;

		cv::Mat i2;
		i1.copyTo(i2, i1WaterMask);
	}


	void run()
	{
		//trainAndWriteWaterClassifier();
		readAndTestWaterClassifier();
	}
}