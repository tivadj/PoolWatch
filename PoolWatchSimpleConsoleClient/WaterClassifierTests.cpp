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
#include <boost/filesystem.hpp>

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

	struct PixelLexicographicalLess
	{
		bool operator()(const cv::Vec3d& x, const cv::Vec3d& y) const
		{
			std::less<double> ld;

			if (x[0] != y[0])
				return ld(x[0], y[0]);
			if (x[1] != y[1])
				return ld(x[1], y[1]);

			return ld(x[2], y[2]);
		}
	};

	void sortAndDistinct(std::vector<cv::Vec3d>& pixels)
	{
		std::sort(begin(pixels), end(pixels), PixelLexicographicalLess());
		auto newEnd = std::unique(begin(pixels), end(pixels));

		pixels.resize(std::distance(begin(pixels), newEnd));
	}

	auto testWaterClassifier(const WaterClassifier& wc) -> float
	{
		auto svgFilter = "*.svg";

		const static uchar Water = 1;
		const static uchar NonWater = 0;
		std::vector<cv::Vec3d> pixels;
		std::vector<uchar> expectedLabels;

		// load pixels

		auto waterMarkupColor = "#0000FF"; // blue
		std::vector<cv::Vec3d> waterPixels;
		auto waterMarkupFiles = "data/waterMarkup/waterMarkupTests/";
		loadWaterPixels(waterMarkupFiles, svgFilter, waterMarkupColor, waterPixels);

		std::copy(std::begin(waterPixels), std::end(waterPixels), std::back_inserter(pixels));
		std::generate_n(std::back_inserter(expectedLabels), waterPixels.size(), []() { return Water; });

		//

		auto nonWaterMarkupColor = "#FFFF00"; // yellow
		std::vector<cv::Vec3d> nonWaterPixels;
		loadWaterPixels(waterMarkupFiles, svgFilter, nonWaterMarkupColor, nonWaterPixels);

		std::copy(std::begin(nonWaterPixels), std::end(nonWaterPixels), std::back_inserter(pixels));
		std::generate_n(std::back_inserter(expectedLabels), nonWaterPixels.size(), []() { return NonWater; });

		// test
		std::vector<uchar> actualLabels(expectedLabels.size());
		auto& classifFun = [&wc](const cv::Vec3d& pixBgr) -> uchar
		{
			bool test = const_cast<WaterClassifier&>(wc).predict(pixBgr);
			return test ? Water : NonWater;
		};
		std::transform(std::begin(pixels), std::end(pixels), std::begin(actualLabels), classifFun);
		int errAbs = 0;
		for (size_t i = 0; i < expectedLabels.size(); ++i)
		{
			errAbs += std::abs(expectedLabels[i] - actualLabels[i]);
		}
		float errRate = errAbs / (float)expectedLabels.size();
		return errRate;
	}

	void trainWaterClassifier(WaterClassifier& wc)
	{
		auto waterMarkupFiles = "data/WaterMarkup/";
		auto svgFilter = "*.svg";

		auto waterMarkupColor = "#0000FF"; // blue
		std::vector<cv::Vec3d> waterPixels;
		loadWaterPixels(waterMarkupFiles, svgFilter, waterMarkupColor, waterPixels);

		auto nonWaterMarkupColor = "#FFFF00"; // yellow
		std::vector<cv::Vec3d> nonWaterPixels;
		loadWaterPixels(waterMarkupFiles, svgFilter, nonWaterMarkupColor, nonWaterPixels);

		//

		bool use1 = true;
		if (use1) sortAndDistinct(waterPixels);
		bool use2 = true;
		if (use2) sortAndDistinct(nonWaterPixels);

		std::vector<cv::Vec3d> waterPixelsExcl;
		std::set_difference(std::begin(waterPixels), std::end(waterPixels), std::begin(nonWaterPixels), std::end(nonWaterPixels), std::back_inserter(waterPixelsExcl), PixelLexicographicalLess());

		//std::random_shuffle(std::begin(waterPixels), std::end(waterPixels));
		std::random_shuffle(std::begin(waterPixelsExcl), std::end(waterPixelsExcl));
		std::random_shuffle(std::begin(nonWaterPixels), std::end(nonWaterPixels));

		cv::Mat waterColorsMat(waterPixelsExcl);
		cv::Mat nonWaterColorsMat(nonWaterPixels);
		wc.trainWater(waterColorsMat, nonWaterColorsMat);
		wc.initCache();
	}

	void runDetermineParametersWaterClassifier()
	{
		auto harnessFun = [](int nclusters, int trainIterCount, int covMatType) -> float // return accuracy
		{
			//TermCriteria term(TermCriteria::COUNT + TermCriteria::EPS, trainIterCount, 0.001);
			TermCriteria term(TermCriteria::COUNT + TermCriteria::EPS, 1000, 0.001);
			auto pc = std::make_unique<WaterClassifier>(nclusters, covMatType, term, term);
			trainWaterClassifier(*pc);
			float errRate = testWaterClassifier(*pc);
			return errRate;
		};

		for (int nclusters = 5; nclusters < 16; )
		{
			int itStep = 10;
			//for (int trainIterCount = 10; trainIterCount < 100; trainIterCount += itStep, itStep += 10)
			{
				for (int covMatType : {cv::EM::COV_MAT_SPHERICAL, cv::EM::COV_MAT_DIAGONAL})
				{
					int trainIterCount = -1;
					float errRate = harnessFun(nclusters, trainIterCount, covMatType);
					cout << "nclusters=" << nclusters
						<< " it=" << trainIterCount
						<< " covMatType" << covMatType
						<< " errRate=" << errRate << endl;
				}
			}
			if (nclusters < 6)
				nclusters++;
			else
				nclusters += 2;
		}
	}

	void runWaterClassifier()
	{
		std::unique_ptr<WaterClassifier> pc;

		cv::FileStorage fs;
		char const* fileName = "cl_water.yml";
		if (!fs.open(fileName, cv::FileStorage::READ))
		{
			int nclusters = 5;
			TermCriteria term(TermCriteria::COUNT + TermCriteria::EPS, 10000, 0.001);
			pc = std::make_unique<WaterClassifier>(nclusters, cv::EM::COV_MAT_SPHERICAL, term, term);
			trainWaterClassifier(*pc);

			if (!fs.open(fileName, cv::FileStorage::WRITE))
				return;
			pc->write(fs);
			fs.release();

		}
		else
		{
			pc = WaterClassifier::read(fs);
		}
		
		WaterClassifier& wc = *pc;
		auto errRate=testWaterClassifier(wc);
		cout << "errRate=" << errRate << endl;

		//
		//cv::Mat image = cv::imread("data/MVI_3177_0127_640x476.png");
		cv::Mat image = cv::imread("../../dinosaur/MVI_4635_640x480_Frame1.png");
		//image = cv::imread("../../dinosaur/waterMarkup/waterMarkupTests/MVI_4635_Frame1_black_flippers.png");
		int zz = 0;

		auto waterClassifFun = [&wc](const cv::Vec3d& pix) -> bool
		{
			auto x1 = pix[0];
			auto x2 = pix[1];
			auto x3 = pix[2];
			return wc.predict(pix);
		};

		cv::Mat_<uchar> waterMask;

		typedef std::chrono::system_clock Clock;
		std::chrono::time_point<Clock> now1 = Clock::now();
		classifyAndGetMask(image, waterClassifFun, waterMask);
		//classifyAndGetMaskParWorkPerPixel(i1, waterClassifFunPure, i1WaterMask);
		//classifyAndGetMaskParWorkLine(i1, waterClassifFunPure, i1WaterMask);
		//classifyAndGetMaskAmpFloat(i1, wc, i1WaterMask);

		std::chrono::time_point<Clock> now2 = Clock::now();

		auto elapsedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now2 - now1).count();
		std::cout << "elapsedMilliseconds=" << elapsedMilliseconds << endl;

		cv::Mat i2;
		image.copyTo(i2, waterMask);
	}


	void run()
	{
		//runWaterClassifier();
		runDetermineParametersWaterClassifier();
	}
}