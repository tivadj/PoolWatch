#include <opencv2\core.hpp>
#include "PoolWatchFacade.h"

namespace SvgImageMaskSerializerNS
{
	using namespace cv;
	using namespace std;

	void loadMaskTest()
	{
		cv::Mat image;
		cv::Mat_<bool> mask;
		loadImageAndMask("data/SkinMarkup/MVI_3177_0127_640x476_nonskinmask.svg", "#FFFFFF", image, mask);
	}

	void loadWaterPixelsTest(const std::string& folderPath)
	{
		auto waterMarkupFiles = "../../dinosaur/waterMarkup/";
		auto svgFilter = "*.svg";

		auto waterMarkupColor = "#0000FF";
		std::vector<cv::Vec3d> pixels;
		loadWaterPixels(waterMarkupFiles, svgFilter, waterMarkupColor, pixels);
	}

	void run()
	{
		//loadMaskTest();
		loadWaterPixelsTest("");
	}
}