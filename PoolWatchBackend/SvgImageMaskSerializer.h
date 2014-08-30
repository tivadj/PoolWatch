#pragma once
#include <vector>
#include <string>

#include <opencv2/core.hpp>

#include "PoolWatchFacade.h"

PW_EXPORTS void loadImageAndMask(const std::string& svgFilePath, const std::string& strokeColor, cv::Mat& outImage, cv::Mat_<bool>& outMask);
PW_EXPORTS void loadWaterPixels(const std::string& folderPath, const std::string& svgFilter, const std::string& strokeStr, std::vector<cv::Vec3d>& pixels, bool invertMask = false, int inflateContourDelta = 0);
