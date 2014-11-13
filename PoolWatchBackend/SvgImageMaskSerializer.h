#pragma once
#include <vector>
#include <string>

#include "PoolWatchFacade.h"

PW_EXPORTS void loadImageAndPolygons(const std::string& svgFilePath, const std::string& strokeColor, cv::Mat& outImage, std::vector<std::vector<cv::Point2f>>& outPolygons);
PW_EXPORTS void loadImageAndMask(const std::string& svgFilePath, const std::string& strokeColor, cv::Mat& outImage, cv::Mat_<bool>& outMask);
PW_EXPORTS void loadWaterPixels(const std::string& folderPath, const std::string& svgFilter, const std::string& strokeStr, std::vector<cv::Vec3d>& pixels, bool invertMask = false, int inflateContourDelta = 0);

PW_EXPORTS void PWDrawContours(const cv::Mat& image, const std::vector<std::vector<cv::Point2i>>& contours, int contourIdx, const cv::Scalar& color, int thickness = 1);