#pragma once
#include <string>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <boost/filesystem.hpp>
#include "PoolWatchFacade.h"
// Logs OpenCV images in a video file. Each set of images are associated with a separate
// stream. Each stream writes to a separated video file using cv::VideoWriter, which is
// initialized lazily.
class PW_EXPORTS VideoLogger
{
	static double fps_;
	static boost::filesystem::path logDir_;
	static std::map<std::string, cv::VideoWriter> videoLoggerNameToObj_;
public:
	static void init(const boost::filesystem::path& logDir, double fps);
	static void logDebug(const char* pStreamName, const cv::Mat& frame);
	static void destroy();
};

