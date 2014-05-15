#include "VideoLogger.h"

double VideoLogger::fps_;
boost::filesystem::path VideoLogger::logDir_;
std::map<std::string, cv::VideoWriter> VideoLogger::videoLoggerNameToObj_;

void VideoLogger::init(const boost::filesystem::path& logDir, double fps)
{
	CV_Assert(logDir.is_absolute());
	logDir_ = logDir;
	fps_ = fps;
}

void VideoLogger::logDebug(const char* pStreamName, const cv::Mat& frame)
{
	if (logDir_.empty())
		return;

	CV_Assert(pStreamName != nullptr);
	CV_Assert(!frame.empty());
	CV_Assert(frame.type() == CV_8UC3 && "Must be color frame");

	std::string streamName(pStreamName);

	cv::VideoWriter* pWriter;
	auto loggerIt = videoLoggerNameToObj_.find(streamName);
	if (loggerIt == videoLoggerNameToObj_.end())
	{
		int sourceFourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

		auto filePath = cv::String((logDir_ / (streamName + ".avi")).normalize().string().c_str());
		cv::VideoWriter writer(filePath, sourceFourcc, fps_, cv::Size(frame.cols, frame.rows));
		auto insOp = videoLoggerNameToObj_.insert(std::make_pair(std::string(streamName), std::move(writer)));
		loggerIt = insOp.first;
	}
	
	cv::VideoWriter& writer = loggerIt->second;
	writer.write(frame);
}

void VideoLogger::destroy()
{
	videoLoggerNameToObj_.clear();
}
