#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> // imread
#include <opencv2/highgui/highgui_c.h> // CV_FOURCC

#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>

#include <log4cxx/logger.h>
#include <log4cxx/helpers/exception.h>
#include <log4cxx/rollingfileappender.h>
#include <log4cxx/patternlayout.h>

#include <QDir>

#include "PoolWatchFacade.h"
#include "algos1.h"
#include "SwimmingPoolObserver.h"

namespace SwimmingPoolVideoFileTrackerTestsNS
{
	using namespace cv;
	using namespace std;
	using namespace PoolWatch;

	using namespace log4cxx;
	using namespace log4cxx::helpers;

	log4cxx::LoggerPtr log_(log4cxx::Logger::getLogger("PW.App"));

	void fixBlobs(std::vector<DetectedBlob>& blobs, const CameraProjector& cameraProjector)
	{
		// update blobs CentroidWorld
		for (auto& blob : blobs)
		{
			blob.CentroidWorld = cameraProjector.cameraToWorld(blob.Centroid);
		}
	}
	
	void configureLogToFileAppender(const QDir& logFolder, const QString& logFileName)
	{
		// create rolling file appending to a runtime generated log

		QString fileRollAbsPath = logFolder.absoluteFilePath(logFileName);

		auto pLayout = log4cxx::helpers::ObjectPtr(new log4cxx::PatternLayout(L"%-5p %c - %m%n"));
		auto pRollingFileApp = log4cxx::helpers::ObjectPtr(new log4cxx::RollingFileAppender(pLayout, fileRollAbsPath.toStdWString()));

		log4cxx::LoggerPtr rootLoggerPtr = log4cxx::Logger::getRootLogger();
		rootLoggerPtr->addAppender(pRollingFileApp);
	}

	void trackVideoFileTest()
	{
		boost::filesystem::path srcDir("../../output");
		
		std::string timeStamp = PoolWatch::timeStampNow();
		boost::filesystem::path outDir = srcDir / "debug" / timeStamp;
		QDir outDir2 = QDir(outDir.string().c_str());

		// 
		configureLogToFileAppender(outDir2, "app.log");
		LOG4CXX_DEBUG(log_, "test debug");
		LOG4CXX_INFO(log_, "test info");
		LOG4CXX_ERROR(log_, "test error");

		// init water classifier

		cv::FileStorage fs;
		if (!fs.open("1.yml", cv::FileStorage::READ))
		{
			LOG4CXX_ERROR(log_, "Can't find file '1.yml' (change working directory)");
			return;
		}
		auto pWc = WaterClassifier::read(fs);
		WaterClassifier& wc = *pWc;

		//

		boost::filesystem::path videoPath = boost::filesystem::absolute("mvi3177_blueWomanLane3.avi", srcDir).normalize();
		cv::VideoCapture videoCapture(videoPath.string());
		if (!videoCapture.isOpened())
		{
			LOG4CXX_ERROR(log_, "can't open video file");
			return;
		}

		float fps = (float)videoCapture.get(cv::CAP_PROP_FPS);
		int frameWidth = (int)videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
		int frameHeight = (int)videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
		int framesCount = (int)videoCapture.get(cv::CAP_PROP_FRAME_COUNT);

		LOG4CXX_INFO(log_, "opened " <<videoPath);
		LOG4CXX_INFO(log_, "WxH=[" << frameWidth << " " << frameHeight << "] framesCount=" << framesCount <<" fps=" << fps);

		//
		const int pruneWindow = 6;
		SwimmingPoolObserver poolObserver(pruneWindow, fps);
		CameraProjector& cameraProjector = *poolObserver.cameraProjector();

		// prepare video writing

		boost::filesystem::path outVideoPath = outDir / "track.avi";

		cv::VideoWriter videoWriter;
		int sourceFourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
		videoWriter.open(outVideoPath.string(), sourceFourcc, fps, cv::Size(frameWidth, frameHeight), true);

#if PW_DEBUG
		boost::filesystem::path videoPathBlobs = outDir / "blobs.avi";

		cv::VideoWriter videoWriterBlobs;
		videoWriterBlobs.open(videoPathBlobs.string(), sourceFourcc, fps, cv::Size(frameWidth, frameHeight), true);
#endif

		// store recent video frames to adorn the correct one with track info

		PoolWatch::CyclicHistoryBuffer<cv::Mat> videoBuffer(pruneWindow + 1);
		videoBuffer.init([=](size_t index, cv::Mat& frame)
		{
			frame = cv::Mat::zeros(frameHeight, frameWidth, CV_8UC3);
		});

		//
#if PW_DEBUG
		PaintHelper paintHelper;
#endif

		// video processing loop

		cv::Mat imageAdornment = cv::Mat::zeros(frameHeight, frameWidth, CV_8UC3);

		int frameOrd = 0;
		for (; frameOrd < framesCount; ++frameOrd)
		{
			LOG4CXX_DEBUG(log_, "frameOrd= " << frameOrd << " of " << framesCount);

			// TODO: tracking doesn't work when processing the same image repeatedly

			//cv::Mat imageFrame = cv::imread("data/MVI_3177_0127_640x476.png");

			cv::Mat& imageFrame = videoBuffer.requestNew();
			bool readOp = videoCapture.read(imageFrame);
			CV_Assert(readOp);

			// find water mask

			cv::Mat_<uchar> waterMask;
			classifyAndGetMask(imageFrame, [&wc](const cv::Vec3d& pix) -> bool
			{
				//bool b1 = wc.predict(pix);
				bool b2 = wc.predictFloat(cv::Vec3f(pix[0], pix[1], pix[2]));
				//assert(b1 == b2);
				return b2;
			}, waterMask);


			// find pool mask

			cv::Mat_<uchar> poolMask;
			getPoolMask(imageFrame, waterMask, poolMask);

			cv::Mat imageFamePoolOnly = cv::Mat::zeros(imageFrame.rows, imageFrame.cols, imageFrame.type());
			imageFrame.copyTo(imageFamePoolOnly, poolMask);

			//
			std::vector<DetectedBlob> blobs;
			getHumanBodies(imageFamePoolOnly, waterMask, blobs);
			fixBlobs(blobs, cameraProjector);

#if PW_DEBUG
			// show blobs
			auto blobsImageDbg = imageFamePoolOnly.clone();
			for (const auto& blob : blobs)
				paintHelper.paintBlob(blob, blobsImageDbg);
			videoWriterBlobs.write(blobsImageDbg);
#endif
			if (log_->isDebugEnabled())
			{
				stringstream bld;
				bld << "Found " << blobs.size() << " blobs" <<endl;
				for (const auto& blob : blobs)
					bld << "Id=" << blob.Id << " Centroid=" << blob.Centroid << endl;
				LOG4CXX_DEBUG(log_, bld.str());
			}

			int frameIndWithTrackInfo;
			poolObserver.processBlobs(frameOrd, imageFrame, blobs, &frameIndWithTrackInfo);

			// visualize tracking

			if (frameIndWithTrackInfo != -1)
			{
				// get frame with frameIndWithTrackInfo index
				int backIndex = - (frameOrd - frameIndWithTrackInfo);
				
				cv::Mat& readyFrame = videoBuffer.queryHistory(backIndex);
				readyFrame.copyTo(imageAdornment);

				int trailLength = 255;
				poolObserver.adornImage(readyFrame, frameIndWithTrackInfo, trailLength, imageAdornment);

				cv::imshow("visualTracking", imageAdornment);

				// dump adorned video to file
				videoWriter.write(imageAdornment);
			}
			else
			{
				// tracking data is not available yet
				// do not write 'blank screen' here to keep original and tracks-adorned video streams in sync
			}

			if (cv::waitKey(1) == 27)
				return;
		}
	}

	void run()
	{
		trackVideoFileTest();
	}
}