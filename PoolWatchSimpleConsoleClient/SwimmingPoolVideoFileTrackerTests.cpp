#include <iostream>
#include <chrono> // std::chrono::system_clock

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

#include "algos1.h"
#include "SwimmingPoolObserver.h"
#include "ProgramUtils.h"
#include "VideoLogger.h"
#include "CoreUtils.h"
#include "PaintHelper.h"

namespace SwimmingPoolVideoFileTrackerTestsNS
{
	using namespace cv;
	using namespace std;
	using namespace PoolWatch;

	using namespace log4cxx;
	using namespace log4cxx::helpers;

	log4cxx::LoggerPtr log_(log4cxx::Logger::getLogger("PW.App"));

	bool guessCameraMat(const QString& videoFile, cv::Matx33f& cameraMat)
	{
		if (videoFile.startsWith("mvi", Qt::CaseInsensitive))
		{
			// Cannon D20 640x480
			float cx = 323.07199373780122f;
			float cy = 241.16033688735058f;
			float fx = 526.96329424435044f;
			float fy = 527.46802103114874f;

			fillCameraMatrix(cx, cy, fx, fy, cameraMat);
			return true;
		}
		return false;
	}

	bool getImageCalibrationPoints(const QString& videoFile, int frameOrd, std::vector<cv::Point3f>& worldPoints, std::vector<cv::Point2f>& imagePoints)
	{
		if (videoFile.startsWith("mvi_3177", Qt::CaseInsensitive))
		{
			// top, origin(0, 0)
			imagePoints.push_back(cv::Point2f(242, 166));
			worldPoints.push_back(cv::Point3f(0, 0, CameraProjector::zeroHeight()));

			//top, 4 marker
			imagePoints.push_back(cv::Point2f(516, 156));
			worldPoints.push_back(cv::Point3f(0, 10, CameraProjector::zeroHeight()));

			//bottom, 2 marker
			imagePoints.push_back(cv::Point2f(-71, 304));
			worldPoints.push_back(cv::Point3f(25, 6, CameraProjector::zeroHeight()));

			// bottom, 4 marker
			imagePoints.push_back(cv::Point2f(730, 365));
			worldPoints.push_back(cv::Point3f(25, 10, CameraProjector::zeroHeight()));
			return true;
		}
		else if (videoFile.startsWith("mvi_4635", Qt::CaseInsensitive))
		{
			// 640x480

			// top-right
			imagePoints.push_back(cv::Point2f(528, 111));
			worldPoints.push_back(cv::Point3f(25, 0, CameraProjector::zeroHeight()));

			// bottom-right is origin
			imagePoints.push_back(cv::Point2f(654, 523));
			worldPoints.push_back(cv::Point3f(0, 0, CameraProjector::zeroHeight()));

			// top of flags string
			imagePoints.push_back(cv::Point2f(414, 108));
			worldPoints.push_back(cv::Point3f(25, 5, CameraProjector::zeroHeight()));

			// bottom of flags string
			imagePoints.push_back(cv::Point2f(149, 481));
			worldPoints.push_back(cv::Point3f(0, 5, CameraProjector::zeroHeight()));
			return true;
		}
		return false;
	}

	void trackVideoFileTest()
	{
		std::string timeStamp = PoolWatch::timeStampNow();
		boost::filesystem::path outDir = boost::filesystem::path("../../output/debug") / timeStamp;
		outDir = boost::filesystem::absolute(outDir, ".").normalize();
		QDir outDirQ = QDir(outDir.string().c_str());

		// 
		configureLogToFileAppender(outDirQ, "app.log");
		LOG4CXX_DEBUG(log_, "test debug");
		LOG4CXX_INFO(log_, "test info");
		LOG4CXX_ERROR(log_, "test error");

		//
		//auto videoPathRel = "../../output/mvi3177_blueWomanLane3.avi";
		auto videoPathRel = "../../dinosaur/mvi_4635_640x480.mp4";
		boost::filesystem::path videoPath = boost::filesystem::absolute(videoPathRel).normalize();
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

		// prepare video writing

		boost::filesystem::path outVideoPath = outDir / "track.avi";

		cv::VideoWriter videoWriter;
		int sourceFourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
		videoWriter.open(outVideoPath.string(), sourceFourcc, fps, cv::Size(frameWidth, frameHeight), true);

		VideoLogger::init(outDir, fps);

		//
		PaintHelper* paintHelperOrNull = nullptr;
#if LOG_DEBUG_EX
		PaintHelper paintHelper;
		paintHelperOrNull = &paintHelper;
#endif
		// prepare video observer

		const int pruneWindow = 5;
		const int NewTrackDelay = 7;
		const float ShapeCentroidNoise = 0.4f;
		auto cameraProjector = make_shared<CameraProjector>();
		auto blobTracker = make_unique<MultiHypothesisBlobTracker>(cameraProjector, pruneWindow, fps);
		blobTracker->initNewTrackDelay_ = NewTrackDelay;
		blobTracker->shapeCentroidNoise_ = ShapeCentroidNoise;
		SwimmingPoolObserver poolObserver(std::move(blobTracker), cameraProjector);
		poolObserver.setLogDir(outDir);
		poolObserver.BlobsDetected = [paintHelperOrNull](const std::vector<DetectedBlob>& blobs, const cv::Mat& imageFamePoolOnly)
		{
			if (paintHelperOrNull != nullptr)
			{
				// show blobs
				auto blobsImageDbg = imageFamePoolOnly.clone();
				for (const auto& blob : blobs)
					paintHelperOrNull->paintBlob(blob, blobsImageDbg);
				VideoLogger::logDebug("blobsOutline", blobsImageDbg);
			}

			if (log_->isDebugEnabled())
			{
				stringstream bld;
				bld << "Found " << blobs.size() << " blobs" << endl;
				for (const auto& blob : blobs)
					bld << "  Id=" << blob.Id << " Centroid=" << blob.Centroid << " Area=" << blob.AreaPix << endl;
				LOG4CXX_DEBUG(log_, bld.str());
			}
		};
		
		tuple<bool,std::string> initOp = poolObserver.init();
		if (!get<0>(initOp))
		{
			LOG4CXX_ERROR(log_, get<1>(initOp));
			return;
		}
		LOG4CXX_INFO(log_, "PoolObserver pruneWindow=" << pruneWindow 
			<< " NewTrackDelay=" << NewTrackDelay
			<< " ShapeCentroidNoise=" << ShapeCentroidNoise);

		// store recent video frames to adorn the correct one with track info

		PoolWatch::CyclicHistoryBuffer<cv::Mat> videoBuffer(pruneWindow + 1);
		videoBuffer.init([=](size_t index, cv::Mat& frame)
		{
			frame = cv::Mat::zeros(frameHeight, frameWidth, CV_8UC3);
		});

		// query camera matrix

		cv::Matx33f cameraMat;
		auto videoFileName = videoPath.filename().string(); 
		if (guessCameraMat(videoFileName.c_str(), cameraMat))
		{
			LOG4CXX_INFO(log_, "Camera matrix: from configuration");
		}
		else
		{
			// init as an "average" camera
			float fovX = deg2rad(62);
			float fovY = deg2rad(49);
			float fx = -1;
			float fy = -1;
			float cx = -1;
			float cy = -1;

			approxCameraMatrix(frameWidth, frameHeight, fovX, fovY, cx, cy, fx, fy);
			fillCameraMatrix(cx, cy, fx, fy, cameraMat);
			LOG4CXX_INFO(log_, "Camera matrix: average");
		}
		
		LOG4CXX_INFO(log_, "cx,cy=" << cameraMat(0, 2) << "," << cameraMat(1, 2) << " fx,fy=" << cameraMat(0, 0) << "," << cameraMat(1, 1));

		std::vector<cv::Point3f> worldPoints;
		std::vector<cv::Point2f> imagePoints;

		// video processing loop

		cv::Mat imageAdornment = cv::Mat::zeros(frameHeight, frameWidth, CV_8UC3);

		int frameOrd = 0;
		for (; frameOrd < framesCount; ++frameOrd)
		{
			LOG4CXX_INFO(log_, "frameOrd= " << frameOrd << " of " << framesCount);

			typedef std::chrono::system_clock Clock;
			std::chrono::time_point<Clock> now1 = Clock::now();

			// TODO: tracking doesn't work when processing the same image repeatedly

			//cv::Mat imageFrame = cv::imread("data/MVI_3177_0127_640x476.png");

			cv::Mat& imageFrame = videoBuffer.requestNew();
			bool readOp = videoCapture.read(imageFrame);
			if (!readOp)
			{
				LOG4CXX_ERROR(log_, "Can't read video frame, frameOrd= " << frameOrd);
				break;
			}

			//
			{
				worldPoints.clear();
				imagePoints.clear();
				bool pointsFound = getImageCalibrationPoints(videoFileName.c_str(), frameOrd, worldPoints, imagePoints);
				CV_Assert(pointsFound);

				bool cameraOriented = cameraProjector->orientCamera(cameraMat, worldPoints, imagePoints);
				CV_Assert(cameraOriented);
				
				cv::Point3f cameraPos = cameraProjector->cameraPosition();
				LOG4CXX_DEBUG(log_, "camera Pos" << cameraPos);
			}

			int readyFrameInd;
			poolObserver.processCameraImage(frameOrd, imageFrame, &readyFrameInd);

			// visualize tracking

			if (readyFrameInd != -1)
			{
				// get frame with frameIndWithTrackInfo index
				int backIndex = - (frameOrd - readyFrameInd);
				
				cv::Mat& readyFrame = videoBuffer.queryHistory(backIndex);
				readyFrame.copyTo(imageAdornment);

				int trailLength = 255;
				poolObserver.adornImage(readyFrameInd, trailLength, imageAdornment);

 				cv::imshow("visualTracking", imageAdornment);

				// dump adorned video to file
				videoWriter.write(imageAdornment);
			}
			else
			{
				// tracking data is not available yet
				// do not write 'blank screen' here to keep original and tracks-adorned video streams in sync
			}

			std::chrono::time_point<Clock> now2 = Clock::now();
			auto elapsedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now2 - now1).count();
			LOG4CXX_INFO(log_, "frameOrd= " << frameOrd << " time=" << elapsedMilliseconds);

			if (cv::waitKey(1) == 27)
				break;
		}
		poolObserver.flushTrackHypothesis(frameOrd);

		std::stringstream bld;
		poolObserver.dumpTrackHistory(bld);
		LOG4CXX_DEBUG(log_, "Tracks Result " << bld.str());
	}

	void run()
	{
		trackVideoFileTest();
	}
}