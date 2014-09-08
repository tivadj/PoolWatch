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
#include <KalmanFilterMovementPredictor.h>

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
		if (videoFile.startsWith("mvi_3177", Qt::CaseInsensitive) ||
			videoFile.startsWith("mvi3177", Qt::CaseInsensitive))

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

			// top of the string of flags
			imagePoints.push_back(cv::Point2f(414, 108));
			worldPoints.push_back(cv::Point3f(25, 5, CameraProjector::zeroHeight()));

			// bottom of the string of flags
			imagePoints.push_back(cv::Point2f(149, 481));
			worldPoints.push_back(cv::Point3f(0, 5, CameraProjector::zeroHeight()));
			return true;
		}
		else if (videoFile.startsWith("mvi_4636", Qt::CaseInsensitive))
		{
			// 640x480

			// top-right
			imagePoints.push_back(cv::Point2f(565, 75));
			worldPoints.push_back(cv::Point3f(25, 0, CameraProjector::zeroHeight()));

			// bottom-right is origin
			imagePoints.push_back(cv::Point2f(641, 484));
			worldPoints.push_back(cv::Point3f(0, 0, CameraProjector::zeroHeight()));

			// top of the string of flags
			imagePoints.push_back(cv::Point2f(444, 76));
			worldPoints.push_back(cv::Point3f(25, 5, CameraProjector::zeroHeight()));

			// bottom of the string of flags
			imagePoints.push_back(cv::Point2f(192, 389));
			worldPoints.push_back(cv::Point3f(0, 5, CameraProjector::zeroHeight()));
			return true;
		}
		else if (videoFile.startsWith("mvi_4637", Qt::CaseInsensitive))
		{
			// 640x480

			// top-right
			imagePoints.push_back(cv::Point2f(508, 76.5));
			worldPoints.push_back(cv::Point3f(25, 0, CameraProjector::zeroHeight()));

			// bottom-right is origin
			imagePoints.push_back(cv::Point2f(828, 410));
			worldPoints.push_back(cv::Point3f(0, 0, CameraProjector::zeroHeight()));

			// top of the string of flags
			imagePoints.push_back(cv::Point2f(398, 78.5));
			worldPoints.push_back(cv::Point3f(25, 5, CameraProjector::zeroHeight()));

			// bottom of the string of flags
			imagePoints.push_back(cv::Point2f(468, 395.5));
			worldPoints.push_back(cv::Point3f(0, 5, CameraProjector::zeroHeight()));
			return true;
		}
		else if (videoFile.startsWith("mvi_4638", Qt::CaseInsensitive))
		{
			// 640x480

			// top-right
			imagePoints.push_back(cv::Point2f(624.5, 114.5));
			worldPoints.push_back(cv::Point3f(25, 0, CameraProjector::zeroHeight()));

			// bottom-right is origin
			imagePoints.push_back(cv::Point2f(631, 503.5));
			worldPoints.push_back(cv::Point3f(0, 0, CameraProjector::zeroHeight()));

			// top of the string of flags
			imagePoints.push_back(cv::Point2f(505.5, 117.5));
			worldPoints.push_back(cv::Point3f(25, 5, CameraProjector::zeroHeight()));

			// bottom of the string of flags
			imagePoints.push_back(cv::Point2f(214.5, 436));
			worldPoints.push_back(cv::Point3f(0, 5, CameraProjector::zeroHeight()));
			return true;
		}
		else if (videoFile.startsWith("mvi_4641", Qt::CaseInsensitive))
		{
			// 640x480

			// bottom of the string of flags
			imagePoints.push_back(cv::Point2f(264, 436.5));
			worldPoints.push_back(cv::Point3f(0, 5, CameraProjector::zeroHeight()));

			// top of the string of flags
			imagePoints.push_back(cv::Point2f(644, 165.5));
			worldPoints.push_back(cv::Point3f(25, 5, CameraProjector::zeroHeight()));

			// top-left
			imagePoints.push_back(cv::Point2f(150.5, 157));
			worldPoints.push_back(cv::Point3f(25, 50, CameraProjector::zeroHeight()));

			// left
			imagePoints.push_back(cv::Point2f(-90, 218.5));
			worldPoints.push_back(cv::Point3f(0, 50, CameraProjector::zeroHeight()));

			return true;
		}

		return false;
	}

	void downsamplePointsInplace(float downsampleRatio, std::vector<cv::Point2f>& imagePoints)
	{
		for (auto& point : imagePoints)
		{
			point.x = point.x * downsampleRatio;
			point.y = point.y * downsampleRatio;
		}
	}

	bool readVideoFrame(cv::VideoCapture& videoCapture, int takeEachKthFrame, cv::Mat& imageFrame)
	{
		// skip takeEachKthFrame-1 frames and read/return the next frame

		for (int skip = 0; skip < takeEachKthFrame; skip++)
		{
			bool readOp = videoCapture.read(imageFrame);
			if (!readOp)
				return false;
		}
		return true;
	}

	void trackVideoFileTest()
	{
		std::string timeStamp = PoolWatch::timeStampNow();
		boost::filesystem::path outDir = boost::filesystem::path("../output") / timeStamp;
		outDir = boost::filesystem::absolute(outDir, ".").normalize();
		QDir outDirQ = QDir(outDir.string().c_str());

		// 
		configureLogToFileAppender(outDirQ, "app.log");
		LOG4CXX_DEBUG(log_, "test debug");
		LOG4CXX_INFO(log_, "test info");
		LOG4CXX_ERROR(log_, "test error");

		//
		// new / old frame size ratio (eg. 0.5 or 1)
		const float downsampleRatio = 1; // no downsampling
		//const float downsampleRatio = 0.5; // decreases width and height in 2 times 

		// process only each K-th frame
		//const int takeEachKthFrame = 1; // original FPS
		const int takeEachKthFrame = 5;

		//auto videoPathRel = "../../output/mvi3177_blueWomanLane3.avi"; // 640x480
		auto videoPathRel = "../../dinosaur/mvi_4635_640x480.mp4";
		//auto videoPathRel = "../../dinosaur/mvi_4636_640x480.mp4";
		//auto videoPathRel = "../../dinosaur/mvi_4637_640x480.mp4";
		//auto videoPathRel = "../../dinosaur/mvi_4638_640x480.mp4";
		//auto videoPathRel = "../../dinosaur/mvi_4641_640x480.mp4";
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

		LOG4CXX_INFO(log_, "opened FramesCount=" << framesCount <<" File=" <<videoPath);
		LOG4CXX_INFO(log_, "original WxH=[" << frameWidth << " " << frameHeight <<"] FPS=" << fps);

		// actual number of frames to process may be smaller
		frameWidth = frameWidth * downsampleRatio;
		frameHeight = frameHeight * downsampleRatio;
		fps = fps / takeEachKthFrame;
		framesCount = static_cast<int>(framesCount / takeEachKthFrame);
		LOG4CXX_INFO(log_, "downsamp WxH=[" << frameWidth << " " << frameHeight << "] FPS=" << fps);

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
		//const int NewTrackDelay = 7;
		const int NewTrackDelay = 1;

		// 0.4 doesn't work for takeEachKthFrame=5 (dist=0.894914 (>0.86))
		// 0.5 doesn't handle the case when there is a noise blob and then the track can't catch up with the real track(dist 1.06259>0.96)
		// 0.7 doesn't handle the case when there is a noise blob and then the track can't catch up with the real track(dist 1.21266>1.16)
		// 0.9 works
		// 1 is too large, as tracks jumps to nearby noise blobs
		const float ShapeCentroidNoise = 10.0f;
		auto cameraProjector = make_shared<CameraProjector>();

		const float swimmerMaxSpeed = 2.3f;         // max speed for swimmers 2.3m/s

		// the speed increases twice, so that if back and hands are split into two blobs with distance 1.1m, then tracker prefer to continue track
		// instead of associating new track with hands
		//const float swimmerMaxSpeed = 2.3f*2;         // max speed for swimmers 2.3m/s
		// Max shift per frame depends on image size and camera projection
		float maxDistPerFrame = swimmerMaxSpeed / fps;

		auto movementModel = std::make_unique<KalmanFilterMovementPredictor>(maxDistPerFrame);
		auto appearanceModel = std::make_unique<SwimmerAppearanceModel>();

		auto blobTracker = make_unique<MultiHypothesisBlobTracker>(pruneWindow, cameraProjector, std::move(movementModel), std::move(appearanceModel));
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
				bld << "Found " << blobs.size() << " blobs";
				for (const auto& blob : blobs)
					bld << std::endl << "  Id=" << blob.Id << " Centroid=" << blob.Centroid << blob.CentroidWorld << " Area=" << blob.AreaPix;
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
		if (false && guessCameraMat(videoFileName.c_str(), cameraMat))
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

		cv::Mat imageFrameOrigin;

		typedef std::chrono::system_clock Clock;
		std::chrono::time_point<Clock> startProcessing = Clock::now();
		int frameOrd = 0;
		bool isCameraOriented = false;
		for (; frameOrd < framesCount; ++frameOrd)
		{
			LOG4CXX_INFO(log_, "frameOrd= " << frameOrd << " of " << framesCount);

			std::chrono::time_point<Clock> now1 = Clock::now();

			// TODO: tracking doesn't work when processing the same image repeatedly
			bool readOp = readVideoFrame(videoCapture, takeEachKthFrame, imageFrameOrigin);
			if (!readOp)
			{
				LOG4CXX_ERROR(log_, "Can't read video frame, frameOrd= " << frameOrd);
				break;
			}

			// downsample the image
			cv::Mat& imageFrame = videoBuffer.requestNew();
			if (downsampleRatio < 1)
				cv::pyrDown(imageFrameOrigin, imageFrame);
			else
				imageFrameOrigin.copyTo(imageFrame);

			// if camera is fixed, then we may orient camera only once
			// otherwise camera should be oriented independently for each frame

			if (false || !isCameraOriented)
			{
				worldPoints.clear();
				imagePoints.clear();
				bool pointsFound = getImageCalibrationPoints(videoFileName.c_str(), frameOrd, worldPoints, imagePoints);
				CV_Assert(pointsFound);
				downsamplePointsInplace(downsampleRatio, imagePoints);

				bool cameraOriented = cameraProjector->orientCamera(cameraMat, worldPoints, imagePoints);
				CV_Assert(cameraOriented);
				
				cv::Point3f cameraPos = cameraProjector->cameraPosition();
				LOG4CXX_DEBUG(log_, "camera Pos" << cameraPos);
				isCameraOriented = true;
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

		std::chrono::time_point<Clock> finishProcessing = Clock::now();
		auto processingMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finishProcessing - startProcessing).count();
		int processedFramesCount = frameOrd + 1;
		auto processingFPS = processedFramesCount * 1000.0 / processingMilliseconds;
		LOG4CXX_INFO(log_, "Processed FramesCount=" << processedFramesCount <<" in " << processingMilliseconds << "ms" << " procFPS=" << processingFPS);
	}

	void run()
	{
		trackVideoFileTest();
	}
}