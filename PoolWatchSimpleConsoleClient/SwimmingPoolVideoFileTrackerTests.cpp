#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> // imread
#include <opencv2/highgui/highgui_c.h> // CV_FOURCC

#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>

#include "PoolWatchFacade.h"
#include "algos1.h"
#include "SwimmingPoolObserver.h"

namespace SwimmingPoolVideoFileTrackerTestsNS
{
	using namespace cv;
	using namespace std;
	using namespace PoolWatch;

	void fixBlobs(std::vector<DetectedBlob>& blobs, const CameraProjector& cameraProjector)
	{
		// update blobs CentroidWorld
		for (auto& blob : blobs)
		{
			blob.CentroidWorld = cameraProjector.cameraToWorld(blob.Centroid);
		}
	}

	void trackVideoFileTest()
	{
		// init water classifier
		cv::FileStorage fs;
		if (!fs.open("1.yml", cv::FileStorage::READ))
			return;
		auto pWc = WaterClassifier::read(fs);
		WaterClassifier& wc = *pWc;

		boost::filesystem::path srcDir("../../output");
		boost::filesystem::path videoPath = srcDir / "mvi3177_blueWomanLane3.avi";
		cv::VideoCapture videoCapture(videoPath.string());
		if (!videoCapture.isOpened())
		{
			cerr << "can't open video file";
			return;
		}

		float fps = (float)videoCapture.get(cv::CAP_PROP_FPS);
		int frameWidth = (int)videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
		int frameHeight = (int)videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
		int framesCount = (int)videoCapture.get(cv::CAP_PROP_FRAME_COUNT);

		//
		const int pruneWindow = 6;
		SwimmingPoolObserver poolObserver(pruneWindow, fps);
		CameraProjector& cameraProjector = *poolObserver.cameraProjector();

		// prepare video writing

		std::string timeStamp = PoolWatch::timeStampNow();
		std::string fileNameNoExt = videoPath.stem().string();

		stringstream fileNameBuf;
		fileNameBuf <<fileNameNoExt << "_" << timeStamp <<"_track" << ".avi";
		std::string outVideoFileName = fileNameBuf.str();
		boost::filesystem::path outVideoPath = srcDir / "debug" / outVideoFileName;

		cv::VideoWriter videoWriter;
		int sourceFourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
		videoWriter.open(outVideoPath.string(), sourceFourcc, fps, cv::Size(frameWidth, frameHeight), true);

#if PW_DEBUG
		std::string videoFileNameBlobs = fileNameNoExt + "_" + timeStamp + "_blobs.avi";
		boost::filesystem::path videoPathBlobs = srcDir / "debug" / videoFileNameBlobs;
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
			cout << "frameOrd= " << frameOrd <<" of " <<framesCount <<endl;

			// TODO: tracking doesn't work when processing the same image repeatedly

			//cv::Mat imageFrame = cv::imread("data/MVI_3177_0127_640x476.png");

			cv::Mat& imageFrame = videoBuffer.requestNew();
			bool readOp = videoCapture.read(imageFrame);
			CV_Assert(readOp);

			// find water mask

			cv::Mat_<uchar> waterMask;
			classifyAndGetMask(imageFrame, [&wc](const cv::Vec3d& pix) -> bool
			{
				return wc.predict(pix);
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

			int frameIndWithTrackInfo;
			poolObserver.processBlobs(frameOrd, imageFrame, blobs, &frameIndWithTrackInfo);

			// visualize tracking

			if (frameIndWithTrackInfo != -1)
			{
				int trailLength = 255;

				// get frame with frameIndWithTrackInfo index
				int backIndex = - (frameOrd - frameIndWithTrackInfo);
				
				cv::Mat& readyFrame = videoBuffer.queryHistory(backIndex);
				readyFrame.copyTo(imageAdornment);

				poolObserver.adornImage(readyFrame, frameIndWithTrackInfo, trailLength, imageAdornment);
			}
			else
			{
				// tracking data is not available yet; just output empty frame
				imageAdornment.setTo(0);
			}

			cv::imshow("visualTracking", imageAdornment);
			if (cv::waitKey(100) == 27)
				return;

			// dump adorned video to file
			videoWriter.write(imageAdornment);
		}
	}

	void run()
	{
		trackVideoFileTest();
	}
}