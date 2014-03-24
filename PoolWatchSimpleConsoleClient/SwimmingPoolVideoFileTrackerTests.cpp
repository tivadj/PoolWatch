#include <iostream>

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp> // imread

#include "PoolWatchFacade.h"
#include "algos1.h"
#include "SwimmingPoolObserver.h"

namespace SwimmingPoolVideoFileTrackerTestsNS
{
	using namespace cv;
	using namespace std;

	void trackVideoFileTest()
	{
		// init water classifier
		cv::FileStorage fs;
		if (!fs.open("1.yml", cv::FileStorage::READ))
			return;
		auto pWc = WaterClassifier::read(fs);
		WaterClassifier& wc = *pWc;

		cv::VideoCapture videoCapture("../../output/mvi3177_blueWomanLane3.avi");
		if (!videoCapture.isOpened())
		{
			cerr << "can't open video file";
			return;
		}

		const int pruneWindow = 6;
		float fps = (float)videoCapture.get(cv::CAP_PROP_FPS);
		SwimmingPoolObserver poolObserver(pruneWindow, fps);
		CameraProjector& cameraProjector = *poolObserver.cameraProjector();


		cv::Mat imageFrame;

		int frameWidth = (int)videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
		int frameHeight = (int)videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
		cv::Mat imageAdornment = cv::Mat::zeros(frameHeight, frameWidth, CV_8UC3);

		int framesCount = (int)videoCapture.get(cv::CAP_PROP_FRAME_COUNT);

		int frameOrd = 0;
		for (; frameOrd < framesCount; ++frameOrd)
		{
			cout << "frameOrd= " << frameOrd <<" of " <<framesCount <<endl;

			// TODO: tracking doesn't work when processing the same image repeatedly

			//cv::Mat imageFrame = cv::imread("data/MVI_3177_0127_640x476.png");
			videoCapture >> imageFrame;

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

			// update blobs CentroidWorld
			for (auto& blob : blobs)
			{
				blob.CentroidWorld = cameraProjector.cameraToWorld(blob.Centroid);
			}

			int frameIndWithTrackInfo;
			poolObserver.processBlobs(frameOrd, imageFrame, blobs, &frameIndWithTrackInfo);

			// visualize tracking

			imageFrame.copyTo(imageAdornment);

			if (frameIndWithTrackInfo != -1)
			{
				int trailLength = 255;
				// TODO: process previous frame from looped cache
				poolObserver.adornImage(imageFrame, frameIndWithTrackInfo, trailLength, imageAdornment);
			}

			cv::imshow("visualTracking", imageAdornment);
			if (cv::waitKey(100) == 27)
				return;
		}
	}

	void run()
	{
		trackVideoFileTest();
	}
}