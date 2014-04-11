#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> // imread
#include <opencv2/highgui/highgui_c.h> // CV_FOURCC

#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>

#include <QDir>

#include "PoolWatchFacade.h"
#include "algos1.h"
#include "SwimmingPoolObserver.h"
#include "TestingUtils.h"

namespace SwimmingPoolObserverTestsNS
{
	using namespace cv;
	using namespace std;
	using namespace PoolWatch;

	log4cxx::LoggerPtr log_(log4cxx::Logger::getLogger("PW.Tests"));

	void parallelMovementTest(const boost::filesystem::path& outDir)
	{
		// two swimmers swim on parallel lanes in one direction

		//
		const int pruneWindow = 2;
		const float fps = 1;
		auto cameraProjector = make_shared<LinearCameraProjector>();
		
		auto blobTracker = make_unique<MultiHypothesisBlobTracker>(cameraProjector, pruneWindow, fps);
		blobTracker->swimmerMaxSpeed_ = 1.1;
		blobTracker->shapeCentroidNoise_ = 0;
		blobTracker->initNewTrackDelay_ = 1;

		auto movementPredictor = std::make_unique<ConstantVelocityMovementPredictor>(cv::Point3f(1,0,0));
		blobTracker->setMovementPredictor(std::move(movementPredictor));

		SwimmingPoolObserver poolObserver(std::move(blobTracker), cameraProjector);
		auto outDirPtr = make_shared<boost::filesystem::path>(outDir);
		poolObserver.setLogDir(outDirPtr);

		std::vector<DetectedBlob> blobs;
		cv::Point2f center1(1, 1);
		cv::Point2f center2(1, 3);
		int readyFrameInd = -1;
		const int framesCount = 4;
		for (int frameInd = 0; frameInd < framesCount; frameInd++)
		{
			blobs.clear();

			DetectedBlob blob;
			blob.Id = 1;
			blob.Centroid = center1 + cv::Point2f(frameInd, 0);
			blobs.push_back(blob);

			blob.Id = 2;
			blob.Centroid = center2 + cv::Point2f(frameInd, 0);
			blobs.push_back(blob);
			
			fixBlobs(blobs, *cameraProjector);

			poolObserver.processBlobs(frameInd, blobs, &readyFrameInd);
		}

		//
		assert(framesCount >= 1 && "Must be at least one frame");
		poolObserver.flushTrackHypothesis(framesCount - 1);

		std::stringstream bld;
		poolObserver.dumpTrackHistory(bld);
		LOG4CXX_DEBUG(log_, "Tracks Result " <<bld.str());
	}

	void run()
	{
		boost::filesystem::path srcDir("../../output");

		std::string timeStamp = PoolWatch::timeStampNow();
		boost::filesystem::path outDir = srcDir / "debugTests" / timeStamp;
		boost::filesystem::create_directory(outDir);

		QDir outDirQ = QDir(outDir.string().c_str());

		// 
		configureLogToFileAppender(outDirQ, "app.log");
		LOG4CXX_DEBUG(log_, "test debug");
		LOG4CXX_INFO(log_, "test info");
		LOG4CXX_ERROR(log_, "test error");

		parallelMovementTest(outDir);
	}
}