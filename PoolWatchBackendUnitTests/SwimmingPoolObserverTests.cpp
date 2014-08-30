#include "stdafx.h"
#include <iostream>
#include <numeric>
#include <random>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> // imread
#include <opencv2/highgui/highgui_c.h> // CV_FOURCC

#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>

#include <QDir>

#include <CppUnitTest.h>

#include "algos1.h"
#include "SwimmingPoolObserver.h"
#include "TestingUtils.h"

namespace PoolWatchBackendUnitTests
{
	using namespace cv;
	using namespace std;
	using namespace PoolWatch;
	using namespace Microsoft::VisualStudio::CppUnitTestFramework;

	extern log4cxx::LoggerPtr log_;

	TEST_CLASS(SwimmingPoolObserverTests)
	{
		const char* ClassName = "SwimmingPoolObserverTests";
		TEST_METHOD_INITIALIZE(MethodInitialize)
		{
			PoolWatchBackendUnitTests_MethodInitilize();
		}
	public:
		// two swimmers swim on parallel lanes in one direction
		TEST_METHOD(parallelMovementTest)
		{
			auto outDir = initTestMethodLogFolder(ClassName, "parallelMovementTest");
			auto logFileScope = scopeLogFileAppenderNew(outDir);

			LOG4CXX_DEBUG(log_, "test debug");
			LOG4CXX_INFO(log_, "test info");
			LOG4CXX_ERROR(log_, "test error");

			const int pruneWindow = 2;
			const float fps = 1;
			auto cameraProjector = make_shared<LinearCameraProjector>();

			auto blobTracker = make_unique<MultiHypothesisBlobTracker>(cameraProjector, pruneWindow, fps);
			blobTracker->setSwimmerMaxSpeed(1.1);
			blobTracker->shapeCentroidNoise_ = 0;
			blobTracker->initNewTrackDelay_ = 1;

			auto movementPredictor = std::make_unique<ConstantVelocityMovementPredictor>(cv::Point3f(1, 0, 0));
			blobTracker->setMovementPredictor(std::move(movementPredictor));

			SwimmingPoolObserver poolObserver(std::move(blobTracker), cameraProjector);
			poolObserver.trackMinDurationFrames_ = 2;
			poolObserver.setLogDir(outDir);

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

			poolObserver.flushTrackHypothesis(framesCount - 1);

			std::stringstream bld;
			poolObserver.dumpTrackHistory(bld);
			LOG4CXX_DEBUG(log_, "Tracks Result " << bld.str());

			auto pTrackHist1 = poolObserver.trackHistoryForBlob(0, 0);
			Assert::IsNotNull(pTrackHist1);
			Assert::AreEqual(framesCount, (int)pTrackHist1->Assignments.size());
			auto& res1 = checkAll(pTrackHist1->Assignments, 1, [](const TrackChangePerFrame& c) { return c.ObservationPosPixExactOrApprox.y; });
			Assert::IsTrue(std::get<0>(res1), std::get<1>(res1).c_str());

			auto pTrackHist2 = poolObserver.trackHistoryForBlob(0, 1);
			Assert::IsNotNull(pTrackHist2);
			Assert::AreEqual(framesCount, (int)pTrackHist2->Assignments.size());
			auto& res2 = checkAll(pTrackHist2->Assignments, 3, [](const TrackChangePerFrame& c) { return c.ObservationPosPixExactOrApprox.y; });
			Assert::IsTrue(std::get<0>(res2), std::get<1>(res2).c_str());
		}

		// Trajectories of two swimmers cross orthogonally.
		TEST_METHOD(orthogonalCross1)
		{
			auto outDir = initTestMethodLogFolder(ClassName, "orthogonalCross1");
			auto logFileScope = scopeLogFileAppenderNew(outDir);

			LOG4CXX_DEBUG(log_, "test debug");
			LOG4CXX_INFO(log_, "test info");
			LOG4CXX_ERROR(log_, "test error");

			const int pruneWindow = 2;
			const float fps = 1;
			auto cameraProjector = make_shared<LinearCameraProjector>();

			auto blobTracker = make_unique<MultiHypothesisBlobTracker>(cameraProjector, pruneWindow, fps);
			blobTracker->setSwimmerMaxSpeed(1.1);
			blobTracker->shapeCentroidNoise_ = 0;
			blobTracker->initNewTrackDelay_ = 1;

			auto movementPredictor = std::make_unique<ConstantVelocityMovementPredictor>(cv::Point3f(0, 0, 0));
			cv::Point3f vel1(1, 0, 0);
			cv::Point3f vel2(0, -1, 0);
			movementPredictor->setSwimmerVelocity(FamilyIdHint(0, 0), vel1);
			movementPredictor->setSwimmerVelocity(FamilyIdHint(0, 1), vel2);
			blobTracker->setMovementPredictor(std::move(movementPredictor));

			SwimmingPoolObserver poolObserver(std::move(blobTracker), cameraProjector);
			poolObserver.trackMinDurationFrames_ = 2;
			poolObserver.setLogDir(outDir);

			std::vector<DetectedBlob> blobs;
			cv::Point2f center1(1, 3);
			cv::Point2f center2(3.3, 5.3);
			int readyFrameInd = -1;
			const int framesCount = 4;
			for (int frameInd = 0; frameInd < framesCount; frameInd++)
			{
				blobs.clear();

				DetectedBlob blob;
				blob.Id = 1;
				blob.Centroid = center1 + cv::Point2f(frameInd * vel1.x, frameInd * vel1.y);
				blobs.push_back(blob);

				blob.Id = 2;
				blob.Centroid = center2 + cv::Point2f(frameInd * vel2.x, frameInd * vel2.y);
				blobs.push_back(blob);

				fixBlobs(blobs, *cameraProjector);

				poolObserver.processBlobs(frameInd, blobs, &readyFrameInd);
			}

			poolObserver.flushTrackHypothesis(framesCount - 1);

			std::stringstream bld;
			poolObserver.dumpTrackHistory(bld);
			LOG4CXX_DEBUG(log_, "Tracks Result " << bld.str());

			auto pTrackHist1 = poolObserver.trackHistoryForBlob(0, 0);
			Assert::IsNotNull(pTrackHist1);
			Assert::AreEqual(framesCount, (int)pTrackHist1->Assignments.size());
			auto& res1 = checkAll(pTrackHist1->Assignments, 3, [](const TrackChangePerFrame& c) { return c.ObservationPosPixExactOrApprox.y; });
			Assert::IsTrue(std::get<0>(res1), std::get<1>(res1).c_str());

			auto pTrackHist2 = poolObserver.trackHistoryForBlob(0, 1);
			Assert::IsNotNull(pTrackHist2);
			Assert::AreEqual(framesCount, (int)pTrackHist2->Assignments.size());
			auto& res2 = checkAll(pTrackHist2->Assignments, 3.3, [](const TrackChangePerFrame& c) { return c.ObservationPosPixExactOrApprox.x; }, 0.01);
			Assert::IsTrue(std::get<0>(res2), std::get<1>(res2).c_str());
		}
		
		void temporalDisappear1Core(boost::filesystem::path outDir)
		{
			const int pruneWindow = 2;
			const float fps = 1;
			auto cameraProjector = make_shared<LinearCameraProjector>();

			auto blobTracker = make_unique<MultiHypothesisBlobTracker>(cameraProjector, pruneWindow, fps);
			blobTracker->setSwimmerMaxSpeed(1.1);
			blobTracker->shapeCentroidNoise_ = 0;
			blobTracker->initNewTrackDelay_ = 1;

			auto movementPredictor = std::make_unique<ConstantVelocityMovementPredictor>(cv::Point3f(1, 0, 0));
			blobTracker->setMovementPredictor(std::move(movementPredictor));

			SwimmingPoolObserver poolObserver(std::move(blobTracker), cameraProjector);
			poolObserver.trackMinDurationFrames_ = 2;
			poolObserver.setLogDir(outDir);

			std::vector<DetectedBlob> blobs;
			cv::Point2f center1(1, 3);
			int readyFrameInd = -1;
			const int framesCount = 5;

			//std::vector<uchar> hasObservation(framesCount, 1);
			//std::random_device rd;
			//std::mt19937 g(rd());
			//int noObsCount = g();
			//if (noObsCount >= hasObservation.size()) noObsCount = noObsCount % hasObservation.size();
			//for (int i = 0; i < noObsCount; ++i)
			//	hasObservation[i] = 0;
			//std::shuffle(begin(hasObservation), end(hasObservation), g);
			//hasObservation[0] = 1;

			for (int frameInd = 0; frameInd < framesCount; frameInd++)
			{
				blobs.clear();

				if (frameInd == 1 || frameInd == 3)
				//if (frameInd == 2)
				//if (frameInd % 2 == 0)
				//if (frameInd >= 5 && frameInd < 15)
				//if (!hasObservation[frameInd])
				{
					// no observations
				}
				else
				{
					DetectedBlob blob;
					blob.Id = 1;
					blob.Centroid = center1 + cv::Point2f(frameInd, 0);
					blobs.push_back(blob);

					fixBlobs(blobs, *cameraProjector);
				}

				poolObserver.processBlobs(frameInd, blobs, &readyFrameInd);
			}

			poolObserver.flushTrackHypothesis(framesCount - 1);

			std::stringstream bld;
			poolObserver.dumpTrackHistory(bld);
			LOG4CXX_DEBUG(log_, "Tracks Result " << bld.str());

			int tracksCount = poolObserver.trackHistoryCount();
			Assert::AreEqual(1, tracksCount);
			auto pTrackHist1 = poolObserver.trackHistoryForBlob(0, 0);
			Assert::IsNotNull(pTrackHist1);
			Assert::AreEqual(framesCount, (int)pTrackHist1->Assignments.size());
			auto& res1 = checkAll(pTrackHist1->Assignments, 3, [](const TrackChangePerFrame& c) { return c.ObservationPosPixExactOrApprox.y; });
			Assert::IsTrue(std::get<0>(res1), std::get<1>(res1).c_str());
		}

		TEST_METHOD(temporalDisappear1)
		{
			boost::filesystem::path outDir = initTestMethodLogFolder(ClassName, "temporalDisappear1");
			auto logFileScope = scopeLogFileAppenderNew(outDir);

			temporalDisappear1Core(outDir);
		}

		TEST_METHOD(noMovementTest)
		{
			auto outDir = initTestMethodLogFolder(ClassName, "noMovementTest");
			auto logFileScope = scopeLogFileAppenderNew(outDir);

			const int pruneWindow = 2;
			const float fps = 1;
			auto cameraProjector = make_shared<LinearCameraProjector>();

			auto blobTracker = make_unique<MultiHypothesisBlobTracker>(cameraProjector, pruneWindow, fps);
			blobTracker->setSwimmerMaxSpeed(1.1);
			blobTracker->shapeCentroidNoise_ = 0;
			blobTracker->initNewTrackDelay_ = 1;

			auto movementPredictor = std::make_unique<ConstantVelocityMovementPredictor>(cv::Point3f(1, 0, 0));
			blobTracker->setMovementPredictor(std::move(movementPredictor));

			SwimmingPoolObserver poolObserver(std::move(blobTracker), cameraProjector);
			poolObserver.trackMinDurationFrames_ = 2;
			poolObserver.setLogDir(outDir);

			std::vector<DetectedBlob> blobs;
			int readyFrameInd = -1;
			const int framesCount = 4;
			for (int frameInd = 0; frameInd < framesCount; frameInd++)
			{
				poolObserver.processBlobs(frameInd, blobs, &readyFrameInd);
			}

			//
			poolObserver.flushTrackHypothesis(framesCount - 1);

			std::stringstream bld;
			poolObserver.dumpTrackHistory(bld);
			LOG4CXX_DEBUG(log_, "Tracks Result " << bld.str());

			auto tracksCount = poolObserver.trackHistoryCount();
			Assert::AreEqual(0, tracksCount, L"There must be zero tracks when there is no observations");
		}
	};
}