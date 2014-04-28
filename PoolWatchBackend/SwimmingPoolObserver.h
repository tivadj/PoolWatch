#pragma once
#include <vector>
#include <sstream>
#include <map>
#include <memory>
#include <tuple>
#include <string>

#include <log4cxx/logger.h>

#include <boost/filesystem/path.hpp>

#include "PoolWatchFacade.h"
#include "MultiHypothesisBlobTracker.h"
#include "SwimmerDetector.h"

/** Decorates frame images with tracks pathes and swimmer shapes. */
class __declspec(dllexport) SwimmingPoolObserver
{
	static log4cxx::LoggerPtr log_;
public:
	std::vector<std::vector<DetectedBlob>> blobsPerFrame_;
private:
	std::map<int, TrackInfoHistory> trackIdToHistory_;
	std::unique_ptr<MultiHypothesisBlobTracker> blobTracker_;
	std::shared_ptr<CameraProjectorBase> cameraProjector_;
	SwimmerDetector swimmerDetector_;
	std::unique_ptr<WaterClassifier> waterClassifier_;
	std::vector<DetectedBlob> expectedBlobs_;
public:
	int trackMinDurationFrames_ = 2; // value>=1; each track must be longer (in frames) than this threshold
	std::function<void(const std::vector<DetectedBlob>& blobs , const cv::Mat& imageFamePoolOnly)> BlobsDetected;
private:
#if PW_DEBUG
	std::shared_ptr<boost::filesystem::path> logDir_;
#endif
public:
	SwimmingPoolObserver(std::unique_ptr<MultiHypothesisBlobTracker> blobTracker, std::shared_ptr<CameraProjectorBase> cameraProjector);
	SwimmingPoolObserver(const SwimmingPoolObserver& tp) = delete;
	std::tuple<bool, std::string> init();
	virtual ~SwimmingPoolObserver();
	void setBlobs(size_t frameOrd, const std::vector<DetectedBlob>& blobs);
	void toString(std::stringstream& bld);
	void dumpTrackHistory(std::stringstream& bld) const;
	void saveSequentialTrackChanges(const std::vector<TrackChangePerFrame>& trackChanges);
	void adornImage(int frameOrd, int trailLength, cv::Mat& resultImage);

	void processCameraImage(size_t frameOrd, const cv::Mat& image, int* pReadyFrameInd);

	void processBlobs(size_t frameOrd, const std::vector<DetectedBlob>& blobs, int* pFrameIndWithTrackInfo = nullptr);
	
	void flushTrackHypothesis(int frameInd);
	const TrackInfoHistory* trackHistoryForBlob(int frameInd, int blobInd) const;
	
	int toLocalAssignmentIndex(const TrackInfoHistory& track, int frameInd) const;

	std::shared_ptr<CameraProjectorBase> cameraProjector();
	int trackHistoryCount() const;
	
	void predictNextFrameBlobs(int frameOrd, const std::vector<DetectedBlob>& blobs, std::vector<DetectedBlob>& expectedBlobs);
private:
	static cv::Scalar getTrackColor(const TrackInfoHistory& trackHist);
	void adornImageInternal(int fromFrameOrd, int toFrameOrd, int trailLength, cv::Mat& resultImage);
public:
#if PW_DEBUG
	void setLogDir(std::shared_ptr<boost::filesystem::path> dir)
	{
		logDir_ = dir;
		blobTracker_->setLogDir(dir);
	}
#endif
};

