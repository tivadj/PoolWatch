#pragma once
#include <vector>
#include <sstream>
#include <map>
#include <memory>

#include <boost/filesystem/path.hpp>

#include "PoolWatchFacade.h"
#include "MultiHypothesisBlobTracker.h"

/** Decorates frame images with tracks pathes and swimmer shapes. */
class __declspec(dllexport) SwimmingPoolObserver
{
	std::vector<std::vector<DetectedBlob>> blobsPerFrame_;
	std::map<int, TrackInfoHistory> trackIdToHistory_;
	std::unique_ptr<MultiHypothesisBlobTracker> blobTracker_;
	std::shared_ptr<CameraProjectorBase> cameraProjector_;
public:
	int trackMinDurationFrames_ = 2; // value>=1; each track must be longer (in frames) than this threshold
private:
#if PW_DEBUG
	std::shared_ptr<boost::filesystem::path> logDir_;
#endif
public:
	SwimmingPoolObserver(std::unique_ptr<MultiHypothesisBlobTracker> blobTracker, std::shared_ptr<CameraProjectorBase> cameraProjector);
	SwimmingPoolObserver(const SwimmingPoolObserver& tp) = delete;
	virtual ~SwimmingPoolObserver();
	void setBlobs(size_t frameOrd, const std::vector<DetectedBlob>& blobs);
	void toString(std::stringstream& bld);
	void dumpTrackHistory(std::stringstream& bld) const;
	void saveSequentialTrackChanges(const std::vector<TrackChangePerFrame>& trackChanges);
	void adornImage(int frameOrd, int trailLength, cv::Mat& resultImage);

	void processBlobs(size_t frameOrd, const std::vector<DetectedBlob>& blobs, int* pFrameIndWithTrackInfo = nullptr);
	
	void flushTrackHypothesis(int frameInd);
	const TrackInfoHistory* trackHistoryForBlob(int frameInd, int blobInd);
	
	int toLocalAssignmentIndex(const TrackInfoHistory& track, int frameInd) const;

	std::shared_ptr<CameraProjectorBase> cameraProjector();
	int trackHistoryCount() const;
	
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

__declspec(dllexport) void getHumanBodies(const cv::Mat& image, const cv::Mat_<uchar>& waterMask, std::vector<DetectedBlob>& blobs);