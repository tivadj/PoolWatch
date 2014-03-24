#pragma once
#include <vector>
#include <sstream>
#include <map>
#include <memory>

#include "PoolWatchFacade.h"
#include "MultiHypothesisBlobTracker.h"

/** Decorates frame images with tracks pathes and swimmer shapes. */
class POOLWATCH_API SwimmingPoolObserver
{
	std::vector<std::vector<DetectedBlob>> blobsPerFrame_;
	std::map<int, TrackInfoHistory> trackIdToHistory_;
	std::unique_ptr<MultiHypothesisBlobTracker> blobTracker_;
	std::shared_ptr<CameraProjector> cameraProjector_;
public:
	SwimmingPoolObserver(int pruneWindow, float fps);
	SwimmingPoolObserver(const SwimmingPoolObserver& tp) = delete;
	virtual ~SwimmingPoolObserver();
	void setBlobs(size_t frameOrd, const std::vector<DetectedBlob>& blobs);
	void toString(std::stringstream& bld);
	void setTrackChangesPerFrame(int frameOrd, const std::vector<TrackChangePerFrame>& trackChanges);
	void adornImage(const cv::Mat& image, int frameOrd, int trailLength, cv::Mat& resultImage);

	void processBlobs(size_t frameOrd, const cv::Mat& image, const std::vector<DetectedBlob>& blobs, int* pFrameIndWithTrackInfo = nullptr);

	std::shared_ptr<CameraProjector> cameraProjector();
	
private:
	static cv::Scalar getTrackColor(const TrackInfoHistory& trackHist);
	void adornImageInternal(const cv::Mat& image, int fromFrameOrd, int toFrameOrd, int trailLength, cv::Mat& resultImage);
};

__declspec(dllexport) void getHumanBodies(const cv::Mat& image, const cv::Mat_<uchar>& waterMask, std::vector<DetectedBlob>& blobs);