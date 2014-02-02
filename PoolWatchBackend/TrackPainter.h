#pragma once
#include <vector>
#include <sstream>
#include <map>

#include "PoolWatchFacade.h"

/** Decorates frame images with tracks pathes and swimmer shapes. */
class TrackPainter
{
	std::vector<std::vector<DetectedBlob>> blobsPerFrame_;
	std::map<int, TrackInfoHistory> trackIdToHistory_;
public:
	TrackPainter();
	~TrackPainter();
	void setBlobs(size_t frameOrd, const std::vector<DetectedBlob>& blobs);
	void toString(std::stringstream& bld);
	void setTrackChangesPerFrame(int frameOrd, const std::vector<TrackChangePerFrame>& trackChanges);
	void adornImage(const cv::Mat& image, int frameOrd, int trailLength, cv::Mat& resultImage);
	void adornImageInternal(const cv::Mat& image, int fromFrameOrd, int toFrameOrd, int trailLength, cv::Mat& resultImage);
	
	static cv::Scalar getTrackColor(const TrackInfoHistory& trackHist);
};

