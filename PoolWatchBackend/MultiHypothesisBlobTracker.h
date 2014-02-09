#pragma once
#include <opencv2/core.hpp>
#include "opencv2/video/tracking.hpp" // cv::KalmanFilter

#include "PoolWatchFacade.h"
#include "TrackHypothesisTreeNode.h"
#include "CameraProjector.h"

/** Represents a tracker which keeps a history of hypothesis to determine moving objects and their positions. */
class MultiHypothesisBlobTracker
{
	const float NullPosX = -1;

	// used to pack (frameInd,observationId) into the single int32_t value
	const int maxObservationsCountPerFrame = 1000; // multiple of 10

	// hypothesis graph as string encoding
	const int32_t openBracket = -1;
	const int32_t closeBracket = -2;
	
	const int DetectionIndNoObservation = -1;
	const int KalmanFilterDynamicParamsCount = 4;
	const float zeroHeight = 0.0f;

	std::shared_ptr<CameraProjector> cameraProjector_;
	TrackHypothesisTreeNode trackHypothesisForestPseudoNode_;
	float fps_;
	int pruneWindow_;
	int nextTrackCandidateId_;
	float swimmerMaxSpeed_;

	// cached objects
	cv::KalmanFilter kalmanFilter_;
public:
	MultiHypothesisBlobTracker(std::shared_ptr<CameraProjector> cameraProjector, int pruneWindow, float fps);
	MultiHypothesisBlobTracker(const MultiHypothesisBlobTracker& mht) = delete;
	virtual ~MultiHypothesisBlobTracker();
	void trackBlobs(int frameInd, const std::vector<DetectedBlob>& blobs, const cv::Mat& image, float fps, float elapsedTimeMs, int& frameIndWithTrackInfo, std::vector<TrackChangePerFrame>& trackStatusList);

	float getFps() { return fps_;  }

private:
	void growTrackHyposhesisTree(int frameInd, const std::vector<DetectedBlob>& blobs, float fps, float swimmerMaxShiftPerFrameM);

	int compoundObservationId(const TrackHypothesisTreeNode& node);

	void hypothesisTreeToTreeStringRec(const TrackHypothesisTreeNode& startFrom, std::vector<int32_t>& encodedTreeString);
	mxArray* createTrackIncopatibilityGraphDLang(const std::vector<int32_t>& encodedTreeString);

	void findBestTracks(const std::vector<TrackHypothesisTreeNode*>& leafSet,
		std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs);

	float calcTrackShiftScore(const TrackHypothesisTreeNode* parentNode,
		const TrackHypothesisTreeNode* trackNode, float fps);

	void calcTrackShiftScoreNew(const TrackHypothesisTreeNode* parentNode,
		const cv::Point3f& blobPosWorld, TrackHypothesisCreationReason hypothesisReason, float fps,
		float& resultScore,
		cv::Point3f& resultEstimatedPosWorld,
		cv::Mat_<float>& resultKalmanFilterState,
		cv::Mat_<float>& resultKalmanFilterStateCovariance);

	int collectTrackChanges(int frameInd, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs, std::vector<TrackChangePerFrame>& trackStatusList);

	void pruneHypothesisTree(const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs);
	std::unique_ptr<TrackHypothesisTreeNode> findNewFamilyRoot(TrackHypothesisTreeNode* leaf);

private:
	inline bool isPseudoRoot(const TrackHypothesisTreeNode& node);
	void getLeafSet(TrackHypothesisTreeNode* startNode, std::vector<TrackHypothesisTreeNode*>& leafSet);

	void initKalmanFilter(cv::KalmanFilter& kalmanFilter, float fps);
};

float kalmanFilterDistance(const cv::KalmanFilter& kalmanFilter, const cv::Mat& observedPos);
