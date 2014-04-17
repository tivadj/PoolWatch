#pragma once
#include <opencv2/core.hpp>

#include <log4cxx/logger.h>

#include <boost/filesystem/path.hpp>

#include "PoolWatchFacade.h"
#include "TrackHypothesisTreeNode.h"
#include "KalmanFilterMovementPredictor.h"

/** Represents a tracker which keeps a history of hypothesis to determine moving objects and their positions. */
class __declspec(dllexport) MultiHypothesisBlobTracker
{
	static log4cxx::LoggerPtr log_;

	const float NullPosX = -1;

	// used to pack (frameInd,observationId) into the single int32_t value
	const int maxObservationsCountPerFrame = 1000; // multiple of 10

	// hypothesis graph as string encoding
	const int32_t openBracket = -1;
	const int32_t closeBracket = -2;
	
	const int DetectionIndNoObservation = -1;

	TrackHypothesisTreeNode trackHypothesisForestPseudoNode_;
	std::shared_ptr<CameraProjectorBase> cameraProjector_;
	std::unique_ptr<SwimmerMovementPredictor> movementPredictor_;
	std::vector<DetectedBlob> prevFrameBlobs;
	float fps_;
	int pruneWindow_; // the depth of track history (valid value>=1; value=0 purges all hypothesis nodes so that hypothesis tree has the single pseudo node)
	int nextTrackCandidateId_;
public: // visible to a test module
	float swimmerMaxSpeed_; // max swimmer speed in m/s (default=2.3)
	float shapeCentroidNoise_; // constant to add to the max swimmer speed to get max possible swimmer shift
	int initNewTrackDelay_ = 7; // value >= 1; generate new track each N frames
private:
#if PW_DEBUG
	std::shared_ptr<boost::filesystem::path> logDir_;
#endif
public:
	MultiHypothesisBlobTracker(std::shared_ptr<CameraProjectorBase> cameraProjector, int pruneWindow, float fps);
	MultiHypothesisBlobTracker(const MultiHypothesisBlobTracker& mht) = delete;
	virtual ~MultiHypothesisBlobTracker();
	void trackBlobs(int frameInd, const std::vector<DetectedBlob>& blobs, float fps, float elapsedTimeMs, int& frameIndWithTrackInfo, std::vector<TrackChangePerFrame>& trackStatusList);

	float getFps() { return fps_;  }

private:
	void growTrackHyposhesisTree(int frameInd, const std::vector<DetectedBlob>& blobs, float fps, float swimmerMaxShiftPerFrameM);

	int compoundObservationId(const TrackHypothesisTreeNode& node);

	void hypothesisTreeToTreeStringRec(const TrackHypothesisTreeNode& startFrom, std::vector<int32_t>& encodedTreeString);
	void createTrackIncopatibilityGraphDLang(const std::vector<int32_t>& encodedTreeString, std::vector<int32_t>& incompGraphEdgePairs) const;

	void findBestTracks(const std::vector<TrackHypothesisTreeNode*>& leafSet,
		std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs);

	TrackChangePerFrame createTrackChange(TrackHypothesisTreeNode* pNode);

	// Truncates hypothesis tree to keep the tree depth fixed and equal 'pruneWindow'. There can be a track change for a track for some former frame.
	// Note, the changes from different tracks may be created for different previous frames.
	// The earliest frame index from all track changes is 'readyFrameInd'.
	void pruneHypothesisTree(int frameInd, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs, int& readyFrameInd, std::vector<TrackChangePerFrame>& trackChanges, int pruneWindow);

	// Finds the old and new root pair for given track path starting at leaf.
	TrackHypothesisTreeNode* findNewFamilyRoot(TrackHypothesisTreeNode* leaf, int pruneWindow);

	// Enumerates nodes from leaf to root but no more than pruneWindow nodes.
	void enumerateBranchNodesReversed(TrackHypothesisTreeNode* leaf, int pruneWindow, std::vector<TrackHypothesisTreeNode*>& result) const;

public:
	// TODO: this method should return "hypothesis tree finalizer" object and client will query it to get the next layer of oldest track nodes
	// Such finalizer object will reuse the set of bestTrackLeafs.
	bool flushTrackHypothesis(int frameInd, int& frameIndWithTrackInfo, std::vector<TrackChangePerFrame>& trackChangeList,
		std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs, bool isBestTrackLeafsInitied, int& pruneWindow);
private:
	inline bool isPseudoRoot(const TrackHypothesisTreeNode& node) const;
	void getLeafSet(TrackHypothesisTreeNode* startNode, std::vector<TrackHypothesisTreeNode*>& leafSet);
public:
	void logVisualHypothesisTree(int frameInd, const std::string& fileNameTag, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs) const;
#if PW_DEBUG
public:
	void setLogDir(std::shared_ptr<boost::filesystem::path> dir)
	{
		logDir_ = dir;
	}
#endif

public:
	void setMovementPredictor(std::unique_ptr<SwimmerMovementPredictor> movementPredictor);
};
