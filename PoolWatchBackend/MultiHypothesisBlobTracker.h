#pragma once
#include <set>
#include <opencv2/core.hpp>

#include <log4cxx/logger.h>

#include <boost/filesystem/path.hpp>

#include "TrackHypothesisTreeNode.h"

/** Represents a tracker which keeps a history of hypothesis to determine moving objects and their positions. */
class __declspec(dllexport) MultiHypothesisBlobTracker
{
	static log4cxx::LoggerPtr log_;

	const float NullPosX = -1;

	// used to pack (frameInd,observationId) into the single int32_t value
	const int MaxObservationsCountPerFrame = 1000; // multiple of 10

	// hypothesis graph as string encoding
	const int32_t OpenBracket = -2;
	const int32_t CloseBracket = -3;
	
	const int DetectionIndNoObservation = -1;

	TrackHypothesisTreeNode trackHypothesisForestPseudoNode_;
	std::shared_ptr<CameraProjectorBase> cameraProjector_;
public:
	std::unique_ptr<SwimmerMovementPredictor> movementPredictor_;
	std::vector<DetectedBlob> prevFrameBlobs;
private:
	float fps_;
	int pruneWindow_; // the depth of track history (valid value>=1; value=0 purges all hypothesis nodes so that hypothesis tree has the single pseudo node)
	int nextTrackCandidateId_;
public: // visible to a test module
	float swimmerMaxSpeed_; // max swimmer speed in m/s (default=2.3)
	float shapeCentroidNoise_ = 0.5f; // constant to add to the max swimmer speed to get max possible swimmer shift
	int initNewTrackDelay_ = 1; // value >= 1; generate new track each N frames
private:
	boost::filesystem::path logDir_;
public:
	MultiHypothesisBlobTracker(std::shared_ptr<CameraProjectorBase> cameraProjector, int pruneWindow, float fps);
	MultiHypothesisBlobTracker(const MultiHypothesisBlobTracker& mht) = delete;
	virtual ~MultiHypothesisBlobTracker();
	void trackBlobs(int frameInd, const std::vector<DetectedBlob>& blobs, float fps, float elapsedTimeMs, int& frameIndWithTrackInfo, std::vector<TrackChangePerFrame>& trackStatusList);

	float getFps() { return fps_;  }

private:
	void growTrackHyposhesisTree(int frameInd, const std::vector<DetectedBlob>& blobs, float fps, float elapsedTimeMs);

	// "Correspondence" hypothesis2: track has correspondent observations in this frame
	void makeCorrespondenceHypothesis(int frameInd, TrackHypothesisTreeNode* leafHyp, const std::vector<DetectedBlob>& blobs, float elapsedTimeMs, int& addedDueCorrespondence, std::map<int, std::vector<TrackHypothesisTreeNode*>>& observationIndToHypNodes);

	// "No observation" hypothesis: track has no observation in this frame
	void makeNoObservationHypothesis(int frameInd, TrackHypothesisTreeNode* leafHyp, float elapsedTimeMs, int& addedDueNoObservation, int noObsOrder, int blobsCount);

	// "New track" hypothesis - track got the initial observation in this frame
	void makeNewTrackHypothesis(int frameInd, const std::vector<DetectedBlob>& blobs, int& addedNew, std::map<int, std::vector<TrackHypothesisTreeNode*>>& observationIndToHypNodes);

#if DO_CACHE_ICL
	void updateIncompatibilityLists(const std::vector<TrackHypothesisTreeNode*>& oldLeafSet, std::map<int, std::vector<TrackHypothesisTreeNode*>>& observationIndToHypNodes);
	void propagateIncompatibilityListsOnTreeGrowth(TrackHypothesisTreeNode* pOldHyp, const std::vector<TrackHypothesisTreeNode*>& oldLeafSet, const std::map<int, TrackHypothesisTreeNode*>& nodeIdToPtr);
	void propagateIncompatibilityListsOnTreeGrowthOne(const TrackHypothesisTreeNode& oldHyp, TrackHypothesisTreeNode& oldHypChild, const std::map<int, TrackHypothesisTreeNode*>& nodeIdToPtr);
	void updateIncompatibilityListsOnHypothesisPruning(const std::vector<TrackHypothesisTreeNode*>& leavesAfterPruning, const std::set<int>& nodeIdSetAfterPruning);

	// Performs incompatibility lists (ICL) validation.
	void validateIncompatibilityLists();

	void setPairwiseObservationIndConflicts(std::vector<TrackHypothesisTreeNode*>& hyps, int conflictFrameInd, int conflictObsInd, bool allowConflictsForTheSameParent, bool forceUniqueCollisionsOnly);

	// The naive implementation of Maximum Weight Independent Set Problem (MWISP) using hash table to access track hypothesis nodes.
	void maximumWeightIndependentSetNaiveMaxFirstCpp(const std::vector<int32_t>& connectedVertices, const std::map<int32_t, TrackHypothesisTreeNode*>& treeNodeIdToNode, std::vector<bool>& indepVertexSet);
#endif

	int compoundObservationId(const TrackHypothesisTreeNode& node);

	void hypothesisTreeToTreeStringRec(const TrackHypothesisTreeNode& startFrom, std::vector<int32_t>& encodedTreeString);
#if DO_CACHE_ICL
	void createTrackIncompatibilityGraphUsingPerNodeICL(const std::vector<TrackHypothesisTreeNode*>& leafSet, const std::map<int, TrackHypothesisTreeNode*>& nodeIdToNode, std::vector<int32_t>& incompatibTrackEdges);
#else
	void createTrackIncopatibilityGraphDLang(const std::vector<int32_t>& encodedTreeString, std::vector<int32_t>& incompGraphEdgePairs) const;
	void validateConformanceDLangImpl();
#endif

	void findBestTracks(const std::vector<TrackHypothesisTreeNode*>& leafSet,
		std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs);

	TrackChangePerFrame createTrackChange(TrackHypothesisTreeNode* pNode);

	// Truncates hypothesis tree to keep the tree depth fixed and equal 'pruneWindow'. There can be a track change for a track for some former frame.
	// Note, the changes from different tracks may be created for different previous frames.
	// The earliest frame index from all track changes is 'readyFrameInd'.
	void pruneHypothesisTree(int frameInd, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs, int& readyFrameInd, std::vector<TrackChangePerFrame>& trackChanges, int pruneWindow);

	void pruneLowScoreTracks(std::vector<TrackChangePerFrame>& trackChanges);

	// Finds the old and new root pair for given track path starting at leaf.
	TrackHypothesisTreeNode* findNewFamilyRoot(TrackHypothesisTreeNode* leaf, int pruneWindow);

	//void enumerateBranchNodesReversed(TrackHypothesisTreeNode* leaf, int pruneWindow, std::vector<TrackHypothesisTreeNode*>& result) const;

public:
	// TODO: this method should return "hypothesis tree finalizer" object and client will query it to get the next layer of oldest track nodes
	// Such finalizer object will reuse the set of bestTrackLeafs.
	bool flushTrackHypothesis(int frameInd, int& frameIndWithTrackInfo, std::vector<TrackChangePerFrame>& trackChangeList,
		std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs, bool isBestTrackLeafsInitied, int& pruneWindow);
	void getMostPossibleHypothesis(int frameInd, std::vector<TrackHypothesisTreeNode*>& hypList);
private:
	inline bool isPseudoRoot(const TrackHypothesisTreeNode& node) const;
	void getLeafSet(TrackHypothesisTreeNode* startNode, std::vector<TrackHypothesisTreeNode*>& leafSet);
	void getSubtreeSet(TrackHypothesisTreeNode* startNode, std::vector<TrackHypothesisTreeNode*>& subtreeSet);
public:
	void logVisualHypothesisTree(int frameInd, const std::string& fileNameTag, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs) const;
public:
	void setLogDir(const boost::filesystem::path& dir)
	{
		logDir_ = dir;
	}

public:
	void setMovementPredictor(std::unique_ptr<SwimmerMovementPredictor> movementPredictor);
};
