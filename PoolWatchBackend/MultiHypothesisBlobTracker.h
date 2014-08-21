#pragma once
#include <set>
#include <map>
#include <opencv2/core.hpp>

#include <log4cxx/logger.h>

#include <boost/filesystem/path.hpp>

#include "TrackHypothesisTreeNode.h"
#include "AppearanceModel.h"

/// Debug time helper class to keep a trail of upddates to a track.
struct TrackChangeClientSide
{
	int TrackId;
	int FrameIndOnNew;
	bool IsLive = true;
	int EndFrameInd; // next sequential track change should be associated with this frameInd

	TrackChangeClientSide(int trackId, int frameIndOnNew);

	/// Terminates the track. Further track changes are not possible.
	void terminate();

	/// Associates next track change with this track.
	void setNextTrackChange(int changeFrameInd);
};

/// Debug time helper class to check that tracker returns consistent sequential track changes to the client.
struct TrackChangeConsistencyChecker
{
	std::map<int, TrackChangeClientSide> trackIdToObj_;
	int FrameIndOnNew;
	void setNextTrackChanges(int readyFrameInd, const std::vector<TrackChangePerFrame>& trackChanges);
};

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

	static const int DetectionIndNoObservation = TrackHypothesisTreeNode::DetectionIndNoObservation;

	TrackHypothesisTreeNode trackHypothesisForestPseudoNode_;
	std::shared_ptr<CameraProjectorBase> cameraProjector_;
public:
	std::unique_ptr<SwimmerMovementPredictor> movementPredictor_;
	std::vector<DetectedBlob> prevFrameBlobs;
private:
	float fps_;
	int pruneWindow_; // the depth of track history (valid value>=1; value=0 purges all hypothesis nodes so that hypothesis tree has the single pseudo node)
	int nextTrackCandidateId_;
	float swimmerMaxSpeed_; // max swimmer speed in m/s (default=2.3)
public: // visible to a test module
	// 0 = blobs from two consequent frames can't be assiciated with a single track and new track is created
	// 0.4 = ok, track doesn't jump to the swimmer moving in opposite direction
	// 0.5 = too much, track may jump to a swimmer which moves in oppositite direction
	float shapeCentroidNoise_ = 0.5f; // constant to add to the max swimmer speed to get max possible swimmer shift
	int initNewTrackDelay_ = 1; // value >= 1; generate new track each N frames
	float trackMinScore_ = -15;
private:
	boost::filesystem::path logDir_;
#if PW_DEBUG
	TrackChangeConsistencyChecker trackChangeChecker_;
#endif
	std::unique_ptr<AppearanceModel> appearanceGmmEstimator_;
public:
	MultiHypothesisBlobTracker(std::shared_ptr<CameraProjectorBase> cameraProjector, int pruneWindow, float fps);
	MultiHypothesisBlobTracker(const MultiHypothesisBlobTracker& mht) = delete;
	virtual ~MultiHypothesisBlobTracker();
	void trackBlobs(int frameInd, const std::vector<DetectedBlob>& blobs, float fps, float elapsedTimeMs, int& frameIndWithTrackInfo, std::vector<TrackChangePerFrame>& trackStatusList);

	// TODO: this method should return "hypothesis tree finalizer" object and client will query it to get the next layer of oldest track nodes
	// Such finalizer object will reuse the set of bestTrackLeafs.
	bool flushTrackHypothesis(int frameInd, int& frameIndWithTrackInfo, std::vector<TrackChangePerFrame>& trackChangeList,
		std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs, bool isBestTrackLeafsInitied, int& pruneWindow);

	/// Propogates track with given Id on the given frameInd in the future.
	/// Returns true if track was found.
	bool getLatestHypothesis(int frameInd, int trackFamilyId, TrackChangePerFrame& outTrackChange);
	void getMostPossibleHypothesis(int frameInd, std::vector<TrackHypothesisTreeNode*>& hypList);

	/// Gets the index of the ready frame for given frameInd if pruning is performed with pruneWindow.
	static int getReadyFrameInd(int frameInd, int pruneWindow);

	void logVisualHypothesisTree(int frameInd, const std::string& fileNameTag, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs) const;

	void setLogDir(const boost::filesystem::path& dir);

	void setMovementPredictor(std::unique_ptr<SwimmerMovementPredictor> movementPredictor);

	float getFps() { return fps_; }

	void setSwimmerMaxSpeed(float swimmerMaxSpeed);

private:
	void growTrackHyposhesisTree(int frameInd, const std::vector<DetectedBlob>& blobs, float fps, float elapsedTimeMs);

	// Updates the node so all its fields are consistent.
	// Some fields, like ChildrensCount or FirstChild, duplicate data in other fields and must be consistent.
	// These helper fields are used in DLang code.
	static void fixHypNodeConsistency(TrackHypothesisTreeNode* pNode);
	
	// Ensures that each node in the hypothesis tree is consistent.
	void checkHypNodesConsistency();

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

	//int compoundObservationId(const TrackHypothesisTreeNode& node);

	void hypothesisTreeToTreeStringRec(const TrackHypothesisTreeNode& startFrom, std::vector<int32_t>& encodedTreeString);
#if DO_CACHE_ICL
	void createTrackIncompatibilityGraphUsingPerNodeICL(const std::vector<TrackHypothesisTreeNode*>& leafSet, const std::map<int, TrackHypothesisTreeNode*>& nodeIdToNode, std::vector<int32_t>& incompatibTrackEdges);
#else
	void createTrackIncopatibilityGraphDLang(const std::vector<int32_t>& encodedTreeString, std::vector<int32_t>& incompGraphEdgePairs) const;
	void createTrackIncopatibilityGraphDLangDirectAccess(std::vector<int32_t>& incompGraphEdgePairs) const;
	void validateConformanceDLangImpl();
#endif

	void findBestTracks(const std::vector<TrackHypothesisTreeNode*>& leafSet, std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs);
	void findBestTracksCpp(const std::vector<TrackHypothesisTreeNode*>& leafSet, std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs);
	void findBestTracksDLang(std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs);

	TrackChangePerFrame createTrackChange(TrackHypothesisTreeNode* pNode);
	void populateTrackChange(TrackHypothesisTreeNode* pNode, TrackChangePerFrame& result);

	// Truncates hypothesis tree to keep the tree depth fixed and equal 'pruneWindow'. There can be a track change for a track for some former frame.
	// Note, the changes from different tracks may be created for different previous frames.
	// The earliest frame index from all track changes is 'readyFrameInd'.
	void pruneHypothesisTreeAndGetTrackChanges(int frameInd, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs, int& readyFrameInd, std::vector<TrackChangePerFrame>& trackChanges, int pruneWindow);

	void pruneLowScoreTracks(int frameInd, std::vector<TrackChangePerFrame>& trackChanges);

	// Finds the old and new root pair for given track path starting at leaf.
	TrackHypothesisTreeNode* findNewFamilyRoot(TrackHypothesisTreeNode* leaf, int pruneWindow);

	//void enumerateBranchNodesReversed(TrackHypothesisTreeNode* leaf, int pruneWindow, std::vector<TrackHypothesisTreeNode*>& result) const;

	/// Gets the sequence of observation statuses for given track hypothesis for the last lastFramesCount frames.
	std::string latestObservationStatus(const TrackHypothesisTreeNode& leafNode, int lastFramesCount, TrackHypothesisTreeNode* leafParentOrNull = nullptr);

	TrackHypothesisTreeNode* findHypothesisWithFrameInd(TrackHypothesisTreeNode* start, int frameInd, int trackFamilyId);

	inline bool isPseudoRoot(const TrackHypothesisTreeNode& node) const;
	void getLeafSet(TrackHypothesisTreeNode* startNode, std::vector<TrackHypothesisTreeNode*>& leafSet);
	void getSubtreeSet(TrackHypothesisTreeNode* startNode, std::vector<TrackHypothesisTreeNode*>& subtreeSet);
};
