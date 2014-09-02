#pragma once
#include <set>
#include <map>
#include <array>
#include <opencv2/core.hpp>

#include <log4cxx/logger.h>

#include <boost/filesystem/path.hpp>

#include "VisualObservationIf.h"
#include "CameraProjector.h"
#include "SwimmerMovementModel.h"

enum class TrackHypothesisCreationReason
{
	SequantialCorrespondence = 1,  // this hypothesis has corresponding observation in the previous frame
	New = 2,                       // this is brand new hypothesis
	NoObservation = 3              // this hypothesis assumes that there is no observation of the tracked object in this frame
};

std::string toString(TrackHypothesisCreationReason reason);

#if DO_CACHE_ICL
// Contains information about the conflicting observation for two hypothesis.
struct ObservationConflict
{
	int FirstNodeId; // the id of this node (first node in the collision nodes pair)
	int OtherFamilyRootId = -1; // the node and all its descendants with which first (this) node conflicts

	int FrameInd; // frame with conflicting observation

	// the observation index for which two hypothesis conflict; in case of multiple frames with conflicting observations, 
	// this observation is for the lowest frameInd.
	// The ObservationInd=-1 means that two hypothesis are children of the same parent which is a 'no observation' hypothesis
	int ObservationInd;

	ObservationConflict(int frameInd, int observationInd, int firstNodeId, int otherFamilyRootId) : FrameInd(frameInd), ObservationInd(observationInd), OtherFamilyRootId(otherFamilyRootId), FirstNodeId(firstNodeId) {}
};
#endif

/** Represents node in the tree of hypothesis. */
struct TrackHypothesisTreeNode
{
	static const int DetectionIndNoObservation = -1;
	static const int AppearanceGmmMaxSize = PoolWatch::ColorSignatureGmmMaxSize;

	// NOTE: must bitwise match corresponding DLang structure
	// struct {
	int32_t Id;
	float Score; // determines validity of the hypothesis(from root to this node); the more is better
	int32_t FrameInd;
	int ObservationOrNoObsId = -1; // >= 0 (eg 0,1,2) for valid observation; <0 (eg -1,-2,-3) to mean no observation for each hypothesis node
	TrackHypothesisTreeNode** ChildrenArray = nullptr; // pointer to the array of children; must be in sync with Children[0]
	int32_t ChildrenCount = 0; // must be in sync with Children.size()
	TrackHypothesisTreeNode* Parent = nullptr;

	// Used by DCode in MWISP algorithm.
	// Stores a pointer to corresponding node in a collision graph of track hypothesis nodes.
	void* MwispNode = nullptr;
	// }

	int FamilyId;
	int ObservationInd; // >= 0 (eg 0,1,2) for valid observation; -1 for 'no observation'
	std::vector<std::unique_ptr<TrackHypothesisTreeNode>> Children;
	cv::Point2f ObservationPos;      // in pixels
	cv::Point3f ObservationPosWorld; // in meters
	cv::Point3f EstimatedPosWorld; // in meters
	TrackHypothesisCreationReason CreationReason;
#if DO_CACHE_ICL
	std::vector<ObservationConflict> IncompatibleNodes; // the list of nodes, this node is incompatible with
#endif
	cv::Matx41f KalmanFilterState; // [X, Y, vx, vy] in meters and m/sec
	cv::Matx44f KalmanFilterStateCovariance;  // [4x4]

	//#if PW_DEBUGXXX
	int Age = 0;  // frames
	//#endif

	// appearance data
	std::array<GaussMixtureCompoenent, AppearanceGmmMaxSize> AppearanceGmm;
	int AppearanceGmmCount = 0;

public:
	void addChildNode(std::unique_ptr<TrackHypothesisTreeNode> childHyp);
	TrackHypothesisTreeNode* getAncestor(int ancestorIndex);
	std::unique_ptr<TrackHypothesisTreeNode> pullChild(TrackHypothesisTreeNode* pChild, bool updateChildrenCollection = false);
};

// Enumerates nodes from leaf to root but no more than pruneWindow nodes.
// For takeCount==1 returns the sequence of one element - the leaf node itself.
void enumerateBranchNodesReversed(TrackHypothesisTreeNode* leaf, int pruneWindow, std::vector<TrackHypothesisTreeNode*>& result, TrackHypothesisTreeNode* leafParentOrNull = nullptr);


enum TrackChangeUpdateType
{
	// Notifies about the creation of a new track. 
	New = 1,

	// TODO: should NoObservation and ObservationUpdate be merged?
	ObservationUpdate,
	NoObservation,

	// Notifies that the track is removed from the tracker. There will be no further updates for this track.
	Pruned
};

void toString(TrackChangeUpdateType trackChange, std::string& result);

struct TrackChangePerFrame
{
	int FamilyId;
	TrackChangeUpdateType UpdateType;
	cv::Point3f EstimatedPosWorld; // [X, Y, Z] corrected by sensor position(in world coord)

	int ObservationInd; // 0 = no observation; >0 observation index
	cv::Point2f ObservationPosPixExactOrApprox; // [X, Y]; required to avoid world->camera conversion on drawing

	int FrameInd; // used for debugging
	float Score; // used for debugging
};

struct TrackInfoHistory
{
	static const int IndexNull = -1;

	//int Id;
	//bool IsTrackCandidate; // true = TrackCandidate
	int TrackCandidateId;
	int FirstAppearanceFrameIdx;
	int LastAppearanceFrameIdx; // inclusive
	//PromotionFramdInd; // the frame when candidate was promoted to track

	std::vector<TrackChangePerFrame> Assignments;

	bool TrackInfoHistory::isFinished() const;
	const TrackChangePerFrame* getTrackChangeForFrame(int frameOrd) const;
};

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
class PW_EXPORTS MultiHypothesisBlobTracker
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
	std::unique_ptr<SwimmerAppearanceModelBase> swimmerAppearanceModel_;
public:
	MultiHypothesisBlobTracker(std::shared_ptr<CameraProjectorBase> cameraProjector, int pruneWindow, float fps);
	MultiHypothesisBlobTracker(const MultiHypothesisBlobTracker& mht) = delete;
	virtual ~MultiHypothesisBlobTracker();

	// TODO: do we need FPS or ElapsedTime?
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
	void getSubtreeSet(TrackHypothesisTreeNode* startNode, std::vector<TrackHypothesisTreeNode*>& subtreeSet, bool includePseudoRoot);
};
