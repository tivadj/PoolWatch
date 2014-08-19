#pragma once
#include <array>
#include <vector>
#include <memory>
#include <tuple>
#include <opencv2/core.hpp>
#include <boost/optional.hpp>

#include "PoolWatchFacade.h"
#include "AppearanceModel.h"

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
	static const int AppearanceGmmMaxSize = ColorSignatureGmmMaxSize;

	// NOTE: must bitwise match corresponding DLang structure
	// struct {
	int32_t Id;
	float Score; // determines validity of the hypothesis(from root to this node); the more is better
	int32_t FrameInd;
	int ObservationOrNoObsId = -1; // >= 0 (eg 0,1,2) for valid observation; <0 (eg -1,-2,-3) to mean no observation for each hypothesis node
	TrackHypothesisTreeNode** ChildrenArray = nullptr; // pointer to the array of children; must be in sync with Children[0]
	int32_t ChildrenCount = 0; // must be in sync with Children.size()
	TrackHypothesisTreeNode* Parent = nullptr;
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
	cv::Mat_<float> KalmanFilterState; // [X, Y, vx, vy] in meters and m/sec
	cv::Mat_<float> KalmanFilterStateCovariance;  // [4x4]
	
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

// Estimates the position of the blob given the state of the blob (position etc).
class SwimmerMovementPredictor
{
public:
	virtual void initScoreAndState(int frameInd, int observationInd, const cv::Point3f& blobCentrWorld, float& score, TrackHypothesisTreeNode& saveNode) = 0;

	// Estimates position and score of movement from position of curNode to blobCentrWorld. Saves state into saveNode.
	// deltaMovementScore = delta score of shifting from current node to proposed blob's center.
	virtual void estimateAndSave(const TrackHypothesisTreeNode& curNode, const boost::optional<cv::Point3f>& blobCentrWorld, cv::Point3f& estPos, float& deltaMovementScore, TrackHypothesisTreeNode& saveNode) = 0;
};
