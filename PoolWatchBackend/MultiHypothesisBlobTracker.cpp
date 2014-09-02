#include <vector>
#include <array>
#include <cassert>
#include <memory>
#include <set>
#include <stack>
#include <algorithm>
#include <functional>

#include <log4cxx/logger.h>

#include "PoolWatchFacade.h"
#include "PoolWatchDLangInterop.h"
#include "algos1.h"
#include "MultiHypothesisBlobTracker.h"
#include "KalmanFilterMovementPredictor.h"
#include "GraphVizHypothesisTreeVisualizer.h"
#include "AppearanceModel.h"

using namespace std;
using namespace log4cxx;

#if !DO_CACHE_ICL
extern "C"
{
	struct Int32PtrPair
	{
		int32_t* pFirst;
		int32_t* pLast;
	};
	Int32PtrPair computeTrackIncopatibilityGraph(const int* pEncodedTree, int encodedTreeLength, int collisionIgnoreNodeId, int openBracketLex, int closeBracketLex, int noObservationId, Int32Allocator int32Alloc);
	Int32PtrPair computeTrackIncopatibilityGraphDirectAccess(const TrackHypothesisTreeNode* pNode, int collisionIgnoreNodeId, Int32Allocator int32Alloc);
	void pwFindBestTracks(const TrackHypothesisTreeNode* pNode, int collisionIgnoreNodeId, int attemptCount, CppVectorPtrWrapper* trackHypVectorWrapper);
}
#endif

log4cxx::LoggerPtr MultiHypothesisBlobTracker::log_ = log4cxx::Logger::getLogger("PW.MultiHypothesisBlobTracker");

MultiHypothesisBlobTracker::MultiHypothesisBlobTracker(std::shared_ptr<CameraProjectorBase> cameraProjector, int pruneWindow, float fps)
	:cameraProjector_(cameraProjector),
	fps_(fps),
	pruneWindow_(pruneWindow),
	nextTrackCandidateId_(1),
	swimmerMaxSpeed_(2.3f)         // max speed for swimmers 2.3m/s
{
	CV_DbgAssert(fps > 0);
	CV_DbgAssert(pruneWindow_ >= 0);

	// root of the tree is an artificial node to gather all trees in hypothesis forest
	trackHypothesisForestPseudoNode_.Id = 0;
	
	// required for compoundId generation
	trackHypothesisForestPseudoNode_.FamilyId = 0;
	trackHypothesisForestPseudoNode_.FrameInd = 0;
	trackHypothesisForestPseudoNode_.ObservationInd = 0;
	trackHypothesisForestPseudoNode_.Parent = nullptr;
	fixHypNodeConsistency(&trackHypothesisForestPseudoNode_);

	// default to Kalman Filter
	movementPredictor_ = std::make_unique<KalmanFilterMovementPredictor>(fps, swimmerMaxSpeed_);
	swimmerAppearanceModel_ = std::make_unique<SwimmerAppearanceModel>();
}


MultiHypothesisBlobTracker::~MultiHypothesisBlobTracker()
{
}

void MultiHypothesisBlobTracker::trackBlobs(int frameInd, const std::vector<DetectedBlob>& blobs, float fps, float elapsedTimeMs, int& readyFrameInd,
	std::vector<TrackChangePerFrame>& trackChanges)
{
#if PW_DEBUG
	// check blobs
	for (const DetectedBlob& blob : blobs)
	{
		CV_Assert(blob.ColorSignatureGmmCount >= 0 && "Blob's color signature must be initialized (required for calculation of appearance score)");
	}
#endif
	growTrackHyposhesisTree(frameInd, blobs, fps, elapsedTimeMs);

	pruneLowScoreTracks(frameInd,trackChanges);

#if PW_DEBUG
	// validate the hypothesis tree after changes (after growth and pruning)
	checkHypNodesConsistency();
#endif

	vector<TrackHypothesisTreeNode*> leafSet;
	getLeafSet(&trackHypothesisForestPseudoNode_, leafSet);

	LOG4CXX_DEBUG(log_, "leafSet.count=" << leafSet.size() << " (after growth)");
	
	vector<TrackHypothesisTreeNode*> bestTrackLeafs; // 
	findBestTracks(leafSet, bestTrackLeafs);

	if (log_->isDebugEnabled())
		logVisualHypothesisTree(frameInd, "1beforePruning", bestTrackLeafs);
	
	// list the best nodes
	if (log_->isDebugEnabled())
	{
		stringstream bld;
		bld << "bestTrackLeafs.count=" << bestTrackLeafs.size();

		if (!bestTrackLeafs.empty())
			for (const TrackHypothesisTreeNode* pLeaf : bestTrackLeafs)
			{
				std::string obsStr = latestObservationStatus(*pLeaf, 5);
				bld << endl << "  FamilyId=" << pLeaf->FamilyId << " LeafId=" << pLeaf->Id << " ObsInd=" << pLeaf->ObservationInd << " " << pLeaf->ObservationPos << " Score=" << pLeaf->Score << " Age=" << pLeaf->Age << " Obs=" << obsStr;
			}
		log_->debug(bld.str());
	}

#if DO_CACHE_ICL
	validateIncompatibilityLists(); // validation after tree growth
#endif

	pruneHypothesisTreeAndGetTrackChanges(frameInd, bestTrackLeafs, readyFrameInd, trackChanges, pruneWindow_);

	if (log_->isDebugEnabled())
		logVisualHypothesisTree(frameInd, "2afterPruning", bestTrackLeafs);

	if (log_->isDebugEnabled())
	{
		// list track hypothesis changes

		stringstream bld;
		bld << "Track Changes for FrameInd>=" << readyFrameInd << " ChangesCount=" << trackChanges.size();
		if (!trackChanges.empty())
		{
			bld << endl;
			for (const TrackChangePerFrame& change : trackChanges)
			{
				std::string changeStr;
				toString(change.UpdateType, changeStr);

				bld <<"  " << changeStr << " FamilyId=" << change.FamilyId << " FrameInd=" << change.FrameInd << " ObsInd=" << change.ObservationInd << " " << change.ObservationPosPixExactOrApprox <<" Score=" <<change.Score <<endl;
			}
		}
		log_->debug(bld.str());
	}

#if DO_CACHE_ICL
	validateIncompatibilityLists(); // validation after pruning the hypothesis tree
#endif

	//
	prevFrameBlobs.resize(blobs.size());
	std::copy(begin(blobs), end(blobs), begin(prevFrameBlobs));

#if PW_DEBUG
	trackChangeChecker_.setNextTrackChanges(readyFrameInd, trackChanges);

	// validate the hypothesis nodes after pruning
	checkHypNodesConsistency();
#endif
}

void MultiHypothesisBlobTracker::makeCorrespondenceHypothesis(int frameInd, TrackHypothesisTreeNode* leafHyp, const std::vector<DetectedBlob>& blobs, float elapsedTimeMs, int& addedDueCorrespondence, std::map<int, std::vector<TrackHypothesisTreeNode*>>& observationIndToHypNodes)
{
	// associate hypothesis node with each observation

	for (int blobInd = 0; blobInd<blobs.size(); ++blobInd)
	{
		const auto& blob = blobs[blobInd];
		cv::Point3f blobCentrWorld = blob.CentroidWorld;

		// constraint: blob can't shift too much (can't leave the blob's gate)
		{
			//swimmerMaxShiftM = elapsedTimeMs * this.v.swimmerMaxSpeed / 1000 + this.humanDetector.shapeCentroidNoise;
			float swimmerMaxShiftM = (elapsedTimeMs * 0.001f) * swimmerMaxSpeed_ + shapeCentroidNoise_;

			auto dist = cv::norm(leafHyp->EstimatedPosWorld - blobCentrWorld);
			if (dist > swimmerMaxShiftM)
				continue;
		}

		// constraint: area can't change too much
		// TODO: area should be converted (and compared) to world coordinates
		// works bad for swimmer detection based on skin color compared to one based on water color
		// because skin based blobs tend to be smaller and the shape vary more dynamically
		if (false && leafHyp->ObservationInd != -1)
		{
			assert(leafHyp->ObservationInd < prevFrameBlobs.size() && "Cache of blobs for the previous frame was not updated");
			const auto& prevBlob = prevFrameBlobs[leafHyp->ObservationInd];
			auto areaChangeRatio = std::abs(blob.AreaPix - prevBlob.AreaPix) / prevBlob.AreaPix;
			const float MaxAreaChangeRatio = 0.6;
			if (areaChangeRatio > MaxAreaChangeRatio)
				continue;
		}

		//
		auto id = nextTrackCandidateId_++;
		auto hypothesisReason = TrackHypothesisCreationReason::SequantialCorrespondence;

		auto pChildHyp = make_unique<TrackHypothesisTreeNode>();
		TrackHypothesisTreeNode& childHyp = *pChildHyp;
		childHyp.Id = id;
		childHyp.FamilyId = leafHyp->FamilyId;
		childHyp.ObservationInd = blobInd;
		childHyp.ObservationOrNoObsId = blobInd;
		childHyp.FrameInd = frameInd;
		childHyp.ObservationPos = blob.Centroid;
		childHyp.ObservationPosWorld = blobCentrWorld;
		childHyp.CreationReason = hypothesisReason;
		childHyp.KalmanFilterState = cv::Mat(4, 1, CV_32FC1);
		childHyp.KalmanFilterStateCovariance = cv::Mat(4, 4, CV_32FC1);

		// compute score

		float movementScore;
		cv::Point3f estPos;
		movementPredictor_->estimateAndSave(*leafHyp, blobCentrWorld, estPos, movementScore, childHyp);

		// merge color signature
		swimmerAppearanceModel_->mergeTwoGaussianMixtures(leafHyp->AppearanceGmm.data(), leafHyp->AppearanceGmmCount, blob.ColorSignature.data(), blob.ColorSignatureGmmCount,
			pChildHyp->AppearanceGmm.data(), pChildHyp->AppearanceGmm.size(), pChildHyp->AppearanceGmmCount);

		// calculate appearance score

		float appearDist = -1;
		float appearanceScore = 0;
		if (leafHyp->AppearanceGmmCount > 0 && pChildHyp->AppearanceGmmCount > 0)
		{
			appearanceScore = swimmerAppearanceModel_->appearanceScore(leafHyp->AppearanceGmm.data(), leafHyp->AppearanceGmmCount, blob.ColorSignature.data(), blob.ColorSignatureGmmCount);
		}

		childHyp.Score = leafHyp->Score + movementScore + appearanceScore;

		//
		childHyp.EstimatedPosWorld = estPos;
		childHyp.Age = leafHyp->Age + 1;

#if LOG_DEBUG_EX
		std::string obsStr = latestObservationStatus(*pChildHyp, 5, leafHyp);
		LOG4CXX_DEBUG(log_, "grow Corresp FamilyId=" << leafHyp->FamilyId << " LeafId=" << leafHyp->Id << " ChildId=" << pChildHyp->Id << " ObsInd=" << pChildHyp->ObservationInd << " " << pChildHyp->ObservationPos << " Score=" << pChildHyp->Score << " (dM=" << movementScore << " dA=" << appearanceScore << " dist2=" << appearDist << ") Age=" << pChildHyp->Age << " Obs=" << obsStr);
#endif
		//

		if (observationIndToHypNodes.find(blobInd) == end(observationIndToHypNodes))
		{
			observationIndToHypNodes.insert(make_pair(blobInd, std::vector<TrackHypothesisTreeNode*>()));
		}
		std::vector<TrackHypothesisTreeNode*>& hypsPerObs = observationIndToHypNodes[blobInd];
		hypsPerObs.push_back(pChildHyp.get());

		//

		leafHyp->addChildNode(std::move(pChildHyp));

		if (log_->isDebugEnabled())
			addedDueCorrespondence++;
	}
}

void MultiHypothesisBlobTracker::makeNoObservationHypothesis(int frameInd, TrackHypothesisTreeNode* leafHyp, float elapsedTimeMs, int& addedDueNoObservation, int noObsOrder, int blobsCount)
{
	auto id = nextTrackCandidateId_++;
	auto hypothesisReason = TrackHypothesisCreationReason::NoObservation;

	auto pChildHyp = make_unique<TrackHypothesisTreeNode>();

	int noObsBeyondBlobsId = -1 - noObsOrder;

	TrackHypothesisTreeNode& childHyp = *pChildHyp;
	childHyp.Id = id;
	childHyp.FamilyId = leafHyp->FamilyId;
	childHyp.ObservationInd = DetectionIndNoObservation;
	childHyp.ObservationOrNoObsId = noObsBeyondBlobsId;
	childHyp.FrameInd = frameInd;
	childHyp.ObservationPos = cv::Point2f(NullPosX, NullPosX);
	childHyp.ObservationPosWorld = cv::Point3f(NullPosX, NullPosX, NullPosX);
	childHyp.CreationReason = hypothesisReason;
	childHyp.KalmanFilterState = cv::Mat(4, 1, CV_32FC1);
	childHyp.KalmanFilterStateCovariance = cv::Mat(4, 4, CV_32FC1);

	//
	float movementScore;
	cv::Point3f estPos;
	movementPredictor_->estimateAndSave(*leafHyp, nullptr, estPos, movementScore, childHyp);
	childHyp.Score = leafHyp->Score + movementScore;
	childHyp.EstimatedPosWorld = estPos;

	// propogate color signature from parent track hypothesis node
	std::copy(std::begin(leafHyp->AppearanceGmm), std::begin(leafHyp->AppearanceGmm) + leafHyp->AppearanceGmmCount, std::begin(pChildHyp->AppearanceGmm));
	pChildHyp->AppearanceGmmCount = leafHyp->AppearanceGmmCount;

	childHyp.Age = leafHyp->Age;

#if LOG_DEBUG_EX
	std::string obsStr = latestObservationStatus(*pChildHyp, 5, leafHyp);
	LOG4CXX_DEBUG(log_, "grow NoObs FamilyId=" << leafHyp->FamilyId << " LeafId=" << leafHyp->Id << " ChildId=" << pChildHyp->Id << " Score=" << pChildHyp->Score << " Age=" << pChildHyp->Age << " Obs=" << obsStr);
#endif

	leafHyp->addChildNode(std::move(pChildHyp));

	if (log_->isDebugEnabled())
		addedDueNoObservation++;
}

void MultiHypothesisBlobTracker::makeNewTrackHypothesis(int frameInd, const std::vector<DetectedBlob>& blobs, int& addedNew, std::map<int, std::vector<TrackHypothesisTreeNode*>>& observationIndToHypNodes)
{
	// associate hypothesis node with each observation
	for (int blobInd = 0; blobInd < (int)blobs.size(); ++blobInd)
	{
		const auto& blob = blobs[blobInd];

		// NOTE: here we can't avoid allocating new track hypotheses for close blobs,
		// because there is no way to choose the right blob.
		// Multiple close track hypotheses should be handled in blob detector or by hypothesis tree pruning mechanism.

		cv::Point3f blobCentrWorld = blob.CentroidWorld;

		auto id = nextTrackCandidateId_++;
		auto hypothesisReason = TrackHypothesisCreationReason::New;

		auto pChildHyp = make_unique<TrackHypothesisTreeNode>();

		TrackHypothesisTreeNode& childHyp = *pChildHyp;
		childHyp.Id = id;
		childHyp.FamilyId = id; // new familyId=id of the initial hypothesis tree node
		childHyp.ObservationInd = blobInd;
		childHyp.ObservationOrNoObsId = blobInd;
		childHyp.FrameInd = frameInd;
		childHyp.ObservationPos = blob.Centroid;
		childHyp.ObservationPosWorld = blobCentrWorld;
		childHyp.CreationReason = hypothesisReason;
		childHyp.EstimatedPosWorld = blobCentrWorld;
		childHyp.KalmanFilterState = cv::Mat(4, 1, CV_32FC1);
		childHyp.KalmanFilterStateCovariance = cv::Mat(4, 4, CV_32FC1);
		float movementScore;
		movementPredictor_->initScoreAndState(frameInd, blobInd, blobCentrWorld, movementScore, childHyp);
		childHyp.Score = movementScore;

		// init color signature with one from the blob
		assert(blob.ColorSignatureGmmCount <= pChildHyp->AppearanceGmm.size());
		std::copy(std::begin(blob.ColorSignature), std::begin(blob.ColorSignature) + blob.ColorSignatureGmmCount, std::begin(pChildHyp->AppearanceGmm));
		pChildHyp->AppearanceGmmCount = blob.ColorSignatureGmmCount;

		childHyp.Age = 0;

#if LOG_DEBUG_EX
		LOG4CXX_DEBUG(log_, "grow New ChildId=" << pChildHyp->Id << " ObsInd=" << pChildHyp->ObservationInd << " " << pChildHyp->ObservationPos << " Score=" << pChildHyp->Score);
#endif

		//

		if (observationIndToHypNodes.find(blobInd) == end(observationIndToHypNodes))
		{
			observationIndToHypNodes.insert(make_pair(blobInd, std::vector<TrackHypothesisTreeNode*>()));
		}
		std::vector<TrackHypothesisTreeNode*>& hypsPerObs = observationIndToHypNodes[blobInd];
		hypsPerObs.push_back(pChildHyp.get());

		//

		trackHypothesisForestPseudoNode_.addChildNode(std::move(pChildHyp));

		if (log_->isDebugEnabled())
			addedNew++;
	}
}

void MultiHypothesisBlobTracker::setSwimmerMaxSpeed(float swimmerMaxSpeed)
{
	swimmerMaxSpeed_ = swimmerMaxSpeed;
}

void MultiHypothesisBlobTracker::growTrackHyposhesisTree(int frameInd, const std::vector<DetectedBlob>& blobs, float fps, float elapsedTimeMs)
{
	vector<TrackHypothesisTreeNode*> leafSet;
	getLeafSet(&trackHypothesisForestPseudoNode_, leafSet);

#if 1 // && log.isDebugEnabled()
	int addedDueNoObservation = 0;
	int addedDueCorrespondence = 0;
	int addedNew = 0;
#endif

	// contains the list of hypothesis nodes per given observation
	std::map<int, std::vector<TrackHypothesisTreeNode*>> observationIndToHypNodes;

	for (auto pLeaf : leafSet)
	{
		makeCorrespondenceHypothesis(frameInd, pLeaf, blobs, elapsedTimeMs, addedDueCorrespondence, observationIndToHypNodes);
	}

	int noObsOrder = 0;
	for (auto pLeaf : leafSet)
	{
		makeNoObservationHypothesis(frameInd, pLeaf, elapsedTimeMs, addedDueNoObservation, noObsOrder, blobs.size());
		noObsOrder++;
	}

	for (auto pLeaf : leafSet)
		fixHypNodeConsistency(pLeaf);

	// initiate new track sparingly(each N frames)

	if (frameInd % initNewTrackDelay_ == 0)
	{
		makeNewTrackHypothesis(frameInd, blobs, addedNew, observationIndToHypNodes);
		
		// fix root, as new hypothesis were added to it
		fixHypNodeConsistency(&trackHypothesisForestPseudoNode_);
	}

	LOG4CXX_DEBUG(log_, "addedNew=" << addedNew << " addedDueCorrespondence=" << addedDueCorrespondence << " addedDueNoObservation=" << addedDueNoObservation);

#if DO_CACHE_ICL
	updateIncompatibilityLists(leafSet, observationIndToHypNodes);
#endif
}

void MultiHypothesisBlobTracker::fixHypNodeConsistency(TrackHypothesisTreeNode* pNode)
{
	int count = (int)pNode->Children.size();
	pNode->ChildrenCount = count;
	pNode->ChildrenArray = count > 0 ? reinterpret_cast<TrackHypothesisTreeNode**>(&pNode->Children.front()) : nullptr;
}

void MultiHypothesisBlobTracker::checkHypNodesConsistency()
{
	std::vector<TrackHypothesisTreeNode*> allNodes;
	getSubtreeSet(&trackHypothesisForestPseudoNode_, allNodes, true);

	for (TrackHypothesisTreeNode* node : allNodes)
	{
		CV_Assert(node->ChildrenCount == (int)node->Children.size());
		if (node->Children.empty())
			CV_Assert(node->ChildrenArray == nullptr);
		else
			CV_Assert(node->ChildrenArray == (void*)&node->Children.front());
	}
}

#if DO_CACHE_ICL
void MultiHypothesisBlobTracker::updateIncompatibilityLists(const std::vector<TrackHypothesisTreeNode*>& oldLeafSet, std::map<int, std::vector<TrackHypothesisTreeNode*>>& observationIndToHypNodes)
{
	// add observation conflicts between all children of any oldLeaf hypothesis

	{
		std::vector<TrackHypothesisTreeNode*> nodeSet;

		for (TrackHypothesisTreeNode* pOldLeaf : oldLeafSet)
		{
			nodeSet.resize(pOldLeaf->Children.size());
			std::transform(begin(pOldLeaf->Children), end(pOldLeaf->Children), begin(nodeSet), [](const std::unique_ptr<TrackHypothesisTreeNode>& n)
			{
				return n.get();
			});

			setPairwiseObservationIndConflicts(nodeSet, pOldLeaf->FrameInd, pOldLeaf->ObservationInd, true, false);
		}
	}

	//

	std::map<int, TrackHypothesisTreeNode*> nodeIdToPtr;
	for (TrackHypothesisTreeNode* pOldLeaf : oldLeafSet)
	{
		nodeIdToPtr.insert(make_pair(pOldLeaf->Id, pOldLeaf));
	}

	// update incompatibility lists due to tree growth

	for (TrackHypothesisTreeNode* pOldHyp : oldLeafSet)
	{
		propagateIncompatibilityListsOnTreeGrowth(pOldHyp, oldLeafSet, nodeIdToPtr);
	}

	// add conflicts created due to collision of observations in current frame

	for (std::pair<const int, vector<TrackHypothesisTreeNode*>>& obsIdAndHyps : observationIndToHypNodes)
	{
		vector<TrackHypothesisTreeNode*>& hyps = obsIdAndHyps.second;
		if (hyps.size() <= 1)
			continue;

		int conflictObsInd = hyps[0]->ObservationInd;
		int conflictFrameInd = hyps[0]->FrameInd;
		
		CV_Assert(conflictObsInd >= 0 && "Hypothesis may share existent observation (obsInd>=0)");
		for (TrackHypothesisTreeNode* hyp : hyps)
		{
			CV_Assert(conflictFrameInd == hyp->FrameInd && conflictObsInd == hyp->ObservationInd && "Conflicting hypothesis must have the same observation");
		}

		// each hypothesis with shared observation conflicts with each other hypothesis
		// except conflicts between two children of the same parent
		setPairwiseObservationIndConflicts(hyps, conflictFrameInd, conflictObsInd, false, true);
	}

	//validateIncompatibilityLists();

	// purge memory for incompatibility liests of the old leaves

	for (TrackHypothesisTreeNode* pOldLeaf : oldLeafSet)
	{
		//pOldLeaf->IncompatibleNodes.shrink_to_fit();
		//pOldLeaf->IncompatibleNodes.clear();
		pOldLeaf->IncompatibleNodes.swap(vector<ObservationConflict>());
	}
}

void MultiHypothesisBlobTracker::setPairwiseObservationIndConflicts(std::vector<TrackHypothesisTreeNode*>& hyps, int conflictFrameInd, int conflictObsInd, bool allowConflictsForTheSameParent, bool forceUniqueCollisionsOnly)
{
	for (TrackHypothesisTreeNode* hyp : hyps)
	{
		hyp->IncompatibleNodes.reserve(hyps.size());
	}

	// generate conflicts of observations

	for (TrackHypothesisTreeNode* hyp : hyps)
	{
		int firstNodeId = hyp->Id;

		for (TrackHypothesisTreeNode* conflictHyp : hyps)
		{
			if (conflictHyp == hyp)
				continue;

			int conflictNodeId = conflictHyp->Id;

			// if conflicts from the same parent are prohibited
			// then take only conflicts from two children of different parents
			if (!allowConflictsForTheSameParent && hyp->Parent == conflictHyp->Parent)
				continue;

			if (false && forceUniqueCollisionsOnly)
			{
				auto it = std::find_if(begin(hyp->IncompatibleNodes), end(hyp->IncompatibleNodes), [=](ObservationConflict& n)
				{
					return n.OtherFamilyRootId == conflictNodeId;
				});
				if (it != end(hyp->IncompatibleNodes))
				{
					// collision between two hypothesis already exist
					continue;
				}
			}
#if PW_DEBUG
			// check incompatibility list has no node with such id
			auto it = std::find_if(begin(hyp->IncompatibleNodes), end(hyp->IncompatibleNodes), [=](ObservationConflict& n)
			{
				return n.OtherFamilyRootId == conflictNodeId;
			});
			if (it != end(hyp->IncompatibleNodes))
			{
				CV_Assert(false && "Two incompatible hypothesis must be unique");
			}
#endif
			ObservationConflict incomp(conflictFrameInd, conflictObsInd, firstNodeId, conflictNodeId);
			incomp.OtherFamilyRootId = conflictNodeId;
			hyp->IncompatibleNodes.push_back(incomp);
		}
	}
}

void MultiHypothesisBlobTracker::updateIncompatibilityListsOnHypothesisPruning(const std::vector<TrackHypothesisTreeNode*>& leavesAfterPruning, const std::set<int>& nodeIdSetAfterPruning)
{
	// TODO: how to handle multiple observations conflicts
	// assumes two hypothesis conflict on the single observation

	// we may update only leaves, because ancestors of each leaf do not participate in incompatibility graph generation

	for (TrackHypothesisTreeNode* leaf : leavesAfterPruning)
	{
		// remove all conflicting nodes, due to pruned hypothesis node

		auto it = begin(leaf->IncompatibleNodes);
		for (; it != end(leaf->IncompatibleNodes);)
		{
			int frameInd = it->FrameInd;
			int obsInd = it->ObservationInd;

			bool notFound1 = nodeIdSetAfterPruning.find(it->FirstNodeId) == end(nodeIdSetAfterPruning);
			bool notFound2 = nodeIdSetAfterPruning.find(it->OtherFamilyRootId) == end(nodeIdSetAfterPruning);

			if (notFound1 || notFound2)
			{
				it = leaf->IncompatibleNodes.erase(it);
				// it is not changed
			}
			else
			{
				++it;
			}
		}
	}
}

void MultiHypothesisBlobTracker::propagateIncompatibilityListsOnTreeGrowth(TrackHypothesisTreeNode* pOldHyp, const std::vector<TrackHypothesisTreeNode*>& oldLeafSet, const std::map<int, TrackHypothesisTreeNode*>& nodeIdToPtr)
{
	//for (std::unique_ptr<TrackHypothesisTreeNode>& pOldHypChild : pOldHyp->Children)
	//{
	//	propagateIncompatibilityListsOnTreeGrowthOne(*pOldHyp, *pOldHypChild, nodeIdToPtr);
	//}

	for (std::unique_ptr<TrackHypothesisTreeNode>& pOldHypChild : pOldHyp->Children)
	{
		// redirect all conflicts of the current hypothesis node to its children

		for (const ObservationConflict& incompNode : pOldHyp->IncompatibleNodes)
		{
			ObservationConflict newConflict = incompNode;

			// redirect conflict to the child
			newConflict.FirstNodeId = pOldHypChild->Id;
			pOldHypChild->IncompatibleNodes.push_back(newConflict);
		}
	}
}

void MultiHypothesisBlobTracker::propagateIncompatibilityListsOnTreeGrowthOne(const TrackHypothesisTreeNode& oldHyp, TrackHypothesisTreeNode& oldHypChild, const std::map<int, TrackHypothesisTreeNode*>& nodeIdToPtr)
{
//	int firstNodeId = oldHypChild.Id;
//
//	for (const ObservationConflict& incompNode : oldHyp.IncompatibleNodes)
//	{
//		int conflictObsInd = incompNode.ObservationInd; // the same conflicting observation
//		int conflictFrameInd = incompNode.FrameInd; // the same conflicting observation
//
//		//auto it = nodeIdToPtr.find(incompNode.OtherNodeId);
//		auto it = nodeIdToPtr.find(incompNode.OtherFamilyRootId);
//		CV_Assert(it != end(nodeIdToPtr) && "Other incompatible node from old leaf set must be found");
//
//		TrackHypothesisTreeNode* pOtherOldIncompNode = it->second;
//
//		//
//
//		// find all derived hypothesis from old incompatible nodes
//		for (const std::unique_ptr<TrackHypothesisTreeNode>& incompNodeChild : pOtherOldIncompNode->Children)
//		{
//			int conflictNodeId = incompNodeChild->Id;
//
//#if PW_DEBUG
//			// check incompatibility list has no node with such id
//			auto it = std::find_if(begin(oldHypChild.IncompatibleNodes), end(oldHypChild.IncompatibleNodes), [=](ObservationConflict& n)
//			{
//				//return n.OtherNodeId == conflictNodeId;
//				return n.OtherFamilyRootId == conflictNodeId;
//			});
//			if (it != end(oldHypChild.IncompatibleNodes))
//			{
//				CV_Assert(false && "Two incompatible hypothesis must be unique");
//			}
//#endif
//
//			//ObservationConflict updIncomp(conflictFrameInd, conflictObsInd, firstNodeId, conflictNodeId);
//			//oldHypChild.IncompatibleNodes.push_back(updIncomp);
//		}
//	}
}

void MultiHypothesisBlobTracker::validateIncompatibilityLists()
{
#if PW_DEBUG
	vector<TrackHypothesisTreeNode*> newLeafSet;
	getLeafSet(&trackHypothesisForestPseudoNode_, newLeafSet);
	std::map<int, TrackHypothesisTreeNode*> leavesNodeIdToPtr;
	for (TrackHypothesisTreeNode* pLeaf : newLeafSet)
	{
		leavesNodeIdToPtr.insert(make_pair(pLeaf->Id, pLeaf));
	}

	for (TrackHypothesisTreeNode* pLeaf : newLeafSet)
	{
		// all incompatible nodes are unique

		std::vector<int> incompNodeIds(pLeaf->IncompatibleNodes.size());
		std::transform(begin(pLeaf->IncompatibleNodes), end(pLeaf->IncompatibleNodes), begin(incompNodeIds), [](ObservationConflict& n)
		{
			return n.OtherFamilyRootId;
		});
		std::sort(begin(incompNodeIds), end(incompNodeIds));
		auto uniqueIt = std::unique(begin(incompNodeIds), end(incompNodeIds));
		int uniqueCount = uniqueIt - begin(incompNodeIds);
		CV_Assert(uniqueCount == pLeaf->IncompatibleNodes.size() && "All ids of incompatible nodes must be unique");
	}

#if !DO_CACHE_ICL
	validateConformanceDLangImpl();
#endif
#endif
}

#if !DO_CACHE_ICL
void MultiHypothesisBlobTracker::validateConformanceDLangImpl()
{
	vector<int32_t> encodedTreeString;
	hypothesisTreeToTreeStringRec(trackHypothesisForestPseudoNode_, encodedTreeString);

	// incompatibility graph in the form of list of edges, each edge is a pair of vertices
	vector<int32_t> incompatibTrackEdges;
	createTrackIncopatibilityGraphDLang(encodedTreeString, incompatibTrackEdges); // [1x(2*edgesCount)]

	int edgesCount = incompatibTrackEdges.size() / 2;

	std::map<int32_t, std::vector<int32_t>> nodeIdToIncompNodes;
	for (int i = 0; i < edgesCount; ++i)
	{
		int i1 = i * 2 + 0;
		int i2 = i * 2 + 1;
		int node1Id = incompatibTrackEdges[i1];
		int node2Id = incompatibTrackEdges[i2];
		if (nodeIdToIncompNodes.find(node1Id) == end(nodeIdToIncompNodes))
		{
			nodeIdToIncompNodes.insert(make_pair(node1Id, std::vector<int32_t>()));
		}
		if (nodeIdToIncompNodes.find(node2Id) == end(nodeIdToIncompNodes))
		{
			nodeIdToIncompNodes.insert(make_pair(node2Id, std::vector<int32_t>()));
		}
		nodeIdToIncompNodes[node1Id].push_back(node2Id);
		nodeIdToIncompNodes[node2Id].push_back(node1Id);
	}
	// sort each list of neightbourly nodes
	for (auto& pair : nodeIdToIncompNodes)
		std::sort(begin(pair.second), end(pair.second));

	vector<TrackHypothesisTreeNode*> newLeafSet;
	getLeafSet(&trackHypothesisForestPseudoNode_, newLeafSet);

	vector<TrackHypothesisTreeNode*> allNodesSet;
	getSubtreeSet(&trackHypothesisForestPseudoNode_, allNodesSet, false);
	std::map<int, TrackHypothesisTreeNode*> nodeIdToNode;
	for (TrackHypothesisTreeNode* pLeaf : allNodesSet)
		nodeIdToNode[pLeaf->Id] = pLeaf;
		
	// compare ICL for each leaf
	vector<TrackHypothesisTreeNode*> conflictNodes;
	for (TrackHypothesisTreeNode* pLeaf : newLeafSet)
	{
		// actual ICL list
		std::vector<int> incompNodeIds;
		//std::vector<int> incompNodeIds(pLeaf->IncompatibleNodes.size());
		//std::transform(begin(pLeaf->IncompatibleNodes), end(pLeaf->IncompatibleNodes), begin(incompNodeIds), [](ObservationConflict& n)
		//{
		//	return n.OtherNodeId;
		//});
		
		for (ObservationConflict& oc : pLeaf->IncompatibleNodes)
		{
			TrackHypothesisTreeNode* pNode = nodeIdToNode[oc.OtherFamilyRootId];
			CV_Assert(pNode != nullptr);

			conflictNodes.clear();
			getLeafSet(pNode, conflictNodes);

			std::transform(begin(conflictNodes), end(conflictNodes), back_inserter(incompNodeIds), [](TrackHypothesisTreeNode* n)
			{
				return n->Id;
			});
		}
		std::sort(begin(incompNodeIds), end(incompNodeIds));
		auto uniqueEnd = std::unique(begin(incompNodeIds), end(incompNodeIds));
		incompNodeIds.erase(uniqueEnd, end(incompNodeIds));

		// expected ICL list
		std::vector<int> expectedIcl = nodeIdToIncompNodes[pLeaf->Id];
		CV_Assert(incompNodeIds.size() == expectedIcl.size());
		bool eachEq = std::equal(begin(incompNodeIds), end(incompNodeIds), begin(expectedIcl));
		CV_Assert(eachEq);
	}
}
#endif // DO_CACHE_ICL
#endif


//int MultiHypothesisBlobTracker::compoundObservationId(const TrackHypothesisTreeNode& node)
//{
//	//if (node.ObservationInd == DetectionIndNoObservation)
//	//	return DetectionIndNoObservation;
//	//return node.FrameInd*MaxObservationsCountPerFrame + node.ObservationInd;
//	return node.FrameInd*MaxObservationsCountPerFrame + node.ObservationOrNoObsId;
//}

void MultiHypothesisBlobTracker::hypothesisTreeToTreeStringRec(const TrackHypothesisTreeNode& startFrom, vector<int32_t>& encodedTreeString)
{
	//int compoundId = compoundObservationId(startFrom);
	
	encodedTreeString.push_back(startFrom.Id);
	encodedTreeString.push_back(startFrom.FrameInd);
	encodedTreeString.push_back(startFrom.ObservationOrNoObsId);

	if (!startFrom.Children.empty())
	{
		encodedTreeString.push_back(OpenBracket);

		for (const auto& pChild : startFrom.Children)
			hypothesisTreeToTreeStringRec(*pChild, encodedTreeString);

		encodedTreeString.push_back(CloseBracket);
	}
}

#if DO_CACHE_ICL
void MultiHypothesisBlobTracker::createTrackIncompatibilityGraphUsingPerNodeICL(const std::vector<TrackHypothesisTreeNode*>& leafSet, const std::map<int, TrackHypothesisTreeNode*>& nodeIdToNode, vector<int32_t>& incompatibTrackEdges)
{
#if PW_DEBUG
	// check all incompatible references are live

	for (TrackHypothesisTreeNode* leaf : leafSet)
	{
		TrackHypothesisTreeNode* pLeaf = nodeIdToNode[leaf->Id];
		CV_Assert(pLeaf != nullptr);

		for (const auto& conflict : pLeaf->IncompatibleNodes)
		{
			TrackHypothesisTreeNode* pConflictNode = nodeIdToNode[conflict.OtherFamilyRootId];
			CV_Assert(pConflictNode != nullptr);
		}
	}
#endif
	vector<TrackHypothesisTreeNode*> contextNodeLeaves;
	for (TrackHypothesisTreeNode* leaf : leafSet)
	{
		int firstId = leaf->Id;
		for (ObservationConflict& obsConflict : leaf->IncompatibleNodes)
		{
			CV_Assert(firstId == obsConflict.FirstNodeId);
			int secondId = obsConflict.OtherFamilyRootId;

			TrackHypothesisTreeNode* secondNode = nodeIdToNode[secondId];

			// find all leaves of the conflicting node
			contextNodeLeaves.clear();
			getLeafSet(secondNode, contextNodeLeaves);

			for (const TrackHypothesisTreeNode* conflictNode : contextNodeLeaves)
			{
				int secondId = conflictNode->Id;

				// edge x-y and y-x will be visited twice; we will take only one (when x<y)
				if (firstId < secondId)
				{
					incompatibTrackEdges.push_back(firstId);
					incompatibTrackEdges.push_back(secondId);
				}
			}
		}
	}
}
#else

void MultiHypothesisBlobTracker::createTrackIncopatibilityGraphDLang(const vector<int32_t>& encodedTreeString, vector<int32_t>& incompGraphEdgePairs) const
{
	Int32Allocator int32Alloc;
	int32Alloc.pUserData = &incompGraphEdgePairs;
	int32Alloc.CreateArrayInt32 = [](size_t celem, void* pUserData) -> int32_t*
	{
		CV_Assert(celem > 0);
		std::vector<int32_t>* pVec = reinterpret_cast<std::vector<int32_t>*>(pUserData);
		pVec->resize(celem);

		return &(*pVec)[0];
	};
	int32Alloc.DestroyArrayInt32 = [](int32_t* pInt32, void* pUserData) -> void
	{
		// vector manages its memory
	};

	// Note: we ignore incompNodesRange because the result is populated in incompGraphEdgePairs
	Int32PtrPair incompNodesRange = computeTrackIncopatibilityGraph(&encodedTreeString[0], (int)encodedTreeString.size(), trackHypothesisForestPseudoNode_.Id, OpenBracket, CloseBracket, DetectionIndNoObservation, int32Alloc);

#if PW_DEBUG
	if (incompNodesRange.pFirst == nullptr)
	{
		CV_Assert(incompNodesRange.pLast == nullptr);
		CV_Assert(incompGraphEdgePairs.empty());
	}
	else
	{
		size_t sz = incompNodesRange.pLast - incompNodesRange.pFirst;
		CV_Assert(incompGraphEdgePairs.size() == sz);
		CV_Assert(incompGraphEdgePairs.size() % 2 == 0 && "Vertices list contains pair (from,to) of vertices for each edge");
	}
#endif
}

void MultiHypothesisBlobTracker::createTrackIncopatibilityGraphDLangDirectAccess(vector<int32_t>& incompGraphEdgePairs) const
{
	Int32Allocator int32Alloc;
	int32Alloc.pUserData = &incompGraphEdgePairs;
	int32Alloc.CreateArrayInt32 = [](size_t celem, void* pUserData) -> int32_t*
	{
		CV_Assert(celem > 0);
		std::vector<int32_t>* pVec = reinterpret_cast<std::vector<int32_t>*>(pUserData);
		pVec->resize(celem);

		return &(*pVec)[0];
	};
	int32Alloc.DestroyArrayInt32 = [](int32_t* pInt32, void* pUserData) -> void
	{
		// vector manages its memory
	};

	// Note: we ignore incompNodesRange because the result is populated in incompGraphEdgePairs
	Int32PtrPair incompNodesRange = computeTrackIncopatibilityGraphDirectAccess(&trackHypothesisForestPseudoNode_, trackHypothesisForestPseudoNode_.Id, int32Alloc);

#if PW_DEBUG
	if (incompNodesRange.pFirst == nullptr)
	{
		CV_Assert(incompNodesRange.pLast == nullptr);
		CV_Assert(incompGraphEdgePairs.empty());
	}
	else
	{
		size_t sz = incompNodesRange.pLast - incompNodesRange.pFirst;
		CV_Assert(incompGraphEdgePairs.size() == sz);
		CV_Assert(incompGraphEdgePairs.size() % 2 == 0 && "Vertices list contains pair (from,to) of vertices for each edge");
	}
#endif
}
#endif


void MultiHypothesisBlobTracker::findBestTracks(const std::vector<TrackHypothesisTreeNode*>& leafSet, std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs)
{
	findBestTracksDLang(bestTrackLeafs);

#if PW_DEBUG
	std::vector<TrackHypothesisTreeNode*> hyps;
	findBestTracksCpp(leafSet, hyps);

	//

	std::sort(std::begin(bestTrackLeafs), std::end(bestTrackLeafs));
	std::sort(std::begin(hyps), std::end(hyps));

	bool eqSize = hyps.size() == bestTrackLeafs.size();
	CV_Assert(eqSize);
	for (int i = 0; i < (int)hyps.size(); ++i)
		CV_Assert(bestTrackLeafs[i] == hyps[i]);
#endif
}

void MultiHypothesisBlobTracker::findBestTracksCpp(std::vector<TrackHypothesisTreeNode*> const& leafSet, std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs)
{
	// incompatibility graph in the form of list of edges, each edge is a pair of vertices
	vector<int32_t> incompatibTrackEdges;

	// construct NodeId -> Node map
	std::map<int, TrackHypothesisTreeNode*> nodeIdToNode;
	std::vector<TrackHypothesisTreeNode*> fullSubtree;
	getSubtreeSet(&trackHypothesisForestPseudoNode_, fullSubtree, false);
	for (auto pNode : fullSubtree)
		nodeIdToNode[pNode->Id] = pNode;

	// construct collision edges
	// we do it using cached incompatibility lists if DO_CACHE_ICL is set, or construct ICL dynamically otherwise
#if DO_CACHE_ICL
	createTrackIncompatibilityGraphUsingPerNodeICL(leafSet, nodeIdToNode, incompatibTrackEdges);
#else
	{
		//vector<int32_t> encodedTreeString;
		//hypothesisTreeToTreeStringRec(trackHypothesisForestPseudoNode_, encodedTreeString);
		//createTrackIncopatibilityGraphDLang(encodedTreeString, incompatibTrackEdges); // [1x(2*edgesCount)]

		//vector<int32_t> incompatibTrackEdgesNew;
		createTrackIncopatibilityGraphDLangDirectAccess(incompatibTrackEdges); // [1x(2*edgesCount)]

		//
		//CV_Assert(incompatibTrackEdges.size() == incompatibTrackEdgesNew.size());
		//auto toPairsFun = [](const std::vector<int>& edgesList)->std::vector < std::pair<int, int> >
		//{
		//	std::vector<std::pair<int, int>> result;
		//	for (int i = 0; i < (int)edgesList.size() / 2; ++i)
		//	{
		//		int i1 = edgesList[i * 2];
		//		int i2 = edgesList[i * 2 + 1];
		//		int left = std::min(i1, i2);
		//		int right = std::max(i1, i2);
		//		result.push_back(std::make_pair(left, right));
		//	}
		//	std::sort(std::begin(result), std::end(result), [](const std::pair<int, int>& x, const std::pair<int, int>& y)
		//	{
		//		if (x.first != y.first)
		//			return x.first < y.first;
		//		return x.second < y.second;
		//	});
		//	return result;
		//};

		//std::vector<std::pair<int, int>> ps1 = toPairsFun(incompatibTrackEdges);
		//std::vector<std::pair<int, int>> ps2 = toPairsFun(incompatibTrackEdgesNew);
		//for (int i = 0; i < ps1.size(); ++i)
		//{
		//	CV_Assert(ps1[i] == ps2[i]);
		//}
	}
#endif

	int edgesCount = incompatibTrackEdges.size() / 2;
	//!LOG4CXX_INFO(log_, "edgesCount=" << edgesCount);

	// find vertices array

	vector<int32_t> connectedVertices = incompatibTrackEdges;
	std::sort(begin(connectedVertices), end(connectedVertices));
	auto it = std::unique(begin(connectedVertices), end(connectedVertices));
	auto newSize = std::distance(begin(connectedVertices), it);
	connectedVertices.resize(newSize);

	// find entire hypothesis tree vertex ids

	vector<int32_t> allTrackIds;
	allTrackIds.reserve(leafSet.size());
	std::transform(begin(leafSet), end(leafSet), back_inserter(allTrackIds), [](TrackHypothesisTreeNode* pNode) { return (int32_t)pNode->Id; });
	std::sort(begin(allTrackIds), end(allTrackIds));

	// isolated vertices are always in the set of best hypothesis

	vector<int32_t> isolatedVertices; // 
	std::set_difference(begin(allTrackIds), end(allTrackIds), begin(connectedVertices), end(connectedVertices), back_inserter(isolatedVertices));
	std::transform(begin(isolatedVertices), end(isolatedVertices), back_inserter(bestTrackLeafs), [&](int32_t id) { return nodeIdToNode[id]; });

	//

	vector<uchar> isInIndepVertexSet;
#if DO_CACHE_ICL
	maximumWeightIndependentSetNaiveMaxFirstCpp(connectedVertices, nodeIdToNode, indepVertexSet);
#else
	// find weights

	vector<double> vertexWeights(connectedVertices.size());
	for (size_t i = 0; i < connectedVertices.size(); ++i)
	{
		auto vertexId = connectedVertices[i];
		auto pNode = nodeIdToNode[vertexId];
		vertexWeights[i] = pNode->Score;
	}

	auto gMap = createFromEdgeList(connectedVertices, incompatibTrackEdges);
	auto g = get<0>(gMap);

	for (int i = 0; i < vertexWeights.size(); ++i)
		g.setVertexPayload(i, vertexWeights[i]);

	//maximumWeightIndependentSetNaiveMaxFirst(g, indepVertexSet);
	maximumWeightIndependentSetNaiveMaxFirstMultipleSeeds(g, isInIndepVertexSet);
	assert(isInIndepVertexSet.size() == connectedVertices.size());
#endif

	for (size_t i = 0; i < isInIndepVertexSet.size(); ++i)
		if (isInIndepVertexSet[i])
		{
		int32_t vertexId = connectedVertices[i];
		auto pNode = nodeIdToNode[vertexId];
		bestTrackLeafs.push_back(pNode);
		}
}

void MultiHypothesisBlobTracker::findBestTracksDLang(std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs)
{
#if PW_DEBUG
	// validate the hypothesis tree after changes (after growth and pruning)
	checkHypNodesConsistency();
#endif

	CppVectorPtrWrapper trackHypsWrapper;
	PoolWatch::bindVectorWrapper(trackHypsWrapper, bestTrackLeafs);

	// attempt count for maximum weight independent set algorithm
	// actually, such number of naive implemenation of MWISP algorithm are run, each using slightly perturbed order of collision graph nodes
	const int attemptCount = 5;
	pwFindBestTracks(&trackHypothesisForestPseudoNode_, trackHypothesisForestPseudoNode_.Id, attemptCount, &trackHypsWrapper);
}

#if DO_CACHE_ICL
void MultiHypothesisBlobTracker::maximumWeightIndependentSetNaiveMaxFirstCpp(const std::vector<int32_t>& connectedVertices, const std::map<int32_t, TrackHypothesisTreeNode*>& treeNodeIdToNode, std::vector<bool>& indepVertexSet)
{
	struct NodeInfo
	{
		const TrackHypothesisTreeNode* pNode = nullptr;
		bool visited = false;
		bool inMaxWeightSet = false;
		std::vector<NodeInfo*> neighbours;
	};

	vector<NodeInfo> nodeInfos(connectedVertices.size());
	map<int, NodeInfo*> nodeIdToNodeInfo;

	{
		for (size_t i = 0; i < connectedVertices.size(); ++i)
		{
			int32_t vertexId = connectedVertices[i];
			// TODO: why the line below doesn't compile?
			// const TrackHypothesisTreeNode* pNode = treeNodeIdToNode[vertexId];
			TrackHypothesisTreeNode* pNode = nullptr;
			auto noteId = treeNodeIdToNode.find(vertexId);
			if (noteId != end(treeNodeIdToNode))
				pNode = noteId->second;
			CV_Assert(pNode != nullptr);

			NodeInfo info;
			info.pNode = pNode;
			nodeInfos[i] = info;

			nodeIdToNodeInfo[vertexId] = &nodeInfos[i];
		}
	}
	
	// init edges

	vector<TrackHypothesisTreeNode*> leavesTmp;
	for (NodeInfo& leafInfo : nodeInfos)
	{
		for (const ObservationConflict& oc : leafInfo.pNode->IncompatibleNodes)
		{
			auto it = treeNodeIdToNode.find(oc.OtherFamilyRootId);
			CV_Assert(it != end(treeNodeIdToNode));

			TrackHypothesisTreeNode* otherNode = it->second;

			leavesTmp.clear();
			this->getLeafSet(otherNode, leavesTmp);
			for (TrackHypothesisTreeNode* pLeaf : leavesTmp)
			{
				NodeInfo* otherInfo = nodeIdToNodeInfo[pLeaf->Id];
				CV_Assert(otherInfo != nullptr);
				leafInfo.neighbours.push_back(otherInfo);

				// TODO: should work without it
				otherInfo->neighbours.push_back(&leafInfo);
			}
		}
	}

	vector<int> indicesByScoreDesc(nodeInfos.size());
	for (size_t i = 0; i < nodeInfos.size(); ++i)
		indicesByScoreDesc[i] = i;

	std::sort(begin(indicesByScoreDesc), end(indicesByScoreDesc), [&nodeInfos](int ind1, int ind2)
	{
		const auto& info1 = nodeInfos[ind1];
		const auto& info2 = nodeInfos[ind2];
		return std::greater<float>()(info1.pNode->Score, info2.pNode->Score);
	});

	indepVertexSet.resize(connectedVertices.size());

	for (int highScoreInd : indicesByScoreDesc)
	{
		NodeInfo& highScoreInfo = nodeInfos[highScoreInd];
		if (highScoreInfo.visited)
			continue;
		highScoreInfo.visited = true;
		highScoreInfo.inMaxWeightSet = true;

		indepVertexSet[highScoreInd] = true;

		for (NodeInfo* pNeigh : highScoreInfo.neighbours)
		{
			if (pNeigh->visited)
			{
				CV_Assert(!pNeigh->inMaxWeightSet);
			}
			else
			{
				pNeigh->visited = true;
				pNeigh->inMaxWeightSet = false;
			}
		}
	}
}
#endif

TrackHypothesisTreeNode* MultiHypothesisBlobTracker::findNewFamilyRoot(TrackHypothesisTreeNode* leaf, int pruneWindow)
{
	CV_Assert(pruneWindow >= 0);
	if (pruneWindow == 0)
	{
		CV_Assert(isPseudoRoot(*leaf->Parent));
	}
	assert(leaf != nullptr);
	assert(!isPseudoRoot(*leaf) && "Assume starting from terminal, not pseudo node");

	if (pruneWindow == 0)
		return leaf->Parent;

	// find new root
	auto current = leaf;
	int stepBack = 1;
	while (true)
	{
		if (stepBack == pruneWindow)
			return current;

		assert(current->Parent != nullptr && "Current node always have the parent node or pseudo root");

		// stop if parent is the pseudo root
		if (isPseudoRoot(*current->Parent))
			return current;

		current = current->Parent;
		stepBack++;
	}
}

std::string MultiHypothesisBlobTracker::latestObservationStatus(TrackHypothesisTreeNode const& leafNode, int lastFramesCount, TrackHypothesisTreeNode* leafParentOrNull)
{
	std::vector<TrackHypothesisTreeNode*> nodes;
	enumerateBranchNodesReversed(const_cast<TrackHypothesisTreeNode*>(&leafNode), lastFramesCount, nodes, leafParentOrNull);

	std::reverse(std::begin(nodes), std::end(nodes));

	std::stringstream buf;
	TrackHypothesisTreeNode* prevNode = nullptr;
	for (TrackHypothesisTreeNode* pNode : nodes)
	{
		if (isPseudoRoot(*pNode))
			continue; // root is the first node in the sequence

		// insert comma if no one of two consequtive nodes is noObs
		if (prevNode != nullptr && 
			prevNode->ObservationInd != TrackHypothesisTreeNode::DetectionIndNoObservation &&
			pNode   ->ObservationInd != TrackHypothesisTreeNode::DetectionIndNoObservation)
			buf << ",";

		if (pNode->ObservationInd == TrackHypothesisTreeNode::DetectionIndNoObservation)
			buf << "-"; // noObs
		else
			buf << pNode->ObservationInd; // gotObs

		prevNode = pNode;
	}
	return buf.str();
}

//void MultiHypothesisBlobTracker::enumerateBranchNodesReversed(TrackHypothesisTreeNode* leaf, int pruneWindow, std::vector<TrackHypothesisTreeNode*>& result) const
//{
//	assert(leaf != nullptr);
//	assert(!isPseudoRoot(*leaf) && "Assume starting from terminal, not pseudo node");
//
//	// find new root
//	auto current = leaf;
//	int stepBack = 1;
//	while (true)
//	{
//		result.push_back(current);
//
//		if (stepBack == pruneWindow)
//			return;
//
//		assert(current->Parent != nullptr && "Current node always have the parent node or pseudo root");
//
//		// stop if parent is the pseudo root
//		if (isPseudoRoot(*current->Parent))
//			return;
//
//		current = current->Parent;
//		stepBack++;
//	}
//}

bool MultiHypothesisBlobTracker::isPseudoRoot(const TrackHypothesisTreeNode& node) const
{
	return node.Id == trackHypothesisForestPseudoNode_.Id;	
}

void MultiHypothesisBlobTracker::getLeafSet(TrackHypothesisTreeNode* startNode, std::vector<TrackHypothesisTreeNode*>& leafSet)
{
	assert(startNode != nullptr);

	if (startNode->Children.empty())
	{
		if (isPseudoRoot(*startNode))
			return;
		leafSet.push_back(startNode);
	}
	else
	{
		for (auto& child : startNode->Children)
		{
			getLeafSet(&*child, leafSet);
		}
	}
}

void MultiHypothesisBlobTracker::getSubtreeSet(TrackHypothesisTreeNode* startNode, std::vector<TrackHypothesisTreeNode*>& subtreeSet, bool includePseudoRoot)
{
	assert(startNode != nullptr);

	bool isRoot = isPseudoRoot(*startNode);
	if (includePseudoRoot || !isRoot)
	{
		subtreeSet.push_back(startNode);
	}

	for (auto& child : startNode->Children)
	{
		getSubtreeSet(&*child, subtreeSet, includePseudoRoot);
	}
}

TrackChangePerFrame MultiHypothesisBlobTracker::createTrackChange(TrackHypothesisTreeNode* pNode)
{
	TrackChangePerFrame result;
	populateTrackChange(pNode, result);
	return result;
}

void MultiHypothesisBlobTracker::populateTrackChange(TrackHypothesisTreeNode* pNode, TrackChangePerFrame& result)
{
	result.FamilyId = pNode->FamilyId;

	if (pNode->CreationReason == TrackHypothesisCreationReason::New)
		result.UpdateType = TrackChangeUpdateType::New;
	else if (pNode->CreationReason == TrackHypothesisCreationReason::SequantialCorrespondence)
		result.UpdateType = TrackChangeUpdateType::ObservationUpdate;
	else if (pNode->CreationReason == TrackHypothesisCreationReason::NoObservation)
		result.UpdateType = TrackChangeUpdateType::NoObservation;

	auto estimatedPos = pNode->EstimatedPosWorld;

	result.EstimatedPosWorld = estimatedPos;
	result.ObservationInd = pNode->ObservationInd;

	auto obsPos = pNode->ObservationPos;
	if (obsPos.x == NullPosX)
		obsPos = cameraProjector_->worldToCamera(estimatedPos);
	result.ObservationPosPixExactOrApprox = obsPos;

	result.FrameInd = pNode->FrameInd;
	result.Score = pNode->Score;
}

int MultiHypothesisBlobTracker::getReadyFrameInd(int frameInd, int pruneWindow)
{
	int readyFrameInd = frameInd - pruneWindow;
	if (readyFrameInd < 0)
	{
		readyFrameInd = -1;
	}

	return readyFrameInd;
}

void MultiHypothesisBlobTracker::pruneHypothesisTreeAndGetTrackChanges(int frameInd, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs, int& readyFrameInd, std::vector<TrackChangePerFrame>& trackChanges, int pruneWindow)
{
	// find frame with ready track info

	readyFrameInd = getReadyFrameInd(frameInd, pruneWindow);
	if (readyFrameInd < 0)
	{
		// no ready track info yet
		return;
	}

	// store the set of leaves before pruning
	std::vector<TrackHypothesisTreeNode*> leavesBeforePruning;
	getLeafSet(&trackHypothesisForestPseudoNode_, leavesBeforePruning);


	// gather new family roots
	// family roots, not gathered in this process, treated as pruned

	vector<unique_ptr<TrackHypothesisTreeNode>> newFamilyRoots;
	for (const auto pLeaf : bestTrackLeafs)
	{
		TrackHypothesisTreeNode* newRoot = findNewFamilyRoot(pLeaf, pruneWindow);
		
		if (isPseudoRoot(*newRoot))
		{
			// leaf is a child of pseudo root
			CV_Assert(pruneWindow == 0);
			CV_Assert(isPseudoRoot(*pLeaf->Parent));

			// do not store any newRoot
			// further the leaf will be treated as pruned node
			continue;
		}

		TrackHypothesisTreeNode* newRootParent = newRoot->Parent;
		if (isPseudoRoot(*newRootParent))
		{
			// track change is not ready yet so old and new root will be the same node
		}
		else
		{
			TrackChangePerFrame change = createTrackChange(newRootParent);
			trackChanges.push_back(change);
		}

		// remember new root

		auto newRootUnique = std::move(newRootParent->pullChild(newRoot));
		newFamilyRoots.push_back(std::move(newRootUnique));
	}

	// save family ids before pruning

	std::set<int> familyIdSetAfterPrune;
	for (const std::unique_ptr<TrackHypothesisTreeNode>& pRoot : newFamilyRoots)
	{
		familyIdSetAfterPrune.insert(pRoot->FamilyId);
	}

	// find pruned families (the set of tracks, initiated from the same observation)
	// = oldFamilyRoots - newFamilyRoots

	const auto& oldFamilyRoots = trackHypothesisForestPseudoNode_.Children;
	for (const std::unique_ptr<TrackHypothesisTreeNode>& pRoot : oldFamilyRoots)
	{
		// gathered family roots are removed from pseudo root children and nulls are left
		if (pRoot == nullptr)
			continue;

		int familyId = pRoot->FamilyId;

		bool found = familyIdSetAfterPrune.find(familyId) != familyIdSetAfterPrune.end();
		if (!found)
		{
			// notify a client about pruning only if the track with this familyId has been previously published as 'New'

			bool wasPublishedAsNew = pRoot->CreationReason != TrackHypothesisCreationReason::New;
			if (wasPublishedAsNew)
			{
				// family was pruned
				int trackId = pRoot->FamilyId;
				TrackChangePerFrame change = createTrackChange(pRoot.get());
				change.UpdateType = TrackChangeUpdateType::Pruned;
				trackChanges.push_back(change);
			}
		}
	}

	//

#if DO_CACHE_ICL
	std::vector<TrackHypothesisTreeNode*> leavesAfterPruning;
	std::vector<TrackHypothesisTreeNode*> nodesSubtreeTmp;
	std::set<int> nodeIdSetAfterPruning;

	for (const std::unique_ptr<TrackHypothesisTreeNode>& newRoot : newFamilyRoots)
	{
		getLeafSet(newRoot.get(), leavesAfterPruning);

		nodesSubtreeTmp.clear();
		getSubtreeSet(newRoot.get(), nodesSubtreeTmp, false);
		for (TrackHypothesisTreeNode* node : nodesSubtreeTmp)
		{
			nodeIdSetAfterPruning.insert(node->Id);
		}
	}

	updateIncompatibilityListsOnHypothesisPruning(leavesAfterPruning, nodeIdSetAfterPruning);
#endif

	// do pruning
	// assume the set of newFamilyRoots are unique, this is because bestTrackLeafs do not collide on any observation

	trackHypothesisForestPseudoNode_.Children.clear();
	for (auto& familtyRoot : newFamilyRoots)
	{
		trackHypothesisForestPseudoNode_.addChildNode(std::move(familtyRoot));
	}
	fixHypNodeConsistency(&trackHypothesisForestPseudoNode_);
}

void MultiHypothesisBlobTracker::pruneLowScoreTracks(int frameInd, std::vector<TrackChangePerFrame>& trackChanges)
{
	vector<TrackHypothesisTreeNode*> leafSet;
	getLeafSet(&trackHypothesisForestPseudoNode_, leafSet);

	for (TrackHypothesisTreeNode* pNode : leafSet)
	{
		if (pNode->Score < trackMinScore_)
		{
			// find the root of the subtree to remove
			TrackHypothesisTreeNode* pNodeToRemove = pNode;
			while(true)
			{
				auto parent = pNodeToRemove->Parent;
				if (parent->Children.size() > 1)
				{
					// parent remains in the tree
					// current node should be pruned
					break;
				}

				if (isPseudoRoot(*parent))
					break;
				
				pNodeToRemove = parent;
			}

			// remove child
			auto parent = pNodeToRemove->Parent;
			std::unique_ptr<TrackHypothesisTreeNode> pChild = std::move(parent->pullChild(pNodeToRemove, true));
			fixHypNodeConsistency(parent);

			// family root was pruned?
			if (isPseudoRoot(*parent))
			{
				bool wasPublishedAsNew = pChild->CreationReason != TrackHypothesisCreationReason::New;
				if (wasPublishedAsNew)
				{
					int trackId = pChild->FamilyId;
					TrackChangePerFrame change = createTrackChange(pChild.get());
					change.UpdateType = TrackChangeUpdateType::Pruned;
					trackChanges.push_back(change);
				}
			}
		}
	}
}

bool MultiHypothesisBlobTracker::flushTrackHypothesis(int frameInd, int& readyFrameInd, std::vector<TrackChangePerFrame>& trackChanges,
	vector<TrackHypothesisTreeNode*>& bestTrackLeafsCache, bool isBestTrackLeafsInitied, int& pruneWindow)
{
	if (!isBestTrackLeafsInitied)
	{
		vector<TrackHypothesisTreeNode*> leafSet;
		getLeafSet(&trackHypothesisForestPseudoNode_, leafSet);

		findBestTracks(leafSet, bestTrackLeafsCache);
	}

	if (pruneWindow == -1)
	{
		// init pruneWindow on first call
		pruneWindow = pruneWindow_ - 1;
		CV_Assert(pruneWindow >= 0);
	}

	// collect oldest ready hypothesis nodes
	pruneHypothesisTreeAndGetTrackChanges(frameInd, bestTrackLeafsCache, readyFrameInd, trackChanges, pruneWindow);
	
	CV_Assert(readyFrameInd != -1 && "Can't collect track changes from hypothesis tree");

	bool continue1 = pruneWindow > 0;
	return continue1;
}

void MultiHypothesisBlobTracker::getMostPossibleHypothesis(int frameInd, std::vector<TrackHypothesisTreeNode*>& hypList)
{
	vector<TrackHypothesisTreeNode*> leafSet;
	getLeafSet(&trackHypothesisForestPseudoNode_, leafSet);

	vector<TrackHypothesisTreeNode*> bestTrackLeafs; // 
	findBestTracks(leafSet, bestTrackLeafs);

	std::vector<TrackHypothesisTreeNode*> pathNodes;
	for (TrackHypothesisTreeNode* pLeaf: bestTrackLeafs)
	{
		pathNodes.clear();
		enumerateBranchNodesReversed(pLeaf, pruneWindow_, pathNodes);

		// find the ancestor with requested frameInd
		TrackHypothesisTreeNode* mostProbHyp = nullptr;
		for (TrackHypothesisTreeNode* pCur : pathNodes)
		{
			if (pCur->FrameInd <= frameInd)
			{
				if (pCur->FrameInd == frameInd)
					mostProbHyp = pCur;
				break;
			}
		}

		if (mostProbHyp != nullptr)
			hypList.push_back(mostProbHyp);
	}
}

TrackHypothesisTreeNode* MultiHypothesisBlobTracker::findHypothesisWithFrameInd(TrackHypothesisTreeNode* start, int frameInd, int trackFamilyId)
{
	std::stack<TrackHypothesisTreeNode*> nodesToProcess;
	nodesToProcess.push(start);
	while (!nodesToProcess.empty())
	{
		TrackHypothesisTreeNode* node = nodesToProcess.top();
		nodesToProcess.pop();

		if (node->FrameInd >= frameInd)
		{
			if (node->FamilyId == trackFamilyId)
				return node;

			// do not process further this branch
		}
		else
		{
			for (std::unique_ptr<TrackHypothesisTreeNode>& childHyp : node->Children)
				nodesToProcess.push(childHyp.get());
		}
	}

	return nullptr;
}

bool MultiHypothesisBlobTracker::getLatestHypothesis(int frameInd, int trackFamilyId, TrackChangePerFrame& outTrackChange)
{
	// find family root
	TrackHypothesisTreeNode* familyRoot = nullptr;
	for (std::unique_ptr<TrackHypothesisTreeNode>& trackHyp : trackHypothesisForestPseudoNode_.Children)
	{
		if (trackHyp->FamilyId == trackFamilyId)
		{
			familyRoot = trackHyp.get();
			break;
		}
	}
	if (familyRoot == nullptr)
		return false;

	// find track node with the given frameInd and trackId
	TrackHypothesisTreeNode* initialTrackHyp = findHypothesisWithFrameInd(familyRoot, frameInd, trackFamilyId);
	if (initialTrackHyp == nullptr)
		return false;

	vector<TrackHypothesisTreeNode*> leafSet;
	getLeafSet(initialTrackHyp, leafSet);

	assert(!leafSet.empty() && "Leaves are populated from seed node, hence there must be at least one leaf");

	auto it = std::max_element(std::begin(leafSet), std::end(leafSet), 
		[](TrackHypothesisTreeNode* h1, TrackHypothesisTreeNode* h2)
	{
		return h1->Score < h2->Score;
	});

	assert(it != std::end(leafSet) && "Element must be found");

	TrackHypothesisTreeNode* mostProbHyp = *it;
	populateTrackChange(mostProbHyp, outTrackChange);
	return true;
}

void MultiHypothesisBlobTracker::logVisualHypothesisTree(int frameInd, const std::string& fileNameTag, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs) const
{
	if (!logDir_.empty())
		writeVisualHypothesisTree(logDir_, frameInd, fileNameTag, bestTrackLeafs, trackHypothesisForestPseudoNode_, pruneWindow_);
}

void MultiHypothesisBlobTracker::setLogDir(const boost::filesystem::path& dir)
{
	logDir_ = dir;
}

void MultiHypothesisBlobTracker::setMovementPredictor(unique_ptr<SwimmerMovementPredictor> movementPredictor)
{
	movementPredictor_.swap(std::move(movementPredictor));
}


void TrackChangeConsistencyChecker::setNextTrackChanges(int readyFrameInd, const std::vector<TrackChangePerFrame>& trackChanges)
{
	if (readyFrameInd == -1)
	{
		CV_Assert(trackChanges.empty() && "Track changes must be empty if there is no updates");
		return;
	}

	for (const TrackChangePerFrame& change : trackChanges)
	{
		CV_Assert(change.FrameInd == readyFrameInd && "Got track change for FrameInd != readyFrameInd");

		int trackId = change.FamilyId;

		switch (change.UpdateType)
		{
		case TrackChangeUpdateType::New:
		{
			TrackChangeClientSide trackItem(trackId, readyFrameInd);
			trackItem.setNextTrackChange(readyFrameInd);

			trackIdToObj_.insert(std::make_pair(trackId, trackItem));

			break;
		}
		case TrackChangeUpdateType::ObservationUpdate:
		case TrackChangeUpdateType::NoObservation:
		{
			auto it = trackIdToObj_.find(trackId);
			if (it == std::end(trackIdToObj_))
			{
				CV_Assert(false && "Got track change for not existent track");
			}
			else
			{
				TrackChangeClientSide& trackItem = it->second;
				trackItem.setNextTrackChange(readyFrameInd);
			}
			break;
		}
		case TrackChangeUpdateType::Pruned:
		{
			auto it = trackIdToObj_.find(trackId);
			if (it == std::end(trackIdToObj_))
			{
				CV_Assert(false && "Got track change for not existent track");
			}
			else
			{
				TrackChangeClientSide& trackItem = it->second;
				trackItem.terminate();
			}
			break;
		}
		}
	}
}
