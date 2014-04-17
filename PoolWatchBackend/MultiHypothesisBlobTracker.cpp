#include <vector>
#include <map>
#include <functional>
#include <cassert>
#include <iostream>
#include <memory>
#include <array>
#include <set>

#include "opencv2/video/tracking.hpp" // cv::KalmanFilter

#include <log4cxx/logger.h>
#include <log4cxx/helpers/exception.h>

#ifdef HAVE_GRAPHVIZ
extern "C"
{
#include <gvc.h>
#include "gvplugin.h"
#include "gvconfig.h"
#include <cghdr.h> // agfindnode_by_id
}
#endif

#include "MatlabInterop.h"
#include "algos1.h"
#include "MultiHypothesisBlobTracker.h"

using namespace std;
using namespace log4cxx;
//using namespace log4cxx::helpers;

extern "C"
{
	struct Int32PtrPair
	{
		int32_t* pFirst;
		int32_t* pLast;
	};
	Int32PtrPair computeTrackIncopatibilityGraph(const int* pEncodedTree, int encodedTreeLength, int collisionIgnoreNodeId, int openBracketLex, int closeBracketLex, Int32Allocator int32Alloc);
}

log4cxx::LoggerPtr MultiHypothesisBlobTracker::log_ = log4cxx::Logger::getLogger("PW.MultiHypothesisBlobTracker");

MultiHypothesisBlobTracker::MultiHypothesisBlobTracker(std::shared_ptr<CameraProjectorBase> cameraProjector, int pruneWindow, float fps)
	:cameraProjector_(cameraProjector),
	fps_(fps),
	pruneWindow_(pruneWindow),
	nextTrackCandidateId_(1),
	swimmerMaxSpeed_(2.3f),         // max speed for swimmers 2.3m/s
	shapeCentroidNoise_(0.5f)
{
	CV_DbgAssert(fps > 0);
	CV_DbgAssert(pruneWindow_ >= 0);

	// root of the tree is an artificial node to gather all trees in hypothesis forest
	trackHypothesisForestPseudoNode_.Id = 0;
	
	// required for compoundId generation
	trackHypothesisForestPseudoNode_.FamilyId = 0;
	trackHypothesisForestPseudoNode_.FrameInd = 0;
	trackHypothesisForestPseudoNode_.ObservationInd = 0;

	// default to Kalman Filter
	movementPredictor_ = std::make_unique<KalmanFilterMovementPredictor>(fps);
}


MultiHypothesisBlobTracker::~MultiHypothesisBlobTracker()
{
}

void MultiHypothesisBlobTracker::trackBlobs(int frameInd, const std::vector<DetectedBlob>& blobs, float fps, float elapsedTimeMs, int& readyFrameInd,
	std::vector<TrackChangePerFrame>& trackChanges)
{
	//swimmerMaxShiftM = elapsedTimeMs * this.v.swimmerMaxSpeed / 1000 + this.humanDetector.shapeCentroidNoise;
	float swimmerMaxShiftM = (elapsedTimeMs * 0.001f) * swimmerMaxSpeed_ + shapeCentroidNoise_;
	growTrackHyposhesisTree(frameInd, blobs, fps, swimmerMaxShiftM);

	vector<TrackHypothesisTreeNode*> leafSet;
	getLeafSet(&trackHypothesisForestPseudoNode_, leafSet);

	LOG4CXX_DEBUG(log_, "leafSet.count=" << leafSet.size() << " (after growth)");
	
	vector<TrackHypothesisTreeNode*> bestTrackLeafs; // 
	findBestTracks(leafSet, bestTrackLeafs);

	LOG4CXX_DEBUG(log_, "bestTrackLeafs.count=" << bestTrackLeafs.size());
#if PW_DEBUG_DETAIL
	if (log_->isDebugEnabled())
		logVisualHypothesisTree(frameInd, "1beforePruning", bestTrackLeafs);
#endif
	
	if (log_->isDebugEnabled() && !bestTrackLeafs.empty())
	{
		// list the best nodes

		stringstream bld;
		for (const TrackHypothesisTreeNode* pLeaf : bestTrackLeafs)
		{
			bld <<endl << "  FamilyId=" << pLeaf->FamilyId << " LeafId=" << pLeaf->Id << " ObsInd=" << pLeaf->ObservationInd << " " << pLeaf->ObservationPos << " Score=" << pLeaf->Score;
		}
		log_->debug(bld.str());
	}

	pruneHypothesisTree(frameInd, bestTrackLeafs, readyFrameInd, trackChanges, pruneWindow_);

#if PW_DEBUG_DETAIL
	if (log_->isDebugEnabled())
		logVisualHypothesisTree(frameInd, "2afterPruning", bestTrackLeafs);
#endif

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

	//
	prevFrameBlobs.resize(blobs.size());
	std::copy(begin(blobs), end(blobs), begin(prevFrameBlobs));
}

void MultiHypothesisBlobTracker::growTrackHyposhesisTree(int frameInd, const std::vector<DetectedBlob>& blobs, float fps, float swimmerMaxShiftM)
{
	vector<TrackHypothesisTreeNode*> leafSet;
	getLeafSet(&trackHypothesisForestPseudoNode_, leafSet);

#if 1 // && log.isDebugEnabled()
	int addedDueNoObservation = 0;
	int addedDueCorrespondence = 0;
	int addedNew = 0;
#endif

	//
	// penalty for missed observation
	// prob 0.4 - penalty - 0.9163
	// prob 0.6 - penalty - 0.5108
	const float probDetection = 0.6f;
	const float penalty = log(1 - probDetection);

	// "Correspondence" hypothesis2: track has correspondent observations in this frame

	for (auto pLeaf : leafSet)
	{
		// associate hypothesis node with each observation

		for (int blobInd = 0; blobInd<blobs.size(); ++blobInd)
		{
			const auto& blob = blobs[blobInd];

			// constraint: blob can shift too much (can't leave the blob's gate)

			cv::Point3f blobCentrWorld = blob.CentroidWorld;
			auto dist = cv::norm(pLeaf->EstimatedPosWorld - blobCentrWorld);
			if (dist > swimmerMaxShiftM)
				continue;

			// constraint: area can't change too much
			if (pLeaf->ObservationInd != -1)
			{
				assert(pLeaf->ObservationInd < prevFrameBlobs.size() && "Cache of blobs for the previous frame was not updated");
				const auto& prevBlob = prevFrameBlobs[pLeaf->ObservationInd];
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
			childHyp.FamilyId = pLeaf->FamilyId;
			childHyp.ObservationInd = blobInd;
			childHyp.FrameInd = frameInd;
			childHyp.ObservationPos = blob.Centroid;
			childHyp.ObservationPosWorld = blobCentrWorld;
			childHyp.CreationReason = hypothesisReason;
			childHyp.KalmanFilterState = cv::Mat(4, 1, CV_32FC1);
			childHyp.KalmanFilterStateCovariance = cv::Mat(4, 4, CV_32FC1);

			//
			float score;
			cv::Point3f estPos;
			movementPredictor_->estimateAndSave(*pLeaf, blobCentrWorld, estPos, score, childHyp);
			childHyp.Score = score;
			childHyp.EstimatedPosWorld = estPos;

#if PW_DEBUG_DETAIL
			LOG4CXX_DEBUG(log_, "grow Corresp FamilyId=" << pLeaf->FamilyId << " LeafId=" << pLeaf->Id << " ChildId=" << pChildHyp->Id << " ObsInd=" << pChildHyp->ObservationInd <<" " << pChildHyp->ObservationPos <<" Score=" <<pChildHyp->Score);
#endif

			pLeaf->addChildNode(std::move(pChildHyp));

			if (log_->isDebugEnabled())
				addedDueCorrespondence++;
		}
	}

	// "No observation" hypothesis: track has no observation in this frame
	{
		for (auto pLeaf : leafSet)
		{
			auto id = nextTrackCandidateId_++;
			auto hypothesisReason = TrackHypothesisCreationReason::NoObservation;

			auto pChildHyp = make_unique<TrackHypothesisTreeNode>();

			TrackHypothesisTreeNode& childHyp = *pChildHyp;
			childHyp.Id = id;
			childHyp.FamilyId = pLeaf->FamilyId;
			childHyp.ObservationInd = DetectionIndNoObservation;
			childHyp.FrameInd = frameInd;
			childHyp.ObservationPos = cv::Point2f(NullPosX, NullPosX);
			childHyp.ObservationPosWorld = cv::Point3f(NullPosX, NullPosX, NullPosX);
			childHyp.CreationReason = hypothesisReason;
			childHyp.KalmanFilterState = cv::Mat(4, 1, CV_32FC1);
			childHyp.KalmanFilterStateCovariance = cv::Mat(4, 4, CV_32FC1);

			//
			float score;
			cv::Point3f estPos;
			movementPredictor_->estimateAndSave(*pLeaf, nullptr, estPos, score, childHyp);
			childHyp.Score = score;
			childHyp.EstimatedPosWorld = estPos;

#if PW_DEBUG_DETAIL
			LOG4CXX_DEBUG(log_, "grow NoObs FamilyId=" << pLeaf->FamilyId << " LeafId=" << pLeaf->Id << " ChildId=" << pChildHyp->Id <<" Score=" <<pChildHyp->Score);
#endif

			pLeaf->addChildNode(std::move(pChildHyp));

			if (log_->isDebugEnabled())
				addedDueNoObservation++;
		}
	}

	// "New track" hypothesis - track got the initial observation in this frame
	
	// initiate new track sparingly(each N frames)

	if (frameInd % initNewTrackDelay_ == 0)
	{
		// associate hypothesis node with each observation
		for (int blobInd = 0; blobInd < (int)blobs.size(); ++blobInd)
		{
			const auto& blob = blobs[blobInd];

			cv::Point3f blobCentrWorld = blob.CentroidWorld;

			auto id = nextTrackCandidateId_++;
			auto hypothesisReason = TrackHypothesisCreationReason::New;

			auto pChildHyp = make_unique<TrackHypothesisTreeNode>();

			TrackHypothesisTreeNode& childHyp = *pChildHyp;
			childHyp.Id = id;
			childHyp.FamilyId = id; // new familyId=id of the initial hypothesis tree node
			childHyp.ObservationInd = blobInd;
			childHyp.FrameInd = frameInd;
			childHyp.ObservationPos = blob.Centroid;
			childHyp.ObservationPosWorld = blobCentrWorld;
			childHyp.CreationReason = hypothesisReason;
			childHyp.EstimatedPosWorld = blobCentrWorld;
			childHyp.KalmanFilterState = cv::Mat(4, 1, CV_32FC1);
			childHyp.KalmanFilterStateCovariance = cv::Mat(4, 4, CV_32FC1);
			float score;
			movementPredictor_->initScoreAndState(frameInd, blobInd, blobCentrWorld, score, childHyp);
			childHyp.Score = score;
			
#if PW_DEBUG_DETAIL
			LOG4CXX_DEBUG(log_, "grow New ChildId=" << pChildHyp->Id << " ObsInd=" << pChildHyp->ObservationInd << " " << pChildHyp->ObservationPos <<" Score=" <<pChildHyp->Score);
#endif

			trackHypothesisForestPseudoNode_.addChildNode(std::move(pChildHyp));
			
			if (log_->isDebugEnabled())
				addedNew++;
		}
	}

	LOG4CXX_DEBUG(log_, "addedNew=" << addedNew << " addedDueCorrespondence=" << addedDueCorrespondence << " addedDueNoObservation=" << addedDueNoObservation);
}

int MultiHypothesisBlobTracker::compoundObservationId(const TrackHypothesisTreeNode& node)
{
	return node.FrameInd*maxObservationsCountPerFrame + node.ObservationInd;
}

void MultiHypothesisBlobTracker::hypothesisTreeToTreeStringRec(const TrackHypothesisTreeNode& startFrom, vector<int32_t>& encodedTreeString)
{
	int compoundId = compoundObservationId(startFrom);
	
	encodedTreeString.push_back(startFrom.Id);
	encodedTreeString.push_back(compoundId);

	if (!startFrom.Children.empty())
	{
		encodedTreeString.push_back(openBracket);

		for (const auto& pChild : startFrom.Children)
			hypothesisTreeToTreeStringRec(*pChild, encodedTreeString);

		encodedTreeString.push_back(closeBracket);
	}
}

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

	Int32PtrPair incompNodesRange = computeTrackIncopatibilityGraph(&encodedTreeString[0], (int)encodedTreeString.size(),
		trackHypothesisForestPseudoNode_.Id, openBracket, closeBracket, int32Alloc);
	
	// Note, we ignore incompNodesRange because the result is populated in incompGraphEdgePairs
	if (incompNodesRange.pFirst == nullptr)
	{
		CV_Assert(incompNodesRange.pLast == nullptr);
		CV_Assert(incompGraphEdgePairs.empty());
	}
	else
	{
		size_t sz = incompNodesRange.pLast - incompNodesRange.pFirst;
		CV_Assert(incompGraphEdgePairs.size() == sz);
	}
}

void MultiHypothesisBlobTracker::findBestTracks(const std::vector<TrackHypothesisTreeNode*>& leafSet,
	std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs)
{
	vector<int32_t> encodedTreeString;
	hypothesisTreeToTreeStringRec(trackHypothesisForestPseudoNode_, encodedTreeString);

	// incompatibility graph in the form of list of edges, each edge is a pair of vertices
	vector<int32_t> incompatibTrackEdges;
	createTrackIncopatibilityGraphDLang(encodedTreeString, incompatibTrackEdges); // [1x(2*edgesCount)]

	CV_Assert(incompatibTrackEdges.size() % 2 == 0 && "Vertices list contains pair (from,to) of vertices for each edge");

	int edgesCount = incompatibTrackEdges.size() / 2;

	// find vertices array

	vector<int32_t> connectedVertices = incompatibTrackEdges;
	std::sort(begin(connectedVertices), end(connectedVertices));
	auto it = std::unique(begin(connectedVertices), end(connectedVertices));
	auto newSize = std::distance(begin(connectedVertices), it);
	connectedVertices.resize(newSize);

	// find map TreeNodeId->TreeNode
	function<void(TrackHypothesisTreeNode& node, map<int32_t, TrackHypothesisTreeNode*>& treeNodeIdToNode)> populateTreeNodeIdToNodeMapFun;
	
	// TODO: why function can't be captured by value (works only when captured by reference)?
	populateTreeNodeIdToNodeMapFun = [&populateTreeNodeIdToNodeMapFun](TrackHypothesisTreeNode& node, map<int32_t, TrackHypothesisTreeNode*>& treeNodeIdToNode) -> void
	{
		treeNodeIdToNode[node.Id] = &node;

		for (const auto& pChild : node.Children)
			populateTreeNodeIdToNodeMapFun(*pChild, treeNodeIdToNode);
	};
	map<int32_t, TrackHypothesisTreeNode*> treeNodeIdToNode;
	populateTreeNodeIdToNodeMapFun(trackHypothesisForestPseudoNode_, treeNodeIdToNode);

	// find entire hypothesis tree vertex ids

	vector<int32_t> allTrackIds;
	allTrackIds.reserve(leafSet.size());
	std::transform(begin(leafSet), end(leafSet), back_inserter(allTrackIds), [](TrackHypothesisTreeNode* pNode) { return (int32_t)pNode->Id; });
	std::sort(begin(allTrackIds), end(allTrackIds));

	// isolated vertices are always in the set of best hypothesis

	vector<int32_t> isolatedVertices; // 
	std::set_difference(begin(allTrackIds), end(allTrackIds), begin(connectedVertices), end(connectedVertices), back_inserter(isolatedVertices));
	std::transform(begin(isolatedVertices), end(isolatedVertices), back_inserter(bestTrackLeafs), [&](int32_t id) { return treeNodeIdToNode[id]; });

	// find weights

	vector<double> vertexWeights(connectedVertices.size());
	for (size_t i = 0; i < connectedVertices.size(); ++i)
	{
		auto vertexId = connectedVertices[i];
		auto pNode = treeNodeIdToNode[vertexId];
		vertexWeights[i] = pNode->Score;
	}
	
	auto gMap = createFromEdgeList(connectedVertices, incompatibTrackEdges);
	auto g = get<0>(gMap);

	for (int i = 0; i < vertexWeights.size(); ++i)
		g.setVertexPayload(i, vertexWeights[i]);

	vector<bool> indepVertexSet;
	maximumWeightIndependentSetNaiveMaxFirst(g, indepVertexSet);
	assert(indepVertexSet.size() == connectedVertices.size());

	for (size_t i = 0; i < indepVertexSet.size(); ++i)
		if (indepVertexSet[i])
		{
			int32_t vertexId = connectedVertices[i];
			auto pNode = treeNodeIdToNode[vertexId];
			bestTrackLeafs.push_back(pNode);
		}
}

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

void MultiHypothesisBlobTracker::enumerateBranchNodesReversed(TrackHypothesisTreeNode* leaf, int pruneWindow, std::vector<TrackHypothesisTreeNode*>& result) const
{
	assert(leaf != nullptr);
	assert(!isPseudoRoot(*leaf) && "Assume starting from terminal, not pseudo node");

	// find new root
	auto current = leaf;
	int stepBack = 1;
	while (true)
	{
		result.push_back(current);

		if (stepBack == pruneWindow)
			return;

		assert(current->Parent != nullptr && "Current node always have the parent node or pseudo root");

		// stop if parent is the pseudo root
		if (isPseudoRoot(*current->Parent))
			return;

		current = current->Parent;
		stepBack++;
	}
}

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

TrackChangePerFrame MultiHypothesisBlobTracker::createTrackChange(TrackHypothesisTreeNode* pNode)
{
	TrackChangePerFrame result;
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

	return result;
}

void MultiHypothesisBlobTracker::pruneHypothesisTree(int frameInd, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs, int& readyFrameInd, std::vector<TrackChangePerFrame>& trackChanges, int pruneWindow)
{
	// find frame with ready track info

	readyFrameInd = frameInd - pruneWindow;
	if (readyFrameInd < 0)
	{
		// no ready track info yet
		readyFrameInd = -1;
		return;
	}

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
			// family was pruned
			TrackChangePerFrame change = createTrackChange(pRoot.get());
			change.UpdateType = TrackChangeUpdateType::Pruned;
			trackChanges.push_back(change);
		}
	}

	// do pruning
	// assume the set of newFamilyRoots are unique, this is because bestTrackLeafs do not collide on any observation

	trackHypothesisForestPseudoNode_.Children.clear();
	for (auto& familtyRoot : newFamilyRoots)
	{
		trackHypothesisForestPseudoNode_.addChildNode(std::move(familtyRoot));
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
	pruneHypothesisTree(frameInd, bestTrackLeafsCache, readyFrameInd, trackChanges, pruneWindow);
	
	CV_Assert(readyFrameInd != -1 && "Can't collect track changes from hypothesis tree");

	bool continue1 = pruneWindow > 0;
	return continue1;
}

#ifdef HAVE_GRAPHVIZ

const char* LayoutNodeNameStr = "name";
extern "C"
{
	__declspec(dllimport) gvplugin_library_t gvplugin_dot_layout_LTX_library;
	__declspec(dllimport) gvplugin_library_t gvplugin_core_LTX_library;
}

char *nodeIdToText(void *state, int objtype, unsigned long id)
{
	Agraph_t *g = (Agraph_t *)state;
	auto node = agfindnode_by_id(g, id);
	if (node == nullptr)
		return "";
	char* name = agget(node, const_cast<char*>(LayoutNodeNameStr));
	return name;
}

void generateHypothesisLayoutTree(Agraph_t *g, Agnode_t* parentOrNull, const TrackHypothesisTreeNode& hypNode, const std::set<int>& liveHypNodeIds)
{
	Agnode_t* layoutNode = agnode(g, nullptr, 1);

	std::stringstream name;
	if (hypNode.Parent == nullptr)
		name << "R";
	else
	{
		if (hypNode.Parent != nullptr && hypNode.Parent->Parent == nullptr) // family root
			name << "F" << hypNode.FamilyId;

		name << "#" << hypNode.Id;

		if (hypNode.ObservationInd != -1)
			name << hypNode.ObservationPos;
		else
			name << "X"; // no observation sign
	}
	
	agsafeset(layoutNode, const_cast<char*>(LayoutNodeNameStr), const_cast<char*>(name.str().c_str()), ""); // label

	// tooltip
	name.str("");
	name << "Id=" << hypNode.Id << " ";
	name << "FrameInd=" << hypNode.FrameInd << "\r\n";
	name << "FamilyId=" << hypNode.FamilyId << "\n\r";
	name << "ObsInd=" << hypNode.ObservationInd << "\r";
	name << "ObsPos=" << hypNode.ObservationPos << "\n";
	name << "Score=" << hypNode.Score << endl;
	name << "Reason=" << toString(hypNode.CreationReason);
	agsafeset(layoutNode, "tooltip", const_cast<char*>(name.str().c_str()), "");

	agsafeset(layoutNode, "margin", "0", ""); // edge length=0
	agsafeset(layoutNode, "fixedsize", "false", ""); // size is dynamically calculated
	agsafeset(layoutNode, "width", "0", ""); // windth=minimal
	agsafeset(layoutNode, "height", "0", "");

	if (liveHypNodeIds.find(hypNode.Id) != std::end(liveHypNodeIds))
		agsafeset(layoutNode, "color", "green", "");

	if (parentOrNull != nullptr)
	{
		Agedge_t* edge = agedge(g, parentOrNull, layoutNode, nullptr, 1);
	}

	for (const std::unique_ptr<TrackHypothesisTreeNode>& pChildHyp : hypNode.Children)
	{
		generateHypothesisLayoutTree(g, layoutNode, *pChildHyp.get(), liveHypNodeIds);
	}
}

#endif

void MultiHypothesisBlobTracker::logVisualHypothesisTree(int frameInd, const std::string& fileNameTag, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs) const
{
#ifdef HAVE_GRAPHVIZ
	CV_Assert(logDir_ != nullptr && "Log directory must be set");

	// prepare the set of new hypothesis
	std::set<int> liveNodeIds;
	std::vector<TrackHypothesisTreeNode*> pathNodes;
	for (const auto pLeaf : bestTrackLeafs)
	{
		pathNodes.clear();
		enumerateBranchNodesReversed(pLeaf, pruneWindow_, pathNodes);

		for (auto pNode : pathNodes)
		{
			liveNodeIds.insert(pNode->Id);
		}
	}

	lt_symlist_t lt_preloaded_symbols[] = {
		{ "gvplugin_core_LTX_library", (void*)(&gvplugin_core_LTX_library) },
		{ "gvplugin_dot_layout_LTX_library", (void*)(&gvplugin_dot_layout_LTX_library) },
		{ 0, 0 }
	};

	/* set up a graphviz context */
	const int demandLoading = 1;
	GVC_t *gvc = gvContextPlugins(lt_preloaded_symbols, demandLoading);

	/* Create a simple digraph */
	Agraph_t *g = agopen(nullptr, Agdirected, nullptr);
	
	// change node formatting function
	g->clos->disc.id->print = nodeIdToText; 
	
	agsafeset(g, "rankdir", "LR", "");
	//agsafeset(g, "ratio", "1.3", "");
	agsafeset(g, "ranksep", "0", "");
	agsafeset(g, "nodesep", "0", ""); // distance between nodes in one rank

	generateHypothesisLayoutTree(g, nullptr, trackHypothesisForestPseudoNode_, liveNodeIds);

	gvLayout(gvc, g, "dot");

	std::stringstream outFileName;
	outFileName << "hypTree_";
	outFileName.fill('0');
	outFileName.width(4);
	outFileName << frameInd << "_" << fileNameTag << ".svg";
	boost::filesystem::path outFilePath = *logDir_ / outFileName.str();

	gvRenderFilename(gvc, g, "svg", outFilePath.string().c_str());

	/* Free layout data */
	gvFreeLayout(gvc, g);

	/* Free graph structures */
	agclose(g);

	/* close output file, free context, and return number of errors */
	gvFreeContext(gvc);
#endif
}

void MultiHypothesisBlobTracker::setMovementPredictor(unique_ptr<SwimmerMovementPredictor> movementPredictor)
{
	movementPredictor_.swap(std::move(movementPredictor));
}

