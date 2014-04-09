#include <vector>
#include <map>
#include <functional>
#include <cassert>
#include <iostream>
#include <memory>
#include <array>
#include <set>

#include "opencv2/video/tracking.hpp"

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

MultiHypothesisBlobTracker::MultiHypothesisBlobTracker(std::shared_ptr<CameraProjector> cameraProjector, int pruneWindow, float fps)
	:cameraProjector_(cameraProjector),
	fps_(fps),
	pruneWindow_(pruneWindow),
	nextTrackCandidateId_(1),
	swimmerMaxSpeed_(2.3f),         // max speed for swimmers 2.3m/s
	kalmanFilter_(cv::KalmanFilter(4, 2))
{
	CV_DbgAssert(fps > 0);
	CV_DbgAssert(pruneWindow_ >= 0);

	// root of the tree is an artificial node to gather all trees in hypothesis forest
	trackHypothesisForestPseudoNode_.Id = 0;
	
	// required for compoundId generation
	trackHypothesisForestPseudoNode_.FamilyId = 0;
	trackHypothesisForestPseudoNode_.FrameInd = 0;
	trackHypothesisForestPseudoNode_.ObservationInd = 0;

	//
	initKalmanFilter(kalmanFilter_, fps);
}


MultiHypothesisBlobTracker::~MultiHypothesisBlobTracker()
{
}

void MultiHypothesisBlobTracker::trackBlobs(int frameInd, const std::vector<DetectedBlob>& blobs, float fps, float elapsedTimeMs, int& frameIndWithTrackInfo,
	std::vector<TrackChangePerFrame>& trackStatusList)
{
#if 0 //_DEBUG
	// the set of tracked families
	// it must not decrease during track hypothesis generation and pruning
	std::set<int> familyIdSet;
	for (const std::unique_ptr<TrackHypothesisTreeNode>& pRoot : trackHypothesisForestPseudoNode_.Children)
		familyIdSet.insert(pRoot->FamilyId);
#endif

	float shapeCentroidNoise = 0.5f;
	//swimmerMaxShiftPerFrameM = elapsedTimeMs * this.v.swimmerMaxSpeed / 1000 + this.humanDetector.shapeCentroidNoise;
	float swimmerMaxShiftPerFrameM = elapsedTimeMs * swimmerMaxSpeed_ / 1000 + shapeCentroidNoise;
	growTrackHyposhesisTree(frameInd, blobs, fps, swimmerMaxShiftPerFrameM);

	if (log_->isDebugEnabled())
		logHypothesisTree(frameInd, "beforePruning", vector<TrackHypothesisTreeNode*>());

	vector<TrackHypothesisTreeNode*> leafSet;
	getLeafSet(&trackHypothesisForestPseudoNode_, leafSet);

	LOG4CXX_DEBUG(log_, "leafSet.count=" << leafSet.size() << " (after growth)");
	
	vector<TrackHypothesisTreeNode*> bestTrackLeafs; // 
	findBestTracks(leafSet, bestTrackLeafs);

	LOG4CXX_DEBUG(log_, "bestTrackLeafs.count=" << bestTrackLeafs.size());

	std::vector<TrackChangePerFrame> trackStatusListTmp;
	int frameIndWithTrackInfo2 = collectTrackChanges(frameInd, bestTrackLeafs, trackStatusListTmp);
	
	pruneHypothesisTreeNew(frameInd, bestTrackLeafs, frameIndWithTrackInfo, trackStatusList);

	if (log_->isDebugEnabled())
		logHypothesisTree(frameInd, "afterPruning", bestTrackLeafs);

	//CV_Assert(frameIndWithTrackInfo2 == frameIndWithTrackInfo);
	//CV_Assert(trackStatusListTmp.size() == trackStatusList.size());

	if (frameIndWithTrackInfo != -1)
	{
		//assert(!trackStatusList.empty());
	}

	if (log_->isDebugEnabled())
	{
		// list track hypothesis changes

		stringstream bld;
		bld << "frameIndWithTrackInfo=" << frameIndWithTrackInfo << " trackStatusList=" << trackStatusList.size() <<endl;
		for (const TrackChangePerFrame& change : trackStatusList)
		{
			switch (change.UpdateType)
			{
			case TrackChangeUpdateType::New:
				bld << "new";
				break;
			case TrackChangeUpdateType::ObservationUpdate:
				bld << "upd";
				break;
			case TrackChangeUpdateType::NoObservation:
				bld << "noObs";
				break;
			case TrackChangeUpdateType::Pruned:
				bld << "del";
				break;
			}

			bld << " FamilyId=" << change.FamilyId <<" ObsInd=" <<change.ObservationInd <<" " <<change.ObservationPosPixExactOrApprox <<endl;
		}
		log_->debug(bld.str());
	}
	
	//pruneHypothesisTree(bestTrackLeafs);

#if 0//_DEBUG
	// check all family roots are preserved (generation of new families is possible)
	std::set<int> familyIdSetAfter;
	for (const std::unique_ptr<TrackHypothesisTreeNode>& pRoot : trackHypothesisForestPseudoNode_.Children)
		familyIdSetAfter.insert(pRoot->FamilyId);
	
	CV_Assert(familyIdSet.size() <= familyIdSetAfter.size());
	for (int familyId : familyIdSet)
	{
		bool found = familyIdSetAfter.find(familyId) != familyIdSetAfter.end();
		CV_Assert(found && "OldRoot from previous iteration was not traversed from new set of best leaves");
	}
#endif
}

void MultiHypothesisBlobTracker::growTrackHyposhesisTree(int frameInd, const std::vector<DetectedBlob>& blobs, float fps, float swimmerMaxShiftPerFrameM)
{
	vector<TrackHypothesisTreeNode*> leafSet;
	getLeafSet(&trackHypothesisForestPseudoNode_, leafSet);

#if _DEBUG
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

			cv::Point3f blobCentrWorld = blob.CentroidWorld;
			auto dist = cv::norm(pLeaf->EstimatedPosWorld - blobCentrWorld);
			if (dist > swimmerMaxShiftPerFrameM)
				continue;

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

			//cv::Mat_<float> state(1, 4, 0.0f); // [X,Y, vx=0, vy=0]
			//state(0, 0) = blobCentrWorld.x;
			//state(0, 1) = blobCentrWorld.y;
			//childHyp.KalmanFilterState = state;
			//childHyp.KalmanFilterStateCovariance = cv::Mat_<float>::eye(state.cols, state.cols);
			//childHyp.KalmanFilterStatePrev = cv::Mat_<float>();
			//childHyp.KalmanFilterStateCovariancePrev = cv::Mat_<float>();

			//float score = calcTrackShiftScoreNew(pLeaf, blobCentrWorld, hypothesisReason, fps,
			//	childHyp.EstimatedPosWorld,
			//	childHyp.KalmanFilterState,
			//	childHyp.KalmanFilterStateCovariance);
			//childHyp.ScoreKalman = score;

			// prepare Kalman Filter state
			cv::KalmanFilter& kalmanFilter = kalmanFilter_; // use cached Kalman Filter object
			kalmanFilter.statePost = pLeaf->KalmanFilterState;
			kalmanFilter.errorCovPost = pLeaf->KalmanFilterStateCovariance;

			//
			cv::Mat predictedPos2 = kalmanFilter.predict();
			cv::Point3f predictedPos = cv::Point3f(predictedPos2.at<float>(0, 0), predictedPos2.at<float>(1, 0), zeroHeight);

			//
			auto obsPosWorld2 = cv::Mat_<float>(2, 1); // ignore Z
			obsPosWorld2(0) = blobCentrWorld.x;
			obsPosWorld2(1) = blobCentrWorld.y;

			auto estPosMat = kalmanFilter.correct(obsPosWorld2);
			childHyp.EstimatedPosWorld = cv::Point3f(estPosMat.at<float>(0, 0), estPosMat.at<float>(1, 0), zeroHeight);

			auto shiftScore = kalmanFilterDistance(kalmanFilter, obsPosWorld2);
			childHyp.Score = pLeaf->Score + shiftScore;

			// save Kalman Filter state
			childHyp.KalmanFilterState = kalmanFilter.statePost;
			childHyp.KalmanFilterStateCovariance = kalmanFilter.errorCovPost;

			LOG4CXX_DEBUG(log_, "grow FamilyId=" << pLeaf->FamilyId << " LeafId=" << pLeaf->Id << " Corresp ChildId=" << pChildHyp->Id << " ObsInd=" << pChildHyp->ObservationInd <<" " << pChildHyp->ObservationPos);

			pLeaf->addChildNode(std::move(pChildHyp));
#if _DEBUG
			addedDueCorrespondence++;
#endif
		}
	}

	// "No observation" hypothesis: track has no observation in this frame
	{
		for (auto pLeaf : leafSet)
		{
			auto id = nextTrackCandidateId_++;
			auto hypothesisReason = TrackHypothesisCreationReason::NoObservation;

			auto pChildHyp = make_unique<TrackHypothesisTreeNode>();

			auto blobCentrWorld = cv::Point3f(NullPosX, NullPosX, NullPosX);

			TrackHypothesisTreeNode& childHyp = *pChildHyp;
			childHyp.Id = id;
			childHyp.FamilyId = pLeaf->FamilyId;
			childHyp.ObservationInd = DetectionIndNoObservation;
			childHyp.FrameInd = frameInd;
			childHyp.ObservationPos = cv::Point2f(NullPosX, NullPosX);
			childHyp.ObservationPosWorld = blobCentrWorld;
			childHyp.CreationReason = hypothesisReason;

			//float score = calcTrackShiftScoreNew(nullptr, blobCentrWorld, hypothesisReason, fps,
			//	childHyp.EstimatedPosWorld,
			//	childHyp.KalmanFilterState,
			//	childHyp.KalmanFilterStateCovariance);
			//childHyp.ScoreKalman = score;

			// prepare Kalman Filter state
			cv::KalmanFilter& kalmanFilter = kalmanFilter_; // use cached Kalman Filter object
			kalmanFilter.statePost = pLeaf->KalmanFilterState;
			kalmanFilter.errorCovPost = pLeaf->KalmanFilterStateCovariance;

			//
			cv::Mat predictedPos2 = kalmanFilter.predict();
			cv::Point3f predictedPos = cv::Point3f(predictedPos2.at<float>(0, 0), predictedPos2.at<float>(1, 0), 0); // z=0
			
			// as there is no observation, the predicted position is the best estimate
			childHyp.EstimatedPosWorld = predictedPos;
			childHyp.Score = pLeaf->Score + penalty; // punish for no observation

			// save Kalman Filter state
			childHyp.KalmanFilterState = kalmanFilter.statePost;
			childHyp.KalmanFilterStateCovariance = kalmanFilter.errorCovPost;

			LOG4CXX_DEBUG(log_, "grow FamilyId=" << pLeaf->FamilyId << " LeafId=" << pLeaf->Id << " NoObs ChildId=" << pChildHyp->Id);

			pLeaf->addChildNode(std::move(pChildHyp));
#if _DEBUG
			addedDueNoObservation++;
#endif
		}
	}

	// "New track" hypothesis - track got the initial observation in this frame
	
	// try to initiate new track sparingly(each N frames)
	const int initNewTrackDelay = 7;

	if (frameInd % initNewTrackDelay == 0)
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
			
			// initial track score
			int nt = 5; // number of expected targets
			int fa = 25; // number of expected FA(false alarms)
			float precision = nt / (float)(nt + fa);
			float initialScore = -log(precision);

			// if initial score is large, then tracks with missed detections may
			// be evicted from hypothesis tree
			initialScore = abs(6 * penalty);

			childHyp.Score = initialScore;

			//float score = calcTrackShiftScoreNew(nullptr, blobCentrWorld, hypothesisReason, fps,
			//	childHyp.EstimatedPosWorld,
			//	childHyp.KalmanFilterState,
			//	childHyp.KalmanFilterStateCovariance);
			//childHyp.ScoreKalman = score;

			//float score = calcTrackShiftScore(nullptr, &childHyp, fps);
			//childHyp.ScoreKalman = score;

			// save Kalman Filter state
			cv::Mat_<float> state(KalmanFilterDynamicParamsCount, 1, 0.0f); // [X,Y, vx=0, vy=0]'
			state(0) = blobCentrWorld.x;
			state(1) = blobCentrWorld.y;
			childHyp.KalmanFilterState = state;
			childHyp.KalmanFilterStateCovariance = cv::Mat_<float>::eye(KalmanFilterDynamicParamsCount, KalmanFilterDynamicParamsCount);
			
			LOG4CXX_DEBUG(log_, "grow New ChildId=" << pChildHyp->Id << " ObsInd=" << pChildHyp->ObservationInd << " " << pChildHyp->ObservationPos);

			trackHypothesisForestPseudoNode_.addChildNode(std::move(pChildHyp));
#if _DEBUG
			addedNew++;
#endif
		}
	}

#if _DEBUG
	LOG4CXX_DEBUG(log_, "addedNew=" << addedNew << " addedDueCorrespondence=" << addedDueCorrespondence << " addedDueNoObservation=" << addedDueNoObservation);
#endif
}

float MultiHypothesisBlobTracker::calcTrackShiftScore(const TrackHypothesisTreeNode* parentNode,
	const TrackHypothesisTreeNode* trackNode, float fps)
{
	// penalty for missed observation
	// prob 0.4 - penalty - 0.9163
	// prob 0.6 - penalty - 0.5108
	float probDetection = 0.6f;
	float penalty = log(1 - probDetection);

	// initial track score
	int nt = 5; // number of expected targets
	int fa = 25; // number of expected FA(false alarms)
	float precision = nt / (float)(nt + fa);
	float initialScore = -log(precision);

	// if initial score is large, then tracks with missed detections may
	// be evicted from hypothesis tree
	initialScore = abs(6 * penalty);

	if (trackNode->CreationReason == TrackHypothesisCreationReason::New)
	{
		assert(parentNode == nullptr);
		return initialScore;
	}
	
	assert(parentNode != nullptr);

	//cv::KalmanFilter kalmanFilter(4, 2);
	//kalmanFilter.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
	//
	//// 2.3m / s is max speed for swimmers
	//// let say 0.5m / s is an average speed
	//// estimate sigma as one third of difference between max and mean shift per frame
	//float maxShiftM = 2.3f / fps;
	//float meanShiftM = 0.5f / fps;
	//float sigma = (maxShiftM - meanShiftM) / 3;
	//kalmanFilter.processNoiseCov = (cv::Mat_<float>(1, 1) << sigma ^ 2);

	//kalmanFilter.measurementMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
	//kalmanFilter.measurementNoiseCov = (cv::Mat_<float>(1, 1) << (5.0 / 3) ^ 2);

	cv::KalmanFilter& kalmanFilter = kalmanFilter_;
	initKalmanFilter(kalmanFilter, fps);
	kalmanFilter.statePost= parentNode->KalmanFilterState;
	kalmanFilter.errorCovPost = parentNode->KalmanFilterStateCovariance;

	//
	cv::Mat predictedPos2 = kalmanFilter.predict();
	cv::Point3f predictedPos = cv::Point3f(predictedPos2.at<float>(0, 0), predictedPos2.at<float>(1, 0), 0); // z=0
	
	if (trackNode->ObservationPosWorld.x == NullPosX)
		return penalty;

	auto obsPosWorld = cv::Mat_<float>(3, 1);
	obsPosWorld(0, 0) = trackNode->ObservationPosWorld.x;
	obsPosWorld(1, 0) = trackNode->ObservationPosWorld.y;
	obsPosWorld(2, 0) = trackNode->ObservationPosWorld.z;
	
	auto result = kalmanFilterDistance(kalmanFilter, obsPosWorld);	
	return result;
}

void MultiHypothesisBlobTracker::calcTrackShiftScoreNew(const TrackHypothesisTreeNode* parentNode,
	const cv::Point3f& blobPosWorld, TrackHypothesisCreationReason hypothesisReason, float fps,
	float& resultScore,
	cv::Point3f& resultEstimatedPosWorld,
	cv::Mat_<float>& resultKalmanFilterState,
	cv::Mat_<float>& resultKalmanFilterStateCovariance)
{
	// penalty for missed observation
	// prob 0.4 - penalty - 0.9163
	// prob 0.6 - penalty - 0.5108
	const float probDetection = 0.6f;
	float penalty = log(1 - probDetection);

	if (hypothesisReason == TrackHypothesisCreationReason::New)
	{
		assert(parentNode == nullptr);
		
		// initial track score
		int nt = 5; // number of expected targets
		int fa = 25; // number of expected FA(false alarms)
		float precision = nt / (float)(nt + fa);
		float initialScore = -log(precision);

		// if initial score is large, then tracks with missed detections may
		// be evicted from hypothesis tree
		initialScore = abs(6 * penalty);

		//
		resultScore = initialScore;
		resultEstimatedPosWorld = blobPosWorld;

		cv::Mat_<float> state(1, 4, 0.0f); // [X,Y, vx=0, vy=0]
		state(0, 0) = blobPosWorld.x;
		state(0, 1) = blobPosWorld.y;
		resultKalmanFilterState = state;

		resultKalmanFilterStateCovariance = cv::Mat_<float>::eye(state.cols, state.cols);
	}

	assert(parentNode != nullptr);

	cv::KalmanFilter& kalmanFilter = kalmanFilter_; // use cached Kalman Filter object
	kalmanFilter.statePost = parentNode->KalmanFilterState;
	kalmanFilter.errorCovPost = parentNode->KalmanFilterStateCovariance;

	//
	cv::Mat predictedPos2 = kalmanFilter.predict();
	cv::Point3f predictedPos = cv::Point3f(predictedPos2.at<float>(0, 0), predictedPos2.at<float>(1, 0), 0); // z=0

	if (hypothesisReason == TrackHypothesisCreationReason::NoObservation)
	{
		assert(blobPosWorld.x == NullPosX);

		resultScore = penalty;
		resultEstimatedPosWorld = predictedPos;
	}
	else
	{
		assert(hypothesisReason == TrackHypothesisCreationReason::SequantialCorrespondence);

		auto obsPosWorld = cv::Mat_<float>(3, 1);
		obsPosWorld(0, 0) = blobPosWorld.x;
		obsPosWorld(1, 0) = blobPosWorld.y;
		obsPosWorld(2, 0) = blobPosWorld.z;

		auto estPosMat = kalmanFilter.correct(obsPosWorld);
		resultEstimatedPosWorld = cv::Point3f(estPosMat.at<float>(0, 0), estPosMat.at<float>(1, 0), estPosMat.at<float>(2, 0));

		resultScore = kalmanFilterDistance(kalmanFilter, obsPosWorld);
	}
	
	resultKalmanFilterState = kalmanFilter.statePost;
	resultKalmanFilterStateCovariance = kalmanFilter.errorCovPost;
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

mxArray* MultiHypothesisBlobTracker::createTrackIncopatibilityGraphDLang(const vector<int32_t>& encodedTreeString)
{
	mxArray* pIncompNodesMat = nullptr;

	Int32Allocator int32Alloc;
	int32Alloc.pUserData = &pIncompNodesMat;
	int32Alloc.CreateArrayInt32 = [](size_t celem, void* pUserData) -> int32_t*
	{
		mxArray** ppMat = reinterpret_cast<mxArray**>(pUserData);
		assert(*ppMat == nullptr && "computeTrackIncopatibilityGraph must request memory only once");

		*ppMat = mxCreateNumericMatrix(1, celem, mxINT32_CLASS, mxREAL);

		return reinterpret_cast<int32_t*>(mxGetData(*ppMat));
	};
	int32Alloc.DestroyArrayInt32 = [](int32_t* pInt32, void* pUserData) -> void
	{
		mxArray** ppMat = reinterpret_cast<mxArray**>(pUserData);
		assert(*ppMat != nullptr && "computeTrackIncopatibilityGraph at first must allocate memory");
		mxDestroyArray(*ppMat);
	};

	Int32PtrPair incompNodesRange = computeTrackIncopatibilityGraph(&encodedTreeString[0], (int)encodedTreeString.size(),
		trackHypothesisForestPseudoNode_.Id, openBracket, closeBracket, int32Alloc);
	
	assert(pIncompNodesMat != nullptr);
	assert(incompNodesRange.pFirst == mxGetData(pIncompNodesMat));

	return pIncompNodesMat;
}

void MultiHypothesisBlobTracker::findBestTracks(const std::vector<TrackHypothesisTreeNode*>& leafSet,
	std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs)
{
	vector<int32_t> encodedTreeString;
	hypothesisTreeToTreeStringRec(trackHypothesisForestPseudoNode_, encodedTreeString);

	mxArray* incompatibTrackEdgesMat = createTrackIncopatibilityGraphDLang(encodedTreeString); // [1x(2*edgesCount)]
	auto pIncompatibTrackEdgesMat = unique_ptr<mxArray, mxArrayDeleter>(incompatibTrackEdgesMat);

	size_t dim1 = mxGetM(incompatibTrackEdgesMat);
	assert(dim1 == 1);

	size_t ncols = mxGetN(incompatibTrackEdgesMat);
	assert(ncols % 2 == 0 && "Vertices list contains pair (from,to) of vertices for each edge");

	int edgesCount = (int)ncols / 2;

	auto pVertices = reinterpret_cast<int32_t*>(mxGetData(incompatibTrackEdgesMat));

	// find vertices array

	vector<int32_t> connectedVertices(pVertices, pVertices + edgesCount*2);
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
	
	auto edgesListMat = incompatibTrackEdgesMat;
	auto edgesListDataPtr = (int*)mxGetPr(edgesListMat);

	//vector<int> edgeList(edgesCount * 2);
	//for (int i = 0; i < edgesCount; ++i)
	//{
	//	edgeList[i * 2] = edgesListDataPtr[i];
	//	//edgeList[i * 2 + 1] = edgesListDataPtr[edgesCount + i];
	//	edgeList[i * 2 + 1] = edgesListDataPtr[i * 2 + 1];
	//}
	vector<int> edgeList(edgesListDataPtr, edgesListDataPtr + edgesCount * 2);
	auto gMap = createFromEdgeList(connectedVertices, edgeList);
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

unique_ptr<TrackHypothesisTreeNode> MultiHypothesisBlobTracker::findNewFamilyRoot(TrackHypothesisTreeNode* leaf)
{
	// find new root
	auto current = leaf;
	int stepBack = 1;
	while (true)
	{
		if (stepBack == pruneWindow_)
			break;

		// stop if parent is the pseudo root
		if (current->Parent == nullptr || isPseudoRoot(*current->Parent))
			break;

		current = current->Parent;
		stepBack++;
	}

	assert(current->Parent != nullptr);

	// ask parent to find unique_ptr corresponding to current
	unique_ptr<TrackHypothesisTreeNode> result;
	for (auto& childPtr : current->Parent->Children)
	{
		if (childPtr.get() == current)
		{
			result.swap(childPtr);
			break;
		}
	}
	assert(result != nullptr);
	return std::move(result);
}

TrackHypothesisTreeNode* MultiHypothesisBlobTracker::findNewFamilyRoot2(TrackHypothesisTreeNode* leaf)
{
	assert(leaf != nullptr);
	assert(!isPseudoRoot(*leaf) && "Assume starting from terminal, not pseudo node");

	// find new root
	auto current = leaf;
	int stepBack = 1;
	while (true)
	{
		if (stepBack == pruneWindow_)
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

void MultiHypothesisBlobTracker::initKalmanFilter(cv::KalmanFilter& kalmanFilter, float fps)
{
	kalmanFilter.init(KalmanFilterDynamicParamsCount, 2, 0);
	
	kalmanFilter.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);

	// 2.3m / s is max speed for swimmers
	// let say 0.5m / s is an average speed
	// estimate sigma as one third of difference between max and mean shift per frame
	float maxShiftM = 2.3f / fps;
	float meanShiftM = 0.5f / fps;
	float sigma = (maxShiftM - meanShiftM) / 3;
	setIdentity(kalmanFilter.processNoiseCov, cv::Scalar::all(sigma*sigma));

	kalmanFilter.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
	const float measS = 5.0f / 3;
	setIdentity(kalmanFilter.measurementNoiseCov, cv::Scalar::all(measS*measS));
}

void MultiHypothesisBlobTracker::pruneHypothesisTree(const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs)
{
	vector<unique_ptr<TrackHypothesisTreeNode>> newFamilyRoots;
	// gather new family roots
	for (const auto pLeaf : bestTrackLeafs)
	{
		newFamilyRoots.push_back(findNewFamilyRoot(pLeaf));
	}

	// assume the set of newFamilyRoots are unique, this is because bestTrackLeafs do not collide on any observation
	trackHypothesisForestPseudoNode_.Children.clear();
	for (auto& familtyRoot : newFamilyRoots)
	{
		trackHypothesisForestPseudoNode_.addChildNode(std::move(familtyRoot));
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

	return result;
}

int MultiHypothesisBlobTracker::collectTrackChanges(int frameInd, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs, std::vector<TrackChangePerFrame>& trackStatusList)
{
	int readyFrameInd = frameInd - pruneWindow_;
	if (readyFrameInd < 0)
		readyFrameInd = -1;

	for (const auto pNode : bestTrackLeafs)
	{
		auto pFixedAncestor = pNode->getAncestor(pruneWindow_); 
		
		if (pFixedAncestor == nullptr)
			continue;
		if (isPseudoRoot(*pFixedAncestor))
			continue;

		TrackChangePerFrame change;
		change.FamilyId = pFixedAncestor->FamilyId;
		
		if (pFixedAncestor->CreationReason == TrackHypothesisCreationReason::New)
			change.UpdateType = TrackChangeUpdateType::New;
		else if (pFixedAncestor->CreationReason == TrackHypothesisCreationReason::SequantialCorrespondence)
			change.UpdateType = TrackChangeUpdateType::ObservationUpdate;
		else if (pFixedAncestor->CreationReason == TrackHypothesisCreationReason::NoObservation)
			change.UpdateType = TrackChangeUpdateType::NoObservation;

		auto estimatedPos = pFixedAncestor->EstimatedPosWorld;

		change.EstimatedPosWorld = estimatedPos;
		change.ObservationInd = pFixedAncestor->ObservationInd;
		
		auto obsPos = pFixedAncestor->ObservationPos;
		if (obsPos.x == NullPosX)
			obsPos = cameraProjector_->worldToCamera(estimatedPos);
		change.ObservationPosPixExactOrApprox = obsPos;

		trackStatusList.push_back(change);
	}

	return readyFrameInd;
}

void MultiHypothesisBlobTracker::pruneHypothesisTreeNew(int frameInd, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs, int& readyFrameInd, std::vector<TrackChangePerFrame>& trackChanges)
{
	// find frame with ready track info

	readyFrameInd = frameInd - pruneWindow_;
	if (readyFrameInd < 0)
	{
		// no ready track info yet
		readyFrameInd = -1;
		return;
	}

	//

	vector<unique_ptr<TrackHypothesisTreeNode>> newFamilyRoots;
	// gather new family roots
	for (const auto pLeaf : bestTrackLeafs)
	{
		TrackHypothesisTreeNode* newRoot = findNewFamilyRoot2(pLeaf);

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

	// find pruned families (the set of tracks, initiated from the same observation)

	std::set<int> familyIdSetBeforePrune;
	for (const std::unique_ptr<TrackHypothesisTreeNode>& pRoot : trackHypothesisForestPseudoNode_.Children)
	{
		if (pRoot != nullptr)
			familyIdSetBeforePrune.insert(pRoot->FamilyId);
	}

	// collect pruned hypothesis changes

	for (const std::unique_ptr<TrackHypothesisTreeNode>& pRoot : trackHypothesisForestPseudoNode_.Children)
	{
		if (pRoot == nullptr)
			continue;
		int familyId = pRoot->FamilyId;

		bool found = familyIdSetBeforePrune.find(familyId) != familyIdSetBeforePrune.end();
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

#ifdef HAVE_GRAPHVIZ

const char* LayoutNodeNameStr = "name";

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
	if (hypNode.Parent != nullptr && hypNode.Parent->Parent == nullptr) // family root
		name << "F" << hypNode.FamilyId;
	name << "#" << hypNode.Id;
	agsafeset(layoutNode, const_cast<char*>(LayoutNodeNameStr), const_cast<char*>(name.str().c_str()), ""); // label

	// tooltip
	name.str("");
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

extern "C"
{
	__declspec(dllimport) gvplugin_library_t gvplugin_dot_layout_LTX_library;
	__declspec(dllimport) gvplugin_library_t gvplugin_core_LTX_library;
}

#endif

void MultiHypothesisBlobTracker::logHypothesisTree(int frameInd, const std::string& fileNameTag, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs) const
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

float normalizedDistance(const cv::Mat& pos, const cv::Mat& mu, const cv::Mat& sigma)
{
	cv::Mat zd = pos - mu;
	cv::Mat mahalanobisDistance = zd.t() * sigma.inv() * zd;
	//auto md = mahalanobisDistance.at<float>(0, 0);
	double determinant = cv::determinant(sigma);
	float dist = mahalanobisDistance.at<float>(0, 0) + static_cast<float>(log(determinant));
	return dist;
}

/// Computes distance between predicted position of the Kalman Filter and given observed position.
/// It corresponds to Matlab's vision.KalmanFilter.distance() function.
/// Parameter observedPos is a column vector.
// http://www.mathworks.com/help/vision/ref/vision.kalmanfilter.distance.html
float kalmanFilterDistance(const cv::KalmanFilter& kalmanFilter, const cv::Mat& observedPos)
{
	cv::Mat residualCovariance = kalmanFilter.measurementMatrix * kalmanFilter.errorCovPost * kalmanFilter.measurementMatrix.t() + kalmanFilter.measurementNoiseCov;
	//auto rc = residualCovariance.at<float>(0, 0);

	cv::Mat mu = kalmanFilter.measurementMatrix * kalmanFilter.statePre;
	//auto mm = mu.at<float>(0, 0);

	float result = normalizedDistance(observedPos, mu, residualCovariance);
	return result;
}