#pragma once
#include <vector>
#include <memory>
#include <opencv2/core.hpp>

#include "PoolWatchFacade.h"

enum class TrackHypothesisCreationReason
{
	SequantialCorrespondence = 1,  // this hypothesis has corresponding observation in the previous frame
	New = 2,                       // this is brand new hypothesis
	NoObservation = 3              // this hypothesis assumes that there is no observation of the tracked object in this frame
};

std::string toString(TrackHypothesisCreationReason reason);


/** Represents node in the tree of hypothesis. */
struct TrackHypothesisTreeNode
{
	int Id;
	int FamilyId;
	float Score; // determines validity of the hypothesis(from root to this node); the more is better
	std::vector<std::unique_ptr<TrackHypothesisTreeNode>> Children;
	TrackHypothesisTreeNode* Parent;
	int ObservationInd;
	int FrameInd;
	cv::Point2f ObservationPos;      // in pixels
	cv::Point3f ObservationPosWorld; // in meters
	cv::Point3f EstimatedPosWorld;
	TrackHypothesisCreationReason CreationReason;
	cv::Mat_<float> KalmanFilterState; // [X, Y, vx, vy] in meters and m/sec
	cv::Mat_<float> KalmanFilterStateCovariance;  // [4x4]

	void addChildNode(std::unique_ptr<TrackHypothesisTreeNode> childHyp);
	TrackHypothesisTreeNode* getAncestor(int ancestorIndex);
	std::unique_ptr<TrackHypothesisTreeNode> pullChild(TrackHypothesisTreeNode* pChild);
};

