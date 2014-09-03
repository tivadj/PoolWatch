#pragma once
#include <opencv2/core.hpp>
#include <boost/optional.hpp>

struct TrackHypothesisTreeNode;

// Estimates the position of the blob given the state of the blob (position etc).
class PW_EXPORTS SwimmerMovementPredictor
{
public:
	virtual ~SwimmerMovementPredictor() = default;

	virtual void initScoreAndState(int frameInd, int observationInd, const cv::Point3f& blobCentrWorld, float& score, TrackHypothesisTreeNode& saveNode) = 0;

	// Estimates position and score of movement from position of curNode to blobCentrWorld. Saves state into saveNode.
	// deltaMovementScore = delta score of shifting from current node to proposed blob's center.
	virtual void estimateAndSave(const TrackHypothesisTreeNode& curNode, const boost::optional<cv::Point3f>& blobCentrWorld, cv::Point3f& estPos, float& deltaMovementScore, TrackHypothesisTreeNode& saveNode) = 0;

	virtual float maxShiftPerFrame() const = 0;
};
