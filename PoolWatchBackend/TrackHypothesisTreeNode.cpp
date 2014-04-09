#include "TrackHypothesisTreeNode.h"

std::string toString(TrackHypothesisCreationReason reason)
{
	const char* reasonStr;
	switch (reason)
	{
	case TrackHypothesisCreationReason::SequantialCorrespondence:
		reasonStr = "upd";
		break;
	case TrackHypothesisCreationReason::New:
		reasonStr = "new";
		break;
	case TrackHypothesisCreationReason::NoObservation:
		reasonStr = "noObs";
		break;
	default:
		reasonStr = "TrackHypothesisCreationReason";
		break;
	}
	return std::string(reasonStr);
}

void TrackHypothesisTreeNode::addChildNode(std::unique_ptr<TrackHypothesisTreeNode> childHyp)
{
	Children.push_back(std::move(childHyp));
	
	assert(childHyp == nullptr);

	Children.back()->Parent = this;
}

TrackHypothesisTreeNode* TrackHypothesisTreeNode::getAncestor(int ancestorIndex)
{
	assert(ancestorIndex >= 0);
	TrackHypothesisTreeNode* result = this;
	while (ancestorIndex > 0)
	{
		result = result->Parent;
		ancestorIndex--;

		if (result == nullptr)
			break;
	}
	return result;
}

// Ask this node to find unique_ptr corresponding to the child.
std::unique_ptr<TrackHypothesisTreeNode> TrackHypothesisTreeNode::pullChild(TrackHypothesisTreeNode* pChild)
{
	std::unique_ptr<TrackHypothesisTreeNode> result;
	for (auto& childPtr : Children)
	{
		if (childPtr.get() == pChild)
		{
			result.swap(childPtr);
			break;
		}
	}
	return std::move(result);
}