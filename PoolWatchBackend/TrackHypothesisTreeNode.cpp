#include "TrackHypothesisTreeNode.h"


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