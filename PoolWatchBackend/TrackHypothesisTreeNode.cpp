#include "TrackHypothesisTreeNode.h"
#include <cassert>

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
std::unique_ptr<TrackHypothesisTreeNode> TrackHypothesisTreeNode::pullChild(TrackHypothesisTreeNode* pChild, bool updateChildrenCollection)
{
	std::unique_ptr<TrackHypothesisTreeNode> result;
	int i = 0;
	for (auto& childPtr : Children)
	{
		if (childPtr.get() == pChild)
		{
			result.swap(childPtr);
			if (updateChildrenCollection)
				Children.erase(std::begin(Children) + i);
			break;
		}
		++i;
	}
	return std::move(result);
}

void enumerateBranchNodesReversed(TrackHypothesisTreeNode* leaf, int takeCount, std::vector<TrackHypothesisTreeNode*>& result, TrackHypothesisTreeNode* leafParentOrNull)
{
	assert(leaf != nullptr);
	//assert(!isPseudoRoot(*leaf) && "Assume starting from terminal, not pseudo node");

	// find new root
	auto current = leaf;
	int stepBack = 1;
	while (true)
	{
		result.push_back(current);

		if (stepBack == takeCount)
			return;

		// find parent
		TrackHypothesisTreeNode* parent;
		if (current == leaf && leafParentOrNull != nullptr)
		{
			parent = leafParentOrNull;
		}
		else
			parent = current->Parent;

		// stop if get root
		if (parent == nullptr)
			return;

		current = parent;
		stepBack++;
	}
}

