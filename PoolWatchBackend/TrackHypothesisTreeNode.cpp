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

void enumerateBranchNodesReversed(TrackHypothesisTreeNode* leaf, int pruneWindow, std::vector<TrackHypothesisTreeNode*>& result)
{
	assert(leaf != nullptr);
	//assert(!isPseudoRoot(*leaf) && "Assume starting from terminal, not pseudo node");

	// find new root
	auto current = leaf;
	int stepBack = 1;
	while (true)
	{
		result.push_back(current);

		if (stepBack == pruneWindow)
			return;

		//assert(current->Parent != nullptr && "Current node always have the parent node or pseudo root");

		// stop if parent is the pseudo root
		//if (isPseudoRoot(*current->Parent))
		//	return;

		current = current->Parent;
		if (current == nullptr)
			return;

		stepBack++;
	}
}