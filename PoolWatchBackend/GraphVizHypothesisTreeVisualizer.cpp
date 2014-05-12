#include "GraphVizHypothesisTreeVisualizer.h"
#include <set>
#include <sstream>

using namespace std;

#ifdef LOG_VISUAL_HYPOTHESIS_TREE
extern "C"
{
#include <gvc.h>
#include "gvplugin.h"
#include "gvconfig.h"
#include <cghdr.h> // agfindnode_by_id
}

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

void writeVisualHypothesisTree(const boost::filesystem::path& logDir, int frameInd, const std::string& fileNameTag, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs, const TrackHypothesisTreeNode& root, int pruneWindow)
{
#ifdef LOG_VISUAL_HYPOTHESIS_TREE
	CV_Assert(!logDir.empty() && "Log directory must be set");

	// prepare the set of new hypothesis
	std::set<int> liveNodeIds;
	std::vector<TrackHypothesisTreeNode*> pathNodes;
	for (const auto pLeaf : bestTrackLeafs)
	{
		pathNodes.clear();
		enumerateBranchNodesReversed(pLeaf, pruneWindow, pathNodes);

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

	generateHypothesisLayoutTree(g, nullptr, root, liveNodeIds);

	gvLayout(gvc, g, "dot");

	std::stringstream outFileName;
	outFileName << "hypTree_";
	outFileName.fill('0');
	outFileName.width(4);
	outFileName << frameInd << "_" << fileNameTag << ".svg";
	boost::filesystem::path outFilePath = logDir / outFileName.str();

	gvRenderFilename(gvc, g, "svg", outFilePath.string().c_str());

	/* Free layout data */
	gvFreeLayout(gvc, g);

	/* Free graph structures */
	agclose(g);

	/* close output file, free context, and return number of errors */
	gvFreeContext(gvc);
#endif
}
