module PoolWatchHelpersDLang.MultiHypothesisBlobTracker;
import std.stdio;
import std.stdint;
import std.algorithm;
import std.format;
import std.string;
import std.array;
import std.range;
import std.typecons; // tuple
//import std.copy;
import std.container; // SList
import std.math;

import PoolWatchHelpersDLang.UndirectedAdjacencyVectorGraph;
import PoolWatchHelpersDLang.RootedUndirectedTree;

// NOTE: must bitwise match corresponding CPP structure
struct TrackHypothesisTreeNode
{
	int32_t Id;
	float Score; // determines validity of the hypothesis(from root to this node); the more is better
	int32_t FrameInd;
	int ObservationOrNoObsId = -1; // >= 0 (eg 0,1,2) for valid observation; <0 (eg -1,-2,-3) to mean no observation for each hypothesis node
	TrackHypothesisTreeNode** ChildrenArray = null; // pointer to the array of children
	int32_t ChildrenCount = 0;
	TrackHypothesisTreeNode* Parent = null;
	void* MwispNode = null;
}

struct HypTreeAdapter
{
	struct NodeId
	{
		alias pNode_ this;
		TrackHypothesisTreeNode* pNode_;
		this(TrackHypothesisTreeNode* pNode) { pNode_ = pNode; }
		bool isNull()
		{
			return pNode_ == null;
		}
		//TrackHypothesisTreeNode* get()
		//{
		//    return pNode_;
		//}
	}

	NodeId hypTreeNode;

	this(TrackHypothesisTreeNode* hypTreeNode)
	{
		this.hypTreeNode = NodeId(hypTreeNode);
	}

	bool hasChildren(NodeId node)
	{
		return node.ChildrenCount > 0;
	}

	NodeId parent(NodeId node)
	{
		return NodeId(node.Parent);
	}

	struct HypNodesIterator
	{
		NodeId startFrom;
		this(NodeId startFrom) 
		{
			this.startFrom = startFrom;
		}
		private int traverseTreeRec(NodeId current, int delegate(NodeId) dg)
		{
			int result = dg(current);
			if (result) return result;

			auto childrenPtrs = current.ChildrenArray[0..current.ChildrenCount];
			foreach (NodeId pChild; childrenPtrs)
			{
				result = traverseTreeRec(pChild, dg);
				if (result) return result;
			}

			return 0;
		}
		int opApply(int delegate(NodeId) dg)
		{
			return traverseTreeRec(startFrom, dg);
		}
	}

	auto nodes()
	{
		return HypNodesIterator(hypTreeNode);
	}
}

void findCollisionEdges(RootedTreeT,OutRangeT)(ref RootedTreeT hypTree, OutRangeT edgesList, int collisionIgnoreNodeId)
{
	alias RootedTreeT.NodeId NodeId;

	// find leaves

	RootedTreeT.NodeId[] leaves;
	RefAppender!(RootedTreeT.NodeId[]) leavesApppender = appender(&leaves);

	getLeaves(hypTree, leavesApppender);

	// check compatibility of each pair of tracks

	struct FrameAndObsInd
	{
		int FrameInd;
		int ObservationInd;
	}
	bool[FrameAndObsInd] observIdSet; // 

	for (int n1=0; n1 < leaves.length-1; n1++)
	{
		auto leaf1 = leaves[n1];
		//writeln("leaf1ObservId=", leaf1Data.ObservationId);

		// populate set of observationIds for the first leaf
		// NOTE: two hypothesis can't conflict on 'no observation'

		observIdSet.clear;

		auto populateBranchObservationIds = delegate void(ref RootedTreeT tree, RootedTreeT.NodeId leaf1)
		{
			foreach(RootedTreeT.NodeId node; tree.branchReversed(leaf1))
			{
				auto nodeId = node.Id;
				if (nodeId == collisionIgnoreNodeId)
					break;

				auto frameInd = node.FrameInd;
				auto obsInd = node.ObservationOrNoObsId;

				FrameAndObsInd conflictObs;
				conflictObs.FrameInd = frameInd;
				conflictObs.ObservationInd = obsInd;
				observIdSet[conflictObs] = true;
			}
		};
		populateBranchObservationIds(hypTree, leaf1);

		//write("keys:");
		//foreach(id; observIdSet.keys)
		//{
		//    write(" ", id);
		//}
		//writeln;

		// check collisions

		for (int n2=n1+1; n2 < leaves.length; n2++)
		{
			auto leaf2 = leaves[n2];

			bool isCollision = false;

			// iterate through leaf2 branch
			foreach(RootedTreeT.NodeId internalNode2; hypTree.branchReversed(leaf2))
			{
				if (internalNode2.Id == collisionIgnoreNodeId)
					break;

				auto frameInd2 = internalNode2.FrameInd;
				auto obsInd2 = internalNode2.ObservationOrNoObsId;
				//writeln("node2 ObservId=", obsId2);

				FrameAndObsInd conflictObs2;
				conflictObs2.FrameInd = frameInd2;
				conflictObs2.ObservationInd = obsInd2;

				auto collide = observIdSet.get(conflictObs2, false);
				if (collide)
				{
					isCollision = true;
					break;
				}
			}

			if (isCollision)
			{
				auto edge = tuple(leaf1.Id,  leaf2.Id);
				edgesList.put(edge);
				//writeln("edge ", edge[0], " ", edge[1]);
			}
		}
	}
}

struct MwispPayload
{
	bool IsVisited;
	bool InIndependentSet;
	TrackHypothesisTreeNode* HypNode;
}

void findCollisionGraph(RootedTreeT)(ref RootedTreeT hypTree, int collisionIgnoreNodeId, RootedTreeT.NodeId[] leaves, ref UndirectedAdjacencyGraphVectorImpl!MwispPayload mwispGraph)
{
	alias UndirectedAdjacencyGraphVectorImpl!MwispPayload MwispGraph;

	// init bidirectional HypTreeNode-MwispGraphNode association
	foreach(RootedTreeT.NodeId leaf; leaves)
	{
		MwispGraph.NodeId mwispNode = mwispGraph.addNode();
		mwispNode.NodePayload.HypNode =leaf;

		leaf.MwispNode = mwispNode;
	}

	// check compatibility of each pair of tracks

	struct FrameAndObsInd
	{
		int FrameInd;
		int ObservationInd;
	}
	bool[FrameAndObsInd] observIdSet; // 

	for (int n1=0; n1 < leaves.length-1; n1++)
	{
		auto leaf1 = leaves[n1];

		// populate set of observationIds for the first leaf
		// NOTE: two hypothesis can't conflict on 'no observation'

		observIdSet.clear;

		auto populateBranchObservationIds = delegate void(ref RootedTreeT tree, RootedTreeT.NodeId leaf1)
		{
			foreach(RootedTreeT.NodeId node; tree.branchReversed(leaf1))
			{
				auto nodeId = node.Id;
				if (nodeId == collisionIgnoreNodeId)
					break;

				auto frameInd = node.FrameInd;
				auto obsInd = node.ObservationOrNoObsId;

				FrameAndObsInd conflictObs;
				conflictObs.FrameInd = frameInd;
				conflictObs.ObservationInd = obsInd;
				observIdSet[conflictObs] = true;
			}
		};
		populateBranchObservationIds(hypTree, leaf1);

		// check collisions

		for (int n2=n1+1; n2 < leaves.length; n2++)
		{
			auto leaf2 = leaves[n2];

			bool isCollision = false;

			// iterate through leaf2 branch
			foreach(RootedTreeT.NodeId internalNode2; hypTree.branchReversed(leaf2))
			{
				if (internalNode2.Id == collisionIgnoreNodeId)
					break;

				auto frameInd2 = internalNode2.FrameInd;
				auto obsInd2 = internalNode2.ObservationOrNoObsId;
				//writeln("node2 ObservId=", obsId2);

				FrameAndObsInd conflictObs2;
				conflictObs2.FrameInd = frameInd2;
				conflictObs2.ObservationInd = obsInd2;

				auto collide = observIdSet.get(conflictObs2, false);
				if (collide)
				{
					isCollision = true;
					break;
				}
			}

			if (isCollision)
			{
				auto edge = tuple(leaf1.Id,  leaf2.Id);
				auto node1 = cast(MwispGraph.NodeId)leaf1.MwispNode;
				auto node2 = cast(MwispGraph.NodeId)leaf2.MwispNode;
				mwispGraph.addEdge(node1, node2);
			}
		}
	}
}

float findMaximumWeightIndependentSetMaxFirst(ref UndirectedAdjacencyGraphVectorImpl!MwispPayload mwispGraph, UndirectedAdjacencyGraphVectorImpl!(MwispPayload).NodeId[] descWeightNodes)
{
	assert(mwispGraph.nodesCount == descWeightNodes.length);

	alias UndirectedAdjacencyGraphVectorImpl!MwispPayload MwisGraph;

	// prepare workspace
	foreach(MwisGraph.NodeId n; mwispGraph.nodes)
	{
		MwispPayload* data = &n.NodePayload;
		data.IsVisited = false;
		data.InIndependentSet = false;
	}

	float independentSetWeight = 0;

	foreach(MwisGraph.NodeId n; descWeightNodes)
	{
		MwispPayload* data = &n.NodePayload;
		if (data.IsVisited)
			continue;

		// include in the independent set
		data.IsVisited = true;
		data.InIndependentSet = true;
		independentSetWeight += data.HypNode.Score;

		// exclude all neigbour edges
		foreach(MwisGraph.NodeId adj; mwispGraph.adjacentNodes(n))
		{
			MwispPayload* adjData = &adj.NodePayload;

			assert(!adjData.InIndependentSet, "Neighbour node can't be in independent set");

			adjData.IsVisited = true;
		}
	}

	return independentSetWeight;
}

// The algorithm sorts nodes in weight descending order. Then it swaps first node and k-th node (k=1..minAttemptsCount)
// and performs naive max weight independent set implementation.
// minAttemptsCount [1..numNodes]=number of perturbations of the initially sorted (in weight desc order) nodes.
// This parameter should be > 1, otherwise algo choses non-optimal solution even for simplest scenarios
void findMaximumWeightIndependentSet_MaxFirstMultiAttempts(ref UndirectedAdjacencyGraphVectorImpl!MwispPayload mwispGraph, int minAttemptsCount, UndirectedAdjacencyGraphVectorImpl!(MwispPayload).NodeId[]* pMaxIndependentSet)
{
	assert(minAttemptsCount >= 1);

	alias UndirectedAdjacencyGraphVectorImpl!MwispPayload MwisGraph;

	MwisGraph.NodeId[] nodes;
	nodes.reserve(mwispGraph.nodesCount);
	auto app = appender(&nodes);

	//copy(mwispGraph.nodes, nodes); // TODO: ERROR:
	foreach(MwisGraph.NodeId n; mwispGraph.nodes)
	{
		app.put(n);
	}

	auto decWeightFun = function (MwisGraph.NodeId n1, MwisGraph.NodeId n2) { return n1.NodePayload.HypNode.Score > n2.NodePayload.HypNode.Score; };
	sort!decWeightFun(nodes);


	//
	int attemptCount = cast(int)log(nodes.length);
	if (attemptCount < minAttemptsCount)
		attemptCount = minAttemptsCount;
	if (attemptCount > nodes.length)
		attemptCount = cast(int)nodes.length;

	// try to use naive max weight independent set on slightly perturbed list
	// of sorted in weight descending order nodes

	float maxIndependentSetWeight = -float.max;
	
	
	int bringToFrontNode = 0;
	for (int attempt=0; attempt < minAttemptsCount; ++attempt, bringToFrontNode++)
	{
		swap(nodes[0],  nodes[bringToFrontNode]); // perturb weight desc order

		//
		float independentSetWeight = findMaximumWeightIndependentSetMaxFirst(mwispGraph, nodes);
		if (independentSetWeight > maxIndependentSetWeight)
		{
			// save the better result
			maxIndependentSetWeight = independentSetWeight;

			pMaxIndependentSet.length = 0;

			scope RefAppender!(MwisGraph.NodeId[]) maxIndependentSetApp = appender(pMaxIndependentSet);
			foreach(MwisGraph.NodeId n; mwispGraph.nodes)
			{
				MwispPayload* data = &n.NodePayload;
				if (data.InIndependentSet)
					maxIndependentSetApp.put(n);
			}
		}
		
		swap(nodes[0],  nodes[bringToFrontNode]); // restore weight desc order
	}
}


void findBestTracks(OutHypRangeT)(TrackHypothesisTreeNode* hypTree, int collisionIgnoreNodeId, int minAttemptsCount, OutHypRangeT bestTracks)
{
	auto hypTreeAdapt = HypTreeAdapter(hypTree);

	// find leaves

	HypTreeAdapter.NodeId[] leaves;
	RefAppender!(HypTreeAdapter.NodeId[]) leavesApppender = appender(&leaves);

	getLeaves(hypTreeAdapt, leavesApppender);

	//
	alias UndirectedAdjacencyGraphVectorImpl!MwispPayload MwisGraph;
	MwisGraph mwispGraph;
	findCollisionGraph(hypTreeAdapt, collisionIgnoreNodeId, leaves, mwispGraph);

	// find isolated vertices, they always are the best candidates
	// leaves - mwispGraph.nodes
	sort(leaves);
	auto sortedConnectedNodes = new HypTreeAdapter.NodeId[mwispGraph.nodesCount];
	foreach(i,n; mwispGraph.nodes)
		sortedConnectedNodes[i] = n.NodePayload.HypNode;

	sort(sortedConnectedNodes);

	auto isolatedNodes = setDifference(leaves, sortedConnectedNodes);

	//
	MwisGraph.NodeId[] maxIndependentSet;
	findMaximumWeightIndependentSet_MaxFirstMultiAttempts(mwispGraph, minAttemptsCount, &maxIndependentSet);

	// populate result

	foreach(TrackHypothesisTreeNode* hypNode; isolatedNodes)
		bestTracks.put(hypNode);

	foreach(MwisGraph.NodeId n; maxIndependentSet)
	{
		MwispPayload* data = &n.NodePayload;
		TrackHypothesisTreeNode* hypNode = data.HypNode;
		bestTracks.put(hypNode);
	}
}
