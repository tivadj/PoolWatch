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
import PoolWatchHelpersDLang.GraphHelpers;

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
	alias TrackHypothesisTreeNode* NodeId;

	NodeId hypTreeNode;

	this(TrackHypothesisTreeNode* hypTreeNode) @nogc
	{
		this.hypTreeNode = hypTreeNode;
	}

	bool isNull(NodeId node) @nogc
	{
		return node == null;
	}

	bool hasChildren(NodeId node) @nogc
	{
		return node.ChildrenCount > 0;
	}

	NodeId parent(NodeId node) @nogc
	{
		return NodeId(node.Parent);
	}

	auto nodes() @nogc
	{
		static struct HypNodesIterator
		{
			NodeId startFrom;

			private int traverseTreeRec(NodeId current, int delegate(NodeId) @nogc dg) @nogc
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
			int opApply(int delegate(NodeId) @nogc dg) @nogc
			{
				return traverseTreeRec(startFrom, dg);
			}
		}

		return HypNodesIterator(hypTreeNode);
	}

	unittest
	{
		//static assert(isInputRange!HypNodesIterator);
	}
}

void findCollisionEdges(RootedTreeT,OutRangeT)(ref RootedTreeT hypTree, OutRangeT edgesList, int collisionIgnoreNodeId)
{
	alias RootedTreeT.NodeId NodeId;

	// find leaves

	CppVector!(RootedTreeT.NodeId) leaves;
	foreach(n; getLeaves(hypTree))
		leaves.pushBack(n);

	// check compatibility of each pair of tracks

	struct FrameAndObsInd
	{
		int FrameInd;
		int ObservationInd;
	}
	scope bool[FrameAndObsInd] observIdSet; // 

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

void findCollisionGraph(TrackHypothesisTreeT,ConflictingTracksHypothesisGraphT)(ref TrackHypothesisTreeT hypTree, int collisionIgnoreNodeId, CppVector!(HypTreeAdapter.NodeId) leaves, ref ConflictingTracksHypothesisGraphT conflictTrackGraph)
{
	// init bidirectional hypothesis tree node - track conflicting graph association
	
	conflictTrackGraph.reserveNodes(leaves.length);

	foreach(TrackHypothesisTreeT.NodeId leaf; leaves)
	{
		ConflictingTracksHypothesisGraphT.NodeId mwispNode = conflictTrackGraph.addNode();
		mwispNode.NodePayload.HypNode =leaf;

		leaf.MwispNode = mwispNode;
	}

	// check compatibility of each pair of tracks

	struct FrameAndObsInd
	{
		int FrameInd;
		int ObservationInd;
	}
	scope bool[FrameAndObsInd] observIdSet; // TODO: require @nogc hash table

	for (int n1=0; n1 < leaves.length-1; n1++)
	{
		auto leaf1 = leaves[n1];

		// populate set of observationIds for the first leaf
		// NOTE: two hypothesis can't conflict on 'no observation'

		observIdSet.clear;

		auto populateBranchObservationIds = delegate void(ref TrackHypothesisTreeT tree, TrackHypothesisTreeT.NodeId leaf1)
		{
			foreach(TrackHypothesisTreeT.NodeId node; tree.branchReversed(leaf1))
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
			foreach(TrackHypothesisTreeT.NodeId internalNode2; hypTree.branchReversed(leaf2))
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
				auto node1 = cast(ConflictingTracksHypothesisGraphT.NodeId)leaf1.MwispNode;
				auto node2 = cast(ConflictingTracksHypothesisGraphT.NodeId)leaf2.MwispNode;
				conflictTrackGraph.addEdge(node1, node2);
			}
		}
	}
}

// The payload to find Maximum Weigth Independent Set (MWIS) payload.
struct MwisPayload
{
	bool IsVisited;
	bool InIndependentSet;
	TrackHypothesisTreeNode* HypNode; // reference to a tree hypothesis node
}

@nogc float findMaximumWeightIndependentSetMaxFirst(ref UndirectedAdjacencyGraphVectorImpl!MwisPayload mwisGraph, UndirectedAdjacencyGraphVectorImpl!(MwisPayload).NodeId[] descWeightNodes)
{
	assert(mwisGraph.nodesCount == descWeightNodes.length);

	alias UndirectedAdjacencyGraphVectorImpl!MwisPayload MwisGraph;

	// prepare workspace
	foreach(MwisGraph.NodeId n; mwisGraph.nodes)
	{
		MwisPayload* data = &n.NodePayload;
		data.IsVisited = false;
		data.InIndependentSet = false;
	}

	float independentSetWeight = 0;

	foreach(MwisGraph.NodeId n; descWeightNodes)
	{
		MwisPayload* data = &n.NodePayload;
		if (data.IsVisited)
			continue;

		// include in the independent set
		data.IsVisited = true;
		data.InIndependentSet = true;
		independentSetWeight += data.HypNode.Score;

		// exclude all neigbour edges
		foreach(MwisGraph.NodeId adj; mwisGraph.adjacentNodes(n))
		{
			MwisPayload* adjData = &adj.NodePayload;

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
void findMaximumWeightIndependentSet_MaxFirstMultiAttempts(MwisGraphT)(ref MwisGraphT mwisGraph, int minAttemptsCount, CppVector!(MwisGraphT.NodeId)* pMaxIndependentSet)
{
	assert(minAttemptsCount >= 1);

	CppVector!(MwisGraphT.NodeId) nodes;
	nodes.reserve(mwisGraph.nodesCount);

	//copy(mwisGraph.nodes, nodes); // TODO: ERROR:
	foreach(MwisGraphT.NodeId n; mwisGraph.nodes)
	{
		nodes.pushBack(n);
	}

	auto decWeightFun = function (MwisGraphT.NodeId n1, MwisGraphT.NodeId n2) @nogc { return n1.NodePayload.HypNode.Score > n2.NodePayload.HypNode.Score; };
	sort!decWeightFun(nodes[]);

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
	for (int attempt=0; attempt < attemptCount; ++attempt, bringToFrontNode++)
	{
		swap(nodes[0],  nodes[bringToFrontNode]); // perturb weight desc order

		//
		float independentSetWeight = findMaximumWeightIndependentSetMaxFirst(mwisGraph, nodes[]);
		if (independentSetWeight > maxIndependentSetWeight)
		{
			// save the better result
			maxIndependentSetWeight = independentSetWeight;

			pMaxIndependentSet.clear;

			foreach(MwisGraphT.NodeId n; mwisGraph.nodes)
			{
				MwisPayload* data = &n.NodePayload;
				if (data.InIndependentSet)
					pMaxIndependentSet.pushBack(n);
			}
		}
		
		swap(nodes[0],  nodes[bringToFrontNode]); // restore weight desc order
	}
}


void findBestTracks(TrackHypothesisTreeNode* hypTree, int collisionIgnoreNodeId, int minAttemptsCount, CppVector!(TrackHypothesisTreeNode*)* pBestTracks)
{
	auto hypTreeAdapt = HypTreeAdapter(hypTree);

	// find leaves

	CppVector!(HypTreeAdapter.NodeId) leaves;
	foreach(e; getLeaves(hypTreeAdapt))
		leaves.pushBack(e);

	//
	alias UndirectedAdjacencyGraphVectorImpl!MwisPayload ConflictingTrackHypothesisGraphT;
	ConflictingTrackHypothesisGraphT conflictTrackGraph;
	findCollisionGraph(hypTreeAdapt, collisionIgnoreNodeId, leaves, conflictTrackGraph);

	// find isolated vertices, they always are the best candidates
	// leaves - conflictTrackGraph.nodes
	sort(leaves[]);
	scope auto sortedConnectedNodes = new HypTreeAdapter.NodeId[conflictTrackGraph.nodesCount];
	foreach(i,n; conflictTrackGraph.nodes)
		sortedConnectedNodes[i] = n.NodePayload.HypNode;

	sort(sortedConnectedNodes);

	scope auto isolatedNodes = setDifference(leaves[], sortedConnectedNodes);

	//
	CppVector!(ConflictingTrackHypothesisGraphT.NodeId) maxIndependentSet;
	maxIndependentSet.reserve(conflictTrackGraph.nodesCount);

	findMaximumWeightIndependentSet_MaxFirstMultiAttempts!(ConflictingTrackHypothesisGraphT)(conflictTrackGraph, minAttemptsCount, &maxIndependentSet);

	// populate result
	int expectLength = maxIndependentSet.length * 2;
	pBestTracks.reserve(expectLength);

	foreach(TrackHypothesisTreeNode* hypNode; isolatedNodes)
		pBestTracks.pushBack(hypNode);

	foreach(ConflictingTrackHypothesisGraphT.NodeId n; maxIndependentSet)
	{
		MwisPayload* data = &n.NodePayload;
		TrackHypothesisTreeNode* hypNode = data.HypNode;
		pBestTracks.pushBack(hypNode);
	}
}
