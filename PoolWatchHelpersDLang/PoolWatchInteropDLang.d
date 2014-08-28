module PoolWatchHelpersDLang.PoolWatchInteropDLang;

import std.stdio;
import std.stdint;
import std.algorithm;
import std.format;
import std.string;
import std.array;
import std.range;
import std.typecons; // tuple
import std.container; // SList

import PoolWatchHelpersDLang.traversal;
import PoolWatchHelpersDLang.connect;
import PoolWatchHelpersDLang.GraphHelpers;
import PoolWatchHelpersDLang.MatrixUndirectedGraph;
import PoolWatchHelpersDLang.MatrixUndirectedGraph2;
import PoolWatchHelpersDLang.NearestCommonAncestorOfflineAlgorithm;
import PoolWatchHelpersDLang.RootedUndirectedTree;
import PoolWatchHelpersDLang.UndirectedAdjacencyVectorGraph;
import PoolWatchHelpersDLang.MultiHypothesisBlobTracker;

extern void poolTest1()
{
	writeln("poolTest1");
}

extern(C) int sum(int x, int y) { return x + y; }
extern(C)
{
	int sumScope(int x, int y) { return x + y; }
}

//extern (C) void pwFree(void* pMem)
//{
//    core.stdc.stdlib.free(pMem);
//}
//
//extern (C) void* pwMalloc(size_t cBytes)
//{
//    return core.stdc.stdlib.malloc(cBytes);
//}

// NOTE: Declarations below must be in sync with CPP declarations.
extern (C)
{
	alias void* mxArrayPtr;

	alias mxArrayPtr function(size_t celem) pwCreateArrayInt32Fun;
	alias void* function(mxArrayPtr pMat) pwGetDataPtrFun;
	alias size_t function (mxArrayPtr pMat) pwGetNumberOfElementsFun;
	alias void function(mxArrayPtr pMat) pwDestroyArrayFun;
	alias void function(const(char)* msg) DebugFun;
	
	struct mxArrayFuns_tag
	{
		pwCreateArrayInt32Fun CreateArrayInt32;
		pwGetDataPtrFun GetDataPtr;
		pwGetNumberOfElementsFun GetNumberOfElements;
		pwDestroyArrayFun DestroyArray;
		DebugFun logDebug;
	};

	alias int32_t* function(size_t celem, void* pUserData) pwCreateArrayInt32FunNew;
	alias void function(int32_t* pInt32, void* pUserData) pwDestroyArrayInt32FunNew;

	struct Int32Allocator
	{
		pwCreateArrayInt32FunNew CreateArrayInt32;
		pwDestroyArrayInt32FunNew DestroyArrayInt32;
		void* pUserData; // data which will be passed to Create/Destroy methods by server code
	};

	struct Int32PtrPair
	{
		int32_t* pFirst;
		int32_t* pLast;
	};

	// Wrapper for std::vector<void*>
	struct CppVectorPtrWrapper
	{
		void* Vector;
		void function(CppVectorPtrWrapper* sender, void* ptr) @nogc PushBack;
	}
}

extern (C) void computeMWISP(int* pVertices, int nodesCount, int* pEdgePerRow, int edgePerRowSize, double* pWeights, int weightsSize, bool* pIndepVerticesMask, int indepVerticesMaskSize)
{
	assert(pVertices != null);
	assert(nodesCount >= 0);

	assert(pEdgePerRow != null);
	assert(edgePerRowSize >= 0);

	assert(pWeights != null);
	assert(weightsSize >= 0);

	assert(pIndepVerticesMask != null);
	assert(indepVerticesMaskSize >= 0);

	//

	int[] vertices = pVertices[0..nodesCount];
	int[] edgePerRow = pEdgePerRow[0..edgePerRowSize];
	double[] weights = pWeights[0..weightsSize];
	bool[] indepVerticesMask = pIndepVerticesMask[0..indepVerticesMaskSize];

	bool include = true;
	foreach(ref vert; indepVerticesMask)
	{
		vert = include;
		include = !include;
	}
}

extern (C) int computeConnectedComponentsCount301(int* pEdgeListColumnwise, int edgeListArraySize)
{
	return 888;
}
extern (C) int computeConnectedComponentsCount302(int* pEdgeListColumnwise, int edgeListArraySize, DebugFun debugFun)
{
	debugFun("computeConnectedComponentsCount302 ptr\n".ptr); // works, no '\0' is required
	debugFun("computeConnectedComponentsCount302 toStringz\n".toStringz); // works, no '\0' is required
	return 777;
}
extern (C) int computeConnectedComponentsCount(int* pEdgeListColumnwise, int edgeListArraySize, DebugFun debugFun)
{
	debug(PRINTF) debugFun("computeConnectedComponentsCount toStringz\n".toStringz);

	assert(pEdgeListColumnwise != null);
	assert(edgeListArraySize >= 0);

	//

	int[] edgeListColumnwise = pEdgeListColumnwise[0..edgeListArraySize];
	debug(PRINTF) debugFun(format("edgeListColumnwise.length=%d\n",edgeListColumnwise.length).ptr);

	// normalize vertices ids
	int[] edgeListCopy = new int[edgeListColumnwise.length];
	edgeListCopy[] = edgeListColumnwise[];

	edgeListCopy.sort;
	auto vertices = array(edgeListCopy.uniq);

	// construct map: original vertexId -> zero-based vertex index
	int[int] mapOriginalToNormalized;
	for (int i=0; i<vertices.length; ++i)
	{
		auto origVertex = vertices[i];
		mapOriginalToNormalized[origVertex] = i;
	}
	//for (int j=0; j<vertices.length; ++j)
	//{
	//    auto orig=vertices[j];
	//    writeln(orig);
	//}

	//for (int j=0; j<vertices.length; ++j)
	//{
	//    auto orig=vertices[j];
	//    auto norm=mapOriginalToNormalized[orig];
	//    write(orig);
	//    write(" ");
	//    writeln(norm);
	//}

	//
	auto normalizedEdgeList = new int[edgeListColumnwise.length];
	for (int k=0; k<edgeListColumnwise.length; k++)
	{
		auto orig = edgeListColumnwise[k];
		//write(orig); write(" ");
		auto normVertex = mapOriginalToNormalized[orig];
		//writeln(normVertex);
		normalizedEdgeList[k] = normVertex;
	}

	//auto edgesCount = edgeListColumnwise.length/2;
	//for (int j=0; j<edgesCount; j++)
	//{
	//    auto n1 = normalizedEdgeList[j];
	//    auto n2 = normalizedEdgeList[edgesCount+j];
	//    write(j);
	//    write(" ");
	//    write(n1);
	//    write(" ");
	//    writeln(n2);
	//}

	//
	alias MatrixUndirectedGraph2!EmptyPayload GraphT;
	auto graph = createMatrixGraphNew!(GraphT)(normalizedEdgeList);
	//auto s = graph.toString;
	//writeln(s);

	//
	static struct ClientStructure
	{
		ConnectedComponentsNodeData connComp;
		//alias dfs this;
	}
	auto nodePayload = new ClientStructure[graph.nodesCount];
	auto nodePayloadFun = delegate ConnectedComponentsNodeData*(ref GraphT g, int v) { return &nodePayload[v].connComp;};

	int result = connectedComponentsCount!(GraphT,nodePayloadFun)(graph);
	return result;
}

// encodedTree = for example, [1, 86, -2 , 2, 87, -2, 21, 91, 22, 92, -3, 3, 88, -3]
// $(D collisionIgnoreNodeId) is not taken into account, when calculating incompatibility graph (this id represent eg. rootId)
extern (C) 
Int32PtrPair computeTrackIncopatibilityGraph(int* pEncodedTree, int encodedTreeLength, int collisionIgnoreNodeId, int openBracketLex, int closeBracketLex, int noObservationId, Int32Allocator allocator)
{
	int32_t* p1;
	debug(PRINTF) printf("size=%d\n", p1.sizeof);
	auto p1Size = p1.sizeof;

	assert(pEncodedTree != null);

	int[] encodedTree = pEncodedTree[0..encodedTreeLength];

	//

	alias NcaNodePayload NcaNodeDataT;
	struct HypothesisNode
	{
		int NodeId;
		int FrameInd;
		int ObservationInd;
		//NcaNodeDataT NcaData;
	}
	alias RootedUndirectedTree!HypothesisNode RootedTreeT;

	// initialize tree

	RootedTreeT tree;

	scope auto nodeDataFun = delegate HypothesisNode* (ref RootedTreeT tree, RootedTreeT.NodeId node) { 
		return &tree.nodePayload(node);
	};

	int[3] nodeDataEntriesBuff;
	auto nullNode = tree.root;
	scope auto initNodeDataFun = delegate void(RootedTreeT.NodeId node)
	{
		HypothesisNode* data = &tree.nodePayload(node);
		data.NodeId = nodeDataEntriesBuff[0];
		data.FrameInd = nodeDataEntriesBuff[1];
		data.ObservationInd = nodeDataEntriesBuff[2];
	};
	parseTree(tree, nullNode, encodedTree, openBracketLex, closeBracketLex, nodeDataEntriesBuff, initNodeDataFun);
	debug(PRINTF) mxArrayFuns.logDebug("parsed tree:\n".toStringz);
	//foreach(n; tree.nodes)
	//{
	//    HypothesisNode* data = nodeDataFun(tree, n);
	//    writeln("leaves NodeId=", data.NodeId, " ObsId=", data.ObservationId);
	//}

	//

	debug(PRINTF) 
	{
		auto nodeFormatterFun = delegate void(RootedTreeT.NodeId node, std.array.Appender!string buff)
		{
			auto data = tree.nodePayload(node);
			formattedWrite(buff, "%s %s", data.NodeId, data.ObservationId);
		};

		Appender!(string) buff;
		buff.put("");
		printTree(tree, tree.root, nodeFormatterFun, buff);
		mxArrayFuns.logDebug(buff.data.toStringz);
	}


	// preprocess Nearest Common Ancestor (NCA) algorithm

	// can we pass function instead of 
	//auto ncaNodeDataFun = delegate NcaNodeDataT* (ref TreeT tree, TreeT.NodeId node) { 
	//    return &tree.nodePayload(node);
	//};
	//
	//NearestCommonAncestorOfflineAlgorithm!(RootedTreeT,ncaNodeDataFun) ncaAlgo;
	//ncaAlgo.process(tree);

	// compuate track incompatibility graph
	// two tracks (corresponding to leafs of the hypothesis tree) are incompatible if the share same observation

	// gather leaves

	scope auto leaves = new RootedTreeT.NodeId[tree.nodesCount];
	int leafIndex = 0;
	foreach(n; tree.nodes)
	{
		if (!tree.hasChildren(n))
		{
			leaves[leafIndex] = n;
			leafIndex++;

			//HypothesisNode* data = nodeDataFun(tree, n);
			//writeln("node NodeId=", data.NodeId, " ObsId=", data.ObservationId);
		}
	}
	leaves.length = leafIndex;

	debug(PRINTF) mxArrayFuns.logDebug("checking tracks incompatibility\n");

	// check compatibility of each pair of tracks
	scope SList!(Tuple!(int,int)) edgeList;
	int edgesCount = 0;

	struct FrameAndObsInd
	{
		int FrameInd;
		int ObservationInd;
	}
	bool[FrameAndObsInd] observIdSet; // 

	for (int n1=0; n1 < leaves.length-1; n1++)
	{
		auto leaf1 = leaves[n1];
		auto leaf1Data = nodeDataFun(tree, leaf1);
		//writeln("leaf1ObservId=", leaf1Data.ObservationId);

		// populate set of observationIds for the first leaf
		// NOTE: two hypothesis can't conflict on 'no observation'

		observIdSet.clear;

		auto populateBranchObservationIds = delegate void(ref RootedTreeT tree, RootedTreeT.NodeId leaf1)
		{
			foreach(node; tree.branchReversed(leaf1))
			{
				HypothesisNode* data = nodeDataFun(tree, node);
				auto nodeId = data.NodeId;
				if (nodeId == collisionIgnoreNodeId)
					break;

				auto frameInd = data.FrameInd;
				auto obsInd = data.ObservationInd;
				//if (obsId == noObservationId)
				//    continue;
				FrameAndObsInd conflictObs;
				conflictObs.FrameInd = frameInd;
				conflictObs.ObservationInd = obsInd;
				observIdSet[conflictObs] = true;
			}
		};
		populateBranchObservationIds(tree, leaf1);

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
			foreach(internalNode2; tree.branchReversed(leaf2))
			{
				HypothesisNode* internalNode2Data = nodeDataFun(tree, internalNode2);
				if (internalNode2Data.NodeId == collisionIgnoreNodeId)
					break;

				auto frameInd2 = internalNode2Data.FrameInd;
				auto obsInd2 = internalNode2Data.ObservationInd;
				//writeln("node2 ObservId=", obsId2);

				//if (obsId2 == noObservationId)
				//    continue;

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
				auto leaf2Data = nodeDataFun(tree, leaf2);
				auto edge = tuple(leaf1Data.NodeId,  leaf2Data.NodeId);
				edgeList.insertFront(edge);
				edgesCount++;
				//writeln("edge ", edge[0], " ", edge[1]);
			}
		}
	}

	// populate result
	{
		Int32PtrPair result;

		auto len = edgesCount * 2; // (from,two) edge pairs
		if (len == 0)
		{
			result.pFirst = null;
			result.pLast = null;
		}
		else
		{
			int32_t* pNodeIdsArray = allocator.CreateArrayInt32(len, allocator.pUserData);
			scope(failure) allocator.DestroyArrayInt32(pNodeIdsArray, allocator.pUserData);

			int32_t[] nodeIds = pNodeIdsArray[0..len];

			int outInd = 0;
			foreach(edgeTuple; edgeList)
			{
				nodeIds[outInd++] = edgeTuple[0];
				nodeIds[outInd++] = edgeTuple[1];
			}
			assert(len == outInd, "error in data assignment");
		
			result.pFirst = pNodeIdsArray;
			result.pLast = pNodeIdsArray + len;
		}
		return  result;
	}
}

// Operates directly on the hypothesis tree.
extern (C) 
Int32PtrPair computeTrackIncopatibilityGraphDirectAccess(TrackHypothesisTreeNode* hypTree, int collisionIgnoreNodeId, Int32Allocator allocator)
{
	auto tree = HypTreeAdapter(hypTree);

	//
	Tuple!(int,int)[] edgesList;
	auto edgesListAppender = appender(&edgesList);
	findCollisionEdges(tree, edgesListAppender, collisionIgnoreNodeId);

	// populate result

	Int32PtrPair result;

	auto len = edgesList.length * 2; // (from,two) edge pairs
	if (len == 0)
	{
		result.pFirst = null;
		result.pLast = null;
	}
	else
	{
		int32_t* pNodeIdsArray = allocator.CreateArrayInt32(len, allocator.pUserData);
		scope(failure) allocator.DestroyArrayInt32(pNodeIdsArray, allocator.pUserData);

		int32_t[] nodeIds = pNodeIdsArray[0..len];

		int outInd = 0;
		foreach(edgeTuple; edgesList)
		{
			nodeIds[outInd++] = edgeTuple[0];
			nodeIds[outInd++] = edgeTuple[1];
		}
		assert(len == outInd, "error in data assignment");

		result.pFirst = pNodeIdsArray;
		result.pLast = pNodeIdsArray + len;
	}
	return result;
}

extern (C) 
void pwFindBestTracks(TrackHypothesisTreeNode* hypTree, int collisionIgnoreNodeId, int attemptCount, CppVectorPtrWrapper* bestTracks) 
{
	CppVector!(TrackHypothesisTreeNode*) bestTracksVector;
	findBestTracks(hypTree, collisionIgnoreNodeId, attemptCount, &bestTracksVector);

	// populate result
	foreach(TrackHypothesisTreeNode* hypNode; bestTracksVector)
		bestTracks.PushBack(bestTracks, hypNode);
}
