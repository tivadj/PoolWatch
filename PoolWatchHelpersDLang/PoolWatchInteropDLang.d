module PoolWatchHelpersDLang.PoolWatchInteropDLang;

import std.stdio;
import std.algorithm;
import std.format;
import std.string;
import std.array;
import std.typecons; // tuple
//import std.copy;
import std.container; // SList

import PoolWatchHelpersDLang.traversal;
import PoolWatchHelpersDLang.connect;
import PoolWatchHelpersDLang.GraphHelpers;
import PoolWatchHelpersDLang.MatrixUndirectedGraph;
import PoolWatchHelpersDLang.MatrixUndirectedGraph2;
import PoolWatchHelpersDLang.NearestCommonAncestorOfflineAlgorithm;
import PoolWatchHelpersDLang.RootedUndirectedTree;

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

// encodedTree = for example, [1, 86, -1 , 2, 87, -1, 21, 91, 22, 92, -2, 3, 88, -2, -2]
// $(D collisionIgnoreNodeId) is not taken into account, when calculating incompatibility graph (this id represent eg. rootId)
extern (C) 
mxArrayPtr computeTrackIncopatibilityGraph(int* pEncodedTree, int encodedTreeLength, int collisionIgnoreNodeId, int openBracketLex, int closeBracketLex, mxArrayFuns_tag* mxArrayFuns)
{
	assert(pEncodedTree != null);

	int[] encodedTree = pEncodedTree[0..encodedTreeLength];

	//

	alias NcaNodePayload!(RootedUndirectedTreeFacade.NodeId) NcaNodeDataT;
	struct HypothesisNode
	{
		int NodeId;
		int ObservationId;
		//NcaNodeDataT NcaData;
	}
	alias RootedUndirectedTree!HypothesisNode RootedTreeT;

	// initialize tree

	RootedTreeT tree;

	auto nodeDataFun = delegate HypothesisNode* (ref RootedTreeT tree, RootedTreeT.NodeId node) { 
		return &tree.nodePayload(node);
	};

	int[2] nodeDataEntriesBuff;
	auto nullNode = tree.root;
	auto initNodeDataFun = delegate void(RootedTreeT.NodeId node)
	{
		HypothesisNode* data = &tree.nodePayload(node);
		data.NodeId = nodeDataEntriesBuff[0];
		data.ObservationId = nodeDataEntriesBuff[1];
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
	bool[int] observIdSet; // 

	for (int n1=0; n1 < leaves.length-1; n1++)
	{
		auto leaf1 = leaves[n1];
		auto leaf1Data = nodeDataFun(tree, leaf1);
		//writeln("leaf1ObservId=", leaf1Data.ObservationId);

		// populate set of observationIds for the first leaf

		observIdSet.clear;

		auto populateBranchObservationIds = delegate void(ref RootedTreeT tree, RootedTreeT.NodeId leaf1)
		{
			foreach(node; tree.branchReversed(leaf1))
			{
				HypothesisNode* data = nodeDataFun(tree, node);
				auto nodeId = data.NodeId;
				if (nodeId == collisionIgnoreNodeId)
					break;

				auto obsId = data.ObservationId;
				observIdSet[obsId] = true;
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

				auto obsId2 = internalNode2Data.ObservationId;
				//writeln("node2 ObservId=", obsId2);

				auto collide = observIdSet.get(obsId2, false);
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
		auto len = edgesCount * 2; // (from,two) edge pairs
		mxArrayPtr nodeIdsArray = mxArrayFuns.CreateArrayInt32(len);
		scope(failure) mxArrayFuns.DestroyArray(nodeIdsArray);

		auto pNodeIds = cast(int*)mxArrayFuns.GetDataPtr(nodeIdsArray);
		int outInd = 0;
		foreach(edgeTuple; edgeList)
		{
			pNodeIds[outInd++] = edgeTuple[0];
			pNodeIds[outInd++] = edgeTuple[1];
		}
		assert(len == outInd, "error in data assignment");

		return nodeIdsArray;
	}
}
