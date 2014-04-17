import std.stdio;
import std.string;
import std.stdint;
import std.algorithm;
import PoolWatchInteropDLangTests;
import PoolWatchHelpersDLang.GraphHelpers;
import PoolWatchHelpersDLang.PoolWatchInteropDLang;

void run()
{
	//auto encodedTree = [1, 86, -1 , 2, 87, -1, 21, 91, 22, 92, -2, 3, 88, -2, -2];
	auto encodedTree = [
	0, 0, -1 ,
		86, 6, -1,
			88, 2, -1, 91, 4, -2,
		-2,
		87, 1, -1,
			89, 2, -1, 92, 4, -2,
			90, 3, -1, 
				93, 4, 94, 5,
			-2,
		-2, 		
	-2];

	const int collisionIgnoreNodeId = 0;
	const int openBracketLex = -1;
	const int closeBracketLex = -2;

	Int32Allocator int32Alloc = createInt32Allocator();

	int32_t* p1;
	printf("size=%d\n", p1.sizeof);

	Int32PtrPair nodeIdsRange = computeTrackIncopatibilityGraph		
		(&encodedTree[0], cast(int)encodedTree.length, collisionIgnoreNodeId, 
		 openBracketLex, closeBracketLex, int32Alloc);
	scope(exit) int32Alloc.DestroyArrayInt32(nodeIdsRange.pFirst, int32Alloc.pUserData);

	size_t idsCount = nodeIdsRange.pLast - nodeIdsRange.pFirst;
	size_t edgesCount = idsCount / 2;
	debugFun(format("idsCount=%d\n", idsCount).toStringz);

	assert(edgesCount == 5, "Must be 5 edges");

	auto nodeIds = nodeIdsRange.pFirst[0..idsCount];
	for (auto i=0; i< idsCount; ++i)
	    write(" ", nodeIds[i]);
	writeln;

	// normalize edges from less id to greater id

	int32_t[2][] edgesIds = new int32_t[2][edgesCount];
	for (auto edgeId=0; edgeId < edgesCount; ++edgeId)
	{
		auto a1 = nodeIds[edgeId*2+0];
		auto a2 = nodeIds[edgeId*2+1];
		edgesIds[edgeId][0] = min(a1, a2);
		edgesIds[edgeId][1] = max(a1, a2);
	}

	bool lexFirst(int32_t[2] edgeX, int32_t[2] edgeY) { return edgeX[0] < edgeY[0]; }
	sort!(lexFirst)(edgesIds);

	assert(edgesIds == [[91, 92], [91, 93], [92, 93], [92, 94], [93, 94]]);
}
