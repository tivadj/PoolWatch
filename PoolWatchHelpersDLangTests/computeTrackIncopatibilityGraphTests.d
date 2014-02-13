import std.stdio;
import std.string;
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

	auto mxArrayFuns = createMxArrayFuns;

	mxArrayPtr pIncompEdgeMat = computeTrackIncopatibilityGraph		
		(&encodedTree[0], cast(int)encodedTree.length, collisionIgnoreNodeId, 
		 openBracketLex, closeBracketLex, &mxArrayFuns);
	//scope(exit) pwFree(pIncompEdgeListColumnwise);

	size_t incompEdgeCount = mxArrayFuns.GetNumberOfElements(pIncompEdgeMat);
	debugFun(format("incompEdgeCount=%s", incompEdgeCount).toStringz);

	int* pNodeIds = cast(int*)mxArrayFuns.GetDataPtr(pIncompEdgeMat);
	for (auto i=0; i< incompEdgeCount; ++i)
	    write(" ", pNodeIds[i]);
	writeln;
}
