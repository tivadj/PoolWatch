#include "mex.h"
#include <vector>
#include <array>
#include "algos1.h"
#include "PoolWatchFacade.h"
using namespace std;

// cd E:\devb\DLang\lama\TestDLangVS2013ConsoleApp1\bin\Release
// mex ..\..\CppConsumer\PWMaxWeightInependentSetMaxFirst.cpp ..\..\CppConsumer\MatrixUndirectedGraph.cpp ..\..\CppConsumer\algos1.cpp

void MaxWeightInependentSetMaxFirstMexFunction(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
#if PRINTF
	mexPrintf("inside MaxWeightInependentSetMaxFirstMexFunction int=%d\n", sizeof(int));
#endif

	if (nrhs != 3)
	{
		mexErrMsgIdAndTxt("PoolWatch:error", "Provide 3 arguments: vertex array, edges-list per row and weights array");
		return;
	}

	if (nlhs > 1)
	{
		mexErrMsgIdAndTxt("PoolWatch:error", "Provide 1 output argument: vertices mask");
		return;
	}

	vector<const mxArray*> rhs(prhs, prhs + nrhs);

	// vertices
	const mxArray* verticesMat = rhs[0];
	auto verticesRows = mxGetM(verticesMat);
	auto verticesCount = mxGetN(verticesMat);
#if PRINTF
	mexPrintf("verticesCount=%d\n", verticesCount);
#endif	
	if (!(verticesRows == 1 && mxIsInt32(verticesMat)))
	{
		mexErrMsgIdAndTxt("PoolWatch:error", "Vertices must be int32[1xN]");
		return;
	}

	// edges list
	const mxArray* edgesListMat = rhs[1];
	auto edgesCount = mxGetM(edgesListMat);
	auto edgesListCols = mxGetN(edgesListMat);
#if PRINTF
	mexPrintf("edgesCount=%d\n", edgesCount);
#endif
	if (!(edgesListCols == 2 && mxIsInt32(edgesListMat)))
	{
		mexErrMsgIdAndTxt("PoolWatch:error", "Edges matrix must be int32[Mx2], each edge per row");
		return;
	}

	// weights
	const mxArray* weightsMat = rhs[2];
	auto weightsRows = mxGetM(weightsMat);
	auto weightsCols = mxGetN(weightsMat);

	if (!(weightsRows == 1 && weightsCols == verticesCount && mxIsDouble(weightsMat)))
	{
		mexErrMsgIdAndTxt("PoolWatch:error", "Weights must be double[1xN]");
		return;
	}

	//
	auto verticesDataPtr = (int*)mxGetPr(verticesMat);
	vector<int> vertices(verticesDataPtr, verticesDataPtr + verticesCount);

	auto edgesListDataPtr = (int*)mxGetPr(edgesListMat);
	vector<int> edgeList(edgesCount * 2);
	for (int i = 0; i < edgesCount; ++i)
	{
		edgeList[i * 2] = edgesListDataPtr[i];
		edgeList[i * 2 + 1] = edgesListDataPtr[edgesCount + i];
	}

	auto weightsDataPtr = mxGetPr(weightsMat);
	vector<double> weightsList(weightsDataPtr, weightsDataPtr + verticesCount);

	//
	//for (int i = 0; i < vertices.size(); ++i)
	//	mexPrintf("vertexId=%d\n", vertices[i]);
	//for (int i = 0; i < edgeList.size(); ++i)
	//	mexPrintf("edgeId=%d\n", edgeList[i]);
	//for (int i = 0; i < weightsList.size(); ++i)
	//	mexPrintf("weight[%d]=%f\n", i, weightsList[i]);

	//
	auto gMap = createFromEdgeList(vertices, edgeList);
	auto g = get<0>(gMap);

	for (int i = 0; i < weightsList.size(); ++i)
		g.setVertexPayload(i, weightsList[i]);

	//mexPrintf("V=%d\n", g.verticesCount());
	//for (int i = 0; i < verticesCount; ++i)
	//	mexPrintf("pay[%d]=%f\n", i, g.vertexPayload(i));
	//vector<int> n;
	//for (int i = 0; i < verticesCount; ++i)
	//{
	//	g.getAdjacentVertices(i, n);
	//	for (int j = 0; j < n.size(); ++j)
	//		mexPrintf("adj[%d]=%d\n", i, n[j]);
	//}

	vector<bool> indepVertexSet;
	maximumWeightIndependentSetNaiveMaxFirst(g, indepVertexSet);

	//auto weight = calculateVertexSetWeight(g, indepVertexSet);
	//mexPrintf("weight=%f\n", weight);


	// populate output

	if (nlhs == 1)
	{
		array<mwSize, 2> outLen = { 1, verticesCount };
		mxArray* outMask = mxCreateLogicalArray((mwSize)2, &outLen[0]);
		bool* data = (bool*)mxGetPr(outMask);
		for (int i = 0; i < verticesCount; ++i)
			data[i] = indepVertexSet[i];

		plhs[0] = outMask;
	}
}
