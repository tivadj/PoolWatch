module PoolWatchHelpersDLang.GraphHelpers;

import std.algorithm; // minPos

//////////////////////////////////////////////////////

struct EmptyPayload
{
}

MatrixUndirectedGraph!NodePayloadT createMatrixUndirectedGraph(NodePayloadT)(int[] edgePerRow)
{
	assert(edgePerRow.length >= 0);
	assert(edgePerRow.length % 2 == 0);

	// find vertices count
	auto maxInd = minPos!("a>b")(edgePerRow);
	auto vertCount = maxInd[0] + 1;

	auto result = MatrixUndirectedGraph!double(vertCount);

	for (int i=0; i<edgePerRow.length / 2; i++)
	{
		auto from = edgePerRow[i * 2 + 0];
		auto to   = edgePerRow[i * 2 + 1];
		result.setEdge(from, to);
	}
	return result;
}

GraphT createMatrixGraphNew(GraphT)(int[] edgePerRow)
{
	assert(edgePerRow.length >= 0);
	assert(edgePerRow.length % 2 == 0);

	// find vertices count
	auto maxInd = minPos!("a>b")(edgePerRow);
	auto vertCount = maxInd[0] + 1;

	auto result = GraphT(vertCount);

	auto edgesCount = edgePerRow.length / 2;

	for (int i=0; i<edgesCount; i++)
	{
		auto from = edgePerRow[i];
		auto to   = edgePerRow[edgesCount + i];
		result.setEdge(from, to);
	}
	return result;
}

