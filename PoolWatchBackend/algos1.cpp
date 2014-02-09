#include <algorithm>
#include <cassert>
#include <functional> // std::reference_wrapper
#include <hash_map>
#include <set>
#include <tuple>
#include "algos1.h"

using namespace std;


void maximumWeightIndependentSetNaiveMaxFirst(const MatrixUndirectedGraph& graph, vector<bool>& vertexSet)
{
	auto vertexCount = graph.nodesCount();

	// initialize bookkeeping data

	struct VertexData
	{
		int vertexId;
		double weight;
		bool processed;
	};
	
	vector<VertexData> vertexIdAndWeight(vertexCount);
	for (int vertexId = 0; vertexId < vertexCount; vertexId++)
	{
		VertexData data;
		data.vertexId = vertexId;
		data.weight = graph.nodePayload(vertexId);
		data.processed = false;
		vertexIdAndWeight[vertexId] = data;
	}

	// arrange weights large to small
	sort(begin(vertexIdAndWeight), end(vertexIdAndWeight), [](VertexData& x, VertexData& y) { return x.weight > y.weight;  });

	// TODO: the value=VertexData& doesn't work; when adding new pair, all values become equal to this pair
	hash_map<int, reference_wrapper<VertexData>> vertexIdToData;
	for (VertexData& vertexData : vertexIdAndWeight)
	{
		vertexIdToData.insert(make_pair(vertexData.vertexId, std::ref(vertexData)));
	}

	vertexSet.resize(vertexCount);
	vertexSet.assign(vertexCount, false);
	
	// sequentially include vertices from largest weight to smallest

	vector<int> neighbourVertices;
	for (int dataIndex = 0; dataIndex < vertexCount; dataIndex++)
	{
		auto& vertexData = vertexIdAndWeight[dataIndex];
		if (vertexData.processed)
			continue;

		vertexData.processed = true;

		// include vertex into max weight independent set
		auto vertexId = vertexData.vertexId;
		vertexSet[vertexId] = true;

		// reject all neighbour vertices
		graph.adjacentNodes(vertexId, neighbourVertices);

		for (auto adjVertId : neighbourVertices)
		{
			auto vertexDataIt = vertexIdToData.find(adjVertId);
			if (vertexDataIt != end(vertexIdToData))
				vertexDataIt->second.get().processed = true;
		}			
	}

	auto validation = validateMaximumWeightIndependentSet(graph, vertexSet);
	assert(validation.isValid);
}

IndependentSetValidationResult validateMaximumWeightIndependentSet(const MatrixUndirectedGraph& graph, const vector<bool>& vertexSet)
{
	IndependentSetValidationResult result;
	result.isValid = false;
	result.vertex1 = -1;
	result.vertex1 = -1;

	if (graph.nodesCount() != vertexSet.size())
	{
		result.message = "graph.VertexCount != independentSet.size";
		return result;
	}
	
	vector<int> neighbours;
	for (int vertexId = 0; vertexId < graph.nodesCount(); ++vertexId)
	{
		auto vertexInSet = vertexSet[vertexId];
		if (!vertexInSet)
			continue;

		// check all adjacent vertices are not in independent set
		
		graph.adjacentNodes(vertexId, neighbours);
		
		auto neighIt = find_if(begin(neighbours), end(neighbours), [&vertexSet](int vertexId) { return vertexSet[vertexId]; });
		if (neighIt != end(neighbours))
		{
			result.vertex1 = vertexId;
			result.vertex2 = *neighIt;
			result.message = "Adjacent vertices are both in independent set";
			return result;
		}
	}

	result.isValid = true;
	return result;
}

double calculateVertexSetWeight(const MatrixUndirectedGraph& graph, const std::vector<bool>& vertexSet)
{
	double result = 0;

	for (int vertexId = 0; vertexId < graph.nodesCount(); ++vertexId)
	{
		auto vertexInSet = vertexSet[vertexId];
		if (!vertexInSet)
			continue;

		result += graph.nodePayload(vertexId);
	}
	return result;
}

// result(1) = vertexIndex to original vertex id
tuple<MatrixUndirectedGraph, vector<int>> createFromEdgeList(const vector<int>& vertices, const vector<int>& edgeListByRow)
{
	// TODO: validate edge-list graph
	assert(edgeListByRow.size() % 2 == 0);

	int vertexCount = (int)vertices.size();

	vector<int> vertexIndexToOriginal(begin(vertices), end(vertices));
	sort(begin(vertexIndexToOriginal), end(vertexIndexToOriginal));

	hash_map<int, int> vertexOriginalToIndex;
	for (int vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex)
	{
		auto original = vertexIndexToOriginal[vertexIndex];
		vertexOriginalToIndex[original] = vertexIndex;
	}

	MatrixUndirectedGraph result(vertexCount, vertexCount);

	for (int i = 0; i < edgeListByRow.size() / 2; i++)
	{
		auto origFrom = edgeListByRow[i * 2 + 0];
		auto origTo   = edgeListByRow[i * 2 + 1];

		auto from = vertexOriginalToIndex[origFrom];
		auto to   = vertexOriginalToIndex[origTo];
		result.setEdge(from, to);
	}

	return make_tuple(result, vertexIndexToOriginal);
}