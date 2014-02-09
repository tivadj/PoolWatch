#pragma  once
#include <vector>
#include <tuple>
#include "MatrixUndirectedGraph.h"

void maximumWeightIndependentSetNaiveMaxFirst(const MatrixUndirectedGraph& graph, std::vector<bool>& vertexSet);

struct IndependentSetValidationResult
{
	bool isValid;
	std::string message;
	int vertex1;
	int vertex2;
};

IndependentSetValidationResult validateMaximumWeightIndependentSet(const MatrixUndirectedGraph& graph, const std::vector<bool>& vertexSet);

double calculateVertexSetWeight(const MatrixUndirectedGraph& graph, const std::vector<bool>& vertexSet);

std::tuple<MatrixUndirectedGraph, std::vector<int>> createFromEdgeList(const std::vector<int>& vertices, const std::vector<int>& edgeListByRow);