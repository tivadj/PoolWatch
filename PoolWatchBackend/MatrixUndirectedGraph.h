#pragma once
#include <vector>
#include "PoolWatchFacade.h"

class PW_EXPORTS MatrixUndirectedGraph
{
	std::vector<bool> adjacencyMatrixByRow;
	std::vector<double> vertexPayload_;
	int vertexCount;
	int capacity;

public:
	MatrixUndirectedGraph(int vertexCount, int capacity);
	~MatrixUndirectedGraph();

	void setVertexPayload(int vertexId, double payload);
	double nodePayload(int vertexId) const;
	int nodesCount() const;
	void adjacentNodes(int vertexId, std::vector<int>& neighbourVertices) const;

	void setEdge(int vertexId, int otherVertexId);
	bool getEdge(int fromIndex, int toIndex) const;

private:
	int unaryIndex(int fromIndex, int toIndex) const;
	void validateVertexId(int vertexId) const;
};

