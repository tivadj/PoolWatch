#include "MatrixUndirectedGraph.h"
#include <cassert>
#include <algorithm> 
using namespace std;

MatrixUndirectedGraph::MatrixUndirectedGraph(int vertexCount, int capacity)
{
	assert(vertexCount >= 0);
	assert(vertexCount <= capacity);

	this->vertexCount = vertexCount;
	this->capacity = capacity;
	this->adjacencyMatrixByRow.resize(capacity*capacity);
	this->vertexPayload_.resize(capacity);
}

int MatrixUndirectedGraph::unaryIndex(int fromIndex, int toIndex) const
{
	auto elementIndex = fromIndex * capacity + toIndex;
	assert(elementIndex < capacity * capacity);
	return elementIndex;
}

void MatrixUndirectedGraph::validateVertexId(int vertexId) const
{
	assert(vertexId >= 0);
	assert(vertexId < vertexCount);
}

void MatrixUndirectedGraph::setVertexPayload(int vertexId, double payload)
{
	validateVertexId(vertexId);

	vertexPayload_[vertexId] = payload;
}

double MatrixUndirectedGraph::nodePayload(int vertexId) const
{
	validateVertexId(vertexId);

	return vertexPayload_[vertexId];
}

bool MatrixUndirectedGraph::getEdge(int fromIndex, int toIndex) const
{
	return adjacencyMatrixByRow[unaryIndex(fromIndex, toIndex)];
}

void MatrixUndirectedGraph::setEdge(int vertexId, int otherVertexId)
{
	validateVertexId(vertexId);
	validateVertexId(otherVertexId);

	adjacencyMatrixByRow[unaryIndex(vertexId, otherVertexId)] = true;
	adjacencyMatrixByRow[unaryIndex(otherVertexId, vertexId)] = true;

	assert(getEdge(vertexId, otherVertexId));
	assert(getEdge(otherVertexId, vertexId));
}

MatrixUndirectedGraph::~MatrixUndirectedGraph()
{
}

int MatrixUndirectedGraph::nodesCount() const
{
	return vertexCount;
}

void MatrixUndirectedGraph::adjacentNodes(int vertexId, vector<int>& neighbourVertices) const
{
	validateVertexId(vertexId);

	neighbourVertices.reserve(capacity);
	neighbourVertices.clear();

	for (int col = 0; col < vertexCount; col++)
	{
		if (adjacencyMatrixByRow[unaryIndex(vertexId, col)])
			neighbourVertices.push_back(col);
	}
}
