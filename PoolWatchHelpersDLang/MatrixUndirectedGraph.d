module PoolWatchHelpersDLang.MatrixUndirectedGraph;
import std.stdio;
import std.algorithm;
import std.format;

// no multi edges
// no loops
// Flag, whether edge between two vertices exist is stored in the upper right triangle of the adjacency matrix.
struct MatrixUndirectedGraph(VertexPayloadType)
{
	alias int NodeId;
	enum NullNode = -1;
	alias VertexPayloadType NodePayloadT;
	alias typeof(this) GraphType;

	bool[] adjacencyMatrixByRow;
	NodePayloadT[] nodePayload;
	int vertexCount;
	int capacity;
	//alias _payload this;

	this(int vertexCount)
	{
		this(vertexCount,vertexCount);
	}

	this(int vertexCount, int capacity)
	{
		assert(vertexCount >= 0);
		assert(vertexCount <= capacity);

		this.vertexCount = vertexCount;
		this.capacity = capacity;
		this.adjacencyMatrixByRow = new bool[capacity*capacity];
		this.nodePayload = new NodePayloadT[capacity];
	}

	void Fun1()
	{
		writeln("fun1");
	}
	
	VerticesRange nodes()
	{
		return VerticesRange(0, vertexCount);
	}

	struct VerticesRange
	{
		int start_;
		int endExcl_;

		this(int start, int endExcl)
		{
			assert(start >= 0);
			assert(start < endExcl);

			start_ = start;
			endExcl_ = endExcl;
		}

		@property bool empty() const
		{ 
			return start_ == endExcl_; 
		}

        @property int front()
        {
            assert(!empty);
            return start_;
        }
        void popFront()
        {
            assert(!empty);
            start_ = start_ + 1;
        }
	}

	int getVerticesCount()
	{
		return vertexCount;
	}

	AdjacentVerticesRange adjacentNodes(NodeId vertex)
	{
		return AdjacentVerticesRange(&this, vertex);
	}

	struct AdjacentVerticesRange
	{
		NodeId vertex_;
		GraphType* outer_;
		this(GraphType* outer, NodeId vertex)
		{ 
			outer_ = outer;
			vertex_ = vertex; 
		}
		int opApply(int delegate(NodeId) dg)
		{
			int first = outer_.unaryIndex(vertex_, 0);

			for (NodeId neigh = 0; neigh < outer_.vertexCount; ++neigh)
			{
				if (outer_.adjacencyMatrixByRow[first+neigh]) {
					int result = dg(neigh);
					if (result) return result;
				}
			}

			return 0;
		}
	}

	void setVertex(int vertexId, NodePayloadT payload)
	{
		validateVertexId(vertexId);
		nodePayload[vertexId] = payload;
	}

	ref NodePayloadT vertexPayloadNew(int vertexId)
	{
		validateVertexId(vertexId);

		return nodePayload[vertexId];
	}

	private void validateVertexId(int vertexId)
	{
		assert(vertexId >= 0);
		assert(vertexId < vertexCount);
	}

	private int unaryIndex(int fromIndex, int toIndex)
	{
		return unaryIndexByRow(fromIndex, toIndex, capacity);
	}

	// doesn't work
	private ref bool getEdge(int fromIndex, int toIndex)
	{
		return adjacencyMatrixByRow[unaryIndex(fromIndex, toIndex)];
	}

	void setEdge(int vertexId, int otherVertexId)
	{
		validateVertexId(vertexId);
		validateVertexId(otherVertexId);

		// both (from,to) and (to,from) edge flags must be set to correctly return adjacent vertices for both vertices
		adjacencyMatrixByRow[unaryIndex(vertexId, otherVertexId)] = true;
		adjacencyMatrixByRow[unaryIndex(otherVertexId, vertexId)] = true;

		auto edge2 = getEdge(vertexId, otherVertexId);
		assert(edge2);
		auto edge3 = getEdge(otherVertexId, vertexId);
		assert(edge2);
	}

	int getEdgesCount()
	{
		// process upper part of the matrix

		int result = 0;
		for (int row=0; row < vertexCount; row++)
		{
			for (int col=row; col < vertexCount; col++)
			{
				if (adjacencyMatrixByRow[unaryIndex(row,col)])
					result++;
			}
		}
		return result;
	}

	string toString()
	{
		auto str = std.array.Appender!string();
		str.put("");
		formattedWrite(str, "V%s E%s", getVerticesCount, getEdgesCount);

		// format matrix

		formattedWrite(str, " [");
		for (int row=0; row < vertexCount; row++)
		{
			bool startedWritingRow = false;
			for (int col=0; col < vertexCount; col++)
			{
				if (!adjacencyMatrixByRow[unaryIndex(row,col)])
					continue;

				// lazy writing of 'from' vertex
				if (!startedWritingRow)
				{
					formattedWrite(str, "v%s - ", row);
					startedWritingRow = true;
				}
				formattedWrite(str, "%s ", col);
			}
		}
		formattedWrite(str, "]");

		// print vertex payload
		formattedWrite(str, " (");
		for (int row=0; row < vertexCount; row++)
			formattedWrite(str, "v%s %s ", row, nodePayload[row]);
		formattedWrite(str, ")");

		return str.data;
	}
}

int unaryIndexByRow(int fromIndex, int toIndex, int theCapacity)
{
	auto elementIndex = fromIndex * theCapacity + toIndex;
	assert(elementIndex < theCapacity * theCapacity);
	return elementIndex;
}

