module PoolWatchHelpersDLang.MatrixUndirectedGraph2;
import std.stdio;
import std.format;
import std.exception;
import PoolWatchHelpersDLang.MatrixUndirectedGraph;

struct MatrixUndirectedGraph2(EdgePayloadType)
{
	alias int NodeId;
	enum NullNode = -1;
	alias typeof(this) GraphType;

	private struct EdgeCell
	{
		bool HasEdge;
		EdgePayloadType edgePayload;
	}

	EdgeCell[] adjacencyMatrixByRow;
	int vertexCount;
	int capacity;

	this(int vertexCount)
	{
		this(vertexCount,vertexCount);
	}
	~this()
	{
		//writeln("MatrixUndirectedGraph2 dtr");
	}

	this(int vertexCount, int capacity)
	{
		enforce(vertexCount >= 0, "vertexCount must be positive or zero");
		enforce(vertexCount <= capacity);

		this.vertexCount = vertexCount;
		this.capacity = capacity;
		this.adjacencyMatrixByRow = new EdgeCell[capacity*capacity];
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

	int getVerticesCount() // TODO: remove
	{
		return vertexCount;
	}
	
	int nodesCount()
	{
		return vertexCount;
	}

	AdjacentVerticesRange adjacentNodes(NodeId vertex)
	{
		validateVertexId(vertex);
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
				if (outer_.adjacencyMatrixByRow[first+neigh].HasEdge) {
					int result = dg(neigh);
					if (result) return result;
				}
			}

			return 0;
		}
	}

	private void validateVertexId(int vertexId)
	{
		enforce(vertexId >= 0);
		enforce(vertexId < vertexCount);
	}

	private int unaryIndex(int fromIndex, int toIndex)
	{
		return unaryIndexByRow(fromIndex, toIndex, capacity);
	}

	void setEdge(int vertexId, int otherVertexId)
	{
		validateVertexId(vertexId);
		validateVertexId(otherVertexId);

		// both (from,to) and (to,from) edge flags must be set to correctly return adjacent vertices for both vertices
		adjacencyMatrixByRow[unaryIndex(vertexId, otherVertexId)].HasEdge = true;
		adjacencyMatrixByRow[unaryIndex(otherVertexId, vertexId)].HasEdge = true;
	}

	int getEdgesCount()
	{
		// process upper part of the matrix

		int result = 0;
		for (int row=0; row < vertexCount; row++)
		{
			for (int col=row; col < vertexCount; col++)
			{
				if (adjacencyMatrixByRow[unaryIndex(row,col)].HasEdge)
					result++;
			}
		}
		return result;
	}

	ref EdgePayloadType edgePayload(NodeId vertex, NodeId otherVertex)
	{
		validateVertexId(vertex);
		validateVertexId(otherVertex);
		return adjacencyMatrixByRow[unaryIndex(vertex, otherVertex)].edgePayload;
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
				if (!adjacencyMatrixByRow[unaryIndex(row,col)].HasEdge)
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

		return str.data;
	}
}
