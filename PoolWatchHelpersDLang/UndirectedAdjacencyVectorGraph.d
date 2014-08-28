module PoolWatchHelpersDLang.UndirectedAdjacencyVectorGraph;
import std.container;
import PoolWatchHelpersDLang.GraphHelpers;

// Represents a graph where nodes are associated with other nodes, bypassing edges.
// The list of adjacent nodes is powered by 'CppVector'.
struct UndirectedAdjacencyGraphVectorImpl(NodePayloadT)
{
	private struct UndirectedAdjacencyNode
	{
		CppVector!NodeId AdjacentNodes;
		NodePayloadT NodePayload;
	}

	alias UndirectedAdjacencyNode* NodeId;

	private CppVector!NodeId allNodes;

	~this() @nogc
	{
		for (int i=0; i < allNodes.length; ++i)
		{
			auto pNode = allNodes[i];
			assert(pNode != null);
			core.stdc.stdlib.free(pNode);
		}
	}

	NodeId addNode() @nogc
	{
		auto pNode = cast(UndirectedAdjacencyNode*)core.stdc.stdlib.malloc(UndirectedAdjacencyNode.sizeof);
		pNode = std.conv.emplace!(UndirectedAdjacencyNode)(pNode);

		allNodes.pushBack(pNode);
		return pNode;
	}

	void reserveNodes(int count) @nogc
	{
		allNodes.reserve(count);
	}

	void addEdge(NodeId n1, NodeId n2) @nogc
	{
		n1.AdjacentNodes.pushBack(n2);
		n2.AdjacentNodes.pushBack(n1);
	}

	private static struct NodesApplyImpl
	{
		CppVector!(NodeId)* pNodes;

		int opApply(int delegate(NodeId node) @nogc dg)
		{
			foreach(NodeId n; *pNodes)
			{
				auto result = dg(n);
				if (result) return result;
			}
			return 0;
		}

		int opApply(int delegate(int index, NodeId node) @nogc dg) @nogc
		{
			int i = 0;
			foreach(NodeId n; *pNodes)
			{
				auto result = dg(i, n);
				if (result) return result;
				i++;
			}
			return 0;
		}
	}

	auto nodes() @nogc
	{
		return NodesApplyImpl(&this.allNodes);
	}

	int nodesCount() @nogc
	{
		return allNodes.length;
	}

	private static struct AdjacentVerticesRange
	{
		NodeId node_;

		int opApply(int delegate(NodeId) @nogc dg)
		{
			foreach(NodeId adj; node_.AdjacentNodes)
			{
				int result = dg(adj);
				if (result) return result;
			}
			return 0;
		}
	}

	auto adjacentNodes(NodeId node) @nogc
	{
		return AdjacentVerticesRange(node);
	}
}