module PoolWatchHelpersDLang.UndirectedAdjacencyVectorGraph;
import std.container;

struct UndirectedAdjacencyVectorGraphFacade
{
	//alias UndirectedAdjacencyVectorNode* NodeId;

}


struct UndirectedAdjacencyGraphVectorImpl(NodePayloadT)
{
	alias UndirectedAdjacencyNode* NodeId;
	SList!NodeId allNodes;
	int nodesCount_ = 0;

	struct UndirectedAdjacencyNode
	{
		SList!NodeId AdjacentNodes;
		NodePayloadT NodePayload;
	}

	NodeId addNode()
	{
		auto node = new UndirectedAdjacencyNode;
		allNodes.insertFront(node);
		nodesCount_++;
		return node;
	}

	void addEdge(NodeId n1, NodeId n2)
	{
		n1.AdjacentNodes.insertFront(n2);
		n2.AdjacentNodes.insertFront(n1);
	}

	struct NodesRange
	{
		bool empty()
		{
			return true;
		}
	}

	struct NodesApplyImpl
	{
		SList!NodeId nodes;

		int opApply(int delegate(NodeId node) dg)
		{
			foreach(NodeId n; this.nodes)
			{
				auto result = dg(n);
				if (result) return result;
			}
			return 0;
		}

		int opApply(int delegate(int index, NodeId node) dg)
		{
			int i = 0;
			foreach(NodeId n; this.nodes)
			{
				auto result = dg(i, n);
				if (result) return result;
				i++;
			}
			return 0;
		}

		auto opSlice()
		{
			return this.nodes.Range();
		}
	}

	auto nodes()
	{
		return NodesApplyImpl(this.allNodes);
	}

	int nodesCount()
	{
		return nodesCount_;
	}

	struct AdjacentVerticesRange
	{
		NodeId node_;
		this(NodeId node)
		{
			this.node_ = node;
		}
		int opApply(int delegate(NodeId) dg)
		{
			auto adjRange = node_.AdjacentNodes[];
			foreach(adj; adjRange)
			{
				int result = dg(adj);
				if (result) return result;
			}
			return 0;
		}
	}

	auto adjacentNodes(NodeId node)
	{
		return AdjacentVerticesRange(node);
	}
}