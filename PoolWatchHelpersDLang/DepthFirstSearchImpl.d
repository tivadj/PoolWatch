module PoolWatchHelpersDLang.DepthFirstSearchImpl;
import std.container;
import std.stdio;

struct DepthFirstSearchNodeData
{
	bool Visited;
}

/// Traverses a single connected component in a Depth First Search (DFS) order starting
/// from given node.
/// Returns a range of nodes in DFS order.
/// Prerequirement: DFS data must be initialized
// it works on getAnyNode + adjacentNodes(n) methods.
auto depthFirstSearch(GraphT, alias nodePayloadFun)(ref GraphT graph, GraphT.NodeId node)
	if ( is(typeof(nodePayloadFun(graph, node)) == DepthFirstSearchNodeData*) )
{
	struct DfsSingleComponent
	{
		SList!(GraphT.NodeId) nodesToProcess;
		int opApply(int delegate(GraphT.NodeId) dg)
		{
			nodesToProcess.insertFront(node);

			// traverse connected component

			//writeln("starting loop");
			while (true)
			{
				//writeln("checking if queue is empty");
				//writeln("outer_=",outer_);
				//writeln("&outer_.verticesToProcess=",&outer_.verticesToProcess);
				if (nodesToProcess.empty)
				    break;

				//writeln("take next element");
				auto current = nodesToProcess.front;

				//writeln("next=", current);
				nodesToProcess.removeFront;

				//writeln("query payload");
				//auto payload = outer_.nodePayloadFun(*outer_.graph, current);
				auto dfsData = nodePayloadFun(graph, current);
				if (dfsData.Visited)
				    continue;

				//writeln("assigning visited=true");
				dfsData.Visited = true;

				int result = dg(current);
				if (result) return result;

				auto adjs = graph.adjacentNodes(current);
				foreach(GraphT.NodeId neigh; adjs)
				{
				    //writeln("neigh=",neigh);
					//auto neighPayload = outer_.nodePayloadFun(*outer_.graph, neigh);
					auto neighDfsData = nodePayloadFun(graph, neigh);
					if (!neighDfsData.Visited)
						nodesToProcess.insertFront(neigh);
				}
				//writeln("exit loop for current=", current);
			}
			return 0;
		}
	}
	return DfsSingleComponent();
}

// Traverses the graph in Depth First Search (DFS) fashion. Graph can contain multiple connected components.
// The 'seedVertex' may be used to find a vertex to initiate the traversal.
//struct DepthFirstSearchAlgorithm(GraphT, nodePayloadFun)
// TODO: obsolete
struct DepthFirstSearchAlgorithm(GraphT, alias nodePayloadFun)
{
	alias typeof(this) DepthFirstSearchT;

	GraphT* graph;
	SList!(GraphT.NodeId) verticesToProcess;
	bool initialized_ = false;
	//nodePayloadFun nodePayloadFun;

	//this() // call to init() is required
	//{
	//}
	this(ref GraphT graph)
	{
		// NOTE: doesn't work if GraphT.init is used instead of 'graph'
		static assert(is(typeof(nodePayloadFun(graph, GraphT.NodeId.init)) == DepthFirstSearchNodeData*));

		//string k1=GraphT.init.stringof;
		//string k2=GraphT.NodeId.init.stringof;
		//writeln("inside DFS ctr, &graph=", &graph);
		init(graph);
		//this.nodePayloadFun = nodePayloadFun;
	}
	~this()
	{
		//writeln("DfsAlgo dtr");
	}

	void init(ref GraphT graph)
	{
		this.graph = &graph;
	}

	private void ensureInitialized()
	{
		if (!initialized_)
		{
			// TODO: extract initialization phase out of DFS on a client
			foreach (nodeRef; graph.nodes)
			{
				auto payload = nodePayloadFun(*graph, nodeRef);
				payload.Visited = false;
			}
			initialized_ = true;
		}
	}

	// Returns any unprocessed vertex from any connected component.
	GraphT.NodeId seedVertex()
	{
		ensureInitialized;

		foreach (vertex; graph.nodes)
		{
			auto payload = nodePayloadFun(*graph, vertex);
			if (!payload.Visited)
				return vertex;
		}
		return GraphT.NullNode;
	}

	DfsRange traverse(GraphT.NodeId start)
	{
		// TODO: validate vertex? eg. graph.validateVertex(start)

		ensureInitialized;

		return DfsRange(&this,start);
	}

	struct DfsRange
	{
		DepthFirstSearchT* outer_;
		GraphT.NodeId vertex_;
		this(DepthFirstSearchT* dfs, GraphT.NodeId vertex) { 
			//writeln("inside DFSRange ctr, &dfsRange=", &this, " &dfs=", dfs);

			outer_ = dfs;
			vertex_ = vertex; 
		}
		~this()
		{
			//writeln("DfsRange dtr");
		}
		int opApply(int delegate(GraphT.NodeId) dg)
		{
			// initiate search
			outer_.verticesToProcess.insertFront(vertex_);

			// traverse connected component

			//writeln("starting loop");
			while (true)
			{
				//writeln("checking if queue is empty");
				//writeln("outer_=",outer_);
				//writeln("&outer_.verticesToProcess=",&outer_.verticesToProcess);
				if (outer_.verticesToProcess.empty)
				    break;
				
				//writeln("take next element");
				auto current = outer_.verticesToProcess.front;
				
				//writeln("next=", current);
				outer_.verticesToProcess.removeFront;

				//writeln("query payload");
				//auto payload = outer_.nodePayloadFun(*outer_.graph, current);
				auto payload = nodePayloadFun(*outer_.graph, current);
				if (payload.Visited)
				    continue;
				
				//writeln("assigning visited=true");
				payload.Visited = true;
				
				int result = dg(current);
				if (result) return result;
				
				auto adjs = outer_.graph.adjacentNodes(current);
				foreach(GraphT.NodeId neigh; adjs)
				{
				    //writeln("neigh=",neigh);
					//auto neighPayload = outer_.nodePayloadFun(*outer_.graph, neigh);
					auto neighPayload = nodePayloadFun(*outer_.graph, neigh);
					if (!neighPayload.Visited)
						outer_.verticesToProcess.insertFront(neigh);
				}
				//writeln("exit loop for current=", current);
			}
			return 0;
		}
	}
}