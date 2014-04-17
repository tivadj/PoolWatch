module PoolWatchHelpersDLang.ConnectedComponentsImpl;

import PoolWatchHelpersDLang.traversal;
import PoolWatchHelpersDLang.connect;
import PoolWatchHelpersDLang.GraphHelpers;
import PoolWatchHelpersDLang.MatrixUndirectedGraph;

struct ConnectedComponentsNodeData
{
	DepthFirstSearchNodeData dfsData;
}

int connectedComponentsCount(GraphT, alias nodePayloadFun)(ref GraphT graph)
if ( is(typeof(nodePayloadFun(graph, GraphT.NodeId.init)) == ConnectedComponentsNodeData*) )
{
	auto dfsDataFun = delegate DepthFirstSearchNodeData*(ref GraphT graph, GraphT.NodeId node) { 
		ConnectedComponentsNodeData* ccData = nodePayloadFun(graph,node);
		return &ccData.dfsData; 
	};

	int result = 0;
	foreach(seed; graph.nodes())
	{
	    auto data = dfsDataFun(graph, seed);
		if (data.Visited)
			continue;

	    result++;

		// traverse component

	    auto trav = depthFirstSearch!(GraphT, dfsDataFun)(graph, seed);
	    foreach (int v; trav)
	    {
	    }
	}
	return result;
}


