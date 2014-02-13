module DepthFirstSearchTests;

import std.stdio;
import PoolWatchHelpersDLang.MatrixUndirectedGraph2;
import PoolWatchHelpersDLang.GraphHelpers;
import PoolWatchHelpersDLang.traversal;

void run()
{
	testDepthFirstSearchFacadeFunction();
}


void testDepthFirstSearchFacadeFunction()
{
	alias MatrixUndirectedGraph2!EmptyPayload GraphT;

	auto g = GraphT(6);
	g.setEdge(1,0);
	g.setEdge(2,0);
	g.setEdge(0,3);
	g.setEdge(3,2);
	g.setEdge(4,5);
	
	static struct Struct1
	{
		DepthFirstSearchNodeData dfs;
	}

	auto nodePayload = new Struct1[g.nodesCount];

	auto dfsRange = depthFirstSearch!(GraphT, (g,v) => &nodePayload[v].dfs)(g, 0);
	foreach(n; dfsRange)
		writeln(n);
}