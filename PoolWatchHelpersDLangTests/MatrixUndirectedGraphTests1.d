module MatrixUndirectedGraphTests1;

import std.stdio;

import PoolWatchHelpersDLang.MatrixUndirectedGraph;
import PoolWatchHelpersDLang.MatrixUndirectedGraph2;
import PoolWatchHelpersDLang.GraphHelpers;

void test1()
{
	testSimple();
	
	testCreateMatrixUndirectedGraphNew();
}

void testSimple()
{
	auto g = MatrixUndirectedGraph!double(3);
	g.setEdge(1,0);
	g.setEdge(2,0);
	g.setEdge(1,2);
	g.Fun1;

	g.setVertex(0, 17.1);
	g.setVertex(1, 14.4);
	g.setVertex(2, 22.3);

	foreach(v; g.nodes)
	{
		writeln(v);
		foreach(neigh; g.adjacentNodes(v))
		{
			write(neigh, ' ');
		}
		writeln;
	}

	auto s = g.toString;
	writeln(s);
}

void testCreateMatrixUndirectedGraphNew()
{
	auto edgeList = [
		2, 1, 2, 3,
		1, 0, 0, 4];
	auto g = createMatrixGraphNew!(MatrixUndirectedGraph2!EmptyPayload)(edgeList);
	assert(g.getVerticesCount == 5);
	assert(g.getEdgesCount == 4);
	auto str = g.toString;
	writeln(str);
}
