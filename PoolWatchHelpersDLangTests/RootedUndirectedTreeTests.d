import std.stdio;
import std.string;
import std.format;
import std.array;
import PoolWatchHelpersDLang.RootedUndirectedTree;
import PoolWatchHelpersDLang.GraphHelpers;

void run()
{
	//testDoublePayload;
	//testEmptyPayload;
	//parseTreeTests;
	branchTests;
}

void testDoublePayload()
{
	alias RootedUndirectedTree!double TreeT;
	TreeT tree;
	auto n0 = tree.createRootNode;
	auto n1 = tree.addChildNode(n0);
	auto n2 = tree.addChildNode(n0);

	tree.nodePayload(n0) = 17.1;
	tree.nodePayload(n1) = 14.4;
	tree.nodePayload(n2) = 22.3;

	auto nodeFormatterFun = delegate void(TreeT.NodeId node, std.array.Appender!string buff)
	{
		TreeT.NodePayloadType data = tree.nodePayload(node);
		static assert (is(typeof(data) == double));

		formattedWrite(buff, "%s", data);
	};


	Appender!(string) buff;
	buff.put("");
	printTree(tree, tree.root, nodeFormatterFun, buff); // TODO: compiler can't deduce TreeT, why?
	writeln(buff.data);

}

void testEmptyPayload()
{
	alias RootedUndirectedTree!EmptyPayload TreeT;
	TreeT tree;
	auto n0 = tree.createRootNode;
	auto n1 = tree.addChildNode(n0);
	auto n2 = tree.addChildNode(n0);

	// TODO: nodePayload method should not exist, but compiler complains about double->EmptyPayload conversion error
	//tree.nodePayload(n0) = 17.1;
	//tree.nodePayload(n1) = 14.4;
	//tree.nodePayload(n2) = 22.3;

	auto nodeFormatterFun = delegate void(TreeT.NodeId node, std.array.Appender!string buff)
	{
		formattedWrite(buff, "x");
	};

	Appender!(string) buff;
	buff.put("");
	printTree(tree, tree.root, nodeFormatterFun, buff); // TODO: compiler can't deduce TreeT, why?
	writeln(buff.data);
}


void parseTreeTests()
{
	struct NodeIdObsId
	{
		int NodeId;
		int ObsId;
	}

	alias RootedUndirectedTree!NodeIdObsId RootedTreeT;
	RootedTreeT tree;

	//auto treeStream = [1, 86];
	//auto treeStream = [1, 86, -1 , 2, 87, 3, 88, -2, -2];
	//auto treeStream = [1, 86, -1 , 2, 87, -1, 4, 89, -2, 3, 88, -2, -2];
	//auto treeStream = [1, 86, -1 , 2, 87, -1, 4, 89, -2, 3, 88, -1, 5, 89, -2, -2, -2];
	auto treeStream = [1, 86, -1 , 2, 87, -1, 21, 91, 22, 92, -2, 3, 88, -2, -2];
	int[2] nodeDataEntriesBuff;
	auto nullNode = tree.root;
	auto initNodeDataFun = delegate void(RootedTreeT.NodeId node)
	{
		NodeIdObsId* data = &tree.nodePayload(node);
		data.NodeId = nodeDataEntriesBuff[0];
		data.ObsId = nodeDataEntriesBuff[1];
	};
	parseTree(tree, nullNode, treeStream, -1, -2, nodeDataEntriesBuff, initNodeDataFun);

	//
	auto nodeFormatterFun = delegate void(RootedTreeT.NodeId node, std.array.Appender!string buff)
	{
		NodeIdObsId data = tree.nodePayload(node); // TODO: is it reference
		formattedWrite(buff, "%s %s", data.NodeId, data.ObsId);
	};

	Appender!(string) buff;
	buff.put("");
	printTree(tree, tree.root, nodeFormatterFun, buff);
	writeln(buff.data);
}

void branchTests()
{
	alias RootedUndirectedTree!EmptyPayload TreeT;
	TreeT tree;
	auto n0 = tree.createRootNode;
	auto n1 = tree.addChildNode(n0);
	auto n2 = tree.addChildNode(n0);
	auto nodes=branchReversed(tree, n1);
	foreach(n; nodes)
	{
		writeln("node"); // print two times for [n1,n0]
	}
}