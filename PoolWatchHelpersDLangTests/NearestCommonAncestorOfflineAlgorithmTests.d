module NearestCommonAncestorOfflineAlgorithmTests;
import std.array; // appender
import std.stdio;
import std.string;
import std.format; // formattedWrite
import PoolWatchHelpersDLang.NearestCommonAncestorOfflineAlgorithm;
import PoolWatchHelpersDLang.RootedUndirectedTree;

void run()
{
	testSchieberFig31;
	testOnBinary3;
	testOnSingleRoot;
	testOnLongSequenceGraph;
	testOnRandom1;
}

void testSchieberFig31()
{
	writeln("testSchieberFig31");
	alias NcaNodePayload!(RootedUndirectedTreeFacade.NodeId) NcaNodeDataT;
	alias RootedUndirectedTree!NcaNodeDataT TreeT;

	TreeT tree;
	auto n1 = tree.createRootNode;
	auto n2 = tree.addChildNode(n1);
	auto n11 = tree.addChildNode(n1);
	auto n13 = tree.addChildNode(n1);
	auto n3 = tree.addChildNode(n2);
	
	auto n4 = tree.addChildNode(n3);
	auto n7 = tree.addChildNode(n3);
	auto n10 = tree.addChildNode(n3);

	auto n5 = tree.addChildNode(n4);
	auto n6 = tree.addChildNode(n4);

	auto n8 = tree.addChildNode(n7);
	
	auto n9 = tree.addChildNode(n8);

	auto n12 = tree.addChildNode(n11);

	auto n14 = tree.addChildNode(n13);

	auto n15 = tree.addChildNode(n14);

	auto n16 = tree.addChildNode(n15);
	auto n17 = tree.addChildNode(n15);
	auto n18 = tree.addChildNode(n15);
	auto n22 = tree.addChildNode(n15);

	auto n19 = tree.addChildNode(n18);

	auto n20 = tree.addChildNode(n19);
	auto n21 = tree.addChildNode(n19);

	auto ncaNodeDataFun = delegate NcaNodeDataT* (ref TreeT tree, TreeT.NodeId node) { 
		return &tree.nodePayload(node);
	};

	NearestCommonAncestorOfflineAlgorithm!(TreeT,ncaNodeDataFun) ncaAlgo;
	ncaAlgo.process(tree);

	auto nodeFormatterFun = delegate void(TreeT.NodeId node, std.array.Appender!string buff)
	{
		NcaNodeDataT* val = ncaNodeDataFun(tree, node);
		formattedWrite(buff, "%s", val.Preorder);
		formattedWrite(buff, " %s", val.Ascendant);
	};

	Appender!(string) buff;
	buff.put("");
	printTree!(TreeT)(tree, tree.root, nodeFormatterFun, buff);
	writeln(buff.data);

	// NCA queries

	auto z1 = ncaAlgo.nearestCommonAncestor(n11, n13);
	auto z1Preorder = ncaNodeDataFun(tree,z1).Preorder;
	writeln("z1Preorder=",z1Preorder);
	assert(1 == z1Preorder);
	
	crossCheckTreeOnNca!(TreeT, ncaNodeDataFun)(tree);
}

void testOnBinary3()
{
	writeln("testOnBinary3");
	alias NcaNodePayload!(RootedUndirectedTreeFacade.NodeId) NcaNodeDataT;

	auto tree = generateGraphPerLevel!NcaNodeDataT(1, 2);
	writeln(tree.nodesCount);

	alias typeof(tree) TreeT;

	auto ncaNodeDataFun = delegate NcaNodeDataT* (ref TreeT tree, TreeT.NodeId node) { 
		return &tree.nodePayload(node);
	};

	crossCheckTreeOnNca!(TreeT, ncaNodeDataFun)(tree);
}

void testOnSingleRoot()
{
	alias NcaNodePayload!(RootedUndirectedTreeFacade.NodeId) NcaNodeDataT;
	alias RootedUndirectedTree!NcaNodeDataT TreeT;

	TreeT tree;
	auto r1 = tree.createRootNode;

	auto ncaNodeDataFun = delegate NcaNodeDataT* (ref TreeT tree, TreeT.NodeId node) { 
		return &tree.nodePayload(node);
	};

	NearestCommonAncestorOfflineAlgorithm!(TreeT,ncaNodeDataFun) ncaAlgo;
	ncaAlgo.process(tree);
	auto z1 = nearestCommonAncestorNaive!(TreeT)(tree, tree.root, tree.root);
	assert(z1 == tree.root);
}

private void crossCheckTreeOnNca(TreeT, alias ncaNodeDataFun)(ref TreeT tree)
{
	NearestCommonAncestorOfflineAlgorithm!(TreeT,ncaNodeDataFun) ncaAlgo;
	ncaAlgo.process(tree);

	auto nodeFormatterFun = delegate void(TreeT.NodeId node, std.array.Appender!string buff)
	{
		auto val = ncaNodeDataFun(tree, node);
		formattedWrite(buff, "%s", val.Preorder);
		formattedWrite(buff, " %s", val.InLabel);
		formattedWrite(buff, " %s", val.Ascendant);
	};
	Appender!(string) buff;
	buff.put("");
	printTree(tree, tree.root, nodeFormatterFun, buff);
	//writeln(buff.data);

		
	foreach(n1; tree.nodes())
	{
		foreach(n2; tree.nodes())
		{
			auto p1 = ncaNodeDataFun(tree, n1);
			auto p2 = ncaNodeDataFun(tree, n2);
			if (p1.Preorder < p2.Preorder)
			{
				//write(p1.Preorder, " ", p2.Preorder);
				auto a1 = ncaAlgo.nearestCommonAncestor(n1, n2);
				auto a1Rev = ncaAlgo.nearestCommonAncestor(n2, n1);
				assert(a1 == a1Rev);

				auto a1p = ncaNodeDataFun(tree, a1);

				//writeln(" nca=", a1p.Preorder);

				auto a2 = nearestCommonAncestorNaive!(TreeT)(tree, n1, n2);
				assert(a1==a2);
			}
		}
	}
}

// levelsCount=0 => tree contains root only
// levelsCount=1 => tree contains root and its children
auto generateGraphPerLevel(NodePayloadT)(int levelsCount, int childrenPerNode)
{
	alias RootedUndirectedTree!NodePayloadT TreeT;

	TreeT tree;
	auto root = tree.createRootNode;

	generateGraphPerLevelRec!(TreeT)(levelsCount, childrenPerNode, root, 0, tree);

	return tree;
}

private void generateGraphPerLevelRec(TreeT)(int levelsCount, int childrenPerNode, TreeT.NodeId curNode, int curLevel, ref TreeT tree)
{
	if (curLevel == levelsCount)
		return;

	for (int i=0; i<childrenPerNode; ++i)
	{
		auto node = tree.addChildNode(curNode);
		generateGraphPerLevelRec!(TreeT)(levelsCount, childrenPerNode, node, curLevel + 1, tree);
	}
}

void testOnLongSequenceGraph()
{
	writeln("testOnLongSequenceGraph");
	alias NcaNodePayload!(RootedUndirectedTreeFacade.NodeId) NcaNodeDataT;

	auto tree = generateGraphPerLevel!NcaNodeDataT(40, 1);
	writeln(tree.nodesCount);

	alias typeof(tree) TreeT;

	auto ncaNodeDataFun = delegate NcaNodeDataT* (ref TreeT tree, TreeT.NodeId node) { 
		return &tree.nodePayload(node);
	};

	crossCheckTreeOnNca!(TreeT, ncaNodeDataFun)(tree);
}

void testOnRandom1()
{
	writeln("testOnRandom1");
	alias NcaNodePayload!(RootedUndirectedTreeFacade.NodeId) NcaNodeDataT;

	auto tree = generateGraphPerLevel!NcaNodeDataT(10, 2);
	writeln(tree.nodesCount);

	alias typeof(tree) TreeT;

	auto ncaNodeDataFun = delegate NcaNodeDataT* (ref TreeT tree, TreeT.NodeId node) { 
		return &tree.nodePayload(node);
	};

	crossCheckTreeOnNca!(TreeT, ncaNodeDataFun)(tree);
}