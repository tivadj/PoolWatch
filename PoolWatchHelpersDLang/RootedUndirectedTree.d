module PoolWatchHelpersDLang.RootedUndirectedTree;
import std.stdio;
import std.container;
import std.exception;
import std.format;
import std.array;
import std.traits; // Unqual
import PoolWatchHelpersDLang.NearestCommonAncestorOfflineAlgorithm;
import PoolWatchHelpersDLang.GraphHelpers;

// Returns compile time size of the NodeId for RootedUndirectedTree type.
// static assert(getNodeIdSize!RootedUndirectedTree == 4)
template getNodeIdSize(alias TreeT) 
if (is(TreeT!double == RootedUndirectedTree!(double)))
{
	enum int getNodeIdSize = (void*).sizeof;
}

struct RootedUndirectedTree(NodePayloadT)
{
	private static struct TreeNode
	{
		CppVector!NodeId Children;
		NodeId Parent;

		static if (!is(typeof(NodePayloadT) == PoolWatchHelpersDLang.GraphHelpers.EmptyPayload))
		{
			NodePayloadT NodePayload;
		}
	}

	alias TreeNode* NodeId;
	enum size_t NodeIdSize = getNodeIdSize!RootedUndirectedTree;
	static assert(NodeId.sizeof == NodeIdSize);

	private 
	{
		NodeId root_ = NodeId(null);
		int nodesCount_;
	}

	~this()
	{
		deallocateRec(root_);
	}

	bool isNull(NodeId node) @nogc
	{
		return node == null;
	}

	NodeId createRootNode() @nogc
	{
		return addChildNode(null);
	}

	NodeId root() @nogc
	{
		return root_;
	}

	NodeId addChildNode(NodeId node) @nogc
	{
		const static Exception exc = new Exception("Root node already exist");

		auto child = allocateNewNode;

		if (isNull(node))
		{
			// create root
			if (!isNull(root_))
				throw exc;
			root_ = child;
		}
		else
		{
			node.Children.pushBack(child);
		}
		child.Parent = node;

		nodesCount_++;

		return child;
	}

	NodeId parent(NodeId node) @nogc
	{
		const static Exception exc = new Exception("Reference can't be null");
		//enforce(!isNull(node));
		if (isNull(node))
			throw exc;
		return node.Parent;
	}

	private NodeId allocateNewNode() @nogc
	{
		auto pRoot = cast(TreeNode*)core.stdc.stdlib.malloc(TreeNode.sizeof);
		auto pRoot2 = std.conv.emplace!(TreeNode)(pRoot);
		assert(pRoot == pRoot2);
		return pRoot;
	}

	private void deallocateRec(NodeId startFrom) @nogc
	{
		if (isNull(startFrom))
			return;

		foreach(NodeId child; startFrom.Children[])
		{
			deallocateRec(child);
		}

		core.stdc.stdlib.free(startFrom);
	}

	struct NodeChildrenRange
	{
		NodeId node_;
		int opApply(int delegate(NodeId node) @nogc dg)
		{
			foreach(child; node_.Children)
			{
				int result = dg(child);
				if (result) return result;
			}
			return 0;
		}
	}

	auto children(NodeId node) @nogc
	{
		return NodeChildrenRange(node);
	}

	//int childrenCount(NodeId node)
	//{
	//    return node.Children.length;
	//}

	bool hasChildren(NodeId node) @nogc
	{
		return !node.Children.empty;
	}

	static if (!is(typeof(NodePayloadT) == PoolWatchHelpersDLang.GraphHelpers.EmptyPayload))
	{
		ref NodePayloadT nodePayload(NodeId node) @nogc
		{
			// recover full infromation that TreeNode* is actually TreeNodeInternal*
			// TODO: can we return just 'node.NodePayload' or it the copy will be returned
			auto pPayload = cast(NodePayloadT*)&node.NodePayload;
			return *pPayload;
		}
	}

	private static struct AdjacentNodesRange
	{
		RootedUndirectedTree!(NodePayloadT)* rootedTree_;
		NodeId node_;

		int opApply(scope int delegate(NodeId) @nogc dg) @nogc
		{
			foreach(child; rootedTree_.children(node_))
			{
				int result = dg(child);
				if (result) return result;
			}

			if (!rootedTree_.isNull(node_.Parent))
			{
				int result = dg(node_.Parent);
				if (result) return result;
			}

			return 0;
		}
	}

	auto adjacentNodes(NodeId node) @nogc
	{
		return AdjacentNodesRange(&this, node);
	}

	private static struct AllTreeNodesRange
	{
		RootedUndirectedTree!(NodePayloadT)* rootedTree_;

		int opApply(int delegate(NodeId) @nogc dg) @nogc
		{
			return opApplyRec(rootedTree_.root, dg);
		}
		private int opApplyRec(NodeId startFrom, int delegate(NodeId) @nogc dg) @nogc
		{
			if (rootedTree_.isNull(startFrom))
				return 0;

			int result = dg(startFrom);
			if (result) return result;

			foreach(child; rootedTree_.children(startFrom))
			{
				int result = opApplyRec(child, dg);
				if (result) return result;
			}

			return 0;
		}
	}
	// TODO: should DFS be used here?
	auto nodes() @nogc
	{
		return AllTreeNodesRange(&this);
	}

	int nodesCount() @nogc
	{
		return nodesCount_;
	}
}

void printTree(RootedTreeT,NodeIdT)(ref RootedTreeT rootedTree, NodeIdT startFrom, void delegate(NodeIdT, std.array.Appender!(string)) nodeFormatter, Appender!string buff) 
	if (is(NodeIdT == RootedTreeT.NodeId))
{
	formattedWrite(buff, "(");

	nodeFormatter(startFrom, buff);

	if (rootedTree.hasChildren(startFrom))
	{
		formattedWrite(buff, " ");
		foreach(child; rootedTree.children(startFrom))
		{
			printTree(rootedTree, child, nodeFormatter, buff);
		}
	}

	formattedWrite(buff, ")");
}

// -1 open bracket
// -2 close bracket
// -1 -2 = empty tree
// -1 1 86 -2 = vertexId=1 with observationId=2 tree
// -1 1 86 -1 5 88 6 89 -2 -2 = vertexId=1 with observationId=2 tree
// ()
// 1 = root 1
// 1 (2 3)
// 1 (2 (21 22) 3)
// 'leafToGrow' is null means parsing lexemes stream into the root.
// http://stackoverflow.com/questions/18826348/parsing-a-multi-tree-from-a-string
void parseTree(RootedTreeT,NodeIdT)(ref RootedTreeT tree, NodeIdT leafToGrow, int[] encodedTree, int openBracketLex, int closeBracketLex, 
							int[] nodeDataEntriesBuff, void delegate (NodeIdT node) initNodeDataFun)
if (is(NodeIdT == RootedTreeT.NodeId))
{
	int index = 0;
	SList!(RootedTreeT.NodeId) curLevelParent;
	curLevelParent.insertFront(leafToGrow);
	auto lastCreatedNode = leafToGrow;

	while (index < encodedTree.length)
	{
		int lex = encodedTree[index];

		if (lex == openBracketLex) // initialization of children for current node
		{
			// nodeId goes just before the open bracket
			curLevelParent.insertFront(lastCreatedNode);
			index++;
		}
		else if (lex == closeBracketLex)
		{
			curLevelParent.removeFront;
			index++;
		}
		else
		{
			auto currentParent = curLevelParent.front;
			lastCreatedNode = tree.addChildNode(currentParent);

			// init node with privided data

			for (int i=0; i<nodeDataEntriesBuff.length; ++i)
			{
				assert(index < encodedTree.length);
				auto data = encodedTree[index];
				nodeDataEntriesBuff[i] = data;
				index++;
			}

			auto t1 = nodeDataEntriesBuff[0];
			auto t2 = nodeDataEntriesBuff[1];

			initNodeDataFun(lastCreatedNode);
		}
	}
}

private struct BranchReversedRange(RootedTreeT,NodeIdT)
{
	RootedTreeT* tree;
	NodeIdT pathFromMe;

	int opApply(int delegate(NodeIdT) dg)
	{
		auto current = pathFromMe;
		while (!tree.isNull(current))
		{
			auto result = dg(current);
			if (result) return result;

			current = tree.parent(current);
		}
		return 0;
	}
}

/// Gets path from node $(D pathFromMe) to root, navigating parental relashions. The result is a reversal of branch(leaf) nodes.
// http://www.proofwiki.org/wiki/Definition:Rooted_Tree/Branch
//auto branchReversed(RootedTreeT)(ref RootedTreeT tree, RootedTreeT.NodeId pathFromMe) // NOTE: can't infer parameter types
auto branchReversed(RootedTreeT, NodeIdT)(ref RootedTreeT tree, NodeIdT pathFromMe) @nogc
if (is(NodeIdT == RootedTreeT.NodeId))
{
	return BranchReversedRange!(RootedTreeT,NodeIdT)(&tree, pathFromMe);
}

private struct LeavesRange(RootedTreeT)
{
	RootedTreeT tree_;
	int opApply(int delegate(RootedTreeT.NodeId) @nogc dg) @nogc
	{
		foreach(RootedTreeT.NodeId n; tree_.nodes)
		{
			if (!tree_.hasChildren(n))
			{
				int result = dg(n);
				if (result) return result;
			}
		}
		return 0;
	}
}

// Gather leaves in the rooted tree.
auto getLeaves(RootedTreeT)(ref RootedTreeT tree) @nogc
{
	return LeavesRange!(RootedTreeT)(tree);
}

unittest
{
	import std.range : isInputRange;
	//static assert(isInputRange!LeavesRange);
}
