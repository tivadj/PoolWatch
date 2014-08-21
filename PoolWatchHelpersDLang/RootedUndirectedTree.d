module PoolWatchHelpersDLang.RootedUndirectedTree;
import std.stdio;
import std.container;
import std.exception;
import std.format;
import std.array;
import std.traits; // Unqual
import PoolWatchHelpersDLang.NearestCommonAncestorOfflineAlgorithm;

struct RootedUndirectedTreeFacade
{
	// TODO: if it is const then what modifier should be on Dfs.seedVertex?
	static NodeId NullNode = NodeId(null);

	struct NodeId // TODO: static?
	{
		alias pNode_ this;
		TreeNode* pNode_;
		this(TreeNode* pNode) { pNode_ = pNode; }
		bool isNull()
		{
			return pNode_ == null;
		}
	}

	struct TreeNode
	{
		//int Id;
		SList!NodeId Children;
		NodeId Parent;
	}

	//template treeType(NodePayloadT)
	//{
	//    enum treeType = RootedUndirectedTree(NodePayloadT);
	//}
}

struct RootedUndirectedTree(NodePayloadT)
{
	alias NodePayloadT NodePayloadType;
	alias RootedUndirectedTreeFacade.NodeId NodeId;
	alias RootedUndirectedTreeFacade.NullNode NullNode;

	NodeId root_ = NodeId(null);
	int nodesCount_;
	//int nodeIdGenerator = 0;


	private struct TreeNodeInternal
	{
		//int Id;
		RootedUndirectedTreeFacade.TreeNode CoreData;
		alias CoreData this;
		
		static if (!is(typeof(NodePayloadT) == PoolWatchHelpersDLang.GraphHelpers.EmptyPayload))
		{
			NodePayloadT NodePayload;
		}
	}

	// this() {}

	~this()
	{
		deallocate(root_);
	}

	NodeId createRootNode()
	{
		return addChildNode(NullNode);
	}

	NodeId root()
	{
		return root_;
	}

	NodeId addChildNode(NodeId node)
	{
		auto child = allocateNewNode;

		if (node.isNull)
		{
			// create root
			enforce(root_.isNull, "Root node already exist");
			root_ = child;
		}
		else
		{
			node.Children.insertFront(child);
		}
		child.Parent = node;

		nodesCount_++;

		return child;
	}

	NodeId parent(NodeId node)
	{
		enforce(!node.isNull);
		return node.Parent;
	}

	private NodeId allocateNewNode()
	{
		auto pRoot = new TreeNodeInternal;
		//pRoot.Id = nodeIdGenerator++;

		// reduce information about allocated node being TreeNodeInternal
		// and present it as TreeNode
		auto pCore = cast(RootedUndirectedTreeFacade.TreeNode*)pRoot;
		return NodeId(pCore);
	}

	private void deallocate(NodeId startFrom)
	{
		if (startFrom.isNull)
			return;

		foreach(NodeId child; startFrom.Children)
		{
			deallocate(child);
		}

		destroy(startFrom.pNode_);
	}

	struct NodeChildrenRange
	{
		NodeId node_;
		this(NodeId node) { node_ = node; }
		int opApply(int delegate(NodeId node) dg)
		{
			foreach(child; node_.Children)
			{
				int result = dg(child);
				if (result) return result;
			}
			return 0;
		}
	}

	NodeChildrenRange children(NodeId node)
	{
		return NodeChildrenRange(node);
	}

	//int childrenCount(NodeId node)
	//{
	//    return node.Children.length;
	//}

	bool hasChildren(NodeId node)
	{
		return !node.Children.empty;
	}

	static if (!is(typeof(NodePayloadT) == PoolWatchHelpersDLang.GraphHelpers.EmptyPayload))
	{
		ref NodePayloadT nodePayload(NodeId node)
		{
			// recover full infromation that TreeNode* is actually TreeNodeInternal*
			auto pNodeInternal = cast(TreeNodeInternal*)node.pNode_;
			auto pPayload = cast(NodePayloadT*)&pNodeInternal.NodePayload;
			return *pPayload;
		}
	}

	auto adjacentNodes(NodeId node)
	{
		auto rootedTree = &this;

		struct AdjacentNodesRange
		{
			int opApply(int delegate(NodeId) dg)
			{
				foreach(child; rootedTree.children(node))
				{
					int result = dg(child);
					if (result) return result;
				}

				if (!node.Parent.isNull)
				{
					int result = dg(node.Parent);
					if (result) return result;
				}

				return 0;
			}
		}
		return AdjacentNodesRange();
	}

	// TODO: should DFS be used here?
	auto nodes()
	{
		auto rootedTree = &this;

		struct AllTreeNodesRange
		{
			int opApply(int delegate(NodeId) dg)
			{
				return opApplyRec(rootedTree.root, dg);
			}
			private int opApplyRec(NodeId startFrom, int delegate(NodeId) dg)
			{
				if (startFrom.isNull)
					return 0;

				int result = dg(startFrom);
				if (result) return result;

				foreach(child; rootedTree.children(startFrom))
				{
					int result = opApplyRec(child, dg);
					if (result) return result;
				}

				return 0;
			}
		}
		return AllTreeNodesRange();
	}

	int nodesCount()
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

/// Gets path from node $(D pathFromMe) to root, navigating parental relashions. The result is a reversal of branch(leaf) nodes.
// http://www.proofwiki.org/wiki/Definition:Rooted_Tree/Branch
//auto branchReversed(RootedTreeT)(ref RootedTreeT tree, RootedTreeT.NodeId pathFromMe) // NOTE: can't infer parameter types
auto branchReversed(RootedTreeT, NodeIdT)(ref RootedTreeT tree, NodeIdT pathFromMe) if (is(NodeIdT == RootedTreeT.NodeId))
{
	struct BranchReversedRange
	{
		int opApply(int delegate(RootedTreeT.NodeId) dg)
		{
			auto current = pathFromMe;
			while (!current.isNull)
			{
				auto result = dg(current);
				if (result) return result;

				current = tree.parent(current);
			}
			return 0;
		}
	}
	return BranchReversedRange();
}

// Gather leaves in the rooted tree.
//void getLeaves(RootedTreeT, OutRangeT)(ref RootedTreeT tree, OutputRange!(RootedTreeT.NodeId) result)
void getLeaves(RootedTreeT, OutRangeT)(ref RootedTreeT tree, OutRangeT result)
//if (isOutputRange!(OutRangeT,RootedTreeT.NodeId))
{
	foreach(RootedTreeT.NodeId n; tree.nodes)
	{
		if (!tree.hasChildren(n))
		{
			result.put(n);
		}
	}
}