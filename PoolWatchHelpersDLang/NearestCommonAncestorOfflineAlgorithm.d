module PoolWatchHelpersDLang.NearestCommonAncestorOfflineAlgorithm;
import std.bitmanip;
import core.bitop;
import std.math;
import std.exception;
import std.algorithm; // max
import PoolWatchHelpersDLang.DepthFirstSearchImpl;

// TODO: how this structure would look like if we force normalized (zero-based) nodes for tree.
struct NcaNodePayload(NodeId)
{
	uint Preorder;

	uint InLabel;
	NodeId InLabelNode;

	uint Ascendant;
	uint Level;
	uint Size;
	NodeId HeadNode;
	uint Head; // debug
	DepthFirstSearchNodeData DfsData;
}

// Nearest Common Ancestor (NCA) algorithm.
// Makes preprocessing in O(N) time and performs NCA(x,y) queries in O(1) constant time.
// It makes sense only for rooted trees.
struct NearestCommonAncestorOfflineAlgorithm(RootedTreeT, alias ncaDataFun)
{
	alias NcaNodePayload!(RootedTreeT.NodeId) NcaNodeDataT;
	RootedTreeT* pRootedTree_;
	RootedTreeT.NodeId[] preorderToData_;

	//alias DepthFirstSearchNodeData* delegate (ref RootedTreeT g, RootedTreeT.NodeId n) DfsDataFun;
	//DfsDataFun dfsDataFun_;

	void process(ref RootedTreeT rootedTree)
	{
		static assert(is(typeof(ncaDataFun(rootedTree, RootedTreeT.NodeId.init)) == NcaNodeDataT*));

		// init caches

		pRootedTree_ = &rootedTree;

		if (preorderToData_ == null)
			preorderToData_ = new RootedTreeT.NodeId[pRootedTree_.nodesCount];
		else
			preorderToData_.length = pRootedTree_.nodesCount;

		// calculate Preorder

		int preorder = 1;
		prepareDfsCache;
		auto dfsDataFun = delegate DepthFirstSearchNodeData*(ref RootedTreeT g, RootedTreeT.NodeId n) { return &ncaDataFun(g,n).DfsData; };
		auto dfsNodes = depthFirstSearch!(RootedTreeT, dfsDataFun)(*pRootedTree_, pRootedTree_.root);
		foreach(node; dfsNodes)
		{
			auto nodeData = ncaDataFun(*pRootedTree_, node);
			nodeData.Preorder = preorder;
			preorderToData_[preorder-1] = node;
			preorder++;
		}

		// calculate Level
		int level = 0;
		uint treeSize;
		assignLevelRec(*pRootedTree_, pRootedTree_.root, level, treeSize);

		// calculate inlabel
		prepareDfsCache;

		auto dfsNodes2 = depthFirstSearch!(RootedTreeT, dfsDataFun)(*pRootedTree_, pRootedTree_.root);
		foreach(node; dfsNodes2)
		{
			auto nodeData = ncaDataFun(*pRootedTree_, node);
			auto preorder = nodeData.Preorder;
			auto size = nodeData.Size;

			// i
			auto num = preorder + size - 1;
			auto v1 = (preorder - 1) ^ num;
			
			// leftmost "1"
			int i = v1 == 0 ? 0 : bsr(v1);

			const uint one = 1;
			auto denom = one << i;
			uint fra = num / denom;
			uint inlabel = fra * denom;
			nodeData.InLabel = inlabel;

			auto InLabelNode = getNodeFromPreorderNumber(inlabel);
			nodeData.InLabelNode = InLabelNode;
		}

		// calculate head
		prepareDfsCache;

		auto dfsNodes3 = depthFirstSearch!(RootedTreeT, dfsDataFun)(*pRootedTree_, pRootedTree_.root);
		foreach(node; dfsNodes3)
		{
			auto nodeData = ncaDataFun(*pRootedTree_, node);
			
			auto nodeParent = pRootedTree_.parent(node);

			bool isInLabelChange;
			if (nodeParent.isNull)
				isInLabelChange = true;
			else
			{
				auto nodeParentData = ncaDataFun(*pRootedTree_, nodeParent);
				isInLabelChange = nodeData.InLabel != nodeParentData.InLabel;
			}

			if (isInLabelChange)
			{
				auto inlabelData = ncaDataFun(*pRootedTree_, nodeData.InLabelNode);
				auto inlabPreord = inlabelData.Preorder;
				auto preord = nodeData.Preorder;

				inlabelData.Head = preord;
				inlabelData.HeadNode = node;
			}
			// TODO: head is 0 if it is not propogated for nodes inside the path
			//else
			//{
			//    if (!nodeParent.isNull)
			//        node.Head = node.Parent.Head;
			//}
		}

		calculateAscendants();
	}

	private void calculateAscendants()
	{
		auto root = pRootedTree_.root;
		if (root.isNull)
			return;

		auto completeTreeHeight = completeBinaryTreeHeight(pRootedTree_.nodesCount);
		
		auto dataRoot = ncaDataFun(*pRootedTree_, root);
		auto rootAscendant = 1 << completeTreeHeight;
		dataRoot.Ascendant = rootAscendant;

		prepareDfsCache;

		auto dfsDataFun = delegate DepthFirstSearchNodeData*(ref RootedTreeT g, RootedTreeT.NodeId n) { return &ncaDataFun(g,n).DfsData; };
		auto dfsNodes = depthFirstSearch!(RootedTreeT, dfsDataFun)(*pRootedTree_, pRootedTree_.root);
		foreach(node; dfsNodes)
		{
			auto nodeData = ncaDataFun(*pRootedTree_, node);
			auto nodeParent = pRootedTree_.parent(node);
			if (nodeParent.isNull) // skip root, which has assigned ascendant
				continue;

			auto nodeParentData = ncaDataFun(*pRootedTree_, nodeParent);

			//
			
			if (nodeData.InLabel == nodeParentData.InLabel)
				nodeData.Ascendant = nodeParentData.Ascendant;				
			else
			{
				auto preorder = nodeData.Preorder;
				auto size = nodeData.Size;
				// i
				auto num = preorder + size - 1;
				auto v1 = (preorder - 1) ^ num;

				// leftmost "1"
				int i = v1 == 0 ? 0 : bsr(v1);

				auto asc = nodeParentData.Ascendant + (1 << i);
				nodeData.Ascendant = asc;
			}
		}
	}
	
	// Height is a number of edges in path from leaf to root.
	// n=1 H=0
	// n=2 H=1
	// n=3 H=1
	// n=4 H=2
	private static int completeBinaryTreeHeight(int nodesCount)
	{
		int result = cast(int)(ceil(log2(nodesCount+1)))-1;

		int bsrResult = bsr(nodesCount+1); // index of mostleft "1"
		bool isPowerTwo = (1 << bsrResult) == nodesCount+1;
		int result2 = isPowerTwo ? bsrResult - 1 : bsrResult;
		assert(result == result2);

		return result;
	}

	private void prepareDfsCache()
	{
		foreach(n; pRootedTree_.nodes)
			pRootedTree_.nodePayload(n).DfsData.Visited = false;
	}

	private auto ncaDataFunThis(ref RootedTreeT g, RootedTreeT.NodeId n)
	{
		NcaNodeDataT* ncaData = ncaDataFun(g,n);
		return ncaData;
	}

	private void validatePreorderNumber(uint preorder)
	{
		assert(preorder >= 1);
		assert(preorder <= pRootedTree_.nodesCount);
	}

	private RootedTreeT.NodeId getNodeFromPreorderNumber(uint preorder)
	{
		assert(preorder >= 1);
		assert(preorder <= pRootedTree_.nodesCount);
		return preorderToData_[preorder-1];
	}
	
	private void assignLevelRec(ref RootedTreeT rootedTree, RootedTreeT.NodeId node, uint level, out uint size)
	{
		auto data = ncaDataFun(rootedTree, node);
		data.Level = level;

		size = 1;

		foreach(child; rootedTree.children(node))
		{
			uint childSize;
			assignLevelRec(rootedTree, child, level + 1, childSize);
			size += childSize;
		}

		data.Size = size;
	}

	RootedTreeT.NodeId nearestCommonAncestor(RootedTreeT.NodeId x, RootedTreeT.NodeId y)
	{
		enforce(!x.isNull);
		enforce(!y.isNull);

		auto xData = ncaDataFun(*pRootedTree_, x);
		auto yData = ncaDataFun(*pRootedTree_, y);
		if (xData.InLabel == yData.InLabel)
		{
			auto result = xData.Level < yData.Level ? x : y;
			return result;
		}

		// (l-i) bits of x.InLabel and y.InLabel are the same
		assert(xData.InLabel > 0, "Node was not initialized");
		assert(yData.InLabel > 0, "Node was not initialized");
		auto i1 = bsf(xData.InLabel);
		auto i2 = bsf(yData.InLabel);

		auto xorBoth = xData.InLabel ^ yData.InLabel;
		assert(xorBoth != 0, "inlabel(x) != inlabel(y)");
		auto i3 = bsr(xorBoth);
		auto i = max(i1, i2, i3);

		debug
		{
			auto L = completeBinaryTreeHeight(pRootedTree_.nodesCount);

			// b = NCA(x.inlabel, y.inlabel) in complete binary tree
			// b = (L-i) bits of x.inlabel or y.inlabel, then single "1", then i zeros
			auto b = xData.InLabel >> i;
			b |= 1;
			b = b << i;
		}

		// Step2: find InLabel of NCA(x,y)
		auto common = xData.Ascendant & yData.Ascendant;
		auto factor = (1 << i);
		common =  (common / factor) * factor;

		// j = rightmost "1" in common
		int j1 = bsf(common);
		int j = cast(int)(log2(common - (common & (common-1))));
		assert(j1 == j);

		// clear j bits of x.InLabel (or y.InLabel)
		int inLabelZ = xData.InLabel >> j;
		inLabelZ |= 1; // set j-th bit
		inLabelZ = inLabelZ << j;
		validatePreorderNumber(inLabelZ);

		// Step3: Find xHat
		auto xHat = GetXHatData(x, xData, inLabelZ, j);
		auto yHat = GetXHatData(y, yData, inLabelZ, j);

		auto xHatData = ncaDataFun(*pRootedTree_, xHat);
		auto yHatData = ncaDataFun(*pRootedTree_, yHat);

		auto result = xHatData.Level < yHatData.Level ? xHat : yHat;
		return result;
	}

	private RootedTreeT.NodeId GetXHatData(RootedTreeT.NodeId x, NcaNodeDataT* xData, uint inLabelZ, int j)
	{
		RootedTreeT.NodeId xHat;
		if (xData.InLabel == inLabelZ)
			xHat = x;
		else
		{
			auto shiftLeft = uint.sizeof*8 - j;
			// clear shiftLeft bits from the left
			auto aj = (xData.Ascendant << shiftLeft) >>shiftLeft;

			// leftmost "1" in aj
			int k = bsr(aj);

			uint inLabelW = xData.InLabel;

			// inlabelW = (l-k) bits of inlabel(x), then single set bit, then k zeros
			// set k right bits to 0

			inLabelW = inLabelW >> k;
			inLabelW |= 1;
			inLabelW = inLabelW << k;
			validatePreorderNumber(inLabelW);

			auto labelWNode = getNodeFromPreorderNumber(inLabelW);
			auto labelWData = ncaDataFun(*pRootedTree_, labelWNode);
			auto omega = labelWData.HeadNode;
			xHat = pRootedTree_.parent(omega);
		}
		return xHat;
	}
}

RootedTreeT.NodeId nearestCommonAncestorNaive(RootedTreeT)(ref RootedTreeT tree, RootedTreeT.NodeId x, RootedTreeT.NodeId y)
{
	auto xAnc = x;
	while (!xAnc.isNull)
	{
		auto yAnc = y;
		while (!yAnc.isNull)
		{
			if (xAnc == yAnc)
				return xAnc;

			yAnc = tree.parent(yAnc);
		}

		xAnc = tree.parent(xAnc);
	}
	assert(false, "Both nodes must exist in the tree");
}