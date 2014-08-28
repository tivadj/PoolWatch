module PoolWatchHelpersDLang.GraphHelpers;

import std.algorithm; // minPos
import core.stdc.stdlib;

//////////////////////////////////////////////////////

struct EmptyPayload
{
}

MatrixUndirectedGraph!NodePayloadT createMatrixUndirectedGraph(NodePayloadT)(int[] edgePerRow)
{
	assert(edgePerRow.length >= 0);
	assert(edgePerRow.length % 2 == 0);

	// find vertices count
	auto maxInd = minPos!("a>b")(edgePerRow);
	auto vertCount = maxInd[0] + 1;

	auto result = MatrixUndirectedGraph!double(vertCount);

	for (int i=0; i<edgePerRow.length / 2; i++)
	{
		auto from = edgePerRow[i * 2 + 0];
		auto to   = edgePerRow[i * 2 + 1];
		result.setEdge(from, to);
	}
	return result;
}

GraphT createMatrixGraphNew(GraphT)(int[] edgePerRow)
{
	assert(edgePerRow.length >= 0);
	assert(edgePerRow.length % 2 == 0);

	// find vertices count
	auto maxInd = minPos!("a>b")(edgePerRow);
	auto vertCount = maxInd[0] + 1;

	auto result = GraphT(vertCount);

	auto edgesCount = edgePerRow.length / 2;

	for (int i=0; i<edgesCount; i++)
	{
		auto from = edgePerRow[i];
		auto to   = edgePerRow[edgesCount + i];
		result.setEdge(from, to);
	}
	return result;
}

struct CppVector(T)
{
	private 
	{
		T* pData = null;
		int size = 0;
		int capacity = 0;
	}

	//invariant()
	//{
	//    assert(size >= 0);
	//    assert(capacity >= 0);
	//    assert(size <= capacity);
	//    //assert(size <= 1024);
	//    //assert(capacity <= 1024);
	//}

	// postblit constructor
	this(this) @nogc
	{
		CppVector!T tmp;
		tmp.opAssign(this);

		// clear this data to avoid data destruction when swapping
		pData = null;
		swap(tmp, this);
	}

	// TODO: if we put 'const' modifier on 'rhs', the error occur
	//       Error: cannot implicitly convert expression (rhs.pData[0..cast(ulong)rhs.size]) of type const(TrackHypothesisTreeNode*)[] to TrackHypothesisTreeNode*[]
	ref CppVector opAssign(ref CppVector rhs) @nogc
	{
		reserve(rhs.capacity);
		pData[0..rhs.size] = rhs.pData[0..rhs.size];
		size = rhs.size;
		assert(capacity == rhs.capacity);

		return this;
	}

	~this() @nogc
	{
		core.stdc.stdlib.free(pData);
	}

	int length() @nogc
	{
		return size;
	}
	
	bool empty() @nogc
	{
		return size == 0;
	}

	void pushBack(T item) @nogc
	{
		// extend if needed
		if (size >= capacity)
		{
			// allocate new array
			int capNew = capacity == 0 ? 8 : capacity * 2;
			growArray(capNew);
		}

		pData[size] = item;
		size++;
	}

	void reserve(int capacity) @nogc
	{
		assert(capacity >= 0);

		if (capacity <= this.capacity)
			return;

		growArray(capacity);
		
		assert(this.capacity >= capacity, "Error allocating required space");
	}

	private void growArray(int capNew)
	{
		assert(capNew > capacity, "One should request the increase of capacity");

		const static outOfMemErr = new core.exception.OutOfMemoryError;

		auto pDataNew = cast(T*)core.stdc.stdlib.malloc(capNew * T.sizeof);
		if (pDataNew == null)
			throw outOfMemErr;

		// copy data
		if (pData != null)
		{
			pDataNew[0..size] = pData[0..size];
			
			// elements from size to capNew remain uninitialized

			//ulong off = size * T.sizeof;
			//ulong maxOff = capNew * T.sizeof;
			//(cast(ubyte*)pDataNew)[off..maxOff] = 88;
		}

		// swap
		core.stdc.stdlib.free(pData);
		pData = pDataNew;
		capacity = capNew;
	}

	void clear() @nogc
	{
		size = 0;
	}

	T* data() @nogc
	{
		return pData;
	}

	// Gets whole slice.
	T[] opIndex() @nogc
	{
		return pData[0..size];
	}

	ref T opIndex(int i) @nogc
	{
		assert(i < size);
		return pData[i];
	}

	int opApply(int delegate(T) @nogc dg)
	{
		for (int i=0; i<size; ++i)
		{
			auto element = pData[i];
			auto result = dg(element);
			if (result) return result;
		}
		return 0;
	}
}

struct CppVectorAppender(E)
{
	CppVector!(E) *vector;

	void put(E element)
	{
		vector.pushBack(element);
	}

	unittest
	{
		static assert (isOutputRange!(CppVectorAppender!(E), E));
	}
}

CppVectorAppender!E appender(E)(CppVector!(E)* vector)
{
	return CppVectorAppender!(E)(vector);
}
