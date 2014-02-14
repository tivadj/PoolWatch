import std.stdio;
import std.string;
import std.stdint;
import core.memory;
import PoolWatchHelpersDLang.PoolWatchInteropDLang;
//import PoolWatchHelpersDLang.PoolWatchInteropDLang;

void run()
{
	test1;
	writeln("PWInteropTestHelpers exits");
}

struct ArrayHolder 
{
	int[] Array;
	this (int[] array) { 
		Array = array;
		//writeln("ArrayHolder ctr"); 
	}
	~this() { 
		//writeln("ArrayHolder dtr"); 
	}
}

extern (C) mxArrayPtr pwCreateArrayInt32DImpl(size_t celem)
{
	assert(celem > 0);
	int[] result = new int[celem];
	//writeln("Array data address=", &result[0]);

	auto holder = new ArrayHolder(result);
	//writeln("ArrayHolder address=", holder);

	// protect ArrayHolder from GC
	GC.addRoot(holder);
	GC.setAttr(holder, GC.BlkAttr.NO_MOVE);
	return cast(void*)holder;
}

extern (C) void* pwGetDataPtrDImpl(mxArrayPtr pMat)
{
	ArrayHolder* holder = cast(ArrayHolder*)pMat;
	return &holder.Array[0];
}

extern (C) void pwDestroyArrayDImpl(mxArrayPtr pMat)
{
	ArrayHolder* pHolder = cast(ArrayHolder*)pMat;

	GC.removeRoot(pHolder);
	GC.clrAttr(pHolder, GC.BlkAttr.NO_MOVE);

	destroy(pHolder.Array);
	//destory(pHolder); // TODO: can't deallocate ArrayHolder
}

extern (C) size_t pwGetNumberOfElementsDImpl(mxArrayPtr pMat)
{
	ArrayHolder* pHolder = cast(ArrayHolder*)pMat;
	return pHolder.Array.length;
}

extern (C) void debugFun(const(char)* msg) 
{
	printf(msg); 
	return; 
};


mxArrayFuns_tag createMxArrayFuns()
{
	mxArrayFuns_tag mxArrayFuns;
	mxArrayFuns.CreateArrayInt32 = &pwCreateArrayInt32DImpl;
	mxArrayFuns.GetDataPtr = &pwGetDataPtrDImpl;
	mxArrayFuns.GetNumberOfElements = &pwGetNumberOfElementsDImpl;
	mxArrayFuns.DestroyArray = &pwDestroyArrayDImpl;
	mxArrayFuns.logDebug = &debugFun;

	return mxArrayFuns;
}

Int32Allocator createInt32Allocator()
{
	Int32Allocator alloc;
	alloc.CreateArrayInt32 = function int32_t*(size_t size, void* pUserData) { auto result = new int32_t[size]; return &result[0]; };
	alloc.DestroyArrayInt32 = function void(int32_t* p, void* pUserData) { destroy(p); };
	alloc.pUserData = null; // do not pass around helper data
	return alloc;
}

private void test1()
{
    size_t elcount = 100_000_000; 
	
	mxArrayPtr p = pwCreateArrayInt32DImpl(elcount);
	writeln("client Mat address=", p);

	int* pInt = cast(int*)pwGetDataPtrDImpl(p);
	writeln("client Array address=", pInt);

	GC.collect;

	for (int i=0; i<elcount; ++i)
		pInt[0] = 10 + i;

	pwDestroyArrayDImpl(p);
}
