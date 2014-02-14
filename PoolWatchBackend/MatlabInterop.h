#pragma once

#include "mex.h"
#include "stdint.h"

// NOTE: Declarations below must be in sync with DLang declarations.
extern "C"
{
	typedef void* mxArrayPtr;

	typedef mxArrayPtr (*pwCreateArrayInt32Fun)(size_t celem);
	typedef void* (*pwGetDataPtrFun)(mxArrayPtr pMat);
	typedef size_t (*pwGetNumberOfElementsFun)(mxArrayPtr pMat);
	typedef void (*pwDestroyArrayFun)(mxArrayPtr pMat);
	typedef void (*DebugFun)(const char*);

	mxArrayPtr pwCreateArrayInt32(size_t celem);
	void* pwGetDataPtr(mxArrayPtr pMat);
	size_t pwGetNumberOfElements(mxArrayPtr pMat);
	void pwDestroyArray(mxArrayPtr pMat);
	void PWmexPrintf(const char* msgSz);
	void PWmexPrintfNull(const char*);

	struct mxArrayFuns_tag
	{
		pwCreateArrayInt32Fun CreateArrayInt32;
		pwGetDataPtrFun GetDataPtr;
		pwGetNumberOfElementsFun GetNumberOfElements;
		pwDestroyArrayFun DestroyArray;
		DebugFun logDebug;
	};

	typedef int32_t* (*pwCreateArrayInt32FunNew)(size_t celem, void* pUserData);
	typedef void(*pwDestroyArrayInt32FunNew)(int32_t* pInt32, void* pUserData);

	struct Int32Allocator
	{
		pwCreateArrayInt32FunNew CreateArrayInt32;
		pwDestroyArrayInt32FunNew DestroyArrayInt32;
		void* pUserData; // data which will be passed to Create/Destroy methods by server code
	};

}

/// Custom deleter for mxArray objects.
struct mxArrayDeleter
{
	void operator()(mxArray* p) const
	{
		mxDestroyArray(p);
	}
};
