#pragma once

#if SAMPLE_MATLABPROX

#include "mex.h"

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
}

/// Custom deleter for mxArray objects.
struct mxArrayDeleter
{
	void operator()(mxArray* p) const
	{
		mxDestroyArray(p);
	}
};

typedef void(*MexFunctionDelegate)(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

PW_EXPORTS void executeMexFunctionSafe(MexFunctionDelegate mexFun, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

//

PW_EXPORTS void TrackPaintMexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
PW_EXPORTS void MaxWeightInependentSetMaxFirstMexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);

#endif
