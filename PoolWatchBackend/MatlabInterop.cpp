#include "MatlabInterop.h"

mxArrayPtr pwCreateArrayInt32(size_t celem)
{
	mxArray* outMask = mxCreateNumericMatrix(1, celem, mxINT32_CLASS, mxREAL);
	return outMask;
}

void* pwGetDataPtr(mxArrayPtr pMat)
{
	return mxGetPr(reinterpret_cast<const mxArray*>(pMat));
}

size_t pwGetNumberOfElements(mxArrayPtr pMat)
{
	return mxGetNumberOfElements(reinterpret_cast<const mxArray*>(pMat));
}

void pwDestroyArray(mxArrayPtr pMat)
{
	mxDestroyArray(reinterpret_cast<mxArray*>(pMat));
}

void PWmexPrintf(const char* msgSz)
{
	mexPrintf(msgSz);
}

void PWmexPrintfNull(const char*)
{
	// no output
}
