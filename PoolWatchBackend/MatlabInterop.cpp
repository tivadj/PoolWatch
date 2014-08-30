#include "MatlabInterop.h"

#if SAMPLE_MATLABPROX
#include <opencv2\matlab\mxarray.hpp>

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

/** Safe wrapper around Matlab mexFunction. */
void executeMexFunctionSafe(MexFunctionDelegate mexFun, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	try
	{
		mexFun(nlhs, plhs, nrhs, prhs);
	}
	catch (cv::Exception& e) {
		matlab::error(std::string("cv::exception caught: ").append(e.what()).c_str());
	}
	catch (std::exception& e) {
		matlab::error(std::string("std::exception caught: ").append(e.what()).c_str());
	}
	catch (...) {
		matlab::error("Uncaught exception occurred in mex function");
	}
}
#endif