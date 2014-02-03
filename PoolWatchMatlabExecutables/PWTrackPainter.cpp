#include "mex.h"
#include "PoolWatchFacade.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	executeMexFunctionSafe(TrackPaintMexFunction, nlhs, plhs, nrhs, prhs);
}
