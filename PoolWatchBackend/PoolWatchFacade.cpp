#include <opencv2\core.hpp>
#include <opencv2\matlab\mxarray.hpp>

#include "PoolWatchFacade.h"

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

TrackChangePerFrame* TrackInfoHistory::getTrackChangeForFrame(int frameOrd)
{
	if (frameOrd < FirstAppearanceFrameIdx)
		return nullptr;
	if (frameOrd >= FirstAppearanceFrameIdx + Assignments.size())
		return nullptr;

	return &Assignments[frameOrd - FirstAppearanceFrameIdx];

}

