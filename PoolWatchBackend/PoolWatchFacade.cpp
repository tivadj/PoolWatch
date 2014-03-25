#include <opencv2\core.hpp>
#include <opencv2\matlab\mxarray.hpp>

#include <ctime> // time_t

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

namespace PoolWatch
{
	std::string timeStampNow()
	{
		std::stringstream strBuf;

		time_t  t1 = time(0); // now time

		// Convert now to tm struct for local timezone
		struct tm * now1 = localtime(&t1);

		char buf[80];
		strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", now1); // 20120601070015

		return std::string(buf);
	}
}
