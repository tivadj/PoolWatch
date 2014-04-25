#include "algos1.h"
#include "PoolWatchFacade.h"

namespace CameraProjectorTestsNS
{
	using namespace std;
	using namespace PoolWatch;

	void run()
	{
		int imageWidth = 640;
		int imageHeight = 480;
		float fovX = deg2rad(62);
		float fovY = deg2rad(49);
		float fx = -1;
		float fy = -1;
		float cx = -1;
		float cy = -1;

		approxCameraMatrix(imageWidth, imageHeight, fovX, fovY, cx, cy, fx, fy);
		
		// expected
		// cx,cy = 320,240
		// fx,fy = 532,526
	}
}