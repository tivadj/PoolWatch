#pragma once

#include "mex.h"
#include <opencv2/core.hpp>

//#ifdef POOLWATCH_EXPORTS
//#define POOLWATCH_API __declspec(dllexport)
//#else
//#define POOLWATCH_API __declspec(dllimport)
//#endif

typedef void (*MexFunctionDelegate)(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

/** Rectangular region of tracket target, which is detected in camera's image frame. */
struct DetectedBlob
{
	int Id;
	cv::Rect2f BoundingBox;
	cv::Point2f Centroid;
	cv::Mat OutlinePixels;
	cv::Mat FilledImage;
	cv::Point3f CentroidWorld;
};

enum TrackChangeUpdateType
{
	New = 1,
	ObservationUpdate,
	NoObservation
};

struct TrackChangePerFrame
{
	int TrackCandidateId;
	TrackChangeUpdateType UpdateType;
	cv::Point3f EstimatedPosWorld; // [X, Y, Z] corrected by sensor position(in world coord)

	int ObservationInd; // 0 = no observation; >0 observation index
	cv::Point2f ObservationPosPixExactOrApprox; // [X, Y]; required to avoid world->camera conversion on drawing
};

struct TrackInfoHistory
{
	//int Id;
	//bool IsTrackCandidate; // true = TrackCandidate
	int TrackCandidateId;
	int FirstAppearanceFrameIdx;
	//PromotionFramdInd; // the frame when candidate was promoted to track

	std::vector<TrackChangePerFrame> Assignments;

	TrackChangePerFrame* getTrackChangeForFrame(int frameOrd);
};

__declspec(dllexport) void executeMexFunctionSafe(MexFunctionDelegate mexFun, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

//

__declspec(dllexport) void TrackPaintMexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

