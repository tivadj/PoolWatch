#pragma once
#include <stdint.h>
#include "mex.h"
#include <opencv2/core.hpp>

#ifdef POOLWATCH_EXPORTS
#define POOLWATCH_API __declspec(dllexport)
#else
#define POOLWATCH_API __declspec(dllimport)
#endif

typedef void (*MexFunctionDelegate)(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

/** Rectangular region of tracket target, which is detected in camera's image frame. */
struct DetectedBlob
{
	int Id;
	cv::Rect2f BoundingBox;
	cv::Point2f Centroid;
	cv::Mat_<int32_t> OutlinePixels; // [Nx2], N=number of points; (Y,X) per row
	cv::Mat FilledImage; // [W,H] image contains only bounding box of this blob
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

__declspec(dllexport) void TrackPaintMexFunction                    (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
__declspec(dllexport) void MaxWeightInependentSetMaxFirstMexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);

__declspec(dllexport) void loadImageAndMask(const std::string& svgFilePath, const std::string& strokeColor, cv::Mat& outImage, cv::Mat_<bool>& outMask);
__declspec(dllexport) void loadWaterPixels(const std::string& folderPath, const std::string& svgFilter, const std::string& strokeStr, std::vector<cv::Vec3d>& pixels);

__declspec(dllexport) void getPoolMask(const cv::Mat& image, const cv::Mat_<uchar>& waterMask, cv::Mat_<uchar>& poolMask);