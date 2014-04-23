#pragma once
#include <stdint.h>
#include <functional>
#if SAMPLE_MATLABPROX
#include "mex.h"
#endif
#include <opencv2/core.hpp>

#ifdef POOLWATCH_EXPORTS
#define POOLWATCH_API __declspec(dllexport)
#else
#define POOLWATCH_API __declspec(dllimport)
#endif

//#define PW_EXPORTS __declspec(dllexport)

class CameraProjectorBase
{
public:
	virtual cv::Point2f worldToCamera(const cv::Point3f& world) const = 0;
	virtual cv::Point3f cameraToWorld(const cv::Point2f& imagePos) const = 0;

	// Finds the area of shape in image(in pixels ^ 2) of an object with world position __worldPos__(in m)
	float worldAreaToCamera(const cv::Point3f& worldPos, float worldArea) const;

	// Calculates distance which has a segment of length worldDist at position worldPos when translating to camera coordinates.
	float distanceWorldToCamera(const cv::Point3f& worldPos, float worldDist) const;
};

/// Class to map camera's image coordinates (X,Y in pixels) and swimming 
/// pool world coordinates ([x y z] in meters).
class POOLWATCH_API CameraProjector : public CameraProjectorBase
{
private:
	cv::Mat_<float> cameraMatrix33_;
	cv::Mat_<float> cameraMatrix33Inv_;
	cv::Mat_<float> distCoeffs_;
	cv::Mat_<float> rvec_;
	cv::Mat_<float> tvec_;
	cv::Mat_<float> worldToCamera44_;
	cv::Mat_<float> cameraToWorld44_;
public:
	CameraProjector();
	virtual ~CameraProjector();
private:
	void init();
public:
	cv::Point2f worldToCamera(const cv::Point3f& world) const override;
	cv::Point3f cameraToWorld(const cv::Point2f& imagePos) const override;

	static float zeroHeight() { return 0; }
};

/** Rectangular region of tracket target, which is detected in camera's image frame. */
struct DetectedBlob
{
	int Id;
	cv::Rect2f BoundingBox;
	cv::Point2f Centroid;
	// TODO: analyze usage to check whether to represent it as a vector of points?
	cv::Mat_<int32_t> OutlinePixels; // [Nx2], N=number of points; (Y,X) per row
	cv::Mat FilledImage; // [W,H] image contains only bounding box of this blob
	cv::Point3f CentroidWorld;
	float AreaPix; // area of the blob in pixels
};

__declspec(dllexport) void fixBlobs(std::vector<DetectedBlob>& blobs, const CameraProjectorBase& cameraProjector);

enum TrackChangeUpdateType
{
	New = 1,
	ObservationUpdate,
	NoObservation,
	Pruned
};

void toString(TrackChangeUpdateType trackChange, std::string& result);

struct TrackChangePerFrame
{
	int FamilyId;
	TrackChangeUpdateType UpdateType;
	cv::Point3f EstimatedPosWorld; // [X, Y, Z] corrected by sensor position(in world coord)

	int ObservationInd; // 0 = no observation; >0 observation index
	cv::Point2f ObservationPosPixExactOrApprox; // [X, Y]; required to avoid world->camera conversion on drawing

	int FrameInd; // used for debugging
	float Score; // used for debugging
};

struct TrackInfoHistory
{
	static const int IndexNull = -1;

	//int Id;
	//bool IsTrackCandidate; // true = TrackCandidate
	int TrackCandidateId;
	int FirstAppearanceFrameIdx;
	int LastAppearanceFrameIdx; // inclusive
	//PromotionFramdInd; // the frame when candidate was promoted to track

	std::vector<TrackChangePerFrame> Assignments;

	bool TrackInfoHistory::isFinished() const;
	const TrackChangePerFrame* getTrackChangeForFrame(int frameOrd) const;
};

__declspec(dllexport) void loadImageAndMask(const std::string& svgFilePath, const std::string& strokeColor, cv::Mat& outImage, cv::Mat_<bool>& outMask);
__declspec(dllexport) void loadWaterPixels(const std::string& folderPath, const std::string& svgFilter, const std::string& strokeStr, std::vector<cv::Vec3d>& pixels, bool invertMask = false, int inflateContourDelta = 0);

__declspec(dllexport) void getPoolMask(const cv::Mat& image, const cv::Mat_<uchar>& waterMask, cv::Mat_<uchar>& poolMask);

#if SAMPLE_MATLABPROX
typedef void(*MexFunctionDelegate)(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

__declspec(dllexport) void executeMexFunctionSafe(MexFunctionDelegate mexFun, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

//

__declspec(dllexport) void TrackPaintMexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
__declspec(dllexport) void MaxWeightInependentSetMaxFirstMexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
#endif

namespace PoolWatch
{
	// utils

	__declspec(dllexport) std::string timeStampNow();

	//

	// Represents buffer of elements with cyclic sematics. When new element is requested from buffer, the reference to
	// already allocated element is returned.
	// Use 'queryHistory' method to get old elements.
	template <typename T>
	struct CyclicHistoryBuffer
	{
	private:
		std::vector<cv::Mat> cyclicBuffer_;
		int freeFrameIndex_;
	public:
		CyclicHistoryBuffer(int bufferSize)
			:freeFrameIndex_(0),
			cyclicBuffer_(bufferSize)
		{
		}

		// initializes each element of the buffer
		auto init(std::function<void(size_t index, T& item)> itemInitFun) -> void
		{
			for (size_t i = 0; i < cyclicBuffer_.size(); ++i)
				itemInitFun(i, cyclicBuffer_[i]);
		}

		auto queryHistory(int indexBack) -> T&
		{
			assert(indexBack <= 0);

			// 0(current) = next free element to return on request
			// -1 = last valid data
			int ind = -1 + freeFrameIndex_ + indexBack;
			if (ind < 0)
				ind += cyclicBuffer_.size();

			assert(ind >= 0 && "Buffer index is out of range");
			assert(ind < cyclicBuffer_.size());

			return cyclicBuffer_[ind];
		};

		auto requestNew() -> T&
		{
			cv::Mat& result = cyclicBuffer_[freeFrameIndex_];

			freeFrameIndex_++;
			if (freeFrameIndex_ >= cyclicBuffer_.size())
				freeFrameIndex_ = 0;

			return result;
		};
	};

	// painting

	struct __declspec(dllexport) PaintHelper
	{
	private:
		std::vector<cv::Scalar> trackColors_;

	public:
		static void getWellKnownColors(std::vector<cv::Scalar>& trackColors);

		PaintHelper();

		void paintBlob(const DetectedBlob& blob, cv::Mat& image);
		static void paintTriangleHBase(cv::Mat& image, cv::Point center, float side, cv::Scalar color);
		static void paintTrack(const TrackInfoHistory& track, int fromFrameOrd, int toFrameOrd, const cv::Scalar& color, const std::vector<std::vector<DetectedBlob>>& blobsPerFrame, cv::Mat& resultImage);
	};
}
