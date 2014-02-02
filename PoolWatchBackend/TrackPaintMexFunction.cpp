#include <vector>
#include <tuple>
#include <map>
using namespace std;

#include <opencv2/matlab/bridge.hpp>
#include <opencv2/core.hpp>
using namespace cv;
using namespace matlab;
using namespace bridge;

#include "TrackPainter.h"
#include "PoolWatchFacade.h"

map<int, TrackPainter> trackPaintHandlers;

int createTrackPainter(map<int, TrackPainter>& trackPaintHandlers)
{
	auto nextId = static_cast<int>(trackPaintHandlers.size());
	nextId += 1; // first id=1
	trackPaintHandlers.insert(make_pair(nextId, TrackPainter()));
	return nextId;
}

tuple<bool,string> ParseDetectedBlobStruct(const mxArray* blobsAsStruct, vector<DetectedBlob>& blobs)
{
	bool isStruct = mxIsStruct(blobsAsStruct);
	if (!isStruct)
		return std::make_tuple(false, "Parameter is not a struct");

	size_t blobsCount = mxGetNumberOfElements(blobsAsStruct);
	for (size_t i = 0; i < blobsCount; ++i)
	{
		DetectedBlob blob;
		{
			mxArray* mx = mxGetField(blobsAsStruct, i, "Id");
			if (!mxIsInt32(mx))
				return std::make_tuple(false, "type(Id)==int32 failed");
			int id = Bridge(mx).toInt();
			blob.Id = id;
#if _DEBUG
			mexPrintf("id=%d\n", id);
#endif
		}
		{
			mxArray* mx = mxGetField(blobsAsStruct, i, "Centroid");
			if (!mxIsSingle(mx))
				return std::make_tuple(false, "type(Centroid)==float failed");
			auto pCentr = reinterpret_cast<cv::Point2f*>(mxGetData(mx));
			blob.Centroid = *pCentr;
#if _DEBUG
			mexPrintf("centr=%f %f\n", pCentr->x, pCentr->y);
#endif
		}
		{
			mxArray* mx = mxGetField(blobsAsStruct, i, "CentroidWorld");
			if (!mxIsSingle(mx))
				return std::make_tuple(false, "type(CentroidWorld)==float failed");
			auto pCentr3 = reinterpret_cast<cv::Point3f*>(mxGetData(mx));
			blob.CentroidWorld = *pCentr3;
#if _DEBUG
			mexPrintf("centr=%f %f\n", pCentr3->x, pCentr3->y);
#endif
		}
		{
			mxArray* mx = mxGetField(blobsAsStruct, i, "BoundingBox");
			if (!mxIsSingle(mx))
				return std::make_tuple(false, "type(BoundingBox)==float failed");
			auto pBnd = reinterpret_cast<cv::Rect2f*>(mxGetData(mx));
			blob.BoundingBox = *pBnd;
#if _DEBUG
			mexPrintf("bnd=%f %f %f %f\n", pBnd->x, pBnd->y, pBnd->width, pBnd->height);
#endif
		}
		{
			mxArray* mx = mxGetField(blobsAsStruct, i, "OutlinePixels");
			if (!mxIsInt32(mx))
				return std::make_tuple(false, "type(OutlinePixels)==int32 failed");
			cv::Mat mat = Bridge(mx).toMat();
			blob.OutlinePixels = mat;
#if _DEBUG
			mexPrintf("outPix=%d %d\n", mat.rows, mat.cols);
			if (mat.rows >= 2)
			{
				mexPrintf("--%d %d\n", mat.at<int32_t>(0, 0), mat.at<int32_t>(0, 1));
				mexPrintf("--%d %d\n", mat.at<int32_t>(1, 0), mat.at<int32_t>(1, 1));
			}
#endif
		}
		{
			mxArray* mx = mxGetField(blobsAsStruct, i, "FilledImage");
			if (!mxIsLogical(mx))
				return std::make_tuple(false, "type(FilledImage)==bool failed");
			cv::Mat mat = Bridge(mx).toMat();
			blob.FilledImage = mat;
#if _DEBUG
			mexPrintf("FilledImage=%d %d\n", mat.rows, mat.cols);
#endif
		}
		blobs.push_back(blob);
	}
	return std::make_tuple(true, "");
}


std::tuple<bool,std::string> ParseTrackChangesStruct(const mxArray* trackChangesStruct, vector<TrackChangePerFrame>& trackChanges)
{
	bool isStruct = mxIsStruct(trackChangesStruct);
	if (!isStruct)
		return std::make_tuple(false, "Parameter is not a struct");

	size_t changesCount = mxGetNumberOfElements(trackChangesStruct);
	for (size_t i = 0; i < changesCount; ++i)
	{
		TrackChangePerFrame change;
		{
			mxArray* mx = mxGetField(trackChangesStruct, i, "TrackCandidateId");
			if (!mxIsInt32(mx))
				return std::make_tuple(false, "type(Id)==int32 failed");
			int trackCandidateId = Bridge(mx).toInt();
			change.TrackCandidateId = trackCandidateId;
#if _DEBUG
			mexPrintf("TrackCandidateId=%d\n", trackCandidateId);
#endif
		}
		{
			mxArray* mx = mxGetField(trackChangesStruct, i, "UpdateType");
			if (!mxIsInt32(mx))
				return std::make_tuple(false, "type(UpdateType)==int32 failed");
			int updateType = Bridge(mx).toInt();
			change.UpdateType = static_cast<TrackChangeUpdateType>(updateType);
#if _DEBUG
			mexPrintf("UpdateType=%d\n", updateType);
#endif
		}
		{
			mxArray* mx = mxGetField(trackChangesStruct, i, "EstimatedPosWorld");
			if (!mxIsSingle(mx))
				return std::make_tuple(false, "type(EstimatedPosWorld)==float failed");
			auto pCentr3 = reinterpret_cast<cv::Point3f*>(mxGetData(mx));
			change.EstimatedPosWorld = *pCentr3;
#if _DEBUG
			mexPrintf("EstimatedPosWorld=%f %f\n", pCentr3->x, pCentr3->y);
#endif
		}
		{
			mxArray* mx = mxGetField(trackChangesStruct, i, "ObservationInd");
			if (!mxIsInt32(mx))
				return std::make_tuple(false, "type(ObservationInd)==int32 failed");
			int obsInd = Bridge(mx).toInt() - 1; // -1 due to Matlab indices start from one
			change.ObservationInd = obsInd;
#if _DEBUG
			mexPrintf("ObservationInd=%d\n", obsInd);
#endif
		}
		{
			mxArray* mx = mxGetField(trackChangesStruct, i, "ObservationPosPixExactOrApprox");
			if (!mxIsSingle(mx))
				return std::make_tuple(false, "type(ObservationPosPixExactOrApprox)==float failed");
			auto pCentr = reinterpret_cast<cv::Point2f*>(mxGetData(mx));
			change.ObservationPosPixExactOrApprox = *pCentr;
#if _DEBUG
			mexPrintf("ObservationPosPixExactOrApprox=%f %f\n", pCentr->x, pCentr->y);
#endif
		}
		trackChanges.push_back(change);
	}
	return make_tuple(true, "");
}

void TrackPaintMexFunction(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
#if _DEBUG
	mexPrintf("inside TrackPaintMexFunction int=%d\n", sizeof(int));
#endif
	//const char* argMethodName = "PWTrackPainter";
	//ArgumentParser arguments(argMethodName);
	//arguments.addVariant("new", 2, 0);
	//arguments.addVariant("setBlobs", 4, 0);
	//arguments.addVariant("adornImage", 5, 0);
	//MxArrayVector sorted = arguments.parse(MxArrayVector(prhs, prhs + nrhs));
	MxArrayVector sorted(prhs, prhs + nrhs);
	BridgeVector inputs(sorted.begin(), sorted.end());

	int objId = inputs[0].toInt();
	string methodName = inputs[1].toString();
	if (methodName == "new")
	{
		// fun(0, 'new') -> objId

		if (nlhs > 1)
		{
			matlab::error("Provide single output argument");
			return;
		}

		auto nextId = createTrackPainter(trackPaintHandlers);

		mxArray* mx = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
		*static_cast<int32_t*>(mxGetData(mx)) = static_cast<int32_t>(nextId);

#if _DEBUG
		mexPrintf("objIds count=%d\n", trackPaintHandlers.size());
#endif

		if (nlhs > 0)
			plhs[0] = mx;
	}
	else if (methodName == "setBlobs")
	{
		// eg: fun(1,'setBlobs', frameOrd, blobs) -> void
		auto objIdIt = trackPaintHandlers.find(objId);
		if (objIdIt == trackPaintHandlers.end())
		{
			matlab::error("Object with given ID doesn't exist");
			return;
		}

		if (!mxIsInt32(prhs[2]))
		{
			matlab::error("type(frameOrd)==int32 failed");
			return;
		}
		int frameOrd = inputs[2].toInt() - 1; // -1 due to Matlab indices start from one
		const mxArray* pDetectionsPerFrame = prhs[3];

		vector<DetectedBlob> blobs;
		auto parseRes = ParseDetectedBlobStruct(pDetectionsPerFrame, blobs);
		if (!get<0>(parseRes))
		{
			matlab::error(get<1>(parseRes));
			return;
		}

		TrackPainter& trackPainter = objIdIt->second;
		trackPainter.setBlobs(frameOrd, blobs);
	}
	else if (methodName == "setTrackChangesPerFrame")
	{
		// eg fun(1, 'setTrackChangesPerFrame', frameOrd, trackChanges)

		auto objIdIt = trackPaintHandlers.find(objId);
		if (objIdIt == trackPaintHandlers.end())
		{
			matlab::error("Object with given ID doesn't exist");
			return;
		}

		if (!mxIsInt32(prhs[2]))
		{
			matlab::error("type(frameOrd)==int32 failed");
			return;
		}
		int frameOrd = inputs[2].toInt() - 1; // -1 due to Matlab indices start from one
		const mxArray* pDetectionsPerFrame = prhs[3];

		vector<TrackChangePerFrame> trackChanges;
		auto parseRes = ParseTrackChangesStruct(pDetectionsPerFrame, trackChanges);
		if (!get<0>(parseRes))
		{
			matlab::error(get<1>(parseRes));
			return;
		}

		TrackPainter& trackPainter = objIdIt->second;
		trackPainter.setTrackChangesPerFrame(frameOrd, trackChanges);
	}
	else if (methodName == "adornImage")
	{
		// eg fun(1, 'adornImage', image, frameOrd, trailLength) -> adornedImage

		if (nlhs > 1)
		{
			matlab::error("Provide adornedImage output argument");
			return;
		}

		auto objIdIt = trackPaintHandlers.find(objId);
		if (objIdIt == trackPaintHandlers.end())
		{
			matlab::error("Object with given ID doesn't exist");
			return;
		}

		cv::Mat image = inputs[2].toMat();

		if (!mxIsInt32(prhs[3]))
		{
			matlab::error("type(frameOrd)==int32 failed");
			return;
		}
		int frameOrd = inputs[3].toInt() - 1; // -1 due to Matlab indices start from one
		
		if (!mxIsInt32(prhs[4]))
		{
			matlab::error("type(TrailLength)==int32 failed");
			return;
		}

		int trailLength = inputs[4].toInt();
#if _DEBUG
		mexPrintf("trailLength=%d\n", trailLength);
#endif

		TrackPainter& trackPainter = objIdIt->second;
		cv::Mat adornedImage = image.clone();
		trackPainter.adornImage(image, frameOrd, trailLength, adornedImage);

		if (nlhs > 0)
		{
			// TODO: Bridge::FromMat->deepCopyAndTranspose seems to have a memory leak
			MxArray adornedImageMx = Bridge::FromMat<uint8_t>(adornedImage);
			plhs[0] = adornedImageMx.releaseOwnership();
		}
	}
	else if (methodName == "toString")
	{
		// fun(1, 'toString') -> void

		stringstream bld;
		trackPaintHandlers[objId].toString(bld);
		string msg = bld.str();
		mexPrintf("%s", msg.c_str());
	}
}
