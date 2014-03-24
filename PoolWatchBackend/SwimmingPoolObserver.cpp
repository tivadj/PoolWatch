#include "SwimmingPoolObserver.h"
#include <cassert>
#include <iostream>
#include <stdint.h>

#include "opencv2/contrib/compat.hpp"
#include "opencv2/highgui/highgui_c.h"

using namespace std;

SwimmingPoolObserver::SwimmingPoolObserver(int pruneWindow, float fps)
{
	cameraProjector_ = make_shared<CameraProjector>();
	blobTracker_.swap(make_unique<MultiHypothesisBlobTracker>(cameraProjector_, pruneWindow, fps));
}

SwimmingPoolObserver::~SwimmingPoolObserver()
{
}

void SwimmingPoolObserver::setBlobs(size_t frameOrd, const vector<DetectedBlob>& blobs)
{
	if (frameOrd == blobsPerFrame_.size())
		blobsPerFrame_.resize(frameOrd + 1);
	else if (frameOrd == blobsPerFrame_.size() - 1)
	{
		// ok, space is available
	}
	else
	{
		CV_Assert(false && "Can update observations only for the last frame");
	}

	assert(frameOrd == blobsPerFrame_.size() - 1);
	blobsPerFrame_[frameOrd] = blobs;
}

void SwimmingPoolObserver::processBlobs(size_t frameOrd, const cv::Mat& image, const vector<DetectedBlob>& blobs, int* pFrameIndWithTrackInfo)
{
	setBlobs(frameOrd, blobs);

	auto fps = blobTracker_->getFps();
	auto elapsedTimeMs = 1000.0f / fps;
	int frameIndWithTrackInfo = -1;
	vector<TrackChangePerFrame> trackChangeList;
	blobTracker_->trackBlobs((int)frameOrd, blobs, image, fps, elapsedTimeMs, frameIndWithTrackInfo, trackChangeList);

	if (frameIndWithTrackInfo != -1)
	{
		setTrackChangesPerFrame(frameIndWithTrackInfo, trackChangeList);
	}

	if (pFrameIndWithTrackInfo != nullptr)
		*pFrameIndWithTrackInfo = frameIndWithTrackInfo;
}

void SwimmingPoolObserver::toString(stringstream& bld)
{
	bld << "framesCount=" << blobsPerFrame_.size() << std::endl;
	bld << "trackChanges=" << trackIdToHistory_.size() << std::endl;
}

void SwimmingPoolObserver::setTrackChangesPerFrame(int frameOrd, const vector<TrackChangePerFrame>& trackChanges)
{
	for (auto& change : trackChanges)
	{
		auto trackCandidateId = change.TrackCandidateId;

		// get or create track

		auto it = trackIdToHistory_.find(trackCandidateId);
		if (it == trackIdToHistory_.end())
		{
			// create new track
			TrackInfoHistory track;
			track.TrackCandidateId = trackCandidateId;
			track.FirstAppearanceFrameIdx = frameOrd;
			trackIdToHistory_.insert(make_pair(trackCandidateId, track));
		}

		TrackInfoHistory& track = trackIdToHistory_[trackCandidateId];
		auto localIndex = frameOrd - track.FirstAppearanceFrameIdx;
		
		// allow consequent 'put' requests
		if (localIndex == track.Assignments.size())
		{
			track.Assignments.resize(localIndex + 1); // reserve space for new element
		}
		else if (localIndex == track.Assignments.size() - 1)
		{
			// ok, space for next element is reserved
		}
		else
		{
			CV_Assert(false && "Can't randomly modify track positions");
		}

		assert(localIndex == track.Assignments.size() - 1 && "Modification must be applied to the last element of track history");
		track.Assignments[localIndex] = change;
	}
}

void SwimmingPoolObserver::adornImage(const cv::Mat& image, int frameOrd, int trailLength, cv::Mat& resultImage)
{
	int processedFramesCount = static_cast<int>(blobsPerFrame_.size());
	CV_Assert(frameOrd >= 0 && frameOrd < processedFramesCount && "Parameter frameOrd is out of range");

	int fromFrameOrd = frameOrd - trailLength;
	if (fromFrameOrd < 0)
		fromFrameOrd = 0;
	if (fromFrameOrd >= processedFramesCount)
		fromFrameOrd = processedFramesCount - 1;

	adornImageInternal(image, fromFrameOrd, frameOrd, trailLength, resultImage);
}

void SwimmingPoolObserver::adornImageInternal(const cv::Mat& image, int fromFrameOrd, int toFrameOrd, int trailLength, cv::Mat& resultImage)
{
	for (auto& trackIdToHist : trackIdToHistory_)
	{
		auto& track = trackIdToHist.second;
		auto color = getTrackColor(track);

		// draw circle in the center of initial observation of the track

		if (track.FirstAppearanceFrameIdx == fromFrameOrd)
		{
			auto pChange = track.getTrackChangeForFrame(fromFrameOrd);
			if (pChange == nullptr)
				continue;

			auto& cent = pChange->ObservationPosPixExactOrApprox;
			cv::circle(resultImage, cent, 3, color);
		}

		// draw track path in frames range of interest

		cv::Point2f prevPoint(-1, -1);;
		for (int frameInd = fromFrameOrd; frameInd <= toFrameOrd; ++frameInd)
		{
			auto pChange = track.getTrackChangeForFrame(frameInd);
			// TODO: implement track termination
			// positions for track can't just stop because tracker approximates track in case of no observation
			//assert(pChange != nullptr);
			if (pChange == nullptr)
				continue; // assume track is finished
			
			auto cent = pChange->ObservationPosPixExactOrApprox;

			bool hasPrevPoint = prevPoint.x != -1;
			if (hasPrevPoint)
				cv::line(resultImage, prevPoint, cent, color);
			
			prevPoint = cent;
		}

		// draw shape outline for the last frame

		auto pLastChange = track.getTrackChangeForFrame(toFrameOrd);
		if (pLastChange != nullptr && pLastChange->ObservationInd >= 0)
		{
			const auto& blobs = blobsPerFrame_[toFrameOrd];
			const auto& obs = blobs[pLastChange->ObservationInd];

			const cv::Mat_<int32_t>& outlinePixMat = obs.OutlinePixels; // [Nx2], N=number of points; (Y,X) per row

			// populate points array
			std::vector<cv::Point> outlinePoints(outlinePixMat.rows);
			for (int i = 0; i < outlinePixMat.rows; ++i)
				outlinePoints[i] = cv::Point(outlinePixMat(i, 1), outlinePixMat(i, 0));

			cv::polylines(resultImage, outlinePoints, true, color);
		}
	}
}

cv::Scalar SwimmingPoolObserver::getTrackColor(const TrackInfoHistory& trackHist)
{
	static vector<cv::Scalar> trackColors;
	trackColors.push_back(CV_RGB(0, 255, 0));
	trackColors.push_back(CV_RGB(0, 0, 255));
	trackColors.push_back(CV_RGB(255, 0, 0));
	trackColors.push_back(CV_RGB(0, 255, 255)); // cyan
	trackColors.push_back(CV_RGB(255, 0, 255)); // magenta
	trackColors.push_back(CV_RGB(255, 255, 0)); // yellow

	int colInd = trackHist.TrackCandidateId % trackColors.size();
	return trackColors[colInd];
}

std::shared_ptr<CameraProjector> SwimmingPoolObserver::cameraProjector()
{
	return cameraProjector_;
}