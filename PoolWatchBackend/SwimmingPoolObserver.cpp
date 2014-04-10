#include <cassert>
#include <iostream>
#include <stdint.h>

#include "opencv2/contrib/compat.hpp"
#include "opencv2/highgui/highgui_c.h"

#include <boost/lexical_cast.hpp>

#include "SwimmingPoolObserver.h"
#include "PoolWatchFacade.h"

using namespace std;

SwimmingPoolObserver::SwimmingPoolObserver(unique_ptr<MultiHypothesisBlobTracker> blobTracker, shared_ptr<CameraProjectorBase> cameraProjector)
: cameraProjector_(cameraProjector)
{
	blobTracker_.swap(blobTracker);
}

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

void SwimmingPoolObserver::processBlobs(size_t frameOrd, const vector<DetectedBlob>& blobs, int* pFrameIndWithTrackInfo)
{
	setBlobs(frameOrd, blobs);

	auto fps = blobTracker_->getFps();
	auto elapsedTimeMs = 1000.0f / fps;
	int frameIndWithTrackInfo = -1;
	vector<TrackChangePerFrame> trackChangeList;
	blobTracker_->trackBlobs((int)frameOrd, blobs, fps, elapsedTimeMs, frameIndWithTrackInfo, trackChangeList);

	if (frameIndWithTrackInfo != -1)
	{
		setTrackChangesPerFrame(frameIndWithTrackInfo, trackChangeList);
	}

	if (pFrameIndWithTrackInfo != nullptr)
		*pFrameIndWithTrackInfo = frameIndWithTrackInfo;
}

void SwimmingPoolObserver::finishTrackHistory(int frameInd)
{
	vector<TrackHypothesisTreeNode*> bestTrackLeaves;
	bool isBestTrackLeafsInitied = false;
	int frameIndWithTrackInfo = -1;
	vector<TrackChangePerFrame> trackChangeList;
	int pruneWindow = -1;
	
	for (bool continue1 = true; continue1; pruneWindow--)
	{
		trackChangeList.clear();
		continue1 = blobTracker_->collectPendingTrackChanges(frameInd, frameIndWithTrackInfo, trackChangeList, bestTrackLeaves, isBestTrackLeafsInitied, pruneWindow);

		// bestTrackLeaves is initialized on the first call
		if (!isBestTrackLeafsInitied)
			isBestTrackLeafsInitied = true;

		CV_Assert(frameIndWithTrackInfo != -1);

		setTrackChangesPerFrame(frameIndWithTrackInfo, trackChangeList);
		
		if (pruneWindow == 0)
			bestTrackLeaves.clear(); // leaves are invalid

		blobTracker_->logVisualHypothesisTree(frameInd, "hypTree", bestTrackLeaves);
	}
}

void SwimmingPoolObserver::toString(stringstream& bld)
{
	bld << "framesCount=" << blobsPerFrame_.size() << std::endl;
	bld << "trackChanges=" << trackIdToHistory_.size() << std::endl;
}

void SwimmingPoolObserver::dumpTrackHistory(stringstream& bld) const
{
	bld << "tracks.count=" << trackIdToHistory_.size() << std::endl;
	for (const auto& trackIdHistPair : trackIdToHistory_)
	{
		int trackId = trackIdHistPair.first;
		const TrackInfoHistory& hist = trackIdHistPair.second;
		bld << "track Id=" << trackId;

		assert(!hist.Assignments.empty());
		bld << " FamilyId=" << hist.Assignments[0].FamilyId;

		bld << " FirstFrameInd=" << hist.FirstAppearanceFrameIdx;
		bld << " LastFrameInd=" << hist.LastAppearanceFrameIdx;
		if (hist.LastAppearanceFrameIdx != TrackInfoHistory::IndexNull)
		{
			int trackedOn = hist.LastAppearanceFrameIdx - hist.FirstAppearanceFrameIdx + 1;
			bld << " framesCount=" << trackedOn;
		}
		bld << std::endl;

		for (size_t i = 0; i < hist.Assignments.size(); ++i)
		{
			int frameInd = hist.FirstAppearanceFrameIdx + i;
			bld << "frameInd=" << frameInd;

			const auto& change = hist.Assignments[i];

			std::string changeStr;
			::toString(change.UpdateType, changeStr);
			bld <<" " << changeStr;
				
			bld << " obsInd=" << change.ObservationInd;
			bld << " ImgPos=" << change.ObservationPosPixExactOrApprox;
			bld << " WorldPos=" << change.EstimatedPosWorld;
			bld << endl;
		}
	}
}

void SwimmingPoolObserver::setTrackChangesPerFrame(int frameOrd, const vector<TrackChangePerFrame>& trackChanges)
{
	for (auto& change : trackChanges)
	{
		auto trackCandidateId = change.FamilyId;

		// get or create track

		auto it = trackIdToHistory_.find(trackCandidateId);
		if (it == trackIdToHistory_.end())
		{
			// create new track
			TrackInfoHistory track;
			track.TrackCandidateId = trackCandidateId;
			track.FirstAppearanceFrameIdx = frameOrd;
			track.LastAppearanceFrameIdx = TrackInfoHistory::IndexNull;
			trackIdToHistory_.insert(make_pair(trackCandidateId, track));
		}

		// TODO: how to handle the case, when there is no change for a track for some frame

		TrackInfoHistory& trackHistory = trackIdToHistory_[trackCandidateId];
		auto localIndex = frameOrd - trackHistory.FirstAppearanceFrameIdx;

		// allow consequent 'put' requests
		if (localIndex == trackHistory.Assignments.size())
		{
			trackHistory.Assignments.resize(localIndex + 1); // reserve space for new element
		}
		else if (localIndex == trackHistory.Assignments.size() - 1)
		{
			// allow multiple modification to the last assignment
		}
		else
		{
			CV_Assert(false && "Can't randomly modify track positions");
		}

		if (change.UpdateType == TrackChangeUpdateType::Pruned)
		{
			trackHistory.LastAppearanceFrameIdx = frameOrd;
			assert(trackHistory.LastAppearanceFrameIdx == trackHistory.FirstAppearanceFrameIdx + trackHistory.Assignments.size() - 1 && "FrameInd of the last appearance must be the last in the assignments array");
		}

		CV_Assert(localIndex == trackHistory.Assignments.size() - 1 && "Modification must be applied to the last element of track history");
		trackHistory.Assignments[localIndex] = change;
	}
}

void SwimmingPoolObserver::adornImage(int frameOrd, int trailLength, cv::Mat& resultImage)
{
	int processedFramesCount = static_cast<int>(blobsPerFrame_.size());
	CV_Assert(frameOrd >= 0 && frameOrd < processedFramesCount && "Parameter frameOrd is out of range");

	int fromFrameOrd = frameOrd - trailLength;
	if (fromFrameOrd < 0)
		fromFrameOrd = 0;
	if (fromFrameOrd >= processedFramesCount)
		fromFrameOrd = processedFramesCount - 1;

	adornImageInternal(fromFrameOrd, frameOrd, trailLength, resultImage);
}

void SwimmingPoolObserver::adornImageInternal(int fromFrameOrd, int toFrameOrd, int trailLength, cv::Mat& resultImage)
{
	for (auto& trackIdToHist : trackIdToHistory_)
	{
		const TrackInfoHistory& track = trackIdToHist.second;
		auto color = getTrackColor(track);

		// limit to available observations

		int maxUpper = track.isFinished() ? track.LastAppearanceFrameIdx : (track.FirstAppearanceFrameIdx + track.Assignments.size() - 1);

		int localFromFrameOrd = fromFrameOrd;
		if (localFromFrameOrd > maxUpper)
			continue;
		else if (localFromFrameOrd < track.FirstAppearanceFrameIdx)
			localFromFrameOrd = track.FirstAppearanceFrameIdx;

		int localToFrameOrd = toFrameOrd;
		if (localToFrameOrd < track.FirstAppearanceFrameIdx)
			continue;
		if (localToFrameOrd < maxUpper) // do not show tracks, finished some time ago
			continue;
		else if (localToFrameOrd > maxUpper)
			localToFrameOrd = maxUpper;
		
		assert(localFromFrameOrd <= localToFrameOrd);

		//

		PoolWatch::PaintHelper::paintTrack(track, localFromFrameOrd, localToFrameOrd, color, blobsPerFrame_, resultImage);
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

std::shared_ptr<CameraProjectorBase> SwimmingPoolObserver::cameraProjector()
{
	return cameraProjector_;
}