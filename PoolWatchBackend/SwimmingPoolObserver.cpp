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

void SwimmingPoolObserver::processBlobs(size_t frameOrd, const vector<DetectedBlob>& blobs, int* pReadyFrameInd)
{
	setBlobs(frameOrd, blobs);

	auto fps = blobTracker_->getFps();
	auto elapsedTimeMs = 1000.0f / fps;
	int readyFrameInd = -1;
	vector<TrackChangePerFrame> trackChangeList;
	blobTracker_->trackBlobs((int)frameOrd, blobs, fps, elapsedTimeMs, readyFrameInd, trackChangeList);

	if (pReadyFrameInd != nullptr)
		*pReadyFrameInd = readyFrameInd;

	saveSequentialTrackChanges(trackChangeList);
}

void SwimmingPoolObserver::flushTrackHypothesis(int frameInd)
{
	vector<TrackHypothesisTreeNode*> bestTrackLeaves;
	bool isBestTrackLeafsInitied = false;
	int readyFrameInd = -1;
	vector<TrackChangePerFrame> trackChanges;
	int pruneWindow = -1;
	
	for (bool continue1 = true; continue1; pruneWindow--)
	{
		trackChanges.clear();
		continue1 = blobTracker_->flushTrackHypothesis(frameInd, readyFrameInd, trackChanges, bestTrackLeaves, isBestTrackLeafsInitied, pruneWindow);

		CV_Assert(readyFrameInd != -1 && "Must be track changes gathered from the hypothesis tree");

		// bestTrackLeaves is initialized on the first call
		if (!isBestTrackLeafsInitied)
			isBestTrackLeafsInitied = true;

		saveSequentialTrackChanges(trackChanges);
		
		if (pruneWindow == 0)
			bestTrackLeaves.clear(); // all leaves become invalid pointers

#if PW_DEBUG_DETAIL
		std::stringstream bld;
		bld <<"flush";
		bld.fill('0');
		bld.width(4);
		bld << readyFrameInd;
		blobTracker_->logVisualHypothesisTree(frameInd, bld.str(), bestTrackLeaves);
#endif
	}
}

void SwimmingPoolObserver::toString(stringstream& bld)
{
	bld << "framesCount=" << blobsPerFrame_.size() << std::endl;
	bld << "trackChanges=" << trackIdToHistory_.size() << std::endl;
}

void SwimmingPoolObserver::dumpTrackHistory(stringstream& bld) const
{
	bld << "TracksCount=" << trackIdToHistory_.size() << std::endl;
	for (const auto& trackIdHistPair : trackIdToHistory_)
	{
		int trackId = trackIdHistPair.first;
		const TrackInfoHistory& hist = trackIdHistPair.second;
		bld << "TrackId=" << trackId;

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
			bld << "  frameInd=" << frameInd;

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

void SwimmingPoolObserver::saveSequentialTrackChanges(const vector<TrackChangePerFrame>& trackChanges)
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
			track.FirstAppearanceFrameIdx = change.FrameInd;
			track.LastAppearanceFrameIdx = TrackInfoHistory::IndexNull;
			trackIdToHistory_.insert(make_pair(trackCandidateId, track));
		}

		TrackInfoHistory& trackHistory = trackIdToHistory_[trackCandidateId];
		auto localIndex = change.FrameInd - trackHistory.FirstAppearanceFrameIdx;

		// finish track if it is pruned

		if (change.UpdateType == TrackChangeUpdateType::Pruned)
		{
			trackHistory.LastAppearanceFrameIdx = change.FrameInd;

			// remove short ('noisy') tracks

			int framesDuration = trackHistory.LastAppearanceFrameIdx - trackHistory.FirstAppearanceFrameIdx + 1;
			if (framesDuration < trackMinDurationFrames_)
			{
				trackIdToHistory_.erase(trackCandidateId);
				continue;
			}
		}

		// allocate space for track info in corresponding frame

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

		// allow consequent 'put' requests only
		// this condition always true if for each frame the hypothesis tree grows with 
		// 'correspondence' or 'no observation' hypothesis

		CV_Assert(localIndex == trackHistory.Assignments.size() - 1 && "Modification must be applied to the last element of track history");
		trackHistory.Assignments[localIndex] = change;

		if (trackHistory.isFinished())
		{
			CV_Assert(trackHistory.LastAppearanceFrameIdx == trackHistory.FirstAppearanceFrameIdx + trackHistory.Assignments.size() - 1 && "FrameInd of the last appearance must be the last in the assignments array");
		}
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

int SwimmingPoolObserver::toLocalAssignmentIndex(const TrackInfoHistory& trackHist, int frameInd) const
{
	int maxUpper = trackHist.isFinished() ? trackHist.LastAppearanceFrameIdx : (trackHist.FirstAppearanceFrameIdx + trackHist.Assignments.size() - 1);

	int localAssignIndex = frameInd;
	if (localAssignIndex > maxUpper)
		localAssignIndex = -1;
	else if (localAssignIndex < trackHist.FirstAppearanceFrameIdx)
		localAssignIndex = -1;
	return localAssignIndex;
}

int SwimmingPoolObserver::trackHistoryCount() const
{
	return (int)trackIdToHistory_.size();
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

const TrackInfoHistory* SwimmingPoolObserver::trackHistoryForBlob(int frameInd, int blobInd)
{
	for (const auto& trackIdToTrackHist : trackIdToHistory_)
	{
		int trackId = trackIdToTrackHist.first;
		const TrackInfoHistory& trackHist = trackIdToTrackHist.second;


		int assignInd = toLocalAssignmentIndex(trackHist, frameInd);
		if (assignInd == -1)
			return nullptr;

		const auto& assign = trackHist.Assignments[assignInd];
		if (assign.ObservationInd == blobInd)
			return &trackHist;
	}
	
	return nullptr;
}

std::shared_ptr<CameraProjectorBase> SwimmingPoolObserver::cameraProjector()
{
	return cameraProjector_;
}