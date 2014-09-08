#include "MultiHypothesisBlobTracker.h"

void toString(TrackChangeUpdateType trackChange, std::string& result)
{
	switch (trackChange)
	{
	case TrackChangeUpdateType::New:
		result = "new";
		break;
	case TrackChangeUpdateType::ObservationUpdate:
		result = "upd";
		break;
	case TrackChangeUpdateType::NoObservation:
		result = "noObs";
		break;
	case TrackChangeUpdateType::Pruned:
		result = "end";
		break;
	default:
		result = "TrackChangeUpdateType";
		break;
	}
}

TrackChangeClientSide::TrackChangeClientSide(int trackId, int frameIndOnNew)
	: TrackId(trackId),
	FrameIndOnNew(frameIndOnNew),
	EndFrameInd(frameIndOnNew)
{
}

void TrackChangeClientSide::setNextTrackChange(int changeFrameInd)
{
	CV_Assert(IsLive && "Can update live track only");

	int nextInd = EndFrameInd;
	CV_Assert(nextInd == changeFrameInd && "Updating track with non sequential change");

	EndFrameInd++;
}

void TrackChangeClientSide::terminate()
{
	CV_Assert(IsLive);
	IsLive = false;
}

const TrackChangePerFrame* TrackInfoHistory::getTrackChangeForFrame(int frameOrd) const
{
	if (frameOrd < FirstAppearanceFrameIdx)
		return nullptr;
	if (isFinished() && frameOrd > LastAppearanceFrameIdx)
		return nullptr;

	int localInd = frameOrd - FirstAppearanceFrameIdx;
	assert(localInd >= 0);
	assert(localInd < Assignments.size());

	return &Assignments[localInd];
}

bool TrackInfoHistory::isFinished() const
{
	return LastAppearanceFrameIdx != IndexNull;
}
