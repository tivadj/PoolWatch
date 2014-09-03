#pragma once

#include <vector>
#include <map>
#include <array>
#include <QObject>
#include <QPointF>
#include <QSize>

#include "SwimmingPoolObserver.h"
#include "CoreUtils.h"

struct BlobRegistrationRecord
{
	cv::Point2f ObsPosPixExactOrApprox;
	bool HasObservation;
};

struct TrackUIItem
{
	int TrackId;
	int FrameIndOnNew;
	int PositionsCount = 0;
	bool IsLive = true;
	BlobRegistrationRecord LastPosition;
	//std::vector<BlobRegistrationRecord> Positions;

	void setNextRegistration(int changeFrameInd, const BlobRegistrationRecord& trackPosInfo);
	int latesetFrameIndExcl() const;
	void setDead();
};

struct BugCreature
{
	static const int MOVEMENT_MODEL_CIRCLE = 1;
	static const int MOVEMENT_MODEL_FOUR_PETALS = 2;

	bool Visible = true;
	QPointF Pos;
	std::array<uchar, 3> PrimaryColor;

	// movement model
	int movementModel_ = MOVEMENT_MODEL_CIRCLE;
	float teta_ = 0;
	float radius_ = 0;
};

class TracingDemoModel : public QObject
{
	Q_OBJECT
public:
	TracingDemoModel();
	~TracingDemoModel();
	void updateBugsPositions(std::vector<BugCreature>& bugs);

	// assigns one blob for each bug
	void populateDetectedBlobs_CentroidPerNeighbourBugsGroup(const std::vector<BugCreature>& bugs, std::vector<DetectedBlob>& resultBlobs);
	void fixBlobFilledImageRgb(const BugCreature& bug, int blobWidth, int blobHeight, DetectedBlob& blob);
	void populateBlobRegistration(TrackChangePerFrame const& change, BlobRegistrationRecord& result);

	void nextGameStep(bool updateBugTeta);
	void onGotTrackChanges(int readyFrameInd, const std::vector<TrackChangePerFrame>& trackChanges);

	float interval() const;
	TrackUIItem* getTrack(int trackId);
	void getTrackList(std::vector<TrackUIItem*>& result);
	bool getLatestTrack(int frameInd, int trackFamilyId, TrackChangePerFrame& outTrackChange);
	void switchBugLiveStatus(int bugInd);
	void setEnableTracking(bool enableTracking);
	void setImageSize(QSize imageSize);
	void printStatistics();
signals:
	void changed();
	void trackAdded(int trackId);
	void trackPruned(int trackId);

private:
	QSize imageSize_;
	float tetaStep_ = -1;
	std::shared_ptr<MultiHypothesisBlobTracker> blobTracker_;

	float maxShift_ = 0;
	std::map<int, TrackUIItem> trackIdToObj_;
	bool enableTracking_ = true;
public:
	bool showCurrentBlobs_ = true;
	int pruneWindow_ = -1;
	int frameInd_ = -1;
	int readyFrameInd_ = -1;
	std::unique_ptr<PoolWatch::CyclicHistoryBuffer<std::vector<BugCreature>>> bugsHistory_;
	std::unique_ptr<PoolWatch::CyclicHistoryBuffer<std::vector<DetectedBlob>>> blobsHistory_;
};

