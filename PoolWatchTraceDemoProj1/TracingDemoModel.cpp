#include "TracingDemoModel.h"
#include <sstream>
#include <QDebug>

#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include "algos1.h"
#include "DemoHelpers.h"
#include <log4cxx/logger.h>
#include <KalmanFilterMovementPredictor.h>

using namespace log4cxx;

log4cxx::LoggerPtr log_ = log4cxx::Logger::getLogger("TracingDemoModel");

void TrackUIItem::setNextRegistration(int changeFrameInd, BlobRegistrationRecord const& trackPosInfo)
{
	auto nextInd = latesetFrameIndExcl();
	CV_Assert(nextInd == changeFrameInd);

	LastPosition = trackPosInfo;
	PositionsCount++;
}

int TrackUIItem::latesetFrameIndExcl() const
{
	auto result = FrameIndOnNew + PositionsCount;
	return result;
}

void TrackUIItem::setDead()
{
	CV_Assert(IsLive);
	IsLive = false;
}

//BlobRegistrationRecord TrackUIItem::getReadyRegistration(int readyFrameInd)
//{
//	return {};
//}

TracingDemoModel::TracingDemoModel()
{
	tetaStep_ = PoolWatch::deg2rad(3);

	//auto size = ui->glWidgetTrace->size();
	//auto size = QSize(320, 200);
	auto size = QSize(640, 480);
	qDebug() << "TopViewDemoTracerMainWindow::resizeEvent (" << size.width() << ", " << size.height() << ")";
	setImageSize(size);

	// init tracker

	auto cameraProjector = std::make_shared<LinearCameraProjector>();

	// Evaluated experimentally. This distance must be somewhat greater than evaluated exact maximum distance between consequent blob's positions
	// because tracker uses inexact Estimated track position.
	const float maxDistPerFrame = 22 * 2.5f;

	auto movementModel = std::make_unique<KalmanFilterMovementPredictor>(maxDistPerFrame);
	auto appearanceModel = std::make_unique<SwimmerAppearanceModel>();

	pruneWindow_ = 3;
	blobTracker_ = std::make_shared<MultiHypothesisBlobTracker>(pruneWindow_, cameraProjector, std::move(movementModel), std::move(appearanceModel));

	// init bugs
	bugsHistory_ = std::make_unique<PoolWatch::CyclicHistoryBuffer<std::vector<BugCreature>>>(pruneWindow_ + 1);
	blobsHistory_ = std::make_unique<PoolWatch::CyclicHistoryBuffer<std::vector<DetectedBlob>>>(pruneWindow_ + 1);

	//
	auto& bugs = bugsHistory_->requestNew();
	bugs.resize(2);

	std::vector<std::array<uchar, 3>> colors;
	colors.push_back(std::array < uchar, 3 > {75, 73, 159}); // blue
	colors.push_back(std::array < uchar, 3 > {163, 79, 126}); // pink
	colors.push_back(std::array < uchar, 3 > {163, 151, 79}); // brown
	colors.push_back(std::array < uchar, 3 > {73, 159, 97}); // green
	//colors.push_back(std::array < uchar, 3 > {116, 146, 191});
	//colors.push_back(std::array < uchar, 3 > {231, 176, 82});
	//colors.push_back(std::array < uchar, 3 > {180, 202, 118});

	const float initialTeta = PoolWatch::deg2rad(185);
	float teta = -1;
	for (size_t i = 0; i < bugs.size(); ++i)
	{
		int movementModel;
		float rad = std::min(imageSize_.width(), imageSize_.height()) / 2;
		rad *= 0.9; // prevent being close to the viewport borders
		if (i % 2 == 0)
		{
			// updates angle
			if (teta == -1)
				teta = initialTeta;
			else
				teta += PoolWatch::deg2rad(rand() % 180);

			movementModel = BugCreature::MOVEMENT_MODEL_CIRCLE;
		}
		else
		{
			// keep teta the same
			
			//
			movementModel = BugCreature::MOVEMENT_MODEL_FOUR_PETALS;
		}

		bugs[i].movementModel_ = movementModel;
		bugs[i].teta_ = teta;		
		bugs[i].radius_ = rad;
		
		auto& color = colors[i % colors.size()];
		bugs[i].PrimaryColor = color;
	}
	nextGameStep(false);
}

TracingDemoModel::~TracingDemoModel()
{
}

void TracingDemoModel::updateBugsPositions(std::vector<BugCreature>& bugs)
{
	float len = std::min(imageSize_.width(), imageSize_.height());

	for (auto& bug : bugs)
	{
		float x = 0;
		float y = 0;

		switch(bug.movementModel_)
		{
		case BugCreature::MOVEMENT_MODEL_CIRCLE:
		{
			x = len / 2;
			y = len / 2;
			
			x += bug.radius_ * std::cosf(bug.teta_);
			y -= bug.radius_ * std::sinf(bug.teta_);
			
			break;
		}
		case BugCreature::MOVEMENT_MODEL_FOUR_PETALS:
		{
			x = len / 2;
			y = len / 2;

			float coef = std::sinf(2*bug.teta_);
			x += bug.radius_ * coef * std::cosf(bug.teta_);
			y -= bug.radius_ * coef * std::sinf(bug.teta_);
			
			break;
		}
		}


		bool hasPrevFrame = frameInd_ > 0;
		if (hasPrevFrame)
		{
			// calculate max shift
			float len = (bug.Pos.x() - x)*(bug.Pos.x() - x) + (bug.Pos.y() - y)*(bug.Pos.y() - y);
			len = std::sqrtf(len);
			if (len > maxShift_)
				maxShift_ = len;
		}

		bug.Pos = QPointF(x, y);
	}
}

void TracingDemoModel::populateDetectedBlobs_CentroidPerNeighbourBugsGroup(const std::vector<BugCreature>& bugs, std::vector<DetectedBlob>& resultBlobs)
{
	const float CloseBlobsDist = 7;
	std::vector<uchar> processedBugs(bugs.size(), false);
	for (size_t i = 0; i < bugs.size(); ++i)
	{
		if (processedBugs[i])
			continue;
		processedBugs[i] = true;

		auto& bug = bugs[i];

		float centroidX = bug.Pos.x();
		float centroidY = bug.Pos.y();
		int neighboursCount = 1;

		for (size_t j = i + 1; j < bugs.size(); ++j)
		{
			if (processedBugs[j])
				continue;

			auto& nghBug = bugs[j];

			float len = PoolWatch::sqr(nghBug.Pos.x() - bug.Pos.x()) + PoolWatch::sqr(nghBug.Pos.y() - bug.Pos.y());
			len = std::sqrtf(len);
			if (len < CloseBlobsDist)
			{
				centroidX += nghBug.Pos.x();
				centroidY += nghBug.Pos.y();
				neighboursCount++;
				processedBugs[j] = true;
			}
		}

		centroidX /= neighboursCount;
		centroidY /= neighboursCount;

		const int blobW = 10;
		const int blobH = 10;
		DetectedBlob blob;
		blob.Id = i + 1;
		blob.Centroid = cv::Point2f(centroidX,centroidY);
		blob.CentroidWorld = cv::Point3f(centroidX, centroidY, 0);
		blob.BoundingBox = cv::Rect2f(centroidX - 5, centroidY-5, blobW,blobH);
		blob.FilledImage = cv::Mat(blobW, blobH, CV_8UC1);
		blob.FilledImage.setTo(255);
		fixBlobFilledImageRgb(bug, blobW, blobH, blob);

		blob.AreaPix = blobW * blobH;
		resultBlobs.push_back(blob);
	}

	if (log_->isDebugEnabled())
	{
		std::stringstream bld;
		bld << "Found " << resultBlobs.size() << " blobs" << std::endl;
		for (const auto& blob : resultBlobs)
			bld << "  Id=" << blob.Id << " Centroid=" << blob.Centroid << std::endl;
		LOG4CXX_DEBUG(log_, bld.str());
	}
}

void TracingDemoModel::fixBlobFilledImageRgb(const BugCreature& bug, int blobWidth, int blobHeight, DetectedBlob& blob)
{
	cv::Mat blobImgRgb(blobWidth * blobHeight, 3, CV_8UC1);
	for (int i = 0; i < blobImgRgb.rows; ++i)
	{
		const int ColorDiff = 20;
		std::array<uchar, 3> col;
		col[0] = bug.PrimaryColor[0] + (rand() % ColorDiff) - ColorDiff / 2;
		col[1] = bug.PrimaryColor[1] + (rand() % ColorDiff) - ColorDiff / 2;
		col[2] = bug.PrimaryColor[2] + (rand() % ColorDiff) - ColorDiff / 2;

		blobImgRgb.at<uchar>(i, 0) = col[0];
		blobImgRgb.at<uchar>(i, 1) = col[1];
		blobImgRgb.at<uchar>(i, 2) = col[2];
	}
	blobImgRgb = blobImgRgb.reshape(3, blobHeight);

	// generates RGB image
	blob.FilledImageRgb = blobImgRgb;
	SwimmerDetector::fixColorSignature(blobImgRgb, blob.FilledImage, cv::Vec3b(0, 0, 0), blob);
}

void TracingDemoModel::nextGameStep(bool updateBugTeta)
{
	frameInd_++;
	LOG4CXX_DEBUG(log_, "frameInd=" << frameInd_);

	if (updateBugTeta)
	{
		auto& bugsNext = bugsHistory_->requestNew();
		bugsNext.clear();

		auto& bugsPrev = bugsHistory_->queryHistory(-1);
		std::copy(std::begin(bugsPrev), std::end(bugsPrev), std::back_inserter(bugsNext));

		for (auto& bug : bugsNext)
			bug.teta_ += tetaStep_;
	}

	auto& bugsNext = bugsHistory_->queryHistory(0);

	updateBugsPositions(bugsNext);

	auto& curDetectedBlobs = blobsHistory_->requestNew();
	curDetectedBlobs.clear();
	populateDetectedBlobs_CentroidPerNeighbourBugsGroup(bugsNext, curDetectedBlobs);

	//
	int readyFrameInd = -1;
	
	std::vector<TrackChangePerFrame> trackChanges;
	blobTracker_->trackBlobs(frameInd_, curDetectedBlobs, readyFrameInd, trackChanges);
	onGotTrackChanges(readyFrameInd, trackChanges);

	emit changed();
}

void TracingDemoModel::onGotTrackChanges(int readyFrameInd, const std::vector<TrackChangePerFrame>& trackChanges)
{
	readyFrameInd_ = readyFrameInd;

	for (const auto& change : trackChanges)
	{
		int trackId = change.FamilyId;

		switch (change.UpdateType)
		{
		case TrackChangeUpdateType::New:
		{
			TrackUIItem newTrack;
			newTrack.TrackId = trackId;
			newTrack.FrameIndOnNew = change.FrameInd;

			BlobRegistrationRecord blobInfo;
			populateBlobRegistration(change, blobInfo);

			newTrack.setNextRegistration(readyFrameInd, blobInfo);
			trackIdToObj_.insert(std::make_pair(trackId, newTrack));

			emit trackAdded(trackId);
			break;
		}
		case TrackChangeUpdateType::NoObservation:
		case TrackChangeUpdateType::ObservationUpdate:
		{
			TrackUIItem* pTrack = getTrack(change.FamilyId);
			CV_Assert(pTrack != nullptr);

			BlobRegistrationRecord blobInfo;
			populateBlobRegistration(change, blobInfo);

			pTrack->setNextRegistration(readyFrameInd, blobInfo);
			break;
		}
		case TrackChangeUpdateType::Pruned:
		{
			TrackUIItem* pTrack = getTrack(change.FamilyId);
			CV_Assert(pTrack != nullptr);

			pTrack->setDead();
			emit trackPruned(trackId);
			break;
		}
		}
	}
}

void TracingDemoModel::populateBlobRegistration(TrackChangePerFrame const& change, BlobRegistrationRecord& result)
{
	result.ObsPosPixExactOrApprox = change.ObservationPosPixExactOrApprox;
	result.HasObservation = change.ObservationInd >= 0;
}

float TracingDemoModel::interval() const
{
	return 30;
}

TrackUIItem* TracingDemoModel::getTrack(int trackId)
{
	auto it = trackIdToObj_.find(trackId);
	if (it == std::end(trackIdToObj_))
		return nullptr;
	return &it->second;
}

void TracingDemoModel::getTrackList(std::vector<TrackUIItem*>& result)
{
	for (auto& pair : trackIdToObj_)
		result.push_back(&pair.second);
}

bool TracingDemoModel::getLatestTrack(int frameInd, int trackFamilyId, TrackChangePerFrame& outTrackChange)
{
	return false;
}

void TracingDemoModel::switchBugLiveStatus(int bugInd)
{
}

void TracingDemoModel::setEnableTracking(bool enableTracking)
{
}

void TracingDemoModel::setImageSize(QSize imageSize)
{
	imageSize_ = imageSize;
}

void TracingDemoModel::printStatistics()
{
	qDebug() << "maxShift=" << maxShift_;
	qDebug() << "tracksCount=" << trackIdToObj_.size();
}
