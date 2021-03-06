#include <cassert>
#include <iostream>
#include <stdint.h>

#include <opencv2\highgui\highgui_c.h>
#include <opencv2\imgproc.hpp> // cv::threshold

#include <boost/lexical_cast.hpp>

#include "SwimmingPoolObserver.h"
#include "PoolWatchFacade.h"
#include "algos1.h"
#include "PaintHelper.h"
#include "VisualObservation.h"

using namespace std;

log4cxx::LoggerPtr SwimmingPoolObserver::log_ = log4cxx::Logger::getLogger("PW.SwimmingPoolObserver");

SwimmingPoolObserver::SwimmingPoolObserver(unique_ptr<MultiHypothesisBlobTracker> blobTracker, shared_ptr<CameraProjectorBase> cameraProjector)
: cameraProjector_(cameraProjector),
  swimmerDetector_(cameraProjector)
{
	blobTracker_.swap(blobTracker);
}

std::tuple<bool, std::string> SwimmingPoolObserver::init()
{
	// init water classifier

	cv::FileStorage fs;
	if (!fs.open("cl_water.yml", cv::FileStorage::READ))
	{
		return make_tuple(false, "Can't find file '1.yml' (change working directory)");
	}
	waterClassifier_ = WaterClassifier::read(fs);

	//
	if (!fs.open("cl_reflectedLight.yml", cv::FileStorage::READ))
	{
		return make_tuple(false, "Can't find file 'reflectedLight_clasifier.yml ' (change working directory)");
	}
	reflectedLightClassifier_ = WaterClassifier::read(fs);	

	return make_tuple(true, "");
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

void SwimmingPoolObserver::processCameraImage(size_t frameOrd, const cv::Mat& image, int* pReadyFrameInd)
{
	std::vector<DetectedBlob> blobs;
	cv::Mat imageBlobsDebug;
	//getHumanBodies(imageFamePoolOnly, waterMask, expectedBlobs, blobs);
	//swimmerDetector_.getBlobsSkinColor(image, expectedBlobs_, *waterClassifier_, blobs, imageBlobsDebug);
	swimmerDetector_.getBlobsSubtractive(image, expectedBlobs_, *waterClassifier_, *reflectedLightClassifier_, blobs, imageBlobsDebug);

	BlobsDetected(blobs, imageBlobsDebug);

	processBlobs(frameOrd, blobs, pReadyFrameInd);

	// predict position of next blobs

	expectedBlobs_.clear();
	predictNextFrameBlobs(frameOrd, blobs, expectedBlobs_);
}

void SwimmingPoolObserver::processBlobs(size_t frameOrd, const vector<DetectedBlob>& blobs, int* pReadyFrameInd)
{
	setBlobs(frameOrd, blobs);

	int readyFrameInd = -1;
	vector<TrackChangePerFrame> trackChangeList;
	blobTracker_->trackBlobs((int)frameOrd, blobs, readyFrameInd, trackChangeList);

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

#if LOG_VISUAL_HYPOTHESIS_TREE
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
			bld << " Score=" << change.Score;
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

		PoolWatch::PaintHelper::paintTrack(track, fromFrameOrd, toFrameOrd, color, blobsPerFrame_, resultImage);
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


// frameOrd is the current frame. Method computes blobs for consequent frame (frameInd+1).
void SwimmingPoolObserver::predictNextFrameBlobs(int frameOrd, const std::vector<DetectedBlob>& blobs, std::vector<DetectedBlob>& expectedBlobs)
{
	std::vector<TrackHypothesisTreeNode*> hypList;
	blobTracker_->getMostPossibleHypothesis(frameOrd, hypList);

	TrackHypothesisTreeNode dummyNode;
	for (TrackHypothesisTreeNode* pNode : hypList)
	{
		// enumerate only hypothesis based on observation
		if (pNode->CreationReason == TrackHypothesisCreationReason::NoObservation)
			continue;

		const std::vector<DetectedBlob>& curBlobs = blobsPerFrame_[frameOrd];
		DetectedBlob blob = curBlobs[pNode->ObservationInd];
		
		cv::Point3f estPosWorld;
		float score;
		blobTracker_->movementPredictor().estimateAndSave(*pNode, nullptr, estPosWorld, score, dummyNode);

		cv::Point2f estPos = cameraProjector_->worldToCamera(estPosWorld);

		float shiftX = estPos.x - blob.Centroid.x;
		float shiftY = estPos.y - blob.Centroid.y;

		blob.Centroid = cv::Point2f(estPos.x, estPos.y);
		auto bnd = blob.BoundingBox;
		bnd.x += shiftX;
		bnd.y += shiftY;

		// TODO: limit movement or truncate to image bounds
		const int W = 640;
		const int H = 480;
		if (bnd.x < 0)
			bnd.x = 0;
		if (bnd.br().x >= W)
			bnd.x = W - 1 -bnd.width;
		
		if (bnd.y < 0)
			bnd.y = 0;
		if (bnd.br().y >= H)
			bnd.y = H - 1 - bnd.height;

		blob.BoundingBox = bnd;
		// invalidate unused data fields
		blob.AreaPix = -1;
		blob.CentroidWorld = estPosWorld;
		blob.OutlinePixels = cv::Mat_<int32_t>();
		expectedBlobs.push_back(blob);
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

const TrackInfoHistory* SwimmingPoolObserver::trackHistoryForBlob(int frameInd, int blobInd) const
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