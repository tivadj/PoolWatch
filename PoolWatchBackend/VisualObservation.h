#pragma once
#include <array>
#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>

#include <log4cxx/logger.h>

#include "PoolWatchFacade.h"
#include "VisualObservationIf.h"
#include "SwimmerDetector.h"
#include "CameraProjector.h"
#include "AppearanceModel.h"
#include "WaterClassifier.h"

PW_EXPORTS void getPoolMask(const cv::Mat& image, const cv::Mat_<uchar>& waterMask, cv::Mat_<uchar>& poolMask);
PW_EXPORTS std::tuple<bool, std::string> getSwimLanes(const cv::Mat& image);

// Updates world coordinates of each blob.
PW_EXPORTS void fixBlobs(std::vector<DetectedBlob>& blobs, const CameraProjectorBase& cameraProjector);

PW_EXPORTS void getHumanBodies(const cv::Mat& image, const cv::Mat_<uchar>& waterMask, const std::vector<DetectedBlob>& expectedBlobs, std::vector<DetectedBlob>& blobs);

// Merges two blobs by drawing 'drawBridgeCount' lines between closest points between two blobs.
// 'paintBuffer' is a buffer to draw on.
// Precondition: both blobs are disconnected and are drawn in a buffer.
PW_EXPORTS void mergeBlobs(const std::vector<cv::Point>& blobPoints1, const std::vector<cv::Point>& blobPoints2, cv::Mat& paintBuffer, std::vector<cv::Point>& mergedBlob, int drawBridgeCount);

class PW_EXPORTS SwimmerDetector
{
	static log4cxx::LoggerPtr log_;
	std::unique_ptr<WaterClassifier> fleshClassifier_;
	std::unique_ptr<WaterClassifier> laneSeparatorClassifier_;
	std::shared_ptr<CameraProjectorBase> cameraProjector_;
public:
	float bodyHalfLenth_;
	float bodyHalfLenthPix_; // used for testing
public:
	SwimmerDetector(const std::shared_ptr<CameraProjectorBase> cameraProjector);
	SwimmerDetector(int bodyHalfLenthPix); // used for testing
	void getBlobs(const cv::Mat& image, const std::vector<DetectedBlob>& expectedBlobs, std::vector<DetectedBlob>& blobs);

	// Calculates GMM color signature from blob's RGB image
	static void fixColorSignature(const cv::Mat& blobImageRgb, const cv::Mat& blobImageMask, cv::Vec3b transparentCol, DetectedBlob& resultBlob);
private:
	void getHumanBodies(const cv::Mat& image, const cv::Mat& imageMask, const cv::Mat_<uchar>& waterMask, const std::vector<DetectedBlob>& expectedBlobs, std::vector<DetectedBlob>& blobs);
	void trainLaneSeparatorClassifier(WaterClassifier& wc);
	void trainSkinClassifier(WaterClassifier& wc);
};