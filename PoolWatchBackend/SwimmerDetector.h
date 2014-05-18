#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <log4cxx/logger.h>
#include "MultiHypothesisBlobTracker.h"
#include "WaterClassifier.h"

__declspec(dllexport) void getHumanBodies(const cv::Mat& image, const cv::Mat_<uchar>& waterMask, const std::vector<DetectedBlob>& expectedBlobs, std::vector<DetectedBlob>& blobs);

// Merges two blobs by drawing 'drawBridgeCount' lines between closest points between two blobs.
// 'paintBuffer' is a buffer to draw on.
// Precondition: both blobs are disconnected and are drawn in a buffer.
__declspec(dllexport) void mergeBlobs(const std::vector<cv::Point>& blobPoints1, const std::vector<cv::Point>& blobPoints2, cv::Mat& paintBuffer, std::vector<cv::Point>& mergedBlob, int drawBridgeCount);

struct ContourInfo
{
	std::vector<cv::Point> outlinePixels;
	float area;
	bool markDeleted;
	cv::Point2f contourCentroid;
	cv::Point3f contourCentroidWorld;
};

class __declspec(dllexport) SwimmerDetector
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
private:
	void getHumanBodies(const cv::Mat& image, const cv::Mat& imageMask, const cv::Mat_<uchar>& waterMask, const std::vector<DetectedBlob>& expectedBlobs, std::vector<DetectedBlob>& blobs);
	void trainLaneSeparatorClassifier(WaterClassifier& wc);
	void trainSkinClassifier(WaterClassifier& wc);
};