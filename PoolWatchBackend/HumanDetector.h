#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include "MultiHypothesisBlobTracker.h"
#include "WaterClassifier.h"

__declspec(dllexport) void getHumanBodies(const cv::Mat& image, const cv::Mat_<uchar>& waterMask, const std::vector<DetectedBlob>& expectedBlobs, std::vector<DetectedBlob>& blobs);

// Merges two blobs by drawing 'drawBridgeCount' lines between closest points between two blobs.
// 'paintBuffer' is a buffer to draw on.
// Precondition: both blobs are disconnected and are drawn in a buffer.
__declspec(dllexport) void mergeBlobs(const std::vector<cv::Point>& blobPoints1, const std::vector<cv::Point>& blobPoints2, cv::Mat& paintBuffer, std::vector<cv::Point>& mergedBlob, int drawBridgeCount);

class __declspec(dllexport) SwimmerDetector
{
	std::unique_ptr<WaterClassifier> fleshClassifier_;
	std::unique_ptr<WaterClassifier> laneSeparatorClassifier_;
public:
	SwimmerDetector();
	void getBlobs(const cv::Mat& image, const std::vector<DetectedBlob>& expectedBlobs, std::vector<DetectedBlob>& blobs);
private:
	void trainLaneSeparatorClassifier(WaterClassifier& wc);
	void trainSkinClassifier(WaterClassifier& wc);
};