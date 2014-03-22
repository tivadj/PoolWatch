#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2\ml.hpp>

#include "PoolWatchFacade.h"

class POOLWATCH_API WaterClassifier
{
public:
	cv::EM waterMixGauss_;
	cv::EM nonWaterMixGauss_;
	int nClusters_;
	int covMatType_;
	cv::Mat cacheL;
	//cv::Mat_<double>& oneColor;
public:
	WaterClassifier(int nClusters, int covMatType);
private:
	WaterClassifier();
public:
	WaterClassifier(WaterClassifier const& wc) = delete;
	virtual ~WaterClassifier();
	void trainWater(const cv::Mat_<double>& waterColors, const cv::Mat_<double>& nonWaterColors);
	bool predict(const cv::Vec3d& pix);

	void write(cv::FileStorage& fs);
	static std::unique_ptr<WaterClassifier> read(cv::FileStorage& fs);

private:
	void initCache();
};

