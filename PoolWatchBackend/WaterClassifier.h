#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2\ml.hpp>

#include "PoolWatchFacade.h"

class PW_EXPORTS WaterClassifier
{
public:
	cv::EM waterMixGauss_;
	cv::EM nonWaterMixGauss_;
	int nClusters_;
	int covMatType_;
	cv::Mat cacheL;
	cv::Mat cacheLFloat;
	//cv::Mat_<double>& oneColor;
	//
	std::vector<float> invCovs1;
	std::vector<float> invCovs2;
	std::vector<double> invCovsDbl1;
	std::vector<double> invCovsDbl2;
	std::vector<float> meanFloats1;
	std::vector<float> meanFloats2;
	std::vector<float> logWeights1;
	std::vector<float> logWeights2;
public:
	WaterClassifier(int nClusters, int covMatType, const cv::TermCriteria& termWater, const cv::TermCriteria& termNonWater);
	WaterClassifier(int nClusters, int covMatType);
private:
	WaterClassifier();
public:
	WaterClassifier(WaterClassifier const& wc) = delete;
	virtual ~WaterClassifier();
	void trainWater(const cv::Mat_<double>& waterColors, const cv::Mat_<double>& nonWaterColors);
	void trainOne(const cv::Mat_<double>& colors, bool isFirst);
	bool predict(const cv::Vec3d& pix);
	bool predictFloat(const cv::Vec3f& pix) const;
	double computeOne(const cv::Vec3d& pix, bool isFirst) const;

	void write(cv::FileStorage& fs);
	static std::unique_ptr<WaterClassifier> read(cv::FileStorage& fs);

public:
	void initCache();
};

