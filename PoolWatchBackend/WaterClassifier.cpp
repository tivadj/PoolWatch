#include <cassert>
#include <array>
#include <opencv2/ml.hpp>

#include "WaterClassifier.h"
#include "algos_amp.cpp"

using namespace std;
using namespace cv;



WaterClassifier::WaterClassifier(int nClusters, int covMatType)
:nClusters_(nClusters),
covMatType_(covMatType)
{
	waterMixGauss_ = cv::EM(nClusters, covMatType);
	nonWaterMixGauss_ = cv::EM(nClusters, covMatType);
	initCache();
}

WaterClassifier::WaterClassifier()
{
	// initialized on read
}

void WaterClassifier::initCache()
{
	cacheL.create(1, nClusters_, CV_64FC1);
}


WaterClassifier::~WaterClassifier()
{
}

void WaterClassifier::trainWater(const cv::Mat_<double>& waterColors, const cv::Mat_<double>& nonWaterColors)
{
	bool success = waterMixGauss_.train(waterColors);
	assert(success);

	bool success2 = nonWaterMixGauss_.train(nonWaterColors);
	assert(success2);
}

bool WaterClassifier::predict(const cv::Vec3d& pix)
{
	//cv::Mat_<double> oneColor(1,3,CV_64FC1);

	auto& c1 = static_cast<EMQuick&>(waterMixGauss_);
	auto& c2 = static_cast<EMQuick&>(nonWaterMixGauss_);

	//cv::Vec2d p1 = MyEM::predict2(oneColor, c1.getMeans(), c1.getInvCovsEigenValuesPar(), c1.getLogWeightDivDetPar(), cacheL);
	//cv::Vec2d p2 = MyEM::predict2(oneColor, c2.getMeans(), c2.getInvCovsEigenValuesPar(), c2.getLogWeightDivDetPar(), cacheL);

	//bool label;
	//if (p1[0] < p2[0]) // first is closer => result is water
	//	label = true;
	//else
	//	label = false;

	const int nclusters = 6;

	std::array<double, 6> invCovs1;
	std::array<double, 6> invCovs2;
	for (int i = 0; i < nclusters; ++i)
	{
		invCovs1[i] = c1.getInvCovsEigenValuesPar()[i].at<double>(0);
		invCovs2[i] = c2.getInvCovsEigenValuesPar()[i].at<double>(0);
	}

	double logProb1 = computeGaussMixtureModel(nclusters, (double*)c1.getMeans().data, &invCovs1[0], (double*)c1.getLogWeightDivDetPar().data, (double*)&pix[0], (double*)cacheL.data);
	double logProb2 = computeGaussMixtureModel(nclusters, (double*)c2.getMeans().data, &invCovs2[0], (double*)c2.getLogWeightDivDetPar().data, (double*)&pix[0], (double*)cacheL.data);
	bool label2 = logProb1 > logProb2 ? true : false;
	//CV_Assert(label == label2);

	return label2;
}

const char* waterEMStr = "waterEM";
const char* nonWaterEMStr = "nonWaterEM";

void WaterClassifier::write(cv::FileStorage& fs)
{
	fs << waterEMStr << "{";
	auto& waterEM = static_cast<EMQuick&>(waterMixGauss_);
	((Algorithm&)waterMixGauss_).write(fs);
	fs << "}";

	fs << nonWaterEMStr << "{";
	auto& nonWaterEM = static_cast<EMQuick&>(nonWaterMixGauss_);
	((Algorithm&)nonWaterEM).write(fs);
	fs << "}";
}

std::unique_ptr<WaterClassifier> WaterClassifier::read(cv::FileStorage& fs)
{
	auto result = std::unique_ptr<WaterClassifier>(new WaterClassifier());
	WaterClassifier& wc = *result;

	auto& waterEM = static_cast<EMQuick&>(wc.waterMixGauss_);
	wc.waterMixGauss_.read(fs[waterEMStr]);

	auto& nonWaterEM = static_cast<EMQuick&>(wc.nonWaterMixGauss_);
	wc.nonWaterMixGauss_.read(fs[nonWaterEMStr]);

	wc.nClusters_ = waterEM.get<int>("nclusters");
	wc.covMatType_ = waterEM.get<int>("covMatType");

	assert(wc.nClusters_ == nonWaterEM.get<int>("nclusters"));
	assert(wc.covMatType_ == nonWaterEM.get<int>("covMatType"));

	wc.initCache();

	return result;
}
