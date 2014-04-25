#include <cassert>
#include <array>
#include <opencv2/ml.hpp>

#include <ppl.h>

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
	//initCache();
}

WaterClassifier::WaterClassifier()
{
	// initialized on read
}

void WaterClassifier::initCache()
{
	cacheL.create(1, nClusters_, CV_64FC1);
	cacheLFloat.create(1, nClusters_, CV_32FC1);

	// cache EM's double internals as floats
	auto& c1 = static_cast<EMQuick&>(waterMixGauss_);
	auto& c2 = static_cast<EMQuick&>(nonWaterMixGauss_);

	invCovs1.resize(nClusters_);
	invCovs2.resize(nClusters_);
	invCovsDbl1.resize(nClusters_);
	invCovsDbl2.resize(nClusters_);
	for (size_t i = 0; i < nClusters_; ++i)
	{
		invCovs1[i] = (float)c1.getInvCovsEigenValuesPar()[i].at<double>(0);
		invCovs2[i] = (float)c2.getInvCovsEigenValuesPar()[i].at<double>(0);
		invCovsDbl1[i] = c1.getInvCovsEigenValuesPar()[i].at<double>(0);
		invCovsDbl2[i] = c2.getInvCovsEigenValuesPar()[i].at<double>(0);
	}

	meanFloats1.resize(nClusters_ * 3);
	meanFloats2.resize(nClusters_ * 3);
	for (size_t i = 0; i < meanFloats1.size(); ++i)
	{
		meanFloats1[i] = (float)c1.getMeans().at<double>(i);
		meanFloats2[i] = (float)c2.getMeans().at<double>(i);
	}

	logWeights1.resize(nClusters_);
	logWeights2.resize(nClusters_);
	for (size_t i = 0; i < logWeights1.size(); ++i)
	{
		logWeights1[i] = (float)c1.getLogWeightDivDetPar().at<double>(i);
		logWeights2[i] = (float)c2.getLogWeightDivDetPar().at<double>(i);
	}
}


WaterClassifier::~WaterClassifier()
{
}

void WaterClassifier::trainWater(const cv::Mat_<double>& waterColors, const cv::Mat_<double>& nonWaterColors)
{
	CV_Assert(!waterColors.empty());
	CV_Assert(!nonWaterColors.empty());

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

	double logProb1 = computeGaussMixtureModel(nClusters_, (double*)c1.getMeans().data, &invCovsDbl1[0], (double*)c1.getLogWeightDivDetPar().data, (double*)&pix[0], (double*)cacheL.data);
	double logProb2 = computeGaussMixtureModel(nClusters_, (double*)c2.getMeans().data, &invCovsDbl2[0], (double*)c2.getLogWeightDivDetPar().data, (double*)&pix[0], (double*)cacheL.data);
	bool label2 = logProb1 > logProb2 ? true : false;
	//CV_Assert(label == label2);

	return label2;
}

double WaterClassifier::computeOne(const cv::Vec3d& pix, bool isFirst)
{
	auto& c1 = static_cast<EMQuick&>(waterMixGauss_);
	auto& c2 = static_cast<EMQuick&>(nonWaterMixGauss_);
	
	EMQuick& cc = isFirst ? c1 : c2;

	double logProb = computeGaussMixtureModel(nClusters_, (double*)cc.getMeans().data, &invCovsDbl1[0], (double*)cc.getLogWeightDivDetPar().data, (double*)&pix[0], (double*)cacheL.data);
	return logProb;
}

bool WaterClassifier::predictFloat(const cv::Vec3f& pix)
{
	float logProb1;
	auto probFun1 = [&]()
	{
		logProb1 = computeGaussMixtureModelGen<float>(nClusters_, &meanFloats1[0], &invCovs1[0], &logWeights1[0], (float*)&pix[0], (float*)cacheL.data);
	};

	float logProb2;
	cv::Vec3f pixCopy = pix;
	auto probFun2 = [&]()
	{
		logProb2 = computeGaussMixtureModelGen<float>(nClusters_, &meanFloats2[0], &invCovs2[0], &logWeights2[0], (float*)&pixCopy[0], (float*)cacheLFloat.data);
	};

	Concurrency::parallel_invoke(probFun1, probFun2);	
	
	bool label2 = logProb1 > logProb2 ? true : false;
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
