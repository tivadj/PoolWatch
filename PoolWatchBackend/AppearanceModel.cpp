#include <array>
#include <algorithm>
#include <numeric> // accumulate
#include <iostream>
#include "AppearanceModel.h"
#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h> // IplImage
#include "algos1.h"
//#include <opencv2/highgui/highgui_c.h> // CV_FOURCC
using namespace cv;

AppearanceModel::AppearanceModel(int numModes)
	:numModes_(numModes)
{
	int width = 1;
	int height = 1;

	//Algorithms::BackgroundSubtraction::GrimsonParams params;
	//params.SetFrameSize(width, height);
	//params.LowThreshold() = 3.0f*3.0f;
	//params.HighThreshold() = 2 * params.LowThreshold();	// Note: high threshold is used by post-processing 
	//params.Alpha() = 0.005f;
	//params.MaxModes() = 2;

	//Algorithms::BackgroundSubtraction::GrimsonGMM bgs;
	//bgs.Initalize(params);


	Algorithms::BackgroundSubtraction::ZivkovicParams params;
	params.SetFrameSize(width, height);
	params.LowThreshold() = 25;
	params.HighThreshold() = 2 * params.LowThreshold();	// Note: high threshold is used by post-processing 
	//params.HighThreshold() = 50;
	params.Alpha() = 0.01f;
	params.MaxModes() = numModes;

	bgs_.Initalize(params);
	bgs_.InitModel(static_cast<const RgbImage&>(nullptr));

	// setup marks to hold results of low and high thresholding
	low_threshold_mask_ = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	low_threshold_mask_.Ptr()->origin = IPL_ORIGIN_BL;

	high_threshold_mask_ = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	high_threshold_mask_.Ptr()->origin = IPL_ORIGIN_BL;

	//
	singlePixelMat_ = cv::Mat_<cv::Vec3b>(height, width, CV_8UC3);
}


AppearanceModel::~AppearanceModel()
{
}

void AppearanceModel::initGmmComponents()
{
	bgs_.InitModel(static_cast<const RgbImage&>(nullptr));
}

void AppearanceModel::loadGmmComponents(GaussMixtureCompoenent const* gmmsList, int gmmsListSize)
{
	// initialize BGS from old GMMs
	initGmmComponents();
	bgs_.loadGmmState(gmmsList, gmmsListSize);
}

void AppearanceModel::pushNextPixels(cv::Mat const& blobRgb, const cv::Vec3b& transparentPix)
{
	// singlePixelMat_ as IplImage
	IplImage iplImage = (IplImage)singlePixelMat_;
	RgbImage singlePixelRgbImage(&iplImage);
	singlePixelRgbImage.ReleaseMemory(false); // prohibit the destruction of underlying memory, as it is owned by singlePixelMat_

	for (int row = 0; row < blobRgb.rows; ++row)
		for (int col = 0; col < blobRgb.cols; ++col)
		{
			const cv::Vec3b& pix3 = blobRgb.at<cv::Vec3b>(row, col);
			if (pix3 == transparentPix) // ignore background pixels
				continue;

			singlePixelMat_.at<cv::Vec3b>(0, 0) = pix3;

			const int unused1 = -1;
			bgs_.Subtract(unused1, singlePixelRgbImage, low_threshold_mask_, high_threshold_mask_);
		}
}

void AppearanceModel::saveGmmComponents(GaussMixtureCompoenent* gmm, int gmmSize, int& newGmmSize)
{
	bgs_.saveGmsState(gmm, gmmSize, newGmmSize);
}

// xs.size=3
float evalGmm(const float* xs, GaussMixtureCompoenent const* gmms, int gmmCount)
{
	CV_Assert(gmmCount > 0);

	float result = 0;
	for (int gmmi = 0; gmmi < gmmCount; ++gmmi)
	{
		auto gmm = gmms[gmmi];

		std::array<float, 3> mu = { gmm.muR, gmm.muG, gmm.muB };
		float p1 = PoolWatch::normalProb(3, xs, &mu[0], gmm.sigma);
		
		result += gmm.weight * p1;
	}
	return result;
}

template <typename PixFun>
void iterateRgbTable(int tableStep, PixFun pixFun)
{
	std::array<float, 3> pix;

	for (int x1 = 0; x1 < 256; x1 += tableStep)
	{
		pix[0] = x1;

		for (int x2 = 0; x2 < 256; x2 += tableStep)
		{
			pix[1] = x2;

			for (int x3 = 0; x3 < 256; x3 += tableStep)
			{
				pix[2] = x3;

				pixFun(pix);
			}
		}
	}
}

float normalizedL2DistanceNaive(GaussMixtureCompoenent const* gmm1, int gmm1Size, GaussMixtureCompoenent const* gmm2, int gmm2Size, int tableStep)
{
	CV_Assert(gmm1Size > 0 && gmm2Size > 0);

	float denom1 = 0;
	iterateRgbTable(tableStep, [=, &denom1](const std::array<float, 3>& pix)
	{
		float p1 = evalGmm(&pix[0], gmm1, gmm1Size);
		denom1 += p1 * p1;
	});
	denom1 = std::sqrtf(denom1);
	assert(denom1 > 0);

	float denom2 = 0;
	iterateRgbTable(tableStep, [=, &denom2](const std::array<float, 3>& pix)
	{
		float p1 = evalGmm(&pix[0], gmm2, gmm2Size);
		denom2 += p1 * p1;
	});
	denom2 = std::sqrtf(denom2);
	assert(denom2 > 0);

	float result = 0;
	iterateRgbTable(tableStep, [=, &result](const std::array<float, 3>& pix)
	{
		float p1 = evalGmm(&pix[0], gmm1, gmm1Size);
		p1 /= denom1;
		float p2 = evalGmm(&pix[0], gmm2, gmm2Size);
		p2 /= denom2;
		result += PoolWatch::sqr(p1 - p2);
	});

	return result;
}

// mu1 [Lx1]
// Computes Integrate[N1(mu1, cov1) * N2(mu2, cov2), dx]
bool integrateTwoGaussiansProduct(const cv::Matx31f& mu1, const cv::Matx33f& cov1, const cv::Matx31f& mu2, const cv::Matx33f& cov2, float& result)
{
	const int L = 3;

	cv::Matx33f covSum = cov1 + cov2;
	float det = cv::determinant(covSum);
	float piL = std::powf(2 * M_PI, L); // constexpr

	float coefLeft = std::powf(piL * det, -0.5);

	//

	const int InvMethod = DECOMP_LU;

	bool invOp;
	cv::Matx33f covSumInv = covSum.inv(InvMethod, &invOp);
	if (!invOp)
		return false;

	cv::Matx31f dMu = mu1 - mu2;
	cv::Matx13f dMuT = dMu.t();

	cv::Matx<float,1,1> trioMat = dMuT * covSumInv * dMu;

	float expVal = std::expf(-0.5 * trioMat.val[0]);
	
	//
	result = coefLeft * expVal;
	return true;
}

// Computes Integrate[GMM1(Mus1, Covs1) * GMM2(Mus2, Covs2), dX]
bool integrateTwoGmmsProduct(GaussMixtureCompoenent const* gmm1, int gmm1Size, GaussMixtureCompoenent const* gmm2, int gmm2Size, float& result)
{
	result = 0;
	for (int i1 = 0; i1 < gmm1Size; ++i1)
	{
		const GaussMixtureCompoenent& gmmComp1 = gmm1[i1];
		cv::Matx31f mu1(gmmComp1.muR, gmmComp1.muG, gmmComp1.muB);
		cv::Matx33f cov1 = cv::Matx33f::eye() * gmmComp1.sigma*gmmComp1.sigma;

		for (int i2 = 0; i2 < gmm2Size; ++i2)
		{
			const GaussMixtureCompoenent& gmmComp2 = gmm2[i2];
			cv::Matx31f mu2(gmmComp2.muR, gmmComp2.muG, gmmComp2.muB);
			cv::Matx33f cov2 = cv::Matx33f::eye() * gmmComp2.sigma*gmmComp2.sigma;

			float integrValue = -1;
			bool integrOp = integrateTwoGaussiansProduct(mu1, cov1, mu2, cov2, integrValue);
			if (!integrOp)
				return false;
			
			float part = gmmComp1.weight * gmmComp2.weight * integrValue;
			result += part;
		}
	}
	return true;
}

// Computes Integrate[GMM1(Mus, Covs) ^ 2), dX]
bool integrateGmmSqr(GaussMixtureCompoenent const* gmm, int gmmSize, float& result)
{
	result = 0;

	// sum(pi ^ 2)
	for (int i = 0; i < gmmSize; ++i)
	{
		const GaussMixtureCompoenent& gmmComp = gmm[i];
		cv::Matx33f cov1 = cv::Matx33f::eye() * PoolWatch::sqr(gmmComp.sigma);

		// two GMM components are the same(use simplified formula)

		float det = cv::determinant(4 * M_PI * cov1);
		float prod = 1 / std::sqrtf(det);
		float part = PoolWatch::sqr(gmmComp.weight) * prod;
		result += part;
	}


	// sum(+2 * pi * pj)
	for (int i1 = 0; i1 < gmmSize; ++i1)
	{
		const GaussMixtureCompoenent& gmmComp1 = gmm[i1];
		cv::Matx31f mu1(gmmComp1.muR, gmmComp1.muG, gmmComp1.muB);
		cv::Matx33f cov1 = cv::Matx33f::eye() * gmmComp1.sigma*gmmComp1.sigma;

		for (int i2 = i1+1; i2 < gmmSize; ++i2)
		{
			const GaussMixtureCompoenent& gmmComp2 = gmm[i2];
			cv::Matx31f mu2(gmmComp2.muR, gmmComp2.muG, gmmComp2.muB);
			cv::Matx33f cov2 = cv::Matx33f::eye() * gmmComp2.sigma*gmmComp2.sigma;

			float integrValue = -1;
			bool integrOp = integrateTwoGaussiansProduct(mu1, cov1, mu2, cov2, integrValue);
			if (!integrOp)
				return false;
			
			float part = gmmComp1.weight * gmmComp2.weight * integrValue;
			result += part;
		}
	}
	return true;
}

bool normalizedL2Distance(GaussMixtureCompoenent const* gmm1, int gmm1Size, GaussMixtureCompoenent const* gmm2, int gmm2Size, float& resultDistance)
{
	CV_Assert(gmm1Size > 0 && gmm2Size > 0);

	float sumP1 = -1;
	bool integrOp = integrateGmmSqr(gmm1, gmm1Size, sumP1);
	if (!integrOp)
		return false;

	float sumP2 = -1;
	integrOp = integrateGmmSqr(gmm2, gmm2Size, sumP2);
	if (!integrOp)
		return false;

	float denom1 = std::sqrtf(sumP1);
	float denom2 = std::sqrtf(sumP2);

	float p12 = -1;
	integrOp = integrateTwoGmmsProduct(gmm1, gmm1Size, gmm2, gmm2Size, p12);
	if (!integrOp)
		return false;

	resultDistance = 2 * (1 - (1 / (denom1*denom2)) * p12);
	return true;
}

// Finds "inside" "point" on the "line" between two gaussian components
bool ridgeLine(const cv::Matx31f& mu1, const cv::Matx33f& cov1Inv, const cv::Matx31f& mu2, const cv::Matx33f& cov2Inv, float alpha, cv::Matx31f& insidePoint)
{
	const int InvMethod = DECOMP_LU;

	bool invOp = false;
	cv::Matx33f coef1 = ((1 - alpha)*cov1Inv + alpha*cov2Inv).inv(InvMethod, &invOp);
	if (!invOp)
		return false;

	cv::Matx31f coef2 = (1 - alpha)*cov1Inv*mu1 + alpha*cov2Inv*mu2;

	insidePoint = coef1 * coef2;
	return true;
}

// Finds how similar are the two GMM components. Large value(eg 0.8)
// means that two components are approximatly the same and might be merged.
bool twoGaussianMixtureComponentsRidgeRatioSimilarity(
	const cv::Matx31f& mu1, const cv::Matx33f& cov1Inv,
	const cv::Matx31f& mu2, const cv::Matx33f& cov2Inv,
	std::function<float(cv::Matx31f)> gmmFun, float alphaValuesCount, float& similarity)
{
	CV_Assert(alphaValuesCount >= 2);
	
	const int InvMethod = DECOMP_LU;

	// find point on the ridge line with min value of GMM

	cv::Matx31f ridgeLinePoint;
	float gmmMinValue = std::numeric_limits<float>::max();

	float alphaStep = 1 / (alphaValuesCount - 1);
	for (float alpha = 0; alpha <= 1.0f; alpha += alphaStep)
	{
		cv::Matx31f insidePoint;
		bool ridgePointOp = ridgeLine(mu1, cov1Inv, mu2, cov2Inv, alpha, insidePoint);
		if (!ridgePointOp)
			return false;

		float probValue = gmmFun(insidePoint);
		if (probValue < gmmMinValue)
		{
			gmmMinValue = probValue;
			ridgeLinePoint = insidePoint;
		}
	}

	// find the second mode of the mixture
	// = min of the two centers
	float probMu1 = gmmFun(mu1);
	float probMu2 = gmmFun(mu2);
	float mode2 = std::min(probMu1, probMu2);

	similarity = gmmMinValue / mode2;
	return true;
}

void mergeGaussianMixtureCompoenents(const GaussMixtureCompoenent& gmmComp1, const GaussMixtureCompoenent& gmmComp2, GaussMixtureCompoenent& resultGmmComp)
{
	float resW = gmmComp1.weight + gmmComp2.weight;

	cv::Matx31f mu1(gmmComp1.muR, gmmComp1.muG, gmmComp1.muB);
	cv::Matx31f mu2(gmmComp2.muR, gmmComp2.muG, gmmComp2.muB);

	cv::Matx31f resMu = gmmComp1.weight*mu1 + gmmComp2.weight*mu2;
	resMu = (1 / resW) * resMu;

	cv::Matx33f cov1 = cv::Matx33f::eye() * gmmComp1.sigma*gmmComp1.sigma;
	cv::Matx33f cov2 = cv::Matx33f::eye() * gmmComp2.sigma*gmmComp2.sigma;

	cv::Matx33f resCov = gmmComp1.weight*(cov1 + (resMu - mu1)*(resMu - mu1).t()) + gmmComp2.weight*(cov2 + (resMu - mu2)*(resMu - mu2).t());
	resCov = (1 / resW) * resCov;

	// populate the result
	resultGmmComp.weight = resW;
	resultGmmComp.muR = resMu.val[0];
	resultGmmComp.muG = resMu.val[1];
	resultGmmComp.muB = resMu.val[2];
	resultGmmComp.sigma = std::sqrtf(resCov.val[0]); // sigma is a square root of a variance
}

bool mergeTwoGaussianMixtures(GaussMixtureCompoenent const* gmm1, int gmm1Size, GaussMixtureCompoenent const* gmm2, int gmm2Size, float maxRidgeRatio, float componentMinWeight, float learningRate,
	GaussMixtureCompoenent * resultGmm, int resultGmmMaxSize, int& rsultGmmSize)
{
	CV_Assert(learningRate > componentMinWeight && "Learned data is always truncated (no learning occur)");

	struct GmmPair
	{
		int Gmm1Ind;
		int Gmm2Ind;
		float Similarity;
	};

	std::vector<GmmPair> similarities;
	similarities.reserve(gmm1Size*gmm2Size);
	for (int i1 = 0; i1 < gmm1Size; ++i1)
	{
		const GaussMixtureCompoenent& g1 = gmm1[i1];
		cv::Matx31f mu1(g1.muR, g1.muG, g1.muB);
		cv::Matx33f cov1 = cv::Matx33f::eye() * g1.sigma * g1.sigma;

		const int InvMethod = DECOMP_LU;

		bool invOp = false;
		cv::Matx33f cov1Inv = cov1.inv(InvMethod, &invOp);
		if (!invOp)
			return false;

		for (int i2 = 0; i2 < gmm2Size; ++i2)
		{
			const GaussMixtureCompoenent& g2 = gmm2[i2];
			cv::Matx31f mu2(g2.muR, g2.muG, g2.muB);
			cv::Matx33f cov2 = cv::Matx33f::eye() * g2.sigma * g2.sigma;

			cv::Matx33f cov2Inv = cov1.inv(InvMethod, &invOp);
			if (!invOp)
				return false;

			auto gmmFun = [=](const cv::Matx31f& pix) -> float 
			{
				float p1 = PoolWatch::normalProb(3, pix.val, (float*)mu1.val, g1.sigma);
				float p2 = PoolWatch::normalProb(3, pix.val, (float*)mu2.val, g2.sigma);
				return g1.weight * p1 + g2.weight * p2;
			};

			float similarity = -1;
			bool similarityOp = twoGaussianMixtureComponentsRidgeRatioSimilarity(mu1, cov1Inv, mu2, cov2Inv, gmmFun, 14, similarity);
			CV_Assert(similarityOp);
			similarities.push_back(GmmPair{ i1, i2, similarity });
		}
	}

	// sort similarity from most to least
	auto descSimilarFun = [](GmmPair g1, GmmPair g2) { return g1.Similarity > g2.Similarity; };
	std::sort(std::begin(similarities), std::end(similarities), descSimilarFun);

	// merge close components
	std::vector<GaussMixtureCompoenent> mergedComponents;
	mergedComponents.reserve(gmm1Size * gmm2Size);
	std::vector<uchar> isMerged1(gmm1Size);
	std::vector<uchar> isMerged2(gmm2Size);
	for (int i = 0; i < (int)similarities.size(); ++i)
	{
		GmmPair twoGmms = similarities[i];
		auto i1 = twoGmms.Gmm1Ind;
		auto i2 = twoGmms.Gmm2Ind;
		
		if (isMerged1[i1] || isMerged2[i2])
			continue;
		
		if (twoGmms.Similarity < maxRidgeRatio)
		{
			// iteration goes from most to least similar component pairs
			// do not merge remaining distinct component pairs
			break;
		}

		const GaussMixtureCompoenent& g1 = gmm1[i1];
		GaussMixtureCompoenent g2Copy = gmm2[i2];
		g2Copy.weight *= learningRate; // diminish value of the component to learn from

		GaussMixtureCompoenent merged;
		mergeGaussianMixtureCompoenents(g1, g2Copy, merged);
		mergedComponents.push_back(merged);

		isMerged1[i1] = true;
		isMerged2[i2] = true;
	}

	// append unmerged components into the result
	for (int i1 = 0; i1 < gmm1Size; ++i1)
	{
		if (!isMerged1[i1])
			mergedComponents.push_back(gmm1[i1]);
	}
	for (int i2 = 0; i2 < gmm2Size; ++i2)
	{
		if (!isMerged2[i2])
		{
			mergedComponents.push_back(gmm2[i2]);
			mergedComponents[mergedComponents.size()-1].weight *= learningRate; // diminish value of the component to learn from
		}
	}

	// remove negligible components
	for (int i = (int)mergedComponents.size() - 1; i >= 0; --i)
	{
		// when we sum two GMMs, compound weight is 2
		const float twoGmmsWeight = 2;

		if ((mergedComponents[i].weight / twoGmmsWeight) < componentMinWeight)
		{
			mergedComponents.erase(std::begin(mergedComponents) + i);
		}
	}

	CV_Assert(!mergedComponents.empty(), "There must be at least one component after merging");

	// ensure sum of weights is one

	float sumWeight = std::accumulate(std::begin(mergedComponents), std::end(mergedComponents), 0.0f,
		[](float ax, const GaussMixtureCompoenent& c) { return ax + c.weight; });

	for (GaussMixtureCompoenent& c : mergedComponents)
		c.weight /= sumWeight;

	// for readability sort the GMM components by weight descending
	std::sort(std::begin(mergedComponents), std::end(mergedComponents), [](const GaussMixtureCompoenent& c1, const GaussMixtureCompoenent& c2) { return c1.weight > c2.weight; });

	// populate the result
	rsultGmmSize = std::min(resultGmmMaxSize, (int)mergedComponents.size());
	
	auto endIt = std::begin(mergedComponents);
	std::advance(endIt, rsultGmmSize);
	std::copy(std::begin(mergedComponents), endIt, resultGmm);

	return true;
}

namespace {
	struct GmmPair
	{
		int Gmm1Ind;
		int Gmm2Ind;
		float Similarity;
	};
}

bool mergeMostSimilarGmmComponents(float maxRidgeRatio, float componentMinWeight, std::vector<GaussMixtureCompoenent>& mergedComponents)
{
	std::vector<GmmPair> similarities;
	int n = (int)mergedComponents.size();
	similarities.reserve(n*(n - 1) / 2);

	for (int i1 = 0; i1 < (int)mergedComponents.size(); ++i1)
	{
		const GaussMixtureCompoenent& g1 = mergedComponents[i1];
		cv::Matx31f mu1(g1.muR, g1.muG, g1.muB);
		cv::Matx33f cov1 = cv::Matx33f::eye() * g1.sigma * g1.sigma;

		const int InvMethod = DECOMP_LU;

		bool invOp = false;
		cv::Matx33f cov1Inv = cov1.inv(InvMethod, &invOp);
		if (!invOp)
			return false;

		for (int i2 = i1 + 1; i2 < (int)mergedComponents.size(); ++i2)
		{
			const GaussMixtureCompoenent& g2 = mergedComponents[i2];
			cv::Matx31f mu2(g2.muR, g2.muG, g2.muB);
			cv::Matx33f cov2 = cv::Matx33f::eye() * g2.sigma * g2.sigma;

			cv::Matx33f cov2Inv = cov1.inv(InvMethod, &invOp);
			if (!invOp)
				return false;

			auto gmmFun = [=](const cv::Matx31f& pix) -> float
			{
				float p1 = PoolWatch::normalProb(3, pix.val, (float*)mu1.val, g1.sigma);
				float p2 = PoolWatch::normalProb(3, pix.val, (float*)mu2.val, g2.sigma);
				return g1.weight * p1 + g2.weight * p2;
			};

			float similarity = -1;
			bool similarityOp = twoGaussianMixtureComponentsRidgeRatioSimilarity(mu1, cov1Inv, mu2, cov2Inv, gmmFun, 14, similarity);
			if (!similarityOp)
				return false;
			similarities.push_back(GmmPair{ i1, i2, similarity });
		}
	}

	// sort similarity from most to least
	auto descSimilarFun = [](GmmPair g1, GmmPair g2) { return g1.Similarity > g2.Similarity; };
	std::sort(std::begin(similarities), std::end(similarities), descSimilarFun);

	// merge close components
	int mergeInd1 = -1;
	int mergeInd2 = -1;
	for (int i = 0; i < (int)similarities.size(); ++i)
	{
		GmmPair twoGmms = similarities[i];
		auto i1 = twoGmms.Gmm1Ind;
		auto i2 = twoGmms.Gmm2Ind;

		if (twoGmms.Similarity < maxRidgeRatio)
		{
			// iteration goes from most to least similar component pairs
			// do not merge remaining distinct component pairs
			break;
		}

		mergeInd1 = i1;
		mergeInd2 = i2;
		break;
	}

	if (mergeInd1 == -1)
		return false;

	// do two components merging

	GaussMixtureCompoenent oneCopy = mergedComponents[mergeInd1];
	const GaussMixtureCompoenent& other = mergedComponents[mergeInd2];

	// put the merging result into the first component
	GaussMixtureCompoenent& result = mergedComponents[mergeInd1];

	mergeGaussianMixtureCompoenents(oneCopy, other, result);

	// remove other component
	mergedComponents.erase(std::begin(mergedComponents) + mergeInd2);
}

void mergeGaussianMixtureComponents(const GaussMixtureCompoenent* gmm, int gmmSize, float maxRidgeRatio, float componentMinWeight, GaussMixtureCompoenent * resultGmm, int resultGmmMaxSize, int& rsultGmmSize)
{
	std::vector<GaussMixtureCompoenent> mergedComponents(gmm, gmm + gmmSize);

	// merging loop

	while (mergedComponents.size() > 1)
	{
		bool mergedSomething = mergeMostSimilarGmmComponents(maxRidgeRatio, componentMinWeight, mergedComponents);
		if (!mergedSomething)
			break;
	}

	CV_Assert(!mergedComponents.empty(), "There must be at least one component after merging");

	// ensure sum of weights is one
	fixGmmWeights(mergedComponents.data(), mergedComponents.size());

	// for readability sort the GMM components by weight descending
	std::sort(std::begin(mergedComponents), std::end(mergedComponents), [](const GaussMixtureCompoenent& c1, const GaussMixtureCompoenent& c2) { return c1.weight > c2.weight; });

	// populate the result
	rsultGmmSize = std::min(resultGmmMaxSize, (int)mergedComponents.size());

	auto endIt = std::begin(mergedComponents);
	std::advance(endIt, rsultGmmSize);
	std::copy(std::begin(mergedComponents), endIt, resultGmm);
}

void fixGmmWeights(GaussMixtureCompoenent* gmm, int gmmSize)
{
	float sumWeight = std::accumulate(gmm, gmm+gmmSize, 0.0f,
		[](float ax, const GaussMixtureCompoenent& c) { return ax + c.weight; });

	for (int i = 0; i < gmmSize; ++i)
	{
		GaussMixtureCompoenent& c = gmm[i];
		c.weight /= sumWeight;
	}
}