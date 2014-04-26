//#pragma once // allow multiple inclusions in each translation unit

#include <amp.h>
#include <amp_math.h> // fast_math
#include <amp_graphics.h>

#include "algos1.h"

using namespace concurrency; // amp
using namespace concurrency::graphics; // float_3

inline double computeGaussMixtureModel(int nclusters, const double* pMeans, const double* pInvCovsEigenValues, const double* pLogWeightDivDet, double* pSample, double* pCacheL)
restrict(cpu, amp) 
{
	// L_ik = log(weight_k) - 0.5 * log(|det(cov_k)|) - 0.5 *(x_i - mean_k)' cov_k^(-1) (x_i - mean_k)]
	// q = arg(max_k(L_ik))
	// probs_ik = exp(L_ik - L_iq) / (1 + sum_j!=q (exp(L_ij - L_iq))
	// see Alex Smola's blog http://blog.smola.org/page/2 for
	// details on the log-sum-exp trick

	//const Mat& means = meansPar;
	//const std::vector<Mat>& invCovsEigenValues = invCovsEigenValuesPar;
	//const Mat& logWeightDivDet = logWeightDivDetPar;
	//int nclusters = means.rows;

	//CV_Assert(!means.empty());
	//CV_Assert(sample.type() == CV_64FC1);
	//CV_Assert(sample.rows == 1);
	//CV_Assert(sample.cols == means.cols);

	//int dim = sample.cols;
	const int dim = 3;

	//Mat L(1, nclusters, CV_64FC1);
	//Mat& L = cacheL;
	//Mat& L = cacheL;

	int label = 0;
	for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
	{
		//const auto& clusterMeanRow = means.row(clusterIndex);
		//sample -= clusterMeanRow;
		for (int i = 0; i < dim; ++i)
			//sample.at<double>(0, i) -= clusterMeanRow.at<double>(0, i);
			//sample.at<double>(0, i) -= means.at<double>(clusterIndex, i);
			pSample[i] -= pMeans[clusterIndex * dim + i];

		//Mat& centeredSample = sample;

		//Mat rotatedCenteredSample = covMatType != EM::COV_MAT_GENERIC ?
		//centeredSample : centeredSample * covsRotateMats[clusterIndex];
		//Mat& rotatedCenteredSample = centeredSample;
		double* pRotatedCenteredSample = pSample;

		double Lval = 0;
		for (int di = 0; di < dim; di++)
		{
			//double w = invCovsEigenValues[clusterIndex].at<double>(covMatType != EM::COV_MAT_SPHERICAL ? di : 0);
			double w = pInvCovsEigenValues[clusterIndex];
			double val = pRotatedCenteredSample[di];
			Lval += w * val * val;
		}
		//CV_DbgAssert(!logWeightDivDet.empty());
		pCacheL[clusterIndex] = pLogWeightDivDet[clusterIndex] - 0.5 * Lval;

		if (pCacheL[clusterIndex] > pCacheL[label])
			label = clusterIndex;

		//sample += clusterMeanRow;
		for (int i = 0; i < dim; ++i)
			//sample.at<double>(0, i) += clusterMeanRow.at<double>(0, i);
			//sample.at<double>(0, i) += means.at<double>(clusterIndex, i);
			pSample[i] += pMeans[clusterIndex * 3 + i];
	}

	double maxLVal = pCacheL[label];

	//Mat expL_Lmax = L; // exp(L_ij - L_iq)
	//for (int i = 0; i < L.cols; i++)
	//	expL_Lmax.at<double>(i) = std::exp(pCacheL[i] - maxLVal);
	//double expDiffSum = sum(expL_Lmax)[0]; // sum_j(exp(L_ij - L_iq))

	double expDiffSum = 0;
	for (int i = 0; i < nclusters; i++)
	{
		//double expL = std::exp(pCacheL[i] - maxLVal);
		double expL = concurrency::precise_math::exp(pCacheL[i] - maxLVal);
		pCacheL[i] = expL;
		expDiffSum += expL;
	}

	//Vec2d res;
	//res[0] = std::log(expDiffSum) + maxLVal - 0.5 * dim * CV_LOG2PI;
	//res[1] = label;

	//double logProb = std::log(expDiffSum) + maxLVal - 0.5 * dim * CV_LOG2PI;
	double logProb = concurrency::precise_math::log(expDiffSum) + maxLVal - 0.5 * dim * CV_LOG2PI;

	return logProb;
}

template <typename T>
inline T computeGaussMixtureModelGen(int nclusters, const T* pMeans, const T* pInvCovsEigenValues, const T* pLogWeightDivDet, T* pSample, T* pCacheL) restrict(cpu, amp)
{
	// L_ik = log(weight_k) - 0.5 * log(|det(cov_k)|) - 0.5 *(x_i - mean_k)' cov_k^(-1) (x_i - mean_k)]
	// q = arg(max_k(L_ik))
	// probs_ik = exp(L_ik - L_iq) / (1 + sum_j!=q (exp(L_ij - L_iq))
	// see Alex Smola's blog http://blog.smola.org/page/2 for
	// details on the log-sum-exp trick

	//const Mat& means = meansPar;
	//const std::vector<Mat>& invCovsEigenValues = invCovsEigenValuesPar;
	//const Mat& logWeightDivDet = logWeightDivDetPar;
	//int nclusters = means.rows;

	//CV_Assert(!means.empty());
	//CV_Assert(sample.type() == CV_64FC1);
	//CV_Assert(sample.rows == 1);
	//CV_Assert(sample.cols == means.cols);

	//int dim = sample.cols;
	const int dim = 3;

	//Mat L(1, nclusters, CV_64FC1);
	//Mat& L = cacheL;
	//Mat& L = cacheL;

	int label = 0;
	for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
	{
		//const auto& clusterMeanRow = means.row(clusterIndex);
		//sample -= clusterMeanRow;
		for (int i = 0; i < dim; ++i)
			//sample.at<T>(0, i) -= clusterMeanRow.at<T>(0, i);
			//sample.at<T>(0, i) -= means.at<T>(clusterIndex, i);
			pSample[i] -= pMeans[clusterIndex * dim + i];

		//Mat& centeredSample = sample;

		//Mat rotatedCenteredSample = covMatType != EM::COV_MAT_GENERIC ?
		//centeredSample : centeredSample * covsRotateMats[clusterIndex];
		//Mat& rotatedCenteredSample = centeredSample;
		T* pRotatedCenteredSample = pSample;

		T Lval = 0;
		for (int di = 0; di < dim; di++)
		{
			//T w = invCovsEigenValues[clusterIndex].at<T>(covMatType != EM::COV_MAT_SPHERICAL ? di : 0);
			T w = pInvCovsEigenValues[clusterIndex];
			T val = pRotatedCenteredSample[di];
			Lval += w * val * val;
		}
		//CV_DbgAssert(!logWeightDivDet.empty());
		pCacheL[clusterIndex] = pLogWeightDivDet[clusterIndex] - 0.5 * Lval;

		if (pCacheL[clusterIndex] > pCacheL[label])
			label = clusterIndex;

		//sample += clusterMeanRow;
		for (int i = 0; i < dim; ++i)
			//sample.at<T>(0, i) += clusterMeanRow.at<T>(0, i);
			//sample.at<T>(0, i) += means.at<T>(clusterIndex, i);
			pSample[i] += pMeans[clusterIndex * 3 + i];
	}

	T maxLVal = pCacheL[label];

	//Mat expL_Lmax = L; // exp(L_ij - L_iq)
	//for (int i = 0; i < L.cols; i++)
	//	expL_Lmax.at<T>(i) = std::exp(pCacheL[i] - maxLVal);
	//T expDiffSum = sum(expL_Lmax)[0]; // sum_j(exp(L_ij - L_iq))

	T expDiffSum = 0;
	for (int i = 0; i < nclusters; i++)
	{
		//T expL = std::exp(pCacheL[i] - maxLVal);
		T expL = concurrency::fast_math::exp(pCacheL[i] - maxLVal);
		pCacheL[i] = expL;
		expDiffSum += expL;
	}

	//Vec2d res;
	//res[0] = std::log(expDiffSum) + maxLVal - 0.5 * dim * CV_LOG2PI;
	//res[1] = label;

	//T logProb = std::log(expDiffSum) + maxLVal - 0.5 * dim * CV_LOG2PI;
	T logProb = concurrency::fast_math::log(expDiffSum) + maxLVal - 0.5 * dim * CV_LOG2PI;

	return logProb;
}

void classifyAndGetMaskAmpFloat(const cv::Mat& image, WaterClassifier& wc, cv::Mat& mask)
{
	assert(image.channels() == 3);
	//assert(image.depth() == CV_8U);
	assert(image.depth() == CV_32F);

	//
	mask.create(image.rows, image.cols, CV_32SC1);
	bool bbb = mask.isContinuous();

	int nrows = image.rows;
	int ncols = image.cols;
	if (image.isContinuous())
	{
		ncols = ncols * nrows;
		nrows = 1;
		mask = mask.reshape(1, 1);
	}

	auto& c1 = static_cast<EMQuick&>(wc.waterMixGauss_);
	auto& c2 = static_cast<EMQuick&>(wc.nonWaterMixGauss_);

	const int nclusters = 6;
	const int dims = 3;

	//array_view<const float_3, 1> pixsArray(ncols*nrows, (float_3*)image.data);
	std::vector<int32_t> maskData1(ncols*nrows);
	//array_view<int32_t, 1> maskArray(ncols*nrows, (int32_t*)&maskData1[0]);
	array_view<int32_t, 1> maskArray(ncols*nrows, (int32_t*)mask.data);
	//array_view<int32_t, 1> maskArray(ncols*nrows, (int32_t*)mask.data);
	//maskArray.discard_data(); // no need to copy maskArray to GPU

	//Mat meansFloat = c1.getMeans();
	//meansFloat.convertTo(meansFloat, CV_32FC1);
	//array_view<const float, 1> meansArray1(nclusters*dims, (float*)meansFloat.data);

	//std::array<float, 6> invCovs1;
	//std::array<float, 6> invCovs2;
	//for (int i = 0; i < nclusters; ++i)
	//{
	//	invCovs1[i] = (float)c1.getInvCovsEigenValuesPar()[i].at<double>(0);
	//	invCovs2[i] = (float)c2.getInvCovsEigenValuesPar()[i].at<double>(0);
	//}
	//array_view<const float, 1> invCovsArray1(nclusters, &invCovs1[0]);

	//Mat logWeightsFloat = c1.getLogWeightDivDetPar();
	//logWeightsFloat.convertTo(logWeightsFloat, CV_32FC1);

	//array_view<const float, 1> logWeightDivDetArray1(nclusters, (float*)logWeightsFloat.data);

	parallel_for_each(maskArray.extent, [=](index<1> idx) restrict(cpu, amp)
	{
		//float_3 v = pixsArray[idx];
		maskArray[idx] = 255;

		//const int nclusters = 6;
		//float cacheL[nclusters];

		//float logProb1 = computeGaussMixtureModelGen<float>(nclusters, meansArray1.data(), invCovsArray1.data(), logWeightDivDetArray1.data(), (float*)&v, &cacheL[0]);
	});

	// reshape back
	if (image.isContinuous())
	{
		mask = mask.reshape(1, image.rows);
	}
	mask.convertTo(mask, CV_8UC1);
}

void classifyAndGetMaskAmp(const cv::Mat& image, WaterClassifier& wc, cv::Mat_<uchar>& mask)
{
	assert(image.channels() == 3);
	assert(image.depth() == CV_8U);

	cv::Mat& mask2 = mask;
	mask2.create(image.rows, image.cols, CV_32SC1);

	int nrows = image.rows;
	int ncols = image.cols;
	if (image.isContinuous())
	{
		ncols = ncols * nrows;
		nrows = 1;
		mask2 = mask2.reshape(1, 1);
	}

	auto& c1 = static_cast<EMQuick&>(wc.waterMixGauss_);
	auto& c2 = static_cast<EMQuick&>(wc.nonWaterMixGauss_);

	const int nclusters = 6;
	const int dims = 3;

	std::array<double, 6> invCovs1;
	std::array<double, 6> invCovs2;
	for (int i = 0; i < nclusters; ++i)
	{
		invCovs1[i] = c1.getInvCovsEigenValuesPar()[i].at<double>(0);
		invCovs2[i] = c2.getInvCovsEigenValuesPar()[i].at<double>(0);
	}

	array_view<const double_3, 1> pixsArray(ncols*nrows, (double_3*)image.data);
	array_view<int32_t, 1> maskArray(ncols*nrows, (int32_t*)&mask.data);
	maskArray.discard_data(); // no need to copy maskArray to GPU

	array_view<const double, 1> meansArray1(nclusters*dims, (double*)c1.getMeans().data);
	array_view<const double, 1> invCovsArray1(nclusters, &invCovs1[0]);
	array_view<const double, 1> logWeightDivDetArray1(nclusters, (double*)c1.getLogWeightDivDetPar().data);

	//array_view<double, 1> sampleArray(nclusters, (double*)oneColor.data);

	parallel_for_each(maskArray.extent, [=](index<1> idx) restrict(cpu, amp)
	{
		double_3 v = pixsArray[idx];
		float a1 = (float)v.x;
		//Vec3d v = pixsArray[idx];
		//maskArray[idx] = pixsArray[idx](0) > 0 ? 255 : 0;
		maskArray[idx] = 255;

		const int nclusters = 6;
		double cacheL[nclusters];

		double logProb1 = computeGaussMixtureModel(nclusters, meansArray1.data(), invCovsArray1.data(), logWeightDivDetArray1.data(), (double*)&v, &cacheL[0]);
	});

	// reshape back
	if (image.isContinuous())
	{
		mask2 = mask2.reshape(1, image.rows);
	}

	mask2.convertTo(mask2, CV_8UC1);
}
