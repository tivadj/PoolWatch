#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional> // std::reference_wrapper
#include <hash_map>
#include <set>
#include <tuple>

#include <ppl.h>

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>

#include <boost/thread/tss.hpp>

#include "algos1.h"

using namespace std;
using namespace Concurrency; // task_group
using namespace cv;

namespace PoolWatch
{
	auto deg2rad(const float& degree) -> float
	{
		return degree * M_PI / 180.0f;
	}
}

void maximumWeightIndependentSetNaiveMaxFirst(const MatrixUndirectedGraph& graph, vector<bool>& vertexSet)
{
	auto vertexCount = graph.nodesCount();

	// initialize bookkeeping data

	struct VertexData
	{
		int vertexId;
		double weight;
		bool processed;
	};
	
	vector<VertexData> vertexIdAndWeight(vertexCount);
	for (int vertexId = 0; vertexId < vertexCount; vertexId++)
	{
		VertexData data;
		data.vertexId = vertexId;
		data.weight = graph.nodePayload(vertexId);
		data.processed = false;
		vertexIdAndWeight[vertexId] = data;
	}

	// arrange weights large to small
	sort(begin(vertexIdAndWeight), end(vertexIdAndWeight), [](VertexData& x, VertexData& y) { return x.weight > y.weight;  });

	// TODO: the value=VertexData& doesn't work; when adding new pair, all values become equal to this pair
	hash_map<int, reference_wrapper<VertexData>> vertexIdToData;
	for (VertexData& vertexData : vertexIdAndWeight)
	{
		vertexIdToData.insert(make_pair(vertexData.vertexId, std::ref(vertexData)));
	}

	vertexSet.resize(vertexCount);
	vertexSet.assign(vertexCount, false);
	
	// sequentially include vertices from largest weight to smallest

	vector<int> neighbourVertices;
	for (int dataIndex = 0; dataIndex < vertexCount; dataIndex++)
	{
		auto& vertexData = vertexIdAndWeight[dataIndex];
		if (vertexData.processed)
			continue;

		vertexData.processed = true;

		// include vertex into max weight independent set
		auto vertexId = vertexData.vertexId;
		vertexSet[vertexId] = true;

		// reject all neighbour vertices
		graph.adjacentNodes(vertexId, neighbourVertices);

		for (auto adjVertId : neighbourVertices)
		{
			auto vertexDataIt = vertexIdToData.find(adjVertId);
			if (vertexDataIt != end(vertexIdToData))
				vertexDataIt->second.get().processed = true;
		}			
	}

	auto validation = validateMaximumWeightIndependentSet(graph, vertexSet);
	assert(validation.isValid);
}

IndependentSetValidationResult validateMaximumWeightIndependentSet(const MatrixUndirectedGraph& graph, const vector<bool>& vertexSet)
{
	IndependentSetValidationResult result;
	result.isValid = false;
	result.vertex1 = -1;
	result.vertex1 = -1;

	if (graph.nodesCount() != vertexSet.size())
	{
		result.message = "graph.VertexCount != independentSet.size";
		return result;
	}
	
	vector<int> neighbours;
	for (int vertexId = 0; vertexId < graph.nodesCount(); ++vertexId)
	{
		auto vertexInSet = vertexSet[vertexId];
		if (!vertexInSet)
			continue;

		// check all adjacent vertices are not in independent set
		
		graph.adjacentNodes(vertexId, neighbours);
		
		auto neighIt = find_if(begin(neighbours), end(neighbours), [&vertexSet](int vertexId) { return vertexSet[vertexId]; });
		if (neighIt != end(neighbours))
		{
			result.vertex1 = vertexId;
			result.vertex2 = *neighIt;
			result.message = "Adjacent vertices are both in independent set";
			return result;
		}
	}

	result.isValid = true;
	return result;
}

double calculateVertexSetWeight(const MatrixUndirectedGraph& graph, const std::vector<bool>& vertexSet)
{
	double result = 0;

	for (int vertexId = 0; vertexId < graph.nodesCount(); ++vertexId)
	{
		auto vertexInSet = vertexSet[vertexId];
		if (!vertexInSet)
			continue;

		result += graph.nodePayload(vertexId);
	}
	return result;
}

// result(1) = vertexIndex to original vertex id
tuple<MatrixUndirectedGraph, vector<int>> createFromEdgeList(const vector<int>& vertices, const vector<int>& edgeListByRow)
{
	// TODO: validate edge-list graph
	assert(edgeListByRow.size() % 2 == 0);

	int vertexCount = (int)vertices.size();

	vector<int> vertexIndexToOriginal(begin(vertices), end(vertices));
	sort(begin(vertexIndexToOriginal), end(vertexIndexToOriginal));

	hash_map<int, int> vertexOriginalToIndex;
	for (int vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex)
	{
		auto original = vertexIndexToOriginal[vertexIndex];
		vertexOriginalToIndex[original] = vertexIndex;
	}

	MatrixUndirectedGraph result(vertexCount, vertexCount);

	for (int i = 0; i < edgeListByRow.size() / 2; i++)
	{
		auto origFrom = edgeListByRow[i * 2 + 0];
		auto origTo   = edgeListByRow[i * 2 + 1];

		auto from = vertexOriginalToIndex[origFrom];
		auto to   = vertexOriginalToIndex[origTo];
		result.setEdge(from, to);
	}

	return make_tuple(result, vertexIndexToOriginal);
}

void classifyAndGetMaskParWorkPerPixel(const cv::Mat& image, std::function<bool(const cv::Vec3d&, cv::Mat_<double>& oneColor, std::vector<uchar>& classifResult)>  pred, cv::Mat_<uchar>& mask)
{
	assert(image.channels() == 3);
	assert(image.depth() == CV_8U);

	cv::Mat& mask2 = mask;
	mask2.create(image.rows, image.cols, CV_8UC1);

	//
	uchar* pRowBeg = image.data;

	int nrows = image.rows;
	int ncols = image.cols;
	//if (image.isContinuous())
	//{
	//	ncols = ncols * nrows;
	//	nrows = 1;
	//	mask2 = mask2.reshape(1, 1);
	//}


	task_group tg;

	for (int r = 0; r < nrows; ++r)
	{
		const Vec3b* pRowIt = image.ptr<Vec3b>(r);

		for (int c = 0; c < ncols; ++c)
		{
			Vec3b pix = *pRowIt;
			Vec3d pixDbl = Vec3d(pix(0), pix(1), pix(2));

			//bool classif = pred(pixDbl);
			//mask(r, c) = classif ? (uchar)255 : 0;

			tg.run([&pred, pixDbl, &mask, r, c]()
			{
				boost::thread_specific_ptr<cv::Mat_<double>> pOneColor;
				boost::thread_specific_ptr<std::vector<uchar>> pClassifResult;

				if (pOneColor.get() == nullptr)
				{
					pOneColor.reset(new cv::Mat_<double>(1, 3, CV_64FC1));
				}
				if (pClassifResult.get() == nullptr)
				{
					pClassifResult.reset(new std::vector<uchar>(1));
				}

				bool classif = pred(pixDbl, *pOneColor, *pClassifResult);
				mask(r, c) = classif ? (uchar)255 : 0;
			});

			pRowIt++;
		}

		pRowBeg += image.step; // jump to the next line
	}

	tg.wait();

	// reshape back
	//if (image.isContinuous())
	//{
	//	mask2 = mask2.reshape(1, image.rows);
	//}
}

void classifyAndGetMaskParWorkLine(const cv::Mat& image, std::function<bool(const cv::Vec3d&, cv::Mat_<double>& oneColor, std::vector<uchar>& classifResult)>  pred, cv::Mat_<uchar>& mask)
{
	assert(image.channels() == 3);
	assert(image.depth() == CV_8U);

	cv::Mat& mask2 = mask;
	mask2.create(image.rows, image.cols, CV_8UC1);

	//
	uchar* pRowBeg = image.data;

	int nrows = image.rows;
	int ncols = image.cols;
	//if (image.isContinuous())
	//{
	//	ncols = ncols * nrows;
	//	nrows = 1;
	//	mask2 = mask2.reshape(1, 1);
	//}


	task_group tg;

	for (int r = 0; r < nrows; ++r)
	{
		tg.run([&pred, &image, &mask, ncols, r]()
		{
			boost::thread_specific_ptr<cv::Mat_<double>> pOneColor;
			boost::thread_specific_ptr<std::vector<uchar>> pClassifResult;

			if (pOneColor.get() == nullptr)
			{
				pOneColor.reset(new cv::Mat_<double>(1, 3, CV_64FC1));
			}
			if (pClassifResult.get() == nullptr)
			{
				pClassifResult.reset(new std::vector<uchar>(1));
			}

			const Vec3b* pRowIt = image.ptr<Vec3b>(r);

			for (int c = 0; c < ncols; ++c)
			{
				Vec3b pix = *pRowIt;
				Vec3d pixDbl = Vec3d(pix(0), pix(1), pix(2));

				bool classif = pred(pixDbl, *pOneColor, *pClassifResult);
				mask(r, c) = classif ? (uchar)255 : 0;

				pRowIt++;
			}
		});

		pRowBeg += image.step; // jump to the next line
	}

	tg.wait();

	// reshape back
	//if (image.isContinuous())
	//{
	//	mask2 = mask2.reshape(1, image.rows);
	//}
}


void classifyAndGetMask(const cv::Mat& image, std::function<bool(const cv::Vec3d&)>  pred, cv::Mat_<uchar>& mask)
{
	assert(image.channels() == 3);
	assert(image.depth() == CV_8U);

	cv::Mat& mask2 = mask;
	mask2.create(image.rows, image.cols, CV_8UC1);

	//
	uchar* pRowBeg = image.data;

	int nrows = image.rows;
	int ncols = image.cols;
	if (image.isContinuous())
	{
		ncols = ncols * nrows;
		nrows = 1;
		mask2 = mask2.reshape(1, 1);
	}

	for (int r = 0; r < nrows; ++r)
	{
		const Vec3b* pRowIt = image.ptr<Vec3b>(r);

		for (int c = 0; c < ncols; ++c)
		{
			Vec3b pix = *pRowIt;
			Vec3d pixDbl = Vec3d(pix(0), pix(1), pix(2));

			bool classif = pred(pixDbl);
			mask(r, c) = classif ? (uchar)255 : 0;

			pRowIt++;
		}

		pRowBeg += image.step; // jump to the next line
	}

	// reshape back
	if (image.isContinuous())
	{
		mask2 = mask2.reshape(1, image.rows);
	}
}

void estimateClassifier(const cv::Mat& image, std::function<double(const cv::Vec3d&)> computeOne, cv::Mat_<double>& mask)
{
	assert(image.channels() == 3);
	assert(image.depth() == CV_8U);

	cv::Mat& mask2 = mask;
	mask2.create(image.rows, image.cols, CV_64FC1);

	//
	uchar* pRowBeg = image.data;

	int nrows = image.rows;
	int ncols = image.cols;
	if (image.isContinuous())
	{
		ncols = ncols * nrows;
		nrows = 1;
		mask2 = mask2.reshape(1, 1);
	}

	for (int r = 0; r < nrows; ++r)
	{
		const Vec3b* pRowIt = image.ptr<Vec3b>(r);

		for (int c = 0; c < ncols; ++c)
		{
			Vec3b pix = *pRowIt;
			Vec3d pixDbl = Vec3d(pix(0), pix(1), pix(2));

			double estimate = computeOne(pixDbl);
			mask(r, c) = estimate;

			pRowIt++;
		}

		pRowBeg += image.step; // jump to the next line
	}

	// reshape back
	if (image.isContinuous())
	{
		mask2 = mask2.reshape(1, image.rows);
	}
}

Vec2d EMQuick::predict2(InputArray _sample, const cv::Mat& meansPar, const std::vector<cv::Mat>& invCovsEigenValuesPar, const cv::Mat& logWeightDivDetPar, Mat& cacheL)
{
	Mat sample = _sample.getMat();
	//CV_Assert(isTrained());

	CV_Assert(!sample.empty());
	CV_Assert(sample.type() == CV_64FC1);
	CV_Assert(sample.rows == 1);

	return computeProbabilitiesInplace(sample, meansPar, invCovsEigenValuesPar, logWeightDivDetPar, cacheL);
}

Vec2d EMQuick::computeProbabilitiesInplace(Mat& sample, const Mat& meansPar, const std::vector<Mat>& invCovsEigenValuesPar, const Mat& logWeightDivDetPar, Mat& cacheL)
{
	// L_ik = log(weight_k) - 0.5 * log(|det(cov_k)|) - 0.5 *(x_i - mean_k)' cov_k^(-1) (x_i - mean_k)]
	// q = arg(max_k(L_ik))
	// probs_ik = exp(L_ik - L_iq) / (1 + sum_j!=q (exp(L_ij - L_iq))
	// see Alex Smola's blog http://blog.smola.org/page/2 for
	// details on the log-sum-exp trick

	const Mat& means = meansPar;
	const std::vector<Mat>& invCovsEigenValues = invCovsEigenValuesPar;
	const Mat& logWeightDivDet = logWeightDivDetPar;
	int nclusters = means.rows;

	CV_Assert(!means.empty());
	CV_Assert(sample.type() == CV_64FC1);
	CV_Assert(sample.rows == 1);
	CV_Assert(sample.cols == means.cols);

	int dim = sample.cols;

	//Mat L(1, nclusters, CV_64FC1);
	//Mat& L = cacheL;
	Mat& L = cacheL;

	int label = 0;
	for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
	{
		//const auto& clusterMeanRow = means.row(clusterIndex);
		//sample -= clusterMeanRow;
		for (int i = 0; i < sample.cols; ++i)
			//sample.at<double>(0, i) -= clusterMeanRow.at<double>(0, i);
			sample.at<double>(0, i) -= means.at<double>(clusterIndex, i);

		Mat& centeredSample = sample;

		//Mat rotatedCenteredSample = covMatType != EM::COV_MAT_GENERIC ?
		//centeredSample : centeredSample * covsRotateMats[clusterIndex];
		Mat& rotatedCenteredSample = centeredSample;

		double Lval = 0;
		for (int di = 0; di < dim; di++)
		{
			//double w = invCovsEigenValues[clusterIndex].at<double>(covMatType != EM::COV_MAT_SPHERICAL ? di : 0);
			double w = invCovsEigenValues[clusterIndex].at<double>(0);
			double val = rotatedCenteredSample.at<double>(di);
			Lval += w * val * val;
		}
		CV_DbgAssert(!logWeightDivDet.empty());
		L.at<double>(clusterIndex) = logWeightDivDet.at<double>(clusterIndex) -0.5 * Lval;

		if (L.at<double>(clusterIndex) > L.at<double>(label))
			label = clusterIndex;

		//sample += clusterMeanRow;
		for (int i = 0; i < sample.cols; ++i)
			//sample.at<double>(0, i) += clusterMeanRow.at<double>(0, i);
			sample.at<double>(0, i) += means.at<double>(clusterIndex, i);
	}

	double maxLVal = L.at<double>(label);
	Mat expL_Lmax = L; // exp(L_ij - L_iq)
	for (int i = 0; i < L.cols; i++)
		expL_Lmax.at<double>(i) = std::exp(L.at<double>(i) -maxLVal);
	double expDiffSum = sum(expL_Lmax)[0]; // sum_j(exp(L_ij - L_iq))

	Vec2d res;
	res[0] = std::log(expDiffSum) + maxLVal - 0.5 * dim * CV_LOG2PI;
	res[1] = label;

	return res;
}
