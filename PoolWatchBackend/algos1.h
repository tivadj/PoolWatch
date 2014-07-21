#pragma  once
#include <vector>
#include <tuple>
#include <functional>

#include <opencv2\core.hpp>

#include "MatrixUndirectedGraph.h"
#include "WaterClassifier.h"

__declspec(dllexport) void classifyAndGetMask(const cv::Mat& image, std::function<bool(const cv::Vec3d&)>  pred, cv::Mat_<uchar>& mask);
__declspec(dllexport) void estimateClassifier(const cv::Mat& image, std::function<double(const cv::Vec3d&)> computeOne, cv::Mat_<double>& mask);

__declspec(dllexport) void maximumWeightIndependentSetNaiveMaxFirst(const MatrixUndirectedGraph& graph, std::vector<uchar>& vertexSet);
__declspec(dllexport) void maximumWeightIndependentSetNaiveMaxFirstMultipleSeeds(const MatrixUndirectedGraph& graph, std::vector<uchar>& vertexSet);

struct IndependentSetValidationResult
{
	bool isValid;
	std::string message;
	int vertex1;
	int vertex2;
};

IndependentSetValidationResult validateMaximumWeightIndependentSet(const MatrixUndirectedGraph& graph, const std::vector<uchar>& vertexSet);

double calculateVertexSetWeight(const MatrixUndirectedGraph& graph, const std::vector<bool>& vertexSet);

std::tuple<MatrixUndirectedGraph, std::vector<int>> createFromEdgeList(const std::vector<int>& vertices, const std::vector<int>& edgeListByRow);

namespace PoolWatch
{
	template <typename T>
	auto sqr(const T& x) -> T
	{
		return x*x;
	}

	__declspec(dllexport) auto deg2rad(const float& degree) -> float;
	__declspec(dllexport) auto normalProb(float x, float mu, float sigma) -> float;
}

class EMQuick : public cv::EM
{
public:
	EMQuick(const EMQuick&) = delete;
	static cv::Vec2d predict2(cv::InputArray _sample, const cv::Mat& meansPar, const std::vector<cv::Mat>& invCovsEigenValuesPar, const cv::Mat& logWeightDivDetPar, cv::Mat& cacheL);
	static cv::Vec2d computeProbabilitiesInplace(cv::Mat& sample, const cv::Mat& meansPar, const std::vector<cv::Mat>& invCovsEigenValuesPar, const cv::Mat& logWeightDivDetPar, cv::Mat& cacheL);

	int getNClusters()
	{
		return this->nclusters;
	}
	const cv::Mat& getMeans()
	{
		return this->means;
	}
	const std::vector<cv::Mat>& getInvCovsEigenValuesPar()
	{
		return this->invCovsEigenValues;
	}
	const cv::Mat& getLogWeightDivDetPar()
	{
		return this->logWeightDivDet;
	}
};


