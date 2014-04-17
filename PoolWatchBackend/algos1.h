#pragma  once
#include <vector>
#include <tuple>
#include <functional>

#include <opencv2\core.hpp>

#include "MatrixUndirectedGraph.h"
#include "WaterClassifier.h"

void maximumWeightIndependentSetNaiveMaxFirst(const MatrixUndirectedGraph& graph, std::vector<bool>& vertexSet);

__declspec(dllexport) void classifyAndGetMask(const cv::Mat& image, std::function<bool(const cv::Vec3d&)>  pred, cv::Mat_<uchar>& mask);

struct IndependentSetValidationResult
{
	bool isValid;
	std::string message;
	int vertex1;
	int vertex2;
};

IndependentSetValidationResult validateMaximumWeightIndependentSet(const MatrixUndirectedGraph& graph, const std::vector<bool>& vertexSet);

double calculateVertexSetWeight(const MatrixUndirectedGraph& graph, const std::vector<bool>& vertexSet);

std::tuple<MatrixUndirectedGraph, std::vector<int>> createFromEdgeList(const std::vector<int>& vertices, const std::vector<int>& edgeListByRow);

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
