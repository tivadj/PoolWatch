#pragma  once
#include <vector>
#include <tuple>
#include <functional>

#include <opencv2\core.hpp>

#include "PoolWatchFacade.h"
#include "MatrixUndirectedGraph.h"
#include "WaterClassifier.h"

PW_EXPORTS void classifyAndGetMask(const cv::Mat& image, std::function<bool(const cv::Vec3d&)>  pred, cv::Mat_<uchar>& mask);
PW_EXPORTS void estimateClassifier(const cv::Mat& image, std::function<double(const cv::Vec3d&)> computeOne, cv::Mat_<double>& mask);

PW_EXPORTS void maximumWeightIndependentSetNaiveMaxFirst(const MatrixUndirectedGraph& graph, std::vector<uchar>& vertexSet);
PW_EXPORTS void maximumWeightIndependentSetNaiveMaxFirstMultipleSeeds(const MatrixUndirectedGraph& graph, std::vector<uchar>& vertexSet);

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

	PW_EXPORTS auto deg2rad(const float& degree) -> float;
	PW_EXPORTS auto normalProb(float x, float mu, float sigma) -> float;
	PW_EXPORTS auto normalProb(int spaceDim, const float* pX, float* pMu, float sigma) -> float;

	// Finds the intersection of two lines, or returns false.
	// The lines are defined by (o1, p1) and (o2, p2).
	bool intersectLines(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point2f& r);
	
	// Finds the distance of the perpendicular from point freePoint to line (p1,p2).
	// Returns false if two points are the same.
	bool distLinePoint(cv::Point2f p1, cv::Point2f p2, cv::Point2f freePoint, float& dist);
}

// TODO: this should have an internal usage
class PW_EXPORTS EMQuick : public cv::EM
{
public:
	EMQuick(int nclusters, int covMatType);
	EMQuick(int nclusters, int covMatType, const cv::TermCriteria& termCrit);
	EMQuick(const EMQuick&) = delete;
	static cv::Vec2d predict2(cv::InputArray _sample, const cv::Mat& meansPar, const std::vector<cv::Mat>& invCovsEigenValuesPar, const cv::Mat& logWeightDivDetPar, cv::Mat& cacheL);
	static cv::Vec2d computeProbabilitiesInplace(cv::Mat& sample, const cv::Mat& meansPar, const std::vector<cv::Mat>& invCovsEigenValuesPar, const cv::Mat& logWeightDivDetPar, cv::Mat& cacheL);

	int getNClusters()
	{
		return this->nclusters;
	}
	const cv::Mat& getWeights()
	{
		return this->weights;
	}
	const cv::Mat& getMeans()
	{
		return this->means;
	}
	const std::vector<cv::Mat>& getCovs()
	{
		return this->covs;
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


