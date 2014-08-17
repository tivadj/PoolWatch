#pragma once
#include <opencv2/core/mat.hpp>
#include "BgSubLib/ZivkovicAGMM.hpp"

typedef Algorithms::BackgroundSubtraction::ZivkovicAGMM::GMM GaussMixtureCompoenent;

class __declspec(dllexport) AppearanceModel
{
	int numModes_;
	Algorithms::BackgroundSubtraction::ZivkovicAGMM bgs_;
	BwImage low_threshold_mask_;
	BwImage high_threshold_mask_;
	cv::Mat_<cv::Vec3b> singlePixelMat_;
	//RgbImage singlePixelRgbImage_;
public :
	AppearanceModel(int numModes);
	AppearanceModel(const AppearanceModel&) = delete;
	virtual ~AppearanceModel();

	void initGmmComponents();
	void loadGmmComponents(const GaussMixtureCompoenent* gmmsList, int gmmsListSize);
	void pushNextPixels(const cv::Mat& blobRgb, const cv::Vec3b& transparentPix);
	void saveGmmComponents(GaussMixtureCompoenent* gmmsList, int gmmsListSize, int& newGmmSize);
};

// Computes GMM probability value in the point xs.
float evalGmm(const float* xs, GaussMixtureCompoenent const* gmms, int gmmCount);

// Computes the distance between two Gaussian Mixture Models (GMM) using naive sampling of the RGB space with step tableStep
// for all directions. Slow.
// source: "Evaluation of distance measures between gaussian mixture models of MFCCS", Jensen
__declspec(dllexport) float normalizedL2DistanceNaive(GaussMixtureCompoenent const* gmm1, int gmm1Size, GaussMixtureCompoenent const* gmm2, int gmm2Size, int tableStep);

// Computes the distance between two Gaussian Mixture Models (GMM) using close formula for integration of the product of gaussians.
// source: "The Multivariate Gaussian Probability Distribution", Ahrendt, 2005, formulas (5.2) and (5.6)
__declspec(dllexport) bool normalizedL2Distance(GaussMixtureCompoenent const* gmm1, int gmm1Size, GaussMixtureCompoenent const* gmm2, int gmm2Size, float& resultDistance);

// Merges two gmms with learning rate applied to the second gmm.
__declspec(dllexport) void mergeTwoGaussianMixtures(GaussMixtureCompoenent const* gmm1, int gmm1Size, GaussMixtureCompoenent const* gmm2, int gmm2Size, float maxRidgeRatio, float componentMinWeight, float learningRate,
	GaussMixtureCompoenent * resultGmm, int resultGmmMaxSize, int& rsultGmmSize);

// Merges similar comonents in one GMM.
__declspec(dllexport) void mergeGaussianMixtureComponents(const GaussMixtureCompoenent* gmm, int gmmSize, float maxRidgeRatio, float componentMinWeight, GaussMixtureCompoenent * resultGmm, int resultGmmMaxSize, int& rsultGmmSize);

// Ensures that GMM has the total sum of weights for each GMM component equals one.
void fixGmmWeights(GaussMixtureCompoenent* gmm, int gmmSize);

