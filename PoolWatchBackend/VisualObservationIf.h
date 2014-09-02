#pragma once
#include "BgSubLib/ZivkovicAGMM.hpp"
#include <cstdint>

namespace PoolWatch
{
	// The number of components for a GMM color signature.
	const int ColorSignatureGmmMaxSize = 3;
}

typedef Algorithms::BackgroundSubtraction::ZivkovicAGMM::GMM GaussMixtureCompoenent;

/** Rectangular region of tracket target, which is detected in camera's image frame. */
struct DetectedBlob
{
	int Id;
	cv::Rect2f BoundingBox;
	cv::Point2f Centroid;
	// TODO: analyze usage to check whether to represent it as a vector of points?
	cv::Mat_<std::int32_t> OutlinePixels; // [Nx2], N=number of points; (Y,X) per row

	cv::Mat FilledImage; // [W,H] CV_8UC1 image contains only bounding box of this blob

	// used in appearance modeling
	cv::Mat FilledImageRgb; // [W,H] CV_8UC3 image contains only bounding box of this blob
	std::array<GaussMixtureCompoenent, PoolWatch::ColorSignatureGmmMaxSize> ColorSignature;
	int ColorSignatureGmmCount = 0;

	//
	cv::Point3f CentroidWorld;
	float AreaPix; // area of the blob in pixels
};

class SwimmerAppearanceModelBase
{
public:
	SwimmerAppearanceModelBase() = default;
	SwimmerAppearanceModelBase(const SwimmerAppearanceModelBase&) = delete;
	virtual ~SwimmerAppearanceModelBase() {}

	// Calculates the appearance score of a blob to change the color signature from 'gmm1' to 'gmm2'.
	virtual float appearanceScore(GaussMixtureCompoenent const* gmm1, int gmm1Size, GaussMixtureCompoenent const* gmm2, int gmm2Size) = 0;

	// Merges two GMMs and put the result into 'resultGmm'.
	virtual void mergeTwoGaussianMixtures(GaussMixtureCompoenent const* gmm1, int gmm1Size, GaussMixtureCompoenent const* gmm2, int gmm2Size, GaussMixtureCompoenent * resultGmm, int resultGmmMaxSize, int& resultGmmSize) = 0;
};
