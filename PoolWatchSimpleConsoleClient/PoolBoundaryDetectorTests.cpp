#include <memory>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> // imread
#include <opencv2/imgproc/types_c.h> // cv::cvtColor
#include <opencv2/video.hpp> // background subtractor
#include "VisualObservation.h"
#include "CoreUtils.h"

#include "VisualObservation.h"
#include "algos1.h"
#include "VideoLogger.h"

namespace SkinClassifierTestsNS
{
	void trainAndWriteSkinClassifier(WaterClassifier& wc);
}

namespace PoolBoundaryDetectorTestsNS
{
	using namespace cv;

	void testPoolBoundary()
	{
		cv::Mat i1 = cv::imread("data/MVI_3177_0127_640x476.png");
		
		// find water mask
		cv::FileStorage fs;
		if (!fs.open("1.yml", cv::FileStorage::READ))
			return;
		auto pWc = WaterClassifier::read(fs);
		WaterClassifier& wc = *pWc;

		cv::Mat_<uchar> i1WaterMask;
		classifyAndGetMask(i1, [&wc](const cv::Vec3d& pix) -> bool
		{
			return wc.predict(pix);
		}, i1WaterMask);

		cv::Mat_<uchar> poolMask;
		getPoolMask(i1, i1WaterMask, poolMask);

		cv::Mat i2;
		i1.copyTo(i2, poolMask);
	}

	void testRemoveWaterUsingLargestWeightWaterColorSignatureGmm(const cv::Mat& image)
	{
		cv::Mat pixsMat = image.reshape(3, image.rows * image.cols); // uchar[ch3 N,1]

		// remove transparent pixels
		std::vector<cv::Vec3b> pixsVector;
		pixsMat.copyTo(pixsVector);
		const cv::Vec3b transpPix(0, 0, 0);
		auto remIt = std::remove_if(std::begin(pixsVector), std::end(pixsVector), [=](const cv::Vec3b& pix) { return pix == transpPix; });
		pixsVector.erase(remIt, std::end(pixsVector));

		// EM training works with double type
		pixsMat = cv::Mat(pixsVector, false).reshape(1, pixsVector.size()); // uchar[N,3]
		cv::Mat pixsDouble; // double[N,3]
		pixsMat.convertTo(pixsDouble, CV_64FC1);

		auto termCrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 1000, 0.001);
		int nclusters = 2;
		EMQuick em(nclusters, cv::EM::COV_MAT_SPHERICAL, termCrit);
		bool trainOp = em.train(pixsDouble);
		CV_Assert(trainOp);

		auto weights = em.getWeights();
		double* pWeightsDbl = (double*)weights.data;
		double* pWeightMax = std::max_element(pWeightsDbl, pWeightsDbl + nclusters);
		int i = pWeightMax - pWeightsDbl;
		//int i = 0;

		std::array<float, 3> mean = { em.getMeans().at<double>(i, 0), em.getMeans().at<double>(i, 1), em.getMeans().at<double>(i, 2) };
		auto variance = em.getCovs()[i].at<double>(0, 0); // assumes diagonal spherical covariance matrix
		float sigma = std::sqrtf(variance);
		float thresh = 0.95;
		float thresh2 = sigma * 3;

		cv::Mat_<uchar> waterMask;
		float max = 0;
		classifyAndGetMask(image, [&mean, sigma, thresh, &max, thresh2](const cv::Vec3d& pix) -> bool
		{
			//float r = (float)pix[0];
			//float g = (float)pix[1];
			//float b = (float)pix[2];
			//std::array<float, 3> xFloat{ r, g, b };

			//float p1 = PoolWatch::normalProb(3, xFloat.data(), mean.data(), sigma);
			//max = std::max(max, p1);
			//return p1 > thresh;

			float dist = cv::norm(pix - cv::Vec3d(mean[0], mean[1], mean[2]));
			return dist < thresh2;
		}, waterMask);

		// try K-means clustering
		std::vector<cv::Vec3f> blockPixels;
		std::transform(std::begin(pixsVector), std::end(pixsVector), std::back_inserter(blockPixels) ,[](const cv::Vec3b& v) { return cv::Vec3f(v[0], v[1], v[2]); });
		std::vector<int> bestLabels;
		cv::TermCriteria term(cv::TermCriteria::EPS, 0, 0.001);
		cv::Mat colorCenters; // float Kx3, size(rgb)=3
		const int K = 1;
		auto compactness = cv::kmeans(blockPixels, K, bestLabels, term, 1, cv::KMEANS_RANDOM_CENTERS, colorCenters);

		std::cout << "test" << std::endl;
	}

	void removeWater()
	{
		//cv::Mat image = cv::imread("../../dinosaur/poolBoundary/MVI_4636_640x480_Frame1.png");
		cv::Mat image = cv::imread("../../dinosaur/pool1.png");

		testRemoveWaterUsingLargestWeightWaterColorSignatureGmm(image);
	}

	void testDetectSwimmingLanesWithBackgroundSubtractorOpenCV()
	{
		auto videoPathRel = "../../dinosaur/mvi_4636_640x480.mp4";
		boost::filesystem::path videoPath = boost::filesystem::absolute(videoPathRel).normalize();
		cv::VideoCapture videoCapture(videoPath.string());
		if (!videoCapture.isOpened())
			return;

		cv::Mat image;
		cv::Mat imageGray;
		cv::Mat foreMask;
		cv::Mat bgMask;
		cv::Mat foreMaskBin;
		cv::Mat foreImage;
		cv::Mat bgImage;
		cv::Mat bgImageGray;
		const int HistSize = 100;
		//auto bs = createBackgroundSubtractorMOG(HistSize, 5, 0.7, 0); // doesn't support background image
		auto bs = createBackgroundSubtractorMOG2(HistSize, 16, false);
		//auto bs = createBackgroundSubtractorKNN(HistSize, 400, false); bs->setkNNSamples(3);
		//auto bs = createBackgroundSubtractorGMG(120, 0.8); // doesn't support background image

		//
		float fps = (float)videoCapture.get(CAP_PROP_FPS);
		auto outDir = boost::filesystem::absolute("../output/!default/");
		VideoLogger::init(outDir, fps);

		int frameInd = 0;
		while (true)
		{
			//if (frameInd > 600)
			//	break;
			for (int i = 0; i < 100; ++i)
			{
				bool readOp = videoCapture.read(image);
				if (!readOp)
					return;
				frameInd++;
				//auto learningRate = 0.005;
				auto learningRate = -1;
				bs->apply(image, foreMask, learningRate);
				image.copyTo(foreImage, foreMask);
				VideoLogger::logDebug("mvi_4636_640x480_MOG2_thr16_fg", foreImage);

				// NOTE: not supported by MOG; supported by MOG2,KNN
				if (true)
				{
					bs->getBackgroundImage(bgImage);
				}
				else
				{
					cv::bitwise_not(foreMask, bgMask);
					image.copyTo(bgImage, bgMask);
				}

				VideoLogger::logDebug("mvi_4636_640x480_MOG2_thr16_bg", bgImage);
			}
			

			std::cout << "frameInd=" << frameInd << std::endl;
			//auto lanesOp = getSwimLanes(bgImage);
			
		}
	}

	void testDetectSwimmingLanesWithBackgroundSubtractorCustom()
	{
		auto videoPathRel = "../../dinosaur/mvi_4636_640x480.mp4";
		boost::filesystem::path videoPath = boost::filesystem::absolute(videoPathRel).normalize();
		cv::VideoCapture videoCapture(videoPath.string());
		if (!videoCapture.isOpened())
			return;

		cv::Mat image;
		cv::Mat imageGray;
		cv::Mat foreMask;
		cv::Mat foreMaskBin;
		cv::Mat foreImage;
		cv::Mat bgImage;
		cv::Mat bgImageGray;
		const int HistSize = 100;
		PoolWatch::MedianBackgroundModel<HistSize> medianBg;
		
		//{ Median Tile
		//const int ClustCount = 3;
		//const int BlockSize = 3;
		//PoolWatch::BlockMedianBackgroundModel<ClustCount> blockBgModel(HistSize, cv::Size2i(BlockSize, BlockSize));

		//int frameWidth = (int)videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
		//int frameHeight = (int)videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
		//blockBgModel.setUpImageParameters(frameWidth, frameHeight);
		//}

		//std::unique_ptr<WaterClassifier> pc;
		//cv::FileStorage fs;
		//if (!fs.open("skin_clasifier.yml", cv::FileStorage::READ))
		//{
		//	pc = std::make_unique<WaterClassifier>(10, cv::EM::COV_MAT_SPHERICAL);
		//	SkinClassifierTestsNS::trainAndWriteSkinClassifier(*pc);

		//	if (!fs.open("skin_clasifier.yml", cv::FileStorage::WRITE))
		//		return;
		//	pc->write(fs);
		//	fs.release();
		//}
		//else
		//{
		//	pc = WaterClassifier::read(fs);
		//}

		//WaterClassifier& wc = *pc;

		//auto fleshClassifFun = [&wc](const cv::Vec3d& pix) -> bool
		//{
		//	return wc.predict(pix);
		//};

		//

		int frameInd = 0;
		while (true)
		{
			for (int i = 0; i < 100; ++i)
			{
				bool readOp = videoCapture.read(image);
				if (!readOp)
					return;
				frameInd++;

				//

				//cv::Mat_<uchar> bgMask;
				//classifyAndGetMask(image, fleshClassifFun, bgMask);
				//cv::subtract(255, bgMask, bgMask);
				//cv::Mat bgImage;
				//image.copyTo(bgImage, bgMask);

				//blockBgModel.apply(bgImage);
				//medianBg.apply(image);

				if (false && frameInd >= HistSize)
				{
					//medianBg.getBackgroundImage(bgImage);
					//blockBgModel.getBackgroundImage(bgImage);
					//blockBgModel.getForeMask(image, foreImage);

					//double minValue,maxValue;
					//cv::minMaxLoc(foreImage, &minValue, &maxValue, nullptr, nullptr, cv::noArray());

					//double alpha = 255 / maxValue;
					//cv::Mat foreImageScaled;
					//foreImage.convertTo(foreImageScaled, -1, alpha, 0);

					cv::threshold(foreImage, foreMaskBin, 1, 255, THRESH_BINARY);

					cv::cvtColor(image, imageGray, CV_BGR2GRAY);
					cv::cvtColor(bgImage, bgImageGray, CV_BGR2GRAY);
					cv::absdiff(imageGray, bgImageGray, foreMask);
					cv::threshold(foreMask, foreMaskBin, -1, 255, THRESH_BINARY | THRESH_OTSU);
				}
			}

			std::cout << "frameInd=" << frameInd << std::endl;
			//auto lanesOp = getSwimLanes(bgImage);
		}
	}

	void testDetectSwimmingLanes()
	{
		//cv::Mat image = cv::imread("../../dinosaur/poolBoundary/MVI_4636_640x480_Frame1.png");
		cv::Mat image = cv::imread("../../artefacts/SwimLaneSeparator/pool_clean1.png");

		std::vector<std::vector<cv::Point2f>> swimLanes;
		auto lanesOp = getSwimLanes(image, swimLanes);
	}

	void run()
	{
		//testPoolBoundary();
		//removeWater();
		testDetectSwimmingLanesWithBackgroundSubtractorOpenCV();
		//testDetectSwimmingLanesWithBackgroundSubtractorCustom();
		//testDetectSwimmingLanes();
	}
}