#pragma once
#include <string>
#include <functional>
#include <numeric>
#include <array>
#include <vector>
#include <cassert>
#include <random>
#include "opencv2/core.hpp"

#include "PoolWatchFacade.h"

namespace PoolWatch
{
	PW_EXPORTS std::string timeStampNow();

	template <typename PointLikeT>
	void convertPointList(const std::vector<cv::Point2f>& polygon, std::vector<PointLikeT>& outList)
	{
		outList.resize(polygon.size());
		for (size_t i = 0; i < polygon.size(); ++i)
		{
			outList[i] = polygon[i];
		}
	}

	void copyTo(const cv::Mat& sourceImageRgb, const cv::Mat& sourceImageMask, cv::Vec3b transparentCol, std::vector<cv::Vec3b>& resultPixels);

	//

	// Represents buffer of elements with cyclic sematics. When new element is requested from buffer, the reference to
	// already allocated element is returned.
	// Use 'queryHistory' method to get old elements.
	template <typename T>
	struct CyclicHistoryBuffer
	{
	private:
		std::vector<T> cyclicBuffer_;
		int freeFrameIndex_;
	public:
		CyclicHistoryBuffer(int bufferSize)
			:freeFrameIndex_(0),
			cyclicBuffer_(bufferSize)
		{
		}

		// initializes each element of the buffer
		auto init(std::function<void(size_t index, T& item)> itemInitFun) -> void
		{
			for (size_t i = 0; i < cyclicBuffer_.size(); ++i)
				itemInitFun(i, cyclicBuffer_[i]);
		}

		auto queryHistory(int indexBack) -> T&
		{
			assert(indexBack <= 0);

			// 0(current) = next free element to return on request
			// -1 = last valid data
			int ind = -1 + freeFrameIndex_ + indexBack;
			if (ind < 0)
				ind += (int)cyclicBuffer_.size();

			assert(ind >= 0 && "Buffer index is out of range");
			assert(ind < cyclicBuffer_.size());

			return cyclicBuffer_[ind];
		};

		auto requestNew() -> T&
		{
			auto& result = cyclicBuffer_[freeFrameIndex_];

			freeFrameIndex_++;
			if (freeFrameIndex_ >= cyclicBuffer_.size())
				freeFrameIndex_ = 0;

			return result;
		};
	};

	// Represents the background image as the median of the last HistorySize frames.
	template <int HistorySize>
	struct MedianBackgroundModel
	{
	private:
		std::array<cv::Mat, HistorySize> historyImages_;
		int nextImageInd_ = 0;
		std::array<uchar, HistorySize> bufferRed_;
		std::array<uchar, HistorySize> bufferGreen_;
		std::array<uchar, HistorySize> bufferBlue_;
	public:
		void apply(const cv::Mat& image)
		{
			cv::Mat& bufImage = historyImages_[nextImageInd_];
			nextImageInd_ = (nextImageInd_+1) % HistorySize;

			//bufImage.create(image.rows, image.cols, image.type());

			image.copyTo(bufImage);
		}

		void getBackgroundImage(cv::Mat& outBgImage)
		{
			int rowsCount = historyImages_[0].rows;
			int colsCount = historyImages_[0].cols;

			outBgImage.create(rowsCount, colsCount, CV_8UC3);

			for (int y = 0; y < rowsCount; ++y)
			for (int x = 0; x < colsCount; ++x)
			{
				cv::Vec3b medianPix = getMedianPixel(y, x);
				outBgImage.at<cv::Vec3b>(y, x) = medianPix;
			}
		}

	private:
		cv::Vec3b getMedianPixel(int y, int x)
		{
			for (size_t histInd = 0; histInd < historyImages_.size(); ++histInd)
			{
				cv::Vec3b pix = historyImages_[histInd].at<cv::Vec3b>(y, x);

				bufferRed_[histInd] = pix(0);
				bufferGreen_[histInd] = pix(1);
				bufferBlue_[histInd] = pix(2);
			}

			size_t midInd = historyImages_.size() / 2;

			std::nth_element(std::begin(bufferRed_), std::begin(bufferRed_) + midInd, std::end(bufferRed_));
			std::nth_element(std::begin(bufferGreen_), std::begin(bufferGreen_) + midInd, std::end(bufferGreen_));
			std::nth_element(std::begin(bufferBlue_), std::begin(bufferBlue_) + midInd, std::end(bufferBlue_));

			cv::Vec3b result(bufferRed_[midInd], bufferGreen_[midInd], bufferBlue_[midInd]);
			return result;
		}
	};

	// Block - based model:
	// x may speed up computation because color appearance is stored for the group of pixels
	// x may account for movement when computing difference image(if comparing pixel valuewith corresponding 9 neighbour blocks)
	// block-based model idea from "An automatic drowning detection surveillance system for challenging outdoor", How-Lung, ICCV 2003
	template <size_t ClusterCountPerBlock>
	class BlockMedianBackgroundModel
	{
	public:
		typedef std::array<cv::Vec3b, ClusterCountPerBlock> ColorClustersPerBlockType;

	private:
		int nextImageInd_ = 0;
		cv::Size2i blockSize_;
		int histSize_;
		std::vector<cv::Mat> historyImages_;
		int imageWidth_;
		int imageHeight_;

		std::vector<ColorClustersPerBlockType> bgBlockModel_;
		//cv::Mat_<ColorClustersPerBlockType> bgBlockModel_;
		//cv::Mat bgBlockModel_;
		int blockCountHorz_;
		int blockCountVert_;
	public:
		BlockMedianBackgroundModel(int histSize, cv::Size2i blockSize) 
			: histSize_(histSize),
			blockSize_(blockSize)
			//clusterCountPerTile_(clusterCountPerTile)
		{
		}

		void setUpImageParameters(int width, int height)
		{
			historyImages_.resize(histSize_);
			for (int i = 0; i < histSize_; ++i)
				historyImages_[i].create(height, width, CV_8UC3);

			blockCountHorz_ = 1 + (width-1) / (float)blockSize_.width;
			blockCountVert_ = 1 + (height-1) / (float)blockSize_.height;
			bgBlockModel_.resize(blockCountHorz_ * blockCountVert_);

			imageWidth_ = width;
			imageHeight_ = height;
		}

		cv::Mat& requestBufImage()
		{
			int imgInd = nextImageInd_;
			nextImageInd_ = (nextImageInd_ + 1) % histSize_;

			return historyImages_[imgInd];
		}

		void apply(const cv::Mat& image)
		{
			cv::Mat& bufImage = historyImages_[nextImageInd_];
			nextImageInd_ = (nextImageInd_ + 1) % histSize_;

			image.copyTo(bufImage);
		}

		void getBackgroundImage(cv::Mat& outBgImage)
		{
			outBgImage.create(imageHeight_, imageWidth_, CV_8UC3);

			buildBackgroundTiledModel();

			std::default_random_engine dre(111);
			std::uniform_int_distribution<int> rand(0, 2); // inclusive[min,max]

			for (int row = 0; row < blockCountVert_; ++row)
			{
				int yMin = row * blockSize_.height;
				int yMax = std::min((row + 1)*blockSize_.height, imageHeight_);

				for (int col = 0; col < blockCountHorz_; ++col)
				{
					int blockUnaryIndex = row * blockCountHorz_ + col;
					const auto& colClusters = bgBlockModel_[blockUnaryIndex];

					// propogate these clusters into a background tile
					
					int xMin = col * blockSize_.width;
					int xMax = std::min((col + 1)*blockSize_.width, imageWidth_);
					for (int y = yMin; y < yMax; ++y)
					{
						for (int x = xMin; x < xMax; ++x)
						{
							//int colorInd = rand(dre);
							int colorInd = 0;
							const cv::Vec3b& color = colClusters[colorInd];
							outBgImage.at<cv::Vec3b>(y, x) = color;
						}
					}
				}
			}
		}

		void getForeMask(const cv::Mat& image, cv::Mat& foreMask)
		{
			CV_Assert(image.rows == imageHeight_);
			CV_Assert(image.cols == imageWidth_);

			foreMask.create(imageHeight_, imageWidth_, CV_8UC1);

			int maxDiffDist = 0;
			std::vector<int> distToBlocks;
			distToBlocks.reserve(3 * 3); // analyze 3x3 neighbour blocks to account for movement

			for (size_t pixY = 0; pixY < image.rows; ++pixY)
			{
				int blockY = pixY / blockSize_.height;

				for (size_t pixX = 0; pixX < image.cols; ++pixX)
				{
					auto minDistanceFromClusterCentroidFun = [=,&image](int blockY, int blockX, const cv::Vec3b& pix) -> int
					{
						int blockUnaryIndex = this->getBlockUnaryIndex(blockY, blockX);
						const auto& block = this->bgBlockModel_[blockUnaryIndex];

						// 
						std::array<int, ClusterCountPerBlock> distToClusters; // distances to each cluster in the given block
						for (size_t i = 0; i < distToClusters.size(); ++i)
						{
							const cv::Vec3b& clusterCenter = block[i];
							int distL1 = std::abs(clusterCenter[0] - pix[0]) + std::abs(clusterCenter[1] - pix[1]) + std::abs(clusterCenter[2] - pix[2]);
							distToClusters[i] = distL1;
						}

						// sum(rgb) <= 255*3
						int distL1Min = *std::min_element(std::begin(distToClusters), std::end(distToClusters));

						// fit value into uchar
						distL1Min /= 3;

						return distL1Min;
					};

					int blockX = pixX / blockSize_.width;
					const cv::Vec3b& pix = image.at<cv::Vec3b>(pixY, pixX);

					// analyze 3x3 neighbour blocks to account for movement
					distToBlocks.clear();
					for (int neighIndY = -1; neighIndY <= 1; ++neighIndY)
					{
						int neighBlockY = blockY + neighIndY;
						if (neighBlockY < 0 || neighBlockY >= blockCountVert_)
							continue;
						for (int neighIndX = -1; neighIndX <= 1; ++neighIndX)
						{
							int neighBlockX = blockX + neighIndX;
							if (neighBlockX < 0 || neighBlockX >= blockCountHorz_)
								continue;
							int distL1Min = minDistanceFromClusterCentroidFun(neighBlockY, neighBlockX, pix);
							distToBlocks.push_back(distL1Min);
						}
					}

					int distMin = *std::min_element(std::begin(distToBlocks), std::end(distToBlocks));
					maxDiffDist = std::max(maxDiffDist, distMin);
					foreMask.at<uchar>(pixY, pixX) = (uchar)distMin;
				}
			}

			if (maxDiffDist > 1)
			{
				double alpha = 255 / maxDiffDist;
				foreMask.convertTo(foreMask, -1, alpha, 0);
			}
		}

	private:
		void buildBackgroundTiledModel()
		{
			std::vector<cv::Vec3f> blockPixels;
			blockPixels.reserve(blockSize_.area() * histSize_);
			std::vector<int> bestLabels;
			cv::Mat colorCenters; // float Kx3, size(rgb)=3

			bgBlockModel_.clear();

			for (int blockRow = 0; blockRow < blockCountVert_; ++blockRow)
			{
				int yMin = blockRow * blockSize_.height;
				int yMax = std::min((blockRow + 1)*blockSize_.height, imageHeight_);

				for (int blockCol = 0; blockCol < blockCountHorz_; ++blockCol)
				{
					int xMin = blockCol * blockSize_.width;
					int xMax = std::min((blockCol + 1)*blockSize_.height, imageWidth_);

					// populate the tile's pixels through the history

					blockPixels.clear();
					for (size_t histInd = 0; histInd < historyImages_.size(); ++histInd)
					{
						const auto& histImg = historyImages_[histInd];

						for (int y = yMin; y < yMax; ++y)
						{
							for (int x = xMin; x < xMax; ++x)
							{
								auto pix = histImg.at<cv::Vec3b>(y, x);
								blockPixels.push_back(cv::Vec3f(pix[0],pix[1],pix[2]));
							}
						}
					}

					//

					cv::TermCriteria term(cv::TermCriteria::EPS, 0, 0);
					auto compactness = cv::kmeans(blockPixels, ClusterCountPerBlock, bestLabels, term, 1, cv::KMEANS_RANDOM_CENTERS, colorCenters);
					CV_Assert(colorCenters.rows == ClusterCountPerBlock && "Must get K clusters");

					//
					ColorClustersPerBlockType centroids;
					for (size_t i = 0; i < centroids.size(); ++i)
					{
						// convert from float centroid to UChar8 by truncation
						//centroids[i] = cv::Vec3b(colorCenters.at<float>(i, 1), colorCenters.at<float>(i, 1), colorCenters.at<float>(i, 2));

						uchar r = (uchar)std::roundf(colorCenters.at<float>(i, 0));
						uchar g = (uchar)std::roundf(colorCenters.at<float>(i, 1));
						uchar b = (uchar)std::roundf(colorCenters.at<float>(i, 2));
						centroids[i] = cv::Vec3b(r, g, b);
					}

					bgBlockModel_.push_back(centroids);
				}
			}
		}

		int getBlockUnaryIndex(int blockY, int blockX) { return blockY * blockCountHorz_ + blockX; }
	};
}
