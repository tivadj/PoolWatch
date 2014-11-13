#include "CoreUtils.h"
#include <ctime> // time_t
#include <cassert>
#include <sstream>

namespace PoolWatch
{
	std::string timeStampNow()
	{
		std::stringstream strBuf;

		time_t  t1 = time(0); // now time

		// Convert now to tm struct for local timezone
		tm now1;
		errno_t err = localtime_s(&now1, &t1);
		assert(err == 0);

		char buf[80];
		strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &now1); // 20120601070015

		return std::string(buf);
	}

	// transparentCol=color to avoid copying
	void copyTo(const cv::Mat& sourceImageRgb, const cv::Mat& sourceImageMask, cv::Vec3b transparentCol, std::vector<cv::Vec3b>& resultPixels)
	{
		CV_Assert(sourceImageRgb.type() == CV_8UC3);
		CV_Assert(sourceImageMask.type() == CV_8UC1);
		CV_Assert(sourceImageRgb.cols == sourceImageMask.cols);
		CV_Assert(sourceImageRgb.rows == sourceImageMask.rows);

		if (false && sourceImageRgb.isContinuous() && sourceImageMask.isContinuous())
		{
			cv::Mat blobPixList = sourceImageRgb.reshape(1, 3);
			cv::Mat blobMaskList = sourceImageMask.reshape(1, 1);
			assert(blobPixList.cols == blobMaskList.cols && "Image and mask must be of the same size");

			cv::Vec3b* pPixSrc = reinterpret_cast<cv::Vec3b*>(blobPixList.data);
			uchar* pMask = blobMaskList.data;
			for (int x = 0; x < blobMaskList.cols; ++x, ++pMask, ++pPixSrc)
			{
				bool isBlob = *pMask;
				if (isBlob)
				{
					resultPixels.push_back(*pPixSrc);
				}
			}
		}
		else
		{
			uchar* pSrcRowStart = sourceImageRgb.data;
			uchar* pSrcMaskRowStart = sourceImageMask.data;
			for (int y = 0; y < sourceImageRgb.rows; ++y)
			{
				cv::Vec3b* pSrcPix = reinterpret_cast<cv::Vec3b*>(pSrcRowStart);
				uchar* pSrcMask = pSrcMaskRowStart;

				for (int x = 0; x < sourceImageRgb.cols; ++x)
				{
					bool isBlob = *pSrcMask;
					if (isBlob)
					{
						resultPixels.push_back(*pSrcPix);
					}
					++pSrcPix;
					++pSrcMask;
				}

				pSrcRowStart += sourceImageRgb.step;
				pSrcMaskRowStart += sourceImageMask.step;
			}
		}

		// ensure all fore pixels were copied
		int sum1 = cv::countNonZero(sourceImageMask);
		CV_DbgAssert(sum1 == resultPixels.size());
	}
}
