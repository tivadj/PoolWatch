#include <vector>
#include <type_traits> // remove_reference

#include "opencv2\core.hpp"
#include "opencv2\core\core_c.h" // CV_FILLED
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"

#include <QtDebug>
#include <QDomDocument>
#include <QFile>
#include <QDir>

#include "PoolWatchFacade.h"

using namespace std;

void loadImageAndMask(const std::string& svgFilePath, const std::string& strokeColor, cv::Mat& outImage, cv::Mat_<bool>& outMask)
{
	QFile file(svgFilePath.c_str());
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
		return;

	QDomDocument xml;
	if (!xml.setContent(&file))
		return;

	file.close();

	QDomElement docElem = xml.documentElement();
	qDebug() << docElem.tagName();

	//

	QDomNodeList nodeList = docElem.childNodes();
	for (int i = 0; i<nodeList.count(); ++i)
	{
		QDomNode n = nodeList.at(i);
		QDomElement e = n.toElement();
		if (e.isNull())
			continue;
		if (e.tagName() == "image")
		{
			int width = e.attribute("width").toInt();
			int height = e.attribute("width").toInt();
			QString fileName = e.attribute("xlink:href");
			qDebug() <<"image " <<width <<"x" <<height <<" name=" <<fileName;

			// abs svg path
			QDir dir(svgFilePath.c_str());
			auto dirUp = dir.cdUp();
			assert(dirUp);
			QString svgAbsPath = dir.absoluteFilePath(fileName);
			qDebug() <<svgAbsPath;


			auto image = cv::imread(svgAbsPath.toStdString());
			outImage = image;
			outMask = cv::Mat_<bool>::zeros(outImage.rows, outImage.cols);
		}
		else if (e.tagName() == "polygon")
		{
			QString strokeStr = e.attribute("stroke");
			if (!strokeColor.empty() && strokeStr.toStdString() != strokeColor)
				continue;

			QString pointsStr = e.attribute("points");
			qDebug() << "polygon stroke=" << strokeStr << "points=" << pointsStr;

			QRegExp rx(R"([,\s])"); // separate by space and comma
			QStringList stringList = pointsStr.split(rx, QString::SkipEmptyParts);
			vector<float> pointsXY(stringList.size());
			std::transform(begin(stringList), end(stringList), begin(pointsXY), [](const QString s)
			{
				return s.toFloat();
			});

			CV_Assert(pointsXY.size() % 2 == 0 && "There must be list of (X,Y) pairs");

			vector<cv::Point2i> points;
			points.resize(pointsXY.size() / 2);
			int index = 0;
			for (size_t i = 0; i < points.size(); ++i)
			{
				cv::Point2f pnt;
				pnt.x = pointsXY[index++];
				pnt.y = pointsXY[index++];
				points[i] = pnt;
			}

			//
			assert(!outImage.empty() && "Svg file must have the first tag to be the image tag");
			assert(!outMask.empty());

			vector<vector<cv::Point2i>> contoursList(1);
			contoursList[0] = std::move(points);
			cv::drawContours(outMask, contoursList, 0, cv::Scalar::all(255), CV_FILLED);
		}
		else
		{
			qDebug() << e.tagName();
		}
	}

}

void loadWaterPixelsOne(const QString& svgFilePath, const std::string& strokeStr, std::vector<cv::Vec3d>& pixels, bool invertMask, int inflateContourDelta)
{
	cv::Mat image;
	cv::Mat_<bool> maskOrig;
	loadImageAndMask(svgFilePath.toStdString(), strokeStr, image, maskOrig);

	cv::Mat_<bool> mask;;
	if (inflateContourDelta != 0)
	{
		int op = inflateContourDelta > 0 ? cv::MORPH_DILATE : cv::MORPH_ERODE;

		int r = inflateContourDelta;
		auto sel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(r, r));
		cv::morphologyEx(maskOrig, mask, op, sel);
	}
	else
		mask = maskOrig;

	cv::Mat maskContinous = mask.reshape(1, 1);
	cv::Mat imageContinous = image.reshape(1, 1);

	bool* pPixFlag = reinterpret_cast<bool*>(maskContinous.data);
	uchar* pPix = reinterpret_cast<uchar*>(imageContinous.data);

	pixels.reserve(maskContinous.cols);
	
	for (int i = 0; i < maskContinous.cols; ++i)
	{
		bool usePixel = pPixFlag[i];
		if (invertMask)
			usePixel = !usePixel;
		if (usePixel)
		{
			cv::Vec3d pix(pPix[0], pPix[1], pPix[2]);
			pixels.push_back(pix);

			//if (pixels.size() > 1000)
			//	return;
		}
		// static_assert(std::remove_reference<decltype(pixels.front())>::type == cv::Vec3d);
		//static_assert(std::remove_reference<decltype(pixels.front())>::type::channels == 3);
		//static_assert(decltype(pixels.front())::channels == 3);
		pPix += 3;
	}
}

void loadWaterPixels(const std::string& folderPath, const std::string& svgFilter, const std::string& strokeStr, std::vector<cv::Vec3d>& pixels, bool invertMask, int inflateContourDelta)
{
	QFileInfo fileInfo = QFileInfo(QString(folderPath.c_str()));
	if (fileInfo.isDir())
	{
		QDir dir = fileInfo.dir();
		
		QStringList filterList;
		filterList.append(svgFilter.c_str());
		
		QStringList files = dir.entryList(filterList);
		for (int i = 0; i < files.count(); ++i)
		{
			QString svgAbsPath = dir.absoluteFilePath(files[i]);
			loadWaterPixelsOne(svgAbsPath, strokeStr, pixels, invertMask, inflateContourDelta);
		}
	}
	else if (fileInfo.isFile())
	{
		loadWaterPixelsOne(fileInfo.absoluteFilePath(), strokeStr, pixels, invertMask, inflateContourDelta);
	}
	else
		return;
}
