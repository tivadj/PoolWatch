#pragma once
#include <QDir>
#include <opencv2/core.hpp>

#include "CameraProjector.h"
#include "PoolWatchFacade.h"

void configureLogToFileAppender(const QDir& logFolder, const QString& logFileName);

class LinearCameraProjector : public CameraProjectorBase
{
public:
	LinearCameraProjector();
	virtual ~LinearCameraProjector();

	cv::Point2f worldToCamera(const cv::Point3f& world) const override;
	cv::Point3f cameraToWorld(const cv::Point2f& imagePos) const override;
};

