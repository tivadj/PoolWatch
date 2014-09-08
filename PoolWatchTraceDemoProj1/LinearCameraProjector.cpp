#pragma once
#include "DemoHelpers.h"

LinearCameraProjector::LinearCameraProjector()
{
	
}

LinearCameraProjector::~LinearCameraProjector()
{
	
}

cv::Point2f LinearCameraProjector::worldToCamera(cv::Point3f const& world) const
{
	return cv::Point2f(world.x, world.y);
}

cv::Point3f LinearCameraProjector::cameraToWorld(cv::Point2f const& imagePos) const
{
	return cv::Point3f(imagePos.x, imagePos.y, 0);
}