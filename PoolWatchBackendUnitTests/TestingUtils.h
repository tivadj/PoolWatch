#pragma once
#include <string>
#include <tuple>

#include <opencv2/core.hpp>

#include <QDir>

#include <boost/filesystem/path.hpp>

#include <log4cxx/logger.h>

#include "PoolWatchFacade.h"
#include "KalmanFilterMovementPredictor.h"

boost::filesystem::path getSrcDir();
boost::filesystem::path getTestResultsDir();

boost::filesystem::path initTestMethodLogFolder(const std::string& className, const std::string& methodName);


struct LogFileAppenderUnsubscriber
{
	void operator()(log4cxx::helpers::ObjectPtrT<log4cxx::Appender>* pAppender) const;
};

std::unique_ptr<log4cxx::helpers::ObjectPtrT<log4cxx::Appender>, LogFileAppenderUnsubscriber> scopeLogFileAppenderNew(const boost::filesystem::path& logFolder);

class LinearCameraProjector : public CameraProjectorBase
{
public:
	LinearCameraProjector();
	virtual ~LinearCameraProjector();

	cv::Point2f worldToCamera(const cv::Point3f& world) const override;
	cv::Point3f cameraToWorld(const cv::Point2f& imagePos) const override;
};

class ConstantVelocityMovementPredictor : public SwimmerMovementPredictor
{
	cv::Point3f velocity_;
	float sigma_; // in N(mu,sigma^2) shows how actual and estimated position agree
public:
	ConstantVelocityMovementPredictor(const cv::Point3f& velocity);
	ConstantVelocityMovementPredictor(const ConstantVelocityMovementPredictor&) = delete;
	virtual ~ConstantVelocityMovementPredictor();

	void initScoreAndState(const cv::Point3f& blobCentrWorld, float& score, TrackHypothesisTreeNode& saveNode) override;

	void estimateAndSave(const TrackHypothesisTreeNode& curNode, const cv::Point3f& blobCentrWorld, cv::Point3f& estPos, float& score, TrackHypothesisTreeNode& saveNode) override;

private:
	// use Kalman Filter state matrix just to store position of a blob
	// state = [x y vx vy]' and (vx,vy) components are ignored
	const cv::Mat& nodeState(const TrackHypothesisTreeNode& node) const;
	      cv::Mat& nodeState(      TrackHypothesisTreeNode& node) const;
};

template <typename Cont, typename ExpectT, typename SelectFun>
std::tuple<bool, std::wstring> checkAll(const Cont& cont, ExpectT expectedValue, const SelectFun& actualFun)
{
	size_t index = 0;
	typename Cont::const_iterator& it = std::cbegin(cont);

	for (; it != std::cend(cont); ++it, ++index)
	{
		const auto& elem = cont[index];

		const auto& actual = actualFun(elem);
		if (expectedValue != actual)
		{
			std::wstringstream buf;
			buf << L"Failed on index=" << index << L" expected=" << expectedValue << L" actual=" << actual;
			return std::make_tuple(false, buf.str());
		}
	}
	return std::make_tuple(true, L"");
}
