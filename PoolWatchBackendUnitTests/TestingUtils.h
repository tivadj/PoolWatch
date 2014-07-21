#pragma once
#include <string>
#include <tuple>

#include <opencv2/core.hpp>

#include <QDir>

#include <boost/filesystem/path.hpp>
#include <boost/optional.hpp>

#include <log4cxx/logger.h>

#include "PoolWatchFacade.h"
#include "KalmanFilterMovementPredictor.h"
#include <SwimmingPoolObserver.h>

boost::filesystem::path getSrcDir();
boost::filesystem::path getTestResultsDir();

boost::filesystem::path initTestMethodLogFolder(const std::string& className, const std::string& methodName);
void PoolWatchBackendUnitTests_MethodInitilize();

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


// Data to indirectly compute the FamilyId. For example, if we know ObservationInd in a given frame FrameInd, we can uniquely infer 
// the corresponding FamilyId.
struct FamilyIdHint
{
	int FrameInd;
	int ObservationInd;
	FamilyIdHint(int frameInd, int observationInd) : FrameInd(frameInd), ObservationInd(observationInd) { }
};

class ConstantVelocityMovementPredictor : public SwimmerMovementPredictor
{
	struct SwimmerState
	{
		float x;
		float y;
		float vx;
		float vy;
	};
	struct TrackVelocity
	{
		::FamilyIdHint FamilyIdHint;
		cv::Point3f Velocity;
		TrackVelocity(::FamilyIdHint familyIdHint, const cv::Point3f& velocity) : FamilyIdHint(familyIdHint), Velocity(velocity) {}
	};
	cv::Point3f defaultVelocity_; // the speed of any swimmer, if exact value is not specified using 'getSwimmerVelocity'
	float sigma_; // in N(mu,sigma^2) shows how actual and estimated position agree
	std::vector<TrackVelocity> trackVelocityList_; // list of artificially specified swimmer velocities
public:
	ConstantVelocityMovementPredictor(const cv::Point3f& defaultVelocity);
	ConstantVelocityMovementPredictor(const ConstantVelocityMovementPredictor&) = delete;
	virtual ~ConstantVelocityMovementPredictor();

	void initScoreAndState(int frameInd, int observationInd, const cv::Point3f& blobCentrWorld, float& score, TrackHypothesisTreeNode& saveNode) override;

	void estimateAndSave(const TrackHypothesisTreeNode& curNode, const boost::optional<cv::Point3f>& blobCentrWorld, cv::Point3f& estPos, float& score, TrackHypothesisTreeNode& saveNode) override;
	
	void setSwimmerVelocity(FamilyIdHint familyIdHint, const cv::Point3f& velocity);
	boost::optional<cv::Point3f> getSwimmerVelocity(int frameInd, int observationInd) const;
private:
	const SwimmerState& nodeState(const TrackHypothesisTreeNode& node) const;
	      SwimmerState& nodeState(TrackHypothesisTreeNode& node) const;
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

template <typename Cont, typename ExpectT, typename SelectFun>
std::tuple<bool, std::wstring> checkAll(const Cont& cont, ExpectT expectedValue, const SelectFun& actualFun, ExpectT tolerance)
{
	size_t index = 0;
	typename Cont::const_iterator& it = std::cbegin(cont);

	for (; it != std::cend(cont); ++it, ++index)
	{
		const auto& elem = cont[index];

		const auto& actual = actualFun(elem);
		if (!(std::abs(expectedValue - actual) < tolerance))
		{
			std::wstringstream buf;
			buf << L"Failed on index=" << index << L" expected=" << expectedValue << L" actual=" << actual <<L" tolerance=" <<tolerance;
			return std::make_tuple(false, buf.str());
		}
	}
	return std::make_tuple(true, L"");
}

namespace Details
{
	template <typename ContT>
	auto dumpCont(std::wstringstream& buf, const ContT& cont) -> void
	{
		for (auto el : cont)
			buf << el << " ";
	};
}

template <typename ContT, typename ContT2, typename ExpectT>
std::tuple<bool, std::wstring> sequenceEqual(const ContT& actualCont, ContT2 expectCont, ExpectT tolerance)
{
	size_t index = 0;
	typename ContT::const_iterator& it1 = std::cbegin(actualCont);
	typename ContT2::const_iterator& it2 = std::cbegin(expectCont);

	auto printFooterFun = [&actualCont, &expectCont](std::wstringstream& buf)
	{
		buf << "Actual ";
		Details::dumpCont(buf, actualCont);

		buf << std::endl;
		buf << "Expect ";
		Details::dumpCont(buf, expectCont);
	};

	for (; it1 != std::cend(actualCont) && it2 != std::cend(expectCont); ++it1, ++it2, ++index)
	{
		const auto& actual = *it1;
		const auto& expectValue = *it2;

		// equality to correctly compare ints with zero tolerance
		if (!(std::abs(actual - expectValue) <= tolerance))
		{
			std::wstringstream buf;
			buf << L"Failed on index=" << index << L" expected=" << expectValue << L" actual=" << actual <<L" tolerance=" <<tolerance;
			
			buf << std::endl;
			printFooterFun(buf);

			return std::make_tuple(false, buf.str());
		}

		if (it1 != std::cend(actualCont) ^ it2 != std::cend(expectCont))
		{
			std::wstringstream buf;
			buf << "Containers' size mismatch" <<std::endl;
			printFooterFun(buf);
			return std::make_tuple(false, buf.str());
		}
	}
	return std::make_tuple(true, L"");
}
