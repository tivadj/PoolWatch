#include "stdafx.h"
#include <sstream>
#include <tuple>

#include <log4cxx/logger.h>
#include <log4cxx/helpers/exception.h>
#include <log4cxx/rollingfileappender.h>
#include <log4cxx/patternlayout.h>

#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>

#include <CppUnitTest.h>

#include "TestingUtils.h"
#include "CoreUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

boost::filesystem::path g_testResultsDir;

namespace PoolWatchBackendUnitTests
{
	log4cxx::LoggerPtr log_(log4cxx::Logger::getLogger("PW.Tests"));
}

boost::filesystem::path getTestResultsDir()
{
	return g_testResultsDir;
}

TEST_MODULE_INITIALIZE(PoolWatchBackendUnitTests_ModuleInitialize)
{
	std::stringstream ss;
	ss << "Unit tests started CD=" << boost::filesystem::current_path() <<std::endl;
	Logger::WriteMessage(ss.str().c_str());

	//
	boost::filesystem::path outDir("../../../../output/debugTests");
	
	std::string timeStamp = PoolWatch::timeStampNow();
	outDir = outDir / timeStamp;
	outDir = boost::filesystem::absolute(outDir).normalize();
	g_testResultsDir = outDir;

	boost::filesystem::create_directories(outDir);

	ss.str("");
	ss <<"Test results directory=" << outDir << std::endl;
	Logger::WriteMessage(ss.str().c_str());
}

void PoolWatchBackendUnitTests_MethodInitilize()
{
	boost::filesystem::current_path("../../../MatlabProto");
}

boost::filesystem::path initTestMethodLogFolder(const std::string& className, const std::string& methodName)
{
	auto resultsDir = getTestResultsDir();

	std::stringstream dirName;
	dirName << className << "_" << methodName;
	boost::filesystem::path logDir = resultsDir / dirName.str();
	boost::filesystem::create_directories(logDir);

	return logDir;
}

void LogFileAppenderUnsubscriber::operator()(log4cxx::helpers::ObjectPtrT<log4cxx::Appender>* pAppender) const
{
	log4cxx::LoggerPtr rootLoggerPtr = log4cxx::Logger::getRootLogger();
	rootLoggerPtr->removeAppender(*pAppender);
}

std::unique_ptr<log4cxx::helpers::ObjectPtrT<log4cxx::Appender>, LogFileAppenderUnsubscriber> scopeLogFileAppenderNew(const boost::filesystem::path& logFolder)
{
	// create rolling file appending to a runtime generated log

	boost::filesystem::path fileRollAbsPath = logFolder / "app.log";

	auto pLayout = log4cxx::helpers::ObjectPtr(new log4cxx::PatternLayout(L"%-5p %c - %m%n"));
	auto pApp = new log4cxx::RollingFileAppender(pLayout, fileRollAbsPath.c_str());
	auto pRollingFileApp = std::unique_ptr<log4cxx::helpers::ObjectPtrT<log4cxx::Appender>, LogFileAppenderUnsubscriber>(
		new log4cxx::helpers::ObjectPtrT<log4cxx::Appender>(pApp), LogFileAppenderUnsubscriber());

	log4cxx::LoggerPtr rootLoggerPtr = log4cxx::Logger::getRootLogger();
	rootLoggerPtr->addAppender(*pRollingFileApp);

	auto destroyFun = [](log4cxx::helpers::ObjectPtrT<log4cxx::Appender>* pAppender) {
		log4cxx::LoggerPtr rootLoggerPtr = log4cxx::Logger::getRootLogger();
		rootLoggerPtr->removeAppender(*pAppender);
	};

	return std::move(pRollingFileApp);
}
