#include "ProgramUtils.h"

#include <log4cxx/logger.h>
#include <log4cxx/helpers/exception.h>
#include <log4cxx/rollingfileappender.h>
#include <log4cxx/patternlayout.h>

void configureLogToFileAppender(const QDir& logFolder, const QString& logFileName)
{
	// create rolling file appending to a runtime generated log

	QString fileRollAbsPath = logFolder.absoluteFilePath(logFileName);

	auto pLayout = log4cxx::helpers::ObjectPtr(new log4cxx::PatternLayout(L"%-5p %c - %m%n"));
	auto pRollingFileApp = log4cxx::helpers::ObjectPtr(new log4cxx::RollingFileAppender(pLayout, fileRollAbsPath.toStdWString()));

	log4cxx::LoggerPtr rootLoggerPtr = log4cxx::Logger::getRootLogger();
	rootLoggerPtr->addAppender(pRollingFileApp);
}