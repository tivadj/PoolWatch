#include <string>

#include <QDir>
#include <QApplication>

#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>

#include <log4cxx/logger.h>
#include <log4cxx/helpers/exception.h>
#include <log4cxx/rollingfileappender.h>
#include <log4cxx/patternlayout.h>

#include "PoolWatchFacade.h"
#include "topviewdemotracermainwindow.h"
#include "DemoHelpers.h"

using namespace cv;
using namespace PoolWatch;
using namespace std;

using namespace log4cxx;
using namespace log4cxx::helpers;

namespace
{
	log4cxx::LoggerPtr log_(log4cxx::Logger::getLogger("PWDemo1.main"));
}


int main(int argc, char *argv[])
{
	auto workDir=boost::filesystem::path(argv[0]).parent_path();
	std::string timeStamp = PoolWatch::timeStampNow();
	boost::filesystem::path outDir = workDir / boost::filesystem::path("output") / timeStamp;
	outDir = boost::filesystem::absolute(outDir, ".").normalize();
	QDir outDirQ = QDir(outDir.string().c_str());

	// 
	configureLogToFileAppender(outDirQ, "app.log");
	LOG4CXX_DEBUG(log_, "test debug");
	LOG4CXX_INFO(log_, "test info");
	LOG4CXX_ERROR(log_, "test error");

    QApplication a(argc, argv);
    TopViewDemoTracerMainWindow w;
    w.show();

    return a.exec();
}
