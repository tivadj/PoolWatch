// PoolWatchSimpleConsoleClient.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <cassert>

//#define SAMPLE_MATLABPROX 1

#ifdef SAMPLE_MATLABPROX
#include "PWMatlabProxy.h"
#endif

#include <log4cxx/basicconfigurator.h>
#include <log4cxx/PropertyConfigurator.h>

#include <QFile>
#include <QDir>

using namespace std;

namespace SvgImageMaskSerializerNS { void run(); }
namespace WaterClassifierTestsNS { void run(); }
namespace PoolBoundaryDetectorTestsNS { void run(); }
namespace HumanBodiesTestsNS { void run(); }
namespace SwimmingPoolVideoFileTrackerTestsNS { void run(); }
namespace SwimmingPoolObserverTestsNS { void run(); }

#ifdef SAMPLE_MATLABPROX
// NOTE: path to Matlab's runtime dlls 
// (eg. E:\Programs\MATLAB\R2013a\bin\win64\
//      E:\Programs\MATLAB\R2013a\runtime\win64\ )
// must be added to PATH environment variable.
struct PWMatlabProxyInitializer
{
	bool initMatlab;

	PWMatlabProxyInitializer() : initMatlab(false) { }
	~PWMatlabProxyInitializer() 
	{ 
		if (initMatlab) {
			PWMatlabProxyTerminate();
			initMatlab = false;
		}
	}
	bool init()
	{
		initMatlab = PWMatlabProxyInitialize();
		return initMatlab;
	}
	bool init(mclOutputHandlerFcn error_handler, mclOutputHandlerFcn print_handler)
	{
		initMatlab = PWMatlabProxyInitializeWithHandlers(error_handler, print_handler);
		return initMatlab;
	}
};

int testMatlabRuntime()
{
	cout << "Init Matlab runtime" << endl;

	PWMatlabProxyInitializer matlabProxy;
	auto errHandler = [](const char* s) { cerr << s; return 1; };
	auto printHandler = [](const char* s) { cout << s; return 1; };
	bool initMatlab = matlabProxy.init(errHandler, printHandler);
	if (!initMatlab)
	{
		cerr << "PWMatlabProxy initialization failed" << endl;
		return 1;
	}

	cout << "Initializing SwimmingPoolObserver... (takes 1 min)" << endl;

	mwArray poolObserver;
	mwArray debug(1, 1, mxLOGICAL_CLASS, mxREAL);
	mxLogical debugValue = true;
	debug.SetLogicalData(&debugValue, 1);

	//utilsCreateSwimmingPoolObserver(1, poolObserver, debug);

	cout << "Getting blobs..." << endl;

	mwArray blobs;
	mwArray imageSwimmers;
	//utilsGetSwimmerBlobs(2, imageSwimmers, blobs, poolObserver, poolObserver, debug);

	return 0;
}
#endif

void configureLog4cxx(QDir logFolder)
{
	// Set up a simple configuration that logs on the console.
	//::log4cxx::BasicConfigurator::configure();

	QString logConfigAbsPath = logFolder.absoluteFilePath("log4cxx.properties");

	::log4cxx::PropertyConfigurator::configure(logConfigAbsPath.toStdString());
}

int _tmain(int argc, _TCHAR* argv[])
{
	_TCHAR* exePath = argv[0];

	//
	QDir logFolder(QString::fromWCharArray(exePath));
	auto dirUpOp = logFolder.cdUp();
	assert(dirUpOp);
	
	configureLog4cxx(logFolder);

	//return testMatlabRuntime();
	//SvgImageMaskSerializerNS::run();
	//WaterClassifierTestsNS::run();
	//PoolBoundaryDetectorTestsNS::run();
	//HumanBodiesTestsNS::run();
	//SwimmingPoolVideoFileTrackerTestsNS::run();
	SwimmingPoolObserverTestsNS::run();

	return 0;
}
