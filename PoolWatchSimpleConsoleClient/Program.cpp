// PoolWatchSimpleConsoleClient.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>

//#define SAMPLE_MATLABPROX 1

#ifdef SAMPLE_MATLABPROX
#include "PWMatlabProxy.h"
#endif

using namespace std;

namespace SvgImageMaskSerializerNS { void run(); }
namespace WaterClassifierTestsNS { void run(); }
namespace PoolBoundaryDetectorTestsNS { void run(); }

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


int _tmain(int argc, _TCHAR* argv[])
{
	//return testMatlabRuntime();
	//SvgImageMaskSerializerNS::run();
	//WaterClassifierTestsNS::run();
	PoolBoundaryDetectorTestsNS::run();

	return 0;
}

