#pragma once

#if defined PWAPI_EXPORTS
#define PW_EXPORTS __declspec(dllexport)
#else
//#define PW_EXPORTS __declspec(dllimport)
#define PW_EXPORTS
#endif


