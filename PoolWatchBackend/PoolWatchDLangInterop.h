#pragma once

#include <cstdint>

extern "C"
{
	typedef int32_t* (*pwCreateArrayInt32FunNew)(size_t celem, void* pUserData);
	typedef void(*pwDestroyArrayInt32FunNew)(int32_t* pInt32, void* pUserData);

	struct Int32Allocator
	{
		pwCreateArrayInt32FunNew CreateArrayInt32;
		pwDestroyArrayInt32FunNew DestroyArrayInt32;
		void* pUserData; // data which will be passed to Create/Destroy methods by server code
	};
}