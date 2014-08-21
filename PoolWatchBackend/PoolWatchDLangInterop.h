#pragma once

#include <cstdint>
#include <vector>

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

	struct CppVectorPtrWrapper
	{
		std::vector<void*>* Vector;
		void (*PushBack)(CppVectorPtrWrapper* vectorWrapper, void* ptr);
	};
}

namespace PoolWatch
{
	// Binds std::vector<T> to a helper structure which allow C code to call push_back method on a vector.
	template <typename T>
	void bindVectorWrapper(CppVectorPtrWrapper& vectorWrapper, std::vector<T>& vector)
	{
		vectorWrapper.Vector = reinterpret_cast<std::vector<void*>*>(&vector);
		vectorWrapper.PushBack = [](CppVectorPtrWrapper* vectorWrapper, void* ptr)
		{
			auto pV = reinterpret_cast<std::vector<T>*>(vectorWrapper->Vector);

			auto typedPtr = reinterpret_cast<T>(ptr);
			pV->push_back(typedPtr);
		};
	}
}
