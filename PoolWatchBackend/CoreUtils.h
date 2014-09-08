#pragma once
#include <string>
#include <functional>
#include <vector>
#include <cassert>

#include "PoolWatchFacade.h"

namespace PoolWatch
{
	PW_EXPORTS std::string timeStampNow();

	//

	// Represents buffer of elements with cyclic sematics. When new element is requested from buffer, the reference to
	// already allocated element is returned.
	// Use 'queryHistory' method to get old elements.
	template <typename T>
	struct CyclicHistoryBuffer
	{
	private:
		std::vector<T> cyclicBuffer_;
		int freeFrameIndex_;
	public:
		CyclicHistoryBuffer(int bufferSize)
			:freeFrameIndex_(0),
			cyclicBuffer_(bufferSize)
		{
		}

		// initializes each element of the buffer
		auto init(std::function<void(size_t index, T& item)> itemInitFun) -> void
		{
			for (size_t i = 0; i < cyclicBuffer_.size(); ++i)
				itemInitFun(i, cyclicBuffer_[i]);
		}

		auto queryHistory(int indexBack) -> T&
		{
			assert(indexBack <= 0);

			// 0(current) = next free element to return on request
			// -1 = last valid data
			int ind = -1 + freeFrameIndex_ + indexBack;
			if (ind < 0)
				ind += (int)cyclicBuffer_.size();

			assert(ind >= 0 && "Buffer index is out of range");
			assert(ind < cyclicBuffer_.size());

			return cyclicBuffer_[ind];
		};

		auto requestNew() -> T&
		{
			auto& result = cyclicBuffer_[freeFrameIndex_];

			freeFrameIndex_++;
			if (freeFrameIndex_ >= cyclicBuffer_.size())
				freeFrameIndex_ = 0;

			return result;
		};
	};
}
