#include "stdafx.h"
#include <vector>
#include <array>
#include "TestingUtils.h"
#include <MatrixUndirectedGraph.h>
#include <algos1.h>

using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace PoolWatchBackendUnitTests
{
	TEST_CLASS(MaxWeightIndependentSetTests)
	{
		TEST_METHOD_INITIALIZE(MethodInitialize)
		{
			PoolWatchBackendUnitTests_MethodInitilize();
		}

	public:

		TEST_METHOD(Papageorgiou1)
		{
			MatrixUndirectedGraph g(5,5);
			g.setVertexPayload(0, 3.4);
			g.setVertexPayload(1, 9.1);
			g.setVertexPayload(2, 7.5);
			g.setVertexPayload(3, 4.8);
			g.setVertexPayload(4, 10.1);
			g.setEdge(0, 1);
			g.setEdge(0, 2);
			g.setEdge(1, 2);
			g.setEdge(1, 3);
			g.setEdge(2, 3);
			g.setEdge(2, 4);

			std::vector<uchar> vertexSet;
			maximumWeightIndependentSetNaiveMaxFirstMultipleSeeds(g, vertexSet);

			std::array<uchar, 5> expectVertexSet = { 0, 1, 0, 0, 1 };
			auto r = sequenceEqual(vertexSet, expectVertexSet, 0);
			Assert::IsTrue(std::get<0>(r), std::get<1>(r).c_str());
		}

		TEST_METHOD(Four1)
		{
			MatrixUndirectedGraph g(4,4);
			g.setVertexPayload(0, 4);
			g.setVertexPayload(1, 2);
			g.setVertexPayload(2, 8);
			g.setVertexPayload(3, 1);
			g.setEdge(0, 2);
			g.setEdge(1, 2);
			g.setEdge(2, 3);

			std::vector<uchar> vertexSet;
			maximumWeightIndependentSetNaiveMaxFirstMultipleSeeds(g, vertexSet);

			std::array<uchar, 4> expectVertexSet = { 0, 0, 1, 0 };
			auto r = sequenceEqual(vertexSet, expectVertexSet, 0);
			Assert::IsTrue(std::get<0>(r), std::get<1>(r).c_str());
		}

		TEST_METHOD(Four2)
		{
			MatrixUndirectedGraph g(4,4);
			g.setVertexPayload(0, 4);
			g.setVertexPayload(1, 2);
			g.setVertexPayload(2, 8);
			g.setVertexPayload(3, 3);
			g.setEdge(0, 2);
			g.setEdge(1, 2);
			g.setEdge(2, 3);

			std::vector<uchar> vertexSet;
			maximumWeightIndependentSetNaiveMaxFirstMultipleSeeds(g, vertexSet);

			std::array<uchar, 4> expectVertexSet = { 1, 1, 0, 1 };
			auto r = sequenceEqual(vertexSet, expectVertexSet, 0);
			Assert::IsTrue(std::get<0>(r), std::get<1>(r).c_str());
		}

		TEST_METHOD(DoubleTri1)
		{
			MatrixUndirectedGraph g(6,6);
			g.setVertexPayload(0, 3);
			g.setVertexPayload(1, 1);
			g.setVertexPayload(2, 5);
			g.setVertexPayload(3, 6);
			g.setVertexPayload(4, 4);
			g.setVertexPayload(5, 2);
			g.setEdge(0, 2);
			g.setEdge(1, 2);
			g.setEdge(2, 3);
			g.setEdge(3, 4);
			g.setEdge(3, 5);

			std::vector<uchar> vertexSet;
			maximumWeightIndependentSetNaiveMaxFirstMultipleSeeds(g, vertexSet);

			std::array<uchar, 6> expectVertexSet = { 0, 0, 1, 0, 1, 1 };
			auto r = sequenceEqual(vertexSet, expectVertexSet, 0);
			Assert::IsTrue(std::get<0>(r), std::get<1>(r).c_str());
		}

		TEST_METHOD(K4)
		{
			MatrixUndirectedGraph g(4,4);
			g.setVertexPayload(0, 3);
			g.setVertexPayload(1, 7);
			g.setVertexPayload(2, 4);
			g.setVertexPayload(3, 6);
			g.setEdge(0, 1);
			g.setEdge(0, 2);
			g.setEdge(0, 3);
			g.setEdge(1, 2);
			g.setEdge(1, 3);
			g.setEdge(2, 3);

			std::vector<uchar> vertexSet;
			maximumWeightIndependentSetNaiveMaxFirstMultipleSeeds(g, vertexSet);

			std::array<uchar, 6> expectVertexSet = { 0, 1, 0, 0 };
			auto r = sequenceEqual(vertexSet, expectVertexSet, 0);
			Assert::IsTrue(std::get<0>(r), std::get<1>(r).c_str());
		}
	};
}