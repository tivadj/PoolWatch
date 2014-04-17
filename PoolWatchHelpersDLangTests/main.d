import std.stdio;
import std.algorithm;
import std.format;
import std.string;
import PoolWatchHelpersDLang.GraphHelpers;
import PoolWatchHelpersDLang.MatrixUndirectedGraph;
import MatrixUndirectedGraphTests1;
import DepthFirstSearchTests;
import ConnectedComponentsTests;
import RootedUndirectedTreeTests;
import NearestCommonAncestorOfflineAlgorithmTests;
import computeTrackIncopatibilityGraphTests;

int main(string[] argv)
{
	//writeln("Hello D-World!");
	//poolTest1();

	//MatrixUndirectedGraphTests1.test1();

	//DepthFirstSearchTests.run();
	//ConnectedComponentsTests.test1();
	RootedUndirectedTreeTests.run;
	//NearestCommonAncestorOfflineAlgorithmTests.run;
	//computeTrackIncopatibilityGraphTests.run;
	//PWInteropTestHelpers.run;

    return 0;
}
