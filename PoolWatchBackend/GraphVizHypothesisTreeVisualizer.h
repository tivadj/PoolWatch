#pragma once
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include "MultiHypothesisBlobTracker.h"

void writeVisualHypothesisTree(const boost::filesystem::path& logDir, int frameInd, const std::string& fileNameTag, const std::vector<TrackHypothesisTreeNode*>& bestTrackLeafs, const TrackHypothesisTreeNode& root, int pruneWindow);