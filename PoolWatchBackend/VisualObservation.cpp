#include "VisualObservation.h"
#include <vector>

void fixBlobs(std::vector<DetectedBlob>& blobs, const CameraProjectorBase& cameraProjector)
{
	// update blobs CentroidWorld
	for (auto& blob : blobs)
	{
		blob.CentroidWorld = cameraProjector.cameraToWorld(blob.Centroid);
	}
}
