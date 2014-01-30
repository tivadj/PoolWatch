classdef BlobsHelper
    
methods(Static)
  
% Merges two blobs with a thin bridge.    
% NOTE
% x When two blobs have parallel closest segments, what should we treat as a thin blob merging?
% (see PrismTipAndWall.svg)
% x If any blob is not convex then holes may appear as a result of merging
% because closest lines between two blobs may be far away from each other (see TwoCrescents.svg)
function mergedBlobsMask = mergeBlobs(connComp, blobInd1, blobInd2)
    % extract two blobs of interest
    pixInds1 = connComp.PixelIdxList{blobInd1};
    pixInds2 = connComp.PixelIdxList{blobInd2};

    imageMaskSize = connComp.ImageSize;
    twoBlobsMaskColumn = false(prod(imageMaskSize), 1);
    twoBlobsMaskColumn(pixInds1) = true;
    twoBlobsMaskColumn(pixInds2) = true;
    
    twoBlobsImage = reshape(twoBlobsMaskColumn, imageMaskSize(1), []);
    bnd = bwboundaries(twoBlobsImage, 8, 'noholes');
    assert(length(bnd) == 2, 'Must be exactly two blobs');
    
    pixCount1 = length(bnd{1});
    pixCount2 = length(bnd{2});

    xs1 = bnd{1}(:,2);
    ys1 = bnd{1}(:,1);
    xs2 = bnd{2}(:,2);
    ys2 = bnd{2}(:,1);
    
    % determine representative pixels to connect two blobs
    
    [colInds, rowInds] = meshgrid(1:pixCount1, 1:pixCount2);
    colInds = reshape(colInds, [], 1);
    rowInds = reshape(rowInds, [], 1);
    colRow1DInds = 1:pixCount1*pixCount2;
    xyDistMat = [xs1(colInds(colRow1DInds)) ys1(colInds(colRow1DInds)) xs2(rowInds(colRow1DInds)) ys2(rowInds(colRow1DInds)) zeros(length(colRow1DInds),1)];
    
    distFun = @(i) norm([xyDistMat(i,1) - xyDistMat(i,3), xyDistMat(i,2) - xyDistMat(i,4)]);
    xyDistMat(:,end) = arrayfun(distFun, colRow1DInds)';
    
    % choose pairs with shortest distance
    xyDistMatAsc = sortrows(xyDistMat, 5);    
    minDist = xyDistMatAsc(1, end);
    
    % All lines between two blobs which are approx equal to the shortest distance
    % are treated as a part of bridge. This parameter influence how close those distances may be.
    shortDistDelta = 5; 
    closestXYDistMat = xyDistMatAsc(xyDistMatAsc(:,end) <= minDist + shortDistDelta,:);

    % find convex hull
    portPoints = [closestXYDistMat(:,1:2); closestXYDistMat(:,3:4)];
    convInds = convhulln(portPoints);
    convInds = convInds(:,1);
    
    glueMask = poly2mask(portPoints(convInds,1), portPoints(convInds,2), imageMaskSize(1), imageMaskSize(2));
    mergedBlobsMask = twoBlobsImage | glueMask;
end

function [minDist, pix1, pix2] = twoBlobsClosestDistancePixels(connComp, blobInd1, blobInd2)
    % extract two blobs of interest
    pixInds1 = connComp.PixelIdxList{blobInd1};
    pixInds2 = connComp.PixelIdxList{blobInd2};

    imageMaskSize = connComp.ImageSize;
    twoBlobsMaskColumn = false(prod(imageMaskSize), 1);
    twoBlobsMaskColumn(pixInds1) = true;
    twoBlobsMaskColumn(pixInds2) = true;
    
    twoBlobsImage = reshape(twoBlobsMaskColumn, imageMaskSize(1), []);
    bnd = bwboundaries(twoBlobsImage, 8, 'noholes');
    assert(length(bnd) == 2, 'Must be exactly two blobs');
    
    pixCount1 = length(bnd{1});
    pixCount2 = length(bnd{2});

    xs1 = bnd{1}(:,2);
    ys1 = bnd{1}(:,1);
    xs2 = bnd{2}(:,2);
    ys2 = bnd{2}(:,1);
    
    % determine representative pixels to connect two blobs
    
    [colInds, rowInds] = meshgrid(1:pixCount1, 1:pixCount2);
    colInds = reshape(colInds, [], 1);
    rowInds = reshape(rowInds, [], 1);
    colRow1DInds = 1:pixCount1*pixCount2;
    xyDistMat = [xs1(colInds(colRow1DInds)) ys1(colInds(colRow1DInds)) xs2(rowInds(colRow1DInds)) ys2(rowInds(colRow1DInds)) zeros(length(colRow1DInds),1)];
    
    distFun = @(i) norm([xyDistMat(i,1) - xyDistMat(i,3), xyDistMat(i,2) - xyDistMat(i,4)]);
    xyDistMat(:,end) = arrayfun(distFun, colRow1DInds)';
    
    % choose pairs with shortest distance
    xyDistMatAsc = sortrows(xyDistMat, 5);    
    minDist = xyDistMatAsc(1, end);
    pix1 = xyDistMatAsc(1, 1:2);
    pix2 = xyDistMatAsc(1, 3:4);
end
    
end
end

