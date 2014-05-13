classdef PoolBoundaryDetector
methods(Static)

% Radius of the circle to separate lane divider from water or pool tiles.
function pad = dividerPadding()
    pad = 3;
end

function dividersMask = getLaneDividersMask(image, imagePoolBnd, waterMask, fleshClassifierFun, debug)
    % remove water and flesh from pool boundary
    
    fleshMask = utils.PixelClassifier.applyToImage(image, fleshClassifierFun);
    dividersMask = imagePoolBnd & ~(waterMask | fleshMask);
    if debug
        imshow(utils.applyMask(image, dividersMask))
    end

    connComp = bwconncomp(dividersMask);
    connCompProps = regionprops(dividersMask, 'Eccentricity');
    almostLinesMask = [connCompProps.Eccentricity] > 0.99;
    mask = HumanDetector.removeIslands(connComp, dividersMask, ~almostLinesMask);
    if debug
        imshow(mask);
    end
    
    % slightly enlarge dividers
    sel = strel('diamond', PoolBoundaryDetector.dividerPadding);
    dividersMaskEnl = imdilate(mask, sel);
    if debug
        imshow(dividersMaskEnl);
        imshow(utils.applyMask(image,dividersMaskEnl))
    end
    
    dividersMask = dividersMaskEnl;    
end

function imageCalibPnts = getCalibrationPoints(poolImage, watClassifFun, debug)
    waterMask = utils.PixelClassifier.applyToImage(poolImage, watClassifFun, debug);
    [imgNoHoles] = PoolBoundaryDetector.getPoolMask(poolImage, waterMask, true, debug);
    if debug
        imshow(imgNoHoles)
    end
    
    laneMarkerMask = waterMask;
    % try to clean the mask
    connComp = bwconncomp(laneMarkerMask);
    connCompProps = regionprops(connComp, 'Area');
    smallBlobs = [connCompProps.Area] < 10;
    laneMarkerSmoothMask = HumanDetector.removeIslands(connComp, laneMarkerMask, smallBlobs);
    imshow(laneMarkerSmoothMask);
    
    [vanishPoint,lines2] = PoolBoundaryDetector.getVanishingPoint(laneMarkerSmoothMask);
    if debug
        hold on
        plot(lines2(:, [1 2])', lines2(:, [3 4])')
        hold off
    end
    
    imgNoHolesU8 = im2uint8(imgNoHoles);
    bndPolyline = PoolBoundaryDetector.getPoolBoundaryPolyline(imgNoHolesU8, vanishPoint);

    
    imageCalibPnts = PoolBoundaryDetector.getCalibrationPointsHelper(imgNoHoles, lines2, vanishPoint, bndPolyline);
    if debug
        fprintf('Image calibration points\n');
        display(imageCalibPnts);
    end
end

function poolMask = getPoolMask(image, waterMask, forceSingleBlob, debug)
    imgWater = utils.applyMask(image, waterMask);
    if debug
        imshow(imgWater);
    end
    
%     sel = strel('disk', 11, 0);
%     imgSmooth = imclose(imgWater, sel);
%     imshow(imgSmooth);
    
    imgSmoothGray = rgb2gray(imgWater);
    connComp = bwconncomp(imgSmoothGray);
    connCompProps = regionprops(connComp, 'Area');
    
    % assume pool is big, so remove components with small area
    poolAreaMinPix = 5000;
    smallBlobs = [connCompProps.Area] < poolAreaMinPix;
    imgSmoothNoSmall = HumanDetector.removeIslands(connComp, imgSmoothGray, smallBlobs);
    if debug
        imshow(imgSmoothNoSmall);
    end

    %
    sel = strel('disk', 9, 0);
    imgSmooth = imclose(imgSmoothNoSmall, sel);
    if debug
        imshow(imgSmooth);
    end

%     imgBlobInfo = utils.drawRegionProps(imgSmoothGray, connComp, 50);
%     imshow(imgBlobInfo);
    
    imgNoHoles = imfill(imgSmooth,'holes');
    if debug
        imshow(imgNoHoles)
    end
    
    % convert to mask (to avoid lines inside the pool)
    imgNoHoles(imgNoHoles > 0)=255;
    if debug
        imshow(imgNoHoles)
    end
    
    % leave some padding between water and pool tiles to avoid stripes of 
    % tiles to be associated with a swimmer
    sel = strel('diamond', PoolBoundaryDetector.dividerPadding);
    poolMaskIndent = imerode(imgNoHoles, sel);
    if debug
        imshow(poolMaskIndent);
    end
    
    poolMaskSingle = poolMaskIndent;

    % leave only largest blob
    if forceSingleBlob
        poolMaskTmp = poolMaskIndent;
        compIndToArea = [1:connComp.NumObjects; connCompProps.Area]';
        compIndToArea = compIndToArea(~smallBlobs,:);
        compIndToArea = sortrows(compIndToArea, -2); % -2=descending by Area

        removeBlobMask = true(1,connComp.NumObjects);
        removeBlobMask(compIndToArea(1,1)) = 0; % leave largest blob
        imgLargestBlob = HumanDetector.removeIslands(connComp, poolMaskTmp, removeBlobMask);
        if debug
            imshow(imgLargestBlob)
        end
        
        poolMaskSingle = imgLargestBlob;
    end
    
    % pool boundary is convex (to avoid erasing swimmer shape by pool boundary with unexpected cavities)
    connComps = bwconncomp(poolMaskSingle);
    connCompProps = regionprops(connComps, 'BoundingBox', 'ConvexImage')
    convexPoolMask = false(size(image,1), size(image,2));
    for i=1:connComps.NumObjects
        bnd = connCompProps(i).BoundingBox;
        convexPoolMask(bnd(2):bnd(2)+bnd(4)-1, bnd(1):bnd(1)+bnd(3)-1) = connCompProps(i).ConvexImage;
    end
    
    if debug
        imshow(convexPoolMask);
    end
    
    poolMask = convexPoolMask;
end

function [vanishPoint,lines2] = getVanishingPoint(laneMarkerSmoothMask)
    lines2 = APPgetLargeConnectedEdges(double(laneMarkerSmoothMask), 50);
    hold on
    plot(lines2(:, [1 2])', lines2(:, [3 4])')
    hold off

    
%     imgWaterBnd = utils.applyMask(i1, imgNoHoles & ~waterMask);
%     imshow(imgWaterBnd);

    % find vanishing point
    % TODO: we may somehow filter which lines to intersect
    intersecPoints = [];
    numLines = size(lines2, 1);
    numPoints = numLines*(numLines-1)/2;
    for i=1:numLines
    for i2=i+1:numLines
        l1 = [lines2(i,1),lines2(i,3),lines2(i,2),lines2(i,4)];
        l2 = [lines2(i2,1),lines2(i2,3),lines2(i2,2),lines2(i2,4)];
        [x,y]=lineintersect(l1,l2,false);
        intersecPoints = [intersecPoints; x y]; 
    end
    end
    
    vanishPoint = median(intersecPoints);
    hold on
    plot(intersecPoints(:,1), intersecPoints(:,2),'yo');
    plot(vanishPoint(1),vanishPoint(2),'ro');
    hold off
    
    % find more precise vanishing point by removing outliers
    
    if true % refine vanishing point
    
        % find distanc
        distsToVanishPoint = sqrt(sum((intersecPoints - repmat(vanishPoint,numPoints,1)).^2, 2));
        goodIntersecPoints = intersecPoints(distsToVanishPoint < 30,:);

        vp1 = mean(goodIntersecPoints);
        vp2 = median(goodIntersecPoints);
    end
    
    % collect all lines which may be associated with each lane marker
    % select next line collect a bunch of associa
    % implement grouping similar to clustering 
    % distance between segments
    % for two segments there are 4 endpoints
    % find farthest from vanishing point
    % for another segment find point which lie on a line and have the 
    % same distance from vanishing point
    % result is distance between these two points
    % it must be lesser than MarkerLaneMaxWidthPix=
end

function bndPolyline = getPoolBoundaryPolyline(imgNoHoles, vanishPoint)
    % show lines

    lines = APPgetLargeConnectedEdges(imgNoHoles, 50);
    hold on
    plot(lines(:, [1 2])', lines(:, [3 4])')
    hold off

%         inds=[8 5; 8 6; 8 10; 8 11];
% %     for i=1:numLines
% %     for i1=i+1:numLines
%     for k=1:size(inds,1)
%         i = inds(k,1);
%         i1 = inds(k,2);
%     
%         p1 = [lines(i,1) lines(i,3)];
%         p2 = [lines(i,2) lines(i,4)];
%         p3 = [lines(i1,1) lines(i1,3)];
%         p4 = [lines(i1,2) lines(i1,4)];
%         %cost1 = RunHumanDetector.getTwoSegmentsGroupingCost(p1,p2,p3,p4);
%         cost1 = RunHumanDetector.getTwoSegmentsAngleCostMidPoint(p1,p2,p3,p4);
%         
%         fprintf('%d-%d cost=%.6f\n', i,i1,cost1);    
%         
%         %ang = RunHumanDetector.getSegmentVanishingPointAngle(p1, p2, vanishPoint);
%     end
    
    %
    
    segmentsSet = LogicalLineSegmentSet;

    numLines = size(lines,1);
    for i=1:numLines
        p1 = [lines(i,1) lines(i,3)];
        p2 = [lines(i,2) lines(i,4)];
        segment = LogicalLineSegment([p1; p2]);
        segment.Id = i;
        segmentsSet.Segments{end+1} = segment;
    end
    
    maxDivergFromVanishPoint = deg2rad(6);
    PoolBoundaryDetector.groupSegmentsByVanishingPoint(segmentsSet, vanishPoint, maxDivergFromVanishPoint);
    
    % NOTE: pool boundary is used to 1) find dividers lines; 2) find refined pool boundary
    
    % Group segments into boundary
    % group segments on the same line with higher priority compared to 
    % grouping closely positioned segments to prefer bigger shapes
    
	PoolBoundaryDetector.groupSegmentsOnTheSameLine(segmentsSet);
    
    PoolBoundaryDetector.groupCloseSegments(segmentsSet);
    
    PoolBoundaryDetector.groupSegmentsByExtendingEndRays(segmentsSet);

    segAsChain = PoolBoundaryDetector.getLongestChain(segmentsSet);

    bndPolyline = segAsChain.asPolyline;
end

function groupSegmentsByVanishingPoint(segmentsSet, vanishPoint, maxDivergFromVanishPoint)
    numAllLines = length(segmentsSet.Segments);
    
    % select aligned to vanishing point segments
    angs = arrayfun(@(i) PoolBoundaryDetector.getSegmentVanishingPointAngle(segmentsSet.Segments{i}, vanishPoint), 1:numAllLines);
    
    maxDivergToVanishPoint = deg2rad(2);
    alignedLinesIndices = find(angs < maxDivergToVanishPoint);
    
    lineClusters = cell(1,0);
    processedAlignedSegments = false(1, numAllLines);
    
    for clustInd=alignedLinesIndices
        if processedAlignedSegments(clustInd)
            continue;
        end
        
        clusterAng = segmentsSet.Segments{clustInd}.slopeAngle;
        processedAlignedSegments(clustInd) = true;
        clusterSegmentInds = [clustInd];
        
        % collect all lines with close angle
        for segInd2=alignedLinesIndices
            if processedAlignedSegments(segInd2)
                continue;
            end
            
            otherAng = segmentsSet.Segments{segInd2}.slopeAngle;
            
            if abs(otherAng - clusterAng) < maxDivergFromVanishPoint
                clusterSegmentInds = [clusterSegmentInds segInd2];
                processedAlignedSegments(segInd2) = true;
            end
        end
        
        % arrange segments by increase of centroid's X coordinate
        midXs = zeros(size(clusterSegmentInds));
        for i=1:length(clusterSegmentInds)
            clustInd = clusterSegmentInds(i);
            
            pnts = segmentsSet.Segments{clustInd}.terminalPoints;
            midXs(i) = mean(pnts(:,1));
        end
        
        ordSegs = sortrows([clusterSegmentInds' midXs'], 2);
        
        lineClusters{end+1} =  ordSegs(:,1)';
    end
    
    % merge segments on the same line
    segmentsToRemove = false(1, numAllLines);
    
    for clustInd=1:length(lineClusters)
        cluster = lineClusters{clustInd};
        
        segInd1 = cluster(1);
        seg = segmentsSet.Segments{segInd1};
        
        for segInd2=cluster(2:end)
            otherSeg = segmentsSet.Segments{segInd2};
            seg.integrate(otherSeg);
            
            segmentsToRemove(segInd2) = true;
        end
        
        % lock this group to avoid merging with other groups, even if
        % they approximately lie on the same line
        seg.CanBeMerged = false; 
    end
    
    % clean up merged sements
    segmentsSet.Segments(segmentsToRemove) = [];
end

function segAsChain = getLongestChain(segmentsSet)
    % leave the longest chain
    chains = cell(1,0);
    numSegments = length(segmentsSet.Segments);
    processedSegments = containers.Map('KeyType', 'int32', 'ValueType', 'uint8');
    for i=1:numSegments
        curSeg = segmentsSet.Segments{i};

        if processedSegments.isKey(curSeg.Id)
            continue;
        end
        
        segChainIds = [curSeg.Id];
        processedSegments(curSeg.Id) = true;
        
        % collect entire chain
        
        nextSeg = curSeg.Next;
        while ~isempty(nextSeg)
            if processedSegments.isKey(nextSeg.Id)
                break;
            end
            
            segChainIds(end+1) = nextSeg.Id;
            processedSegments(nextSeg.Id) = true;
            
            nextSeg = nextSeg.Next;
        end
        
        prevSeg = curSeg.Prev;
        while ~isempty(prevSeg)
            if processedSegments.isKey(prevSeg.Id)
                break;
            end
            
            segChainIds = [prevSeg.Id segChainIds];
            processedSegments(prevSeg.Id) = true;
            
            prevSeg = prevSeg.Prev;
        end
        
        chains{end+1} = segChainIds;        
    end
    
    % 
    [chainSize, chainInd] = max(cellfun(@length, chains));
    longestChain = chains{chainInd};
    segAsChain = PoolBoundaryDetector.segmentById(segmentsSet, longestChain(1));
end

function segment = segmentById(segmentsSet, id)
    for i=1:length(segmentsSet.Segments)
        seg = segmentsSet.Segments{i};
        if seg.Id == id
            segment = seg;
            break;
        end
    end
end

function groupSegmentsOnTheSameLine(segmentsSet)
    numSegments = length(segmentsSet.Segments);
    
    processedSegm = false(1, numSegments);
    segmentsToRemove = false(1, numSegments);

    % group segments on the same line
    for i1=1:numSegments
        if processedSegm(i1)
            continue;
        end
        
        codirSegInds = [i1];
        processedSegm(i1) = true;
        
        seg1 = segmentsSet.Segments{i1};
        seg1Pnts = seg1.terminalPoints;
        v1 = seg1Pnts(2,:)-seg1Pnts(1,:);
        
        distToSeg1 = utils.PW.distanceOriginToLine(seg1Pnts(1,:),seg1Pnts(2,:));
        
        % select all segments on the line of this segment
        
        for i2=i1+1:numSegments
            if processedSegm(i2)
                continue;
            end
            
            seg2 = segmentsSet.Segments{i2};
            seg2Pnts = seg2.terminalPoints;
            v2 = seg2Pnts(2,:)-seg2Pnts(1,:);
            
            % two segments lie on the same line if slope is approximately the same
            % and stripe covering two segments is thin
            
            ang1 = utils.PW.angleTwoVectors(v1,v2);
            sameLineMaxAngle = deg2rad(4);
            
            isSameLine = ang1 < sameLineMaxAngle || sameLineMaxAngle > (pi-sameLineMaxAngle);
            if ~isSameLine
                continue;
            end
            
            maxLineWidthPix = 12; % line across window can 'bend' (eg >8pix)
            distToSeg2 = utils.PW.distanceOriginToLine(seg2Pnts(1,:),seg2Pnts(2,:));
            isSameLineByWidth = abs(distToSeg2-distToSeg1) < maxLineWidthPix;
            
            if isSameLineByWidth
                codirSegInds(end+1) = i2;
                processedSegm(i2) = true;
            end            
        end
        
        if length(codirSegInds) <= 1
            continue;
        end
        
        % arrange all segments according to X coordinate
        midXs = zeros(size(codirSegInds));
        for i=1:length(codirSegInds)
            segInd = codirSegInds(i);
            
            pnts = segmentsSet.Segments{segInd}.terminalPoints;
            midXs(i) = mean(pnts(:,1));
        end
        
        ordSegs = sortrows([codirSegInds' midXs'], 2);
        
        % try to merge and link all segments along the line
        prevSeg = [];
        segIndInd = 1;
        while segIndInd <= length(ordSegs)
            segInd=ordSegs(segIndInd,1);
            seg = segmentsSet.Segments{segInd};
            
            if seg.CanBeMerged
                % merge segments
                segIndInd2 = segIndInd+1;
                while segIndInd2 <= length(ordSegs)
                    segInd2 = ordSegs(segIndInd2);
                    seg2 = segmentsSet.Segments{segInd2};
                    if ~seg2.CanBeMerged
                        break;
                    end
                        
                    seg.integrate(seg2);
                    
                    segmentsToRemove(segInd2) = true;
                    segIndInd2 = segIndInd2 + 1;
                end
                segIndInd = segIndInd2 - 1; % stop on last mergable segment
            end
            
            % assert: current segment is the longest mergable logical segment
            
            if ~isempty(prevSeg)
                prevSeg.linkSegment(seg);
            end
            
            prevSeg = seg;
            segIndInd = segIndInd + 1;
        end
    end

    segmentsSet.Segments(segmentsToRemove) = [];
end

function groupCloseSegments(segmentsSet)
    % group closely positioned groups
    numSegments = length(segmentsSet.Segments);
    groupPairs = nchoosek(numSegments:-1:1, 2);
          
    maxMergeSegmentsDistance = 10;
    for i=1:size(groupPairs,1)
        k1 = groupPairs(i,1);
        k2 = groupPairs(i,2);
        
        seg1 = segmentsSet.Segments{k1};
        seg2 = segmentsSet.Segments{k2};
        
        
        [where, otherInd, dist] = seg1.closestIntegration(seg2);
        if dist < maxMergeSegmentsDistance
            if strcmp(where, 'begin') && isempty(seg1.Prev) || strcmp(where, 'end') && isempty(seg1.Next)
                seg1.linkSegment(seg2);
            end
        end
    end
end

% Extends segments till intersection with other segments. In case of 
% valid intersection, both segments are linked.
function groupSegmentsByExtendingEndRays(segmentsSet)
    % extend some segments and find intersection
    while true
    madeChanges = 0;
    segEndPoints = PoolBoundaryDetector.getAllSegmentRays(segmentsSet);
    for i=1:size(segEndPoints,1)
        seg1Ind = segEndPoints(i, 1);
        seg1Where = segEndPoints(i, 2);
        seg2Ind = segEndPoints(i, 3);
        seg2Where = segEndPoints(i, 4);
        
        seg1 = segmentsSet.Segments{seg1Ind};
        seg2 = segmentsSet.Segments{seg2Ind};
        
        segtPnts1 = seg1.terminalPoints;
        segtPnts2 = seg2.terminalPoints;
        
        % seg1(p1,p2) and seg2(p3,p4)
        if seg1Where == 1
            org1 = segtPnts1(2,:);
            open1 = segtPnts1(1,:);
        else
            org1 = segtPnts1(1,:);
            open1 = segtPnts1(2,:);
        end
        if seg2Where == 1
            org2 = segtPnts2(2,:);
            open2 = segtPnts2(1,:);
        else
            org2 = segtPnts2(1,:);
            open2 = segtPnts2(2,:);
        end
        
        v1 = open1 - org1;
        v2 = open2 - org2;
        
        vMat = [v1' -v2'];
        pMat = (org2 - org1)';
        t = vMat \ pMat;
        
        if t(1) > 1 && t(2) > 1 % hit forward to each ray
            cent = org1 + t(1)*v1;
            
            %seg1.link(seg2, seg1Ind, seg2Ind, cent);
            if seg1Where == 1
                seg1.Points(1,:) = cent;
                seg1.Prev = seg2;
            else
                seg1.Points(seg1.pointsCount,:) = cent;
                seg1.Next = seg2;
            end
            if seg2Where == 1
                seg2.Points(1,:) = cent;
                seg2.Prev = seg1;
            else
                seg2.Points(seg2.pointsCount,:) = cent;
                seg2.Next = seg1;
            end
            
            madeChanges = 1;
            break;
        end
    end
    
    if madeChanges == 0
        break;
    end
    end
end

function segEndPoints = getAllSegmentRays(segmentsSet)
    segEndPoints = zeros(0,4,'int32');
    
    numSegs = length(segmentsSet.Segments);
    for i1=1:numSegs
        seg1 = segmentsSet.Segments{i1};
        
        for i2=i1+1:numSegs
            seg2 = segmentsSet.Segments{i2};
            
            if isempty(seg1.Prev) && isempty(seg2.Prev)
                segEndPoints(end+1,:) = [i1 1 i2 1];
            end
            if isempty(seg1.Prev) && isempty(seg2.Next)
                segEndPoints(end+1,:) = [i1 1 i2 2];
            end
            if isempty(seg1.Next) && isempty(seg2.Prev)
                segEndPoints(end+1,:) = [i1 2 i2 1];
            end
            if isempty(seg1.Next) && isempty(seg2.Next)
                segEndPoints(end+1,:) = [i1 2 i2 2];
            end
        end
    end
end

function ang = getSegmentVanishingPointAngle(segment, vanishPoint)
    allPoints = segment.terminalPoints;
    
    p1 = allPoints(1,:);
    p2 = allPoints(2,:);
    [~, minInd] = min([norm(vanishPoint - p1) norm(vanishPoint - p2)]);
    
    closePnt = allPoints(minInd,:);
    furtherPnt = allPoints(3 - minInd,:);
    
    v1 = closePnt - furtherPnt;
    dirToVanish = vanishPoint - furtherPnt;
    
    ang = utils.PW.angleTwoVectors(v1, dirToVanish);
end

function imageCalibPnts = getCalibrationPointsHelper(imgNoHoles, lines2, vanishPoint, bndPolyline)
    % keep lines along lane dividers (remove lines which do not hit vanishing point)

    pointDistFun = @(x1,y1,x2,y2) utils.PW.distancePointLine(vanishPoint, [x1 y1],[x2 y2]);
    dists = arrayfun(pointDistFun, lines2(:,1),lines2(:,3),lines2(:,2),lines2(:,4));
    
    vanishPointMaxDivergence = 16; % in pixels
    lines3 = lines2;
    lines3(abs(dists) > vanishPointMaxDivergence,:) = [];
    

    %
    maxDividerWidthAng = deg2rad(6);
    dividerLines = PoolBoundaryDetector.assignLineSegmentsIntoClusters(lines3, maxDividerWidthAng);
    
    % find average line representative for each divider
    dividerLineAngs = cellfun(@(lineInds) mean(lines3(lineInds,5)), dividerLines);
    negAngs = dividerLineAngs<0;
    dividerLineAngs(negAngs) = dividerLineAngs(negAngs) + pi; % for convenience, make line angles positive
    dividerLineAngs = sort(dividerLineAngs,'descend'); % for conveniece, arrange dividers from left to right
    
    % find pool boundary polyline
%     poolBnd=bwboundaries(imgNoHoles); % row=(Y,X) for some reason?
%     poolBnd=poolBnd{1}; % focus on largest
%     hold on
%     plot(poolBnd(:,2), poolBnd(:,1),'r-')
%     hold off
% 
%     % simplify pool boundary
%     poolBndSimple=dpsimplify(poolBnd,30);
    poolBndSimple = bndPolyline;
    hold on
    plot(poolBndSimple(:,1), poolBndSimple(:,2),'g-')
    plot(poolBndSimple(:,1), poolBndSimple(:,2),'co')
    hold off
    
    % construct points for calibration
    imageCalibPnts = zeros(0,2);
    
    % add boundary corners
    assert(size(poolBndSimple,1) >= 4, 'At least two corners of the pool must be detected');
    imageCalibPnts = [imageCalibPnts; poolBndSimple(2:end-1,:)];
    
    % add intersection points of pool boundary and
    
    % process dividers
    for divInd=1:length(dividerLineAngs)
        ang = dividerLineAngs(divInd);
        dir1 = [cos(ang) sin(ang)];
        
        rad = 60000;
        p2 = vanishPoint + dir1 * rad;
        pNeg = vanishPoint;
        hold on
        plot([pNeg(1) p2(1)], [pNeg(2) p2(2)], 'g');
        hold off
        
        % cross line with pool boundary
        [crossX crossY] = polyxpoly(poolBndSimple(:,1),poolBndSimple(:,2), [pNeg(1) p2(1)], [pNeg(2) p2(2)]);
        hold on
        plot(crossX, crossY,'mo')
        hold off
        imageCalibPnts = [imageCalibPnts; [crossX crossY]];
    end
end

% lines: Nx6 matrix, result of APPgetLargeConnectedEdges()
% dividerLines: List<List<int>>
function dividerLines = assignLineSegmentsIntoClusters(lines, maxDividerWidthAng)
    % assign divider id to each line
    numLines = size(lines,1);
    lineToLaneDivider = zeros(numLines,1,'int32');
    nextClusterId = 1;
    for i=1:numLines
        if lineToLaneDivider(i) > 0
            continue;
        end
        
        % create new cluster
        curClusterId = nextClusterId;
        nextClusterId = nextClusterId + 1;
        
        lineToLaneDivider(i) = curClusterId;
        clusterAng = lines(i,5);
        
        % collect all lines with close angle
        for i2=i+1:numLines
            if lineToLaneDivider(i2) > 0
                continue;
            end
            
            if abs(lines(i2,5) - clusterAng) < maxDividerWidthAng
                lineToLaneDivider(i2) = curClusterId;
            end
        end
    end
    
    % reshape result into a list form
    dividerLines = cell(1,nextClusterId-1);
    for i=1:numLines
        clustId = lineToLaneDivider(i);
        
        dividerLines{clustId} = [[dividerLines{clustId}] i];
    end
end

function drawSegmentsAll(segmentsSet)
    c_list = ['g' 'r' 'b' 'c' 'm' 'y'];

    hold on
    for i=1:length(segmentsSet.Segments)
        % pick color
        color = c_list(1+mod(i-1, length(c_list)));

        seg = segmentsSet.Segments{i};
        plot(seg.Points(:,1), seg.Points(:,2), color);
        
        mid = mean(seg.terminalPoints);
        text(mid(1), mid(2), num2str(i),'Color',color)
    end
    hold off
end

function drawSegments(segmentsSet, indsToDraw, col)
    hold on
    for i=indsToDraw
        plot(segmentsSet.Segments{i}.Points(:,1),segmentsSet.Segments{i}.Points(:,2), col);
    end
    hold off
end

end
end