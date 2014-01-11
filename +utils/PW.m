classdef PW

methods(Static)

function stampStr = timeStampNow
    stampStr = datestr(now, 'yyyymmddhhMMss');
end

function debugImage(sender, msg, image)
    imshow(image);
    title(msg);
end

% removes each row of the mat for which condition rowPred is true.
function cleanedMat = matRemoveIf(mat, rowPred)
    for i=size(mat,1):-1:1 % traverse in back order
        if rowPred(mat(i,:))
            mat(i,:) = [];
        end
    end
    cleanedMat = mat;
end

% Find distance from origin to line(p1,p2).
function dist = distanceOriginToLine(p1,p2)
    % implicit line equation Ax+By+C=0
    A = p1(2) - p2(2);
    B = - (p1(1) - p2(1));
    C = p1(1)*p2(2) - p2(1)*p1(2);
    
    lineVec = p2 - p1;
    len = norm(lineVec);
    
    % n(nx,ny) = direction of the perpendicular from origin to the line
    nx = abs(lineVec(2)) / len;
    ny = abs(lineVec(1)) / len;
    
    dist = - C / (A*nx + B*ny);
end

% Computes distance from point to line(p1,p2).
function dist = distancePointLine(point, p1,p2)
    A = p1(2) - p2(2); % y1-y2
    B = - p1(1) + p2(1); % -x1+x2
    C = p1(1)*p2(2) - p1(2)*p2(1); % x1 y2 - y1 x2
    
    dist = [A B C] * [point 1]' / norm([A B]);
end

function angle = angleTwoVectors(v1,v2)
    angle = acos(v1 * v2' / (norm(v1)*norm(v2)));
end

% Facade function to create skin(flesh) classifier.
function skinClassifierFun = createSkinClassifier(debug)
    % initialize classifier
    cl2=SkinClassifierStatics.create;
    SkinClassifierStatics.populateSurfPixels(cl2);
    SkinClassifierStatics.prepareTrainingDataMakeNonoverlappingHulls(cl2, debug);
    SkinClassifierStatics.findSkinPixelsConvexHullWithMinError(cl2, debug);
    
    %svmClassifierFun=@(XByRow) utils.SvmClassifyHelper(obj.v.skinClassif, XByRow, 1000);
    skinHullClassifierFun=@(XByRow) utils.inhull(XByRow, cl2.v.skinHullClassifHullPoints, cl2.v.skinHullClassifHullTriInds, 0.2);
    skinClassifierFun = skinHullClassifierFun;
end

function waterClassifierFun = createWaterClassifier(debug)
    % init water classifer
    humanDetectorRunner = RunHumanDetector.create;
    %waterClassifierFun = RunHumanDetector.getWaterClassifierAsConvHull(humanDetectorRunner, debug);
    waterClassifierFun = RunHumanDetector.getWaterClassifierAsMixtureOfGaussians(humanDetectorRunner,6,debug);
end

function tracker = createSwimmerTracker(debug)
    skinClassifierFun = utils.PW.createSkinClassifier(debug);
    waterClassifierFun = utils.PW.createWaterClassifier(debug);
    
    poolRegionDetector = PoolRegionDetector(skinClassifierFun, waterClassifierFun);

    distanceCompensator = CameraDistanceCompensator;
    
    humanDetector = HumanDetector(skinClassifierFun, waterClassifierFun, distanceCompensator);
    
    colorAppearance = ColorAppearanceController;
    
    %
    tracker = SwimmerTracker(poolRegionDetector, distanceCompensator, humanDetector, colorAppearance);
end

function result = hasCommonObservation(track, otherTrack)
    assert(~isempty(track));
    assert(~isempty(otherTrack));
    
    result = false;
    
    % node.DetectionInd is empty for root pseudo node or node for missed observation hypothesis
    
    childAnc = track;
    while ~isempty(childAnc)
        if ~isempty(childAnc.DetectionInd)
        
            otherAnc = otherTrack;
            while ~isempty(otherAnc)
                if ~isempty(otherAnc.DetectionInd)
                    
                    if childAnc.FrameInd == otherAnc.FrameInd && childAnc.DetectionInd == otherAnc.DetectionInd
                        result = true;
                        return;
                    end
                end
                otherAnc = otherAnc.Parent;
            end
        end
        childAnc = childAnc.Parent;
    end
end

% Finds connected component in the graph, represented as list of edges.
% connectedComponents([1 2; 2 3; 3 1; 4 5; 5 6])
% result = {[1 2; 2 3; 3 1], [4 5; 5 6]}
function edgeGraphList = connectedComponents(edgeGraph)
    edgeCount = size(edgeGraph,1);
    
    componentIds = zeros(edgeCount, 1, 'int32');
    
    verticesToProcess = cell(1,0);
    curComponent = 1;
    
    % for each component
    
    while true
        pendingEdgesInds = find(componentIds(:) == 0);
        if isempty(pendingEdgesInds)
            break;
        end
        
        pendingEdge = edgeGraph(pendingEdgesInds(1), :);
        verticesToProcess{end+1} = pendingEdge(1);
        verticesToProcess{end+1} = pendingEdge(2);
        
        % gather component
        
        while ~isempty(verticesToProcess)
            vertex = verticesToProcess{1};
            verticesToProcess(1) = [];

            fromEdges = edgeGraph(:,1) == vertex;
            toEdges   = edgeGraph(:,2) == vertex;
            pending   = componentIds(:,1) == 0;
            
            incidentEdgesMask = (fromEdges | toEdges) & pending;
            componentIds(incidentEdgesMask, 1) = curComponent;
            
            for adjVertex = reshape(edgeGraph(incidentEdgesMask, :), 1, [])
                verticesToProcess{end+1} = adjVertex;
            end
        end
        
        curComponent = curComponent + 1;
    end
    
    compCount = curComponent - 1;
    edgeGraphList = cell(1,compCount);
    for compInd=1:compCount
        edgeGraphList{compInd} = edgeGraph(componentIds(:,1) == compInd, :);
    end
end

function result = connectedComponentsCount(edgeGraph)
    edgeGraphList = utils.PW.connectedComponents(edgeGraph);
    result = length(edgeGraphList);
end

function result = connectedComponentsCountNative(edgeGraph)
    %countMatlab = utils.PW.connectedComponentsCount(edgeGraph);
    
    countNative = PWConnectedComponentsCount(edgeGraph);
    %assert(countMatlab == countNative);
    result = countNative;
end

end
    
end

