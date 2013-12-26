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

end
    
end

