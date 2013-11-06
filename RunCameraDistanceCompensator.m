classdef RunCameraDistanceCompensator
methods(Static)

function obj = create()
    obj = utils.TypeErasedClass();
end    

function run(obj)
    debug=1;
    %RunCameraDistanceCompensator.test1(debug);
    RunCameraDistanceCompensator.testBackProjection(obj, debug);
    %RunCameraDistanceCompensator.testWorldAreaToCamera(obj,debug);
end    

% shows how the trace path would look in a camera view if a swimmer
% moves strictly vertically in a world coordinates (which 99% times true)
function testBackProjection(obj, debug)
    i1 = imread('data/MVI_3177_0127_640x476.png');
    imshow(i1);

    comp = CameraDistanceCompensator.create;
    obj.v.comp = comp;

    cameraPosHist = [];
    prevCameraPos = [];
    poolSize = CameraDistanceCompensator.poolSize;
    laneLength = poolSize(2);
    hold on
    for x=0:1:laneLength
        worldPos = [x 5 0]; % put on arbitrary lane
        
        cameraPos = CameraDistanceCompensator.worldToCamera(comp, worldPos);
        fprintf('worldPos=(%.2f,%.2f) cameraPos=(%.2f,%.2f)\n', worldPos(1),worldPos(2), cameraPos(1), cameraPos(2));
        
        plot(cameraPos(1), cameraPos(2), 'r.');
        if ~isempty(prevCameraPos)
            plot([prevCameraPos(1) cameraPos(1)], [prevCameraPos(2) cameraPos(2)],'r');
        end
        
        prevCameraPos = cameraPos;
        cameraPosHist = [cameraPosHist; cameraPos];
        
        % check projection
        worldPosGuess = CameraDistanceCompensator.cameraToWorld(comp, cameraPos);
        errorDist = norm(worldPosGuess - worldPos);
        fprintf('errorDist=%.2f\n', errorDist);

        % check area unprojection
        camArea = CameraDistanceCompensator.worldAreaToCamera(comp, worldPos, 1);
        rad = sqrt(camArea/pi);
        ang=0:pi/10:2*pi;
        plot(cameraPos(1) + rad*cos(ang),cameraPos(2) - rad*sin(ang), 'g');
    end
    hold off
    
    
end    
    
function test1(obj, debug)

i1=imread(fullfile('data/MVI_3177_0127_640x476.png'));
imshow(i1)

%cv.findHomography
srcCoordScaled=[181 124; 387 117; 563 267; -417 196]; % on scaled image W480 x H357
srcCoord=[srcCoordScaled(:,1)*640/480 srcCoordScaled(:,2)*476/357] % W640 x H476
%srcCoord=[243 166; ]; 640x476
%dstCoord=[0 0; 1 0; 1 1; 0 1];
dstCoord=[0 0; 640 0; 640 476; 0 476];


T=cv.getPerspectiveTransform(srcCoord, dstCoord)
x1=T*[srcCoord ones(4,1)]';
x11=[x1(:,1)/x1(3,1) x1(:,2)/x1(3,2) x1(:,3)/x1(3,3) x1(:,4)/x1(3,4)] % normalize homog coordinates
i2=cv.warpPerspective(i1, T);
imshow(i2)

T2 = [T(:,1)/T(3,1) T(:,2)/T(3,2) T(:,3)/T(3,3)];

p1=T*[463;162;1]
p11=p1/p1(3)

% take the Y coordinate
% flip Y coordinate so that 0 starts at camera view (for convenience)
yPixCam = 476-p11(2);

% fit line Area ~ Distance from camera
areaVsDist = polyfit([96 466], [58 6], 1); % diameter
areaVsDist = polyfit([0 476], [2674 27], 1); % = -0.1092d+58
x1=0:476;
plot(x1,areaVsDist(1)*x1+areaVsDist(2))
plot(x1,pi*(areaVsDist(1)*x1+areaVsDist(2)).^2)

expArea = areaVsDist * [yPixCam; 1];

% assume max area must be twice big
% areaMax = 
%%
equivDiameterVsY = [demo1.v.shapeHist(:,2), 2*sqrt(demo1.v.shapeHist(:,3)/pi)];
%save('artefacts/CameraSwimmerDistanceCompensator/EquivDiameterVsY.mat', 'equivDiameterVsY')

    % fit EquivDiameter ~ y
%     plot(demo1.v.shapeHist(:,2), 2*sqrt(demo1.v.shapeHist(:,3)/pi),'.')
%     x1=polyfit(demo1.v.shapeHist(:,2), 2*sqrt(demo1.v.shapeHist(:,3)/pi),1)
%     hold on
%     plot(xs, x1(1)*xs+x1(2),'r')
%     hold off

% find upper/lower bounds for DiamerterVsY

equivDiameterVsYBounds=zeros(length(equivDiameterVsY), 2);
r=10;
for i=1:length(equivDiameterVsY)
    i1 = max([1, i - r]);
    i2 = min([length(equivDiameterVsY), i + r]);
    
%     sigmaAccept=4;
%     sig3 = sigmaAccept*std(equivDiameterVsY(i1:i2,2));
%     bnd1 = equivDiameterVsY(i,2)-sig3;
%     bnd2 = equivDiameterVsY(i,2)+sig3;

    bnd1 = min(equivDiameterVsY(i1:i2,2));
    bnd2 = max(equivDiameterVsY(i1:i2,2));
    
    equivDiameterVsYBounds(i,:) = [bnd1 bnd2];
end

plot(equivDiameterVsY(:,1), equivDiameterVsY(:,2),'b.')

x1  =polyfit(equivDiameterVsY(:,1), equivDiameterVsY(:,2),1);
x1dw=polyfit(equivDiameterVsY(:,1), equivDiameterVsYBounds(:,1),1);
x1up=polyfit(equivDiameterVsY(:,1), equivDiameterVsYBounds(:,2),1);

xs=390:480;
hold on
plot(xs, x1(1)*xs+x1(2),'b')

plot(equivDiameterVsY(:,1), equivDiameterVsYBounds(:,1),'g.')
plot(xs, x1dw(1)*xs+x1dw(2),'g')

plot(equivDiameterVsY(:,1), equivDiameterVsYBounds(:,2),'r.')
plot(xs, x1up(1)*xs+x1up(2),'r')
hold off

end
   
function testWorldAreaToCamera(obj,debug)
    areaCamHist=[];
    
    comp = CameraDistanceCompensator.create;
    
    worldX  = 8.703;
    
    poolSize = CameraDistanceCompensator.poolSize;
    ys= linspace(0,poolSize(2),20);
    
    for y=ys
        areaCam = CameraDistanceCompensator.worldAreaToCamera(comp,[worldX y],999);
        areaCamHist = [areaCamHist; areaCam];
    end
    
    plot(ys, areaCamHist);
end

end
end