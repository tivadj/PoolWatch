% classdef CameraSwimmerDistanceCompensator
% 
% properties
% %     MRectPool;
% %     areaVsDist;
% %     v;
% end
% 
% methods



% eg comp.getExpectedArea([463 249])
% function expArea = getExpectedArea(obj, swimmerImagePos)
%     rectPosHom = obj.MRectPool * [swimmerImagePos 1]';
%     rectPos = rectPosHom/rectPosHom(3); % normalize homog coordinate
% 
%     % take the Y coordinate
%     % flip Y coordinate so that 0 starts at camera view (for convenience)
%     yPixCam = 476-rectPos(2);
% 
% 
%     %expArea = obj.areaVsDist * [yPixCam; 1];
%     expArea = obj.areaVsDist *[yPixCam^2 yPixCam 1]'; % eval parabola
% end

% function poolPosM = poolCoord(obj, imagePos)
%     rectPosHom = obj.MRectPool * [imagePos 1]';
%     rectPos = rectPosHom/rectPosHom(3); % normalize homog coordinate
%     
%     swimmingPoolDimM = [50 25];
%     
%     poolPosMX = rectPos(1)/640*swimmingPoolDimM(1);
%     poolPosMY = rectPos(2)/476*swimmingPoolDimM(2);
%     poolPosM = [poolPosMX poolPosMY];
% end
    
% end
% end