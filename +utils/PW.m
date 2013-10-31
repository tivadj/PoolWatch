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

end
    
end

