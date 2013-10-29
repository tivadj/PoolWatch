classdef PW

methods(Static)

function stampStr = timeStampNow
    stampStr = datestr(now, 'yyyymmddhhMMss');
end

function debugImage(sender, msg, image)
    imshow(image);
    title(msg);
end

end
    
end

