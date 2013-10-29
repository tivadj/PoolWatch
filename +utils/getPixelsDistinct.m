%pixels = [N,3]
function pixels = getPixelsDistinct(svgPathPattern, invertMask, strokeColor)
if ~exist('strokeColor','var')
    strokeColor='';
end
skinPixelsM=utils.getPixelsFromLabelledImage(svgPathPattern, invertMask, strokeColor);

% unique pixels
skinPixelsMUnique = unique(skinPixelsM, 'rows');
pixels = skinPixelsMUnique;
end