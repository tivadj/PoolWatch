%pixels = [N,3]
function pixels = getPixelsFromLabelledImage(svgPathPattern, invertMask, strokeColor)
if ~exist('strokeColor','var')
    strokeColor='';
end

resFolder = fileparts(svgPathPattern);
svgFiles=dir(svgPathPattern);

skinPixels = {};
for dirInd=1:length(svgFiles)
    svgFileName=svgFiles(dirInd).name;
    fprintf(1, 'svg=%s\n', svgFileName);
    svgFilePath=fullfile(resFolder, svgFileName);
    
    [i1,m1]=utils.getMaskAll(svgFilePath, strokeColor);
    
    if invertMask
        m1 = ~m1;
    end

    i1R = i1(:,:,1);
    i1G = i1(:,:,2);
    i1B = i1(:,:,3);
    imgPixels = cat(2, i1R(m1), i1G(m1), i1B(m1));
    % TODO: how to batch assign to cell array?
    for pixInd=1:size(imgPixels, 1)
        skinPixels{end+1} = imgPixels(pixInd, :);
    end
end

skinPixelsM=reshape(cell2mat(skinPixels), 3, [])';
pixels = skinPixelsM;
end