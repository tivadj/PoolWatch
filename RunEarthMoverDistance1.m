% s1 = single([1 0; 0 1]);
% s2 = single([0 0; 1 1]);
% 
% d=cv.EMD(s1, s2)
% 
% implay(fullfile(cd, 'rawdata/MVI_3178.mov'))

%
appearDir = fullfile('data/appearance');
appearFilesPattern = fullfile(appearDir, '*.svg');
svgFiles = dir(appearFilesPattern);

% construct histograms for each image
imagesCount = length(svgFiles);

imageHistograms = {};
for fileObjInd=1:imagesCount
    filePath = fullfile(appearDir, svgFiles(fileObjInd).name);
    pixels = utils.getPixelsFromLabelledImage(filePath, false, '#00FFFF');
    
    % construct RGB cubic histogram
    % TODO: too slow, => directly construct records with collection.Map
    tic;
    rgbCube = zeros(256,256,256,'int32');
    for pixInd=1:length(pixels)
        pix = pixels(pixInd,:) + 1; % +1 to Matlab index
        rgbCube(pix(1),pix(2),pix(3)) = rgbCube(pix(1),pix(2),pix(3)) + 1;
    end
    fprintf(1, 'rgbCube constructed in % sec\n', toc);
    
    % construct histogram bar record
    tic;
    histBarRecord = {};
    for r1=1:256
    for g1=1:256
    for b1=1:256
        barValue = rgbCube(r1,g1,b1);
        if barValue > 0
            histBarRecord{end+1,1} = single([barValue r1 g1 b1]);
        end
    end
    end
    end
    fprintf(1, 'histBarRecord constructed in % sec\n', toc);
    
    histBarRecordMat = cell2mat(histBarRecord);
    imageHistograms{end+1,1} = histBarRecordMat;
end

distMat = zeros(imagesCount,imagesCount);
% process matrix upper triangle
for row=1:(imagesCount-1)
for col=(row+1):imagesCount
    imageBarRecordA = imageHistograms{row};
    imageBarRecordB = imageHistograms{col};
    dist=cv.EMD(imageBarRecordA, imageBarRecordB);
    distMat(row,col) = dist;
end
end
display(svgFiles);
display(distMat);

% MVI_3177_175_woman_black.svg
% MVI_3177_2047_woman_black.svg
% MVI_3178_1913_black1.svg
% MVI_3178_2297_kid1.svg
% MVI_3178_2453_kid1.svg
% MVI_3178_602_black1.svg
%          0    4.6321   43.0203   25.5680   36.3205   50.2176
%          0         0   12.7576    6.4162    7.8527   10.6642
%          0         0         0   10.8882   19.5881   23.1406
%          0         0         0         0   13.3240   11.7520
%          0         0         0         0         0   16.8781
%          0         0         0         0         0         0
