% Labels regions in the image as indices from connected component 
% __connComps__ list. 
% Also shifts labels by __labelShift__ to avoid labels overlapping.
%
function imageProps = drawRegionProps(imageGray, connComps, labelShift)
    if ndims(imageGray) > 2
        error('imageGray must be a gray image');
    end
    
    imageRgb = cat(3, imageGray, imageGray, imageGray);
    grayPixelsCount = size(imageGray,1)*size(imageGray,2);
    
    c_list = ['g' 'r' 'b' 'c' 'm' 'y'];
    c_list = utils.convert_color(c_list)*255;

    imageProps = imageRgb;
    
    allRowsCols = [];
    
    for compInd=1:connComps.NumObjects
        oneInds=connComps.PixelIdxList{compInd};
        [rows,cols] = ind2sub(size(imageGray), oneInds);
        allRowsCols = [allRowsCols; rows,cols];
    end
    totalCentroid = mean(allRowsCols);
    totalCentroid = circshift(totalCentroid, [0 1]); % X=Cols Y=Rows
    
    for compInd=1:connComps.NumObjects
        % pick component color
        color = c_list(1+mod(compInd-1, length(c_list)),:);

        oneInds=connComps.PixelIdxList{compInd};
        imageProps(oneInds)=color(1);
        imageProps(oneInds + grayPixelsCount)=color(2);
        imageProps(oneInds + 2*grayPixelsCount)=color(3);

        % find centroid
        [rows,cols] = ind2sub(size(imageGray), oneInds);
        centroid = mean([cols rows], 1);
        
        % shift label out of component
        dir = centroid-totalCentroid;
        dir = dir / norm(dir);
        offsetCentroid = centroid + labelShift*dir;
        imageProps = cv.line(imageProps, centroid, offsetCentroid, 'Color', color);
        
        imageProps = cv.putText(imageProps, sprintf('%d', compInd), offsetCentroid, 'Color', color);
    end
end

