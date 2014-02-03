function result = chooseBoundaryByDensity(densitySet, hitSet, gatherRadius, takeCount, logLevelSize)
    result = zeros(0,size(densitySet,2), class(hitSet));
    hitSetInt32 = int32(hitSet);
    
    numPixLogLevel=0;
    while length(result) < takeCount
        % choose random skin pixels
        seedPixelsCount = 1000;
        pixInd = randi(length(densitySet), seedPixelsCount,1);
        pix = int32(densitySet(pixInd,:));

        % generate some satellite pixels
        satelliteCount = 10;
        pixsAround = repmat(pix, satelliteCount,1) + randi([-gatherRadius gatherRadius], satelliteCount*length(pix),3,'int32');
        newPixels = intersect(pixsAround, hitSetInt32, 'rows');
        newPixelsOut = cast(newPixels, class(hitSet));
        result = union(result, newPixelsOut, 'rows');
        
        curNumPix = floor(length(result) / logLevelSize);
        if curNumPix > numPixLogLevel
            fprintf(1, 'gathered %d pixels\n', length(result));
            numPixLogLevel = curNumPix;
        end
    end
end