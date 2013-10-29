function maskedImage = applyMask(image, mask)

maskInt=im2uint8(mask);

if ndims(image) == 2
    maskRgb=maskInt;
else
    maskRgb=cat(3,maskInt,maskInt,maskInt);
end

maskedImage=bitand(image,maskRgb);

end