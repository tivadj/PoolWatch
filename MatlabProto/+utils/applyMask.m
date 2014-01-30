function maskedImage = applyMask(image, mask)
assert(isa(image,'int8') || isa(image,'uint8'));

maskInt=im2uint8(mask);

if ndims(image) == 2
    maskRgb=maskInt;
else
    maskRgb=cat(3,maskInt,maskInt,maskInt);
end

maskedImage=bitand(image,maskRgb);

end