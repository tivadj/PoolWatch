function [ XNorm ] = normalizeHomog(XHomogByRow)
%NORMALIZEHOMOG divides point by W coordinate

hom = size(XHomogByRow,2);
if (hom ~= 3 && hom ~= 4)
    error('Must be 2D points');
end
    
num = size(XHomogByRow,1);

XNorm=zeros(num,hom-1);
for i=1:num
    XNorm(i,:) = XHomogByRow(i,1:hom-1) / XHomogByRow(i,hom);
end

end

