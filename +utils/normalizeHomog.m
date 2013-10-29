function [ XNorm ] = normalizeHomog(XHomogByRow)
%NORMALIZEHOMOG divides point by W coordinate
assert(size(XHomogByRow,2) == 3, 'Must be 2D points');

num = size(XHomogByRow,1);

XNorm=zeros(num,2);
for i=1:num
    XNorm(i,:) = XHomogByRow(i,1:2) / XHomogByRow(i,3);
end

end

