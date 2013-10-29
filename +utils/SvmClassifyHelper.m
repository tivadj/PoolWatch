% split whole classification job into buckets because of 'Out of Memory'
% exception.
function ClassifRes = SvmClassifyHelper(cl, XByRow, bucketSize)
if ~isa(XByRow, 'double') 
    error('svmclassify requires points to be of double type');
end

tic;

n=length(XByRow);

completeParts = floor(n / bucketSize);

ClassifRes = zeros(completeParts+1,bucketSize);
%ClassifRes = zeros(n,1);

parfor i=1:completeParts
    fprintf(1, 'svmclassify part %d of %d\n', i,completeParts+1);

    len = bucketSize;
    r1 = (i-1) * len + 1;
    r2 = r1 + len - 1;
    res = svmclassify(cl, XByRow(r1:r2,:));
    
    %ClassifRes(((i-1)*bucketSize+1) : (i*bucketSize)) = res'; % linear
    %ClassifRes(r1:r2,1) = res;
    ClassifRes(i,:) = res';
end

% classify last segment
r1 = completeParts*bucketSize+1;
len = mod(n, bucketSize);
res = svmclassify(cl, XByRow(r1:(r1+len-1),:));
ClassifRes(completeParts+1,1:len) = res';

% reshape result
ClassifRes = reshape(ClassifRes', [], 1);
ClassifRes = ClassifRes(1:n);

fprintf(1, 'SvmClassifyHelper took %f\n', toc);

end
