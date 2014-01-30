%% read data from SVG documents
svg1Path=fullfile(cd, 'data/markup/MVI_3178_363_black1.svg');

[i1,m1]=utils.getMaskAll(svg1Path,'#FFFF00');

m1B=im2uint8(~m1);
i2=cat(3, bitand(i1(:,:,1),m1B), bitand(i1(:,:,2), m1B), bitand(i1(:,:,3), m1B));
figure(1), imshow(i2)

m2B=im2uint8(m1);
i3=cat(3, bitand(i1(:,:,1),m2B), bitand(i1(:,:,2), m2B), bitand(i1(:,:,3), m2B));
figure(2), imshow(i3)

sum(sum(m1 > 0))