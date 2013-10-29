
fprintf(1,'composing: video + boundaries');
tic;

% highlight human bodies
parfor i=1:framesCount
    bodyDescrs = detectionsPerFrame{i,1};
    
    %
    num = framesToTake(i);
    image = read(obj, num);
    for k=1:size(bodyDescrs,1)
        box=bodyDescrs(k).BoundingBox;
        image=cv.rectangle(image, box, 'Color', [0 255 0]);
        image=cv.circle(image, bodyDescrs(k).Centroid, 3, 'Color', [0 255 0]);
        
        textPos = [max(box(1),box(1) + box(3) - 26), box(2) - 13];
        image = cv.putText(image, int2str(k), textPos, 'Color', [0 255 0]);
    end
    %imshow(image)
    %refresh
    %pause(0.001);
   
%
    videoUpd(:,:,:,i) = image;
end
fprintf(1, 'took time=%d\n', toc);

%{
% write output
writerObj = VideoWriter('data/out1');
writerObj.open();
writerObj.writeVideo(videoUpd);
writerObj.close();
%}

% play movie
new_movie = immovie(videoUpd);
implay(new_movie);

