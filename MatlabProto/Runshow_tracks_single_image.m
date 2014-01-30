imshow(I)
xlim([0 640]);
ylim([0 480]);
hold on
for r=1:length(tracks)
    track=tracks{r,1};
    prevPos=[];
    for i=1:length(track.Assignments)
        ass = track.Assignments{i,1};
        if length(ass) == 0
            continue;
        end
        
        %pos = ass.DetectionOrPrediction; % TODO: not working (error: No appropriate method, property, or field DetectionOrPrediction for class ShapeAssignment.)
        box=[];
        if ass.IsDetectionAssigned
            error('NotImplementedExceptino');
            %pos=ass.DetectionCentroid;
            %box=ass.DetectionBoundingBox;
        else
            pos=ass.PredictedPos;
        end
        
        if length(prevPos) > 0
            plot([prevPos(1) pos(1)], [prevPos(2) pos(2)], 'g');
        else
            plot(pos(1), pos(2),'go');
        end
        
        % draw box around occurence
        if (length(box) > 0)
            boxXs = [box(1), box(1) + box(3), box(1) + box(3), box(1), box(1)];
            boxYs = [box(2), box(2), box(2) + box(4), box(2) + box(4), box(2)];
            %plot(boxXs, boxYs, 'g');
        end
        
        prevPos = pos;
    end
end
hold off
