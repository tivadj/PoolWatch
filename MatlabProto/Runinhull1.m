%% 2D test
L = linspace(0,2.*pi,6);
xv = cos(L)';
yv = sin(L)';
xv = [xv ; xv(1)];
yv = [yv ; yv(1)];
plot(xv,yv);

x = randn(250,1);
y = randn(250,1);
%in = inpolygon(x,y,xv,yv);
in = utils.inhull([x y], [xv, yv])
plot(xv,yv,x(in),y(in),'r+',x(~in),y(~in),'bo')


%% 3D test
allPoints = 100*mvnrnd([0 0 0], diag([1 3 5]), 100);
scatter3(allPoints(:,1), allPoints(:,2), allPoints(:,3));

% choose some points close to the center
cent = mean(allPoints);
dists = sqrt(sum((repmat(cent, length(allPoints), 1) - allPoints).^2, 2));

rad = 200;
isInHull = dists < rad;
fprintf('hull contains %d triangles\n', sum(isInHull));
ballPoints = allPoints(isInHull,:);
plot3(ballPoints(:,1),ballPoints(:,2),ballPoints(:,3),'r+',...
      allPoints(~isInHull,1),allPoints(~isInHull,2),allPoints(~isInHull,3),'bo');
  
triInds=convhull(ballPoints(:,1),ballPoints(:,2),ballPoints(:,3));
hold on
trisurf(triInds,ballPoints(:,1),ballPoints(:,2),ballPoints(:,3),'FaceColor','c')
hold off

hullPoints = ballPoints(unique(triInds(:)),:);

tess = convhulln(hullPoints);
%tess=[];
in = utils.inhull(allPoints, hullPoints, tess, 0.2);
fprintf(1, 'hull has %d points\n', sum(in));
errs=sum(isInHull ~= in)
