% swimmer moves with constant velocity
%
dt=1;
F=[1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];
Q=[dt^2 0    dt 0; 
   0    dt^2 0  dt; 
   dt   0    1  0; 
   0    dt   0  1];
H=[1 0 0 0; 0 1 0 0];
R=diag([10 10]);
n=120;

%% simulate actual (real) flow
cent=[100 100];
xActArray=zeros(4,n);
yArray=zeros(2,n);
xprev=zeros(4,1);
x=zeros(4,1);
for k=0:n
    ang=(pi/(4*20))*k;
    ra=50;
    x(1) = cent(1) + ra*cos(ang);
    x(2) = cent(2) + ra*sin(ang);
    if (xprev(1) ~= 0)
        x(3) = x(1) - xprev(1);
        x(4) = x(2) - xprev(2);

        xActArray(:,k)=x;
        yArray(:,k)=H*x + (sqrt(R)*randn(2,1));
    end
    xprev = x;
end
%% plot real data
figure(1);
plot(xActArray(1,:),xActArray(2,:),'r');
title('(x,y)');
axis([50-30 150+30 50-30 150+30]);
axis square
hold on
plot(yArray(1,:),yArray(2,:),'k');
hold off

figure(3);
plot(xActArray(3,:),'--k');
title('speed');
hold on
plot(xActArray(4,:),'-.k');
plot(sqrt(xActArray(3,:).^2+xActArray(4,:).^2),'r');
hold off


% Kalman filter for swimmer position
xestp =[100; 100; 0; 0];  % x0
Pp=diag([250 250 250 250]); % P0
x=xestp;
xEstArray=zeros(4,n+1);
xEstArray(:,1)=x;
for k=1:n
    xestm = F*xestp;
    Pm = F*Pp*F' + Q;
    K = Pm*H'*inv(H*Pm*H' + R);
   
    %if k>=20 && k<30 %&& mod(k,2)==0
    %if mod(k,2)==0
    if mod(floor(mod(k,10)),2) == 0
        xestp = xestm;
    else
        yk = yArray(:,k);
        xestp = xestm + K*(yk - H*xestm);
    end
    
    Pp = (eye(4)-K*H)*Pm*(eye(4)-K*H)' + K*R*K';
    
    xEstArray(:,k+1) = xestp;
end
figure(1);
hold on
plot(xEstArray(1,:),xEstArray(2,:),'-g.','LineWidth',2);
hold off

figure(2);
errsMeas=sum((yArray-xActArray(1:2,:)).^2);
fprintf(1, 'std dev meas err=%f\n', std(errsMeas));
errsEst=sum((xEstArray(1:2,2:end)-xActArray(1:2,:)).^2);
fprintf(1, 'std dev est err=%f\n', std(errsEst));
hold on
plot(errsEst,'g');
hold off
title('meas error & estim error');

figure(3);
hold on
plot(xEstArray(3,:),'--g');
title('speed');
plot(xEstArray(4,:),'-.g');
plot(sqrt(xEstArray(3,:).^2+xEstArray(4,:).^2),'g');
hold off
ylim([-5 5]);


figure(4)
for k=1:n
    plot(xActArray(1,:),xActArray(2,:),'r');
    title('(x,y)');
    axis([50-30 150+30 50-30 150+30]);
    axis square
    hold on
    plot(yArray(1,:),yArray(2,:),'k');
    
    
    c=0:0.5:2*pi;
    plot(xActArray(1,k),xActArray(2,k),'r.'); %plot swimmer
    plot(yArray(1,k),yArray(2,k),'k.'); %plot sensor
    plot(xEstArray(1,k) + 3*cos(c),xEstArray(2,k)+3*sin(c),'g');
    hold off
    pause(0.2);
end

figure(1);
