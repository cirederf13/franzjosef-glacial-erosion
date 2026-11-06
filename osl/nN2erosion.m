% Heat-advection + OSL
clear all; close all

% load data file
kparams=xlsread('P84290minus.xlsx');
fileout='output/P84290minus.mat';

%% physical parameters for the heat transfer equation
kappa0 = 30;
Tb_top = 10.;
Tb_bot = 700.;
Vxa    = 0.;
Vya1 = 10;
Lx     = 3.;
Ly     = 25.;
%% numerics: discretization, time steps and plotting frequency
nx    = 4;
ny    = 26;
dx    = Lx/(nx-1);
dy    = Ly/(ny-1);
tmax  = 6;
dt    = min(min(dx,dy)^2/kappa0/2.1,min(dx,dy)/100/max(Vxa,Vya1));

%% Location of the data points to t2rack the history for
nd  = 1;              % number of data point
age2D=ones(1,nd);
age3D=ones(1,nd);
age4D=ones(1,nd);
age5D=ones(1,nd);
xd0  = Lx/2.+0.1;     % pick a random location
yd0  = ones(nd,1)*Ly; % make sure the initial position is on top of the box

%% intialize some tables
Te    = ones(nx  ,ny  );
Te0    = ones(nx  ,ny  );
qx    = zeros(nx-1,ny-2);
qy    = zeros(nx-2,ny-1);
dTedt = zeros(nx  ,ny  );
Vx    = Vxa*ones(nx+1,ny  );
Vy    = Vya1*ones(nx  ,ny+1);
kappa = kappa0*ones(nx  ,ny  );
redisuals=zeros(1,8);
for iy = 1:ny
  yc(iy) = (iy-1)*dy;
  for ix = 1:nx
    xc(ix) = (ix-1)*dx;
    Te0(ix,iy) = Tb_bot+(Tb_top-Tb_bot)*yc(iy)/Ly;
  end
end

%% for loop for an inversion on Vya0, Vya1 and t

for jr=1:500
%tic    

%% randomly selecting the free variables
 Vya1=6+(2.*rand-1.)*5.9;

%% initial conditions for the thermal model
Te=Te0;

%calculate the initial position of the rock that will then be advected
%towards the surface
xd = xd0;
time=tmax;
yd=yd0;
nt=0;
while (time>0) 
    Vya=Vya1; 
    dt    = min(min(dx,dy)^2/kappa0/2.1,min(dx,dy)/100/max(Vxa,Vya));
    time=time-dt;
    yd = yd-Vya*dt;
    nt=nt+1;
end

%% heat equation solver
time=0;it=0;
while (time < tmax)
    
    % pick the vertical velocity and time step
    Vya=Vya1; 
    dt    = min(min(dx,dy)^2/kappa0/2.1,min(dx,dy)/100/max(Vxa,Vya));
    Vy    = Vya*ones(nx  ,ny+1);
    time=time+dt;it=it+1;
    
    % compute diffusion
    qx   =  -   diff(Te(1:end-0,2:end-1),1,1)/dx;    
    qy   =  -   diff(Te(2:end-1,1:end-0),1,2)/dy;
    dTedt(2:end-1,2:end-1) = kappa(2:end-1,2:end-1).*(- diff(qx,1,1)/dx - diff(qy,1,2)/dy); 
      
    % compute advection
    dTedt(:,2:end  ) = dTedt(:,2:end  ) - (Vy(:,2:end-1)>0).*Vy(:,2:end-1).*diff(Te,1,2)/dy;
      
    % update time step
    Te = Te + dt*dTedt;
    
    % set boundary conditions
    %edges
    Te(1,:)=Te(2,:);Te(nx,:)=Te(nx-1,:);
    %top
    Te(:,ny)=Tb_top; 
    %bottom boundary conditions
    Te(:,1)=Tb_bot;
    
    % advect rocks back toward the surface
    yd=yd+Vya*dt;

    % save Tt history. It first finds where the rock is in the mesh and
    % then interpolates the temperature and save it
    id=1;
      ix=floor(xd(id)/dx)+1;
      iy=floor(yd(id)/dy)+1;
      if (ix>0) && (ix<nx) && (iy>0) && (iy<ny)
        t = (xd(id)-xc(ix))/(xc(ix+1)-xc(ix));
        u = (yd(id)-yc(iy))/(yc(iy+1)-yc(iy));
        Tdata(1,it,id) = (1-t)*(1-u)*Te(ix,iy)+t*(1-u)*Te(ix+1,iy)+ ...
                          t*u*Te(ix+1,iy+1)+(1-t)*u*Te(ix,iy+1);
      else
      Tdata(1,it,id) = Tb_top;
      end
    Tdata(2,it,id)=time;  
    
end
 
%% compute OSL nN data and misfit between model prediction and observation
% it first interpolates the Tt path on a higher resolution to make sure it
% remains stable in trapping_BTM_FAD
timemax=max(Tdata(2,:,1));
nstep=500000;
Dtime=(timemax-0.5)/(nstep-1); 
vec=(timemax-0.5):Dtime:timemax;
tempM=interp1(Tdata(2,:,1),Tdata(1,:,1),vec,'linear');
timeM=vec;

% compute nN, residuals and misfits
for k=1:1
[nNf] = trapping_BTM_FAD(timeM,tempM,kparams(k,:));
residuals(k)=(kparams(k,7)-nNf(end))^2/(0.1*kparams(k,7))^2.;
end
Misfit=sum(residuals);

%% save output
output(jr,1)=Vya1;
output(jr,2)=Misfit;

%% plot figure
figure(1);
plot(output(:,1),output(:,2),'o');xlabel('exhumation rates');ylabel('Misfit'); axis square
set(gca,'YScale','log')
drawnow

%% display progress
fprintf('V0= %f m= %f nN_m = %f nN_o= %f trial %i \n',output(jr,1),output(jr,2), nNf(end), kparams(k,7), jr)

%toc
Misfit=0.;resdiduals=0.;
clear Tdata;
end

% save output
save(fileout,'output');