function [nNf] = trapping_BTM_FAD(time,temp,kparams);

Ma = 3600*24*365*1e6;
% Extract parameters
% Here I hard code the parameters
ddot = kparams(1)/(1e3*365*24*3600); %convert from Gy/kyr to Gy/seconds
D0 = kparams(2);
s = 10.^kparams(3);
rhop = 10.^kparams(4);
Et = kparams(5);
Eu = kparams(6);

% Define constances
kb = 8.617343e-5; alpha = 1; Hs = 3e15; %s value after Huntley (2006) J. Phys. D.
magic_ratio = ddot/D0;
nrp = 100;
nEb = 50;
nstep = length(time);
dt = ((max(time)-min(time))/(nstep-1))*Ma; %time step in Myr is converted into second.

% Define rprime range and tau athermic
rprime = linspace(0.,2.5,nrp); %create vector of rprime distances
pr = 3.*rprime'.^2.*exp(-rprime'.^3.); %calc p(r') eq 3 in Kars et al 2008
npr = sum(pr);

inv_tauath = ones(nEb,1)*(Hs*exp(-(rhop.^-(1./3)).*rprime)); %combine eq 1 and 3 from Kars et al 2008, convert to Ma

% Create variables for lili2013 BTM
Eb = linspace(0.01,Et+0.01,nEb)';
pEb = exp(-Eb'/Eu);
npEb = sum(pEb);
T = temp+273.15;

% computes nN for a Tt path
nN = zeros(nEb,nrp,nstep);
nNf = zeros(1,nstep);
for i = 2:nstep
	inv_tauth = s*exp(-(Et-Eb)./(kb.*T(i-1)))*ones(1,nrp);
	alpha = magic_ratio+inv_tauth+inv_tauath;
	nN(:,:,i) = (nN(:,:,i-1)+magic_ratio.*dt).*(1./(1+dt.*alpha));
	nNf(i) = pEb*nN(:,:,i)*pr;
end
nNf = nNf./npEb./npr;
