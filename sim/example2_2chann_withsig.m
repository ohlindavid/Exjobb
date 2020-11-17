Fs=256;
noreal=10; % Number of realizations
N=1024; % Simulated data length
M=N/2; % Spectrum model frequency values 0 to Fs/2


% Weighting factors for the different parts of the EEG-noise

alpha_fact=5; 
oneoverf_fact=5;
measn_fact=1;

totnorm=sqrt(alpha_fact^2+oneoverf_fact^2+measn_fact^2); % Normalized to total energy one

alpha_fact=alpha_fact/totnorm;
oneoverf_fact=oneoverf_fact/totnorm;
measn_fact=measn_fact/totnorm;




% White noise simulations for the different parts of the EEG-noise,
% channel 1 and 2

alpha_wn1=randn(N,noreal);
oneoverf_wn1=randn(N,noreal);
measn_wn1=randn(N,noreal); 

alpha_wn2=randn(N,noreal);
oneoverf_wn2=randn(N,noreal);
measn_wn2=randn(N,noreal); 


% Simulation of alpha-noise between 8 and 12 Hz, 6th order Butterworth

[B,A] = butter(3,[8/Fs*2 12/Fs*2]);

S_alpha=abs(freqz(B,A,M)).^2;
S_alpha=S_alpha./sum(S_alpha)/2*N;

S_alpha=alpha_fact.^2*S_alpha;

alpha_noise1=zeros(N,noreal);
for i=1:noreal
   alpha_noise1(:,i)=filter(B,A,alpha_wn1(:,i));
   alpha_noise1(:,i)=alpha_noise1(:,i)./sqrt(alpha_noise1(:,i)'*alpha_noise1(:,i))*sqrt(N); % Normalized to power one
end



alpha_noise2=zeros(N,noreal);
for i=1:noreal
   alpha_noise2(:,i)=filter(B,A,alpha_wn2(:,i));
   alpha_noise2(:,i)=alpha_noise2(:,i)./sqrt(alpha_noise2(:,i)'*alpha_noise2(:,i))*sqrt(N); % Normalized to power one
end

alpha_noise1=alpha_fact*alpha_noise1;
alpha_noise2=alpha_fact*alpha_noise2;


% Simulation of the  1/f-noise

f_slope=1;
f=[0:M-1]'/(M*2)*Fs;
S_oneoverf = 1./(f.^(f_slope));
S_oneoverf(1)=0;
S_oneoverf=S_oneoverf./sum(S_oneoverf)/2*N;

S_oneoverf=(oneoverf_fact.^2)*S_oneoverf;

[oneoverf_noise1] = oneoverfnoise(oneoverf_wn1,f_slope,Fs); % Normalized to power one
[oneoverf_noise2] = oneoverfnoise(oneoverf_wn2,f_slope,Fs); % Normalized to power one

oneoverf_noise1=oneoverf_fact*oneoverf_noise1;
oneoverf_noise2=oneoverf_fact*oneoverf_noise2;

% Simulation of measurement noise

for i=1:noreal
    measn_wn1(:,i)=measn_wn1(:,i)./sqrt(measn_wn1(:,i)'*measn_wn1(:,i))*sqrt(N); % Normalized to mean power one
    measn_wn2(:,i)=measn_wn2(:,i)./sqrt(measn_wn2(:,i)'*measn_wn2(:,i))*sqrt(N); % Normalized to mean power one
end

measn_wn1=measn_fact*measn_wn1;
measn_wn2=measn_fact*measn_wn2;




%%

dist=[5]  % Distance between electrodes in cm

% Approximately the coherence shape of Figure 4 and, top in Srinivasan07. 
% Note that Figure 4 shows the squared coherence, i.e. coh_th.^2 

coh_th_oneoverf=exp(-0.01*(dist.^2))


% Stronger coherence for the alpha-activity

coh_th_alpha=exp(-0.002*(dist.^2)) % Reasonable approximation based on figure 7


% Amplitude of the different mixing factors to give the corresponding separate coherence 
% withour meas. noise solving coh_th=2*amp_H*S./((1+amp_H^2)*S);    

ampH_oneoverf=(1./coh_th_oneoverf-sqrt(1./(coh_th_oneoverf).^2-1)) 
ampH_alpha=(1./coh_th_alpha-sqrt(1./(coh_th_alpha).^2-1)) 

% All the noise including mixing from the other channel to the separate weighting of 1/f and
% alpha, and measurement noise

total_noise1=oneoverf_noise1+alpha_noise1+ampH_oneoverf*oneoverf_noise2+ampH_alpha*alpha_noise2+measn_wn1;
total_noise2=oneoverf_noise2+alpha_noise2+ampH_oneoverf*oneoverf_noise1+ampH_alpha*alpha_noise1+measn_wn2;

figure

plot([0:N-1]/Fs,[total_noise1(:,1) total_noise2(:,1)+5])
xlabel('Time(s)')
title('Corresponding noise sequences of the two channels')

'Average correlation between two channels'
mean(diag(corr(total_noise1,total_noise2)))

%%



S_12=2*ampH_alpha*S_alpha+2*ampH_oneoverf*S_oneoverf;
S_11=(1+ampH_alpha.^2)*S_alpha+(1+ampH_oneoverf.^2)*S_oneoverf+(measn_fact).^2;
S_22=S_11;

COH=S_12./sqrt(S_11.*S_22);
 
figure
subplot(121)
plot([0:M-1]/N*Fs,S_11)
axis([0 30 0 max(S_11)])
title('Spectral density')
xlabel('Frequency(Hz)')
ylabel('S_1(f)=S_2(f)')


subplot(122)
plot([0:M-1]/(2*M)*Fs,COH.^2)
axis([0 30 0 1])
xlabel('Frequency(Hz)')
ylabel('Squared coherence')
title('Squared coherence between the 2 channels')



%%

SNR=20; % SNR in dB, relating the total energy of the signal to the total power of the noise
lambda=10^(SNR/10); % Corresponding SNR-factor 


phasediff=pi/2;

% Gaussian envelope multi-component signal simulation

A1=[1];
A2=[1];


phasemat1=rand(noreal,1)*2*pi; % Uniformly distributed phases of the transient components
phasemat2=phasemat1+ones(noreal,1)*phasediff; % Phases of channel 2
 
[signal1,T]=multigaussdata(N,[512],A1,[1],[4],phasemat1,Fs);
[signal2,T]=multigaussdata(N,[512],A2,[1],[4],phasemat2,Fs);

 
% Adding the simulated signals and the EEG-noise together with the correct SNR-factor 
sim_signal1=lambda/sqrt(lambda+1)*signal1+1/sqrt(lambda+1)*total_noise1;
sim_signal2=lambda/sqrt(lambda+1)*signal2+1/sqrt(lambda+1)*total_noise2;


figure

plot([0:N-1]/Fs,[sim_signal1(:,1) sim_signal2(:,1)+5])
xlabel('Time(s)')
title('Corresponding signal+noise sequences of the two channels')