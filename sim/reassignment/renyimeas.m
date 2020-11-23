function [Renyi]=renyimeas(S,tint,fint,NFFT,NSTEP,Fs);

% [Renyi]=renyimeas(S,tint,fint,NFFT,NSTEP,Fs); 
% computes the Renyi entropy in the defined time- and frequency area
%
%
% Output data
%
% Renyi: The Renyi entropy
%
% Input data
%
% S: The time-frequency image
% tint: The timeinterval [tstart tend] defined in seconds.
% fint: The frequency interval [fstart fend] defined in Hz.
% NFFT: The number of FFT-samples.
% NSTEP: The time-step between to spectrum calculations. 
% Fs: Sample frequency. 
%


t0i=fix(tint(1)*Fs/NSTEP);
t1i=fix(tint(2)*Fs/NSTEP);
f0i=fix(fint(1)/Fs*NFFT);
f1i=fix(fint(2)/Fs*NFFT);

Stest=S(f0i+1:f1i+1,t0i+1:t1i+1);
Stestn=Stest./sum(sum(Stest));
[m,n]=size(Stest);

% Renyi-entropy
p=3;
Renyi=1/(1-p)*log2(sum(sum((Stestn).^p)));
