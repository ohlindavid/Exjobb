function [SS,MSS,TI,FI,H]=screassignspectrogram(X,lambda,candsig,NFFT,NSTEP,Fs,e);

% SCREASSIGNSPECTROGRAM [SS,MSS,TI,FI,H]=screassignspectrogram(X,lambda,candsig,NFFT,NSTEP,Fs,e); 
% computes and plots the windowed spectrogram and the scaled reassigned spectrogram.
% 
%
% Output data
%
% SS:  The Gaussian windowed spectrogram
% MSS: The scaled reassigned windowed spectrogram
% TI:  Time vector for the time-frequency plots
% FI:  Frequency vector for the time-frequency plots
% H:   The Gaussian window
%
% Input data
%
% X:    Data sequence 
% lambda:    Parameter of Gaussian window.
% candsig: Candidate sigma, the assumed scaling factor of the signal
% NFFT: The number of FFT-samples, default NFFT=2048.
% NSTEP:The time-step between to spectrum calculations, default NSTEP=1. 
% Fs:   Sample frequency, default Fs=1 
% e:    Smaller spectrum values than this number are not reassigned, default e=0.
%



if nargin<3
    'Error: No data, window length lambda, or candidate sigma input'
end
if nargin<7
   e=0;
end
if nargin<6
   Fs=1; 
end
if nargin<4
   NSTEP=1; 
end
if nargin<4
   NFFT=2048; 
end


%%

% Gaussian window calculation

Hl=12*lambda; %Long enough window
H=exp(-0.5*([-Hl/2:Hl/2-1]'/lambda).^2);

% TH and DH needed for the reassignment

Tvect=[-Hl/2+1:Hl/2]';
TH=Tvect.*H;
DHd=diff(H);
DHd2=interp(DHd,2);
DH=[0;DHd2(2:2:end)];

data=X;
data=data(:);

% Spectrogram calculation


%mvect=[0:NFFT-1];
data=[zeros(fix(Hl/2),1);data;zeros(fix(Hl/2),1)];
datal=length(data(:,1));

timevect=[0:NSTEP:datal-Hl-1];
TI=[];
FF=[];
TFF=[];
DFF=[];
MSS=zeros(NFFT/2,length(timevect));
nmat0=zeros(NFFT,length(timevect));
mmat0=zeros(NFFT,length(timevect));
nmat=zeros(NFFT,length(timevect));
mmat=zeros(NFFT,length(timevect));
for i=0:NSTEP:datal-Hl-1
   testdata=data(i+1:i+Hl);
   testdata=testdata-mean(testdata); % Mean value reduction!
   F=fft(H.*testdata,NFFT);
   TF=fft(TH.*testdata,NFFT);
   DF=fft(DH.*testdata,NFFT);
   FF=[FF F(1:NFFT/2)];
   TFF=[TFF TF(1:NFFT/2)];
   DFF=[DFF DF(1:NFFT/2)];
   TI=[TI i];
end
SS=abs(FF).^2;
% SS = SS + 0.1; % Add epsilon to check effect on final spectrogram
TI=TI/Fs;
FI=[0:NFFT/2-1]'/NFFT*Fs;

% Scaling factors for the scaled Gaussian reassignment

fact=(lambda^2+candsig^2)/(lambda^2);     
fact2=(lambda^2+candsig^2)/(candsig^2);

% Scaled reassignment calculation

% imaginary = max(max(imag(FF)))

for n=1:length(TI)
    for m=1:NFFT/2
        if SS(m,n)>e
            nmat0(m,n)=fact/NSTEP*(real(TFF(m,n).*conj(FF(m,n))./SS(m,n)));
            mmat0(m,n)=NFFT/2/pi*fact2*(imag(DFF(m,n).*conj(FF(m,n))./SS(m,n)));
            nmat(m,n)=n+round(nmat0(m,n));
            mmat(m,n)=m-round(mmat0(m,n));
            if mmat(m,n)>0 & mmat(m,n)<=NFFT/2 & nmat(m,n)>0 & nmat(m,n)<=length(TI) 
                MSS(mmat(m,n),nmat(m,n))=MSS(mmat(m,n),nmat(m,n))+SS(m,n);
            else
                mmat(m,n)=0;
                nmat(m,n)=0;
            end
        end
     end
end