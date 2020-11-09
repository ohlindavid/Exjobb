function [Y] = oneoverfnoise(pinkwn,alpha,Fs);
% function: [Y] = oneoverfnoise(pinkwn,alpha,Fs);

% A modified version of the spectral-based code of coloured noise
% sequences based on Gaussian white noise; Mean value is zero and variance is one.
%
% H. Zhivomirov. A Method for Colored Noise Generation. Romanian Journal of Acoustics and 
%               Vibration, ISSN: 1584-7284, Vol. XV, No. 1, pp. 14-19, 2018. 
%               
% H. Zhivomirov (2020). Pink, Red, Blue and Violet Noise Generation with Matlab 

% Y - output matrix N x numRe
%
% pinkwn - matrix of white noise realizations N x numRe
% alpha - 0 = white noise; 1 = pink noise; 2 = Brownian (red) noise; 
% Fs - sample frequency
%
% Example: 100 5s-realizations of EEG, sample frequency 256 (1280 samples) : 
% Y = oneoverfnoise(1280,100,2,256);


[N,numRe]=size(pinkwn);
Y=zeros(N,numRe);


for i=1:numRe
    % define the length of the vector
    % ensure that the M is even
    if rem(N,2)
        M = N+1;
    else
        M = N;
    end
    
    X = fft(pinkwn(:,i));
    
    % prepare a vector for 1/f multiplication
    NumUniquePts = M/2 + 1;
    f=([0 [1:NumUniquePts-1]]'/NumUniquePts*Fs/2);
    oneoverf_noise = 1./(f.^alpha);
    oneoverf_noise(1)=0;
    
    % multiplicate the left half of the spectrum so the power spectral density
    % is proportional to the frequency by factor 1/f, i.e. the
    % amplitudes are proportional to 1/sqrt(f)
    X(1:NumUniquePts) = X(1:NumUniquePts).*sqrt(oneoverf_noise);
    
    % prepare a right half of the spectrum - a copy of the left one,
    % except the DC component and Nyquist frequency - they are unique
    X(NumUniquePts+1:M) = real(X(M/2:-1:2)) -1i*imag(X(M/2:-1:2));
    y = real(ifft(X));
    
    % ensure unity standard deviation and zero mean value
    y = y - mean(y);
    y = y/sqrt(sum(y.^2))*sqrt(N);
    Y(:,i)=y;
end

% NFFT=(2^round(log2(N)))*8;
% normtruef=sum(oneoverf_alpha(2:end))*2;
% 
% 
% figure(1)
% plot([0:NFFT-1]/NFFT*fs,mean(abs(fft(Y,NFFT))'.^2)/N/length(Y(:,1)),'r')
% hold
% plot(f,oneoverf_alpha/normtruef,'b')
% hold
% axis([0 30 0 0.1])
% xlabel('Frequency (Hz)')
% title('True spectral density (blue) and mean of periodograms of all realizations (red)')
% 
% if N>=3
% figure(2)
% plot([0:N-1]/fs,Y(:,2:4))
% xlabel('Time (s)')
% title('Three example realizations')

%S_oneoverf=oneoverf_alpha./sum(oneoverf_alpha)/2;

end