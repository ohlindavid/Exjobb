clear;
close all;
clc;
E = 60;
time = 4;
N = 1024;
M = N/2;
noreal = 10;
Fs = 256;

for j=1:100

    % Placeholder matrix where element (i,j) notes the distance matrix between
    % channel i and j. 
    dMatrix = rand(E,E);
    dMatrix = dMatrix/norm(dMatrix);
    dMatrix = (dMatrix'+dMatrix)/2;

    % Noise weight vector. [alpha , 1/f , m]
    w_noise = [1 , 1 , 1]';
    w_noise = w_noise/norm(w_noise);

    % Simulate alpha noise
    [B,A] = butter(3,[8/Fs*2 12/Fs*2]);
    alpha_power = abs(freqz(B,A,M)).^2;
    alpha_power = alpha_power./sum(alpha_power)/2*N;
    alpha_noise = zeros(N,E);
    alpha_wn1 = randn(N,E);
    for i=1:E
       alpha_noise(:,i)=filter(B,A,alpha_wn1(:,i));
       alpha_noise(:,i)=alpha_noise(:,i)./sqrt(alpha_noise(:,i)'*alpha_noise(:,i))*sqrt(N); % Normalized to power one
    end

    % Simulate pink noise
    x = pinknoise(N,E)*20;
    y = fft(x);
    n = length(x);          % number of samples
    f = (0:n-1)*(Fs/n);     % frequency range
    power = abs(y).^2/n;    % power of the DFT
    oof_noise  = x;

    % Simulate measurement noise
    measurement_noise = randn(N,E);

    % Define and calc Coherence between channels of different noises.
    Coh_oof = exp(-0.01*(dMatrix.^2));
    Coh_alpha = exp(-0.002*(dMatrix.^2));


    tot_noise = w_noise(1)*alpha_noise + w_noise(2)*measurement_noise + w_noise(3)*oof_noise;


    in = linspace(0,time,N);
    for i=1:5
        z = 0.1*rand();
     %   in2(:,i) =  exp(-1/5*in).*sin(3*in+2+z) + exp(-10*in).*sin(10*in) - 5*sin(12*in-pi/5+z).*exp(-2*in);
        in2(:,i) = exp(-2*in).*sin(8*in+2) + exp(-in).*sin(2*in+z) - 5*sin(2*in-pi/0.8+z).*exp(-2*in);
    end
    in = in2;
    SNR=20;
    lambda=10^(SNR/10);
    out = sqrt(lambda/(lambda+1)).*in./norm(in)+1/sqrt(lambda+1)*tot_noise(:,5)/norm(tot_noise(:,5));
    csvwrite('sim_test_2ch_' + string(j),out)
end
