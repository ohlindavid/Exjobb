
NFFT=2048;
NSTEP=1;
Fs=128;
e=0;

[X,T]=multigaussdata(1024,[40 20 10],[1 1 1],[1 1.5 3],[4 12 9],[pi/2 pi pi],Fs);

%[X,T]=multigaussdata(1024,[30],[1],[1.5],[12],[pi],Fs);

candsigvect=[10:2:40]; % Candidate sigma for the unknown sigma of X
Renyivect=zeros(1,length(candsigvect));
figure

plot(T,real(X))

lambda=20 % Window parameter


for i=1:length(candsigvect)

candsig=candsigvect(i)
    
[SS,MSS,TI,FI,H]=screassignspectrogram1(real(X),lambda,candsig,NFFT,NSTEP,Fs,e);

figure(1)

subplot(121)
c=[min(min(SS)) max(max(SS))];
pcolor(TI,FI,SS)  
shading interp
caxis(c)
axis([0 4 0 30])
ylabel('Frequency (Hz)')
xlabel('Time (s)')
title('Spectrogram')

subplot(122)
c=[min(min(MSS)) max(max(MSS))/10];   % Sometimes I divide the maximum value with some random number for a better view
pcolor(TI,FI,MSS)  
shading interp
caxis(c)
axis([0 4 0 30])
ylabel('Frequency (Hz)')
xlabel('Time (s)')
title('Scaled reassigned spectrogram')

figure(2)
subplot(121)
plot(TI,MSS)
title('SRS')
xlabel('Time (s)')
subplot(122)
plot(FI,MSS')
title('SRS')
xlabel('Frequency (Hz)')

'Renyi entropy'

[Renyi]=renyimeas(MSS,[1 2],[5 20],NFFT,NSTEP,Fs)

Renyivect(i)=Renyi;

%pause


end

figure(3)
plot(candsigvect,Renyivect)
title('Renyi entropy')
xlabel('Candidate sigma')








