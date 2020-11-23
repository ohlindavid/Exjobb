[X,T]=multigaussdata1(1024,[40 20 10],[1 1 1],[1 1.5 3],[4 12 9],[pi/2 pi pi],Fs);

figure

plot(T,real(X))


cpar = 20 %Optimal to the second component
[SS,MSS,TI,FI,H]=screassignspectrogram(real(X),cpar,1024,1,255);
    
figure

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






