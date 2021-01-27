clear


NSTEP=1;
Fs=128;
e=0;
NFFT=2048;
nch = 6;
nsig = 5;
len= 321;
resize = 1;

trial = dir;

for k = 3:(length(trial)-2)

load(trial(k).name)

trial(k).name

% [X,T]=multigaussdata1((1024/128)*Fs,[40 20 10],[1 1 1],[1 1.5 3],[4 12 9],[pi/2 pi pi],Fs);
% [X,T]=multigaussdata1(1024,[30],[1],[1.5],[12],[pi],Fs);

processed = zeros(nch,nsig,NFFT/(2*resize),len);

for ch = 1:nch

temp = eval(trial(k).name);

X = decimate(temp(:,ch),4);

candsigvect = [12 16 20 24 28]; % Candidate sigma for the unknown sigma of X
% Renyivect=zeros(1,length(candsigvect));

lambda = 20; % Window parameter

for i=1:length(candsigvect)

candsig = candsigvect(i)

[SS,MSS,TI,FI,H]=screassignspectrogram1(real(X),lambda,candsig,NFFT,NSTEP,Fs,e);

MSS = imresize(MSS, [ NFFT/(2*resize) 321 ]);

FIr = linspace(FI(1),FI(end),NFFT/(2*resize))';

processed(ch,i,:,:) = MSS;

figure(1)

subplot(121)
image(SS/200)

% c=[min(min(SS)) max(max(SS))];
% pcolor(TI,FI,SS)
% shading interp
% caxis(c)
% axis([0 2.5 0 30])
% ylabel('Frequency (Hz)')
% xlabel('Time (s)')
% title('Spectrogram')
%
subplot(122)
image(MSS/200)

%pause(1)

% c=[min(min(MSS)) max(max(MSS))/100];   % Sometimes I divide the maximum value with some random number for a better view
% pcolor(TI,FIr,MSS)
% shading interp
% caxis(c)
% axis([0 2.5 0 30])
% ylabel('Frequency (Hz)')
% xlabel('Time (s)')
% title('Scaled reassigned spectrogram')

% figure(2)
% subplot(121)
% plot(TI,MSS)
% title('SRS')
% xlabel('Time (s)')
% subplot(122)
% plot(FI,MSS')
% title('SRS')
% xlabel('Frequency (Hz)')

%'Renyi entropy'

%[Renyi]=renyimeas(MSS,[1 2],[5 20],NFFT,NSTEP,Fs)

%Renyivect(i)=Renyi;

% pause


end

end

dlmwrite("r" + trial(k).name,processed);

end

%figure(3)
%plot(candsigvect,Renyivect)
%title('Renyi entropy')
%xlabel('Candidate sigma')
