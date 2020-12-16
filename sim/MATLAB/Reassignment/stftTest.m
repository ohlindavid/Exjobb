Fs = 1024;
tmax = 0.5;
fmax = 20;

[x1,T] = multigaussdata1(1024,[40 20 10],[1 1 1],[1 1.5 3],[3 20 9],[pi/2 pi pi],Fs);

data = real(x1');
step = 1; %default
nfft = 2048;

epsilon = 0.1;
lambda = 20;
sigma = 26;

Hl=12*lambda; %Long enough window
H=exp(-0.5*([-Hl/2:Hl/2-1]'/lambda).^2);
h = H';
rows = length(data) + 1;
cols = length(data) + Hl;

data = [ zeros(1,Hl/2) data zeros(1,Hl/2) ];
data = repmat(data, [rows 1]);

dh = ([ h 0 0 ] - [ 0 0 h ])/2;
dh = dh(2:(Hl-1));
th = linspace(-Hl/2 + 1,Hl/2, Hl).*h

hmat = zeros(size(data));
dhmat = hmat;
thmat = hmat;
M = hmat;

for i=1:rows
    M(i,:) = [ zeros(1,i-1) ones(1, length(h)) zeros(1,cols-i-Hl+1) ];
    hmat(i,:) = [ zeros(1,i-1) h zeros(1,cols-i-Hl+1) ];
    dhmat(i,:) = [ zeros(1,i-1) dh zeros(1,cols-i-Hl+1) ];
    thmat(i,:) = [ zeros(1,i-1) th zeros(1,cols-i-Hl+1) ];
end

data = data.*M;
data = data - sum(data,2)/Hl.*M;
fh = data.*hmat;
fdh = data.*dhmat;
fth = data.*thmat;

% for i = 1:rows
%     fh(i,:) = [ fh(i, i:(cols)) zeros(1,i-1) ];
% end

Fh = fft(fh, nfft, 2);
Fh = Fh(:, 1:(nfft/2))';
Fdh = fft(fdh, nfft, 2);
Fdh = Fdh(:, 1:(nfft/2))';
Fth = fft(fth, nfft,2);
Fth = Fth(:, 1:(nfft/2))';
ss = abs(Fh).^2;
ss = ss + epsilon;
% ssdh = abs(Fdh).^2';
% sst = abs(Fth).^2';

ct = (lambda^2+sigma^2)/(lambda^2);
cw = (lambda^2+sigma^2)/(sigma^2);

j0s = ct/step*real(Fth.*conj(Fh)./ss);
i0s = cw*nfft/2/pi*imag(Fdh.*conj(Fh)./ss);
is = linspace(1, nfft/2, nfft/2)';
js = linspace(1, rows, rows);
is = repmat(is, [1, rows]);
js = repmat(js, [nfft/2, 1]);
is = is + round(i0s);
js = js + round(j0s);

marg = 2000;
xs = (nfft/2) + 2*marg; % Marginal på båda sidor enligt marg
ys = (rows-1) + 2*marg; % -"-
rss = zeros(xs, ys);

for m = 1:nfft/2
    for n = 1:(rows-1)
        rss(marg + is(m,n), marg + js(m,n)) = rss(marg + is(m,n), marg + js(m,n)) + ss(m,n);
    end
end

rss = rss(marg:(xs-marg-1),marg:(ys-marg-1));

TI = linspace(0, rows-1,rows-1)/Fs;
FI = [0:nfft/2-1]'/nfft*Fs;

figure(1)

subplot(121)
c=[min(min(ss)) max(max(ss))];
pcolor(TI,FI,ss(:,1:rows-1)) 
shading interp
caxis(c)
axis([0 tmax 0 fmax])
ylabel('Frequency (Hz)')
xlabel('Time (s)')
title('Spectrogram')

subplot(122)
c=[min(min(rss)) max(max(rss))/10];   % Sometimes I divide the maximum value with some random number for a better view
pcolor(TI,FI,rss)  
shading interp
caxis(c)
axis([0 tmax 0 fmax])
ylabel('Frequency (Hz)')
xlabel('Time (s)')
title('Scaled reassigned spectrogram')
