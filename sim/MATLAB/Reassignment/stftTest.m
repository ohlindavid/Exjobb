Fs = 128;

[x1,T] = multigaussdata1(1024,[40 20 10],[1 1 1],[1 1.5 3],[4 12 9],[pi/2 pi pi],Fs);;
Hl=12*lambda; %Long enough window
H=exp(-0.5*([-Hl/2:Hl/2-1]'/lambda).^2);

data = real(x1');
step = 1; %default
nfft = 2048;

epsilon = 0.1;
lambda = 20;
sigma = 18;
ct = (lambda^2+sigma^2)/(lambda^2);
cw = (lambda^2+sigma^2)/(sigma^2);

h = H';
rows = length(data) + 1;
cols = length(data) + length(h);

data = [ zeros(1,length(h)/2) data zeros(1,length(h)/2) ];
data = repmat(data, [rows 1]);

dh = ([ h 0 0 ] - [ 0 0 h ])/2;
dh = dh(2:(length(dh)-1));
th = linspace(-length(h)/2 + 1,length(h)/2, length(h)).*h;

hmat = zeros(size(data));
dhmat = hmat;
thmat = hmat;
M = hmat;

for i=1:rows
    M(i,:) = [ zeros(1,i-1) ones(1, length(h)) zeros(1,cols-i-length(h)+1) ];
    hmat(i,:) = [ zeros(1,i-1) h zeros(1,cols-i-length(h)+1) ];
    dhmat(i,:) = [ zeros(1,i-1) dh zeros(1,cols-i-length(h)+1) ];
    thmat(i,:) = [ zeros(1,i-1) th zeros(1,cols-i-length(h)+1) ];
end

data = data.*M;
data = data - sum(data,2)/length(h).*M;
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

j0s = ct/step*real(Fth).*conj(Fh)./ss;
i0s = cw*nfft/(2*pi)*imag(Fdh).*conj(Fh)./ss;
is = linspace(1, rows, rows);
is = repmat(is, [nfft/2,1]);
js = linspace(1, nfft/2, nfft/2)';
js = repmat(js, [1, rows]);
is = is + round(abs(i0s));
js = js + round(abs(j0s));

marg = 1/(10*epsilon);
xs = (nfft/2)*2*marg; % Marginal på båda sidor enligt marg
ys = 2*(rows-1)*marg; % -"-
rss = zeros(xs, ys);

for m = 1:nfft/2
    for n = 1:(rows-1)
        rss(xs/2 + is(m,n),ys/2 + js(m,n)) = rss(xs/2 + is(m,n), ys/2 + js(m,n)) + ss(m,n);
    end
end

rss = rss((xs/2):(xs/2 + nfft/2),(ys/2):(ys/2 + rows-1));
image(rss)
