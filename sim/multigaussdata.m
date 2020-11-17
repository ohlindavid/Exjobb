function [X,T]=multigaussdata(N,Hlvect,Avect,Tvect,Fvect,PHmat,Fs);

% MULTIGAUSSDATA creates a complex-valued data vector with K different Gaussian components.
%   [X,T]=multigaussdata(N,Hlvect, Avect,Tvect,Fvect,PHvect,Fs) generates a complex-valued signal of total length N 
%   including a number of Gaussian components, of different lengths, specified in the (1 X K) vectors Hlvect. 
%   The amplitudes are specified in the vector (1 X K) Avect, the centre timepoints in (1 X K)
%   Tvect, the frequencies in (1 X K) Fvect and the random phases  in
%   (noreal X K) PHmat
%
%   X: output data vector of length N.
%   T: corresponding time vector of length N.
%   N: data vector length.
%   Hlvect: 1 X K vector of Gaussian component lengths.
%   Avect: 1 X K vector of amplitudes.
%   Tvect: 1 X K vector of centre values, in seconds. 
%   Fvect: 1 X K vector of frequency values of each component, in Hz.
%   PHmat: noreal X K matrix of phases.
%   Fs: sample frequency, default=1.
%


if nargin<7
    Fs=1;
end

[noreal,K]=size(PHmat);

X=zeros(N,noreal);
Tvect=fix(Tvect*Fs);
Fvect=Fvect/Fs;

for l=1:noreal


for k=1:K
    
    Hk=Hlvect(k);

    c=1/8*Hk;
    Hcompk=exp(-0.5*([-Hk/2:Hk/2-1]'/c).^2);

    nvect=[max(Tvect(k)-Hk/2,1):min(Tvect(k)+Hk/2-1,N)]';
    X(nvect,l)=X(nvect,l)+Avect(k)*exp(i*(2*pi*Fvect(k).*(nvect-Tvect(k))+PHmat(l,k))).*Hcompk(nvect-Tvect(k)+Hk/2+1);

end

X(:,l)=real(X(:,l));
X(:,l)=X(:,l)/sqrt(X(:,l)'*X(:,l));

end

T=[0:N-1]'/Fs;




    