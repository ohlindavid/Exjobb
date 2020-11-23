function [X,T]=multigaussdata1(N,cvect,Avect,Tvect,Fvect,PHvect,Fs);

% MULTIGAUSSDATA creates a complex-valued data vector with K different Gaussian components.
%   [X,T]=multigaussdata(N,cvect,Avect,Tvect,Fvect,PHvect,Fs) generates a complex-valued signal of total length N 
%   including a number of Gaussian components, specified in the (K X 1) vectors cvect. 
%   The amplitudes are specified in the vector (K X 1) Avect, the centre timepoints in (K X 1)
%   Tvect, the frequencies in (K X 1) Fvect and the phases in (K X 1) PHvect
%
%   X: output data vector of length N.
%   T: corresponding time vector of length N.
%   N: data vector length.
%   cvect: K X 1 vector of Gaussian component parameters 
%   Avect: K X 1 vector of amplitudes.
%   Tvect: K X 1 vector of centre values, in seconds. 
%   Fvect: K X 1 vector of frequency values of each component, in Hz.
%   Phvect: K X 1 vectors of phases.
%   Fs: sample frequency, default=1.
%


if nargin<7
    Fs=1;
end


X=zeros(N,1);
Tvect=fix(Tvect*Fs);
Fvect=Fvect/Fs;


for k=1:length(cvect)
    
    Hk=12*cvect(k); %Long enough time vector
    Hcompk=exp(-0.5*([-Hk/2:Hk/2-1]'/cvect(k)).^2);
    

    nvect=[max(Tvect(k)-Hk/2,1):min(Tvect(k)+Hk/2-1,N)]';
    X(nvect)=X(nvect)+Avect(k)*exp(j*(2*pi*Fvect(k).*(nvect-Tvect(k))+PHvect(k))).*Hcompk(nvect-Tvect(k)+Hk/2+1);
end

T=[0:N-1]'/Fs;


    