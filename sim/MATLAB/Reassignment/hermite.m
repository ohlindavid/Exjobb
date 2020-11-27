function [H]=hermite(c,K);

% Computation of K Hermite functions with parameter c

M=12*c;

if abs(M/2-fix(M/2))<0.1
  t1=[-M/2+1:M/2]'/c;
else
  t1=[-(M-1)/2:(M-1)/2]'/c;
end

h(:,1)=ones(M,1);
h(:,2)=2*t1;

if K>1
for i=2:K
    h(:,i+1)=2*t1.*h(:,i)-2*(i-1)*h(:,i-1);
end
end

for i=0:K
  H(:,i+1)=(h(:,i+1).*exp(-(t1.^2)/2)/sqrt(c*sqrt(pi)*2^(i)*factorial(i)));
end

H=H(:,1:K); %The number of K final Hermite functions
