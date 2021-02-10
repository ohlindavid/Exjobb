%% Wavelets

a1 = 15;
b1 = 10;
a2 = 15;
b2 = 5;

morlet1 = @(t) exp(-(a1.^2.*t.^2)/2).*cos(2*pi.*b1.*t);
morlet2 = @(t) exp(-(a2.^2.*t.^2)/2).*cos(2*pi.*b2.*t);

T = -0.18:0.001:0.18

hold on

plot(morlet1(T));
plot(morlet2(T));
