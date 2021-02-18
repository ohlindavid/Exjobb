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

%%

%Equation of Love, 2021-02-10%L=Anna's Love for David
%t=time from 2020-01-25 in seconds
%(1000*exp(t)) is the increase of love with respect to time t
%(abs(f)) is the love given to David at each timestep t
%f=David mood factor, -100= extremely sad, 100=extremely happy, at a 
%certain time t
%f=(-100:100);    
%Side note: one can plot the mood with respect to time, but this would     
%require measurements from advanced technology
%L=(1000*exp(t))-(abs(f)))
%where t->infinity
%Example:
t=33264000; %in seconds, approx 1 year and 20 days
f=-5; %translates to a little sad/worried, perhaps about future
L=(1000*exp(t))-(abs(f))