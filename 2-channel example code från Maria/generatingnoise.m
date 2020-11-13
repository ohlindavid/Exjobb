clear;
close all;
clc;
N = 60;

% Placeholder matrix where element (i,j) notes the distance matrix between
% channel i and j. 
dMatrix = rand(N,N);
dMatrix = dMatrix/norm(dMatrix);
dMatrix = (dMatrix'+dMatrix)/2;

% Noise weight vector. [alpha , 1/f , m]
w_noise = [1 , 1 , 1]';
w_noise = w_noise/norm(w_noise);

Coh_oof = [];
Coh_alpha = [];