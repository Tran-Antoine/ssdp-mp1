%% Main
% this script compare the spectrum estimation of an clean/noisy harmonic 
% signal using a optimization technic which consist on minimizing the 
% estimation error and an anihilating filter

% BEFORE RUNNING: In the annihilating_filter.m make sure 
% X = linsolve(A,b.'); is uncommented and X = linsolve(A,b); is commented


clear all;
close all;
%% We create Data without Noise
fs = 5000; % [Hz]
time = 0:1/fs:2-1/fs;
f1 = 440; % [Hz]
A1 = 10;
f2 = 490; % [Hz]
A2 = 7;
x1 = A1*sin(2*pi*f1*time);
x2 = A2*sin(2*pi*f2*time);
x = x1+x2;

%% 
K = 4;
%% Recover the frequencies with the Annihilating Filter
f = annihilating_filter(x, fs, K);

% Plot the retrieve frequencies
plot_retrieved_frequencies(x, fs, f, 'FFT and retrieved frequencies of the clean signal')


%% Recover the frequencies with the Augmented Annihilating Filter
[b, c] = annihilating_filter_augmented(x, K);

h = [1 c.'];

r = roots(h);
angles = angle(r);
f_est = angles/(2*pi);
f = f_est*fs;

% We plot the frequencies found
plot_retrieved_frequencies(x, fs, f, 'FFT and retrieved frequencies of the clean signal')


%% Then we add noise to the data
noise = rand(1, numel(x));
x_n = x+noise;

% Recover the frequencies with the Annihilating Filter
f_n = annihilating_filter(x_n, fs, K);

% Plot the retrieve frequencies
plot_retrieved_frequencies(x_n, fs, f_n,'FFT and retrieved frequencies of the noisy signal' )

%% Recover the frequencies with the Augmented Annihilating Filter
[b_n, c_n] = annihilating_filter_augmented(x_n, K);

h_n = [1 c_n.'];

r_n = roots(h_n);
angles_n = angle(r_n);
f_est_n = angles_n/(2*pi);
f_n = f_est_n*fs;

% We plot the frequencies found
plot_retrieved_frequencies(x_n, fs, f_n, 'FFT and retrieved frequencies of the noisy signal')
