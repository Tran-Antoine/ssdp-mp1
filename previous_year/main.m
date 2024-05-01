%% Main
% this script compare the spectrum estimation of an clean/noisy harmonic 
% signal using a optimization technic which consist on minimizing the 
% estimation error and an anihilating filter

% BEFORE RUNNING: In the annihilating_filter.m make sure X = linsolve(A,b);
% is uncommented and X = linsolve(A,b.'); is commented

clear all;
close all;

%% Data parameters
folder_name = 'data/';
clean_filename = strcat(folder_name,'Clean bass.wav');
noisy_filename = strcat(folder_name,'Noisy bass.wav');

%% Clean Signal
% First we analyse the clean signal
    %% Data load

    [signal,fs] = audioread(clean_filename);
    l_y = signal(:,1);
    r_y = signal(:,2);

    N = length(l_y);

    %% Periodogram
    % Let's have a first look on the spectre of the Clean signal.
    % Our music is not harmonic, but we can see that there is spikes in the 
    % periodogram, so we interpreted as spikes with noise, so we decided to
    % interpret the periodogram as our harmonic signal
    p = periodogram_audio(N, fs, l_y, r_y);
    
    % First, notice the spectre of the left channel and right are
    % simmilar. This is because the left and the right channel is nearly the
    % same signal.

    % For further analyse, we will consider only one channel, the left one. 
    y= l_y;

    % Seconly, we can see that the harmonic signal is composed of K = 8 spikes.
    K = 16;
    %% Annihilating Filter
    % First we apply the anihilating filter
    f = annihilating_filter(fft(p), fs, K);

    % TODO: Find a way to verify our results
    plot_retrieved_frequencies(p, fs, f, 'FFT and retrieved frequencies of the clean signal')

    %% Annihilating Filter Augmented
    % Then we apply the augmented anihilating filter
    %[b, c] = annihilating_filter_augmented(y, K);

    % TODO: Apply augmented anihilating filter
    
    % TODO: Find a way to verify our results
    %h = [1 c.'];

    %r = roots(h);
    %angles = angle(r);
    %f_est = angles/(2*pi);
    %f = f_est*fs;

    % We plot the frequencies found
    %plot_retrieved_frequencies(x, fs, f, 'FFT and retrieved frequencies of the clean signal')
    
%% Noisy Signal
% Then we analyse the Noisy signal
    %% Data load
    [signal_n,fs_n] = audioread(noisy_filename);
    y_n = signal_n(:,1);

    N_n = length(y_n);

    %% Periodogram
    % Let's have a first look on the spectre of the Clean signal.

    periodogram_audio_noisy(N_n, fs_n, y_n);

    % Seconly, we can see that the harmonic signal is composed of K = 8 spikes.

    %% Annihilating Filter
    % First we apply the anihilating filter
    f_n = annihilating_filter(y_n, fs_n, K);
    
    % TODO: Find a way to verify our results
    plot_retrieved_frequencies(y_n, fs_n, f_n, 'FFT and retrieved frequencies of the noisy signal')


    %% Annihilating Filter Augmented
    % Then we apply the augmented anihilating filter
    %[b_n, c_n] = annihilating_filter_augmented(y_n, K);

    % TODO: Find a way to verify our results



