function periodogram_audio_noisy(N, fs, y)
% PERIODOGRAM_AUDIO_NOISY plot the periodogram of the noisy signal
%   N: length of the signal
%   fs: the sampling frequency
%   y: the signal

    freq_axis = (0:N/2)*fs/N;
    freq_axis_limit = [0 500];

    p_l = periodogram(y,[],N,'one-sided');

    figure
    plot(freq_axis,p_l)
    ylabel('|y_l|')
    xlim(freq_axis_limit)
    title('Periodogram noisy signal')
    
end