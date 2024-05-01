function plot_retrieved_frequencies(x, fs, f, t)
% PLOT_RETRIEVED_FREQUENCIES  apply the annihilating_filter to 
% the signal and retrieve the frequencies
%   x: our signal
%   fs: the sampling frequency
%   f: the retrieved frequencies
%   t: title of the plot

    fft_x = fftshift(fft(x));
    df = fs/length(x);
    freq = -fs/2:df:fs/2-df;

    ystart = zeros(1,numel(f));
    yend = ones(1,numel(f))*10e3;
    
    figure; hold on;
    title(t);
    plot(freq, abs(fft_x))
    for idx = 1 : numel(f)
        plot([f(idx) f(idx)], [ystart(idx) yend(idx)], 'r');
    end

end