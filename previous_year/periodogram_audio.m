function p = periodogram_audio(N, fs, l_y, r_y)
% PERIODOGRAM_AUDIO plot the periodogram of the clean signal
%   N: length of the signal
%   fs: the sampling frequency
%   l_y: the left side of the signal
%   r_y: the right side of the signal

    freq_axis = (0:N/2)*fs/N;
    freq_axis_limit = [0 500];

    p_l = periodogram(l_y,[],N,'one-sided');
    p_r = periodogram(r_y,[],N,'one-sided');

    figure
    subplot(2,1,1)
    plot(freq_axis,p_l)
    ylabel('|y_l|')
    xlim(freq_axis_limit)
    title('Clean signal periodogram left')

    subplot(2,1,2)
    plot(freq_axis,p_r)
    ylabel('|y_r|')
    xlim(freq_axis_limit)
    title('Clean signal periodogram right')

    err = norm(l_y - r_y);

    disp(strcat('The difference between the left and the right channel : ', num2str(err,'%0.4f')))
    
    p = p_l;
end