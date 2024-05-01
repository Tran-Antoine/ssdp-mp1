function f = annihilating_filter(x, fs, K)
% ANNIHILSTIN_FILTER  apply the annihilating_filter to the signal and
% retrieve the frequencies
%   x: our signal
%   fs: the sampling frequency
%   K: the number of spikes
%   
%   f: the retrieved frequencies
    
    A = zeros(K,K);

    for i = 0:K-1
    A(:,i+1) =  x(K - i : 2 * K - i-1);
    end

    b = -x(K+1 : 2 * K);

    %%
    % if music data provided on moodle uncomment this line
    %X = linsolve(A,b);
    % if harmonic signal created by a sum of sinusoid uncomment this line
    X = linsolve(A,b.');
    %%
    h = [1 X.'];

    %% r are the zeroes of H(z)
    r = roots(h);
    angles = angle(r);
    f_est = angles/(2*pi);
    %%
    f = f_est*fs;

end