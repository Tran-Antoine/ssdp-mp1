%% Annihilathing filter augmented
% The goal is to find the frequecy values of the harmonic signals by
% resolving an optimization problem. It is solved iteratively by solving a
% linear matrix equation.
function [b, c] = annihilating_filter_augmented(x,K)

% Use an harmonic signal for testing


m = K;

% random initialization 
c0 = randn([m 1]); 
c0 = c0/norm(c0);


N = length(x);
x_f = fft(x,N);

%freq = 0
figure
plot(abs(x_f))


    
G = zeros(m,m);

for i = 0:m-1
G(:,i+1) =  x(m - i : 2 * m - i-1);
end


G = eye(m);
G_h = G.';

a = x; % 
a = a(1:m); % 
beta = inv(G_h*G)*G_h*a';


cn= c0;
T_beta = right_dual(beta,m);

max_inter = 1000;

Gdot = G.' * G;
for i= 1:max_inter
    
R_cn = right_dual(cn,m);


% the linear equation to solve.

A = [zeros([m,m])  T_beta.' zeros([m,m]) c0;
    T_beta zeros(2*m-1) -R_cn zeros([2*m-1 1]);
    zeros(m) -R_cn.' Gdot zeros([m 1]);
    c0.' zeros([1 2*m-1]) zeros([1 m]) 0];

fri = zeros([size(A,2),1]);
fri(end) = 1;

cn_solve = linsolve(A,fri);

cn = cn_solve(1:m); % update cn



R_cn = right_dual(cn,m);

B = [Gdot R_cn.';
    R_cn zeros(2*m-1)];
temp = [G.'*a';zeros([2*m-1,1])];

solve =  linsolve(B,temp);

% Update the b
b = solve(1:m);

err = norm(a.'- G*b);

end

c =cn;
end

    



