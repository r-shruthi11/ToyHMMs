function [toy_params, xdat, zdat] = HMM_genDataRAND(N);

% Generate toy HMM data to perform inference

% Set dimensionality of latents and observations
K = 3;   % number of states
D = 4;   % number of possible observations

toy_params = struct();

% fun = @(a,b) 50*power(a, b);
% N = bsxfun(fun, 2, 0:1:len_N); % Lengths of datasets

toy_params.N = N;


% Observations are 101, 102, 103 and 104
Obs = 100 + [1:1:D];

% States are 1, 2, 3, 4 and 5
States = 1:1:K;

%% Build simulated transition matrix by sampling each column from the uniform random distribution

A = rand(K,K);
A = A./repmat(sum(A,2),1,K); % normalize so rows sum to 1

toy_params.A = A;


%% Build simulated emissions matrix by sampling each column from the uniform random distribution

Phi = rand(K,D);
Phi = bsxfun(@rdivide, Phi, sum(Phi,2)); % normalize so rows sum to 1 (using bsxfun)

toy_params.Phi = Phi;


% Initialize starting state from a uniform random distribution
Pi0 = repmat(1/K, K, 1);  % initial state distribution for first latent

toy_params.Pi0 = Pi0;

for i = 1:length(N)
    
    z_lats = zeros(1,N(i));  % create space for latents
    x_obs = zeros(1,N(i));   % create space fo observations

    z_lats(1) = randsample(States, 1, true, Pi0);
    x_obs(1) = randsample(Obs, 1, true, Phi(z_lats(1), :));

    for ii = 2:N(i)
        z_lats(ii) = randsample(States, 1, true, A(z_lats(ii-1), :));
        x_obs(ii) = randsample(Obs, 1, true, Phi(z_lats(ii), :));

    end
    
    xdat(i).x_obs = x_obs;
    zdat(i).z_lats = z_lats;
    
end

end

% save('genHMMdata.mat', 'x_obs', 'z_lats', 'A', 'Phi', 'Pi0');

