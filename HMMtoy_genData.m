function [] = HMMtoy_genData();

% Generate toy HMM data to perform inference

% Set dimensionality of latents and observations
K = 3;   % number of states
D = 4;   % number of possible observations
N = 250;  % number of time bins

% Observations are 101, 102, 103 and 104
Obs = 100 + [1:1:D];

% States are 1, 2, 3, 4 and 5
States = 1:1:K;

%% Build simulated transition matrix by sampling each column from Dirichlet

a_transitions = 1;  % concentration parameter (higher makes more uniform probabilities)
adiag = 5; % extra concentration for diagonal entries
A = gamrnd(a_transitions*ones(K)+adiag*diag(ones(K,1)),1);   % draw gamma RVs
A = A./repmat(sum(A,2),1,K); % normalize so rows sum to 1


%% Build simulated emissions matrix by sampling each column from Dirichlet
a_emissions = 1; % concentration parameter
Phi = gamrnd(a_emissions,1,K, D);
Phi = bsxfun(@rdivide, Phi, sum(Phi,2)); % normalize so rows sum to 1 (using bsxfun)


% Initialize starting state from a uniform random distribution
Pi0 = repmat(1/K, 1, K);  % initial state distribution for first latent
z_lats = zeros(1,N);  % create space for latents
x_obs = zeros(1,N);   % create space fo observations

z_lats(1) = randsample(States, 1, true, Pi0);
x_obs(1) = randsample(Obs, 1, true, Phi(z_lats(1), :));

for i = 2:N
    z_lats(i) = randsample(States, 1, true, A(z_lats(i-1), :));
    x_obs(i) = randsample(Obs, 1, true, Phi(z_lats(i), :));
    
end

save('genHMMdata.mat', 'x_obs', 'z_lats', 'A', 'Phi', 'Pi0');

