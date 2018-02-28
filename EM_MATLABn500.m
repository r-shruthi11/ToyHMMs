clear;

clc;

% Generate and perform inference on 1000 samples of the same dataset
% starting from different initial conditions

%% Generate Data
clear;

clc;

% N = [500, 750, 1000, 1500, 2000];

N = [1000];

[params, xdat, zdat] = HMM_genData(N);

% [params, xdat, zdat] = HMM_genDataRAND(N);

K = 3;

%% Convert observations to an ordered list

x_obs = xdat(1).x_obs;

[categ, ~, ic] = unique(x_obs);

D = numel(categ); % Number of unique characters emitted

ord_list = 1:D;

x = ord_list(ic);

%% Perform EM inference

nTimes = 500;

for nt = 1:nTimes
    
    nt
    
    A0 = rand(K,K); % transition probabilities matrix
    A0 = A0./repmat(sum(A0,2),1,K); % normalize so rows sum to 1
    
    Phi0 = rand(K,D); % emission probabilities matrix
    Phi0 = Phi0./repmat(sum(Phi0,2), 1, D); 
    
    
    % Run EM
    
    [A, Phi, logliks] = hmmtrain(x,A0,Phi0, 'MaxIterations', 500);
    
    Aest(:, :, nt) = A;
    
    Phiest(:, :, nt) = Phi;
    
    FinalLLH(:, nt) = logliks(end);
    
end
    
plot(FinalLLH);
    
    
    
    
    
    
    
    
    
    