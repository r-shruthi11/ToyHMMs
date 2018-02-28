CheckForwardBackwardAlgorithm(params, x_obs, z_lats);
% Check custom forward-backward algorithm with MATLAB's implementation 

% params contains the model parameters used to generate the data
% params.A: transition probability matrix
% params.Phi: emissions matrix
% params.Pi: initial state distribution

% x_obs is the sequence of characters emitted by the (hidden) markov chain
% of states z_lats

%% Generate the data

N = [100 500 1000 5000]

[params, xdat, zdat] = HMM_genData(N);

x_obs = xdat(3).x_obs;

%% Convert the characters in x_obs to an ordered sequence of numbers

n = length(x_obs);

[categ, ~, ic] = unique(x_obs);

D = numel(categ); % Number of unique characters emitted 

ord_list = 1:D;

x = ord_list(ic);

X = sparse(x, 1:n, 1, D, n); % X is a sparse matrix with each row corresponding to one 
                             % observed character and each column corresponding to one
                             % timestep 
                             % X(i,j) = 1 if in the jth timestep, the
                             % emission was character i
                             
%% Generate likelihood matrix
M = params.Phi*X;
                       

%% Run custom forward-backward
[alpha_fwd, beta_bwd, gamma_smoothed, epsilon_joint, c] = compFwdBwdHMM_sr(M, params.A, params.Pi0);


%% Run MATLAB's forward-backward
[PSTATES,logpseq,FORWARD,BACKWARD,S] = hmmdecode(x, params.A, params.Phi);


%% Visualise the differences in alpha, beta

close all;
figure();
subplot(2,1,1);
imagesc(alpha_fwd);
title('alpha fwd');
subplot(2,1,2);
imagesc(FORWARD(:, 2:end));
title('FORWARD');

figure();
subplot(2,1,1);
imagesc(beta_bwd);
title('beta bwd');
subplot(2,1,2);
imagesc(BACKWARD(:, 2:end));
title('BACKWARD');

figure();
subplot(2,1,1);
[~, al_est] = max(alpha_fwd);
[~, f_est] = max(FORWARD(:, 2:end));
plot(1:length(alpha_fwd), al_est, 'ro-', 1:length(FORWARD(:, 2:end)), f_est, 'bd-');
title('Max state vs. n');
[~, be_est] = max(beta_bwd);
[~, b_est] = max(BACKWARD(:, 2:end));
subplot(2,1,2);
plot(1:length(beta_bwd), be_est, 'ro-', 1:length(BACKWARD(:, 2:end)), b_est, 'bd-');
title('Max state vs. n');

