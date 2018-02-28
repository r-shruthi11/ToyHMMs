function [params, Afinal, Phifinal] = CheckEMSingleRun_InitialRand()
%% Generate Data
clear;

clc;

N = [1000];

[params, xdat, zdat] = HMM_genData(N);

K = 3;

i = 1;

n = N(i);

%% Convert characters in x_obs to an ordered list of possible emissions

x_obs = xdat(i).x_obs;

[categ, ~, ic] = unique(x_obs);

D = numel(categ); % Number of unique characters emitted

ord_list = 1:D;

x = ord_list(ic);

X = sparse(x, 1:n, 1, D, n); % X is a sparse matrix with each row corresponding to one
% observed category and each column corresponding to one
% timestep
% X(i,j) = 1 if in the jth timestep, the
% emission was in category i


%% Initialize the model parameters

A0 = rand(K,K); % transition probabilities matrix
A0 = A0./repmat(sum(A0,2),1,K); % normalize so rows sum to 1

Phi0 = rand(K,D); % emission probabilities matrix
Phi0 = Phi0./repmat(sum(Phi0,2), 1, D); % normalize so rows sum to 1

% A0 = params.A+rand(size(params.A))*.1; % Initialization
% A0 = A0./repmat(sum(A0,2),1,K);
% 
% Phi0 = params.Phi+rand(size(params.Phi))*0.1; % Initialization
% Phi0 = Phi0./repmat(sum(Phi0,2), 1, D);

A = A0;
Phi = Phi0;
Pi0 = repmat(1/K, 1, K)'; % distribution of initial states
M = Phi*X; % Likelihood term p(x|z) - The nth column of M represents Phi(:,x) where x is the
% emission observed at timestep n

%% EM-Algorithm

Nsteps = 500; % Number of steps of EM

llh = -inf(1, Nsteps); % Log-likelihood function

s = 2; %Loop variable

flag = 1;

% Aestim = zeros(K^2, Nsteps);

% Phiestim = zeros(K*D, Nsteps);

%while(s <= Nsteps && flag > 0)

while(s<=Nsteps)
    
    % E-step
    
    [alpha_fwd, beta_bwd, gamma_smoothed, epsilon_joint, c] = compFwdBwdHMM_sr(M, A, Pi0);
    
    llh(s) = sum(log(c(c>0))); % Compute log likelihood ie, log(P(X))
    
    % M-step
    
    A = sum(epsilon_joint,3)./repmat(sum(sum(epsilon_joint, 3),2), 1, K);
    
%     Aestim(:, s-1) = A(:);
    
    Pi0 = gamma_smoothed(:,1);
    
    Phi = bsxfun(@times,gamma_smoothed*X',1./sum(gamma_smoothed,2));
    
%     Phiestim(:, s-1) = Phi(:);
    
    M = Phi*X;
    
%     if (llh(s)-llh(s-1) < check_conv*abs(llh(s-1)))
%         
%         flag = 0;
%         fprintf('N=%d: Converged in %d steps\n', n, s);
%         
%     end
    
    s = s+1;
    
end

Afinal = A;

Phifinal = Phi;


% Visualise results
% close all;
% 
% subplot(3,1,1);
% plot(Aestim);
% hold on;
% plot(params.A(:), 'k-', 'LineWidth', 2);
% plot(A0(:), 'b-', 'LineWidth', 2);
% xlim([1 K^2]);
% title('Estimates of A, initial A and actual A');
% % text(9,1, 'Blue: Initial \n Black: Original');
% % legend([p1, p2, p3], 'Aestim', 'Ainit', 'Amodel');
% % legend('show');
% 
% subplot(3,1,2);
% plot(Phiestim);
% hold on;
% plot(params.Phi(:), 'k-', 'LineWidth', 2);
% plot(Phi0(:), 'b-', 'LineWidth', 2);
% xlim([1 K*D]);
% title('Estimates of Phi, initial Phi and actual Phi');
% 
% subplot(3,1,3);
% plot(llh, 'k.-');
% title('Log P(X)');

