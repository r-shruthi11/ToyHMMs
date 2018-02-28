function [alpha_fwd, beta_bwd, gamma_smoothed, epsilon_joint, c] = compFwdBwdHMM_sr2(M, A, Pi0)

[K, N] = size(M);

At = A'; %Transpose A - now At(i,j) = P(zn = i| zn-1 = j) or P(z(n,1) = 1| z(n-1, j) = 1)

c = zeros(1, N); % Scaling constants

alpha_fwd = zeros(K, N);

% First time bin
alpha_fwd(:,1) = Pi0.*M(:,1); 

% Define scaling factor
c(1,1) = sum(alpha_fwd(:,1)); 

% Normalize alpha_fwd to get alpha_fwd-hat
alpha_fwd(:,1) = alpha_fwd(:,1)/c(1,1);

for n_iter = 2:N
    
    alpha_fwd(:, n_iter) = (At*alpha_fwd(:,n_iter-1)).*M(:,n_iter);
    c(1,n_iter) = sum(alpha_fwd(:,n_iter));
    alpha_fwd(:, n_iter) = alpha_fwd(:, n_iter)/c(1,n_iter);
    
end


beta_bwd = zeros(K,N);

% Initialize
beta_bwd(:,N) = 1;

for n_iter = N-1:-1:1
    
    % Compute and normalize to get beta_bwd-hat
    beta_bwd(:,n_iter) = A*(beta_bwd(:,n_iter+1).*M(:,n_iter+1))/c(1,n_iter+1); 
    
end

gamma_smoothed = alpha_fwd.*beta_bwd; % Marginal posterior distribution over latent states 

epsilon_joint = zeros(K, K, N-1); % Joint posterior distribution over two successive latents

for n_iter = 2:N
    
    epsilon_joint(:, :, n_iter-1) = (A.*(alpha_fwd(:,n_iter-1)*(M(:, n_iter).*beta_bwd(:, n_iter)*c(1,n_iter))'));
    
end

end

