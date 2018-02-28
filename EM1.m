clear;

clc;

%% Generate toy data

N = [100 500 1000 5000 10000 50000]; % Number of different timesteps that the data is generated for 

[params, xdat, zdat] = HMM_genData(N); % params: Model parameters that generated the data 
                                       % xdat, zdat: length(N)*1 struct
                                       % where each field contains x_obs or
                                       % z_lats for each N
                                       

K = 3; % Number of latent states
D = 4; % NUmber of observed characters

numN = length(N);

AEstim = zeros(K^2,numN); 
PhiEstim = zeros(K*D,numN);


A0 = params.A+rand(size(params.A))*.1; % Initialization
A0 = A0./repmat(sum(A0,2),1,K);

Phi0 = params.Phi+rand(size(params.Phi))*0.1; % Initialization
Phi0 = Phi0./repmat(sum(Phi0,2), 1, D);


%% For each sequence of size n in N, run the EM algorithm
for i = 1:numN
    
    n = N(i);
    
    % Convert characters in x_obs to an ordered list of possible emissions
  
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


    % Initialize the model parameters

%     A = rand(K,K); % transition probabilities matrix
%     A = A./repmat(sum(A,2),1,K); % normalize so rows sum to 1
% 
%     Phi = rand(K,D); % emission probabilities matrix
%     Phi = Phi./repmat(sum(Phi,2), 1, D); % normalize so rows sum to 1

    A = A0; 
    Phi = Phi0;

    Pi0 = repmat(1/K, 1, K)'; % distribution of initial states


    M = Phi*X; % Likelihood term p(x|z) - The nth column of M represents Phi(:,x) where x is the 
               % emission observed at timestep n

    % EM-Algorithm 

    check_conv = 1e-9; % Convergence criterion

    Nsteps = 2000; % Number of steps of EM

    llh = -inf(1, Nsteps); % Log-likelihood function

    s = 2; %Loop variable

    flag = 1;
    
    while(s <= Nsteps && flag > 0)
        
        % E-step 

        [alpha_fwd, beta_bwd, gamma_smoothed, epsilon_joint, c] = compFwdBwdHMM_sr(M, A, Pi0);

        llh(s) = sum(log(c(c>0))); % Compute log likelihood ie, log(P(X))

        % M-step

        A = sum(epsilon_joint,3)./repmat(sum(sum(epsilon_joint, 3),2), 1, K);

        Pi0 = gamma_smoothed(:,1);

        M = bsxfun(@times,gamma_smoothed*X',1./sum(gamma_smoothed,2))*X;

        if (llh(s)-llh(s-1) < check_conv*abs(llh(s-1)))

            flag = 0;
            fprintf('N=%d: Converged in %d steps\n', n, s);

        end

        s = s+1;

    end
     
    if flag==1
           fprintf('N=%d: Did not converge in %d steps\n', n, s);
    end
    plot(llh);
   
   err1 = abs(params.A-A);
   err2 = abs(params.Phi-Phi);
   err3 = abs(params.Pi0-Pi0);
   
   AEstim(:,i) = A(:);
   PhiEstim(:,i) = Phi(:);
   
   error(i).A = sum(err1(:));
   error(i).Phi = sum(err2(:));
   error(i).Pi0 = sum(err3(:));
     
end     

%%


errA0 = sum(sum(abs(params.A-A0)))
errPhi0 = sum(sum(abs(params.Phi-Phi0)))

errAFinal = sum(sum(abs(params.A-A)))
errPhiFinal = sum(sum(abs(params.Phi-Phi)))


subplot(221);
plot(1:numN, [error(:).A], 'b-o', [1 numN], errA0*[1 1], 'k');
title('transitions (A)');
subplot(222);
plot(1:numN, [error(:).Phi], '-ro',[1 numN], errPhi0*[1 1], 'k');
title('emissions (Phi)');
subplot(223);
plot([error(:).Pi0], 'ko-');
title('initial');











