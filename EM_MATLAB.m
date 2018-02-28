
%% Generate Data
clear;

clc;

N = [500, 750, 1000, 1500, 2000];

[params, xdat, zdat] = HMM_genData(N);

% [params, xdat, zdat] = HMM_genDataRAND(N);

K = 3;

%% Run EM for different lengths N

for i = 1:numel(N)
    
    n = N(i);
    
    % Convert characters in x_obs to an ordered list of possible emissions
    
    x_obs = xdat(i).x_obs;
    
    [categ, ~, ic] = unique(x_obs);
    
    D = numel(categ); % Number of unique characters emitted
    
    ord_list = 1:D;
    
    x = ord_list(ic);
    
    % X = sparse(x, 1:n, 1, D, n); % X is a sparse matrix with each row corresponding to one
    % observed category and each column corresponding to one
    % timestep
    % X(i,j) = 1 if in the jth timestep, the
    % emission was in category i
    
    %Initialize model parameters
    
    A0 = rand(K,K); % transition probabilities matrix
    A0 = A0./repmat(sum(A0,2),1,K); % normalize so rows sum to 1
    
    Phi0 = rand(K,D); % emission probabilities matrix
    Phi0 = Phi0./repmat(sum(Phi0,2), 1, D); % normalize so rows sum to 1
    
%     A0 = params.A+rand(size(params.A))*.1; % Initialization
%     A0 = A0./repmat(sum(A0,2),1,K);
% 
%     Phi0 = params.Phi+rand(size(params.Phi))*0.1; % Initialization
%     Phi0 = Phi0./repmat(sum(Phi0,2), 1, D);

    %Run EM
    i
    
    [Aest,Phiest] = hmmtrain(x,A0,Phi0, 'MaxIterations', 250);
    
    Aestim(:, i) = Aest(:);
    
    Phiestim(:, i) = Phiest(:);
    
end


    
    
    %% Visualise results
    close all;
    
    figure();
    
    for j = 1:numel(N)
    
        subplot(numel(N),1,j);
        plot(Aestim(:, j),'r-', 'LineWidth', 2);
        hold on;
        plot(params.A(:), 'k-', 'LineWidth', 2);
        plot(A0(:), 'b-', 'LineWidth', 2);
        xlim([1 K^2]);
        title(['Estimates of A, initial A and actual A for n = ' num2str(N(j))]);
    
    end
    
    figure();
    
     for j = 1:numel(N)
         
         %p = Phiestim(:,j);
         %p = reshape(reshape(p,4,3)',[],1);
         
         
        subplot(numel(N),1,j);
        plot(Phiestim(:,j),'r-', 'LineWidth', 2);
        %plot(p,'r');
        hold on;
        plot(params.Phi(:), 'k-', 'LineWidth', 2);
        plot(Phi0(:), 'b-', 'LineWidth', 2);
        xlim([1 K*D]);
        title(['Estimates of Phi, initial Phi and actual Phi for n = ' num2str(N(j))]);
    
    end
    
    
    
