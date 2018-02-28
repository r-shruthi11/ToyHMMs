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

%% Run EM and check whether the results are permuted 

nTimes = 10;

for nt = 1:nTimes
    
    % Initialize
    
    A0 = rand(K,K); % transition probabilities matrix
    A0 = A0./repmat(sum(A0,2),1,K); % normalize so rows sum to 1
    
    Phi0 = rand(K,D); % emission probabilities matrix
    Phi0 = Phi0./repmat(sum(Phi0,2), 1, D); 
    
    
    % Run EM
    
    [Aest,Phiest] = hmmtrain(x,A0,Phi0, 'MaxIterations', 500);
    
    % Check permutation
    
    [bestPerm, newPhiest] = permRows(Phiest, params.Phi);
    
    % Permute Aest also by same
    
    newAest = Aest(bestPerm, :); % Permute rows
    
    newAest = newAest(:, bestPerm); % Permute columns
    
    errA = sum(sum(abs(Aest-params.A)));
    
    errAnew = sum(sum(abs(newAest-params.A)));
    
    errP = sum(sum(abs(Phiest-params.Phi)));
    
    errPnew = sum(sum(abs(newPhiest-params.Phi)));
    
    Aestim(:, :, nt) = Aest;
    
    Phiestim(:, :, nt) = Phiest;
    
    phiPerms(nt, :) = bestPerm;

    % Plot
    
     f = figure('units','normalized','outerposition',[0 0 1 1], 'visible', 'off');

%     figure();
    
    subplot(3,1,1);
    
%     imagesc(reshape(Phiestimates(5).Phiestim(:, end), [3,4]));

    imagesc(Phiest);
    
    title(['Final estimate of Phi for run ' num2str(nt) ' with error = ' num2str(errP)]);

    subplot(3,1,2);

    imagesc(params.Phi);
    
    title(['Model parameter Phi for run ' num2str(nt)]);
    
    subplot(3,1,3);
    
    imagesc(newPhiest);
    
    title(['Permuted to match Phi for run ' num2str(nt) ' with error = ' num2str(errPnew)]);
    
    saveas(f, ['./Figs/PhiPerm_' num2str(nt)], 'epsc');
    
    
    g = figure('units','normalized','outerposition',[0 0 1 1], 'visible', 'off');
    
%     figure();
    
    subplot(3,1,1);
    
    imagesc(Aest);
    
    title(['Final estimate of A for run ' num2str(nt) ' with error = ' num2str(errA)]);

    subplot(3,1,2);

    imagesc(params.A);
    
    title(['Model parameter A for run ' num2str(nt)]);
    
    subplot(3,1,3);
    
    imagesc(newAest);
    
    title(['Permuted to match A for run ' num2str(nt) ' with error = ' num2str(errAnew)]);

    saveas(g, ['./Figs/APerm_' num2str(nt)], 'epsc');
    
    
end



    
    
    
    
    
    