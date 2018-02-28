clear;

clc;

close all;

%% Load data from EM runs

load('MATLAB_EM500sims.mat', 'Phiest', 'Aest', 'FinalLLH', 'params');

%% Find top 10% LLH estimates

llhN = histc(FinalLLH, -560:1:-540); % Total 470

perc10LLH = find(FinalLLH<-552 & FinalLLH>-555); % ~ Bottom 10% of this distribution and their indices

%% Find corresponding parameter estimates 

Phi_ = Phiest(:, :, perc10LLH); % Phiest: Estimate of Phi from the EM runs

A_ = Aest(:, :, perc10LLH); % Aest: Estimate of A from the EM runs

%% Extract original model parameters for comparison

Phi = params.Phi; % Actual Phi

A = params.A; % Actual A

%% Find 10 random samples from here

K = randsample(1:1:numel(perc10LLH), 10);

%% Generate summary figs
for k = 1:numel(K)
    
    i = K(k);
    
    [bestPerm, newPhi] = permRows(Phi_(:, :, i), Phi);
    
    Atemp = A_(:, :, i);
    
    newA = Atemp(bestPerm, :); % Permute rows
    
    newA = newA(:, bestPerm); % Permute columns
    
    
    f = figure('units','normalized','outerposition',[0.5 0.5 0.2 0.5], 'visible', 'on');

    subplot(3,1,1);

    imagesc(Phi_(:, :, i));

    title(['Final estimate of Phi for run ' num2str(i)]);

    subplot(3,1,2);

    imagesc(Phi);

    title(['Model parameter Phi for run ' num2str(i)]);

    subplot(3,1,3);

    imagesc(newPhi);

    title(['Permuted to match Phi for run ' num2str(i)]);
    
    set(f, 'PaperUnits', 'centimeters');
    set(f, 'PaperPosition', [0 0 10 15]);

    saveas(f, ['./Summary/ControlPhi_' num2str(i)], 'png');

    g = figure('units','normalized','outerposition',[0 0 0.2 0.5], 'visible', 'off');

    subplot(3,1,1);

    imagesc(A_(:, :, i));

    title(['Final estimate of A for run ' num2str(i)]);

    subplot(3,1,2);

    imagesc(A);

    title(['Model parameter A for run ' num2str(i)]);

    subplot(3,1,3);

    imagesc(newA);

    title(['Permuted to match A for run ' num2str(i)]);
    
     set(g, 'PaperUnits', 'centimeters');
    set(g, 'PaperPosition', [0 0 10 15]);

    saveas(g, ['./Summary/ControlA_' num2str(i)], 'png');
    
end









